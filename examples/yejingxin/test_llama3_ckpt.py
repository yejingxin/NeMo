# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from dataclasses import dataclass

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.lightning.io.pl import MegatronCheckpointIO
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model.llama import Llama3Config, LlamaModel
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO
from nemo.utils.callbacks.dist_ckpt_io import DistributedCheckpointIO
import logging
from torch.distributed import TCPStore
import datetime
import signal
import os, sys
import torch.distributed as dist

@dataclass
class Llama3Config36M(Llama3Config):
    rotary_base: int = 500_000
    seq_length: int = 8192
    num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16

@dataclass
class Llama3Config8B(Llama3Config):
    rotary_base: int = 500_000
    seq_length: int = 8192
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32

class TCPStoreSignalHandler:
   def __init__(self, store): 
       self.rank = dist.get_rank()
       self.received_signal = False
       self.store = store
       self.shutdown_key = "shutdown_signal"
       self.ready_key_prefix = "ready_to_save_"
       
       if self.rank == 0:
           self.store.set(self.shutdown_key, "0")
           
       signal.signal(signal.SIGTERM, self._handle_signal)
       
       self.logger = logging.getLogger(__name__)
       self.logger.setLevel(logging.INFO)

   def _handle_signal(self, signum, frame):
       self.received_signal = True
       self.logger.info(f"Process {self.rank} received signal {signum}")
       try:
           self.store.set(self.shutdown_key, "1")
       except Exception as e:
           self.logger.error(f"Failed to set shutdown signal: {e}")

   def mark_ready_to_save(self):
       """Mark this rank as ready to save checkpoint"""
       self.store.set(f"{self.ready_key_prefix}{self.rank}", "1")

   def should_stop(self):
       """Check if should stop and all ranks are ready"""
       try:
           # First check shutdown signal
           if self.store.get(self.shutdown_key) == b"1":
               self.received_signal = True
               self.logger.info(f"Process {self.rank} detected shutdown signal")
               return True
               # Mark this rank as ready
               self.mark_ready_to_save()
               
               # Wait for all ranks to be ready
               world_size = dist.get_world_size()
               ready_count = 0
               for r in range(world_size):
                   try:
                       if self.store.get(f"{self.ready_key_prefix}{r}") == b"1":
                           ready_count += 1
                   except Exception:
                       pass
                       
               if ready_count == world_size:
                   return True
                   
           return False
           
       except Exception as e:
           self.logger.error(f"Failed to check signals: {e}")
           return False

class AutoCheckpointCallback(Callback):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.signal_handler = None
        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_train_start(self, trainer, pl_module):
        store = dist.distributed_c10d._get_default_store() #_get_default_group().store
        self.signal_handler = TCPStoreSignalHandler(store=store)
    
    def _finalize_if_async_save(self, trainer):
        checkpoint_io = trainer.strategy.checkpoint_io
        if not isinstance(checkpoint_io, AsyncFinalizableCheckpointIO):
            return
        checkpoint_io.maybe_finalize_save_checkpoint(blocking=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if self.signal_handler and self.signal_handler.should_stop():
            print(f"rank {dist.get_rank()} saving emergency ckpt.")
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"emergency_ckpt_step_{trainer.global_step}.ckpt"
            )
            trainer.save_checkpoint(checkpoint_path)
            self._finalize_if_async_save(trainer)
            if trainer.is_global_zero:
                print(f"Emergency checkpoint saved at: {checkpoint_path}")
            trainer.strategy.barrier()  # Ensure all processes are synced
            if trainer.is_global_zero:
                print(f"Exit Training.")
            sys.exit(0)


def get_trainer(args, callbacks):
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=DistributedDataParallelConfig(),
    )
    checkpoint_io = DistributedCheckpointIO(
        save_ckpt_format='torch_dist',
        load_directly_on_device=True,
        async_save = True,
        torch_dist_multiproc = 8,
        assume_constant_structure = False,
        parallel_save = True,
        parallel_save_within_dp = False,
        parallel_load = True,
    )
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        max_time={"seconds": args.max_runtime},
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=None, #args.val_check_interval,
        limit_val_batches=None, #args.limit_val_batches,
        plugins=[nl.MegatronMixedPrecision(precision="bf16-mixed"), 
                 AsyncFinalizableCheckpointIO(checkpoint_io)
                 ],
        strategy=strategy,
        #default_root_dir="gcs://yejingxin-terraform"
    )
    return trainer


def print_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)


def get_parser():
    parser = argparse.ArgumentParser(description="Llama3 Pretraining on a local node")

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="tokenizer.model",
        help="Path to the tokenizer model file",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="How many nodes to use",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Specify the number of GPUs per node",
    )
    parser.add_argument('--max-runtime', type=int, default=900)  # in seconds
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1_000_000,
        help="Number of steps to run the training for",
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=80,
        help="Checkpoint saving interval in steps",
    )
    parser.add_argument(
        '--val-check-interval',
        type=int,
        default=40,
        help="Validation check interval in steps",
    )
    parser.add_argument(
        '--limit_val_batches',
        type=int,
        default=10,
        help="How many batches to use for validation",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Output dir.",
        required=False,
        default="./nemo_llama3_fault_tol",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Output dir.",
        required=False,
        default="36M",
    )
    return parser


def main():
    args = get_parser().parse_args()

    mbs = 1
    gbs = mbs * args.num_gpus * args.num_nodes

    data = MockDataModule(
        seq_length=8192,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        tokenizer=SentencePieceTokenizer(model_path=args.tokenizer_path),
    )
    model_config = {
        "36M": Llama3Config36M(),
        "8B": Llama3Config8B(),
    }
    assert args.model in model_config

    model = LlamaModel(config=model_config[args.model])

    checkpoint_callback = ModelCheckpoint(
        #save_last=True,
        #monitor="reduced_train_loss",
        #save_top_k=1,
        every_n_train_steps=args.checkpoint_interval,
        #save_on_train_epoch_end=True,
        #save_optim_on_train_end=True,
        filename='{step}-{epoch}',
    )
    #signal_handler = TCPStoreSignalHandler(world_size=8, rank=dist.get_rank())
    autockpt_callback = AutoCheckpointCallback(args.log_dir)
    trainer = get_trainer(args, callbacks=[checkpoint_callback, autockpt_callback])

    nemo_logger = nl.NeMoLogger(
        log_dir=args.log_dir,
        use_datetime_version=False,
        update_logger_directory=True,
        wandb=None,
        ckpt=checkpoint_callback,
    )

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-2,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        clip_grad=1.0,
        log_num_zeros_in_grad=False,
        timers=None,
        bf16=True,
        use_distributed_optimizer=False,
    )
    optim = MegatronOptimizerModule(config=opt_config)
    #trainer.save_checkpoint('./test/ckpt')

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        resume=nl.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
        ),
        optim=optim,
        tokenizer="data",
    )


if __name__ == "__main__":
    main()
