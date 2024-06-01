from dataclasses import dataclass, field
import time
from .utils import convert_seconds_to_string_time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers.trainer_pt_utils import (
    get_model_param_count,
)
import atexit


import numpy as np
import contextlib

from logging import getLogger
from typing import Optional
import os


def default_scheduler_factory(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


def get_number_from_checkpoint(filename):
    parts = filename.split("_")
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


logger = getLogger()


@dataclass(kw_only=True)
class TrainerState:
    epoch: int = 0
    steps: int = 0
    global_loss: int = 0

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "steps": self.steps,
            "global_loss": self.global_loss,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.steps = state_dict["steps"]
        self.global_loss = state_dict["global_loss"]


@dataclass(kw_only=True)
class Trainer:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    accelerator: Accelerator
    dataset: DataLoader
    device: torch.device
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
    use_scheduler: bool = True
    grad_accum: int = 1
    fp16: bool = False
    use_bfloat16: bool = False
    output_dir: str = "./output"
    checkpoint_every: int = 1000
    load_checkpoint: bool = False
    hf_hub_token: Optional[str] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed:
            set_seed(self.seed)

        os.makedirs(self.output_dir, exist_ok=True)

        self.batch_size = self.dataset.batch_size

        if self.use_scheduler and not self.scheduler:
            self.scheduler = default_scheduler_factory(self.optimizer)

        if self.fp16 and self.use_bfloat16:
            self.f16_dtype = torch.bfloat16
        elif self.fp16:
            self.f16_dtype = torch.float16

        device = self.accelerator.device
        self.device = device

        if self.use_scheduler:
            (
                self.model,
                self.optimizer,
                self.dataset,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.model, self.optimizer, self.dataset, self.scheduler
            )

            self.accelerator.register_for_checkpointing(self.scheduler)
        else:
            self.model, self.optimizer, self.dataset = self.accelerator.prepare(
                self.model, self.optimizer, self.dataset
            )

        self.state = TrainerState()
        self.accelerator.register_for_checkpointing(self.state)

    def _save_dist(self):
        if self.accelerator.is_main_process():
            model = self.accelerator.unwrap_model(self.model)
            model_state = model.state_dict()
            model.save_pretrained(f"{self.output_dir}/dist", state_dict=model_state, token=self.hf_hub_token)

    def _save(self, no_dist=False):
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state()
        if not no_dist:
            self._save_dist()

    def _load_last_checkpoint(self):
        try:
            checkpoints_dir = f"{self.output_dir}/checkpoints/"

            if os.path.exists(checkpoints_dir) and os.listdir(checkpoints_dir):
                files = os.listdir(checkpoints_dir)
                checkpoint_files = [
                    f
                    for f in files
                    if f.startswith("checkpoint_")
                    and get_number_from_checkpoint(f) is not None
                ]
                checkpoint_files_sorted = sorted(
                    checkpoint_files, key=get_number_from_checkpoint
                )

                if checkpoint_files_sorted:
                    last_cp = checkpoint_files_sorted[-1]
                    logger.info(f"Trying to load checkpoint: {last_cp}....")
                    self.accelerator.load_state(
                        f"{self.output_dir}/checkpoints/{last_cp}"
                    )

                    # Update the iteration number so that the next checkpoint name is increased by 1
                    last_cp_num = last_cp.split("_")[-1]
                    self.accelerator.project_configuration.iteration = (
                        int(last_cp_num) + 1
                    )
                    return last_cp

        except FileNotFoundError as e:
            logger.warning(e)
        return None

    def _train_accelerate(self, epochs=1, max_steps=None):
        context_manager = contextlib.nullcontext()

        if self.fp16:
            context_manager = torch.cuda.amp.autocast(
                cache_enabled=False, dtype=self.f16_dtype
            )

        self.model.train()
        total_batches = len(self.dataset) * epochs
        start_time = time.time()

        if self.load_checkpoint:
            last_cp = self._load_last_checkpoint()
            if last_cp:
                logger.info(f"Checkpoint loaded: {last_cp}.")
            else:
                logger.warning("No checkpoint found, starting from scratch.")

        atexit.register(self._save)
        for ep in range(self.state.epoch, epochs):
            dataset = self.dataset

            if self.state.steps > 0:
                dataset = self.accelerator.skip_first_batches(
                    self.dataset, self.state.steps
                )

            for batch_idx, batch in enumerate(dataset):
                with self.accelerator.accumulate(self.model):
                    with context_manager:
                        outputs = self.model(**batch)
                        loss = (
                            outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                        )

                    self.accelerator.backward(loss)
                    batch_loss = loss.item()
                    self.optimizer.step()
                    if self.use_scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                    # calculate current epoch as decimal
                    total_batches_done = ep * len(self.dataset) + batch_idx
                    current_epoch_decimal = total_batches_done / total_batches

                    # calculate time elapsed and estimate remaining time
                    time_elapsed = time.time() - start_time
                    batches_remaining = total_batches - total_batches_done
                    estimated_time_remaining = (
                        (time_elapsed / total_batches_done) * batches_remaining
                        if total_batches_done > 0
                        else 0
                    )

                    # convert estimated_time_remaining to format: dd days, hh hours, mm minutes, ss seconds
                    estimated_time_remaining_ddhhmmss = convert_seconds_to_string_time(
                        estimated_time_remaining
                    )

                    # calculate percentage done
                    percent_done = (total_batches_done / total_batches) * 100

                    print(
                        f"Epoch {current_epoch_decimal:.2f} | Batch {batch_idx}/{len(self.dataset)} | Loss {batch_loss} | {percent_done:.2f}% done | Estimated time remaining: {estimated_time_remaining_ddhhmmss}"
                    )

                    self.state.global_loss += batch_loss

                self.state.steps += 1
                if max_steps is not None and batch_idx >= max_steps:
                    break

                if batch_idx % self.checkpoint_every == 0 and batch_idx > 0:
                    self._save()

            self.state.epoch += 1
        atexit.unregister(self._save)
        self._save()

        return self.state.global_loss / self.state.steps

    def train(self, epochs=1, max_steps=None):
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.dataset):,}")
        logger.info(f"  Num Epochs = {epochs:,}")
        if max_steps:
            logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Instantaneous batch size per device = {self.batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {self.grad_accum}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(self.model, trainable_only=True):,}"
        )

        return self._train_accelerate(epochs=epochs, max_steps=max_steps)
