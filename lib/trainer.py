from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers.trainer_pt_utils import (
    get_model_param_count
)
from logging import getLogger
import time

def default_scheduler_factory(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

logger = getLogger()

@dataclass(kw_only=True)
class Trainer():
    model: nn.Module
    optimizer: torch.optim.Optimizer
    dataset: DataLoader
    device: torch.device
    accelerate: bool
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
    use_scheduler: bool = False
    grad_accum: int = 1
    accelerator_kwargs: dict = field(default_factory=dict)
    fp16: bool = False

    def __post_init__(self):
        self.accelerator_kwargs["gradient_accumulation_steps"] = self.grad_accum

        self.batch_size = self.dataset.batch_size

        if self.use_scheduler and not self.scheduler:
            self.scheduler = default_scheduler_factory(self.optimizer)

        if self.accelerate:
            # https://github.com/huggingface/accelerate/issues/1460
            # if self.use_scheduler:
            #     self.accelerator_kwargs["adjust_scheduler_to_accumulation"] = True

            self.accelerator = Accelerator(**self.accelerator_kwargs)
            device = self.accelerator.device
            self.device = device

            if self.use_scheduler:
                self.model, self.optimizer, self.dataset, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.dataset, self.scheduler)
            else:
                self.model, self.optimizer, self.dataset = self.accelerator.prepare(self.model, self.optimizer, self.dataset)

    def _train(self, epochs=1, max_steps=None):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for ep in range(epochs):
            for batch_idx, batch in enumerate(self.dataset):
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss.backward()

                if self.grad_accum > 1 and (batch_idx+1) % self.grad_accum == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.use_scheduler:
                        self.scheduler.step()
                elif self.grad_accum == 1:
                    self.optimizer.zero_grad()
                    self.optimizer.step()
                    if self.use_scheduler:
                        self.scheduler.step()

                batch_loss = loss.item() / self.grad_accum

                print(f"Epoch {ep} | Batch {batch_idx} | Loss {batch_loss}")

                train_loss += batch_loss
                total += 1

                if max_steps is not None and batch_idx >= max_steps:
                    break

        return train_loss / total

    def _train_accelerate(self, epochs=1, max_steps=None):
        self.model.train()
        train_loss = 0
        total = 0
        for ep in range(epochs):
            for batch_idx, batch in enumerate(self.dataset):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

                    self.accelerator.backward(loss)

                    self.optimizer.step()
                    if self.use_scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                    batch_loss = loss.item() / self.grad_accum

                    print(f"Epoch {ep} | Batch {batch_idx} | Loss {batch_loss}")

                    train_loss += batch_loss
                    total += 1

                    if max_steps is not None and batch_idx >= max_steps:
                        break

        return train_loss / total

    def train(self, epochs=1, max_steps=None):
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.dataset):,}")
        logger.info(f"  Num Epochs = {epochs:,}")
        if max_steps:
            logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Instantaneous batch size per device = {self.batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {self.grad_accum}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(self.model, trainable_only=True):,}")


        if self.accelerate:
            return self._train_accelerate(epochs=epochs, max_steps=max_steps)
        else:
            return self._train(epochs=epochs, max_steps=max_steps)
        
