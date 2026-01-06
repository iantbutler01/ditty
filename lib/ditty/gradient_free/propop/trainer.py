"""
PropOpTrainer: Training loop for PropOp local learning.

No optimizer, no gradients - uses PropOpWrapper's learn() method instead.
"""
from dataclasses import dataclass, field
import time
import os
import atexit
from logging import getLogger
from typing import Optional, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...processors import PreProcessor, PostProcessor, Context
from ...checkpoint import CheckpointManager
from ...utils import convert_seconds_to_string_time
from .wrapper import PropOpWrapper

logger = getLogger("ditty_training")


@dataclass(kw_only=True)
class PropOpTrainerState:
    """Training state for PropOpTrainer."""
    epoch: int = 0
    steps: int = 0
    total_steps: int = 0
    global_loss: float = 0.0

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "steps": self.steps,
            "total_steps": self.total_steps,
            "global_loss": self.global_loss,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict.get("epoch", 0)
        self.steps = state_dict.get("steps", 0)
        self.total_steps = state_dict.get("total_steps", 0)
        self.global_loss = state_dict.get("global_loss", 0.0)


@dataclass(kw_only=True)
class PropOpTrainer:
    """
    Training loop for PropOp local learning.

    Unlike standard Trainer:
    - No optimizer (weight updates happen in PropOpWrapper.learn())
    - No backward pass (credit assignment replaces gradients)
    - No gradient accumulation (each batch updates weights directly)

    Pipeline pattern is preserved:
        batch -> preprocessors -> model.forward -> postprocessors -> learn(output, target)
    """
    model: PropOpWrapper
    dataset: DataLoader
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Pipeline
    preprocessors: List[PreProcessor] = field(default_factory=list)
    postprocessors: List[PostProcessor] = field(default_factory=list)
    target_key: str = "target"

    # Training config
    output_dir: str = "./output"
    checkpoint_every: int = 1000
    seed: Optional[int] = None
    metrics_logger: Optional[Any] = None
    log_every: int = 10
    shuffle_each_epoch: bool = True
    total_batches: Optional[int] = None

    # Pre-loaded state (from CheckpointManager)
    initial_state: Optional[PropOpTrainerState] = None

    def __post_init__(self):
        if self.seed:
            torch.manual_seed(self.seed)

        os.makedirs(self.output_dir, exist_ok=True)

        self.batch_size = self.dataset.batch_size
        self.preprocessors = self.preprocessors or []
        self.postprocessors = self.postprocessors or []

        # Move model to device
        self.model = self.model.to(self.device)

        # Use pre-loaded state if provided
        if self.initial_state is not None:
            self.state = self.initial_state
        else:
            self.state = PropOpTrainerState()

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.output_dir)
        self._checkpoint_iteration = self.checkpoint_manager.get_latest_checkpoint_num() or 0
        if self.initial_state is not None:
            self._checkpoint_iteration += 1

    def _save(self):
        """Save checkpoint."""
        logger.info(f"Saving checkpoint at step {self.state.steps} (total: {self.state.total_steps})")

        self.checkpoint_manager.save(
            checkpoint_num=self._checkpoint_iteration,
            model=self.model,
            optimizer=None,  # No optimizer in PropOp
            training_state=self.state.state_dict(),
            scheduler=None,
            scaler=None,
            loss_calculator=None,
            is_fsdp=False,
            rank=0,
            local_rank=0,
        )
        self._checkpoint_iteration += 1

    def _log_pipeline(self):
        """Log pipeline configuration."""
        from transformers.trainer_pt_utils import get_model_param_count

        logger.info("Pipeline (PropOp):")
        logger.info("  preprocessors:")
        for p in self.preprocessors:
            logger.info(f"    - {p}")
        logger.info(f"  model: PropOpWrapper({self.model.model.__class__.__name__}) ({get_model_param_count(self.model, trainable_only=True):,} params)")
        logger.info("  postprocessors:")
        for p in self.postprocessors:
            logger.info(f"    - {p}")
        logger.info("  learning: PropOp local credit assignment")

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Compute cross-entropy loss for logging only."""
        with torch.no_grad():
            return F.cross_entropy(output, target).item()

    def _compute_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Compute accuracy for logging."""
        with torch.no_grad():
            preds = output.argmax(dim=-1)
            return (preds == target).float().mean().item()

    def train(self, epochs: int = 1, max_steps: Optional[int] = None) -> float:
        """
        Run training loop.

        Args:
            epochs: Number of epochs to train
            max_steps: Optional max total steps (overrides epochs)

        Returns:
            Average loss over training
        """
        from transformers.trainer_pt_utils import get_model_param_count

        logger.info("***** Running PropOp training *****")
        try:
            logger.info(f"  Num examples = {len(self.dataset):,}")
        except TypeError:
            logger.info("  Num examples = unknown (iterable dataset)")
        logger.info(f"  Num Epochs = {epochs:,}")
        if max_steps:
            logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Batch size = {self.batch_size:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(self.model, trainable_only=True):,}")
        logger.info(f"  Learning rate = {self.model.config.lr}")
        logger.info(f"  Cofire: {self.model.config.use_cofire}, Lateral: {self.model.config.use_lateral}, Theta: {self.model.config.use_theta}")

        self.model.train()

        if self.total_batches is not None:
            total_batches = self.total_batches
        else:
            try:
                total_batches = len(self.dataset) * epochs
            except TypeError:
                total_batches = None

        start_time = time.time()
        atexit.register(self._save)

        for ep in range(self.state.epoch, epochs):
            dataset = self.dataset

            if self.shuffle_each_epoch and hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(ep)

            for batch in dataset:
                if batch is None:
                    break

                original_batch = batch
                ctx: Context = {
                    "epoch": ep,
                    "step": self.state.steps,
                    "total_steps": self.state.total_steps,
                    "device": self.device,
                    "original_batch": original_batch,
                    "model": self.model,
                }

                # Preprocessors
                for preprocessor in self.preprocessors:
                    result = preprocessor.process(batch, ctx)
                    if result[0] is None:
                        batch = None
                        break
                    batch, ctx = result

                if batch is None:
                    continue

                # Forward pass (hooks cache activations)
                with torch.no_grad():
                    output = self.model(batch)
                    if not isinstance(output, tuple):
                        output = (output,)

                # Postprocessors
                for postprocessor in self.postprocessors:
                    output, ctx = postprocessor.process(output, ctx)

                # Get target from context
                target = ctx[self.target_key]
                logits = output[0]

                # Compute per-sample losses for error-gain modulated learning
                with torch.no_grad():
                    per_sample_losses = F.cross_entropy(logits, target, reduction='none')

                # PropOp learning (replaces backward + optimizer)
                self.model.begin_batch()
                self.model.learn(logits, target, losses=per_sample_losses)
                self.model.end_batch()

                # Compute metrics for logging
                batch_loss = per_sample_losses.mean().item()
                batch_acc = self._compute_accuracy(logits, target)

                # Timing and progress
                time_elapsed = time.time() - start_time
                if total_batches is not None:
                    batches_per_epoch = total_batches // epochs if epochs > 0 else total_batches
                    total_batches_done = ep * batches_per_epoch + self.state.steps
                    current_epoch_decimal = total_batches_done / total_batches if total_batches > 0 else 0
                    batches_remaining = total_batches - total_batches_done
                    estimated_time_remaining = (
                        (time_elapsed / total_batches_done) * batches_remaining
                        if total_batches_done > 0 else 0
                    )
                    estimated_time_remaining_str = convert_seconds_to_string_time(estimated_time_remaining)
                    percent_done = (total_batches_done / total_batches) * 100 if total_batches > 0 else 0
                    batch_info = f"Batch {self.state.steps}/{batches_per_epoch}"
                    progress_info = f"{percent_done:.2f}% done | ETA: {estimated_time_remaining_str}"
                else:
                    current_epoch_decimal = ep + (self.state.steps / 1000)
                    batch_info = f"Batch {self.state.steps}"
                    progress_info = f"elapsed: {convert_seconds_to_string_time(time_elapsed)}"

                # Logging
                if self.state.steps % self.log_every == 0:
                    logger.info(
                        f"Epoch {current_epoch_decimal:.2f} | {batch_info} | "
                        f"loss: {batch_loss:.4f} | acc: {batch_acc:.4f} | {progress_info}"
                    )

                    if self.metrics_logger:
                        self.metrics_logger.log_scalar("train/loss", batch_loss, self.state.total_steps)
                        self.metrics_logger.log_scalar("train/accuracy", batch_acc, self.state.total_steps)
                        self.metrics_logger.log_scalar("train/epoch", current_epoch_decimal, self.state.total_steps)

                self.state.global_loss += batch_loss
                self.state.steps += 1
                self.state.total_steps += 1

                if max_steps is not None and self.state.total_steps >= max_steps:
                    break

                if self.state.steps % self.checkpoint_every == 0 and self.state.steps > 0:
                    self._save()

            self.state.epoch += 1
            self.state.steps = 0

        atexit.unregister(self._save)
        self._save()

        return self.state.global_loss / self.state.total_steps if self.state.total_steps > 0 else 0
