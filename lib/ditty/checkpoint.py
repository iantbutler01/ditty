import os
import pickle
import random
from dataclasses import dataclass, field
from logging import getLogger
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

logger = getLogger("ditty_checkpoint")


@dataclass
class Checkpoint:
    """Container for all checkpoint data."""
    model_state: Optional[Dict[str, Any]] = None
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    training_state: Dict[str, Any] = field(default_factory=dict)
    scaler_state: Optional[Dict[str, Any]] = None
    rng_states: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """
    Unified checkpoint manager for ditty training.

    Handles saving and loading of:
    - Model weights
    - Optimizer state
    - Scheduler state
    - Training state (epoch, steps, etc.)
    - Gradient scaler state
    - RNG states for reproducibility

    This replaces accelerate's save_state/load_state to give us control
    over the loading order (load before prepare() instead of after).
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoints_dir = os.path.join(output_dir, "checkpoints")

    def _get_checkpoint_path(self, checkpoint_num: int) -> str:
        return os.path.join(self.checkpoints_dir, f"checkpoint_{checkpoint_num}")

    def _get_latest_checkpoint_num(self) -> Optional[int]:
        if not os.path.exists(self.checkpoints_dir):
            return None

        checkpoint_dirs = []
        for name in os.listdir(self.checkpoints_dir):
            if name.startswith("checkpoint_"):
                try:
                    num = int(name.split("_")[1])
                    checkpoint_dirs.append(num)
                except (IndexError, ValueError):
                    continue

        if not checkpoint_dirs:
            return None

        return max(checkpoint_dirs)

    def save(
        self,
        checkpoint_num: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        training_state: Dict[str, Any],
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        is_fsdp: bool = False,
        rank: int = 0,
        local_rank: int = 0,
    ):
        """Save a complete training checkpoint."""
        checkpoint_path = self._get_checkpoint_path(checkpoint_num)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model weights (full state dict for FSDP)
        model_path = os.path.join(self.output_dir, "dist", "model.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Get the unwrapped model if compiled (avoids _orig_mod. prefix in state_dict)
        unwrapped_model = getattr(model, '_orig_mod', model)

        if is_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            from torch.distributed.tensor import DTensor

            # For FSDP2 with fully_shard, we need to use get_model_state_dict
            try:
                from torch.distributed.checkpoint.state_dict import get_model_state_dict
                model_state = get_model_state_dict(unwrapped_model)
                # Convert DTensors to regular tensors for portable checkpoints
                model_state = {
                    k: v.full_tensor().cpu() if isinstance(v, DTensor) else v.cpu()
                    for k, v in model_state.items()
                }
            except ImportError:
                # Fallback for older torch versions
                model_state = {k: v.cpu() for k, v in unwrapped_model.state_dict().items()}

            if rank == 0:
                torch.save(model_state, model_path)
        else:
            if rank == 0:
                torch.save(unwrapped_model.state_dict(), model_path)

        # Save optimizer state
        optimizer_path = os.path.join(checkpoint_path, "optimizer.bin")
        if is_fsdp:
            try:
                from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
                optim_state = get_optimizer_state_dict(model, optimizer)
            except ImportError:
                optim_state = optimizer.state_dict()
        else:
            optim_state = optimizer.state_dict()

        if rank == 0:
            torch.save(optim_state, optimizer_path)

        # Save scheduler state
        if scheduler is not None and rank == 0:
            scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
            torch.save(scheduler.state_dict(), scheduler_path)

        # Save training state
        if rank == 0:
            training_state_path = os.path.join(checkpoint_path, "training_state.pt")
            torch.save(training_state, training_state_path)

        # Save scaler state
        if scaler is not None and rank == 0:
            scaler_path = os.path.join(checkpoint_path, "scaler.pt")
            torch.save(scaler.state_dict(), scaler_path)

        # Save RNG states for this rank
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state(local_rank)

        rng_path = os.path.join(checkpoint_path, f"rng_state_{rank}.pt")
        torch.save(rng_state, rng_path)

        if rank == 0:
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load(self, checkpoint_num: Optional[int] = None) -> Optional[Checkpoint]:
        """
        Load a checkpoint. If checkpoint_num is None, loads the latest.
        Returns None if no checkpoint exists.
        """
        if checkpoint_num is None:
            checkpoint_num = self._get_latest_checkpoint_num()
            if checkpoint_num is None:
                return None

        checkpoint_path = self._get_checkpoint_path(checkpoint_num)
        if not os.path.exists(checkpoint_path):
            return None

        checkpoint = Checkpoint()

        # Load model weights
        model_path = os.path.join(self.output_dir, "dist", "model.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            # Strip _orig_mod. prefix added by torch.compile
            checkpoint.model_state = {
                k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
            }

        # Load optimizer state
        optimizer_path = os.path.join(checkpoint_path, "optimizer.bin")
        if os.path.exists(optimizer_path):
            checkpoint.optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)

        # Load scheduler state
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        if os.path.exists(scheduler_path):
            checkpoint.scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=False)

        # Load training state (new format or legacy accelerate format)
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(training_state_path):
            checkpoint.training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        else:
            # Try legacy accelerate format
            legacy_path = os.path.join(checkpoint_path, "custom_checkpoint_0.pkl")
            if os.path.exists(legacy_path):
                try:
                    checkpoint.training_state = torch.load(legacy_path, map_location="cpu", weights_only=False)
                    logger.info("Loaded training state from legacy accelerate format")
                except Exception as e:
                    logger.warning(f"Failed to load legacy training state: {e}")

        # Load scaler state
        scaler_path = os.path.join(checkpoint_path, "scaler.pt")
        if os.path.exists(scaler_path):
            checkpoint.scaler_state = torch.load(scaler_path, map_location="cpu", weights_only=False)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint

    def load_rng_state(self, checkpoint_num: Optional[int] = None, rank: int = 0, local_rank: int = 0):
        """Load and restore RNG states for a specific rank."""
        if checkpoint_num is None:
            checkpoint_num = self._get_latest_checkpoint_num()
            if checkpoint_num is None:
                return

        checkpoint_path = self._get_checkpoint_path(checkpoint_num)
        rng_path = os.path.join(checkpoint_path, f"rng_state_{rank}.pt")

        if not os.path.exists(rng_path):
            # Try legacy format
            rng_path = os.path.join(checkpoint_path, f"random_states_{rank}.pkl")
            if not os.path.exists(rng_path):
                return

        rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)

        # Handle our new format
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if "cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["cuda"], local_rank)

        # Handle accelerate format
        if "random_state" in rng_state:
            random.setstate(rng_state["random_state"])
        if "numpy_random_seed" in rng_state:
            np.random.set_state(rng_state["numpy_random_seed"])
        if "torch_manual_seed" in rng_state:
            torch.set_rng_state(rng_state["torch_manual_seed"])
        if "torch_cuda_manual_seed" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["torch_cuda_manual_seed"], local_rank)

    def get_latest_checkpoint_num(self) -> Optional[int]:
        return self._get_latest_checkpoint_num()

    def apply_to_model(self, checkpoint: Checkpoint, model: nn.Module):
        """Apply checkpoint model state to a model."""
        if checkpoint.model_state is not None:
            model.load_state_dict(checkpoint.model_state)
            logger.info("Loaded model weights from checkpoint")

    def apply_to_optimizer(self, checkpoint: Checkpoint, optimizer: torch.optim.Optimizer):
        """Apply checkpoint optimizer state to an optimizer."""
        if checkpoint.optimizer_state is not None:
            optimizer.load_state_dict(checkpoint.optimizer_state)
            logger.info("Loaded optimizer state from checkpoint")

    def apply_to_scheduler(self, checkpoint: Checkpoint, scheduler: torch.optim.lr_scheduler.LRScheduler):
        """Apply checkpoint scheduler state to a scheduler."""
        if checkpoint.scheduler_state is not None:
            scheduler.load_state_dict(checkpoint.scheduler_state)
            logger.info("Loaded scheduler state from checkpoint")

    def apply_to_scaler(self, checkpoint: Checkpoint, scaler: torch.amp.GradScaler):
        """Apply checkpoint scaler state to a gradient scaler."""
        if checkpoint.scaler_state is not None:
            scaler.load_state_dict(checkpoint.scaler_state)
            logger.info("Loaded scaler state from checkpoint")
