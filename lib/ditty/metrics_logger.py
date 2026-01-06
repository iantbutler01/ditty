"""Unified metrics logging for ditty training with TensorBoard and WandB support."""
import os
from logging import getLogger
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

logger = getLogger("ditty_metrics_logger")


class MetricsLogger:
    """Unified metrics logger supporting TensorBoard and WandB.

    Handles:
    - Loss/metrics scalar logging
    - Gradient histograms and statistics
    - Learning rate tracking
    - Model parameter watching (WandB)
    """

    def __init__(
        self,
        output_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "ditty-training",
        wandb_config: Optional[dict] = None,
        log_gradients: bool = False,
        gradient_log_every: int = 20,
        gradient_log_histograms: bool = True,
        gradient_components: Optional[List[str]] = None,
    ):
        self.output_dir = output_dir
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.log_gradients_enabled = log_gradients
        self.gradient_log_every = gradient_log_every
        self.gradient_log_histograms = gradient_log_histograms
        self.gradient_components = gradient_components or ["decoder", "denoiser"]

        self._tb_writer = None
        self._wandb_initialized = False
        self._model_watched = False

        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging to {log_dir}")

        if use_wandb:
            import wandb
            if not wandb.run:
                wandb.init(project=wandb_project, config=wandb_config)
            self._wandb_initialized = True
            logger.info(f"WandB logging enabled (project: {wandb_project})")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        if self._tb_writer:
            self._tb_writer.add_scalar(tag, value, step)
        if self.use_wandb:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_scalars(self, main_tag: str, values: Dict[str, float], step: int):
        """Log multiple scalars under a common tag."""
        if self._tb_writer:
            self._tb_writer.add_scalars(main_tag, values, step)
        if self.use_wandb:
            import wandb
            wandb.log({f"{main_tag}/{k}": v for k, v in values.items()}, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        """Log a dictionary of metrics with a prefix."""
        for name, value in metrics.items():
            self.log_scalar(f"{prefix}/{name}", value, step)

    def log_lr(self, optimizer: torch.optim.Optimizer, step: int):
        """Log learning rate from optimizer."""
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group.get("lr", 0)
            tag = "train/lr" if i == 0 else f"train/lr_group_{i}"
            self.log_scalar(tag, lr, step)

    def log_grad_norm(self, grad_norm: float, step: int):
        """Log gradient norm (typically after clipping)."""
        self.log_scalar("train/grad_norm", grad_norm, step)

    def log_model_watch(self, model: nn.Module):
        """Enable WandB automatic gradient tracking."""
        if self.use_wandb and not self._model_watched:
            import wandb
            wandb.watch(model, log="all", log_freq=self.gradient_log_every)
            self._model_watched = True
            logger.info("WandB model watch enabled")

    def _to_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Convert DTensor or other tensor subclasses to regular tensor."""
        # Handle DTensor (FSDP2) - use to_local() not full_tensor()
        # because full_tensor() is collective and we're only on rank 0
        if hasattr(t, 'to_local'):
            t = t.to_local()
        return t.detach().cpu().float()

    def log_gradients(self, model: nn.Module, step: int):
        """Log gradient histograms and scalar stats for model parameters."""
        if not self.log_gradients_enabled:
            return

        # Unwrap compiled/wrapped models
        unwrapped = getattr(model, '_orig_mod', model)
        unwrapped = getattr(unwrapped, 'model', unwrapped)

        for comp_name in self.gradient_components:
            component = getattr(unwrapped, comp_name, None)
            if component is None:
                continue

            for param_name, param in component.named_parameters():
                if param.grad is None:
                    continue

                grad = self._to_tensor(param.grad)
                full_name = f"{comp_name}/{param_name}"

                if self.gradient_log_histograms and self._tb_writer:
                    if grad.numel() > 0 and not torch.isnan(grad).any():
                        self._tb_writer.add_histogram(f"grads/{full_name}", grad, step)

                stats = {
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "min": grad.min().item(),
                    "max": grad.max().item(),
                    "norm": grad.norm().item(),
                    "nonzero_pct": (grad != 0).float().mean().item() * 100,
                }

                if self._tb_writer:
                    for stat_name, stat_val in stats.items():
                        self._tb_writer.add_scalar(f"grads/{full_name}/{stat_name}", stat_val, step)

                if self.use_wandb:
                    import wandb
                    wandb.log({f"grads/{full_name}/{k}": v for k, v in stats.items()}, step=step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log a histogram of values."""
        values = self._to_tensor(values)
        if self._tb_writer:
            self._tb_writer.add_histogram(tag, values, step)
        if self.use_wandb:
            import wandb
            wandb.log({tag: wandb.Histogram(values.numpy())}, step=step)

    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        if self._tb_writer:
            self._tb_writer.add_text(tag, text, step)
        if self.use_wandb:
            import wandb
            wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def flush(self):
        """Flush pending writes."""
        if self._tb_writer:
            self._tb_writer.flush()

    def close(self):
        """Close writers."""
        if self._tb_writer:
            self._tb_writer.close()
            self._tb_writer = None
        if self.use_wandb and self._wandb_initialized:
            import wandb
            if wandb.run:
                wandb.finish()
            self._wandb_initialized = False
