"""
Loss calculator abstraction for ditty trainers.

Architecture:
    batch -> preprocess -> model.forward -> postprocess -> loss_calc(model_output, ctx)

LossCalculator receives the full model output tuple and context dict,
allowing flexible loss computation across multiple model outputs.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn.functional as F

from .base import DittyBase
from .processors import Context


@dataclass
class LossOutput:
    loss: torch.Tensor
    metrics: Dict[str, float] = field(default_factory=dict)


class LossCalculator(DittyBase, ABC):
    def __init__(
        self,
        output_index: int = 0,
        target_key: str = "target",
        mask_key: Optional[str] = None,
        contract: str = "",
    ):
        super().__init__(contract=contract)
        self.output_index = output_index
        self.target_key = target_key
        self.mask_key = mask_key

    def get_prediction(self, model_output: Tuple[Any, ...]) -> torch.Tensor:
        return model_output[self.output_index]

    def get_target(self, ctx: Context) -> torch.Tensor:
        return ctx[self.target_key]

    def get_mask(self, ctx: Context) -> Optional[torch.Tensor]:
        return ctx.get(self.mask_key) if self.mask_key else None

    @abstractmethod
    def compute(self, model_output: Tuple[Any, ...], ctx: Context) -> LossOutput:
        """
        Compute loss from model output and context.

        Args:
            model_output: Tuple of tensors from model forward pass
            ctx: Context dict populated by preprocessors

        Returns:
            LossOutput with loss tensor and metrics dict
        """
        pass


class ReductionLoss(LossCalculator, ABC):
    """Base for losses with reduction and mask support (MSE, L1, etc)."""

    def __init__(self, reduction: str = "mean", mask_key: str = "mask", **kwargs):
        super().__init__(mask_key=mask_key, **kwargs)
        self.reduction = reduction

    def apply_mask(self, loss: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            return loss.sum() / mask.sum().clamp(min=1) if self.reduction == "mean" else loss.sum()
        return loss


class MSELoss(ReductionLoss):
    def compute(self, model_output: Tuple[Any, ...], ctx: Context) -> LossOutput:
        pred, target, mask = self.get_prediction(model_output), self.get_target(ctx), self.get_mask(ctx)
        if mask is not None:
            loss = F.mse_loss(pred * mask, target * mask, reduction="none")
            loss = self.apply_mask(loss, mask)
        else:
            loss = F.mse_loss(pred, target, reduction=self.reduction)
        return LossOutput(loss=loss, metrics={"mse": loss.item()})


class L1Loss(ReductionLoss):
    def compute(self, model_output: Tuple[Any, ...], ctx: Context) -> LossOutput:
        pred, target, mask = self.get_prediction(model_output), self.get_target(ctx), self.get_mask(ctx)
        if mask is not None:
            loss = F.l1_loss(pred * mask, target * mask, reduction="none")
            loss = self.apply_mask(loss, mask)
        else:
            loss = F.l1_loss(pred, target, reduction=self.reduction)
        return LossOutput(loss=loss, metrics={"l1": loss.item()})


class CrossEntropyLoss(LossCalculator):
    def __init__(self, ignore_index: int = -100, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index

    def compute(self, model_output: Tuple[Any, ...], ctx: Context) -> LossOutput:
        pred, target, mask = self.get_prediction(model_output), self.get_target(ctx), self.get_mask(ctx)
        if pred.dim() > 2:
            pred = pred.reshape(-1, pred.size(-1))
        if target.dim() > 1:
            target = target.reshape(-1)
        if mask is not None:
            mask = mask.reshape(-1) if mask.dim() > 1 else mask
            loss_per_token = F.cross_entropy(pred, target, reduction="none")
            loss = (loss_per_token * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index)
        return LossOutput(loss=loss, metrics={"ce": loss.item()})


class CompositeLoss(LossCalculator):
    """Combine multiple loss calculators with weights."""

    def __init__(self, losses: list[tuple[LossCalculator, float]]):
        super().__init__(contract="")
        self.losses = losses

    def compute(self, model_output: Tuple[Any, ...], ctx: Context) -> LossOutput:
        device = ctx.get("device", "cuda")
        total_loss = torch.tensor(0.0, device=device)
        all_metrics = {}

        for loss_calc, weight in self.losses:
            if weight == 0.0:
                continue
            output = loss_calc.compute(model_output, ctx)
            total_loss = total_loss + weight * output.loss
            for k, v in output.metrics.items():
                all_metrics[f"{loss_calc.name}/{k}"] = v

        all_metrics["total"] = total_loss.item()
        return LossOutput(loss=total_loss, metrics=all_metrics)