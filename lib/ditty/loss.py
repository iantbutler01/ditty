"""
Loss calculator abstraction for ditty trainers.

Architecture:
    batch -> preprocess -> model.forward -> postprocess -> loss_calc(model_output, ctx)

LossCalculator receives the full model output tuple and context dict,
allowing flexible loss computation across multiple model outputs.

Includes memory-efficient cross-entropy implementations:
- LigerFusedLinearCrossEntropy: LinkedIn's fused linear + CE (requires liger-kernel)
- CutCrossEntropy: Apple's cut cross-entropy (requires cut-cross-entropy)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DittyBase
from .processors import Context

# Optional imports for memory-efficient CE
try:
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False

try:
    from cut_cross_entropy import linear_cross_entropy
    CCE_AVAILABLE = True
except ImportError:
    CCE_AVAILABLE = False


from torch.distributed.fsdp import fully_shard

@dataclass
class LossOutput:
    loss: torch.Tensor
    metrics: Dict[str, float] = field(default_factory=dict)


class LossCalculator(DittyBase, nn.Module, ABC):
    def __init__(
        self,
        output_index: int = 0,
        target_key: str = "target",
        mask_key: Optional[str] = None,
        contract: str = "",
        fsdp: bool = False,
    ):
        DittyBase.__init__(self, contract=contract)
        nn.Module.__init__(self)
        self.output_index = output_index
        self.target_key = target_key
        self.mask_key = mask_key
        self._fsdp = fsdp

    def setup_fsdp(self):
        """Apply FSDP sharding if enabled and has parameters."""
        if self._fsdp and list(self.parameters()):
            self.to("cpu")
            fully_shard(self)

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


class FusedLinearCrossEntropyLoss(LossCalculator):
    """
    Memory-efficient fused linear + cross-entropy loss.

    Instead of materializing full [batch*seq, vocab] logits tensor, computes
    loss in chunks using either Liger kernel or Apple's cut-cross-entropy.

    Model output indices:
        - hidden_index: Hidden states before projection [batch, lines, tokens, hidden_dim]
        - weight_index: Projection weights [vocab_size, hidden_dim]
        - bias_index: Optional projection bias [vocab_size]
    """

    def __init__(
        self,
        hidden_index: int = 0,
        weight_index: int = 4,
        bias_index: int = 5,
        backend: str = "auto",
        ignore_index: int = -100,
        weight_attr_path: Optional[str] = None,
        bias_attr_path: Optional[str] = None,
        include_padding_in_normalization: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_index = hidden_index
        self.weight_index = weight_index
        self.bias_index = bias_index
        self.ignore_index = ignore_index
        self.weight_attr_path = weight_attr_path
        self.bias_attr_path = bias_attr_path
        self.include_padding_in_normalization = include_padding_in_normalization

        if backend == "auto":
            if LIGER_AVAILABLE:
                self.backend = "liger"
            elif CCE_AVAILABLE:
                self.backend = "cce"
            else:
                self.backend = "chunked"
        else:
            self.backend = backend

        if self.backend == "liger" and not LIGER_AVAILABLE:
            raise ImportError("liger-kernel not installed. Install with: pip install liger-kernel")
        if self.backend == "cce" and not CCE_AVAILABLE:
            raise ImportError("cut-cross-entropy not installed. Install with: pip install cut-cross-entropy")

    def compute(self, model_output: Tuple[Any, ...], ctx: Context) -> LossOutput:
        hidden = model_output[self.hidden_index]
        weight = model_output[self.weight_index]
        bias = model_output[self.bias_index] if self.bias_index < len(model_output) else None

        if hidden is None:
            device = ctx["device"]
            return LossOutput(
                loss=torch.tensor(0.0, device=device),
                metrics={"ce": 0.0}
            )

        target = self.get_target(ctx)
        mask = self.get_mask(ctx)

        # Flatten hidden: [batch, lines, tokens, hidden_dim] -> [batch*lines*tokens, hidden_dim]
        hidden = hidden.reshape(-1, hidden.shape[-1])

        if target.dim() > 1:
            target = target.reshape(-1)
        if mask is not None and mask.dim() > 1:
            mask = mask.reshape(-1)

        # Handle torch.compile + FSDP2: compiled forward returns traced tensors with zero storage
        # Get real weights from the model via _orig_mod if needed
        from torch.distributed.tensor import DTensor, Replicate

        def get_real_tensor(tensor, model, attr_path):
            """Get real tensor, handling torch.compile traced tensors and FSDP2 DTensors."""
            if tensor is None:
                return None
            # Check if this is a traced tensor with zero storage (torch.compile artifact)
            if tensor.untyped_storage().size() == 0:
                # Get the real model (unwrap torch.compile)
                real_model = getattr(model, '_orig_mod', model)
                # Navigate the attribute path to get real weight
                obj = real_model
                for attr in attr_path.split('.'):
                    obj = getattr(obj, attr)
                tensor = obj
            # Handle FSDP2 DTensor sharding - use redistribute for differentiable gather
            if isinstance(tensor, DTensor):
                # redistribute is differentiable, full_tensor is not
                tensor = tensor.redistribute(placements=[Replicate()] * tensor.device_mesh.ndim)
                tensor = tensor.to_local()
            elif hasattr(tensor, 'data') and isinstance(tensor.data, DTensor):
                tensor = tensor.data.redistribute(placements=[Replicate()] * tensor.data.device_mesh.ndim)
                tensor = tensor.to_local()
            return tensor

        model = ctx.get("model")
        if model is not None and self.weight_attr_path:
            weight = get_real_tensor(weight, model, self.weight_attr_path)
            if bias is not None and self.bias_attr_path:
                bias = get_real_tensor(bias, model, self.bias_attr_path)

        if self.backend == "liger":
            loss = self._compute_liger(hidden, weight, bias, target, mask)
        elif self.backend == "cce":
            loss = self._compute_cce(hidden, weight, bias, target, mask)
        else:
            loss = self._compute_chunked(hidden, weight, bias, target, mask)

        return LossOutput(loss=loss, metrics={"ce": loss.item()})

    def _compute_liger(self, hidden, weight, bias, target, mask):
        loss, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
            hidden, weight, target, bias,
            None,  # ce_weight
            self.ignore_index,
            0.0,  # lse_square_scale
            0.0,  # label_smoothing
            "mean" if mask is None else "none",
        )
        if mask is not None:
            divisor = mask.numel() if self.include_padding_in_normalization else mask.sum().clamp(min=1)
            loss = (loss * mask).sum() / divisor
        return loss

    def _compute_cce(self, hidden, weight, bias, target, mask):
        loss = linear_cross_entropy(
            hidden, weight.T, target,
            bias=bias,
            reduction="mean" if mask is None else "none",
        )
        if mask is not None:
            divisor = mask.numel() if self.include_padding_in_normalization else mask.sum().clamp(min=1)
            loss = (loss * mask).sum() / divisor
        return loss

    def _compute_chunked(self, hidden, weight, bias, target, mask, chunk_size: int = 4096):
        """Fallback chunked implementation when neither library is available."""
        device = hidden.device
        total_loss = torch.tensor(0.0, device=device)
        num_tokens = hidden.shape[0]

        for i in range(0, num_tokens, chunk_size):
            end = min(i + chunk_size, num_tokens)
            h_chunk = hidden[i:end]
            t_chunk = target[i:end]
            m_chunk = mask[i:end] if mask is not None else None

            logits_chunk = F.linear(h_chunk, weight, bias)

            if m_chunk is not None:
                loss_chunk = F.cross_entropy(logits_chunk, t_chunk, reduction="none")
                total_loss = total_loss + (loss_chunk * m_chunk).sum()
            else:
                loss_chunk = F.cross_entropy(logits_chunk, t_chunk, reduction="sum")
                total_loss = total_loss + loss_chunk

        if mask is not None:
            divisor = num_tokens if self.include_padding_in_normalization else mask.sum().clamp(min=1)
        else:
            divisor = num_tokens
        return total_loss / divisor


class CompositeLoss(LossCalculator):
    """Combine multiple loss calculators with weights."""

    def __init__(self, losses: list[tuple[LossCalculator, float]], fsdp: bool = False):
        super().__init__(contract="", fsdp=fsdp)
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


class UncertaintyWeightedLoss(CompositeLoss):
    """
    Uncertainty-weighted multi-task loss (Kendall et al. 2018).

    Learns task-specific log-variances that automatically balance losses.
    Loss = sum_i( exp(-log_var_i) * loss_i + log_var_i )

    The log_var_i terms act as regularization to prevent precision going to 0.
    """

    def __init__(self, losses: Sequence[LossCalculator], fsdp: bool = False):
        super().__init__([(l, 1.0) for l in losses], fsdp=fsdp)
        self.log_vars = nn.Parameter(torch.zeros(len(losses)))

    def compute(self, model_output: Tuple[Any, ...], ctx: Context) -> LossOutput:
        from torch.distributed.tensor import DTensor, Replicate

        device = ctx.get("device", "cuda")
        log_vars = self.log_vars.to(device)

        total_loss = log_vars.sum() * 0
        all_metrics = {}

        for i, (loss_calc, _) in enumerate(self.losses):
            output = loss_calc.compute(model_output, ctx)
            loss = output.loss

            # Convert plain tensor to DTensor if log_vars is a DTensor (FSDP2)
            if isinstance(log_vars, DTensor) and not isinstance(loss, DTensor):
                loss = DTensor.from_local(
                    loss,
                    device_mesh=log_vars.device_mesh,
                    placements=[Replicate()] * log_vars.device_mesh.ndim,
                )

            # Precision (weight) is unclamped - can be > 1 when log_var < 0
            precision = torch.exp(-log_vars[i])
            # Regularization only penalizes downweighting (log_var > 0)
            reg_term = log_vars[i].clamp(min=0.0)
            weighted_loss = 0.5 * precision * loss + 0.5 * reg_term
            total_loss = total_loss + weighted_loss

            for k, v in output.metrics.items():
                all_metrics[f"{loss_calc.name}/{k}"] = v
            all_metrics[f"{loss_calc.name}/weight"] = precision.item()
            all_metrics[f"{loss_calc.name}/log_var"] = log_vars[i].item()

        all_metrics["total"] = total_loss.item()
        return LossOutput(loss=total_loss, metrics=all_metrics)