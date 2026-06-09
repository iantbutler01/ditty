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
import inspect
from typing import Dict, Tuple, Optional, Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard

from .base import DittyBase
from .grpo import GRPOConfig, compute_grpo_loss
from .projection import extract_last_hidden_state
from .projection import materialize_model_tensor
from .projection import resolve_output_embeddings
from .projection import resolve_output_projection
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


def resolve_fused_ce_backend(backend: str) -> str:
    if backend == "auto":
        if LIGER_AVAILABLE:
            return "liger"
        if CCE_AVAILABLE:
            return "cce"
        return "chunked"
    if backend == "liger" and not LIGER_AVAILABLE:
        raise ImportError("liger-kernel not installed. Install with: pip install liger-kernel")
    if backend == "cce" and not CCE_AVAILABLE:
        raise ImportError("cut-cross-entropy not installed. Install with: pip install cut-cross-entropy")
    if backend not in {"liger", "cce", "chunked"}:
        raise ValueError(f"Unknown fused CE backend: {backend}")
    return backend


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    backend: str = "auto",
    ignore_index: int = -100,
    include_padding_in_normalization: bool = False,
    chunk_size: int = 4096,
    output_projection: nn.Module | None = None,
) -> torch.Tensor:
    """Compute linear projection + cross entropy without materializing full logits."""

    if hidden.dim() > 2:
        hidden = hidden.reshape(-1, hidden.shape[-1])
    if target.dim() > 1:
        target = target.reshape(-1)
    if mask is not None and mask.dim() > 1:
        mask = mask.reshape(-1)

    backend = resolve_fused_ce_backend(backend)
    if backend == "liger":
        liger_result = LigerFusedLinearCrossEntropyFunction.apply(
            hidden,
            weight,
            target,
            bias,
            None,  # ce_weight
            ignore_index,
            0.0,  # lse_square_scale
            0.0,  # label_smoothing
            "mean" if mask is None else "none",
        )
        loss = liger_result[0] if isinstance(liger_result, tuple) else liger_result
        if mask is not None:
            divisor = hidden.shape[0] if include_padding_in_normalization else mask.sum().clamp(min=1)
            loss = (loss * mask).sum() / divisor
        return loss

    if backend == "cce":
        if bias is not None:
            return _chunked_linear_cross_entropy(
                hidden,
                weight,
                target,
                bias=bias,
                mask=mask,
                ignore_index=ignore_index,
                include_padding_in_normalization=include_padding_in_normalization,
                chunk_size=chunk_size,
                output_projection=output_projection,
            )
        loss = linear_cross_entropy(
            hidden,
            weight,
            target,
            ignore_index=ignore_index,
            reduction="mean" if mask is None else "none",
        )
        if mask is not None:
            divisor = hidden.shape[0] if include_padding_in_normalization else mask.sum().clamp(min=1)
            loss = (loss * mask).sum() / divisor
        return loss

    return _chunked_linear_cross_entropy(
        hidden,
        weight,
        target,
        bias=bias,
        mask=mask,
        ignore_index=ignore_index,
        include_padding_in_normalization=include_padding_in_normalization,
        chunk_size=chunk_size,
        output_projection=output_projection,
    )


def _is_float8_backed_tensor(value: Any) -> bool:
    tensor = value
    if hasattr(tensor, "_tensor"):
        tensor = getattr(tensor, "_tensor")
    if isinstance(tensor, torch.Tensor) and tensor.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
        return True
    data = getattr(tensor, "data", None)
    if isinstance(data, torch.Tensor) and data.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
        return True
    return False


def _chunked_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    output_projection: nn.Module | None = None,
    ignore_index: int = -100,
    include_padding_in_normalization: bool = False,
    chunk_size: int = 4096,
) -> torch.Tensor:
    device = hidden.device
    total_loss = torch.tensor(0.0, device=device)
    num_tokens = hidden.shape[0]

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        h_chunk = hidden[start:end]
        t_chunk = target[start:end]
        m_chunk = mask[start:end] if mask is not None else None

        if output_projection is not None:
            logits_chunk = output_projection(h_chunk)
        else:
            logits_chunk = F.linear(h_chunk, weight, bias)
        if logits_chunk.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
            logits_chunk = logits_chunk.float()
        if m_chunk is not None:
            loss_chunk = F.cross_entropy(
                logits_chunk,
                t_chunk,
                reduction="none",
                ignore_index=ignore_index,
            )
            total_loss = total_loss + (loss_chunk * m_chunk).sum()
        else:
            total_loss = total_loss + F.cross_entropy(
                logits_chunk,
                t_chunk,
                reduction="sum",
                ignore_index=ignore_index,
            )

    if mask is not None:
        divisor = num_tokens if include_padding_in_normalization else mask.sum().clamp(min=1)
    else:
        divisor = num_tokens
    return total_loss / divisor


def _projection_weight_and_bias(projection: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(projection, tuple):
        if len(projection) == 2:
            return projection
        if len(projection) == 1:
            return projection[0], None
    weight = getattr(projection, "weight", None)
    bias = getattr(projection, "bias", None)
    if weight is None and hasattr(projection, "head"):
        head = projection.head
        weight = getattr(head, "weight", None)
        bias = getattr(head, "bias", None)
    if weight is None and hasattr(projection, "output"):
        output = projection.output
        weight = getattr(output, "weight", None)
        bias = getattr(output, "bias", None)
    if weight is None:
        raise RuntimeError(f"Could not resolve projection weight from {type(projection).__name__}.")
    return weight, bias


def _derive_position_ids(input_ids: torch.Tensor, ctx: Context) -> torch.Tensor:
    forward_kwargs = ctx.get("forward_kwargs") or {}
    position_ids = ctx.get("position_ids")
    if position_ids is None:
        position_ids = forward_kwargs.get("position_ids")
    if position_ids is not None:
        return position_ids

    attention_mask = forward_kwargs.get("attention_mask")
    if attention_mask is not None and getattr(attention_mask, "shape", None) == input_ids.shape:
        positions = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
        return positions.masked_fill(attention_mask == 0, 0)

    return torch.arange(
        input_ids.shape[1],
        device=input_ids.device,
        dtype=torch.long,
    ).unsqueeze(0).expand_as(input_ids)


def _valid_token_count(mask: torch.Tensor | None, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    if mask is not None:
        return mask.sum()
    return target.ne(ignore_index).sum()


def _forward_mtp_step(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    positions: torch.Tensor | None,
    previous_hidden_states: torch.Tensor,
    step_idx: int,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    kwargs = {
        "input_ids": input_ids,
        "positions": positions,
        "previous_hidden_states": previous_hidden_states,
        "step_idx": step_idx,
    }
    signature = inspect.signature(model.forward_mtp_step)
    if "attention_mask" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        kwargs["attention_mask"] = attention_mask
    return model.forward_mtp_step(**kwargs)


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
        prediction = model_output[self.output_index]
        return self._extract_prediction_tensor(prediction)

    def get_target(self, ctx: Context) -> torch.Tensor:
        return ctx[self.target_key]

    def get_mask(self, ctx: Context) -> Optional[torch.Tensor]:
        return ctx.get(self.mask_key) if self.mask_key else None

    @staticmethod
    def _extract_prediction_tensor(prediction: Any) -> torch.Tensor:
        if isinstance(prediction, torch.Tensor):
            return prediction
        if hasattr(prediction, "logits") and isinstance(prediction.logits, torch.Tensor):
            return prediction.logits
        if isinstance(prediction, dict):
            logits = prediction.get("logits")
            if isinstance(logits, torch.Tensor):
                return logits
        raise TypeError(
            "Could not extract prediction tensor from model output. "
            f"Expected a tensor or an object with `.logits`, got {type(prediction).__name__}."
        )

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


class GRPOLoss(LossCalculator):
    def __init__(
        self,
        config: Optional[GRPOConfig] = None,
        old_logprob_key: str = "old_logprobs",
        reference_logprob_key: str = "reference_logprobs",
        advantage_key: str = "advantages",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config or GRPOConfig()
        self.old_logprob_key = old_logprob_key
        self.reference_logprob_key = reference_logprob_key
        self.advantage_key = advantage_key

    def compute(self, model_output: Tuple[Any, ...], ctx: Context) -> LossOutput:
        labels = self.get_target(ctx)
        mask = self.get_mask(ctx)
        if mask is None:
            mask = labels.ne(-100)

        logits = None
        hidden_states = None
        if self.config.logprob_source == "hidden_states":
            hidden_states = extract_last_hidden_state(model_output[self.output_index])
        else:
            logits = self.get_prediction(model_output)

        reference_logprobs = ctx.get(self.reference_logprob_key)
        if reference_logprobs is None and self.reference_logprob_key != "ref_logprobs":
            reference_logprobs = ctx.get("ref_logprobs")

        loss, metrics = compute_grpo_loss(
            logits=logits,
            hidden_states=hidden_states,
            labels=labels,
            mask=mask,
            old_logprobs=ctx[self.old_logprob_key],
            advantages=ctx[self.advantage_key],
            reference_logprobs=reference_logprobs,
            config=self.config,
            model=ctx.get("model"),
            logits_positions=ctx.get("logits_positions"),
        )
        for name, value in dict(ctx.get("rollout_metrics") or {}).items():
            if isinstance(value, (int, float)):
                metrics[f"rollout_{name}" if not str(name).startswith("rollout_") else str(name)] = float(value)
        return LossOutput(loss=loss, metrics=metrics)


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
        chunk_size: int = 4096,
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
        self.chunk_size = int(chunk_size)

        self.backend = resolve_fused_ce_backend(backend)

    def compute(self, model_output: Tuple[Any, ...], ctx: Context) -> LossOutput:
        hidden = model_output[self.hidden_index]
        weight = model_output[self.weight_index] if self.weight_index < len(model_output) else None
        bias = model_output[self.bias_index] if self.bias_index < len(model_output) else None

        if hidden is None:
            device = ctx["device"]
            return LossOutput(
                loss=torch.tensor(0.0, device=device),
                metrics={"ce": 0.0}
            )
        hidden = extract_last_hidden_state(hidden)

        target = self.get_target(ctx)
        mask = self.get_mask(ctx)

        # Flatten hidden: [batch, lines, tokens, hidden_dim] -> [batch*lines*tokens, hidden_dim]
        hidden = hidden.reshape(-1, hidden.shape[-1])

        if target.dim() > 1:
            target = target.reshape(-1)
        if mask is not None and mask.dim() > 1:
            mask = mask.reshape(-1)

        model = ctx.get("model")
        if model is not None and self.weight_attr_path:
            weight = materialize_model_tensor(weight, model=model, attr_path=self.weight_attr_path)
            if bias is not None and self.bias_attr_path:
                bias = materialize_model_tensor(bias, model=model, attr_path=self.bias_attr_path)
        if weight is None and model is not None:
            weight, resolved_bias = resolve_output_embeddings(model)
            if bias is None:
                bias = resolved_bias
        if weight is None:
            raise RuntimeError(
                "FusedLinearCrossEntropyLoss requires projection weights in the model output "
                "or a model in ctx exposing output embeddings."
            )

        output_projection = None
        if model is not None and _is_float8_backed_tensor(weight):
            output_projection = resolve_output_projection(model)

        loss = fused_linear_cross_entropy(
            hidden,
            weight,
            target,
            bias=bias,
            mask=mask,
            backend="chunked" if output_projection is not None else self.backend,
            ignore_index=self.ignore_index,
            include_padding_in_normalization=self.include_padding_in_normalization,
            output_projection=output_projection,
            chunk_size=self.chunk_size,
        )

        ce_loss = loss
        metrics = {"ce": ce_loss.item()}
        aux_loss = getattr(model_output, "aux_loss", None)
        if aux_loss is not None:
            coef = 1.0
            config = getattr(model, "config", None) if model is not None else None
            if config is not None:
                text_config = getattr(config, "text_config", None)
                coef = float(
                    getattr(config, "router_aux_loss_coef", None)
                    or getattr(text_config, "router_aux_loss_coef", None)
                    or coef
                )
            router_aux_loss = aux_loss.to(device=ce_loss.device, dtype=ce_loss.dtype)
            loss = ce_loss + (coef * router_aux_loss)
            metrics["router_aux_loss"] = float(router_aux_loss.detach().item())
            metrics["router_aux_loss_coef"] = float(coef)
            metrics["loss"] = loss.item()

        return LossOutput(loss=loss, metrics=metrics)

    def _compute_liger(self, hidden, weight, bias, target, mask):
        return fused_linear_cross_entropy(
            hidden,
            weight,
            target,
            bias=bias,
            mask=mask,
            backend="liger",
            ignore_index=self.ignore_index,
            include_padding_in_normalization=self.include_padding_in_normalization,
        )

    def _compute_cce(self, hidden, weight, bias, target, mask):
        return fused_linear_cross_entropy(
            hidden,
            weight,
            target,
            bias=bias,
            mask=mask,
            backend="cce",
            ignore_index=self.ignore_index,
            include_padding_in_normalization=self.include_padding_in_normalization,
        )

    def _compute_chunked(self, hidden, weight, bias, target, mask, chunk_size: int = 4096):
        """Fallback chunked implementation when neither fused library is available."""
        return fused_linear_cross_entropy(
            hidden,
            weight,
            target,
            bias=bias,
            mask=mask,
            backend="chunked",
            ignore_index=self.ignore_index,
            include_padding_in_normalization=self.include_padding_in_normalization,
            chunk_size=chunk_size,
        )


class MTPAuxFusedCrossEntropyLoss(LossCalculator):
    """Future-token auxiliary CE over model-provided MTP heads using fused linear CE."""

    def __init__(
        self,
        hidden_index: int = 0,
        input_ids_key: str = "input_ids",
        position_ids_key: str = "position_ids",
        attention_mask_key: str = "attention_mask",
        depth: int = 1,
        backend: str = "auto",
        ignore_index: int = -100,
        beta: float = 0.6,
        offset_weights: Optional[Sequence[float]] = None,
        include_padding_in_normalization: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if depth < 1:
            raise ValueError("MTP depth must be at least 1.")
        self.hidden_index = hidden_index
        self.input_ids_key = input_ids_key
        self.position_ids_key = position_ids_key
        self.attention_mask_key = attention_mask_key
        self.depth = int(depth)
        self.backend = resolve_fused_ce_backend(backend)
        self.ignore_index = ignore_index
        self.beta = float(beta)
        self.offset_weights = tuple(float(value) for value in offset_weights) if offset_weights is not None else None
        self.include_padding_in_normalization = include_padding_in_normalization

    def _weights(self, device: torch.device) -> torch.Tensor:
        if self.offset_weights is not None:
            if len(self.offset_weights) != self.depth:
                raise ValueError(
                    f"Expected {self.depth} MTP offset weights, got {len(self.offset_weights)}."
                )
            weights = torch.tensor(self.offset_weights, device=device, dtype=torch.float32)
        elif self.depth == 1:
            weights = torch.ones(1, device=device, dtype=torch.float32)
        else:
            weights = torch.tensor(
                [self.beta ** idx for idx in range(self.depth)],
                device=device,
                dtype=torch.float32,
            )
        return weights / weights.sum().clamp(min=1e-12)

    def compute(self, model_output: Tuple[Any, ...], ctx: Context) -> LossOutput:
        model = ctx.get("model")
        if model is None:
            raise RuntimeError("MTPAuxFusedCrossEntropyLoss requires ctx['model'].")
        if not hasattr(model, "forward_mtp_step"):
            raise RuntimeError(
                f"Model {type(model).__name__} does not implement forward_mtp_step()."
            )
        if not hasattr(model, "get_mtp_output_embeddings"):
            raise RuntimeError(
                f"Model {type(model).__name__} does not implement get_mtp_output_embeddings()."
            )

        hidden = extract_last_hidden_state(model_output[self.hidden_index])
        if hidden.dim() != 3:
            raise ValueError(
                f"MTP auxiliary loss expects hidden shape [batch, seq, hidden], got {tuple(hidden.shape)}."
            )

        input_ids = ctx[self.input_ids_key]
        target = self.get_target(ctx)
        mask = self.get_mask(ctx)
        if mask is None:
            mask = target.ne(self.ignore_index)
        positions = ctx.get(self.position_ids_key)
        if positions is None:
            positions = _derive_position_ids(input_ids, ctx)

        forward_kwargs = ctx.get("forward_kwargs") or {}
        attention_mask = ctx.get(self.attention_mask_key, forward_kwargs.get("attention_mask"))

        device = hidden.device
        weights = self._weights(device=device)
        total_loss = hidden.sum() * 0.0
        metrics: Dict[str, float] = {}
        previous_hidden = hidden

        for step_idx in range(self.depth):
            shift = step_idx + 1
            if input_ids.shape[1] <= shift or previous_hidden.shape[1] <= 1:
                step_loss = hidden.sum() * 0.0
                valid_tokens = torch.tensor(0, device=device)
                metrics[f"mtp_{shift}_ce"] = 0.0
                metrics[f"tokens/mtp_{shift}_valid"] = 0.0
                total_loss = total_loss + weights[step_idx] * step_loss
                continue

            step_input_ids = input_ids[:, shift:]
            step_positions = positions[:, shift:] if positions is not None else None
            step_prev_hidden = previous_hidden[:, :-1]
            step_target = target[:, shift:]
            step_mask = mask[:, shift:] if mask is not None else None
            step_attention_mask = (
                attention_mask[:, shift:]
                if attention_mask is not None and getattr(attention_mask, "dim", lambda: 0)() == 2
                else None
            )

            seq_len = min(step_input_ids.shape[1], step_prev_hidden.shape[1], step_target.shape[1])
            step_input_ids = step_input_ids[:, :seq_len]
            step_prev_hidden = step_prev_hidden[:, :seq_len]
            step_target = step_target[:, :seq_len]
            if step_positions is not None:
                step_positions = step_positions[:, :seq_len]
            if step_mask is not None:
                step_mask = step_mask[:, :seq_len]
            if step_attention_mask is not None:
                step_attention_mask = step_attention_mask[:, :seq_len]

            previous_hidden = _forward_mtp_step(
                model,
                input_ids=step_input_ids,
                positions=step_positions,
                previous_hidden_states=step_prev_hidden,
                step_idx=step_idx,
                attention_mask=step_attention_mask,
            )
            ce_hidden = previous_hidden
            if hasattr(model, "prepare_mtp_hidden_for_output"):
                ce_hidden = model.prepare_mtp_hidden_for_output(ce_hidden, step_idx=step_idx)

            projection = model.get_mtp_output_embeddings(step_idx)
            weight, bias = _projection_weight_and_bias(projection)
            valid_tokens = _valid_token_count(step_mask, step_target, self.ignore_index)
            if int(valid_tokens.detach().item()) == 0:
                step_loss = previous_hidden.sum() * 0.0
            else:
                step_loss = fused_linear_cross_entropy(
                    ce_hidden,
                    weight,
                    step_target,
                    bias=bias,
                    mask=step_mask,
                    backend=self.backend,
                    ignore_index=self.ignore_index,
                    include_padding_in_normalization=self.include_padding_in_normalization,
                )

            total_loss = total_loss + weights[step_idx] * step_loss
            metrics[f"mtp_{shift}_ce"] = float(step_loss.detach().item())
            metrics[f"tokens/mtp_{shift}_valid"] = float(valid_tokens.detach().item())

        metrics["mtp_total"] = float(total_loss.detach().item())
        return LossOutput(loss=total_loss, metrics=metrics)


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
