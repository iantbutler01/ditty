from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Sequence

import torch

from .projection import (
    extract_last_hidden_state,
    gather_selected_logprobs_from_hidden,
    resolve_output_projection,
)


@dataclass(frozen=True)
class GRPOConfig:
    clip_epsilon: float = 0.2
    kl_beta: float = 0.04
    epsilon: float = 1e-8
    normalize_advantages: bool = True
    center_advantages: bool = True
    kl_estimator: str = "low_variance"
    logprob_source: str = "logits"
    logprob_backend: str = "selective"
    logprob_chunk_size: int = 128


@dataclass(frozen=True)
class RolloutSample:
    group_id: str
    sample_id: str
    prompt: str
    completion: str
    reward: float
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutGroup:
    group_id: str
    samples: list[RolloutSample]

    def rewards(self) -> list[float]:
        return [sample.reward for sample in self.samples]


def model_supports_selective_logits(model: Any) -> bool:
    real_model = getattr(model, "_orig_mod", model)
    forward = getattr(real_model, "forward", None)
    if forward is None:
        return False
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return False
    return "logits_to_keep" in signature.parameters


def build_selective_logit_positions(
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    if labels.ndim != 2:
        raise ValueError(f"Expected labels with shape [batch, seq], got {tuple(labels.shape)}")
    if mask.shape != labels.shape:
        raise ValueError(
            f"Expected mask with shape {tuple(labels.shape)}, got {tuple(mask.shape)}"
        )

    shifted_labels = labels[:, 1:]
    valid_mask = mask[:, 1:].bool() & shifted_labels.ne(-100)
    positions = valid_mask.any(dim=0).nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return None
    return positions


def prepare_grpo_forward_kwargs(
    *,
    model: Any,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    forward_kwargs: dict[str, Any] = {"attention_mask": attention_mask}
    if not model_supports_selective_logits(model):
        return forward_kwargs, None

    positions = build_selective_logit_positions(labels, mask)
    if positions is None:
        return forward_kwargs, None

    positions = positions.to(device=labels.device)
    forward_kwargs["logits_to_keep"] = positions
    return forward_kwargs, positions


def compute_group_advantages(
    rewards: torch.Tensor | Sequence[float],
    group_ids: Sequence[Any],
    *,
    normalize: bool = True,
    center: bool = True,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if len(group_ids) == 0:
        return torch.tensor([], dtype=torch.float32)

    rewards_tensor = (
        rewards.clone()
        if isinstance(rewards, torch.Tensor)
        else torch.tensor(rewards, dtype=torch.float32)
    ).float()
    advantages = torch.zeros_like(rewards_tensor)

    grouped_indices: dict[Any, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped_indices.setdefault(group_id, []).append(index)

    for indices in grouped_indices.values():
        index_tensor = torch.tensor(indices, device=rewards_tensor.device, dtype=torch.long)
        group_rewards = rewards_tensor.index_select(0, index_tensor)

        if center:
            group_advantages = group_rewards - group_rewards.mean()
        else:
            group_advantages = group_rewards.clone()

        if normalize:
            std = group_advantages.std(unbiased=False)
            if torch.isfinite(std) and std > epsilon:
                group_advantages = group_advantages / (std + epsilon)
            else:
                group_advantages = torch.zeros_like(group_advantages)

        advantages.index_copy_(0, index_tensor, group_advantages)

    return advantages


def gather_completion_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    backend: str = "selective",
    chunk_size: int = 128,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits with shape [batch, seq, vocab], got {tuple(logits.shape)}"
        )
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"Expected labels with shape {tuple(logits.shape[:2])}, got {tuple(labels.shape)}"
        )

    labels_clamped = labels.clamp_min(0)
    selected_logits = torch.gather(logits, dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)

    if backend == "dense":
        log_probs = torch.log_softmax(logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)

    if backend != "selective":
        raise ValueError(f"Unknown GRPO logprob backend: {backend}")

    if valid_mask is None:
        valid_mask = labels.ne(-100)
    valid_mask = valid_mask.to(device=logits.device).bool()

    # Compute exact selected-token logprobs without materializing a full log_softmax tensor.
    # We only process masked-in rows so prompt positions do not incur extra vocab-wide work.
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_selected = selected_logits.reshape(-1)
    flat_valid = valid_mask.reshape(-1)

    valid_indices = flat_valid.nonzero(as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return torch.zeros_like(selected_logits)

    values = []
    effective_chunk_size = max(int(chunk_size), 1)
    for chunk_indices in valid_indices.split(effective_chunk_size):
        chunk_logits = flat_logits.index_select(0, chunk_indices)
        chunk_selected = flat_selected.index_select(0, chunk_indices)
        values.append(chunk_selected - torch.logsumexp(chunk_logits, dim=-1))

    valid_values = torch.cat(values, dim=0)
    flat_output = torch.zeros_like(flat_selected)
    flat_output.scatter_(0, valid_indices, valid_values)
    return flat_output.view_as(selected_logits)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, *, epsilon: float = 1e-8) -> torch.Tensor:
    mask = mask.to(values.dtype)
    numerator = (values * mask).sum()
    denominator = mask.sum().clamp(min=epsilon)
    return numerator / denominator


def approximate_kl_divergence(
    current_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    *,
    estimator: str = "low_variance",
) -> torch.Tensor:
    if estimator == "low_variance":
        delta = reference_logprobs - current_logprobs
        return torch.exp(delta) - delta - 1.0
    if estimator == "log_ratio":
        return current_logprobs - reference_logprobs
    raise ValueError(f"Unknown KL estimator: {estimator}")


def compute_grpo_loss(
    *,
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    mask: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    config: GRPOConfig,
    reference_logprobs: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
    model: Any = None,
    logits_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction_tensor = hidden_states if hidden_states is not None else logits
    if prediction_tensor is None:
        raise ValueError("compute_grpo_loss requires either logits or hidden_states.")

    device = prediction_tensor.device
    labels = labels.to(device)
    mask = mask.to(device).bool()
    old_logprobs = old_logprobs.to(device, dtype=prediction_tensor.dtype)
    advantages = advantages.to(device, dtype=prediction_tensor.dtype)

    # Causal LM logits at position t score token t+1, so align all supervision
    # with the shifted next-token prediction positions.
    shifted_labels = labels[:, 1:]
    shifted_mask = mask[:, 1:]
    shifted_old_logprobs = old_logprobs[:, 1:]
    shifted_advantages = advantages[:, 1:]
    valid_mask = shifted_mask & shifted_labels.ne(-100)
    if logits_positions is not None:
        logits_positions = logits_positions.to(device=device, dtype=torch.long)
        shifted_labels = shifted_labels.index_select(1, logits_positions)
        shifted_old_logprobs = shifted_old_logprobs.index_select(1, logits_positions)
        shifted_advantages = shifted_advantages.index_select(1, logits_positions)
        valid_mask = valid_mask.index_select(1, logits_positions)

    if config.logprob_source == "hidden_states":
        if hidden_states is None:
            raise ValueError("GRPO hidden-state logprob source requires hidden_states.")
        if model is None:
            raise ValueError("GRPO hidden-state logprob source requires the model in ctx.")
        output_projection = resolve_output_projection(model)
        token_logprobs = gather_selected_logprobs_from_hidden(
            hidden_states[:, :-1, :],
            shifted_labels,
            valid_mask,
            output_projection=output_projection,
            chunk_size=config.logprob_chunk_size,
        )
    elif config.logprob_source == "logits":
        if logits is None:
            raise ValueError("GRPO logits logprob source requires logits.")
        if logits_positions is not None:
            if logits.shape[1] != logits_positions.numel():
                raise ValueError(
                    f"logits shape {tuple(logits.shape)} did not match requested logits positions "
                    f"shape {tuple(logits_positions.shape)}"
                )
            logits_for_loss = logits
        else:
            logits_for_loss = logits[:, :-1, :]
        token_logprobs = gather_completion_logprobs(
            logits_for_loss,
            shifted_labels,
            valid_mask=valid_mask,
            backend=config.logprob_backend,
            chunk_size=config.logprob_chunk_size,
        )
    else:
        raise ValueError(f"Unknown GRPO logprob source: {config.logprob_source}")

    if shifted_old_logprobs.shape != token_logprobs.shape:
        raise ValueError(
            f"old_logprobs shape {tuple(shifted_old_logprobs.shape)} did not match gathered token logprobs "
            f"shape {tuple(token_logprobs.shape)}"
        )

    if shifted_advantages.ndim == 1:
        shifted_advantages = shifted_advantages.unsqueeze(-1).expand_as(token_logprobs)
    elif shifted_advantages.shape != token_logprobs.shape:
        raise ValueError(
            f"advantages shape {tuple(shifted_advantages.shape)} did not match token logprobs "
            f"shape {tuple(token_logprobs.shape)}"
        )

    log_ratio = token_logprobs - shifted_old_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(
        ratio,
        min=1.0 - config.clip_epsilon,
        max=1.0 + config.clip_epsilon,
    )

    unclipped_objective = ratio * shifted_advantages
    clipped_objective = clipped_ratio * shifted_advantages
    policy_objective = torch.minimum(unclipped_objective, clipped_objective)
    policy_loss = -masked_mean(policy_objective, valid_mask, epsilon=config.epsilon)

    kl_loss = torch.tensor(0.0, device=device, dtype=prediction_tensor.dtype)
    if reference_logprobs is not None and config.kl_beta != 0.0:
        reference_logprobs = reference_logprobs.to(device, dtype=prediction_tensor.dtype)[:, 1:]
        if logits_positions is not None:
            reference_logprobs = reference_logprobs.index_select(1, logits_positions)
        if reference_logprobs.shape != token_logprobs.shape:
            raise ValueError(
                f"reference_logprobs shape {tuple(reference_logprobs.shape)} did not match token logprobs "
                f"shape {tuple(token_logprobs.shape)}"
            )
        approx_kl = approximate_kl_divergence(
            token_logprobs,
            reference_logprobs,
            estimator=config.kl_estimator,
        )
        kl_loss = config.kl_beta * masked_mean(approx_kl, valid_mask, epsilon=config.epsilon)

    total_loss = policy_loss + kl_loss
    clip_fraction = masked_mean(
        (ratio.ne(clipped_ratio)).to(prediction_tensor.dtype),
        valid_mask,
        epsilon=config.epsilon,
    )

    metrics = {
        "grpo_total": float(total_loss.item()),
        "grpo_policy": float(policy_loss.item()),
        "grpo_kl": float(kl_loss.item()),
        "grpo_advantage_mean": float(
            masked_mean(shifted_advantages, valid_mask, epsilon=config.epsilon).item()
        ),
        "grpo_ratio_mean": float(masked_mean(ratio, valid_mask, epsilon=config.epsilon).item()),
        "grpo_clipfrac": float(clip_fraction.item()),
    }
    return total_loss, metrics
