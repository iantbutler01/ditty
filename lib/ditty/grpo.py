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
    clip_epsilon_low: float | None = None
    clip_epsilon_high: float | None = None
    kl_beta: float = 0.04
    epsilon: float = 1e-8
    normalize_advantages: bool = True
    center_advantages: bool = True
    loss_type: str = "grpo"
    max_completion_length: int | None = None
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
    visited: set[int] = set()
    candidates = [model]
    while candidates:
        current = candidates.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        forward = getattr(current, "forward", None)
        if forward is not None:
            try:
                signature = inspect.signature(forward)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                parameters = signature.parameters
                if "logits_to_keep" in parameters:
                    return True
                if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
                    return True

        for attr in ("_orig_mod", "module", "_fsdp_wrapped_module", "_checkpoint_wrapped_module"):
            wrapped = getattr(current, attr, None)
            if wrapped is not None and id(wrapped) not in visited:
                candidates.append(wrapped)
    return False


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
    logprob_source: str = "logits",
) -> tuple[dict[str, Any], torch.Tensor | None]:
    forward_kwargs: dict[str, Any] = {"attention_mask": attention_mask}
    if not model_supports_selective_logits(model):
        if logprob_source == "hidden_states":
            forward_kwargs["output_hidden_states"] = True
        return forward_kwargs, None

    if logprob_source == "hidden_states":
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["logits_to_keep"] = 1
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

    flat_output = torch.zeros_like(flat_selected)
    valid_values = torch.cat(values, dim=0).to(dtype=flat_output.dtype)
    flat_output.scatter_(0, valid_indices, valid_values)
    return flat_output.view_as(selected_logits)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, *, epsilon: float = 1e-8) -> torch.Tensor:
    mask = mask.to(values.dtype)
    numerator = (values * mask).sum()
    denominator = mask.sum().clamp(min=epsilon)
    return numerator / denominator


def masked_policy_loss(
    objective: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_type: str,
    max_completion_length: int | None,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = mask.to(objective.dtype)
    objective_sum = (objective * mask).sum()
    active_tokens = mask.sum()
    normalized_loss_type = loss_type.lower()

    if normalized_loss_type in {"grpo", "gspo"}:
        per_sequence_tokens = mask.sum(dim=1).clamp(min=epsilon)
        per_sequence_objective = (objective * mask).sum(dim=1) / per_sequence_tokens
        active_sequences = mask.sum(dim=1).gt(0)
        denominator = active_sequences.sum().to(objective.dtype).clamp(min=epsilon)
        return -(per_sequence_objective * active_sequences.to(objective.dtype)).sum() / denominator, denominator

    if normalized_loss_type in {"bnpo", "dapo"}:
        denominator = active_tokens.clamp(min=epsilon)
        return -objective_sum / denominator, denominator

    if normalized_loss_type in {"dr_grpo", "dr_gspo"}:
        batch_size = objective.shape[0]
        constant_length = max_completion_length if max_completion_length is not None else objective.shape[1]
        denominator = torch.tensor(
            max(batch_size * int(constant_length), 1),
            device=objective.device,
            dtype=objective.dtype,
        ).clamp(min=epsilon)
        return -objective_sum / denominator, denominator

    raise ValueError(
        "Unknown GRPO loss_type="
        f"{loss_type!r}; expected one of 'grpo', 'gspo', 'bnpo', 'dapo', 'dr_grpo', or 'dr_gspo'."
    )


def clip_bounds(config: GRPOConfig) -> tuple[float, float]:
    low = config.clip_epsilon if config.clip_epsilon_low is None else config.clip_epsilon_low
    high = config.clip_epsilon if config.clip_epsilon_high is None else config.clip_epsilon_high
    if low < 0 or high < 0:
        raise ValueError("clip epsilons must be non-negative")
    return 1.0 - low, 1.0 + high


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
    clip_min, clip_max = clip_bounds(config)
    normalized_loss_type = config.loss_type.lower()

    if normalized_loss_type == "gspo":
        # Vanilla GSPO (Eq. 5): one sequence ratio per rollout, gradient flows through s_i
        # directly via the live token log-probs that compose it.
        token_counts = valid_mask.sum(dim=1).clamp(min=config.epsilon).to(log_ratio.dtype)
        sequence_log_ratio = (log_ratio * valid_mask.to(log_ratio.dtype)).sum(dim=1) / token_counts
        sequence_ratio = torch.exp(sequence_log_ratio).unsqueeze(1)
        clipped_sequence_ratio = torch.clamp(sequence_ratio, min=clip_min, max=clip_max)
        unclipped_objective = sequence_ratio * shifted_advantages
        clipped_objective = clipped_sequence_ratio * shifted_advantages
        policy_objective = torch.minimum(unclipped_objective, clipped_objective)
        ratio = sequence_ratio.expand_as(token_logprobs)
        clipped_ratio = clipped_sequence_ratio.expand_as(token_logprobs)
    elif normalized_loss_type == "dr_gspo":
        # GSPO-token (Eq. 13-14) + DR.GRPO constant-token denominator. The sequence ratio
        # is detached so gradient flows only through the live token log-probs, preserving
        # per-token credit semantics when A_{i,t} varies (e.g. FICA per-span credit) while
        # keeping GSPO's variance-bounded importance correction.
        token_counts = valid_mask.sum(dim=1).clamp(min=config.epsilon).to(log_ratio.dtype)
        sequence_log_ratio = (log_ratio * valid_mask.to(log_ratio.dtype)).sum(dim=1) / token_counts
        sequence_ratio = torch.exp(sequence_log_ratio).unsqueeze(1)
        sequence_ratio_detached = sequence_ratio.detach()
        clipped_sequence_ratio = torch.clamp(sequence_ratio_detached, min=clip_min, max=clip_max)
        # token_term equals 1 in forward; gradient flows through live token_logprobs.
        token_term = torch.exp(token_logprobs - token_logprobs.detach())
        unclipped_objective = sequence_ratio_detached * token_term * shifted_advantages
        clipped_objective = clipped_sequence_ratio * token_term * shifted_advantages
        policy_objective = torch.minimum(unclipped_objective, clipped_objective)
        ratio = sequence_ratio_detached.expand_as(token_logprobs)
        clipped_ratio = clipped_sequence_ratio.expand_as(token_logprobs)
    else:
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, min=clip_min, max=clip_max)
        unclipped_objective = ratio * shifted_advantages
        clipped_objective = clipped_ratio * shifted_advantages
        policy_objective = torch.minimum(unclipped_objective, clipped_objective)

    policy_loss, policy_denominator = masked_policy_loss(
        policy_objective,
        valid_mask,
        loss_type=config.loss_type,
        max_completion_length=config.max_completion_length,
        epsilon=config.epsilon,
    )

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
    if not total_loss.requires_grad and prediction_tensor.requires_grad:
        total_loss = total_loss + prediction_tensor.sum() * 0.0
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
        "grpo_policy_denominator": float(policy_denominator.item()),
    }
    if normalized_loss_type in {"gspo", "dr_gspo"}:
        sequence_valid = valid_mask.any(dim=1)
        sequence_mask = sequence_valid.unsqueeze(1).expand_as(ratio)
        metrics["grpo_sequence_ratio_mean"] = float(
            masked_mean(ratio, sequence_mask, epsilon=config.epsilon).item()
        )
        # Per-sequence |s_i - 1| max diagnoses runaway ratio drift before clipping bites.
        if sequence_valid.any():
            seq_dev = (sequence_ratio.squeeze(1).detach() - 1.0).abs()
            seq_dev = seq_dev.masked_fill(~sequence_valid, 0.0)
            metrics["grpo_sequence_ratio_max_abs_dev"] = float(seq_dev.max().item())
        else:
            metrics["grpo_sequence_ratio_max_abs_dev"] = 0.0
    # Zero-advantage sample fraction is a proxy for zero-variance groups upstream.
    sample_zero_adv = (
        masked_mean(
            shifted_advantages.abs().lt(config.epsilon).to(prediction_tensor.dtype),
            valid_mask,
            epsilon=config.epsilon,
        )
    )
    metrics["grpo_zero_advantage_sample_frac"] = float(sample_zero_adv.item())
    return total_loss, metrics
