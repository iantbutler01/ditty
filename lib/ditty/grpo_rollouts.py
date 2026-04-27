from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .grpo import (
    GRPOConfig,
    compute_group_advantages,
    gather_completion_logprobs,
    prepare_grpo_forward_kwargs,
)
from .processors import Context, PreProcessor


@dataclass
class RolloutRecord:
    task: Any
    group_id: str
    sample_id: str
    prompt_text: str
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_text: str
    reward: float
    reward_metrics: dict[str, float] = field(default_factory=dict)


RewardFn = Callable[[Any, str], Any]
PromptFn = Callable[[Any], str]
ProgressFn = Callable[[str], None]
RolloutCallback = Callable[[list[RolloutRecord], dict[str, float], Context], None]


def _iter_module_tree(model: Any):
    seen: set[int] = set()
    stack: list[Any] = [model]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if isinstance(current, nn.Module):
            stack.extend(current.children())
            continue
        for attr in ("module", "_orig_mod", "_fsdp_wrapped_module", "model", "base_model"):
            child = getattr(current, attr, None)
            if child is not current:
                stack.append(child)


def _call_if_present(model: Any, method_name: str) -> bool:
    called = False
    for current in _iter_module_tree(model):
        method = getattr(current, method_name, None)
        if callable(method):
            method()
            called = True
    return called


def _is_gradient_checkpointing_enabled(model: Any) -> bool:
    for current in _iter_module_tree(model):
        value = getattr(current, "is_gradient_checkpointing", None)
        if isinstance(value, bool):
            return value
    return False


def _find_model_config(model: Any) -> Any | None:
    for current in _iter_module_tree(model):
        config = getattr(current, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            return config
    return None


def _coerce_reward(result: Any) -> tuple[float, dict[str, float]]:
    if isinstance(result, tuple):
        reward, metrics = result
        return float(reward), dict(metrics or {})
    if isinstance(result, dict):
        return float(result.get("reward", 0.0)), dict(result.get("metrics") or {})
    if hasattr(result, "reward"):
        return float(result.reward), dict(getattr(result, "metrics", {}) or {})
    return float(result), {}


def _trim_completion(ids: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is None:
        return ids
    if eos_token_id in ids:
        return ids[: ids.index(eos_token_id) + 1]
    return ids


def _sample_next_tokens(logits: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits.float() / temperature
    if top_p >= 1.0:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    filtered_logits = sorted_logits.masked_fill(remove, torch.finfo(sorted_logits.dtype).min)
    filtered_probs = torch.softmax(filtered_logits, dim=-1)
    sampled_sorted = torch.multinomial(filtered_probs, num_samples=1)
    return sorted_indices.gather(dim=-1, index=sampled_sorted).squeeze(-1)


def _manual_generate_completion_ids(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pad_token_id: int,
    eos_token_id: int | None,
    progress_fn: Callable[[int], None] | None = None,
    progress_every: int = 0,
) -> list[list[int]]:
    batch_size = input_ids.shape[0]
    completions: list[list[int]] = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, device=input_ids.device, dtype=torch.bool)
    current_ids = input_ids
    current_mask = attention_mask

    for token_idx in range(max_new_tokens):
        output = model(input_ids=current_ids, attention_mask=current_mask)
        next_tokens = _sample_next_tokens(output.logits[:, -1, :], temperature=temperature, top_p=top_p)
        active = ~finished
        active_flags = active.detach().cpu().tolist()

        for row, token in enumerate(next_tokens.detach().cpu().tolist()):
            if active_flags[row]:
                completions[row].append(int(token))

        if eos_token_id is not None:
            finished = finished | (active & next_tokens.eq(eos_token_id))

        filler = eos_token_id if eos_token_id is not None else pad_token_id
        next_tokens = torch.where(
            active,
            next_tokens,
            torch.full_like(next_tokens, int(filler)),
        )
        current_ids = torch.cat([current_ids, next_tokens.unsqueeze(1)], dim=1)
        current_mask = torch.cat([current_mask, torch.ones_like(next_tokens).unsqueeze(1)], dim=1)
        generated = token_idx + 1
        if progress_fn and progress_every > 0 and (generated % progress_every == 0 or generated == max_new_tokens):
            progress_fn(generated)

    return [_trim_completion(ids, eos_token_id) for ids in completions]


def reward_summary(records: Sequence[RolloutRecord]) -> dict[str, float]:
    rewards = [r.reward for r in records]
    by_group: dict[str, list[float]] = {}
    for record in records:
        by_group.setdefault(record.group_id, []).append(record.reward)

    group_stds = []
    active_groups = 0
    for values in by_group.values():
        if len(values) > 1:
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            group_stds.append(std)
            if std > 1e-8:
                active_groups += 1

    return {
        "reward_mean": sum(rewards) / max(len(rewards), 1),
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "group_reward_std_mean": sum(group_stds) / max(len(group_stds), 1),
        "active_group_fraction": active_groups / max(len(by_group), 1),
    }


@torch.no_grad()
def generate_rollouts(
    *,
    model,
    tokenizer,
    tasks: Sequence[Any],
    render_prompt: PromptFn,
    reward_fn: RewardFn,
    group_id_fn: Callable[[Any], str],
    rollouts_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    step: int,
    rollout_use_cache: bool = True,
    rollout_disable_compile: bool = True,
    rollout_backend: str = "manual",
    rollout_log_every: int = 1,
    rollout_token_log_every: int = 16,
    progress_fn: ProgressFn | None = None,
) -> list[RolloutRecord]:
    model.eval()
    records: list[RolloutRecord] = []

    model_config = _find_model_config(model)
    old_use_cache = getattr(model_config, "use_cache", None) if model_config is not None else None
    old_gradient_checkpointing = _is_gradient_checkpointing_enabled(model)
    if old_use_cache is not None and model_config is not None:
        model_config.use_cache = rollout_use_cache if rollout_backend == "hf_generate" else False
    disabled_gradient_checkpointing = False
    if rollout_backend == "hf_generate" and rollout_use_cache and old_gradient_checkpointing:
        disabled_gradient_checkpointing = _call_if_present(model, "gradient_checkpointing_disable")
    if rollout_backend == "hf_generate" and progress_fn:
        progress_fn(
            "rollout hf_generate setup "
            f"use_cache={getattr(model_config, 'use_cache', None)} "
            f"gradient_checkpointing_was_enabled={old_gradient_checkpointing} "
            f"disabled_gradient_checkpointing={disabled_gradient_checkpointing}"
        )
    try:
        for group_offset, task in enumerate(tasks):
            group_id = group_id_fn(task)
            should_log = (
                progress_fn is not None
                and rollout_log_every > 0
                and (
                    group_offset == 0
                    or (group_offset + 1) % rollout_log_every == 0
                    or group_offset == len(tasks) - 1
                )
            )
            group_start = time.time()
            if should_log:
                progress_fn(f"rollout task {group_offset + 1}/{len(tasks)} start id={group_id}")

            prompt_text = render_prompt(task)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            encoded = tokenizer(
                [prompt_text] * rollouts_per_prompt,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

            if rollout_backend == "manual":
                completion_batches = _manual_generate_completion_ids(
                    model=model,
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    progress_fn=(
                        (
                            lambda generated, group_offset=group_offset, group_id=group_id: progress_fn(
                                f"rollout task {group_offset + 1}/{len(tasks)} tokens_generated={generated}/{max_new_tokens} id={group_id}"
                            )
                        )
                        if should_log and progress_fn is not None
                        else None
                    ),
                    progress_every=rollout_token_log_every,
                )
            elif rollout_backend == "hf_generate":
                input_width = encoded["input_ids"].shape[1]
                output_ids = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    use_cache=rollout_use_cache,
                    disable_compile=rollout_disable_compile,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                completion_batches = [
                    _trim_completion(output[input_width:].detach().cpu().tolist(), tokenizer.eos_token_id)
                    for output in output_ids
                ]
            else:
                raise ValueError(f"Unknown rollout backend: {rollout_backend}")

            completion_token_count = 0
            for rollout_idx, completion_ids in enumerate(completion_batches):
                completion_token_count += len(completion_ids)
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                reward, reward_metrics = _coerce_reward(reward_fn(task, completion_text))
                records.append(
                    RolloutRecord(
                        task=task,
                        group_id=group_id,
                        sample_id=f"s{step}-{group_offset}-{rollout_idx}",
                        prompt_text=prompt_text,
                        prompt_ids=prompt_ids,
                        completion_ids=completion_ids,
                        completion_text=completion_text,
                        reward=reward,
                        reward_metrics=reward_metrics,
                    )
                )

            if should_log:
                progress_fn(
                    f"rollout task {group_offset + 1}/{len(tasks)} done id={group_id} "
                    f"tokens={completion_token_count} elapsed={time.time() - group_start:.1f}s"
                )
        return records
    finally:
        if rollout_backend == "hf_generate" and old_gradient_checkpointing:
            _call_if_present(model, "gradient_checkpointing_enable")
        if old_use_cache is not None and model_config is not None:
            model_config.use_cache = old_use_cache


def collate_rollouts(records: Sequence[RolloutRecord], tokenizer, device: torch.device) -> dict[str, Any]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    max_len = max(len(r.prompt_ids) + len(r.completion_ids) for r in records)
    batch = len(records)
    input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long)
    labels = torch.full((batch, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
    completion_mask = torch.zeros((batch, max_len), dtype=torch.bool)

    for row, record in enumerate(records):
        ids = record.prompt_ids + record.completion_ids
        seq_len = len(ids)
        prompt_len = len(record.prompt_ids)
        input_ids[row, :seq_len] = torch.tensor(ids, dtype=torch.long)
        labels[row, :seq_len] = input_ids[row, :seq_len]
        attention_mask[row, :seq_len] = 1
        completion_mask[row, prompt_len:seq_len] = 1

    rewards = torch.tensor([r.reward for r in records], dtype=torch.float32)
    group_ids = [r.group_id for r in records]
    scalar_advantages = compute_group_advantages(rewards, group_ids)
    token_advantages = torch.zeros((batch, max_len), dtype=torch.float32)
    for row, record in enumerate(records):
        prompt_len = len(record.prompt_ids)
        seq_len = prompt_len + len(record.completion_ids)
        token_advantages[row, prompt_len:seq_len] = scalar_advantages[row]

    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device),
        "completion_mask": completion_mask.to(device),
        "advantages": token_advantages.to(device),
        "rewards": rewards,
        "group_ids": group_ids,
    }


@torch.no_grad()
def compute_old_logprobs(model, batch: dict[str, Any], config: GRPOConfig) -> torch.Tensor:
    model.eval()
    forward_kwargs, logits_positions = prepare_grpo_forward_kwargs(
        model=model,
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        mask=batch["completion_mask"],
    )
    output = model(batch["input_ids"], **forward_kwargs)
    logits = output.logits

    labels = batch["labels"]
    shifted_labels = labels[:, 1:]
    valid_mask = batch["completion_mask"][:, 1:] & shifted_labels.ne(-100)
    full_logprobs = torch.zeros(labels.shape, device=labels.device, dtype=logits.dtype)

    if logits_positions is not None:
        positions = logits_positions.to(device=labels.device, dtype=torch.long)
        selected_labels = shifted_labels.index_select(1, positions)
        selected_valid = valid_mask.index_select(1, positions)
        selected_logprobs = gather_completion_logprobs(
            logits,
            selected_labels,
            valid_mask=selected_valid,
            backend=config.logprob_backend,
            chunk_size=config.logprob_chunk_size,
        )
        full_logprobs[:, 1:].index_copy_(1, positions, selected_logprobs)
    else:
        token_logprobs = gather_completion_logprobs(
            logits[:, :-1, :],
            shifted_labels,
            valid_mask=valid_mask,
            backend=config.logprob_backend,
            chunk_size=config.logprob_chunk_size,
        )
        full_logprobs[:, 1:] = token_logprobs

    return full_logprobs


class GRPORolloutPreProcessor(PreProcessor):
    def __init__(
        self,
        *,
        tokenizer,
        render_prompt: PromptFn,
        reward_fn: RewardFn,
        grpo_config: GRPOConfig | None = None,
        group_id_fn: Callable[[Any], str] | None = None,
        rollouts_per_prompt: int = 4,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        rollout_use_cache: bool = True,
        rollout_disable_compile: bool = True,
        rollout_backend: str = "manual",
        rollout_log_every: int = 1,
        rollout_token_log_every: int = 16,
        progress_fn: ProgressFn | None = None,
        on_rollouts: RolloutCallback | None = None,
    ) -> None:
        super().__init__(contract="")
        self.tokenizer = tokenizer
        self.render_prompt = render_prompt
        self.reward_fn = reward_fn
        self.grpo_config = grpo_config or GRPOConfig()
        self.group_id_fn = group_id_fn or (lambda task: str(task["id"] if isinstance(task, dict) else id(task)))
        self.rollouts_per_prompt = rollouts_per_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.rollout_use_cache = rollout_use_cache
        self.rollout_disable_compile = rollout_disable_compile
        self.rollout_backend = rollout_backend
        self.rollout_log_every = rollout_log_every
        self.rollout_token_log_every = rollout_token_log_every
        self.progress_fn = progress_fn
        self.on_rollouts = on_rollouts

    def process(self, batch: Sequence[Any], ctx: Context) -> tuple[torch.Tensor, Context]:
        model = ctx["model"]
        device = ctx["device"]
        step = int(ctx.get("total_steps", 0))
        if self.progress_fn:
            self.progress_fn(f"step {step} rollout start tasks={len(batch)}")

        step_start = time.time()
        records = generate_rollouts(
            model=model,
            tokenizer=self.tokenizer,
            tasks=batch,
            render_prompt=self.render_prompt,
            reward_fn=self.reward_fn,
            group_id_fn=self.group_id_fn,
            rollouts_per_prompt=self.rollouts_per_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            device=device,
            step=step,
            rollout_use_cache=self.rollout_use_cache,
            rollout_disable_compile=self.rollout_disable_compile,
            rollout_backend=self.rollout_backend,
            rollout_log_every=self.rollout_log_every,
            rollout_token_log_every=self.rollout_token_log_every,
            progress_fn=self.progress_fn,
        )
        if self.progress_fn:
            self.progress_fn(
                f"step {step} rollout done records={len(records)} elapsed={time.time() - step_start:.1f}s"
            )

        grpo_batch = collate_rollouts(records, self.tokenizer, device)
        if self.progress_fn:
            self.progress_fn(f"step {step} old_logprobs start")
        old_logprobs = compute_old_logprobs(model, grpo_batch, self.grpo_config)
        if self.progress_fn:
            self.progress_fn(f"step {step} old_logprobs done")
        model.train()

        forward_kwargs, logits_positions = prepare_grpo_forward_kwargs(
            model=model,
            attention_mask=grpo_batch["attention_mask"],
            labels=grpo_batch["labels"],
            mask=grpo_batch["completion_mask"],
        )
        ctx["forward_kwargs"] = forward_kwargs
        ctx["logits_positions"] = logits_positions
        ctx["target"] = grpo_batch["labels"]
        ctx["mask"] = grpo_batch["completion_mask"]
        ctx["old_logprobs"] = old_logprobs
        ctx["advantages"] = grpo_batch["advantages"]
        ctx["rollout_records"] = records
        ctx["rollout_metrics"] = {
            "advantage_abs_sum": float(grpo_batch["advantages"].abs().sum().item()),
            **reward_summary(records),
        }
        if self.on_rollouts:
            self.on_rollouts(records, ctx["rollout_metrics"], ctx)

        return grpo_batch["input_ids"], ctx

    def config(self) -> dict[str, Any]:
        return {
            "rollouts_per_prompt": self.rollouts_per_prompt,
            "max_new_tokens": self.max_new_tokens,
            "rollout_use_cache": self.rollout_use_cache,
            "rollout_disable_compile": self.rollout_disable_compile,
            "rollout_backend": self.rollout_backend,
            "rollout_token_log_every": self.rollout_token_log_every,
        }
