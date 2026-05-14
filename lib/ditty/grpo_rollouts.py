from __future__ import annotations

import inspect
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.nn as nn

from .grpo import (
    GRPOConfig,
    compute_group_advantages,
    gather_completion_logprobs,
    prepare_grpo_forward_kwargs,
)
from .credit import FunctionalCreditConfig, assign_functional_token_advantages
from .processors import Context, PreProcessor
from .projection import (
    extract_last_hidden_state,
    gather_selected_logprobs_from_hidden,
    resolve_output_projection,
)


@dataclass
class PolicyVersion:
    policy_version: str
    created_step: int
    created_at: float
    policy_checkpoint_id: str | None = None
    rollout_backend: str | None = None
    model_id: str | None = None
    generation: dict[str, Any] = field(default_factory=dict)

    def age_updates(self, current_step: int) -> int:
        return max(int(current_step) - int(self.created_step), 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "policy_checkpoint_id": self.policy_checkpoint_id,
            "created_step": int(self.created_step),
            "created_at": float(self.created_at),
            "rollout_backend": self.rollout_backend,
            "model_id": self.model_id,
            "generation": dict(self.generation or {}),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyVersion":
        return cls(
            policy_version=str(value.get("policy_version") or value.get("id") or "unknown"),
            policy_checkpoint_id=(
                str(value["policy_checkpoint_id"])
                if value.get("policy_checkpoint_id") is not None
                else None
            ),
            created_step=int(value.get("created_step", 0)),
            created_at=float(value.get("created_at", 0.0)),
            rollout_backend=(
                str(value["rollout_backend"]) if value.get("rollout_backend") is not None else None
            ),
            model_id=str(value["model_id"]) if value.get("model_id") is not None else None,
            generation=dict(value.get("generation") or {}),
        )


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
    token_advantages: list[float] | None = None
    policy_version: PolicyVersion | None = None
    task_signature: str | None = None
    skip_reason: str | None = None
    environment_transcript: list[dict[str, Any]] | None = None


@dataclass
class RolloutBatch:
    records: list[RolloutRecord]
    all_records: list[RolloutRecord]
    policy_version: PolicyVersion | None
    current_step: int
    reuse_count: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    skip_reasons: dict[str, int] = field(default_factory=dict)
    signature_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    failure_reasons: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_records(
        cls,
        records: Sequence[RolloutRecord],
        *,
        current_step: int,
        skip_zero_variance_groups: bool = True,
        min_group_reward_std: float = 0.0,
        max_policy_age_updates: int | None = 0,
        max_rollout_reuse: int | None = 1,
        reuse_count: int = 0,
    ) -> "RolloutBatch":
        all_records = list(records)
        selected: list[RolloutRecord] = []
        skipped: dict[str, int] = {}
        grouped: dict[str, list[RolloutRecord]] = {}
        for record in all_records:
            grouped.setdefault(record.group_id, []).append(record)

        policy_versions = {
            record.policy_version.policy_version
            for record in all_records
            if record.policy_version is not None
        }
        policy_version = next(
            (record.policy_version for record in all_records if record.policy_version is not None),
            None,
        )
        if len(policy_versions) > 1:
            raise ValueError(f"RolloutBatch spans multiple policy versions: {sorted(policy_versions)}")
        if policy_version is not None and max_policy_age_updates is not None:
            policy_age = policy_version.age_updates(current_step)
            if policy_age > max_policy_age_updates:
                raise ValueError(
                    "RolloutBatch policy version is stale: "
                    f"age_updates={policy_age} > max_policy_age_updates={max_policy_age_updates}"
                )
        if max_rollout_reuse is not None and reuse_count >= max_rollout_reuse:
            raise ValueError(
                f"RolloutBatch reuse_count={reuse_count} exceeds max_rollout_reuse={max_rollout_reuse}"
            )

        group_stds: list[float] = []
        selected_group_stds: list[float] = []
        for group_records in grouped.values():
            rewards = [record.reward for record in group_records]
            mean = sum(rewards) / max(len(rewards), 1)
            std = (sum((value - mean) ** 2 for value in rewards) / max(len(rewards), 1)) ** 0.5
            group_stds.append(std)
            if skip_zero_variance_groups and std <= min_group_reward_std:
                reason = "zero_variance_group"
                for record in group_records:
                    record.skip_reason = reason
                skipped[reason] = skipped.get(reason, 0) + len(group_records)
                continue
            selected.extend(group_records)
            selected_group_stds.append(std)

        selected_ids = {id(record) for record in selected}
        signature_rows: dict[str, dict[str, Any]] = {}
        failure_reasons: dict[str, int] = {}
        for group_records in grouped.values():
            signature = group_records[0].task_signature or "<missing>"
            rewards = [record.reward for record in group_records]
            mean = sum(rewards) / max(len(rewards), 1)
            std = (sum((value - mean) ** 2 for value in rewards) / max(len(rewards), 1)) ** 0.5
            row = signature_rows.setdefault(
                signature,
                {
                    "signature": signature,
                    "source_groups": 0,
                    "selected_groups": 0,
                    "active_groups": 0,
                    "source_records": 0,
                    "selected_records": 0,
                    "pass_count": 0,
                    "reward_sum": 0.0,
                    "reward_sq_sum": 0.0,
                    "completion_tokens": 0,
                    "failure_reasons": defaultdict(int),
                },
            )
            row["source_groups"] += 1
            row["source_records"] += len(group_records)
            row["pass_count"] += sum(1 for record in group_records if record.reward >= 0.9)
            row["reward_sum"] += sum(rewards)
            row["reward_sq_sum"] += sum(value * value for value in rewards)
            row["completion_tokens"] += sum(len(record.completion_ids) for record in group_records)
            if std > min_group_reward_std:
                row["active_groups"] += 1
            if any(id(record) in selected_ids for record in group_records):
                row["selected_groups"] += 1
                row["selected_records"] += sum(1 for record in group_records if id(record) in selected_ids)
            for record in group_records:
                reason = _failure_reason(record)
                if reason is None:
                    continue
                row["failure_reasons"][reason] += 1
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        signature_stats: dict[str, dict[str, Any]] = {}
        for signature, row in signature_rows.items():
            records = max(int(row["source_records"]), 1)
            reward_mean = float(row["reward_sum"]) / records
            reward_variance = max(float(row["reward_sq_sum"]) / records - reward_mean * reward_mean, 0.0)
            source_groups = max(int(row["source_groups"]), 1)
            signature_stats[signature] = {
                "signature": signature,
                "source_groups": int(row["source_groups"]),
                "selected_groups": int(row["selected_groups"]),
                "active_groups": int(row["active_groups"]),
                "source_records": int(row["source_records"]),
                "selected_records": int(row["selected_records"]),
                "reward_mean": reward_mean,
                "reward_std": reward_variance**0.5,
                "pass_rate": float(row["pass_count"]) / records,
                "active_group_fraction": float(row["active_groups"]) / source_groups,
                "completion_tokens_mean": float(row["completion_tokens"]) / records,
                "failure_reasons": dict(sorted(row["failure_reasons"].items())),
            }

        active_groups = sum(1 for std in group_stds if std > min_group_reward_std)
        policy_age = policy_version.age_updates(current_step) if policy_version is not None else 0
        pass_rate = sum(1 for record in all_records if record.reward >= 0.9) / max(len(all_records), 1)
        completion_tokens_mean = sum(len(record.completion_ids) for record in all_records) / max(len(all_records), 1)
        active_signature_count = sum(
            1 for row in signature_stats.values() if row["active_groups"] > 0
        )
        metrics = {
            "rollout_source_records": float(len(all_records)),
            "rollout_selected_records": float(len(selected)),
            "rollout_source_groups": float(len(grouped)),
            "rollout_selected_groups": float(len({record.group_id for record in selected})),
            "rollout_skipped_records": float(len(all_records) - len(selected)),
            "rollout_skipped_zero_variance_records": float(skipped.get("zero_variance_group", 0)),
            "rollout_skipped_zero_variance_groups": float(
                sum(1 for group_records in grouped.values() if group_records and group_records[0].skip_reason == "zero_variance_group")
            ),
            "rollout_group_reward_std_mean_all": sum(group_stds) / max(len(group_stds), 1),
            "rollout_group_reward_std_mean_selected": sum(selected_group_stds) / max(len(selected_group_stds), 1),
            "rollout_active_group_fraction_all": active_groups / max(len(grouped), 1),
            "rollout_policy_age_updates": float(policy_age),
            "rollout_reuse_count": float(reuse_count),
            "rollout_task_signature_count": float(len(signature_stats)),
            "rollout_task_signature_active_count": float(active_signature_count),
            "rollout_task_signature_active_fraction": active_signature_count / max(len(signature_stats), 1),
            "rollout_pass_rate": float(pass_rate),
            "rollout_completion_tokens_mean": float(completion_tokens_mean),
            "rollout_failure_reason_count": float(len(failure_reasons)),
        }
        for reason, count in skipped.items():
            metrics[f"rollout_skip_reason_{reason}"] = float(count)
        for reason, count in sorted(failure_reasons.items(), key=lambda item: (-item[1], item[0]))[:8]:
            metrics[f"rollout_failure_{_metric_key(reason)}"] = float(count)

        return cls(
            records=selected,
            all_records=all_records,
            policy_version=policy_version,
            current_step=current_step,
            reuse_count=reuse_count,
            metrics=metrics,
            skip_reasons=skipped,
            signature_stats=signature_stats,
            failure_reasons=failure_reasons,
        )

    def validate_for_step(
        self,
        *,
        current_step: int,
        max_policy_age_updates: int | None = 0,
        max_rollout_reuse: int | None = 1,
    ) -> None:
        if self.policy_version is not None and max_policy_age_updates is not None:
            policy_age = self.policy_version.age_updates(current_step)
            if policy_age > max_policy_age_updates:
                raise ValueError(
                    "RolloutBatch policy version is stale: "
                    f"age_updates={policy_age} > max_policy_age_updates={max_policy_age_updates}"
                )
        if max_rollout_reuse is not None and self.reuse_count >= max_rollout_reuse:
            raise ValueError(
                f"RolloutBatch reuse_count={self.reuse_count} exceeds max_rollout_reuse={max_rollout_reuse}"
            )


@dataclass
class SignatureRunningStats:
    signature: str
    groups: int = 0
    active_groups: int = 0
    records: int = 0
    pass_count: int = 0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0
    completion_tokens: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)

    def update(self, summary: Mapping[str, Any]) -> None:
        records = int(summary.get("source_records", 0) or 0)
        reward_mean = float(summary.get("reward_mean", 0.0) or 0.0)
        reward_std = float(summary.get("reward_std", 0.0) or 0.0)
        self.groups += int(summary.get("source_groups", 0) or 0)
        self.active_groups += int(summary.get("active_groups", 0) or 0)
        self.records += records
        self.pass_count += int(round(float(summary.get("pass_rate", 0.0) or 0.0) * records))
        self.reward_sum += reward_mean * records
        self.reward_sq_sum += (reward_std * reward_std + reward_mean * reward_mean) * records
        self.completion_tokens += int(round(float(summary.get("completion_tokens_mean", 0.0) or 0.0) * records))
        reasons = summary.get("failure_reasons")
        if isinstance(reasons, Mapping):
            for reason, count in reasons.items():
                self.failure_reasons[str(reason)] = self.failure_reasons.get(str(reason), 0) + int(count)

    @property
    def reward_mean(self) -> float:
        return self.reward_sum / max(self.records, 1)

    @property
    def reward_std(self) -> float:
        mean = self.reward_mean
        return max(self.reward_sq_sum / max(self.records, 1) - mean * mean, 0.0) ** 0.5

    @property
    def pass_rate(self) -> float:
        return self.pass_count / max(self.records, 1)

    @property
    def active_group_fraction(self) -> float:
        return self.active_groups / max(self.groups, 1)

    @property
    def completion_tokens_mean(self) -> float:
        return self.completion_tokens / max(self.records, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "groups": self.groups,
            "active_groups": self.active_groups,
            "records": self.records,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "pass_rate": self.pass_rate,
            "active_group_fraction": self.active_group_fraction,
            "completion_tokens_mean": self.completion_tokens_mean,
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SignatureRunningStats":
        stats = cls(signature=str(value.get("signature") or "unknown"))
        stats.groups = int(value.get("groups", 0) or 0)
        stats.active_groups = int(value.get("active_groups", 0) or 0)
        stats.records = int(value.get("records", 0) or 0)
        stats.pass_count = int(round(float(value.get("pass_rate", 0.0) or 0.0) * stats.records))
        stats.reward_sum = float(value.get("reward_mean", 0.0) or 0.0) * stats.records
        reward_std = float(value.get("reward_std", 0.0) or 0.0)
        reward_mean = float(value.get("reward_mean", 0.0) or 0.0)
        stats.reward_sq_sum = (reward_std * reward_std + reward_mean * reward_mean) * stats.records
        stats.completion_tokens = int(
            round(float(value.get("completion_tokens_mean", 0.0) or 0.0) * stats.records)
        )
        reasons = value.get("failure_reasons")
        if isinstance(reasons, Mapping):
            stats.failure_reasons = {str(reason): int(count) for reason, count in reasons.items()}
        return stats


@dataclass
class RolloutSchedulerConfig:
    seed: int = 1
    pass_threshold: float = 0.9
    novelty_weight: float = 0.25
    variance_weight: float = 0.35
    active_weight: float = 0.20
    difficulty_weight: float = 0.15
    official_gap_weight: float = 0.20
    cost_weight: float = 0.05
    selection_history_weight: float = 0.05
    zero_variance_common_factor: float = 0.15
    common_signature_groups: int = 3
    max_signature_fraction: float = 0.5
    max_top_candidates: int = 32
    min_score: float = 1e-4
    max_expected_tokens_per_batch: int | None = None
    min_batch_size: int = 1


class RolloutScheduler:
    def __init__(self, config: RolloutSchedulerConfig | None = None) -> None:
        self.config = config or RolloutSchedulerConfig()
        self.signature_stats: dict[str, SignatureRunningStats] = {}
        self.pool_signature_counts: dict[str, int] = {}
        self.selected_signature_counts: dict[str, int] = {}
        self.total_updates = 0
        self.last_plan: dict[str, Any] = {}

    def prime(self, tasks: Sequence[Any], task_signature_fn: TaskSignatureFn | None) -> None:
        counts: dict[str, int] = {}
        for task in tasks:
            signature = _task_signature(task, task_signature_fn)
            counts[signature] = counts.get(signature, 0) + 1
        self.pool_signature_counts = counts

    def score_task_components(
        self,
        task: Any,
        task_signature_fn: TaskSignatureFn | None,
    ) -> dict[str, Any]:
        signature = _task_signature(task, task_signature_fn)
        stats = self.signature_stats.get(signature)
        pool_count = self.pool_signature_counts.get(signature, 1)
        pool_rarity = 1.0 / math.sqrt(max(pool_count, 1))
        if stats is None or stats.records == 0:
            variance_signal = 0.5
            active_signal = 0.5
            difficulty_signal = 0.5
            seen_groups = 0
        else:
            variance_signal = min(stats.reward_std * 2.0, 1.0)
            active_signal = stats.active_group_fraction
            difficulty_signal = max(0.0, 1.0 - abs(stats.pass_rate - 0.5) * 2.0)
            seen_groups = stats.groups

        novelty = 1.0 / math.sqrt(max(seen_groups, 0) + 1.0)
        official_gap = max(_task_float(task, "qwen_gap", "official_gap", "official_instruct_gap"), 0.0)
        expected_tokens = _task_expected_tokens(task)
        cost = min(expected_tokens / 4096.0, 1.0)
        selected_count = self.selected_signature_counts.get(signature, 0)
        selection_penalty = min(
            self.config.selection_history_weight * math.log1p(selected_count),
            1.0,
        )
        score = (
            self.config.novelty_weight * max(novelty, pool_rarity)
            + self.config.variance_weight * variance_signal
            + self.config.active_weight * active_signal
            + self.config.difficulty_weight * difficulty_signal
            + self.config.official_gap_weight * official_gap
            - self.config.cost_weight * cost
            - selection_penalty
        )
        zero_variance_common = False
        if (
            stats is not None
            and stats.groups >= self.config.common_signature_groups
            and stats.active_groups == 0
        ):
            score *= self.config.zero_variance_common_factor
            zero_variance_common = True
        score = max(score, self.config.min_score)
        return {
            "signature": signature,
            "score": float(score),
            "novelty": float(novelty),
            "pool_rarity": float(pool_rarity),
            "variance_signal": float(variance_signal),
            "active_signal": float(active_signal),
            "difficulty_signal": float(difficulty_signal),
            "official_gap": float(official_gap),
            "estimated_cost": float(cost),
            "expected_tokens": float(expected_tokens),
            "selection_penalty": float(selection_penalty),
            "seen_groups": int(seen_groups),
            "pool_count": int(pool_count),
            "selected_count": int(selected_count),
            "zero_variance_common": bool(zero_variance_common),
        }

    def score_task(self, task: Any, task_signature_fn: TaskSignatureFn | None) -> float:
        return float(self.score_task_components(task, task_signature_fn)["score"])

    def select(
        self,
        tasks: Sequence[Any],
        *,
        batch_size: int,
        step: int,
        task_signature_fn: TaskSignatureFn | None,
        worker_offset: int = 0,
    ) -> list[Any]:
        candidates = list(tasks)
        if not candidates or batch_size <= 0:
            return []
        if not self.pool_signature_counts:
            self.prime(candidates, task_signature_fn)
        rng = random.Random(self.config.seed + step * 1009 + worker_offset * 9176)
        selected: list[Any] = []
        selected_plan: list[dict[str, Any]] = []
        available = [
            {
                "task": task,
                "task_id": _task_id(task),
                "components": self.score_task_components(task, task_signature_fn),
            }
            for task in candidates
        ]
        top_candidates = sorted(
            (
                {
                    "task_id": row["task_id"],
                    **dict(row["components"]),
                }
                for row in available
            ),
            key=lambda row: float(row["score"]),
            reverse=True,
        )[: max(int(self.config.max_top_candidates), 0)]
        signature_limit: int | None = None
        if 0.0 < self.config.max_signature_fraction < 1.0:
            signature_limit = max(1, math.ceil(batch_size * self.config.max_signature_fraction))
        local_signature_counts: dict[str, int] = {}
        max_expected_tokens = (
            int(self.config.max_expected_tokens_per_batch)
            if self.config.max_expected_tokens_per_batch is not None
            and int(self.config.max_expected_tokens_per_batch) > 0
            else None
        )
        min_batch_size = max(1, int(self.config.min_batch_size or 1))
        selected_expected_tokens = 0.0
        selection_stopped_reason: str | None = None
        while available and len(selected) < batch_size:
            available_signatures = {
                str(row["components"]["signature"])
                for row in available
                if signature_limit is None
                or local_signature_counts.get(str(row["components"]["signature"]), 0) < signature_limit
            }
            weights = []
            token_eligible_indices: list[int] = []
            for row in available:
                signature = str(row["components"]["signature"])
                capped = (
                    signature_limit is not None
                    and local_signature_counts.get(signature, 0) >= signature_limit
                    and bool(available_signatures)
                )
                expected_tokens = float(row["components"].get("expected_tokens", 0.0) or 0.0)
                token_capped = (
                    max_expected_tokens is not None
                    and len(selected) >= min_batch_size
                    and selected_expected_tokens > 0
                    and selected_expected_tokens + expected_tokens > max_expected_tokens
                )
                if not token_capped:
                    token_eligible_indices.append(len(weights))
                weight = 0.0 if capped or token_capped else float(row["components"]["score"])
                weights.append(weight)
            total = sum(weights)
            if total <= 0:
                if max_expected_tokens is not None and selected and not token_eligible_indices:
                    selection_stopped_reason = "expected_token_budget"
                    break
                fallback_indices = token_eligible_indices or list(range(len(available)))
                index = fallback_indices[rng.randrange(len(fallback_indices))]
            else:
                threshold = rng.random() * total
                cumulative = 0.0
                index = len(available) - 1
                for idx, weight in enumerate(weights):
                    cumulative += weight
                    if cumulative >= threshold:
                        index = idx
                        break
            row = available.pop(index)
            task = row["task"]
            signature = str(row["components"]["signature"])
            local_signature_counts[signature] = local_signature_counts.get(signature, 0) + 1
            self.selected_signature_counts[signature] = self.selected_signature_counts.get(signature, 0) + 1
            selected_expected_tokens += float(row["components"].get("expected_tokens", 0.0) or 0.0)
            selected.append(task)
            selected_plan.append(
                {
                    "task_id": row["task_id"],
                    **dict(row["components"]),
                    "selection_index": len(selected) - 1,
                }
            )
        self.last_plan = {
            "step": int(step),
            "worker_offset": int(worker_offset),
            "requested_batch_size": int(batch_size),
            "pool_size": int(len(candidates)),
            "selected_count": int(len(selected)),
            "signature_limit": signature_limit,
            "max_expected_tokens_per_batch": max_expected_tokens,
            "min_batch_size": int(min_batch_size),
            "selected_expected_tokens": float(selected_expected_tokens),
            "selection_stopped_reason": selection_stopped_reason,
            "local_signature_counts": dict(sorted(local_signature_counts.items())),
            "selected": selected_plan,
            "top_candidates": top_candidates,
        }
        if max_expected_tokens is not None:
            self.last_plan["selected_expected_token_budget_fraction"] = (
                float(selected_expected_tokens) / max(float(max_expected_tokens), 1.0)
            )
            self.last_plan["selected_expected_token_budget_headroom"] = (
                float(max_expected_tokens) - float(selected_expected_tokens)
            )
        return selected

    def update(self, rollout_batch: RolloutBatch) -> dict[str, float]:
        for signature, summary in rollout_batch.signature_stats.items():
            stats = self.signature_stats.setdefault(signature, SignatureRunningStats(signature=signature))
            stats.update(summary)
        self.total_updates += 1
        active_signatures = sum(1 for stats in self.signature_stats.values() if stats.active_groups > 0)
        last_plan = self.last_plan if isinstance(self.last_plan, Mapping) else {}
        selected_expected_tokens = float(last_plan.get("selected_expected_tokens", 0.0) or 0.0)
        expected_token_budget = float(last_plan.get("max_expected_tokens_per_batch", 0.0) or 0.0)
        expected_token_budget_fraction = (
            selected_expected_tokens / expected_token_budget if expected_token_budget > 0 else 0.0
        )
        expected_token_budget_headroom = (
            expected_token_budget - selected_expected_tokens if expected_token_budget > 0 else 0.0
        )
        return {
            "rollout_scheduler_updates": float(self.total_updates),
            "rollout_scheduler_signature_count": float(len(self.signature_stats)),
            "rollout_scheduler_active_signature_count": float(active_signatures),
            "rollout_scheduler_selected_expected_tokens": selected_expected_tokens,
            "rollout_scheduler_expected_token_budget": expected_token_budget,
            "rollout_scheduler_expected_token_budget_fraction": expected_token_budget_fraction,
            "rollout_scheduler_expected_token_budget_headroom": expected_token_budget_headroom,
            "rollout_scheduler_expected_token_budget_exhausted": float(
                last_plan.get("selection_stopped_reason") == "expected_token_budget"
            ),
        }

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            "config": {
                field_name: getattr(self.config, field_name)
                for field_name in self.config.__dataclass_fields__
            },
            "updates": self.total_updates,
            "pool_signature_counts": dict(sorted(self.pool_signature_counts.items())),
            "selected_signature_counts": dict(sorted(self.selected_signature_counts.items())),
            "signatures": {
                signature: stats.to_dict()
                for signature, stats in sorted(self.signature_stats.items())
            },
            "last_plan": dict(self.last_plan),
        }

    def state_dict(self) -> dict[str, Any]:
        return self.stats_snapshot()

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        # Checkpoint state restores scheduler learning/statistics, but the active
        # launch config remains authoritative. This lets a resumed spot run pick
        # up tighter token budgets or other operator overrides without discarding
        # accumulated signature statistics from the checkpoint.
        signatures = value.get("signatures")
        if isinstance(signatures, Mapping):
            self.signature_stats = {
                str(signature): SignatureRunningStats.from_dict(stats)
                for signature, stats in signatures.items()
                if isinstance(stats, Mapping)
            }
        pool_counts = value.get("pool_signature_counts")
        if isinstance(pool_counts, Mapping):
            self.pool_signature_counts = {
                str(signature): int(count) for signature, count in pool_counts.items()
            }
        selected_counts = value.get("selected_signature_counts")
        if isinstance(selected_counts, Mapping):
            self.selected_signature_counts = {
                str(signature): int(count) for signature, count in selected_counts.items()
            }
        self.total_updates = int(value.get("updates", 0) or 0)
        last_plan = value.get("last_plan")
        self.last_plan = dict(last_plan) if isinstance(last_plan, Mapping) else {}

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> "RolloutScheduler":
        scheduler = cls()
        scheduler.load_state_dict(value)
        return scheduler


RewardFn = Callable[[Any, str], Any]
PromptFn = Callable[[Any], str]
TaskSignatureFn = Callable[[Any], str]
ProgressFn = Callable[[str], None]
RolloutCallback = Callable[[list[RolloutRecord], dict[str, Any], Context], None]
EnvironmentReplayFn = Callable[[Any, str], Mapping[str, Any] | None]


def _metric_key(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")[:48] or "unknown"


def _failure_reason(record: RolloutRecord) -> str | None:
    if record.reward >= 0.9:
        return None
    for key in ("failure_reason", "error", "schema_error"):
        value = record.reward_metrics.get(key) if isinstance(record.reward_metrics, Mapping) else None
        if isinstance(value, str) and value:
            return value
    task = record.task if isinstance(record.task, Mapping) else {}
    verifier = task.get("verifier") if isinstance(task.get("verifier"), Mapping) else {}
    family = str(task.get("family") or "unknown")
    verifier_type = str(verifier.get("type") or "unknown")
    return f"{family}:{verifier_type}:reward_below_threshold"


def _task_signature(task: Any, task_signature_fn: TaskSignatureFn | None) -> str:
    if task_signature_fn is not None:
        return task_signature_fn(task)
    if isinstance(task, Mapping):
        family = task.get("family") or "unknown"
        verifier = task.get("verifier") if isinstance(task.get("verifier"), Mapping) else {}
        return f"{family}:{verifier.get('type') or 'unknown'}"
    return type(task).__name__


def _task_id(task: Any) -> str:
    if isinstance(task, Mapping):
        for key in ("id", "task_id", "prompt_id"):
            value = task.get(key)
            if value is not None:
                return str(value)
    return str(id(task))


def _task_float(task: Any, *keys: str) -> float:
    if not isinstance(task, Mapping):
        return 0.0
    sources = [task]
    metadata = task.get("metadata")
    if isinstance(metadata, Mapping):
        sources.append(metadata)
        quality = metadata.get("quality")
        if isinstance(quality, Mapping):
            sources.append(quality)
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    continue
    return 0.0


def _task_expected_tokens(task: Any) -> float:
    return max(
        _task_float(task, "expected_tokens", "rollout_max_new_tokens", "target_response_tokens"),
        0.0,
    )


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


def _call_reward_fn(
    reward_fn: RewardFn,
    task: Any,
    completion_text: str,
    *,
    completion_ids: Sequence[int],
    max_new_tokens: int,
) -> Any:
    try:
        signature = inspect.signature(reward_fn)
    except (TypeError, ValueError):
        return reward_fn(task, completion_text)

    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if accepts_kwargs or "completion_ids" in parameters:
        kwargs["completion_ids"] = list(completion_ids)
    if accepts_kwargs or "max_new_tokens" in parameters:
        kwargs["max_new_tokens"] = int(max_new_tokens)
    if kwargs:
        return reward_fn(task, completion_text, **kwargs)
    return reward_fn(task, completion_text)


def rebalance_records_across_ranks(
    records: Sequence[RolloutRecord],
    *,
    step: int = 0,
) -> list[RolloutRecord]:
    """Gather selected rollout records from every rank, shuffle deterministically,
    and redistribute so every rank ends up with the same count (+/-1).

    Prefer avoiding this path for high-throughput vLLM GRPO. Moving full
    completion payloads through object collectives is expensive; the Ray/vLLM
    path keeps batching aligned before generation instead.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return list(records)
    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return list(records)
    rank = torch.distributed.get_rank()

    local_dicts = [rollout_record_to_dict(r) for r in records]
    gathered: list[list[dict[str, Any]] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, local_dicts)

    all_dicts: list[dict[str, Any]] = []
    for per_rank in gathered:
        if per_rank:
            all_dicts.extend(per_rank)

    if not all_dicts:
        return []

    rng = random.Random(0xCAFE ^ int(step))
    rng.shuffle(all_dicts)

    n_total = len(all_dicts)
    per_rank = n_total // world_size
    remainder = n_total % world_size
    start = rank * per_rank + min(rank, remainder)
    end = start + per_rank + (1 if rank < remainder else 0)
    return [rollout_record_from_dict(d) for d in all_dicts[start:end]]


def rollout_record_to_dict(record: RolloutRecord) -> dict[str, Any]:
    return {
        "task": record.task,
        "group_id": record.group_id,
        "sample_id": record.sample_id,
        "prompt_text": record.prompt_text,
        "prompt_ids": list(record.prompt_ids),
        "completion_ids": list(record.completion_ids),
        "completion_text": record.completion_text,
        "reward": float(record.reward),
        "reward_metrics": dict(record.reward_metrics or {}),
        "token_advantages": list(record.token_advantages) if record.token_advantages is not None else None,
        "policy_version": record.policy_version.to_dict() if record.policy_version is not None else None,
        "task_signature": record.task_signature,
        "skip_reason": record.skip_reason,
        "environment_transcript": (
            [dict(item) for item in record.environment_transcript]
            if record.environment_transcript is not None
            else None
        ),
    }


def rollout_record_from_dict(row: Mapping[str, Any]) -> RolloutRecord:
    missing = [
        key
        for key in (
            "task",
            "group_id",
            "sample_id",
            "prompt_text",
            "prompt_ids",
            "completion_ids",
            "completion_text",
            "reward",
        )
        if key not in row
    ]
    if missing:
        raise ValueError(f"Rollout row is missing required field(s): {', '.join(missing)}")

    return RolloutRecord(
        task=row["task"],
        group_id=str(row["group_id"]),
        sample_id=str(row["sample_id"]),
        prompt_text=str(row["prompt_text"]),
        prompt_ids=[int(token_id) for token_id in row["prompt_ids"]],
        completion_ids=[int(token_id) for token_id in row["completion_ids"]],
        completion_text=str(row["completion_text"]),
        reward=float(row["reward"]),
        reward_metrics=dict(row.get("reward_metrics") or {}),
        token_advantages=(
            [float(value) for value in row["token_advantages"]]
            if isinstance(row.get("token_advantages"), Sequence)
            and not isinstance(row.get("token_advantages"), (str, bytes, bytearray))
            else None
        ),
        policy_version=(
            PolicyVersion.from_dict(row["policy_version"])
            if isinstance(row.get("policy_version"), Mapping)
            else None
        ),
        task_signature=str(row["task_signature"]) if row.get("task_signature") is not None else None,
        skip_reason=str(row["skip_reason"]) if row.get("skip_reason") is not None else None,
        environment_transcript=(
            [dict(item) for item in row["environment_transcript"] if isinstance(item, Mapping)]
            if isinstance(row.get("environment_transcript"), Sequence)
            and not isinstance(row.get("environment_transcript"), (str, bytes, bytearray))
            else None
        ),
    )


def coerce_rollout_record(value: Any) -> RolloutRecord:
    if isinstance(value, RolloutRecord):
        return value
    if isinstance(value, Mapping):
        return rollout_record_from_dict(value)
    raise TypeError(f"Expected RolloutRecord or mapping, got {type(value).__name__}.")


def make_no_signal_keepalive_record(
    record: RolloutRecord,
    *,
    fallback_token_id: int,
) -> RolloutRecord:
    keepalive_token_id = int(fallback_token_id)
    return replace(
        record,
        sample_id=f"{record.sample_id}:keepalive",
        prompt_text="",
        prompt_ids=[keepalive_token_id],
        completion_ids=[keepalive_token_id],
        completion_text="",
        reward=0.0,
        reward_metrics={**dict(record.reward_metrics or {}), "no_signal_keepalive": 1.0},
        token_advantages=[0.0],
        skip_reason="no_signal_keepalive",
    )


def apply_functional_credit_to_records(
    records: Sequence[RolloutRecord],
    *,
    config: FunctionalCreditConfig,
) -> tuple[list[RolloutRecord], dict[str, Any]]:
    rows = [rollout_record_to_dict(record) for record in records]
    credited_rows, summary = assign_functional_token_advantages(rows, config=config)
    return [rollout_record_from_dict(row) for row in credited_rows], summary


def flatten_rollout_records(batch: Any) -> list[RolloutRecord]:
    if isinstance(batch, RolloutRecord):
        return [batch]
    if isinstance(batch, Mapping):
        if "records" in batch:
            return flatten_rollout_records(batch["records"])
        return [rollout_record_from_dict(batch)]
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes, bytearray)):
        records: list[RolloutRecord] = []
        for item in batch:
            records.extend(flatten_rollout_records(item))
        return records
    raise TypeError(f"Could not interpret rollout batch of type {type(batch).__name__}.")


def _normal_stop_token_ids(stop_token_ids: Sequence[int] | int | None) -> list[int]:
    if stop_token_ids is None:
        return []
    if isinstance(stop_token_ids, int):
        return [int(stop_token_ids)]
    return [int(token_id) for token_id in stop_token_ids if token_id is not None]


def _trim_completion(ids: list[int], stop_token_ids: Sequence[int] | int | None) -> list[int]:
    stops = set(_normal_stop_token_ids(stop_token_ids))
    if not stops:
        return ids
    for index, token_id in enumerate(ids):
        if token_id in stops:
            return ids[: index + 1]
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
    stop_token_ids: Sequence[int] | int | None,
    progress_fn: Callable[[int], None] | None = None,
    progress_every: int = 0,
) -> list[list[int]]:
    batch_size = input_ids.shape[0]
    completions: list[list[int]] = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, device=input_ids.device, dtype=torch.bool)
    current_ids = input_ids
    current_mask = attention_mask
    stops = _normal_stop_token_ids(stop_token_ids)
    stop_tensor = (
        torch.tensor(stops, device=input_ids.device, dtype=input_ids.dtype)
        if stops
        else None
    )
    filler = stops[0] if stops else pad_token_id

    for token_idx in range(max_new_tokens):
        output = model(input_ids=current_ids, attention_mask=current_mask)
        next_tokens = _sample_next_tokens(output.logits[:, -1, :], temperature=temperature, top_p=top_p)
        active = ~finished
        active_flags = active.detach().cpu().tolist()

        for row, token in enumerate(next_tokens.detach().cpu().tolist()):
            if active_flags[row]:
                completions[row].append(int(token))

        if stop_tensor is not None:
            finished = finished | (active & torch.isin(next_tokens, stop_tensor))

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
        if bool(finished.all().item()):
            break

    return [_trim_completion(ids, stop_token_ids) for ids in completions]


def reward_summary(records: Sequence[RolloutRecord]) -> dict[str, float]:
    rewards = [r.reward for r in records]
    by_group: dict[str, list[float]] = {}
    reward_metric_values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_group.setdefault(record.group_id, []).append(record.reward)
        if isinstance(record.reward_metrics, Mapping):
            for key, value in record.reward_metrics.items():
                if isinstance(value, (int, float)):
                    reward_metric_values[str(key)].append(float(value))

    group_stds = []
    active_groups = 0
    for values in by_group.values():
        if len(values) > 1:
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            group_stds.append(std)
            if std > 1e-8:
                active_groups += 1

    summary = {
        "reward_mean": sum(rewards) / max(len(rewards), 1),
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "group_reward_std_mean": sum(group_stds) / max(len(group_stds), 1),
        "active_group_fraction": active_groups / max(len(by_group), 1),
    }
    for key, values in sorted(reward_metric_values.items()):
        summary[f"reward_metric_{_metric_key(key)}_mean"] = sum(values) / max(len(values), 1)
    return summary


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
    max_new_tokens: int | Callable[[Any], int],
    temperature: float,
    top_p: float,
    device: torch.device,
    step: int,
    policy_version: PolicyVersion | None = None,
    task_signature_fn: TaskSignatureFn | None = None,
    rollout_do_sample: bool = True,
    rollout_use_cache: bool = True,
    rollout_disable_compile: bool = True,
    rollout_backend: str = "manual",
    rollout_stop_token_ids: Sequence[int] | int | None = None,
    prompt_batch_size: int = 1,
    rollout_log_every: int = 1,
    rollout_token_log_every: int = 16,
    batch_records_fn: Callable[[list[RolloutRecord]], None] | None = None,
    progress_fn: ProgressFn | None = None,
    environment_replay_fn: EnvironmentReplayFn | None = None,
    vllm_engine: Any = None,
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
        prompt_batch_size = max(1, int(prompt_batch_size))
        group_offset = 0
        rank_aligned_ray_vllm = (
            rollout_backend == "ray_vllm"
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        total_task_slots = (
            _distributed_max_int(len(tasks), device=device)
            if rank_aligned_ray_vllm
            else len(tasks)
        )
        while group_offset < total_task_slots:
            if group_offset >= len(tasks):
                # Ray/vLLM generation is itself a distributed collective over trainer ranks.
                # Ranks with fewer local rollout tasks still need to enter the collective
                # for this slot so the following ranks do not drift into a different
                # collective such as rollout rebalancing.
                if rollout_backend == "ray_vllm":
                    if vllm_engine is None:
                        raise ValueError("rollout_backend='ray_vllm' requires a vllm_engine instance")
                    vllm_engine.generate([], [], use_tqdm=False)
                group_offset += prompt_batch_size
                continue
            first_task = tasks[group_offset]
            first_task_max_new_tokens = int(max_new_tokens(first_task) if callable(max_new_tokens) else max_new_tokens)
            task_batch = [first_task]
            batch_end = group_offset + 1
            while (
                rollout_backend == "ray_vllm"
                and batch_end < len(tasks)
                and len(task_batch) < prompt_batch_size
            ):
                candidate = tasks[batch_end]
                task_batch.append(candidate)
                batch_end += 1
            while (
                rollout_backend != "ray_vllm"
                and batch_end < len(tasks)
                and len(task_batch) < prompt_batch_size
            ):
                candidate = tasks[batch_end]
                candidate_max_new_tokens = int(max_new_tokens(candidate) if callable(max_new_tokens) else max_new_tokens)
                if callable(max_new_tokens) and candidate_max_new_tokens != first_task_max_new_tokens:
                    break
                task_batch.append(candidate)
                batch_end += 1
            group_ids = [group_id_fn(task) for task in task_batch]
            task_signatures = [
                task_signature_fn(task) if task_signature_fn is not None else None
                for task in task_batch
            ]
            task_max_new_tokens_batch = [
                int(max_new_tokens(task) if callable(max_new_tokens) else max_new_tokens)
                for task in task_batch
            ]
            task_max_new_tokens = first_task_max_new_tokens
            should_log = (
                progress_fn is not None
                and rollout_log_every > 0
                and (
                    group_offset == 0
                    or batch_end % rollout_log_every == 0
                    or batch_end >= len(tasks)
                )
            )
            group_start = time.time()
            if should_log:
                token_caps = sorted(set(task_max_new_tokens_batch))
                cap_text = (
                    str(token_caps[0])
                    if len(token_caps) == 1
                    else ",".join(str(value) for value in token_caps)
                )
                progress_fn(
                    f"rollout task {group_offset + 1}-{batch_end}/{len(tasks)} start "
                    f"ids={','.join(group_ids)} max_new_tokens={cap_text}"
                )

            prompt_texts = [render_prompt(task) for task in task_batch]
            prompt_ids_batch = [
                tokenizer(prompt_text, add_special_tokens=False).input_ids
                for prompt_text in prompt_texts
            ]
            expanded_prompt_texts = [
                prompt_text
                for prompt_text in prompt_texts
                for _ in range(rollouts_per_prompt)
            ]
            encoded = tokenizer(
                expanded_prompt_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
            stop_token_ids = _normal_stop_token_ids(
                rollout_stop_token_ids
                if rollout_stop_token_ids is not None
                else tokenizer.eos_token_id
            )

            if rollout_backend in {"vllm", "ray_vllm"}:
                if rollout_backend == "vllm":
                    raise ValueError(
                        "rollout_backend='vllm' was the removed colocated backend; "
                        "use rollout_backend='ray_vllm' for split trainer/inference vLLM."
                    )
                if vllm_engine is None:
                    raise ValueError("rollout_backend='ray_vllm' requires a vllm_engine instance")
                from vllm import SamplingParams
                vllm_stops = list(stop_token_ids) if stop_token_ids else None
                sampling_params = [
                    SamplingParams(
                        n=rollouts_per_prompt,
                        temperature=temperature if rollout_do_sample else 0.0,
                        top_p=top_p,
                        max_tokens=int(task_tokens),
                        stop_token_ids=vllm_stops,
                        seed=None,
                    )
                    for task_tokens in task_max_new_tokens_batch
                ]
                # vLLM takes one prompt per task and expands n=G internally. The order in
                # returned outputs matches our existing (task_idx, sample_idx) flattening:
                # task0_s0, task0_s1, ..., task0_sG-1, task1_s0, ...
                vllm_outputs = vllm_engine.generate(prompt_texts, sampling_params, use_tqdm=False)
                completion_batches = []
                for req_out in vllm_outputs:
                    for compl_out in req_out.outputs:
                        completion_batches.append(_trim_completion(list(compl_out.token_ids), stop_token_ids))
            elif rollout_backend == "manual":
                completion_batches = _manual_generate_completion_ids(
                    model=model,
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=task_max_new_tokens,
                    temperature=temperature if rollout_do_sample else 0.0,
                    top_p=top_p,
                    pad_token_id=pad_token_id,
                    stop_token_ids=stop_token_ids,
                    progress_fn=(
                        (
                            lambda generated, group_offset=group_offset, batch_end=batch_end, group_ids=group_ids: progress_fn(
                                f"rollout task {group_offset + 1}-{batch_end}/{len(tasks)} "
                                f"tokens_generated={generated}/{task_max_new_tokens} ids={','.join(group_ids)}"
                            )
                        )
                        if should_log and progress_fn is not None
                        else None
                    ),
                    progress_every=rollout_token_log_every,
                )
            elif rollout_backend == "hf_generate":
                input_width = encoded["input_ids"].shape[1]
                generate_kwargs = {
                    **encoded,
                    "do_sample": rollout_do_sample,
                    "max_new_tokens": task_max_new_tokens,
                    "use_cache": rollout_use_cache,
                    "disable_compile": rollout_disable_compile,
                    "pad_token_id": pad_token_id,
                    "eos_token_id": stop_token_ids[0] if len(stop_token_ids) == 1 else stop_token_ids,
                }
                if rollout_do_sample:
                    generate_kwargs["temperature"] = temperature
                    generate_kwargs["top_p"] = top_p
                output_ids = model.generate(**generate_kwargs)
                completion_batches = [
                    _trim_completion(output[input_width:].detach().cpu().tolist(), stop_token_ids)
                    for output in output_ids
                ]
            else:
                raise ValueError(f"Unknown rollout backend: {rollout_backend}")

            completion_token_count = 0
            batch_records: list[RolloutRecord] = []
            for sequence_idx, completion_ids in enumerate(completion_batches):
                task_batch_idx = sequence_idx // rollouts_per_prompt
                rollout_idx = sequence_idx % rollouts_per_prompt
                task = task_batch[task_batch_idx]
                group_id = group_ids[task_batch_idx]
                task_signature = task_signatures[task_batch_idx]
                task_completion_cap = int(task_max_new_tokens_batch[task_batch_idx])
                prompt_text = prompt_texts[task_batch_idx]
                prompt_ids = prompt_ids_batch[task_batch_idx]
                completion_token_count += len(completion_ids)
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                reward, reward_metrics = _coerce_reward(
                    _call_reward_fn(
                        reward_fn,
                        task,
                        completion_text,
                        completion_ids=completion_ids,
                        max_new_tokens=task_completion_cap,
                    )
                )
                reward_metrics = dict(reward_metrics or {})
                hit_max_new_tokens = len(completion_ids) >= max(int(task_completion_cap), 1)
                reward_metrics.setdefault("completion_tokens", float(len(completion_ids)))
                reward_metrics.setdefault("completion_max_new_tokens", float(task_completion_cap))
                reward_metrics.setdefault("completion_hit_max_new_tokens", float(hit_max_new_tokens))
                reward_metrics.setdefault(
                    "completion_token_fraction_of_max",
                    float(len(completion_ids)) / max(float(task_completion_cap), 1.0),
                )
                environment_transcript = None
                if environment_replay_fn is not None:
                    environment_payload = environment_replay_fn(task, completion_text)
                    if isinstance(environment_payload, Mapping):
                        transcript = environment_payload.get("transcript")
                        if isinstance(transcript, Sequence) and not isinstance(transcript, (str, bytes, bytearray)):
                            environment_transcript = [
                                dict(item) for item in transcript if isinstance(item, Mapping)
                            ]
                        env_metrics = environment_payload.get("metrics")
                        if isinstance(env_metrics, Mapping):
                            reward_metrics = {
                                **dict(reward_metrics or {}),
                                **{
                                    str(key): float(value)
                                    for key, value in env_metrics.items()
                                    if isinstance(value, (int, float))
                                },
                            }
                record = RolloutRecord(
                    task=task,
                    group_id=group_id,
                    sample_id=f"s{step}-{group_offset + task_batch_idx}-{rollout_idx}",
                    prompt_text=prompt_text,
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    completion_text=completion_text,
                    reward=reward,
                    reward_metrics=reward_metrics,
                    policy_version=policy_version,
                    task_signature=task_signature,
                    environment_transcript=environment_transcript,
                )
                records.append(record)
                batch_records.append(record)

            if batch_records_fn is not None:
                batch_records_fn(batch_records)

            if should_log:
                progress_fn(
                    f"rollout task {batch_end}/{len(tasks)} done ids={','.join(group_ids)} "
                    f"tokens={completion_token_count} elapsed={time.time() - group_start:.1f}s"
                )
            group_offset = (
                group_offset + prompt_batch_size
                if rollout_backend == "ray_vllm"
                else batch_end
            )
        return records
    finally:
        if rollout_backend == "hf_generate" and old_gradient_checkpointing:
            _call_if_present(model, "gradient_checkpointing_enable")
        if old_use_cache is not None and model_config is not None:
            model_config.use_cache = old_use_cache


def collate_rollouts(
    records: Sequence[RolloutRecord],
    tokenizer,
    device: torch.device,
    grpo_config: GRPOConfig | None = None,
) -> dict[str, Any]:
    config = grpo_config or GRPOConfig()
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
    scalar_advantages = compute_group_advantages(
        rewards,
        group_ids,
        normalize=config.normalize_advantages,
        center=config.center_advantages,
        epsilon=config.epsilon,
    )
    token_advantages = torch.zeros((batch, max_len), dtype=torch.float32)
    for row, record in enumerate(records):
        prompt_len = len(record.prompt_ids)
        seq_len = prompt_len + len(record.completion_ids)
        if record.token_advantages is None:
            token_advantages[row, prompt_len:seq_len] = scalar_advantages[row]
        else:
            values = torch.tensor(record.token_advantages, dtype=torch.float32)
            usable = min(values.numel(), len(record.completion_ids))
            if usable:
                token_advantages[row, prompt_len : prompt_len + usable] = values[:usable]

    batch_payload = {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device),
        "completion_mask": completion_mask.to(device),
        "advantages": token_advantages.to(device),
        "rewards": rewards,
        "group_ids": group_ids,
    }
    return batch_payload


def _distributed_max_int(value: int, *, device: torch.device) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return int(value)
    tensor = torch.tensor([int(value)], device=device, dtype=torch.int64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return int(tensor.item())


def _slice_tensor_batch(batch: dict[str, Any], start: int, end: int, *, batch_size: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def _model_floating_dtype(model: Any) -> torch.dtype:
    parameters = getattr(model, "parameters", None)
    if parameters is not None:
        try:
            for parameter in parameters():
                if isinstance(parameter, torch.Tensor) and parameter.dtype.is_floating_point:
                    return parameter.dtype
        except Exception:
            pass
    return torch.float32


def _compute_old_logprobs_single(model, batch: dict[str, Any], config: GRPOConfig) -> torch.Tensor:
    forward_kwargs, logits_positions = prepare_grpo_forward_kwargs(
        model=model,
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        mask=batch["completion_mask"],
        logprob_source=config.logprob_source,
    )
    output = model(batch["input_ids"], **forward_kwargs)

    labels = batch["labels"]
    shifted_labels = labels[:, 1:]
    valid_mask = batch["completion_mask"][:, 1:] & shifted_labels.ne(-100)

    if config.logprob_source == "hidden_states":
        hidden_states = extract_last_hidden_state(output)
        token_logprobs = gather_selected_logprobs_from_hidden(
            hidden_states[:, :-1, :],
            shifted_labels,
            valid_mask,
            output_projection=resolve_output_projection(model),
            chunk_size=config.logprob_chunk_size,
        )
        full_logprobs = torch.zeros(labels.shape, device=labels.device, dtype=token_logprobs.dtype)
        full_logprobs[:, 1:] = token_logprobs
        return full_logprobs

    logits = output.logits
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


@torch.no_grad()
def compute_old_logprobs(
    model,
    batch: dict[str, Any],
    config: GRPOConfig,
    *,
    micro_batch_size: int | None = None,
) -> torch.Tensor:
    model.eval()
    labels = batch["labels"]
    batch_size = int(labels.shape[0])
    micro_bs = int(micro_batch_size or 0)
    if micro_bs <= 0:
        return _compute_old_logprobs_single(model, batch, config)

    local_chunks = max((batch_size + micro_bs - 1) // micro_bs, 1)
    num_chunks = _distributed_max_int(local_chunks, device=labels.device)
    if num_chunks <= 1:
        return _compute_old_logprobs_single(model, batch, config)
    effective_micro_bs = (batch_size + num_chunks - 1) // num_chunks
    full_logprobs = torch.zeros(labels.shape, device=labels.device, dtype=_model_floating_dtype(model))

    for chunk_idx in range(num_chunks):
        start = chunk_idx * effective_micro_bs
        end = min(start + effective_micro_bs, batch_size)
        has_real_rows = start < end
        if not has_real_rows:
            start, end = batch_size - 1, batch_size
        chunk = _slice_tensor_batch(batch, start, end, batch_size=batch_size)
        chunk_logprobs = _compute_old_logprobs_single(model, chunk, config)
        if has_real_rows:
            if full_logprobs.dtype != chunk_logprobs.dtype:
                full_logprobs = full_logprobs.to(dtype=chunk_logprobs.dtype)
            full_logprobs[start:end] = chunk_logprobs[: end - start]
        del chunk_logprobs

    return full_logprobs


def prepare_rollout_training_context(
    *,
    model,
    tokenizer,
    records: Sequence[RolloutRecord],
    device: torch.device,
    grpo_config: GRPOConfig,
    ctx: Context,
    rollout_batch: RolloutBatch | None = None,
    progress_fn: ProgressFn | None = None,
    step: int | None = None,
    max_policy_age_updates: int | None = 0,
    max_rollout_reuse: int | None = 1,
    old_logprob_micro_batch_size: int | None = None,
) -> torch.Tensor:
    if rollout_batch is not None:
        rollout_batch.validate_for_step(
            current_step=int(step if step is not None else ctx.get("total_steps", 0)),
            max_policy_age_updates=max_policy_age_updates,
            max_rollout_reuse=max_rollout_reuse,
        )
        records = rollout_batch.records
    if not records:
        raise ValueError("GRPO rollout training context requires at least one selected rollout.")
    grpo_batch = collate_rollouts(records, tokenizer, device, grpo_config=grpo_config)
    if progress_fn:
        step_prefix = f"step {step} " if step is not None else ""
        micro_suffix = (
            f" micro_batch_size={int(old_logprob_micro_batch_size)}"
            if old_logprob_micro_batch_size is not None and old_logprob_micro_batch_size > 0
            else ""
        )
        progress_fn(f"{step_prefix}old_logprobs start{micro_suffix}")
    old_logprobs = compute_old_logprobs(
        model,
        grpo_batch,
        grpo_config,
        micro_batch_size=old_logprob_micro_batch_size,
    )
    if progress_fn:
        step_prefix = f"step {step} " if step is not None else ""
        progress_fn(f"{step_prefix}old_logprobs done")
    model.train()

    forward_kwargs, logits_positions = prepare_grpo_forward_kwargs(
        model=model,
        attention_mask=grpo_batch["attention_mask"],
        labels=grpo_batch["labels"],
        mask=grpo_batch["completion_mask"],
        logprob_source=grpo_config.logprob_source,
    )
    ctx["forward_kwargs"] = forward_kwargs
    ctx["logits_positions"] = logits_positions
    ctx["target"] = grpo_batch["labels"]
    ctx["mask"] = grpo_batch["completion_mask"]
    ctx["old_logprobs"] = old_logprobs
    ctx["advantages"] = grpo_batch["advantages"]
    ctx["rollout_records"] = list(records)
    rollout_metrics = {
        "advantage_abs_sum": float(grpo_batch["advantages"].abs().sum().item()),
        **reward_summary(records),
    }
    if rollout_batch is not None:
        rollout_metrics.update(rollout_batch.metrics)
        ctx["rollout_batch"] = rollout_batch
        ctx["rollout_all_records"] = list(rollout_batch.all_records)
        ctx["rollout_signature_stats"] = dict(rollout_batch.signature_stats)
        ctx["rollout_failure_reasons"] = dict(rollout_batch.failure_reasons)
    ctx["rollout_metrics"] = rollout_metrics
    return grpo_batch["input_ids"]


class GRPORolloutPreProcessor(PreProcessor):
    def __init__(
        self,
        *,
        tokenizer,
        render_prompt: PromptFn,
        reward_fn: RewardFn,
        grpo_config: GRPOConfig | None = None,
        group_id_fn: Callable[[Any], str] | None = None,
        task_signature_fn: TaskSignatureFn | None = None,
        rollouts_per_prompt: int = 4,
        max_new_tokens: int | Callable[[Any], int] = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        rollout_do_sample: bool = True,
        rollout_use_cache: bool = True,
        rollout_disable_compile: bool = True,
        rollout_backend: str = "manual",
        prompt_batch_size: int = 1,
        rollout_log_every: int = 1,
        rollout_token_log_every: int = 16,
        skip_zero_variance_groups: bool = True,
        min_group_reward_std: float = 0.0,
        max_policy_age_updates: int | None = 0,
        max_rollout_reuse: int | None = 1,
        rollout_scheduler: RolloutScheduler | None = None,
        task_pool: Sequence[Any] | None = None,
        environment_replay_fn: EnvironmentReplayFn | None = None,
        functional_credit_config: FunctionalCreditConfig | None = None,
        progress_fn: ProgressFn | None = None,
        on_rollouts: RolloutCallback | None = None,
        rebalance_across_ranks: bool = False,
        loss_micro_batch_size: int | None = None,
        old_logprob_micro_batch_size: int | None = None,
        oversample_multiplier: float = 1.0,
        vllm_engine: Any = None,
        vllm_engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(contract="")
        self.tokenizer = tokenizer
        self.render_prompt = render_prompt
        self.reward_fn = reward_fn
        self.grpo_config = grpo_config or GRPOConfig()
        self.group_id_fn = group_id_fn or (lambda task: str(task["id"] if isinstance(task, dict) else id(task)))
        self.task_signature_fn = task_signature_fn
        self.rollouts_per_prompt = rollouts_per_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.rollout_do_sample = rollout_do_sample
        self.rollout_use_cache = rollout_use_cache
        self.rollout_disable_compile = rollout_disable_compile
        self.rollout_backend = rollout_backend
        self.prompt_batch_size = prompt_batch_size
        self.rollout_log_every = rollout_log_every
        self.rollout_token_log_every = rollout_token_log_every
        self.skip_zero_variance_groups = skip_zero_variance_groups
        self.min_group_reward_std = min_group_reward_std
        self.max_policy_age_updates = max_policy_age_updates
        self.max_rollout_reuse = max_rollout_reuse
        self.rollout_scheduler = rollout_scheduler
        self.task_pool = list(task_pool) if task_pool is not None else None
        self.environment_replay_fn = environment_replay_fn
        self.functional_credit_config = functional_credit_config
        if self.rollout_scheduler is not None and self.task_pool is not None:
            self.rollout_scheduler.prime(self.task_pool, self.task_signature_fn)
        self.progress_fn = progress_fn
        self.on_rollouts = on_rollouts
        self.rebalance_across_ranks = bool(rebalance_across_ranks)
        self.loss_micro_batch_size = (
            int(loss_micro_batch_size) if loss_micro_batch_size and loss_micro_batch_size > 0 else None
        )
        effective_old_logprob_micro_batch_size = (
            loss_micro_batch_size if old_logprob_micro_batch_size is None else old_logprob_micro_batch_size
        )
        self.old_logprob_micro_batch_size = (
            int(effective_old_logprob_micro_batch_size)
            if effective_old_logprob_micro_batch_size and effective_old_logprob_micro_batch_size > 0
            else None
        )
        self.oversample_multiplier = float(oversample_multiplier) if oversample_multiplier else 1.0
        if self.oversample_multiplier < 1.0:
            self.oversample_multiplier = 1.0
        if self.oversample_multiplier > 1.0 and self.task_pool is None:
            raise ValueError(
                "oversample_multiplier > 1 requires task_pool to be supplied so extra prompts can be drawn"
            )
        self.vllm_engine = vllm_engine
        self.vllm_engine_kwargs = dict(vllm_engine_kwargs) if vllm_engine_kwargs else None
        self._last_vllm_weight_sync_step: int = -1
        if self.skip_zero_variance_groups and self.rollouts_per_prompt < 2:
            raise ValueError(
                "skip_zero_variance_groups requires rollouts_per_prompt >= 2; "
                "a one-sample group has no GRPO variance signal."
            )

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.rollout_scheduler is not None:
            state["rollout_scheduler"] = self.rollout_scheduler.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        scheduler_state = state.get("rollout_scheduler") if isinstance(state, Mapping) else None
        if isinstance(scheduler_state, Mapping) and self.rollout_scheduler is not None:
            self.rollout_scheduler.load_state_dict(scheduler_state)

    def shutdown(self) -> None:
        if self.vllm_engine is None:
            return
        if self.progress_fn:
            self.progress_fn("ray_vllm shutdown start")
        try:
            self.vllm_engine.shutdown()
        finally:
            self.vllm_engine = None
            if self.progress_fn:
                self.progress_fn("ray_vllm shutdown done")

    def process(self, batch: Sequence[Any], ctx: Context) -> tuple[torch.Tensor | None, Context]:
        model = ctx["model"]
        device = ctx["device"]
        step = int(ctx.get("total_steps", 0))
        input_tasks = list(batch)
        target_batch_size = len(input_tasks)
        oversample_extra_target = (
            int(round(target_batch_size * (self.oversample_multiplier - 1.0)))
            if self.oversample_multiplier > 1.0
            else 0
        )
        if self.rollout_scheduler is not None:
            candidates = self.task_pool if self.task_pool is not None else input_tasks
            input_tasks = self.rollout_scheduler.select(
                candidates,
                batch_size=target_batch_size + max(oversample_extra_target, 0),
                step=step,
                task_signature_fn=self.task_signature_fn,
                worker_offset=int(os.environ.get("RANK", "0")),
            )
            ctx["scheduled_tasks"] = input_tasks
        # DAPO-style oversampling: draw extra prompts from the task pool to compensate
        # for zero-variance attrition. Extra prompts run through the same rollout +
        # variance-filter pipeline; survivors flow into the rebalanced batch. Trades
        # rollout compute for a higher effective gradient-signal batch when the policy
        # is producing near-deterministic outputs on a meaningful fraction of prompts.
        if self.oversample_multiplier > 1.0 and self.task_pool is not None and self.rollout_scheduler is None:
            extra_target = oversample_extra_target
            if extra_target > 0:
                existing_ids = {self.group_id_fn(t) for t in input_tasks}
                pool = list(self.task_pool)
                rng = random.Random(0xDA90 ^ (int(step) * 1000003 + int(os.environ.get("RANK", "0"))))
                rng.shuffle(pool)
                extras: list[Any] = []
                for task in pool:
                    if len(extras) >= extra_target:
                        break
                    tid = self.group_id_fn(task)
                    if tid in existing_ids:
                        continue
                    existing_ids.add(tid)
                    extras.append(task)
                if extras:
                    input_tasks = list(input_tasks) + extras
                    ctx["oversampled_extra_tasks"] = len(extras)
        elif self.rollout_scheduler is not None and oversample_extra_target > 0:
            ctx["oversampled_extra_tasks"] = max(0, len(input_tasks) - target_batch_size)
        policy_version = PolicyVersion(
            policy_version=f"step-{step}",
            policy_checkpoint_id=(
                str(ctx.get("policy_checkpoint_id"))
                if ctx.get("policy_checkpoint_id") is not None
                else (str(ctx.get("checkpoint_id")) if ctx.get("checkpoint_id") is not None else None)
            ),
            created_step=step,
            created_at=time.time(),
            rollout_backend=self.rollout_backend,
            model_id=str(getattr(getattr(model, "config", None), "_name_or_path", "") or model.__class__.__name__),
            generation={
                "rollouts_per_prompt": self.rollouts_per_prompt,
                "max_new_tokens": (
                    self.max_new_tokens
                    if isinstance(self.max_new_tokens, int)
                    else "task_callable"
                ),
                "temperature": self.temperature,
                "top_p": self.top_p,
                "rollout_do_sample": self.rollout_do_sample,
            },
        )
        if self.progress_fn:
            scheduler_suffix = " scheduler=signature" if self.rollout_scheduler is not None else ""
            self.progress_fn(f"step {step} rollout start tasks={len(input_tasks)}{scheduler_suffix}")

        # Lazy-init the split-role Ray/vLLM engine on first process() call, after
        # torch.distributed and FSDP wrapping have completed in ditty.Pipeline.
        if self.vllm_engine is None and self.vllm_engine_kwargs is not None and self.rollout_backend == "ray_vllm":
            from ditty.ray_vllm_engine import RayVllmRolloutEngine
            if self.progress_fn:
                self.progress_fn(f"step {step} ray_vllm lazy init start")
            self.vllm_engine = RayVllmRolloutEngine(**self.vllm_engine_kwargs)
            if self.progress_fn:
                self.progress_fn(f"step {step} ray_vllm lazy init done")

        # Split-role vLLM weight sync: all trainer ranks participate in the FSDP
        # full_tensor() gathers; rank 0 broadcasts each gathered tensor to the
        # Ray-hosted vLLM TP workers over a side NCCL group.
        if self.vllm_engine is not None and step > self._last_vllm_weight_sync_step:
            if self.progress_fn:
                self.progress_fn(f"step {step} ray_vllm weight sync start")
            sync_start = time.time()
            count = self.vllm_engine.update_weights_from_fsdp_model(model)
            if self.progress_fn:
                self.progress_fn(
                    f"step {step} ray_vllm weight sync done params={count} elapsed={time.time()-sync_start:.1f}s"
                )
            self._last_vllm_weight_sync_step = step

        step_start = time.time()
        records = generate_rollouts(
            model=model,
            tokenizer=self.tokenizer,
            tasks=input_tasks,
            render_prompt=self.render_prompt,
            reward_fn=self.reward_fn,
            group_id_fn=self.group_id_fn,
            rollouts_per_prompt=self.rollouts_per_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            rollout_do_sample=self.rollout_do_sample,
            device=device,
            step=step,
            policy_version=policy_version,
            task_signature_fn=self.task_signature_fn,
            rollout_use_cache=self.rollout_use_cache,
            rollout_disable_compile=self.rollout_disable_compile,
            rollout_backend=self.rollout_backend,
            vllm_engine=self.vllm_engine,

            prompt_batch_size=self.prompt_batch_size,
            rollout_log_every=self.rollout_log_every,
            rollout_token_log_every=self.rollout_token_log_every,
            progress_fn=self.progress_fn,
            environment_replay_fn=self.environment_replay_fn,
        )
        if self.progress_fn:
            self.progress_fn(
                f"step {step} rollout done records={len(records)} elapsed={time.time() - step_start:.1f}s"
            )

        functional_credit_summary: dict[str, Any] | None = None
        if self.functional_credit_config is not None:
            records, functional_credit_summary = apply_functional_credit_to_records(
                records,
                config=self.functional_credit_config,
            )
            ctx["functional_credit_summary"] = dict(functional_credit_summary)
            if self.progress_fn:
                self.progress_fn(
                    f"step {step} functional credit done "
                    f"active_groups={functional_credit_summary.get('active_groups', 0)} "
                    f"active_turns={functional_credit_summary.get('active_turns', 0)}"
                )

        rollout_batch = RolloutBatch.from_records(
            records,
            current_step=step,
            skip_zero_variance_groups=self.skip_zero_variance_groups,
            min_group_reward_std=self.min_group_reward_std,
            max_policy_age_updates=self.max_policy_age_updates,
            max_rollout_reuse=self.max_rollout_reuse,
        )
        if functional_credit_summary is not None:
            for key, value in functional_credit_summary.items():
                if isinstance(value, (int, float)):
                    rollout_batch.metrics[f"functional_credit_{_metric_key(str(key))}"] = float(value)
        if self.rollout_scheduler is not None:
            rollout_batch.metrics.update(self.rollout_scheduler.update(rollout_batch))
            ctx["rollout_scheduler_stats"] = self.rollout_scheduler.stats_snapshot()
        # For split Ray/vLLM rollouts, batching is aligned before generation:
        # RayVllmRolloutEngine gathers per-rank prompts into one global vLLM
        # request and scatters completions back to each trainer rank. Avoid
        # moving generated rollout objects across the FSDP process group.
        if (
            self.rebalance_across_ranks
            and self.rollout_backend != "ray_vllm"
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            pre_local = len(rollout_batch.records)
            rebalanced = rebalance_records_across_ranks(rollout_batch.records, step=step)
            rollout_batch.records = rebalanced
            rollout_batch.metrics["rollout_rebalance_pre_local"] = float(pre_local)
            rollout_batch.metrics["rollout_rebalance_post_local"] = float(len(rebalanced))
        elif self.rebalance_across_ranks and self.rollout_backend == "ray_vllm":
            rollout_batch.metrics["rollout_rebalance_skipped_rank_aligned_vllm"] = 1.0
        if self.progress_fn:
            self.progress_fn(
                f"step {step} rollout selected records={len(rollout_batch.records)}/{len(rollout_batch.all_records)} "
                f"groups={int(rollout_batch.metrics.get('rollout_selected_groups', 0))}/"
                f"{int(rollout_batch.metrics.get('rollout_source_groups', 0))} "
                f"zero_variance_skipped={int(rollout_batch.metrics.get('rollout_skipped_zero_variance_groups', 0))}"
            )
        if not rollout_batch.records:
            if not rollout_batch.all_records:
                ctx["rollout_batch"] = rollout_batch
                ctx["rollout_all_records"] = []
                ctx["rollout_metrics"] = dict(rollout_batch.metrics)
                ctx["rollout_signature_stats"] = dict(rollout_batch.signature_stats)
                ctx["rollout_failure_reasons"] = dict(rollout_batch.failure_reasons)
                return None, ctx
            fallback_token_id = self.tokenizer.eos_token_id
            if fallback_token_id is None:
                fallback_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            keepalive_record = make_no_signal_keepalive_record(
                rollout_batch.all_records[0],
                fallback_token_id=int(fallback_token_id),
            )
            input_ids = prepare_rollout_training_context(
                model=model,
                tokenizer=self.tokenizer,
                records=[keepalive_record],
                device=device,
                grpo_config=self.grpo_config,
                ctx=ctx,
                rollout_batch=None,
                progress_fn=self.progress_fn,
                step=step,
                max_policy_age_updates=self.max_policy_age_updates,
                max_rollout_reuse=self.max_rollout_reuse,
                old_logprob_micro_batch_size=self.old_logprob_micro_batch_size,
            )
            ctx["rollout_batch"] = rollout_batch
            ctx["rollout_all_records"] = list(rollout_batch.all_records)
            ctx["rollout_metrics"] = {
                **dict(ctx.get("rollout_metrics") or {}),
                **dict(rollout_batch.metrics),
                "advantage_abs_sum": 0.0,
                "rollout_no_signal_keepalive_records": 1.0,
            }
            ctx["rollout_signature_stats"] = dict(rollout_batch.signature_stats)
            ctx["rollout_failure_reasons"] = dict(rollout_batch.failure_reasons)
            if self.loss_micro_batch_size is not None:
                ctx["loss_micro_batch_size"] = int(self.loss_micro_batch_size)
            if self.on_rollouts:
                self.on_rollouts(rollout_batch.all_records, ctx["rollout_metrics"], ctx)
            return input_ids, ctx

        input_ids = prepare_rollout_training_context(
            model=model,
            tokenizer=self.tokenizer,
            records=rollout_batch.records,
            device=device,
            grpo_config=self.grpo_config,
            ctx=ctx,
            rollout_batch=rollout_batch,
            progress_fn=self.progress_fn,
            step=step,
            max_policy_age_updates=self.max_policy_age_updates,
            max_rollout_reuse=self.max_rollout_reuse,
            old_logprob_micro_batch_size=self.old_logprob_micro_batch_size,
        )
        if self.loss_micro_batch_size is not None:
            ctx["loss_micro_batch_size"] = int(self.loss_micro_batch_size)
        if self.on_rollouts:
            self.on_rollouts(rollout_batch.all_records, ctx["rollout_metrics"], ctx)

        return input_ids, ctx

    def config(self) -> dict[str, Any]:
        return {
            "rollouts_per_prompt": self.rollouts_per_prompt,
            "max_new_tokens": (
                self.max_new_tokens if isinstance(self.max_new_tokens, int) else "task_callable"
            ),
            "rollout_do_sample": self.rollout_do_sample,
            "rollout_use_cache": self.rollout_use_cache,
            "rollout_disable_compile": self.rollout_disable_compile,
            "rollout_backend": self.rollout_backend,
            "prompt_batch_size": self.prompt_batch_size,
            "rollout_token_log_every": self.rollout_token_log_every,
            "skip_zero_variance_groups": self.skip_zero_variance_groups,
            "min_group_reward_std": self.min_group_reward_std,
            "max_policy_age_updates": self.max_policy_age_updates,
            "max_rollout_reuse": self.max_rollout_reuse,
            "rollout_scheduler": self.rollout_scheduler.__class__.__name__ if self.rollout_scheduler is not None else None,
            "environment_replay": self.environment_replay_fn.__name__ if self.environment_replay_fn is not None else None,
            "functional_credit": (
                self.functional_credit_config.__class__.__name__
                if self.functional_credit_config is not None
                else None
            ),
        }
