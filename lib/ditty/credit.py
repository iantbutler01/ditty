from __future__ import annotations

import json
import re
import string
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class FunctionalCreditConfig:
    min_cluster_size: int = 1
    normalize: bool = True
    text_fallback_chars: int = 240


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = value.strip(string.whitespace + string.punctuation)
    return re.sub(r"\s+", " ", value)


def strip_code_fence(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def extract_json_value(value: str) -> Any:
    value = strip_code_fence(value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    candidates: list[str] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = value.find(opener)
        end = value.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidates.append(value[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _parse_parameter_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    if re.fullmatch(r"[-+]?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    if re.fullmatch(r"[-+]?\d+\.\d+", value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def qwen_tool_calls_from_completion(completion: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for tool_call in re.finditer(
        r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
        completion,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        name = tool_call.group(1).strip()
        body = tool_call.group(2)
        arguments: dict[str, Any] = {}
        for parameter in re.finditer(
            r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            arguments[parameter.group(1).strip()] = _parse_parameter_value(parameter.group(2))
        actions.append({"name": name, "arguments": arguments})
    return actions


def canonical_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonical_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonical_json(item) for item in value]
    if isinstance(value, str):
        return normalize_text(value)
    return value


def actions_from_completion(completion: str) -> list[dict[str, Any]]:
    qwen_actions = qwen_tool_calls_from_completion(completion)
    if qwen_actions:
        return qwen_actions
    parsed = extract_json_value(completion)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("actions"), list):
            return [item for item in parsed["actions"] if isinstance(item, dict)]
        if any(key in parsed for key in ("name", "tool", "action")):
            return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def action_arguments(action: dict[str, Any]) -> dict[str, Any]:
    for key in ("arguments", "args", "tool_input", "parameters", "input"):
        value = action.get(key)
        if isinstance(value, dict):
            return value
    return {
        key: value
        for key, value in action.items()
        if key not in {"name", "tool", "action", "arguments", "args", "tool_input", "parameters", "input"}
    }


def environment_actions_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    transcript = row.get("environment_transcript")
    if not isinstance(transcript, list):
        return []
    actions: list[dict[str, Any]] = []
    for turn in transcript:
        if not isinstance(turn, dict):
            continue
        action = turn.get("action")
        if isinstance(action, dict):
            actions.append(action)
    return actions


def rollout_completion_text(row: dict[str, Any]) -> str:
    return str(row.get("completion_text") or row.get("completion") or "")


def rollout_completion_len(row: dict[str, Any]) -> int:
    ids = row.get("completion_ids")
    if isinstance(ids, list):
        return len(ids)
    return max(len(rollout_completion_text(row).split()), 1)


def rollout_reward(row: dict[str, Any]) -> float:
    try:
        return float(row.get("reward", 0.0))
    except (TypeError, ValueError):
        return 0.0


def structured_action_functional_key(row: dict[str, Any], *, text_fallback_chars: int = 240) -> str:
    actions = environment_actions_from_row(row) or actions_from_completion(rollout_completion_text(row))
    if actions:
        canonical_actions = []
        for action in actions:
            name = normalize_text(str(action.get("name") or action.get("tool") or action.get("action") or ""))
            canonical_actions.append({"name": name, "arguments": canonical_json(action_arguments(action))})
        return json.dumps(canonical_actions, sort_keys=True, separators=(",", ":"))
    return normalize_text(rollout_completion_text(row))[:text_fallback_chars]


def canonical_action_key(action: dict[str, Any]) -> str:
    name = normalize_text(str(action.get("name") or action.get("tool") or action.get("action") or ""))
    return json.dumps(
        {"name": name, "arguments": canonical_json(action_arguments(action))},
        sort_keys=True,
        separators=(",", ":"),
    )


def group_rollout_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("group_id") or row.get("sample_id") or "")].append(row)
    return grouped


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = sum(values) / len(values)
    return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def functional_turns_for_row(
    row: dict[str, Any],
    *,
    completion_len: int,
    functional_key_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    raw_turns = row.get("functional_turns")
    turns: list[dict[str, Any]] = []
    if isinstance(raw_turns, list):
        for ordinal, raw_turn in enumerate(raw_turns):
            if not isinstance(raw_turn, dict):
                continue
            action = raw_turn.get("action")
            key = canonical_action_key(action) if isinstance(action, dict) else None
            key = key or raw_turn.get("functional_key") or raw_turn.get("key")
            if key is None:
                key = raw_turn.get("effect") or raw_turn.get("action")
            if key is None:
                continue
            start = max(_as_int(raw_turn.get("token_start"), 0), 0)
            end = min(_as_int(raw_turn.get("token_end"), completion_len), completion_len)
            if end <= start:
                continue
            turn = {
                "turn_index": _as_int(raw_turn.get("turn_index"), ordinal),
                "functional_key": str(key),
                "token_start": start,
                "token_end": end,
            }
            if isinstance(action, dict):
                turn["action"] = action
            turns.append(turn)
    if turns:
        return turns
    environment_actions = environment_actions_from_row(row)
    if environment_actions:
        return [
            {
                "turn_index": ordinal,
                "functional_key": canonical_action_key(action),
                "token_start": 0,
                "token_end": completion_len,
                "action": action,
                "source": "environment_transcript",
            }
            for ordinal, action in enumerate(environment_actions)
            if completion_len > 0
        ]
    return [
        {
            "turn_index": 0,
            "functional_key": functional_key_fn(row),
            "token_start": 0,
            "token_end": completion_len,
        }
    ]


def assign_functional_token_advantages(
    rows: list[dict[str, Any]],
    *,
    config: FunctionalCreditConfig | None = None,
    functional_key_fn: Callable[[dict[str, Any]], str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign FICA-style token advantages from within-group functional clusters.

    This implements the Tier-1 structured-action case from the FICA note and a
    weak normalized-text fallback. It is intentionally a rollout post-processor:
    rows go in with terminal rewards, rows come out with `token_advantages`.
    """
    cfg = config or FunctionalCreditConfig()
    key_fn = functional_key_fn or (
        lambda row: structured_action_functional_key(row, text_fallback_chars=cfg.text_fallback_chars)
    )
    output: list[dict[str, Any]] = []
    active_groups = 0
    active_clusters = 0
    active_turns = 0
    grouped = group_rollout_rows(rows)
    for group_id, group in grouped.items():
        row_turns: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for row in group:
            row_turns.append(
                (
                    row,
                    functional_turns_for_row(
                        row,
                        completion_len=rollout_completion_len(row),
                        functional_key_fn=key_fn,
                    ),
                )
            )

        turn_rows: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for row, turns in row_turns:
            for turn in turns:
                turn_rows[int(turn["turn_index"])].append((row, turn))

        turn_cluster_advantages: dict[tuple[int, str], float] = {}
        turn_cluster_sizes: dict[tuple[int, str], int] = {}
        turn_baselines: dict[int, float] = {}
        turn_stds: dict[int, float] = {}
        group_has_signal = False
        for turn_index, entries in turn_rows.items():
            rewards = [rollout_reward(row) for row, _turn in entries]
            baseline = sum(rewards) / max(len(rewards), 1)
            reward_std = population_std(rewards)
            scale = reward_std if cfg.normalize else 1.0
            if scale <= 1e-8:
                scale = 1.0
            turn_baselines[turn_index] = baseline
            turn_stds[turn_index] = reward_std

            clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row, turn in entries:
                clusters[str(turn["functional_key"])].append(row)

            turn_has_signal = False
            for key, cluster in clusters.items():
                cluster_id = (turn_index, key)
                turn_cluster_sizes[cluster_id] = len(cluster)
                if len(cluster) < cfg.min_cluster_size:
                    turn_cluster_advantages[cluster_id] = 0.0
                    continue
                cluster_reward = sum(rollout_reward(row) for row in cluster) / len(cluster)
                advantage = (cluster_reward - baseline) / scale
                turn_cluster_advantages[cluster_id] = advantage
                if abs(advantage) > 1e-8:
                    active_clusters += 1
                    turn_has_signal = True
                    group_has_signal = True
            if turn_has_signal:
                active_turns += 1

        if group_has_signal:
            active_groups += 1

        for row, turns in row_turns:
            updated = dict(row)
            token_advantages = [0.0] * rollout_completion_len(row)
            assigned: list[float] = []
            assigned_cluster_sizes: list[int] = []
            assigned_baselines: list[float] = []
            assigned_stds: list[float] = []
            for turn in turns:
                turn_index = int(turn["turn_index"])
                cluster_id = (turn_index, str(turn["functional_key"]))
                advantage = turn_cluster_advantages.get(cluster_id, 0.0)
                start = int(turn["token_start"])
                end = int(turn["token_end"])
                for idx in range(start, end):
                    token_advantages[idx] = advantage
                assigned.append(advantage)
                assigned_cluster_sizes.append(turn_cluster_sizes.get(cluster_id, 0))
                assigned_baselines.append(turn_baselines.get(turn_index, 0.0))
                assigned_stds.append(turn_stds.get(turn_index, 0.0))
            updated["functional_turns"] = [dict(turn) for turn in turns]
            updated["token_advantages"] = token_advantages
            metrics = dict(updated.get("reward_metrics") or {})
            advantage_mean = sum(assigned) / len(assigned) if assigned else 0.0
            metrics.update(
                {
                    "fica_advantage": advantage_mean,
                    "fica_cluster_size": float(max(assigned_cluster_sizes, default=0)),
                    "fica_group_baseline": sum(assigned_baselines) / len(assigned_baselines) if assigned_baselines else 0.0,
                    "fica_group_reward_std": sum(assigned_stds) / len(assigned_stds) if assigned_stds else 0.0,
                    "fica_turns": float(len(turns)),
                }
            )
            updated["reward_metrics"] = metrics
            output.append(updated)

    summary = {
        "input_records": len(rows),
        "output_records": len(output),
        "groups": len(grouped),
        "active_groups": active_groups,
        "active_turns": active_turns,
        "active_clusters": active_clusters,
        "min_cluster_size": cfg.min_cluster_size,
        "normalize": cfg.normalize,
        "text_fallback_chars": cfg.text_fallback_chars,
    }
    return output, summary
