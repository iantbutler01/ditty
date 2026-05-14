from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .credit import action_arguments, actions_from_completion, normalize_text


@dataclass
class EnvironmentStepResult:
    observation: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class Environment(Protocol):
    def reset(self, task: Any) -> str:
        ...

    def step(self, action: dict[str, Any]) -> EnvironmentStepResult:
        ...


def _action_name(action: dict[str, Any]) -> str:
    return normalize_text(str(action.get("name") or action.get("tool") or action.get("action") or ""))


def _values_match(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    return normalize_text(str(actual)) == normalize_text(str(expected))


def _action_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    if _action_name(actual) != _action_name(expected):
        return False
    actual_args = action_arguments(actual)
    expected_args = action_arguments(expected)
    return all(key in actual_args and _values_match(actual_args[key], value) for key, value in expected_args.items())


def expected_actions(task: Any) -> list[dict[str, Any]]:
    if not isinstance(task, dict):
        return []
    verifier = task.get("verifier") if isinstance(task.get("verifier"), dict) else {}
    spec = verifier.get("spec") if isinstance(verifier.get("spec"), dict) else {}
    actions = spec.get("expected_actions") or spec.get("actions") or []
    if isinstance(actions, dict):
        actions = [actions]
    return [action for action in actions if isinstance(action, dict)]


def expects_no_tool(task: Any) -> bool:
    if not isinstance(task, dict):
        return False
    verifier = task.get("verifier") if isinstance(task.get("verifier"), dict) else {}
    spec = verifier.get("spec") if isinstance(verifier.get("spec"), dict) else {}
    actions = spec.get("expected_actions", spec.get("actions"))
    return bool(spec.get("no_tool") or spec.get("expected_no_tool")) and actions == []


class DeterministicToolEnvironment:
    def __init__(self) -> None:
        self.task: Any = None
        self.expected: list[dict[str, Any]] = []
        self.cursor = 0
        self.done = False
        self.wrong_branch = False

    def reset(self, task: Any) -> str:
        self.task = task
        self.expected = expected_actions(task)
        self.cursor = 0
        self.done = False
        self.wrong_branch = False
        return "ready"

    def step(self, action: dict[str, Any]) -> EnvironmentStepResult:
        if self.done:
            return EnvironmentStepResult(
                observation="episode already complete",
                reward=0.0,
                done=True,
                info={"already_done": True, "extra_action": True},
            )
        if self.cursor >= len(self.expected):
            self.done = True
            return EnvironmentStepResult(
                observation="unexpected extra action",
                reward=0.0,
                done=True,
                info={"wrong_branch": True, "extra_action": True},
            )

        expected = self.expected[self.cursor]
        matched = _action_matches(action, expected)
        if not matched:
            self.done = True
            self.wrong_branch = True
            return EnvironmentStepResult(
                observation="wrong branch",
                reward=0.0,
                done=True,
                info={"wrong_branch": True, "expected_action": expected},
            )

        self.cursor += 1
        self.done = self.cursor >= len(self.expected)
        return EnvironmentStepResult(
            observation="accepted" if not self.done else "task complete",
            reward=1.0 if self.done else 0.0,
            done=self.done,
            info={"matched": True, "action_index": self.cursor - 1},
        )


def replay_tool_environment(task: Any, completion: str) -> dict[str, Any]:
    actions = actions_from_completion(completion)
    env = DeterministicToolEnvironment()
    initial_observation = env.reset(task)
    transcript: list[dict[str, Any]] = [{"type": "observation", "content": initial_observation}]
    expected = expected_actions(task)

    if not expected and not expects_no_tool(task):
        return {
            "transcript": transcript,
            "metrics": {
                "env_applicable": 0.0,
                "env_success": 0.0,
                "env_step_count": 0.0,
                "env_expected_steps": 0.0,
                "env_wrong_branch": 0.0,
                "env_recovery_needed": 0.0,
                "env_wasted_turns": 0.0,
                "env_stopped_early": 0.0,
                "env_repeated_tool_call": 0.0,
                "env_invalid_action_parse": 0.0,
            },
        }

    if expects_no_tool(task):
        success = not actions
        return {
            "transcript": transcript,
            "metrics": {
                "env_applicable": 1.0,
                "env_success": float(success),
                "env_step_count": 0.0,
                "env_expected_steps": 0.0,
                "env_wrong_branch": 0.0,
                "env_recovery_needed": 0.0,
                "env_wasted_turns": float(len(actions)),
                "env_stopped_early": 0.0,
                "env_invalid_action_parse": 0.0,
            },
        }

    repeated = 0
    seen: set[str] = set()
    for action in actions:
        key = f"{_action_name(action)}:{action_arguments(action)}"
        if key in seen:
            repeated += 1
        seen.add(key)
        result = env.step(action)
        transcript.append({"type": "action", "action": action})
        transcript.append(
            {
                "type": "observation",
                "content": result.observation,
                "reward": result.reward,
                "done": result.done,
                "info": result.info,
            }
        )
        if result.done:
            break

    success = env.done and not env.wrong_branch and env.cursor >= len(expected)
    stopped_early = bool(actions and not env.done)
    if not actions and expected:
        stopped_early = True
    wasted_turns = max(len(actions) - len(expected), 0)
    wrong_branch = bool(env.wrong_branch or (actions and env.cursor < min(len(actions), len(expected)) and not success))
    return {
        "transcript": transcript,
        "metrics": {
            "env_applicable": 1.0,
            "env_success": float(success),
            "env_step_count": float(min(len(actions), len(transcript) // 2)),
            "env_expected_steps": float(len(expected)),
            "env_wrong_branch": float(wrong_branch),
            "env_recovery_needed": float(wrong_branch or stopped_early),
            "env_wasted_turns": float(wasted_turns),
            "env_stopped_early": float(stopped_early),
            "env_repeated_tool_call": float(repeated),
            "env_invalid_action_parse": float(not actions and bool(expected)),
        },
    }
