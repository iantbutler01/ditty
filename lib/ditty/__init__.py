from .base import DittyBase
from .contract import (
    Contract,
    TensorSpec,
    ContractViolation,
    ContractParseError,
    parse_contract,
    validate_pipeline_chain,
    format_pipeline_contracts,
)
from . import diffusion
from .loss import LossCalculator, LossOutput, MSELoss, L1Loss, CrossEntropyLoss, CompositeLoss
from .loss import GRPOLoss
from .processors import PreProcessor, PostProcessor, Context
from .checkpoint import CheckpointManager, Checkpoint
from .metrics_logger import MetricsLogger
from .grpo import (
    GRPOConfig,
    RolloutGroup,
    RolloutSample,
    build_selective_logit_positions,
    compute_group_advantages,
    model_supports_selective_logits,
    prepare_grpo_forward_kwargs,
)
from .grpo_rollouts import (
    GRPORolloutPreProcessor,
    PolicyVersion,
    RolloutBatch,
    RolloutScheduler,
    RolloutSchedulerConfig,
    RolloutRecord,
    apply_functional_credit_to_records,
    collate_rollouts,
    coerce_rollout_record,
    compute_old_logprobs,
    flatten_rollout_records,
    generate_rollouts,
    prepare_rollout_training_context,
    reward_summary,
    rollout_record_from_dict,
    rollout_record_to_dict,
)
from .credit import (
    FunctionalCreditConfig,
    assign_functional_token_advantages,
    structured_action_functional_key,
)
from .environments import (
    DeterministicToolEnvironment,
    Environment,
    EnvironmentStepResult,
    expected_actions,
    expects_no_tool,
    replay_tool_environment,
)
from . import optimizers


def _missing_optional_export(name, error):
    class MissingOptionalDependency:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise ModuleNotFoundError(
                f"ditty.{name} requires optional dependencies that are not installed: {error}"
            ) from error

    MissingOptionalDependency.__name__ = name
    return MissingOptionalDependency


try:
    from .pipeline import Pipeline
    from .trainer import Trainer, TrainerState
    from .data import Data
except ModuleNotFoundError as exc:
    Pipeline = _missing_optional_export("Pipeline", exc)
    Trainer = _missing_optional_export("Trainer", exc)
    TrainerState = _missing_optional_export("TrainerState", exc)
    Data = _missing_optional_export("Data", exc)

try:
    from .model_factory import (
        CausalLMBackboneTransform,
        ModelConfig,
        ModelFactory,
        TokenizerFactory,
        FSDPConfig,
        QuantConfig,
        PeftConfig,
        ModelTransform,
    )
except ModuleNotFoundError as exc:
    CausalLMBackboneTransform = _missing_optional_export("CausalLMBackboneTransform", exc)
    ModelConfig = _missing_optional_export("ModelConfig", exc)
    ModelFactory = _missing_optional_export("ModelFactory", exc)
    TokenizerFactory = _missing_optional_export("TokenizerFactory", exc)
    FSDPConfig = _missing_optional_export("FSDPConfig", exc)
    QuantConfig = _missing_optional_export("QuantConfig", exc)
    PeftConfig = _missing_optional_export("PeftConfig", exc)
    ModelTransform = _missing_optional_export("ModelTransform", exc)

try:
    from .example import print_pipeline
except ModuleNotFoundError as exc:
    def print_pipeline(*args, **kwargs):
        del args, kwargs
        raise ModuleNotFoundError(
            f"ditty.print_pipeline requires optional dependencies that are not installed: {exc}"
        ) from exc

try:
    from .ray_vllm_engine import RayVllmActor, RayVllmRolloutEngine
except ModuleNotFoundError as exc:
    RayVllmActor = _missing_optional_export("RayVllmActor", exc)
    RayVllmRolloutEngine = _missing_optional_export("RayVllmRolloutEngine", exc)

__all__ = [
    "DittyBase",
    "Contract",
    "TensorSpec",
    "ContractViolation",
    "ContractParseError",
    "parse_contract",
    "validate_pipeline_chain",
    "format_pipeline_contracts",
    "Pipeline",
    "Trainer",
    "TrainerState",
    "Data",
    "LossCalculator",
    "LossOutput",
    "MSELoss",
    "L1Loss",
    "CrossEntropyLoss",
    "GRPOLoss",
    "CompositeLoss",
    "PreProcessor",
    "PostProcessor",
    "Context",
    "ModelFactory",
    "ModelConfig",
    "CausalLMBackboneTransform",
    "TokenizerFactory",
    "FSDPConfig",
    "QuantConfig",
    "PeftConfig",
    "ModelTransform",
    "CheckpointManager",
    "Checkpoint",
    "MetricsLogger",
    "print_pipeline",
    "GRPOConfig",
    "RolloutGroup",
    "RolloutSample",
    "build_selective_logit_positions",
    "compute_group_advantages",
    "model_supports_selective_logits",
    "prepare_grpo_forward_kwargs",
    "GRPORolloutPreProcessor",
    "PolicyVersion",
    "RolloutBatch",
    "RolloutScheduler",
    "RolloutSchedulerConfig",
    "RolloutRecord",
    "apply_functional_credit_to_records",
    "collate_rollouts",
    "coerce_rollout_record",
    "compute_old_logprobs",
    "flatten_rollout_records",
    "generate_rollouts",
    "prepare_rollout_training_context",
    "reward_summary",
    "rollout_record_from_dict",
    "rollout_record_to_dict",
    "FunctionalCreditConfig",
    "assign_functional_token_advantages",
    "structured_action_functional_key",
    "DeterministicToolEnvironment",
    "Environment",
    "EnvironmentStepResult",
    "expected_actions",
    "expects_no_tool",
    "replay_tool_environment",
    "RayVllmActor",
    "RayVllmRolloutEngine",
    "optimizers",
]
