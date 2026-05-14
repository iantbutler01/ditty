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
from .pipeline import Pipeline
from .trainer import Trainer, TrainerState
from .data import Data
from . import diffusion
from .loss import LossCalculator, LossOutput, MSELoss, L1Loss, CrossEntropyLoss, CompositeLoss
from .loss import GRPOLoss
from .processors import PreProcessor, PostProcessor, Context
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
from .checkpoint import CheckpointManager, Checkpoint
from .metrics_logger import MetricsLogger
from .example import print_pipeline
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
from .ray_vllm_engine import RayVllmActor, RayVllmRolloutEngine
from . import optimizers

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
