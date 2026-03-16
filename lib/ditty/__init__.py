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
    "optimizers",
]
