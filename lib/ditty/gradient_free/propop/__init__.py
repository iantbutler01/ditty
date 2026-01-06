"""
PropOp: Local credit assignment with eligibility traces.

A gradient-free learning paradigm that uses local learning rules
instead of backpropagation.
"""
from .config import PropOpConfig
from .state import LayerState
from .wrapper import PropOpWrapper
from .trainer import PropOpTrainer, PropOpTrainerState

__all__ = ["PropOpConfig", "LayerState", "PropOpWrapper", "PropOpTrainer", "PropOpTrainerState"]
