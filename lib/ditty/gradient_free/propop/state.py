"""
Per-layer learning state for PropOp.
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LayerState:
    """Learning state for a single layer. No forward logic - just state."""

    module: nn.Module
    device: torch.device

    # Cached during forward (set by hooks)
    input_cache: Optional[torch.Tensor] = None
    output_cache: Optional[torch.Tensor] = None
    credit: Optional[torch.Tensor] = None

    # Learning state (initialized for weight-bearing layers only)
    eligibility: Optional[torch.Tensor] = None
    cofire: Optional[torch.Tensor] = None
    lateral: Optional[torch.Tensor] = None
    theta: Optional[torch.Tensor] = None
    firing_rate: Optional[torch.Tensor] = None
    credit_echo: Optional[torch.Tensor] = None  # Persistent credit baseline (dopamine-like)

    # Frozen state for batch consistency (snapshot at begin_batch)
    cofire_pre: Optional[torch.Tensor] = None
    lateral_pre: Optional[torch.Tensor] = None

    # MaxPool indices for proper credit routing
    pool_indices: Optional[torch.Tensor] = None

    # Per-layer hyperparameters (computed from layer dimensions)
    lateral_strength: float = 1.0

    # Per-layer learnable echo params (adaptive credit smoothing)
    echo_tau: Optional[float] = None       # Per-layer EMA decay for echo
    echo_strength: Optional[float] = None  # Per-layer echo contribution weight
    credit_variance_ema: Optional[float] = None  # EMA of credit variance (for adaptation)

    # Nested structure for residual blocks (child layers contained within block)
    child_states: Optional[list] = None  # List[LayerState] for residual block internals

    # Reference to activation state following this weight layer (for post-activation lookup)
    post_activation_state: Optional["LayerState"] = None

    # DAG structure (populated from FX trace)
    predecessors: Optional[list] = None  # List[LayerState] - input states for credit backward
