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

    # Frozen state for batch consistency (snapshot at begin_batch)
    cofire_pre: Optional[torch.Tensor] = None
    lateral_pre: Optional[torch.Tensor] = None

    # MaxPool indices for proper credit routing
    pool_indices: Optional[torch.Tensor] = None

    # Per-layer hyperparameters (computed from layer dimensions)
    lateral_strength: float = 1.0
