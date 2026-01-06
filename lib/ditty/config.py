"""
Configuration dataclasses for training paradigms.
"""
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class BackpropConfig:
    """Configuration for standard backprop training."""

    lr: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    grad_accum: int = 1
    fp16: bool = True
    use_bfloat16: bool = False
    use_8bit_optim: bool = False
    optim_backend: str = "torchao"  # "torch", "bnb", or "torchao"
    gradient_checkpointing: bool = True
    optimizer: Optional[torch.optim.Optimizer] = None  # User-provided optimizer
