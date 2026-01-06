"""
PropOp configuration dataclass.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class PropOpConfig:
    """Configuration for PropOp local learning."""

    lr: float = 0.01
    beta: float = 0.9  # Eligibility trace momentum

    # Feature flags
    use_cofire: bool = True
    use_lateral: bool = True
    use_theta: bool = True
    cofire_forward: bool = True  # Use cofire to boost activations in forward
    activity_gate: bool = True  # Gate credit by activity
    normalize_updates: bool = True  # Normalize input in weight updates
    norm_band: Optional[float] = None  # Weight norm banding (None = disabled)

    # Hyperparameters
    cofire_tau: float = 0.33  # EMA decay for cofire matrix
    lateral_tau: float = 0.33  # EMA decay for lateral inhibition
    lateral_decay: float = 0.9995  # Per-batch decay for lateral
    lateral_strength: float = 1.0  # Scaling factor for lateral push
    lam: float = 1000.0  # Lambda for lateral weight updates
    theta_lr: float = 0.001  # Learning rate for theta homeostasis
    theta_clip: float = 3.0  # Max absolute value for theta
    target_fire: float = 0.20  # Target firing rate for homeostasis
    firing_rate_tau: float = 0.9  # EMA decay for firing rate tracking
