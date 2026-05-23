"""Small optimizer composition helpers."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Optimizer


class ChainedOptimizer(Optimizer):
    """Expose several PyTorch optimizers through one Optimizer interface."""

    def __init__(self, optimizers: Iterable[Optimizer]):
        self.optimizers = list(optimizers)
        if not self.optimizers:
            raise ValueError("ChainedOptimizer requires at least one optimizer")
        self._group_counts = [len(optimizer.param_groups) for optimizer in self.optimizers]

        param_groups = [
            dict(group)
            for optimizer in self.optimizers
            for group in optimizer.param_groups
        ]
        super().__init__(param_groups, {})
        self._sync_children_from_self()

    def _sync_children_from_self(self) -> None:
        offset = 0
        for optimizer, group_count in zip(self.optimizers, self._group_counts):
            optimizer.param_groups = self.param_groups[offset:offset + group_count]
            optimizer.state = self.state
            offset += group_count

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        self._sync_children_from_self()
        return super().state_dict()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._sync_children_from_self()
