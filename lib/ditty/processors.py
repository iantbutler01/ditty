"""
PreProcessor and PostProcessor abstractions for ditty trainers.

Architecture:
    dataset -> preprocessors -> model.forward -> postprocessors -> loss_calc

Contracts use terse syntax: "input:rank:dtype -> output:rank:dtype | ctx.key:rank:dtype"
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from .base import DittyBase


Context = Dict[str, Any]


class PreProcessor(DittyBase, ABC):
    @abstractmethod
    def process(self, batch: Any, ctx: Context) -> Tuple[Any, Context]:
        """
        Transform batch for model forward.

        Returns:
            (batch_transformed, ctx)
        """
        pass

    def config(self) -> Dict[str, Any]:
        return {}

    def __repr__(self):
        cfg = self.config()
        if cfg:
            params = ", ".join(f"{k}={v}" for k, v in cfg.items())
            return f"{self.name}({params})"
        return self.name


class PostProcessor(DittyBase, ABC):
    @abstractmethod
    def process(self, model_output: Tuple[Any, ...], ctx: Context) -> Tuple[Tuple[Any, ...], Context]:
        """
        Transform model output for loss calculation.

        Args:
            model_output: Tuple of tensors from model forward
            ctx: Context dict

        Returns:
            (model_output_transformed, ctx)
        """
        pass

    def config(self) -> Dict[str, Any]:
        return {}

    def __repr__(self):
        cfg = self.config()
        if cfg:
            params = ", ".join(f"{k}={v}" for k, v in cfg.items())
            return f"{self.name}({params})"
        return self.name
