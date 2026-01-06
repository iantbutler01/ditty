"""
Gradient-free learning methods.

This module contains non-backprop training paradigms:
- propop: Local credit assignment with eligibility traces
- (future) evo: Evolutionary strategies
- (future) neuro: Neuromorphic learning rules
"""
from . import propop

__all__ = ["propop"]
