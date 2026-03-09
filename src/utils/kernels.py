"""
Kernels utilities

Kernel classes and functions used by local models (e.g., Gaussian,
Epanechnikov).  This module is intentionally independent of any specific
model so that kernels can be swapped or extended without modifying the
algorithms themselves.

The original kernel implementations lived in `src/models/kernels.py`; they
have been consolidated here and the models module removed.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class Kernel(ABC):
    """Base class for kernel functions."""

    @abstractmethod
    def __call__(self, X: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """Compute weights for each row of ``X`` relative to ``x0``."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human‑readable identifier for the kernel."""
        pass


class GaussianKernel(Kernel):
    def __init__(self, bandwidth: float):
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        self.bandwidth = float(bandwidth)

    def __call__(self, X: np.ndarray, x0: np.ndarray) -> np.ndarray:
        diff = X - x0
        dist_sq = np.sum(diff ** 2, axis=1)
        return np.exp(-dist_sq / (2 * self.bandwidth ** 2))

    @property
    def name(self) -> str:
        return f"gaussian_bw{self.bandwidth}"


class EpanechnikovKernel(Kernel):
    def __init__(self, bandwidth: float):
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        self.bandwidth = float(bandwidth)

    def __call__(self, X: np.ndarray, x0: np.ndarray) -> np.ndarray:
        diff = X - x0
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        u = dist / self.bandwidth
        return np.maximum(0, 0.75 * (1 - u**2)) / self.bandwidth

    @property
    def name(self) -> str:
        return f"epanechnikov_bw{self.bandwidth}"
