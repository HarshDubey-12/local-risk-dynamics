"""
Sampling helpers

Subsampling and Monte Carlo sampling utilities.  Provides a thin wrapper
around NumPy's random choice so that the sampling strategy is centralized
and easy to test.  This module currently defines:

* ``subsample_indices`` – select a fixed-size random subset of indices,
  optionally weighted and/or without replacement.

Future work may add stratified, bootstrap, or kernel-based sampling helpers.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def subsample_indices(
    n: int,
    k: int,
    replace: bool = False,
    weights: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Return ``k`` random indices from ``0..n-1``.

    Parameters
    ----------
    n : int
        Population size.
    k : int
        Number of indices to draw.
    replace : bool, default=False
        Whether to sample with replacement.
    weights : array-like, optional
        Probabilities for each element. Must sum to 1 if provided.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (k,) of selected indices.
    """
    if k < 0 or k > n:
        raise ValueError("k must be between 0 and n")

    rng = np.random.default_rng(random_state)

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n,):
            raise ValueError("weights must have length n")
        total = weights.sum()
        if total <= 0:
            raise ValueError("weights must sum to positive value")
        probs = weights / total
        return rng.choice(n, size=k, replace=replace, p=probs)
    else:
        return rng.choice(n, size=k, replace=replace)

