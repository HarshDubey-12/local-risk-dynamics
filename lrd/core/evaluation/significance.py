"""
Forecast comparison statistics

Tests and scores for comparing two sets of out-of-sample forecast errors,
used to judge models against each other and against the historical-mean
benchmark.
"""

from __future__ import annotations

import math

import numpy as np


def oos_r2(errors: np.ndarray, benchmark_errors: np.ndarray) -> float:
    """Out-of-sample R² relative to a benchmark: 1 - SSE(model) / SSE(benchmark).

    Positive values mean the model's squared errors are smaller than the
    benchmark's (typically the historical-mean forecast).
    """
    errors = np.asarray(errors, dtype=float)
    benchmark_errors = np.asarray(benchmark_errors, dtype=float)
    if errors.shape != benchmark_errors.shape:
        raise ValueError("errors and benchmark_errors must have the same shape.")
    return float(1 - np.sum(errors**2) / np.sum(benchmark_errors**2))


def diebold_mariano(errors: np.ndarray, reference_errors: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano test for equal accuracy (squared loss, one-step forecasts).

    Returns ``(t_stat, p_value)``. A negative statistic means ``errors`` are
    smaller than ``reference_errors``; the p-value is two-sided.
    """
    errors = np.asarray(errors, dtype=float)
    reference_errors = np.asarray(reference_errors, dtype=float)
    if errors.shape != reference_errors.shape:
        raise ValueError("errors and reference_errors must have the same shape.")
    d = errors**2 - reference_errors**2
    se = d.std(ddof=1) / np.sqrt(len(d))
    if se == 0:
        return 0.0, 1.0
    t = float(d.mean() / se)
    return t, math.erfc(abs(t) / math.sqrt(2))
