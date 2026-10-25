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


def clark_west(errors: np.ndarray, benchmark_errors: np.ndarray) -> tuple[float, float]:
    """Clark-West (2007) test of equal accuracy against a nested benchmark.

    Designed for comparing a model with a benchmark it nests, such as the
    historical-mean forecast. Diebold-Mariano is undersized in that case;
    Clark-West adjusts the loss differential for the noise the larger model
    adds by estimating parameters whose true values may be zero:

        f_t = e_bench² − (e_model² − (ŷ_bench − ŷ_model)²)

    Since ŷ_bench − ŷ_model = e_model − e_bench, only the errors are needed.

    Returns ``(t_stat, p_value)``; the p-value is one-sided for
    H1: the model is more accurate than the benchmark.
    """
    errors = np.asarray(errors, dtype=float)
    benchmark_errors = np.asarray(benchmark_errors, dtype=float)
    if errors.shape != benchmark_errors.shape:
        raise ValueError("errors and benchmark_errors must have the same shape.")
    f = benchmark_errors**2 - (errors**2 - (errors - benchmark_errors) ** 2)
    se = f.std(ddof=1) / np.sqrt(len(f))
    if se == 0:
        return 0.0, 0.5
    t = float(f.mean() / se)
    return t, 0.5 * math.erfc(t / math.sqrt(2))  # P(Z > t)


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
