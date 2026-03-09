"""
Evaluation metrics

Standalone implementations of common regression and backtesting
scores. These are written in pure NumPy so they can be imported from
anywhere in the project and tested independently of the evaluation
pipeline.

Each function accepts ``y_true`` and ``y_pred`` arrays (and optionally
``y_std`` for calibration metrics) and returns a scalar float.
"""

import numpy as np
from typing import Optional


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error.

    Expressed as a fraction (not percent); division-by-zero yields ``np.nan``
    for the offending elements.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.nanmean(np.abs((y_true - y_pred) / y_true)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Explained variance score."""
    return float(1 - np.var(y_true - y_pred) / np.var(y_true))


def median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median absolute error."""
    return float(np.median(np.abs(y_true - y_pred)))


def coverage_95(
    y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray
) -> float:
    """Fraction of observations within 1.96 sigma of prediction."""
    return float(np.mean(np.abs(y_true - y_pred) <= 1.96 * y_std))


def avg_std(y_std: np.ndarray) -> float:
    """Mean of per-sample standard deviations."""
    return float(np.mean(y_std))


def interval_width_95(y_std: np.ndarray) -> float:
    """Average width of a nominal 95% predictive interval."""
    return float(np.mean(2 * 1.96 * y_std))


def gaussian_nll(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> float:
    """Gaussian negative log-likelihood using model mean and std."""
    sigma = np.maximum(np.asarray(y_std, dtype=float), 1e-8)
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    nll = 0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * (residual**2) / (sigma**2)
    return float(np.mean(nll))


def interval_score_95(
    y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, alpha: float = 0.05
) -> float:
    """Proper interval score for central (1-alpha) intervals; lower is better."""
    z = 1.96
    lower = y_pred - z * y_std
    upper = y_pred + z * y_std
    width = upper - lower
    below_penalty = (2.0 / alpha) * np.maximum(lower - y_true, 0.0)
    above_penalty = (2.0 / alpha) * np.maximum(y_true - upper, 0.0)
    return float(np.mean(width + below_penalty + above_penalty))

