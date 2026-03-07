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

