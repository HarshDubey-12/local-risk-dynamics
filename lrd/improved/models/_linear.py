"""Shared linear-algebra helpers for the improved models."""

from __future__ import annotations

import numpy as np


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones; works for (n, p) and batched (k, n, p) arrays."""
    ones = np.ones(X.shape[:-1] + (1,), dtype=float)
    return np.concatenate([ones, X], axis=-1)


def ridge_solve(D: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    """Ridge coefficients with an unpenalised intercept (first column of D).

    Supports a single problem, D (n, p) / y (n,), or a batch,
    D (k, n, p) / y (k, n), returning (p,) or (k, p).
    """
    p = D.shape[-1]
    P = penalty * np.eye(p)
    P[0, 0] = 0.0
    P += 1e-10 * np.eye(p)  # keeps the system solvable when penalty == 0
    A = np.swapaxes(D, -1, -2) @ D + P
    b = np.swapaxes(D, -1, -2) @ y[..., None]
    return np.linalg.solve(A, b)[..., 0]


def check_fit_inputs(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if X.shape[0] == 0:
        raise ValueError("Cannot fit on an empty dataset.")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("X and y must not contain NaN or infinite values.")
    return X, y


def check_predict_inputs(X: np.ndarray, n_features: int | None) -> np.ndarray:
    if n_features is None:
        raise RuntimeError("Model must be fitted before prediction.")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != n_features:
        raise ValueError(f"Expected a 2D array with {n_features} features.")
    return X
