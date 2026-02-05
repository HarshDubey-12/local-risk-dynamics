"""
Data preprocessing helpers

Place common preprocessing routines here (scaling, feature engineering, etc.).

TODO:
 - Implement preprocessing pipeline used by notebooks.
 - Add tests and docs.
"""

# placeholder for preprocessing utilities
from __future__ import annotations

from typing import Tuple

import numpy as np 
import pandas as pd

def time_series_split(
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.DatetimeIndex,
        train_ratio: float = 0.8,
)-> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
]:
    """
    Perform a chronological train/test split for time-series data.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix ordered in time.

    y : np.ndarray of shape (n_samples,)
        Target vector aligned with X.

    dates : pd.DatetimeIndex of length n_samples
        Timestamps aligned with X and y.

    train_ratio : float, default=0.8
        Fraction of data used for training.
        Must satisfy 0 < train_ratio < 1.

    Returns
    -------
    X_train, X_test : np.ndarray
        Chronologically split feature matrices.

    y_train, y_test : np.ndarray
        Chronologically split targets.

    dates_train, dates_test : pd.DatetimeIndex
        Corresponding timestamps for each split.

    Raises
    ------
    ValueError
        If input lengths mismatch or train_ratio is invalid.

    Notes
    -----
    - No shuffling is performed.
    - Prevents look-ahead bias in financial forecasting.
    - Deterministic and reproducible.
    """

    # -----------Validation---------------
    n_samples = len(X)

    if not (len(y) == n_samples == len(dates)):
        raise ValueError("X, y, and dates must have the same length.")
    
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1(exclusive).")
    
    # -----------Split index--------------
    split_idx = int(np.floor(n_samples*train_ratio))

    if split_idx == 0 or split_idx == n_samples:
        raise ValueError("train_ratio results in empty or test split.")
    
    # -----------Chronological split-------------
    X_train, X_test = X[:split_idx],X[:split_idx:]
    y_train, y_test = y[:split_idx],y[:split_idx:]
    dates_train, dates_test = dates[:split_idx],dates[split_idx:]

    return X_train, X_test, y_train, y_test, dates_train, dates_test