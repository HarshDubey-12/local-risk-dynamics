"""
Fama-French five-factor data loader.

Reads the monthly table of the Kenneth French Data Library file
"F-F_Research_Data_5_Factors_2x3.csv" and builds a forecasting dataset:
features are the five factor returns at month t, the target is a quantity
of the market excess return at month t+1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
ALL_RETURN_COLUMNS = FACTOR_COLUMNS + ["RF"]

# Forecast targets, all built from the market excess return at t+1.
TARGETS = {
    "return": "next-month market excess return",
    "abs_return": "absolute next-month market excess return (a volatility proxy)",
}


def load_fama_french(
    csv_path: str | Path,
    target: str = "return",
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Load and preprocess the Fama-French five-factor dataset.

    Parameters
    ----------
    csv_path : str or Path
        Path to the raw Fama-French CSV file.
    target : {"return", "abs_return"}, default="return"
        ``"return"``: next-month market excess return.
        ``"abs_return"``: its absolute value, a simple proxy for next-month
        market volatility (risk forecasting).

    Returns
    -------
    X : np.ndarray of shape (n_samples, 5)
        Factor returns at month t (decimals).
    y : np.ndarray of shape (n_samples,)
        Target at month t+1.
    dates : pd.DatetimeIndex
        Month t of each row.

    Notes
    -----
    - Returns are converted from percentages to decimals.
    - The target is shifted by one month, so every row is a true forecast.
    - The final month is dropped because it has no next-month target.
    - Files with duplicated or out-of-order months are rejected.
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown target '{target}'. Expected one of: {sorted(TARGETS)}")

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fama-French file not found: {csv_path}")

    # The header is on line 4 (after three note lines and a blank line);
    # skipinitialspace trims the padding in numeric columns.
    df = pd.read_csv(csv_path, skiprows=4, skipinitialspace=True)
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})

    # Keep only the monthly table (YYYYMM); this drops the annual section and notes.
    df = df[df["Date"].astype(str).str.match(r"^\d{6}$")]
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m")

    # A duplicated or out-of-order month would silently corrupt the t -> t+1
    # target alignment, so refuse to load such a file.
    if df["Date"].duplicated().any() or not df["Date"].is_monotonic_increasing:
        raise ValueError(
            f"{csv_path} has duplicated or out-of-order months; "
            "run lrd.core.data.validation.validate_fama_french for details."
        )

    # Percentage returns -> decimals
    df[ALL_RETURN_COLUMNS] = df[ALL_RETURN_COLUMNS].apply(pd.to_numeric, errors="coerce") / 100.0

    X = df[FACTOR_COLUMNS].to_numpy(dtype=float)
    next_month = df["Mkt-RF"].shift(-1).to_numpy(dtype=float)
    y = np.abs(next_month) if target == "abs_return" else next_month

    valid = ~np.isnan(y)
    return X[valid], y[valid], pd.DatetimeIndex(df.loc[valid, "Date"])
