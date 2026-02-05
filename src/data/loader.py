"""
Data loader utilities

This module provides dataset loading functions for the research project.

TODO:
 - Implement `load_fama_french()` to read `data/raw/F-F_Research_Data_5_Factors_2x3.csv`.
 - Add type hints and tests.
"""

# placeholder: implementations should live here
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np 
import pandas as pd

FACTOR_COLUMNS = ["Mkt-RF","SMB","HML","RMW","CMA"]
ALL_RETURN_COLUMNS = FACTOR_COLUMNS + ["RF"]

def load_fama_french(
        csv_path: str | Path,
) -> Tuple[np.ndarray,np.ndarray,pd.DatetimeIndex]:
     """
    Load and preprocess the Fama–French Five-Factor dataset.

    Parameters
    ----------
    csv_path : str or Path
        Path to the raw Fama–French CSV file.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 5)
        Feature matrix containing factor returns at time t.

    y : np.ndarray of shape (n_samples,)
        Target vector containing next-period market excess return (t+1).

    dates : pd.DatetimeIndex
        Dates aligned with X and y after preprocessing.

    Notes
    -----
    - Returns are converted from percentages to decimals.
    - Target is shifted by one period to ensure true forecasting.
    - Final row is dropped due to NaN after shifting.
    """
     
     csv_path = Path(csv_path)

     if not csv_path.exists():
          raise FileNotFoundError(f"Fama-French file not found: {csv_path}")
     
     # Load CSV (skip metadata header rows)
     df = pd.read_csv(csv_path, skiprows = 3)

     # Rename first colummn to 'Date' if unnamed 
     if "Date" not in df.columns:
          df = df.rename(columns = {df.columns[0]: "Date"})

     # Pasre dates(YYYYMM format)
     df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%y%m")

     # Convert percentage returns -> decimals 
     df[ALL_RETURN_COLUMNS] = df[ALL_RETURN_COLUMNS]/100.0

     # Feature matrix at time t
     X = df[FACTOR_COLUMNS].to_numpy(dtype = float)

     # Target: next-period market excess return
     y = df["Mkt-RF"].shift(-1).to_numpy(dtype = float)

     # Remove last row (NaN after shift)
     valid_mask = ~np.isnan(y)
     x = X[valid_mask]
     y = y[valid_mask]
     dates = df.loc[valid_mask,"Date"]

     return X,y,pd.DatetimeIndex(dates)
