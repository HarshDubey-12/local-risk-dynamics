"""Shared pytest fixtures. Tests run from the project root (see pyproject.toml)."""

import pytest

from lrd.core.data.loader import load_fama_french
from lrd.core.data.preprocessing import time_series_split

DATA_PATH = "data/raw/F-F_Research_Data_5_Factors_2x3.csv"


@pytest.fixture
def sample_data():
    """Fama-French data split 80/20 chronologically: X_train, X_test, y_train, y_test."""
    X, y, dates = load_fama_french(DATA_PATH)
    X_train, X_test, y_train, y_test, _, _ = time_series_split(X, y, dates, 0.8)
    return X_train, X_test, y_train, y_test
