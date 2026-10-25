"""Forecast-target options of the loader (lrd.core.data.loader)."""

import numpy as np
import pytest

from lrd.core.data.loader import TARGETS, load_fama_french

DATA_PATH = "data/raw/F-F_Research_Data_5_Factors_2x3.csv"


def test_abs_return_target_is_absolute_next_month_return():
    X, y, dates = load_fama_french(DATA_PATH)
    X_abs, y_abs, dates_abs = load_fama_french(DATA_PATH, target="abs_return")
    np.testing.assert_allclose(X_abs, X)
    assert (dates_abs == dates).all()
    np.testing.assert_allclose(y_abs, np.abs(y))
    # the target of month t is the market factor of month t+1
    np.testing.assert_allclose(y_abs[:-1], np.abs(X[1:, 0]))


def test_unknown_target():
    with pytest.raises(ValueError, match="Unknown target"):
        load_fama_french(DATA_PATH, target="volume")


def test_targets_documented():
    assert set(TARGETS) == {"return", "abs_return"}
