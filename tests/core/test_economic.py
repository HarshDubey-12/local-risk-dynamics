"""Portfolio statistics and economic-value tests (lrd.core.evaluation.backtesting)."""

import numpy as np
import pandas as pd
import pytest

from lrd.core.evaluation.backtesting import (
    certainty_equivalent,
    economic_value,
    sharpe_ratio,
    timing_weights,
    trailing_variance,
    volatility,
)


def test_monthly_annualisation_uses_sqrt_12():
    returns = np.array([0.01, 0.03, -0.02, 0.02, 0.0, 0.01])
    assert sharpe_ratio(returns) == pytest.approx(sharpe_ratio(returns, annualize=False) * np.sqrt(12))
    assert volatility(returns) == pytest.approx(np.std(returns, ddof=1) * np.sqrt(12))


def test_timing_weights_clip_and_scale():
    w = timing_weights(np.array([0.01, -0.01, 1.0]), np.array([0.0025, 0.0025, 0.0025]), risk_aversion=4.0)
    np.testing.assert_allclose(w, [1.0, 0.0, 1.5])


def test_timing_weights_reject_bad_variance():
    with pytest.raises(ValueError):
        timing_weights(np.array([0.01]), np.array([np.nan]))
    with pytest.raises(ValueError):
        timing_weights(np.array([0.01]), np.array([0.0025]), risk_aversion=0)


def test_certainty_equivalent():
    r = np.array([0.01, 0.02, 0.03])
    expected = 12 * (r.mean() - 0.5 * 3 * r.var(ddof=1))
    assert certainty_equivalent(r, 3.0) == pytest.approx(expected)


def test_identical_forecasts_have_no_gain():
    rng = np.random.default_rng(0)
    realized = rng.normal(0.005, 0.04, 200)
    forecast = np.full(200, 0.005)
    result = economic_value(forecast, forecast, realized, np.full(200, 0.0016))
    assert result["cer_gain_%"] == pytest.approx(0.0)
    assert result["turnover"] == 0.0


def test_informative_forecast_has_positive_gain():
    rng = np.random.default_rng(1)
    signal = rng.normal(0.005, 0.02, 600)
    realized = signal + rng.normal(0, 0.03, 600)
    result = economic_value(signal, np.full(600, 0.005), realized, np.full(600, 0.0016))
    assert result["cer_gain_%"] > 0
    assert result["sharpe"] > result["sharpe_benchmark"]


def test_trailing_variance_uses_only_past():
    series = pd.Series(np.arange(100, dtype=float))
    var = trailing_variance(series, window=60, min_periods=24)
    assert var.iloc[:23].isna().all()
    assert var.iloc[59] == pytest.approx(series.iloc[:60].var())
