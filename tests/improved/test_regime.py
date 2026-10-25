"""Regime-state MCLLR tests (lrd.improved.models.regime)."""

import numpy as np
import pytest

from lrd.core.evaluation.rolling import evaluate_walk_forward
from lrd.improved.models import MODEL_REGISTRY, RegimeMCLLR
from lrd.improved.models.regime import market_state
from lrd.improved.preprocessing import PreparedModel, Standardizer


def test_market_state_values():
    state = market_state(np.array([1.0, 3.0, 5.0, 7.0]), window=2)
    np.testing.assert_allclose(state[:, 0], [0.0, 1.0, 1.0, 1.0])  # std of the last two months
    np.testing.assert_allclose(state[:, 1], [1.0, 4.0, 8.0, 12.0])  # sum of the last two months


def test_market_state_has_no_look_ahead():
    rng = np.random.default_rng(0)
    series = rng.normal(size=50)
    changed = series.copy()
    changed[30:] += 10.0  # alter the future
    np.testing.assert_allclose(market_state(series, 12)[:30], market_state(changed, 12)[:30])


def test_market_state_uses_history():
    history = np.array([1.0, 2.0, 3.0])
    with_history = market_state(np.array([4.0]), window=4, history=history)
    np.testing.assert_allclose(with_history, [[np.std([1, 2, 3, 4]), 10.0]])


def test_registered_and_named():
    assert MODEL_REGISTRY["mcllr_regime"] is RegimeMCLLR
    model = RegimeMCLLR(state_window=6)
    assert model.model_name == "mcllr_regime"
    with pytest.raises(ValueError):
        RegimeMCLLR(state_window=1)


def test_kernel_uses_state_not_factors():
    """Two queries with identical factors but different market history get different weights."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = rng.normal(size=200)
    model = RegimeMCLLR(1.0, 5, 40, random_state=0, state_window=6).fit(X, y)
    calm = np.tile([0.01, 0.0, 0.0], (6, 1))
    wild = calm.copy()
    wild[:5, 0] = [3.0, -3.0, 3.0, -3.0, 3.0]
    d_calm = model._query_distances(calm)[-1]
    d_wild = model._query_distances(wild)[-1]
    assert not np.allclose(d_calm, d_wild)


def test_walk_forward_on_real_data_is_calibrated(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    folds = evaluate_walk_forward(
        lambda: PreparedModel([Standardizer()], RegimeMCLLR(1.0, 15, 120, alpha=10.0, random_state=0)),
        X, y, train_window=240, test_window=120,
    )
    pred = np.concatenate([f.result.predictions for f in folds])
    std = np.concatenate([f.result.uncertainties for f in folds])
    truth = np.concatenate([f.result.ground_truth for f in folds])
    assert np.isfinite(pred).all()
    assert np.mean(np.abs(truth - pred) <= 1.96 * std) > 0.85
