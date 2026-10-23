"""Improved model tests (lrd.improved.models)."""

import numpy as np
import pytest

from lrd.improved.models import HistoricalMean, ImprovedMCLLR, RidgeRegression
from lrd.improved.models._linear import add_intercept, ridge_solve
from lrd.improved.preprocessing import PreparedModel, Standardizer


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = 0.5 + X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, 300)
    return X, y


class TestRidgeSolve:
    def test_zero_penalty_is_ols(self, linear_data):
        X, y = linear_data
        D = add_intercept(X)
        np.testing.assert_allclose(ridge_solve(D, y, 0.0), np.linalg.lstsq(D, y, rcond=None)[0], atol=1e-8)

    def test_batched_matches_single(self, linear_data):
        X, y = linear_data
        D = add_intercept(X)
        batch = ridge_solve(np.stack([D[:100], D[100:200]]), np.stack([y[:100], y[100:200]]), 3.0)
        np.testing.assert_allclose(batch[1], ridge_solve(D[100:200], y[100:200], 3.0))

    def test_intercept_not_penalised(self, linear_data):
        X, y = linear_data
        beta = ridge_solve(add_intercept(X), y + 10, 1e9)
        assert beta[0] == pytest.approx(np.mean(y + 10), rel=1e-3)
        assert np.abs(beta[1:]).max() < 1e-3


class TestHistoricalMean:
    def test_predicts_training_mean_and_std(self):
        y = np.array([1.0, 2.0, 3.0, 6.0])
        model = HistoricalMean().fit(np.zeros((4, 2)), y)
        pred, std = model.predict_with_uncertainty(np.zeros((3, 2)))
        np.testing.assert_allclose(pred, 3.0)
        np.testing.assert_allclose(std, np.std(y, ddof=1))

    def test_not_fitted(self):
        with pytest.raises(RuntimeError):
            HistoricalMean().predict(np.zeros((1, 2)))


class TestRidgeRegression:
    def test_recovers_coefficients(self, linear_data):
        X, y = linear_data
        model = RidgeRegression(0.0).fit(X, y)
        np.testing.assert_allclose(model.coef_, [0.5, 1.0, -2.0, 0.5], atol=0.03)
        assert model.sigma_ == pytest.approx(0.1, rel=0.15)

    def test_heavy_penalty_shrinks_to_mean(self, linear_data):
        X, y = linear_data
        pred = RidgeRegression(1e6).fit(X, y).predict(X[:5])
        np.testing.assert_allclose(pred, y.mean(), atol=1e-3)

    @pytest.mark.parametrize("bad", [np.array([[np.nan, 1.0]]), np.array([[np.inf, 1.0]])])
    def test_rejects_non_finite(self, bad):
        with pytest.raises(ValueError, match="NaN or infinite"):
            RidgeRegression().fit(bad, np.array([1.0]))

    def test_negative_alpha(self):
        with pytest.raises(ValueError):
            RidgeRegression(-1.0)


class TestImprovedMCLLR:
    def test_outputs(self, linear_data):
        X, y = linear_data
        model = ImprovedMCLLR(1.0, n_simulations=20, subsample_size=60, alpha=0.1, random_state=0).fit(X, y)
        pred, std = model.predict_with_uncertainty(X[:15])
        assert pred.shape == std.shape == (15,)
        assert np.isfinite(pred).all() and (std > 0).all()
        assert np.abs(pred - y[:15]).mean() < 0.3

    def test_reproducible_per_fit(self, linear_data):
        X, y = linear_data
        model = ImprovedMCLLR(1.0, 10, 50, random_state=7)
        first = model.fit(X, y).predict(X[:5])
        second = model.fit(X, y).predict(X[:5])
        np.testing.assert_allclose(first, second)

    def test_predictive_std_exceeds_epistemic(self, linear_data):
        X, y = linear_data
        kwargs = dict(bandwidth=1.0, n_simulations=30, subsample_size=60, alpha=0.1, random_state=0)
        _, total = ImprovedMCLLR(predictive_std=True, **kwargs).fit(X, y).predict_with_uncertainty(X[:10])
        _, epistemic = ImprovedMCLLR(predictive_std=False, **kwargs).fit(X, y).predict_with_uncertainty(X[:10])
        assert (total > epistemic).all()

    def test_full_neighbourhood_without_penalty_is_global_ols(self, linear_data):
        X, y = linear_data
        n = len(y)
        model = ImprovedMCLLR(1e6, n_simulations=3, subsample_size=n, alpha=0.0, predictive_std=False,
                              random_state=0).fit(X, y)
        pred, std = model.predict_with_uncertainty(X[:5])
        ols = RidgeRegression(0.0).fit(X, y).predict(X[:5])
        np.testing.assert_allclose(pred, ols, atol=1e-8)
        np.testing.assert_allclose(std, 0.0, atol=1e-8)

    def test_neighbourhoods_have_no_duplicates(self, linear_data):
        X, y = linear_data
        model = ImprovedMCLLR(0.5, 25, 40, random_state=0).fit(X, y)
        idx = model._sample_neighbourhoods(np.zeros(len(y)))
        assert idx.shape == (25, 40)
        assert all(len(set(row)) == 40 for row in idx)

    def test_sampling_prefers_nearby_points(self, linear_data):
        X, y = linear_data
        model = ImprovedMCLLR(0.3, 200, 20, random_state=0).fit(X, y)
        sq_dist = np.sum((X - X[0]) ** 2, axis=1)
        idx = model._sample_neighbourhoods(-(sq_dist - sq_dist.min()) / (2 * 0.3**2))
        assert np.mean(sq_dist[idx]) < 0.5 * np.mean(sq_dist)

    def test_tiny_bandwidth_is_finite(self, linear_data):
        X, y = linear_data
        model = ImprovedMCLLR(1e-6, 5, 30, random_state=0).fit(X, y)
        assert np.isfinite(model.predict_with_uncertainty(X[:5] + 10)[0]).all()

    def test_subsample_larger_than_training(self, linear_data):
        X, y = linear_data
        with pytest.raises(ValueError, match="cannot exceed"):
            ImprovedMCLLR(subsample_size=500).fit(X, y)

    @pytest.mark.parametrize(
        "kwargs", [{"bandwidth": 0}, {"n_simulations": 0}, {"subsample_size": 1}, {"alpha": -1}]
    )
    def test_invalid_parameters(self, kwargs):
        with pytest.raises(ValueError):
            ImprovedMCLLR(**kwargs)

    def test_on_real_data_with_standardisation(self, sample_data):
        X_train, X_test, y_train, y_test = sample_data
        model = PreparedModel(
            [Standardizer()], ImprovedMCLLR(2.0, 20, 160, alpha=10.0, random_state=0)
        ).fit(X_train, y_train)
        pred, std = model.predict_with_uncertainty(X_test)
        coverage = np.mean(np.abs(y_test - pred) <= 1.96 * std)
        assert coverage > 0.85  # predictive std is roughly calibrated
