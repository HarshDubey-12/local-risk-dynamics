"""Standard model tests: interface, MCLLR stability and kernel locality."""

import numpy as np
import pytest

from lrd.standard.config import ExperimentConfig
from lrd.standard.models.global_linear import GlobalLinearRegression
from lrd.standard.models.lwlr import LocallyWeightedLinearRegression
from lrd.standard.models.mc_linear import MonteCarloLinearRegression
from lrd.standard.models.mcllr import MonteCarloLocalLinearRegression
from lrd.standard.models.mcllr_ablations import MCLLRNoStochasticity

COMPARISON_CONFIG = "experiments/standard/config_comparison.yaml"


class TestModels:
    def test_global_linear(self, sample_data):
        X_train, X_test, y_train, y_test = sample_data
        model = GlobalLinearRegression()
        model.fit(X_train, y_train)
        # interface
        assert model.is_fitted
        assert isinstance(model.model_name, str)
        assert isinstance(model.hyperparameters, dict)

        y_pred = model.predict(X_test)
        assert y_pred.shape == y_test.shape
        assert model.score(X_test, y_test) > 0  # basic sanity
        # uncertainty api
        y_pred2, y_std = model.predict_with_uncertainty(X_test[:5])
        assert y_pred2.shape == (5,)
        assert y_std is None

    def test_lwlr(self, sample_data):
        X_train, X_test, y_train, y_test = sample_data
        model = LocallyWeightedLinearRegression(bandwidth=1.0)
        model.fit(X_train, y_train)
        assert model.is_fitted
        assert isinstance(model.model_name, str)
        assert isinstance(model.hyperparameters, dict)

        y_pred = model.predict(X_test[:10])  # test subset
        assert y_pred.shape == (10,)
        assert model.score(X_test[:10], y_test[:10]) > 0
        y_pred2, y_std = model.predict_with_uncertainty(X_test[:5])
        assert y_pred2.shape == (5,)
        assert y_std is None

    def test_mc_linear(self, sample_data):
        X_train, X_test, y_train, y_test = sample_data
        model = MonteCarloLinearRegression(n_simulations=5, subsample_size=20, random_state=42)
        model.fit(X_train, y_train)
        assert model.is_fitted
        assert isinstance(model.model_name, str)
        assert isinstance(model.hyperparameters, dict)

        y_pred, y_std = model.predict(X_test[:5], return_std=True)
        assert y_pred.shape == (5,)
        assert y_std.shape == (5,)
        assert np.all(y_std >= 0)
        # stochastic predictions change each call; just verify shape
        y_pred2, y_std2 = model.predict_with_uncertainty(X_test[:5])
        assert y_pred2.shape == (5,)
        assert y_std2.shape == (5,)

    def test_mcllr(self, sample_data):
        X_train, X_test, y_train, y_test = sample_data
        model = MonteCarloLocalLinearRegression(bandwidth=1.0, n_simulations=5, subsample_size=20, random_state=42)
        model.fit(X_train, y_train)
        assert model.is_fitted
        assert isinstance(model.model_name, str)
        assert isinstance(model.hyperparameters, dict)

        y_pred, y_std = model.predict(X_test[:5], return_std=True)
        assert y_pred.shape == (5,)
        assert y_std.shape == (5,)
        assert np.all(y_std >= 0)
        assert model.score(X_test[:5], y_test[:5]) > 0
        # results vary due to randomness
        y_pred2, y_std2 = model.predict_with_uncertainty(X_test[:5])
        assert y_pred2.shape == (5,)
        assert y_std2.shape == (5,)

class TestMCLLRStability:
    @pytest.mark.parametrize("bandwidth", [0.02, 0.01, 0.005, 1e-4])
    def test_small_bandwidths_predict_all_test_points(self, sample_data, bandwidth):
        X_train, X_test, y_train, _ = sample_data
        model = MonteCarloLocalLinearRegression(
            bandwidth=bandwidth, n_simulations=3, subsample_size=80, random_state=0
        ).fit(X_train, y_train)
        y_pred, y_std = model.predict(X_test, return_std=True)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))

    def test_probabilities_are_valid_when_weights_underflow(self, sample_data):
        X_train, _, y_train, _ = sample_data
        model = MonteCarloLocalLinearRegression(
            bandwidth=1e-6, n_simulations=2, subsample_size=50, random_state=0
        ).fit(X_train, y_train)
        # A far-away query would underflow every unshifted Gaussian weight.
        weights = model._compute_kernel_weights(X_train[0] + 1.0)
        probabilities = model._sampling_probabilities(weights)
        assert np.isclose(probabilities.sum(), 1.0)
        assert np.count_nonzero(probabilities) >= model.subsample_size
        assert np.all(probabilities >= 0)

    def test_probabilities_unchanged_for_moderate_bandwidth(self, sample_data):
        X_train, X_test, y_train, _ = sample_data
        model = MonteCarloLocalLinearRegression(
            bandwidth=0.08, n_simulations=2, subsample_size=50, random_state=0
        ).fit(X_train, y_train)
        x0 = X_test[0]
        raw = np.exp(-np.sum((X_train - x0) ** 2, axis=1) / (2 * 0.08 ** 2))
        expected = raw / raw.sum()
        actual = model._sampling_probabilities(model._compute_kernel_weights(x0))
        np.testing.assert_allclose(actual, expected, rtol=1e-10)


class TestLWLRStability:
    def test_tiny_bandwidth_does_not_silently_return_zero(self, sample_data):
        X_train, X_test, y_train, _ = sample_data
        pred = LocallyWeightedLinearRegression(bandwidth=1e-4).fit(X_train, y_train).predict(X_test[:10])
        assert np.isfinite(pred).all()
        assert np.any(pred != 0.0)

    def test_relative_weights_leave_fit_unchanged(self, sample_data):
        X_train, X_test, y_train, _ = sample_data
        model = LocallyWeightedLinearRegression(bandwidth=0.08).fit(X_train, y_train)
        x0 = X_test[0]
        raw = np.exp(-np.sum((X_train - x0) ** 2, axis=1) / (2 * 0.08**2))
        D = np.hstack([np.ones((len(X_train), 1)), X_train])
        beta = np.linalg.pinv(D.T @ (D * raw[:, None])) @ (D.T @ (raw * y_train))
        assert model.predict(X_test[:1])[0] == pytest.approx(np.r_[1.0, x0] @ beta, rel=1e-6)

    def test_compact_kernel_without_neighbours_raises(self, sample_data):
        from lrd.core.utils.kernels import EpanechnikovKernel

        X_train, _, y_train, _ = sample_data
        model = LocallyWeightedLinearRegression(0.01, kernel=EpanechnikovKernel(0.01)).fit(X_train, y_train)
        with pytest.raises(ValueError, match="increase the bandwidth"):
            model.predict(X_train[:1] + 10.0)


class TestLocality:
    def test_comparison_config_bandwidths_induce_locality(self, sample_data):
        X_train, X_test, _, _ = sample_data
        config = ExperimentConfig.from_yaml(COMPARISON_CONFIG)
        bandwidths = {m.bandwidth for m in config.models if m.bandwidth is not None}
        assert bandwidths
        for bw in bandwidths:
            sq_dist = np.sum((X_train - X_test[0]) ** 2, axis=1)
            weights = np.exp(-sq_dist / (2 * bw ** 2))
            # With bandwidth=1.0 every weight was >0.95; locality needs real spread.
            assert np.median(weights) < 0.9
            assert weights.min() < 0.1

    def test_lwlr_differs_from_global_at_configured_bandwidth(self, sample_data):
        X_train, X_test, y_train, _ = sample_data
        global_pred = GlobalLinearRegression().fit(X_train, y_train).predict(X_test[:20])
        lwlr_pred = LocallyWeightedLinearRegression(bandwidth=0.08).fit(
            X_train, y_train
        ).predict(X_test[:20])
        assert not np.allclose(global_pred, lwlr_pred, rtol=1e-2)

    def test_no_stochasticity_ablation_depends_on_bandwidth(self, sample_data):
        X_train, X_test, y_train, _ = sample_data
        narrow = MCLLRNoStochasticity(bandwidth=0.03, subsample_size=80).fit(X_train, y_train)
        wide = MCLLRNoStochasticity(bandwidth=1.0, subsample_size=80).fit(X_train, y_train)
        assert not np.allclose(narrow.predict(X_test[:20]), wide.predict(X_test[:20]))

    def test_no_stochasticity_ablation_small_bandwidth_is_finite(self, sample_data):
        X_train, X_test, y_train, _ = sample_data
        model = MCLLRNoStochasticity(bandwidth=1e-4, subsample_size=80).fit(X_train, y_train)
        assert np.all(np.isfinite(model.predict(X_test)))
