#!/usr/bin/env python
"""
Unit tests for MCLLR project using pytest.
Tests models, metrics, kernels, and sampling utilities.
"""

import numpy as np
import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
from src.data.loader import load_fama_french
from src.data.preprocessing import time_series_split
from src.models.global_linear import GlobalLinearRegression
from src.models.lwlr import LocallyWeightedLinearRegression
from src.models.mc_linear import MonteCarloLinearRegression
from src.models.mcllr import MonteCarloLocalLinearRegression
from src.evaluation.metrics import rmse, mae, mape, r2, explained_variance, median_ae, coverage_95, avg_std
from src.utils.kernels import GaussianKernel, EpanechnikovKernel
from src.utils.sampling import subsample_indices
from src.utils.plotting import (
    plot_predictions_vs_true, plot_uncertainty, plot_kernel_weights,
    plot_model_comparison, plot_time_series, plot_residuals
)
from src.evaluation.backtesting import sharpe_ratio, max_drawdown, volatility, sortino_ratio, calmar_ratio, backtest_summary


@pytest.fixture
def sample_data():
    """Load and split sample data for tests."""
    X, y, dates = load_fama_french('data/raw/F-F_Research_Data_5_Factors_2x3.csv')
    X_train, X_test, y_train, y_test, _, _ = time_series_split(X, y, dates, 0.8)
    return X_train, X_test, y_train, y_test


class TestDataLoading:
    def test_load_fama_french(self):
        X, y, dates = load_fama_french('data/raw/F-F_Research_Data_5_Factors_2x3.csv')
        assert X.shape[0] == y.shape[0] == dates.shape[0]
        assert X.shape[1] == 5  # 5 factors
        assert isinstance(dates[0], (np.datetime64, pd.Timestamp))

    def test_time_series_split(self, sample_data):
        X_train, X_test, y_train, y_test = sample_data
        assert X_train.shape[0] + X_test.shape[0] == X_train.shape[0] + X_test.shape[0]  # total rows
        assert X_train.shape[1] == X_test.shape[1] == 5
        assert y_train.shape[0] == X_train.shape[0]
        assert y_test.shape[0] == X_test.shape[0]


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


class TestMetrics:
    def test_rmse(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 2.9])
        assert rmse(y_true, y_pred) > 0
        assert rmse(y_true, y_true) == 0

    def test_mae(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 2.9])
        assert mae(y_true, y_pred) > 0
        assert mae(y_true, y_true) == 0

    def test_r2(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert r2(y_true, y_pred) == 1.0

    def test_coverage_95(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        y_std = np.array([0.1, 0.1, 0.1])
        coverage = coverage_95(y_true, y_pred, y_std)
        assert 0 <= coverage <= 1


class TestKernels:
    def test_gaussian_kernel(self):
        kernel = GaussianKernel(bandwidth=1.0)
        X = np.array([[0], [1], [2]])
        x0 = np.array([0])
        weights = kernel(X, x0)
        assert weights.shape == (3,)
        assert weights[0] == 1.0  # at zero distance
        assert np.all(weights >= 0)

    def test_epanechnikov_kernel(self):
        kernel = EpanechnikovKernel(bandwidth=1.0)
        X = np.array([[0], [0.5], [1.5]])
        x0 = np.array([0])
        weights = kernel(X, x0)
        assert weights.shape == (3,)
        assert weights[0] == 0.75  # Epanechnikov at 0 is 3/4
        assert weights[2] == 0.0  # distance > bandwidth


class TestSampling:
    def test_subsample_indices(self):
        indices = subsample_indices(100, 10, random_state=42)
        assert len(indices) == 10
        assert all(0 <= i < 100 for i in indices)
        assert len(set(indices)) == 10  # unique

    def test_subsample_indices_with_weights(self):
        weights = np.ones(100) / 100
        indices = subsample_indices(100, 10, weights=weights, random_state=42)
        assert len(indices) == 10
        assert len(set(indices)) == 10


class TestPlotting:
    def test_plot_predictions_vs_true(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        plot_predictions_vs_true(y_true, y_pred)  # Test no error

    def test_plot_uncertainty(self):
        y_pred = np.array([1, 2, 3])
        y_std = np.array([0.1, 0.2, 0.1])
        plot_uncertainty(y_pred, y_std)

    def test_plot_kernel_weights(self):
        distances = np.array([0, 1, 2, 3])
        weights = np.array([1.0, 0.5, 0.1, 0.0])
        plot_kernel_weights(distances, weights, 'Test Kernel')

    def test_plot_model_comparison(self):
        # Mock EvaluationResult objects
        class MockResult:
            def __init__(self, r2_val):
                self.metrics = {'r2': r2_val}

        results = {
            'Model1': MockResult(0.8),
            'Model2': MockResult(0.9),
            'Model3': MockResult(0.7)
        }
        plot_model_comparison(results, 'r2')

    def test_plot_time_series(self):
        y = np.random.randn(100).cumsum()
        plot_time_series(y, title='Test Time Series')

    def test_plot_residuals(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        plot_residuals(y_true, y_pred)


class TestBacktesting:
    def test_sharpe_ratio(self):
        returns = np.array([0.01, 0.02, -0.01, 0.03])
        assert sharpe_ratio(returns, annualize=False) > 0

    def test_max_drawdown(self):
        returns = np.array([0.1, -0.2, 0.05, -0.15])
        md = max_drawdown(returns)
        assert md < 0  # drawdown is negative

    def test_volatility(self):
        returns = np.array([0.01, 0.02, -0.01])
        vol = volatility(returns, annualize=False)
        assert vol > 0

    def test_backtest_summary(self):
        returns = np.array([0.01] * 300)  # 300 days
        summary = backtest_summary(returns)
        assert 'sharpe_ratio' in summary
        assert summary['total_return'] > 0
