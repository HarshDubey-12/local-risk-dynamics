"""Kernel, sampling and plotting utility tests (lrd.core.utils)."""

import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for testing

import numpy as np

from lrd.core.utils.kernels import EpanechnikovKernel, GaussianKernel
from lrd.core.utils.plotting import (
    plot_kernel_weights, plot_model_comparison, plot_predictions_vs_true,
    plot_residuals, plot_time_series, plot_uncertainty,
)
from lrd.core.utils.sampling import subsample_indices



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
        assert weights[2] == 0.0

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
