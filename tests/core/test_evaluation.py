"""Metric and backtesting tests (lrd.core.evaluation)."""

import numpy as np

from lrd.core.evaluation.backtesting import backtest_summary, max_drawdown, sharpe_ratio, volatility
from lrd.core.evaluation.metrics import coverage_95, mae, r2, rmse



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
