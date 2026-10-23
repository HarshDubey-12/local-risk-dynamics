"""
Backtesting helpers

Functions for financial performance evaluation and backtesting.
Includes risk-adjusted metrics and rolling-window analysis.
"""

import numpy as np
from typing import Optional, Tuple


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, annualize: bool = True) -> float:
    """Sharpe ratio: (mean return - rf) / std return."""
    excess_returns = returns - risk_free_rate
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    if std_excess == 0:
        return np.nan
    ratio = mean_excess / std_excess
    return ratio * np.sqrt(252) if annualize else ratio  # assuming daily returns


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown: largest peak-to-trough decline."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return float(np.min(drawdowns))


def volatility(returns: np.ndarray, annualize: bool = True) -> float:
    """Volatility: standard deviation of returns."""
    vol = np.std(returns, ddof=1)
    return vol * np.sqrt(252) if annualize else vol


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, annualize: bool = True) -> float:
    """Sortino ratio: (mean return - rf) / downside std."""
    excess_returns = returns - risk_free_rate
    downside = excess_returns[excess_returns < 0]
    if len(downside) == 0:
        return np.nan
    mean_excess = np.mean(excess_returns)
    downside_std = np.std(downside, ddof=1)
    ratio = mean_excess / downside_std
    return ratio * np.sqrt(252) if annualize else ratio


def calmar_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    if len(returns) < 252:
        # Not enough data for annualization
        total_return = np.prod(1 + returns) - 1
        md = max_drawdown(returns)
        return total_return / abs(md) if md != 0 else np.nan
    else:
        ann_return = np.prod(1 + returns)**(252 / len(returns)) - 1
        md = max_drawdown(returns)
        return ann_return / abs(md) if md != 0 else np.nan


def rolling_backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_size: int = 252,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling window backtest: compute metrics over sliding windows.

    Returns arrays of Sharpe ratios and max drawdowns for each window.
    Assumes y_true and y_pred are returns or errors.
    """
    sharpes = []
    mdds = []
    for i in range(0, len(y_true) - window_size + 1, step):
        window_true = y_true[i:i+window_size]
        window_pred = y_pred[i:i+window_size]
        # Simple strategy: assume returns based on prediction accuracy
        # For simplicity, use absolute error as "loss"
        errors = np.abs(window_true - window_pred)
        # Convert to pseudo-returns (negative for errors)
        pseudo_returns = -errors  # or some other logic
        sharpes.append(sharpe_ratio(pseudo_returns))
        mdds.append(max_drawdown(pseudo_returns))
    return np.array(sharpes), np.array(mdds)


def backtest_summary(
    returns: np.ndarray,
    risk_free_rate: float = 0.0
) -> dict:
    """Compute comprehensive backtest summary."""
    return {
        'total_return': float(np.prod(1 + returns) - 1),
        'annualized_return': float(np.prod(1 + returns)**(252 / len(returns)) - 1) if len(returns) >= 252 else np.nan,
        'volatility': volatility(returns),
        'sharpe_ratio': sharpe_ratio(returns, risk_free_rate),
        'max_drawdown': max_drawdown(returns),
        'sortino_ratio': sortino_ratio(returns, risk_free_rate),
        'calmar_ratio': calmar_ratio(returns, risk_free_rate),
    }
