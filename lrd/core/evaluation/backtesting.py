"""
Portfolio statistics and economic evaluation of return forecasts.

Statistics are annualised with ``periods_per_year`` (12 for the monthly data
used here). The economic evaluation follows the standard market-timing
exercise in the return-predictability literature (e.g. Campbell & Thompson,
2008): a mean-variance investor holds

    w_t = forecast_t / (gamma * sigma²_t),   clipped to [0, 1.5]

in the market and the rest in the risk-free asset, where sigma²_t is a
variance estimate known at time t (here: trailing 60-month variance of the
market excess return). Forecast quality is measured by the certainty-
equivalent return (CER) and Sharpe ratio of the resulting portfolio, compared
with the same investor using the historical-mean forecast.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                 periods_per_year: int = MONTHS_PER_YEAR, annualize: bool = True) -> float:
    """Sharpe ratio: mean excess return / its standard deviation."""
    excess = np.asarray(returns, dtype=float) - risk_free_rate
    std = np.std(excess, ddof=1)
    if std == 0:
        return float("nan")
    ratio = np.mean(excess) / std
    return float(ratio * np.sqrt(periods_per_year)) if annualize else float(ratio)


def volatility(returns: np.ndarray, periods_per_year: int = MONTHS_PER_YEAR, annualize: bool = True) -> float:
    """Standard deviation of returns."""
    vol = np.std(np.asarray(returns, dtype=float), ddof=1)
    return float(vol * np.sqrt(periods_per_year)) if annualize else float(vol)


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                  periods_per_year: int = MONTHS_PER_YEAR, annualize: bool = True) -> float:
    """Sortino ratio: mean excess return / downside deviation."""
    excess = np.asarray(returns, dtype=float) - risk_free_rate
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    ratio = np.mean(excess) / np.std(downside, ddof=1)
    return float(ratio * np.sqrt(periods_per_year)) if annualize else float(ratio)


def max_drawdown(returns: np.ndarray) -> float:
    """Largest peak-to-trough decline of the compounded return path (negative)."""
    cumulative = np.cumprod(1 + np.asarray(returns, dtype=float))
    running_max = np.maximum.accumulate(cumulative)
    return float(np.min((cumulative - running_max) / running_max))


def annualized_return(returns: np.ndarray, periods_per_year: int = MONTHS_PER_YEAR) -> float:
    """Geometric average return per year."""
    returns = np.asarray(returns, dtype=float)
    return float(np.prod(1 + returns) ** (periods_per_year / len(returns)) - 1)


def calmar_ratio(returns: np.ndarray, periods_per_year: int = MONTHS_PER_YEAR) -> float:
    """Annualised return / |maximum drawdown|."""
    md = max_drawdown(returns)
    return float(annualized_return(returns, periods_per_year) / abs(md)) if md != 0 else float("nan")


def backtest_summary(returns: np.ndarray, risk_free_rate: float = 0.0,
                     periods_per_year: int = MONTHS_PER_YEAR) -> Dict[str, float]:
    """Standard performance summary of a return series."""
    returns = np.asarray(returns, dtype=float)
    return {
        "total_return": float(np.prod(1 + returns) - 1),
        "annualized_return": annualized_return(returns, periods_per_year),
        "volatility": volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
    }


# ---------------------------------------------------------------------------
# Economic evaluation of forecasts
# ---------------------------------------------------------------------------

def trailing_variance(market_excess: pd.Series, window: int = 60, min_periods: int = 24) -> pd.Series:
    """Variance of the market excess return over the trailing ``window`` months,
    including the current month (known at forecast time)."""
    return market_excess.rolling(window, min_periods=min_periods).var()


def timing_weights(forecast: np.ndarray, variance: np.ndarray, risk_aversion: float = 3.0,
                   bounds: Tuple[float, float] = (0.0, 1.5)) -> np.ndarray:
    """Mean-variance market weights w = forecast / (gamma * variance), clipped."""
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive.")
    variance = np.asarray(variance, dtype=float)
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("variance must be positive and finite for every month.")
    return np.clip(np.asarray(forecast, dtype=float) / (risk_aversion * variance), *bounds)


def certainty_equivalent(excess_returns: np.ndarray, risk_aversion: float = 3.0,
                         periods_per_year: int = MONTHS_PER_YEAR) -> float:
    """Annualised certainty-equivalent excess return: mean − (gamma/2)·variance."""
    r = np.asarray(excess_returns, dtype=float)
    return float(periods_per_year * (np.mean(r) - 0.5 * risk_aversion * np.var(r, ddof=1)))


def economic_value(forecast: np.ndarray, benchmark_forecast: np.ndarray, realized_excess: np.ndarray,
                   variance: np.ndarray, risk_aversion: float = 3.0,
                   bounds: Tuple[float, float] = (0.0, 1.5)) -> Dict[str, float]:
    """Market-timing performance of a forecast versus the benchmark forecast.

    Returns annualised CER (in %) for both investors, the CER gain (the fee,
    in % per year, an investor would pay to switch to the model), annualised
    Sharpe ratios of the portfolio excess returns, the average market weight
    and average monthly turnover of the model portfolio.
    """
    realized = np.asarray(realized_excess, dtype=float)
    w_model = timing_weights(forecast, variance, risk_aversion, bounds)
    w_bench = timing_weights(benchmark_forecast, variance, risk_aversion, bounds)
    r_model, r_bench = w_model * realized, w_bench * realized
    cer_model = 100 * certainty_equivalent(r_model, risk_aversion)
    cer_bench = 100 * certainty_equivalent(r_bench, risk_aversion)
    return {
        "cer_%": cer_model,
        "cer_benchmark_%": cer_bench,
        "cer_gain_%": cer_model - cer_bench,
        "sharpe": sharpe_ratio(r_model),
        "sharpe_benchmark": sharpe_ratio(r_bench),
        "mean_weight": float(np.mean(w_model)),
        "turnover": float(np.mean(np.abs(np.diff(w_model)))) if len(w_model) > 1 else 0.0,
    }
