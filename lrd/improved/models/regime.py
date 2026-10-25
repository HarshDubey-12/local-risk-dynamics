"""
Regime-state MCLLR: locality defined by the market regime, not by factor returns.

The original hypothesis is about *regimes*, but MCLLR measures similarity on
this month's factor returns, which are mostly noise. ``RegimeMCLLR`` keeps the
improved MCLLR (local ridge, Monte Carlo neighbourhoods, predictive variance)
and the regression on the five factors, but computes kernel distances on a
market-state vector built only from past market returns:

* trailing realised volatility of the market factor over ``state_window`` months
* trailing cumulative market return (momentum) over the same window

Both use months t - state_window + 1 .. t, which are known when forecasting
t + 1, so there is no look-ahead.

Assumption: rows are in chronological order and query rows directly follow
the training rows, as in every walk-forward fold, inner tuning fold and
hold-out split of this project. The state of the first query rows uses the
end of the training window as history.
"""

from __future__ import annotations

import numpy as np

from lrd.improved.models.mcllr import ImprovedMCLLR

MARKET_COLUMN = 0  # Mkt-RF is the first factor (lrd.core.data.loader.FACTOR_COLUMNS)


def market_state(market: np.ndarray, window: int, history: np.ndarray | None = None) -> np.ndarray:
    """(n, 2) trailing volatility and momentum of ``market``, optionally preceded by ``history``.

    The first rows use as many months as are available (volatility 0 for a
    single month), so no row ever uses future data.
    """
    history = np.empty(0) if history is None else np.asarray(history, dtype=float)
    series = np.concatenate([history, np.asarray(market, dtype=float)])
    offset = len(history)
    state = np.empty((len(market), 2))
    for i in range(len(market)):
        end = offset + i + 1
        window_values = series[max(0, end - window):end]
        state[i] = (window_values.std(), window_values.sum())
    return state


class RegimeMCLLR(ImprovedMCLLR):
    """Improved MCLLR whose kernel compares market regimes instead of factor returns.

    Parameters are those of ``ImprovedMCLLR`` plus ``state_window`` (months
    used for the trailing volatility and momentum). ``bandwidth`` is measured
    in standard deviations of the standardised state variables.
    """

    def __init__(self, bandwidth: float = 1.0, n_simulations: int = 50, subsample_size: int = 160,
                 alpha: float = 1.0, predictive_std: bool = True, random_state: int | None = None,
                 state_window: int = 12) -> None:
        super().__init__(bandwidth, n_simulations, subsample_size, alpha, predictive_std, random_state)
        if state_window < 2:
            raise ValueError("state_window must be at least 2.")
        self.state_window = int(state_window)
        self.state_train_: np.ndarray | None = None
        self._state_mean: np.ndarray | None = None
        self._state_scale: np.ndarray | None = None
        self._history: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegimeMCLLR":
        super().fit(X, y)
        raw = market_state(self.X_train_[:, MARKET_COLUMN], self.state_window)
        self._state_mean = raw.mean(axis=0)
        scale = raw.std(axis=0)
        self._state_scale = np.where(scale > 0, scale, 1.0)
        self.state_train_ = (raw - self._state_mean) / self._state_scale
        self._history = self.X_train_[-(self.state_window - 1):, MARKET_COLUMN]
        return self

    def _query_distances(self, X: np.ndarray) -> np.ndarray:
        raw = market_state(X[:, MARKET_COLUMN], self.state_window, history=self._history)
        state = (raw - self._state_mean) / self._state_scale
        return np.sum((state[:, None, :] - self.state_train_[None, :, :]) ** 2, axis=2)

    @property
    def model_name(self) -> str:
        return "mcllr_regime"

    @property
    def hyperparameters(self) -> dict:
        return {**super().hyperparameters, "state_window": self.state_window}
