"""
Improved Monte Carlo Local Linear Regression.

Differences from the standard MCLLR (``lrd.standard.models.mcllr``), each
motivated by docs/MODEL_ASSESSMENT.md:

* Local fits are **ridge** regressions (penalty ``alpha * subsample_size``),
  which controls the variance that made the original model lose to Ridge.
* The kernel acts on whatever features it receives; used inside a
  ``PreparedModel`` with ``Standardizer`` the bandwidth is in standard
  deviations and no factor dominates the distance.
* The reported standard deviation is **predictive**: Monte Carlo spread plus
  the mean residual variance of the local fits, so intervals are calibrated.
* Neighbourhoods are sampled with the Gumbel top-k trick, which draws all
  simulations at once (same distribution as sequential weighted sampling
  without replacement) and solves the local fits as one batched system.
"""

from __future__ import annotations

import numpy as np

from lrd.core.base import LocalRiskModel
from lrd.improved.models._linear import (
    add_intercept,
    check_fit_inputs,
    check_predict_inputs,
    ridge_solve,
)


class ImprovedMCLLR(LocalRiskModel):
    """MCLLR with local ridge shrinkage and predictive uncertainty.

    Parameters
    ----------
    bandwidth : float
        Gaussian kernel bandwidth (in the units of the input features).
    n_simulations : int
        Neighbourhoods sampled per query point.
    subsample_size : int
        Training points per neighbourhood (must not exceed the training size).
    alpha : float
        Ridge penalty per local fit, scaled by ``subsample_size``.
    predictive_std : bool
        Add local residual variance to the Monte Carlo spread (recommended).
    random_state : int or None
        Seed; the generator is re-created on every ``fit`` so each fold is
        reproducible on its own.
    """

    def __init__(
        self,
        bandwidth: float = 2.0,
        n_simulations: int = 50,
        subsample_size: int = 160,
        alpha: float = 1.0,
        predictive_std: bool = True,
        random_state: int | None = None,
    ) -> None:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive.")
        if subsample_size <= 1:
            raise ValueError("subsample_size must be greater than 1.")
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.bandwidth = float(bandwidth)
        self.n_simulations = int(n_simulations)
        self.subsample_size = int(subsample_size)
        self.alpha = float(alpha)
        self.predictive_std = bool(predictive_std)
        self.random_state = random_state

        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.n_features_: int | None = None
        self._rng: np.random.Generator | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ImprovedMCLLR":
        X, y = check_fit_inputs(X, y)
        if self.subsample_size > X.shape[0]:
            raise ValueError(
                f"subsample_size ({self.subsample_size}) cannot exceed the "
                f"number of training samples ({X.shape[0]})."
            )
        self.X_train_, self.y_train_ = X, y
        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        return self

    def _sample_neighbourhoods(self, log_weights: np.ndarray) -> np.ndarray:
        """(n_simulations, subsample_size) indices drawn without replacement ∝ weights."""
        gumbel = self._rng.gumbel(size=(self.n_simulations, log_weights.shape[0]))
        keys = log_weights + gumbel
        return np.argpartition(-keys, self.subsample_size - 1, axis=1)[:, : self.subsample_size]

    def _query_distances(self, X: np.ndarray) -> np.ndarray:
        """Squared kernel distances (n_query, n_train); subclasses may use other locality spaces."""
        return np.sum((X[:, None, :] - self.X_train_[None, :, :]) ** 2, axis=2)

    def _predict_one(self, x0: np.ndarray, sq_dist: np.ndarray) -> tuple[float, float]:
        # Log-weights relative to the nearest point: no underflow at any bandwidth.
        log_w = -(sq_dist - sq_dist.min()) / (2 * self.bandwidth**2)
        idx = self._sample_neighbourhoods(log_w)

        D = add_intercept(self.X_train_[idx])  # (k, m, p+1)
        ys = self.y_train_[idx]  # (k, m)
        beta = ridge_solve(D, ys, self.alpha * self.subsample_size)  # (k, p+1)
        preds = beta @ add_intercept(x0[None, :])[0]  # (k,)

        variance = preds.var()
        if self.predictive_std:
            resid = ys - np.einsum("kmp,kp->km", D, beta)
            dof = max(self.subsample_size - D.shape[-1], 1)
            variance += float(np.mean(np.sum(resid**2, axis=1) / dof))
        return float(preds.mean()), float(np.sqrt(variance))

    def predict_with_uncertainty(self, X: np.ndarray):
        X = check_predict_inputs(X, self.n_features_)
        distances = self._query_distances(X)
        out = np.array([self._predict_one(x0, d) for x0, d in zip(X, distances)]).reshape(-1, 2)
        return out[:, 0], out[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_uncertainty(X)[0]

    @property
    def is_fitted(self) -> bool:
        return self.X_train_ is not None

    @property
    def model_name(self) -> str:
        return "mcllr_improved"

    @property
    def hyperparameters(self) -> dict:
        return {
            "bandwidth": self.bandwidth,
            "n_simulations": self.n_simulations,
            "subsample_size": self.subsample_size,
            "alpha": self.alpha,
            "predictive_std": self.predictive_std,
        }
