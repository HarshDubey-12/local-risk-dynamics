"""Global ridge regression with a predictive standard deviation."""

from __future__ import annotations

import numpy as np

from lrd.core.base import LocalRiskModel
from lrd.improved.models._linear import (
    add_intercept,
    check_fit_inputs,
    check_predict_inputs,
    ridge_solve,
)


class RidgeRegression(LocalRiskModel):
    """Ridge regression; the strongest global baseline in the assessment.

    Parameters
    ----------
    alpha : float
        Penalty relative to sample size: the solver uses ``alpha * n``. With
        standardised features (``X^T X`` diagonal ≈ n) this makes ``alpha``
        comparable across window lengths.

    The predictive standard deviation is the residual standard deviation,
    so 95% intervals reflect return noise, not just parameter noise.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.sigma_: float | None = None
        self.n_features_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        X, y = check_fit_inputs(X, y)
        D = add_intercept(X)
        self.coef_ = ridge_solve(D, y, self.alpha * len(y))
        resid = y - D @ self.coef_
        dof = max(len(y) - D.shape[1], 1)
        self.sigma_ = float(np.sqrt(resid @ resid / dof))
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = check_predict_inputs(X, self.n_features_)
        return add_intercept(X) @ self.coef_

    def predict_with_uncertainty(self, X: np.ndarray):
        pred = self.predict(X)
        return pred, np.full_like(pred, self.sigma_)

    @property
    def is_fitted(self) -> bool:
        return self.coef_ is not None

    @property
    def model_name(self) -> str:
        return "ridge"

    @property
    def hyperparameters(self) -> dict:
        return {"alpha": self.alpha}
