from __future__ import annotations

import numpy as np

from .base import LocalRiskModel


class RidgeLinearRegression(LocalRiskModel):
    """Regularized linear baseline (closed-form ridge regression)."""

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_features_: int | None = None
        self.is_fitted_: bool = False

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeLinearRegression":
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty dataset.")

        self.n_features_ = X.shape[1]
        X_design = self._add_intercept(X)

        p = X_design.shape[1]
        reg = self.alpha * np.eye(p)
        if self.fit_intercept:
            reg[0, 0] = 0.0

        beta = np.linalg.pinv(X_design.T @ X_design + reg) @ X_design.T @ y
        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction.")
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError("Input has invalid shape.")

        if self.fit_intercept:
            beta = np.concatenate(([self.intercept_], self.coef_))
        else:
            beta = self.coef_
        return self._add_intercept(X) @ beta

    @property
    def is_fitted(self) -> bool:
        return self.is_fitted_

    @property
    def model_name(self) -> str:
        return "ridge_linear"

    @property
    def hyperparameters(self) -> dict:
        return {"alpha": self.alpha, "fit_intercept": self.fit_intercept}
