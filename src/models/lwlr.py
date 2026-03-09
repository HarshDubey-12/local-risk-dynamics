"""
Locally Weighted Linear Regression (LWLR) for Local Risk Dynamics.

This module implements deterministic locality using Gaussian
kernel-weighted least squares solved at prediction time.

Scientific role
---------------
Introduces geometric locality into linear regression by allowing
coefficients to vary smoothly across feature space.

Serves as the deterministic locality baseline against which
stochastic local models (MCLLR) are evaluated.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from ..utils.kernels import GaussianKernel, Kernel


from ..models.base import LocalRiskModel

class LocallyWeightedLinearRegression(LocalRiskModel):
    """
    Gaussian Kernel Locally Weighted Linear Regression (LWLR).
    """

    def __init__(
        self,
        bandwidth: float,
        fit_intercept: bool = True,
        kernel: Kernel | None = None,
    ) -> None:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be strictly positive.")

        self.bandwidth = float(bandwidth)
        self.fit_intercept = fit_intercept
        # allow custom kernel; default to Gaussian
        self.kernel = kernel if kernel is not None else GaussianKernel(self.bandwidth)

        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.is_fitted_: bool = False

    # --------------------------------------------------------------

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack((ones, X))

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction.")


    # --------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LocallyWeightedLinearRegression":
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must match in sample size.")
        if X.shape[0] == 0:
            raise ValueError("Empty dataset.")

        self.X_train_ = X.astype(float)
        self.y_train_ = y.astype(float)
        self.n_features_ = X.shape[1]
        self.is_fitted_ = True

        return self

    # --------------------------------------------------------------

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        self._check_is_fitted()

        if X_query.ndim != 2:
            raise ValueError("X_query must be 2D.")
        if X_query.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X_query.shape[1]}"
            )

        X_train = self.X_train_
        y_train = self.y_train_
        X_design = self._add_intercept(X_train)

        y_pred = np.zeros(X_query.shape[0])

        for i, x0 in enumerate(X_query):
            weights = self.kernel(X_train, x0)

            # Avoid explicit diagonal matrix
            W = weights[:, np.newaxis]
            X_weighted = X_design * W

            A = X_design.T @ X_weighted
            b = X_design.T @ (weights * y_train)

            beta = np.linalg.pinv(A) @ b

            x0_design = self._add_intercept(x0.reshape(1, -1))
            y_pred[i] = float(x0_design @ beta)

        return y_pred

    # --------------------------------------------------------------

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        self._check_is_fitted()

        y_pred = self.predict(X)
        return float(np.mean((y - y_pred) ** 2))

    # --------------------------------------------------------------
    # Interface compliance
    # --------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self.is_fitted_

    @property
    def model_name(self) -> str:
        return "lwlr"

    @property
    def hyperparameters(self) -> dict:
        return {"bandwidth": self.bandwidth, "fit_intercept": self.fit_intercept}

    def predict_with_uncertainty(self, X: np.ndarray):
        return self.predict(X), None