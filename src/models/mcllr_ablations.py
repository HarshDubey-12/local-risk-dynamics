from __future__ import annotations

import numpy as np

from .base import LocalRiskModel


class MCLLRNoGeometry(LocalRiskModel):
    """Ablation: stochastic sampling without geometric kernel locality."""

    def __init__(
        self,
        n_simulations: int,
        subsample_size: int,
        fit_intercept: bool = True,
        random_state: int | None = None,
    ) -> None:
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive.")
        if subsample_size <= 1:
            raise ValueError("subsample_size must be greater than 1.")
        self.n_simulations = n_simulations
        self.subsample_size = subsample_size
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.is_fitted_: bool = False

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MCLLRNoGeometry":
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples.")
        if self.subsample_size > X.shape[0]:
            raise ValueError("subsample_size cannot exceed number of samples.")
        self.X_train_ = X.astype(float)
        self.y_train_ = y.astype(float)
        self.n_features_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction.")
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError("Input has invalid shape.")

        n_test = X.shape[0]
        mean_preds = np.zeros(n_test)
        std_preds = np.zeros(n_test)
        X_train_n = self.X_train_.shape[0]
        X_test_design = self._add_intercept(X)

        for i in range(n_test):
            sim_preds = np.zeros(self.n_simulations)
            for k in range(self.n_simulations):
                idx = self._rng.choice(
                    X_train_n, size=self.subsample_size, replace=False
                )
                X_sub = self._add_intercept(self.X_train_[idx])
                y_sub = self.y_train_[idx]
                beta = np.linalg.pinv(X_sub) @ y_sub
                sim_preds[k] = float(X_test_design[i] @ beta)
            mean_preds[i] = float(np.mean(sim_preds))
            std_preds[i] = float(np.std(sim_preds))

        return (mean_preds, std_preds) if return_std else mean_preds

    def predict_with_uncertainty(self, X: np.ndarray):
        return self.predict(X, return_std=True)

    @property
    def is_fitted(self) -> bool:
        return self.is_fitted_

    @property
    def model_name(self) -> str:
        return "mcllr_no_geometry"

    @property
    def hyperparameters(self) -> dict:
        return {
            "n_simulations": self.n_simulations,
            "subsample_size": self.subsample_size,
            "fit_intercept": self.fit_intercept,
        }


class MCLLRNoStochasticity(LocalRiskModel):
    """Ablation: geometric locality without stochastic neighborhood sampling."""

    def __init__(
        self,
        bandwidth: float,
        subsample_size: int,
        fit_intercept: bool = True,
    ) -> None:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")
        if subsample_size <= 1:
            raise ValueError("subsample_size must be greater than 1.")
        self.bandwidth = bandwidth
        self.subsample_size = subsample_size
        self.fit_intercept = fit_intercept

        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.is_fitted_: bool = False

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MCLLRNoStochasticity":
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples.")
        if self.subsample_size > X.shape[0]:
            raise ValueError("subsample_size cannot exceed number of samples.")
        self.X_train_ = X.astype(float)
        self.y_train_ = y.astype(float)
        self.n_features_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def _kernel_weights(self, x0: np.ndarray) -> np.ndarray:
        diff = self.X_train_ - x0
        sq_dist = np.sum(diff**2, axis=1)
        return np.exp(-sq_dist / (2 * self.bandwidth**2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction.")
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError("Input has invalid shape.")

        y_pred = np.zeros(X.shape[0])
        for i, x0 in enumerate(X):
            weights = self._kernel_weights(x0)
            idx = np.argsort(weights)[-self.subsample_size :]
            X_sub = self._add_intercept(self.X_train_[idx])
            y_sub = self.y_train_[idx]
            beta = np.linalg.pinv(X_sub) @ y_sub
            x0_design = self._add_intercept(x0.reshape(1, -1))
            y_pred[i] = float(x0_design @ beta)
        return y_pred

    @property
    def is_fitted(self) -> bool:
        return self.is_fitted_

    @property
    def model_name(self) -> str:
        return "mcllr_no_stochasticity"

    @property
    def hyperparameters(self) -> dict:
        return {
            "bandwidth": self.bandwidth,
            "subsample_size": self.subsample_size,
            "fit_intercept": self.fit_intercept,
        }
