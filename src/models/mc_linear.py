"""
Monte Carlo Linear Regression (MC-LR).

This module implements stochastic subsampling to estimate
uncertainty in global linear regression.

Scientific role
---------------
Models:
    > Uncertainty in regression surface
    > Instability of global linear assumption
    > Sampling variability
"""

from __future__ import annotations
import numpy as np


from ..models.base import LocalRiskModel

class MonteCarloLinearRegression(LocalRiskModel):
    """
    Stochastic subsampling and prediction ensembling.

    Parameters
    ----------
    n_simulations : int
        Number of Monte Carlo runs. Must be positive.

    subsample_size : int
        Number of samples per regression (m). Must satisfy 1 < m <= n.

    fit_intercept : bool, default=True
        Whether to include an intercept term.

    random_state : int | None
        Random seed for reproducibility.

    Attributes
    ----------
    X_train_ : np.ndarray of shape (n_samples, n_features)
        Stored training feature matrix.

    y_train_ : np.ndarray of shape (n_samples,)
        Stored training targets.

    n_features_ : int
        Number of features seen during fitting.

    is_fitted_ : bool
        Indicates whether the model has been fitted.

    _rng : np.random.Generator
        Internal random number generator used for subsampling.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        n_simulations: int,
        subsample_size: int,
        fit_intercept: bool = True,
        random_state: int | None = None,
    ) -> None:

        if n_simulations <= 0:
            raise ValueError("n_simulations must be strictly positive.")

        if subsample_size <= 1:
            raise ValueError("subsample_size must be greater than 1.")

        self.n_simulations = n_simulations
        self.subsample_size = subsample_size
        self.fit_intercept = fit_intercept
        self.random_state = random_state

        # Runtime attributes
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.is_fitted_: bool = False

        # Internal RNG
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Internal Utilities
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction.")

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MonteCarloLinearRegression":
        """
        Store training data for Monte Carlo subsampling.
        """

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        if X.shape[0] == 0:
            raise ValueError("Cannot fit on an empty dataset.")

        if self.subsample_size > X.shape[0]:
            raise ValueError(
                "subsample_size cannot exceed number of training samples."
            )

        self.X_train_ = X
        self.y_train_ = y
        self.n_features_ = X.shape[1]
        self.is_fitted_ = True

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:

        self._check_is_fitted()

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}"
            )

        n_test = X.shape[0]
        predictions = np.zeros((self.n_simulations, n_test))

        # Precompute test design matrix
        X_test_design = self._add_intercept(X)

        # Monte Carlo loop
        for k in range(self.n_simulations):

            # Sample indices without replacement
            # draw without replacement using utility helper
            from ..utils.sampling import subsample_indices
            indices = subsample_indices(
                self.X_train_.shape[0],
                self.subsample_size,
                replace=False,
                random_state=self.random_state if self.random_state is not None else None,
            )

            # Create subsample
            X_sub = self.X_train_[indices]
            y_sub = self.y_train_[indices]

            # Build design matrix for subsample
            X_sub_design = self._add_intercept(X_sub)

            # Solve OLS via pseudo-inverse
            beta = np.linalg.pinv(X_sub_design) @ y_sub

            # Predict
            predictions[k] = X_test_design @ beta

        # Aggregate
        mean_pred = predictions.mean(axis=0)

        if return_std:
            std_pred = predictions.std(axis=0)
            return mean_pred, std_pred

        return mean_pred

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mean Squared Error (MSE) using mean prediction.
        """

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)

        return float(mse)

    # --------------------------------------------------------------
    # Interface compliance
    # --------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self.is_fitted_

    @property
    def model_name(self) -> str:
        return "mc_linear"

    @property
    def hyperparameters(self) -> dict:
        return {
            "n_simulations": self.n_simulations,
            "subsample_size": self.subsample_size,
            "fit_intercept": self.fit_intercept,
        }

    def predict_with_uncertainty(self, X: np.ndarray):
        return self.predict(X, return_std=True)