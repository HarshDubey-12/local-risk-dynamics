from __future__ import annotations
from typing import Tuple

import numpy as np

"""
Global Linear Regression model for Local Risk Dynamics.

This module implements a stationary baseline using closed-form
Ordinary Least Squares (OLS) with a numerically stable pseudo-inverse.

Scientific role
---------------
Serves as the control model assuming constant factor sensitivities
across time. Performance gaps versus local or stochastic models
quantify the effect of non-stationarity in financial data.
"""

from ..models.base import LocalRiskModel

class GlobalLinearRegression(LocalRiskModel):
    """
    Closed_form Ordinary Least Squares linear regression.

    Parameters 
    ----------
    fit_intercept : bool, default = True
        whether to include an intercept term in the model
    
    Attributes 
    ----------
    coef_ : np.ndarray of shape (n_features)
        Estimated slope coefficients.
    
    inetercept_ : float 
        Estimated intercept term (0.0 if fit_intercept = False).
    
    n_features_ : int 
        Number of input features seen during fitting.

    is_fitted_ : bool
        Indicates whether the model has been fitted.
    
    """
    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept

        # Attributes set during fitting 
        self.coef_: np.ndarray | None = None 
        self.intercept_: float | None = None
        self.n_featutres_: int | None = None
        self.is_fitted_: bool = False 

    # -----------------------------------------------------------------
    # Internal Utilities
    # -----------------------------------------------------------------

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add a column of ones to X if intercept is enabled."""
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack((ones, X))

    def _check_is_fitted_(self)->None:
        """Raise an error if the model is not fitted."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling predict or score.")
        
    # -----------------------------------------------------------------
    # Core API
    # -----------------------------------------------------------------

    def fit(self, X: np.mndarray, y: np.ndarray)-> "GlobalLinearRegression":
        """
        Fit the global linear regression model using closed-form OLS.

        parameters 
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training featutre matrix.
        
        y : np.ndarray of shape(n_samples,)
            Target vector.

        Returns 
        -----------
        self : GlobalLinearRegression
            Fitted model instance.
        """

        # ---------- Validation ----------
        if X.ndim != 2:
            raise ValueError("X must ne a 2D array.")
        
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples.")
        
        if X.shape[0] == 0:
            raise ValueError("Cannot fit model on empty dataset.")
        
        # ---------- Store feature count ---------
        self.n_features_ = X.shape[1]

        # ---------- Design matrix ----------
        X_design = self._add_intercept(X)

        # ---------- Closed form OLS via pseudo-inverse ----------
        beta = np.linalg.pinv(X_design) @ y

        # ---------- Separate intercept and coefficients ----------
        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        self.is_fitted_ = True
        return self
    
    # -----------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        prdeict target values using fitted linear model.

        parameters 
        -----------
        X : np.ndarray of shape(n_samples, n_features)
            Input feature matrix
        
        Returns 
        -----------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values. 
        """

        self._check_is_fitted_()

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}"
            )
        
        X_design = self._add_intercept(X)

        beta = (
            np.concatenate(([self.intercept_],self.coef_))
            if self.fit_intercept
            else self.coef_
        )

        return X_design @ beta
    
    # ---------------------------------------------------------------

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mean Squared Error (MSE) of predictions.

        Parameters 
        ------------
        X : np.ndarray of shape (n_samples, n_features)
            Feature Matrix.

        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns 
        ------------
        mse : float 
            Mean squared prediction error.
        """

        self._check_is_fitted_()

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)

        return float(mse)

    # -----------------------------------------------------------------
    # Interface compliance
    # -----------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self.is_fitted_

    @property
    def model_name(self) -> str:
        return "global_linear"

    @property
    def hyperparameters(self) -> dict:
        return {"fit_intercept": self.fit_intercept}
