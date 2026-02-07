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
from typing import Tuple
import numpy as np 

class LocallyWeightedLinearRgression:
    """
    Gaussian Kernel Locality weighted linear Regression.

    parameters 
    ------------
    bandwidth : float
        Kernel bandwidth(tau). must be positive.

    fit_intercept : bool, default = True
        Whether to include an intercept term.

    Attributes
    ------------
    X_train_ : np.ndarray
        Stored training features.
    
    y_train_ : np.ndarray
        Stored training targets.

    n_features_ : int
        Number of features in training data.

    is_fitted_ : bool
        indicates whether the model has been fitted.
    """

    def __init__(self, bandwidth: float, fit_intercept: bool = True) -> None:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be strictly positive.")
        self.bandwidth = bandwidth
        self.fit_intercept = fit_intercept

        self.X_train_ : np.ndarray | None = None
        self.y_train_ : np.ndarray | None = None 
        self.n_features_ : int | None = None
        self.is_fitted_ : bool = False

    # ----------------------------------------------------------------
    # Internal Utilities 
    # ----------------------------------------------------------------

    def _add_intecept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column if enabled."""
        if not self.fit_intercept:
            return X
        
        ones = np.ones((X.shape[0],1),dtype=X.dtype)
        return np.hstack((ones,X))
    
    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction.")
        
    def _gaussian_kernel(self, X: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian kernel weights for query point x0.
        """
        diff = X-x0
        dist_sq = np.sum(diff**2,axis = 1)
        weights = np.exp(-dist_sq/(2*self.bandwidth**2))

    # -----------------------------------------------------------------
    # Core API
    # -----------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LocallyWeightedLinearRgression":
        """ 
        Store training data for lazy local regression.

        Parameters
        ------------
        X : np.ndarray of shape (n_samples, n_featres)
        y : np.ndarray of shape (n_samples,)

        Returns 
        ------------
        self 
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of sapmles.")
        
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty dataset.")
        
        self.X_train_ = X
        self.y_train_ = y
        self.n_features_ = X.shape[1]
        self.is_fitted_ = True

        return self
    
    # ----------------------------------------------------------------

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        """
        Predict using locally weighted regression.

        Parameters 
        ------------
        X_query : np.ndarray of shape (n_queries, n_features)
        
        Returns
        ------------
        y_pred : np.ndarray of shape (n_queries,)
        """

        self.__check_is_fitted()

        if X_query.ndim !=2:
            raise ValueError("X_query must be a 2D array.")
        
        if X_query.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X_query.shape[1]}"
            )
        
        X_train = self.X_train_
        y_train = self.y_train_

        y_pred = []

        for x0 in X_query:
            # Compute kernel weights 
            weights = self._gaussian_kernel(X_train,x0)

            # Construct weighted design matrix 
            X_design = self._add_intecept(X_train)
            w = np.diag(weights)

            # Weighted least squares via pseudo_inverse 
            beta = np.linalg.pinv(X_design.T @ w @ X_design) @ (X_design.T @ w @ y_train)

            # Prepare query design row
            x0_design = self._add_intecept(x0.reshape(1,-1))

            # predict 
            y0 = float(x0_design @ beta)
            y_pred.append(y0)
        
        return np.array(y_pred)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mean Square Error (MSE).

        Parameters 
        ------------
        X : np.ndarray
        y : np.ndarray 

        Return
        ------------
        mse : float
        """
        if y.ndim != 1:
            raise ValueError("y must be 1D.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)

        return float(mse)