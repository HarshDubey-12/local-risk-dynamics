"""
MCLLR (Monte Carlo + LWLR) proposed model scaffold

This module will contain the proposed hybrid model used for experiments.

TODO:
 - Add public API for MCLLR model.
 - Implement training, prediction, and evaluation helpers.
"""

"""
Monte Carlo Local Linear Regression (MCLLR).

This model combines:
    - Geometric locality via kernel weights
    - Stochastic locality via Monte Carlo sampling

It produces uncertainty-aware, locally adaptive predictions.
"""

from __future__ import annotations
import numpy as np

class MonteCarloLocalLinearRegression:
    """
    Monte Carlo Local Linear Regression (MCLLR).

    Combines:
        - Geometric locality via Gaussian kernel weighting
        - Stochastic neighborhood selection via Monte Carlo sampling

    For each query point, the model:
        1. Computes kernel-based similarity weights
        2. Converts them into a probability distribution
        3. Samples local neighborhoods
        4. Fits local linear models
        5. Aggregates predictions into mean and uncertainty

    This produces uncertainty-aware, locally adaptive predictions.

    Parameters
    ------------
    bandwidth : float
        Gaussian kernel bandwidth controlling locality. Must be positive.

    n_simulations : int
        Number of Monte Carlo runs per query point. Must be positive. 

    subsample_size : int
        Number of samples drawn per local neighborhood. Must satisfy 1 < m <= n.

    fit_intercept : bool, default=True
        Whether to include an intercept term.

    random_state : int | None
        Random seed for reproducibility.

    Attributes
    ------------
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
        bandwidth: float,
        n_simulations: int,
        subsample_size: int,
        fit_intercept: bool = True,
        random_state: int | None = None,
    ) -> None:
        if bandwidth <=0:
            raise ValueError("Bandwidth must be positive.")
        
        if n_simulations <= 0:
            raise ValueError("n_simulations must be strictly positive.")

        if subsample_size <= 1:
            raise ValueError("subsample_size must be greater than 1.")
        
        self.bandwidth = bandwidth
        self.n_simulations = n_simulations
        self.subsample_size = subsample_size
        self.fit_intercept = fit_intercept
        self.random_state = random_state

        # Runtime Attributes
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
    # Compute Kernel Weights 
    # ------------------------------------------------------------------
    def _compute_kerne_weights(self, X0:np.ndarray)-> np.ndarray:
        """
        Compute raw Gaussian kernel weights for a single query point.

        Parameters
        ----------
        x0 : np.ndarray of shape (n_features,)
            Query point.

        Returns
        -------
        weights : np.ndarray of shape (n_train,)
            Raw (unnormalized) kernel weights.
        """

        if X0.ndim != 1:
            raise ValueError("X0 must be a 1D array")
        
        if X0.shape != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X0.shape[0]}")
        
        # compute euclidean squared distances
        diff = self.X_train_ - X0
        sq_dist = np.sum(diff**2, axis = 1)

        # Gaussian Kernel 
        weights = np.exp(-sq_dist / (2*self.bandwidth**2) )

        return weights

    # ------------------------------------------------------------------
    # Predict  
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray | tuple[np.ndarray,np.ndarray]:
        
        self._check_is_fitted()

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}"
            )
        
        n_test = X.shape[0]

        mean_predictions = np.zeros(n_test)
        std_predictions = np.zeros(n_test)

        eps = 1e-12

        # Loop over test points(query independent locality)
        for i in range(n_test):

            x0 = X[i]

            # Compute the kernel weights 
            weights = self._compute_kerne_weights(x0)

            # Normalize to probabilities(with epsilon safeguard)
            weight_sum = np.sum(weights)
            probabilities = weights/(weight_sum + eps)

            # Store simulation prediction for this query
            simulation_preds = np.zeros(self.n_simulations)

            # Monte Carlo Loop
            for k in range(self.n_simulations):

                # Sample indices according to local probabilities
                indices = self._rng.choice(
                    self.X_train_.shape[0],
                    size = self.subsample_size,
                    replace = False,
                    p = probabilities
                )

                # Create local subsamples 
                X_sub = self.X_train_[indices]
                y_sub = self.y_train_[indices]

                # Build design matrix 
                x_sub_design = self._add_intercept(X_sub)

                # Solve OlS
                beta = np.linalg.pinv(x_sub_design) @ y_sub

                # Predict at X0
                x0_design = self._add_intercept(x0.reshape(1,-1))
                simulation_preds[k] = (x0_design @ beta ).item()
            
            # Aggregate for this test point 
            mean_predictions[i] = simulation_preds.mean()
            std_predictions[i] = simulation_preds.std()

            if return_std:
                return mean_predictions, std_predictions
            
            return mean_predictions
        
    # ------------------------------------------------------------------
    # Score 
    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y: np.ndarray)->float:
        """
        Compute Mean Squared Error (MSE) using mean prediction.
        """
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(" X and y must have same number of samples.")
        
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)

        return float(mse)