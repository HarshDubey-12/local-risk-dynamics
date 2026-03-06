from __future__ import annotations

"""
Monte Carlo Local Linear Regression (MCLLR).

This module implements the proposed hybrid model combining:
    - Geometric locality via Gaussian kernel weighting
    - Stochastic locality via Monte Carlo sampling

It produces uncertainty-aware, locally adaptive predictions for non-stationary financial data.
"""

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
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MonteCarloLocalLinearRegression":
        """
        Store training data for Monte Carlo local sampling.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        
        y : np.ndarray of shape (n_samples,)
            Training target vector.

        Returns
        -------
        self : MonteCarloLocalLinearRegression
            Fitted model instance.
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
        
        self.X_train_ = X.astype(float)
        self.y_train_ = y.astype(float)
        self.n_features_ = X.shape[1]
        self.is_fitted_ = True
        
        return self

    # ------------------------------------------------------------------
    # Compute Kernel Weights 
    # ------------------------------------------------------------------
    def _compute_kernel_weights(self, x0: np.ndarray) -> np.ndarray:
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

        if x0.ndim != 1:
            raise ValueError("x0 must be a 1D array")
        
        if x0.shape[0] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {x0.shape[0]}")
        
        # Compute squared Euclidean distances (no square root for numerical stability)
        diff = self.X_train_ - x0
        sq_dist = np.sum(diff ** 2, axis=1)

        # Gaussian kernel: w_i(x0) = exp(-||x_i - x0||^2 / (2*tau^2))
        weights = np.exp(-sq_dist / (2 * self.bandwidth ** 2))

        return weights

    # ------------------------------------------------------------------
    # Predict  
    # ------------------------------------------------------------------
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predict target values for query points using MCLLR.

        For each test point x0:
          1. Compute kernel weights w_i(x0) = exp(-||x_i - x0||^2 / (2*tau^2))
          2. Normalize to probabilities p_i = w_i / sum(w_i) + eps
          3. For r=1 to n_simulations:
             a. Sample neighborhood S_r ~ w/o replacement(m, p)
             b. Fit plain OLS on S_r
             c. Predict y_r at x0
          4. Aggregate: mean(y_r), std(y_r)

        Parameters
        ----------
        X : np.ndarray of shape (n_test, n_features)
            Query point matrix.
        
        return_std : bool, default=False
            Whether to return uncertainty (standard deviation).

        Returns
        -------
        y_pred : np.ndarray of shape (n_test,)
            Mean predicted values.
        
        y_std : np.ndarray of shape (n_test,), optional
            Predicted standard deviations (if return_std=True).
        """
        
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

        # Outer loop over test points (query-dependent locality)
        for i in range(n_test):
            x0 = X[i]

            # Step 1: Compute raw kernel weights for this query point
            weights = self._compute_kernel_weights(x0)

            # Step 2: Normalize to probabilities (critical bridge)
            weight_sum = np.sum(weights)
            probabilities = weights / (weight_sum + eps)

            # Store Monte Carlo predictions for aggregation
            simulation_preds = np.zeros(self.n_simulations)

            # Inner loop: Monte Carlo simulations with stochastic neighborhoods
            for k in range(self.n_simulations):
                # Step 3: Sample local neighborhood without replacement
                indices = self._rng.choice(
                    self.X_train_.shape[0],
                    size=self.subsample_size,
                    replace=False,
                    p=probabilities,
                )

                # Create subsampled local neighborhood
                X_sub = self.X_train_[indices]
                y_sub = self.y_train_[indices]

                # Step 4: Fit plain OLS (unweighted) on local subset
                # Locality enforced via sampling, not regression weights
                X_sub_design = self._add_intercept(X_sub)
                beta = np.linalg.pinv(X_sub_design) @ y_sub

                # Predict at x0 with local linear model
                x0_design = self._add_intercept(x0.reshape(1, -1))
                simulation_preds[k] = (x0_design @ beta).item()
            
            # Step 5: Aggregate Monte Carlo predictions
            mean_predictions[i] = np.mean(simulation_preds)
            std_predictions[i] = np.std(simulation_preds)

        # Return after processing all test points
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