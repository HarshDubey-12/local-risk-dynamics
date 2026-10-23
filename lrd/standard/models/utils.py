import numpy as np

def add_intercept(X: np.ndarray, fit_intercept: bool) -> np.ndarray:
    """Shared intercept-addition logic."""
    if not fit_intercept:
        return
    ones = np.ones((X.shape[0],1),dtype=X.dtype)
    return np.hstack((ones, X))

def gaussian_kernel(X: np.ndarray, x0:np.ndarray, bandwidth: float) -> np.ndarray:
    """Shared Gaussian Kernel."""
    diff = X - x0
    dist_sq = np.sum(diff**2, axis=1)
    return np.exp(-dist_sq / (2 * bandwidth ** 2))

def mormalize_weights(weights: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Normalize weights to probabilities."""
    w_sum = np.sum(weights)
    if w_sum < eps:
        w_sum = eps
    return weights / w_sum