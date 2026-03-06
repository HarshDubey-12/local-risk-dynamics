# src/models/kernels.py
from abc import ABC, abstractmethod
import numpy as np

class Kernel(ABC):
    """Base class for kernel functions."""
    
    @abstractmethod
    def __call__(self, X: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """Compute kernel weights."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

class GaussianKernel(Kernel):
    def __init__(self, bandwidth: float):
        self.bandwidth = bandwidth
    
    def __call__(self, X: np.ndarray, x0: np.ndarray) -> np.ndarray:
        diff = X - x0
        dist_sq = np.sum(diff ** 2, axis=1)
        return np.exp(-dist_sq / (2 * self.bandwidth ** 2))
    
    @property
    def name(self) -> str:
        return f"gaussian_bw{self.bandwidth}"

class EpanechnikovKernel(Kernel):
    def __init__(self, bandwidth: float):
        self.bandwidth = bandwidth
    
    def __call__(self, X: np.ndarray, x0: np.ndarray) -> np.ndarray:
        diff = X - x0
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        u = dist / self.bandwidth
        return np.maximum(0, 0.75 * (1 - u**2)) / self.bandwidth
    
    @property
    def name(self) -> str:
        return f"epanechnikov_bw{self.bandwidth}"

# Models accept kernel as dependency
class LocallyWeightedLinearRegression:
    def __init__(self, kernel: Kernel, fit_intercept: bool = True):
        self.kernel = kernel
        self.fit_intercept = fit_intercept
        # ...
    
    def predict(self, X_query):
        for x0 in X_query:
            weights = self.kernel(self.X_train_, x0)  # Pluggable!