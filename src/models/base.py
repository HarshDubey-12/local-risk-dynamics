from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

class LocalRiskModel(ABC):
    """ Base protocol for all the regression models in the framework."""

    @abstractmethod
    def fit(self, X: np.ndarray, y:np.ndarray) -> "LocalRiskModel":
        """Fit model to training data."""
        pass 

    @abstractmethod
    def predict(self, X:np.ndarray) -> np.ndarray:
        """Return point predictions."""
        pass 

    def predict_with_uncertainty(self, X:np.ndarray) ->Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (predictions, Uncertainty), Optional for stochastic models."""
        return self.predict(X), None
    
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        pass 

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique identifier for model type."""
        pass 

    @property
    @abstractmethod
    def hyperparameters(self) -> dict:
        """Return hyperparameter dict for logging/comparison"""
        pass 