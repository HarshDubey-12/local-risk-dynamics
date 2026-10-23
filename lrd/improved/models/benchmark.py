"""Historical-mean benchmark: the forecast every return model must beat."""

from __future__ import annotations

import numpy as np

from lrd.core.base import LocalRiskModel
from lrd.improved.models._linear import check_fit_inputs, check_predict_inputs


class HistoricalMean(LocalRiskModel):
    """Predict the training-window mean; uncertainty is the training std.

    This is the standard benchmark in return predictability: out-of-sample
    R² of every other model is measured relative to it.
    """

    def __init__(self) -> None:
        self.mean_: float | None = None
        self.std_: float | None = None
        self.n_features_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistoricalMean":
        X, y = check_fit_inputs(X, y)
        self.mean_ = float(y.mean())
        self.std_ = float(y.std(ddof=1)) if len(y) > 1 else 0.0
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = check_predict_inputs(X, self.n_features_)
        return np.full(X.shape[0], self.mean_)

    def predict_with_uncertainty(self, X: np.ndarray):
        pred = self.predict(X)
        return pred, np.full_like(pred, self.std_)

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None

    @property
    def model_name(self) -> str:
        return "historical_mean"

    @property
    def hyperparameters(self) -> dict:
        return {}
