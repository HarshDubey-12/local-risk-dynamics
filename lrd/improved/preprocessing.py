"""
Model-specific data preparation for the improved pipeline.

Every transformer learns its parameters in ``fit`` from training data only and
then applies them unchanged in ``transform``. ``PreparedModel`` bundles the
transformers with an estimator so preparation is re-learned inside every
``fit`` call - and therefore inside every walk-forward fold - without any
information from the test window leaking into training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from lrd.core.base import LocalRiskModel


class Transformer(ABC):
    """Feature transformation fitted on training data only."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "Transformer":
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class Standardizer(Transformer):
    """Z-score each feature with the training mean and standard deviation.

    Puts all factors on a common scale, so kernel bandwidths are measured in
    standard deviations and ridge penalties treat every factor equally.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)  # constant columns pass through
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Standardizer must be fitted before transform.")
        return (X - self.mean_) / self.scale_

    @property
    def name(self) -> str:
        return "standardize"


class Winsorizer(Transformer):
    """Clip each feature to training-set quantiles.

    Limits the leverage of extreme months on the fitted coefficients. Months
    are never removed, and test features are clipped with the *training*
    thresholds; targets are never modified.
    """

    def __init__(self, quantile: float = 0.01) -> None:
        if not 0.0 <= quantile < 0.5:
            raise ValueError("quantile must be in [0, 0.5).")
        self.quantile = quantile
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Winsorizer":
        self.lower_ = np.quantile(X, self.quantile, axis=0)
        self.upper_ = np.quantile(X, 1.0 - self.quantile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower_ is None:
            raise RuntimeError("Winsorizer must be fitted before transform.")
        return np.clip(X, self.lower_, self.upper_)

    @property
    def name(self) -> str:
        return f"winsorize({self.quantile:g})"


TRANSFORMERS = {"standardize": Standardizer, "winsorize": Winsorizer}


def build_transformer(spec: str | dict) -> Transformer:
    """Create a transformer from a config entry: ``"standardize"`` or ``{"winsorize": 0.01}``."""
    if isinstance(spec, str):
        name, args = spec, ()
    elif isinstance(spec, dict) and len(spec) == 1:
        name, value = next(iter(spec.items()))
        args = (value,)
    else:
        raise ValueError(f"Invalid preprocessing step: {spec!r}")
    if name not in TRANSFORMERS:
        raise ValueError(f"Unknown preprocessing step '{name}'. Expected one of: {sorted(TRANSFORMERS)}")
    return TRANSFORMERS[name](*args)


class PreparedModel(LocalRiskModel):
    """A model with its preparation steps, fitted together on training data.

    Parameters
    ----------
    steps : sequence of Transformer
        Applied in order. Fresh instances should be passed for every model.
    model : LocalRiskModel
        Estimator receiving the transformed features.
    """

    def __init__(self, steps: Sequence[Transformer], model: LocalRiskModel) -> None:
        self.steps = list(steps)
        self.model = model

    def _transform(self, X: np.ndarray) -> np.ndarray:
        for step in self.steps:
            X = step.transform(X)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PreparedModel":
        Xt = np.asarray(X, dtype=float)
        for step in self.steps:
            Xt = step.fit_transform(Xt)
        self.model.fit(Xt, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._transform(np.asarray(X, dtype=float)))

    def predict_with_uncertainty(self, X: np.ndarray):
        return self.model.predict_with_uncertainty(self._transform(np.asarray(X, dtype=float)))

    @property
    def is_fitted(self) -> bool:
        return self.model.is_fitted

    @property
    def model_name(self) -> str:
        return self.model.model_name

    @property
    def hyperparameters(self) -> dict:
        return {"preprocessing": [s.name for s in self.steps], **self.model.hyperparameters}
