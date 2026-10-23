"""
Nested walk-forward hyperparameter tuning.

``TunedModel`` is itself a model: its ``fit`` runs an inner walk-forward
over the training data it is given, picks the candidate with the lowest
inner mean squared error, and refits that candidate on the full training
window. Used inside an outer walk-forward, every fold therefore selects its
hyperparameters from its own past only - no test data is ever seen.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Dict, List

import numpy as np

from lrd.core.base import LocalRiskModel
from lrd.core.evaluation.rolling import walk_forward_splits


def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """All combinations of a parameter grid, in a stable order."""
    if not grid:
        return [{}]
    names = list(grid)
    return [dict(zip(names, values)) for values in product(*(grid[n] for n in names))]


class TunedModel(LocalRiskModel):
    """Select hyperparameters by inner walk-forward on the training window.

    Parameters
    ----------
    builder : callable
        ``builder(params) -> LocalRiskModel`` creating a fresh, unfitted model
        (including its preparation steps) for one candidate.
    candidates : list of dict
        Parameter sets to compare (see ``expand_grid``).
    inner_train_window, inner_test_window : int
        Inner walk-forward windows, in months, within the training data.
    """

    def __init__(
        self,
        builder: Callable[[Dict[str, Any]], LocalRiskModel],
        candidates: List[Dict[str, Any]],
        inner_train_window: int,
        inner_test_window: int,
    ) -> None:
        if not candidates:
            raise ValueError("candidates must not be empty.")
        self.builder = builder
        self.candidates = candidates
        self.inner_train_window = inner_train_window
        self.inner_test_window = inner_test_window

        self.best_params_: Dict[str, Any] | None = None
        self.scores_: List[float] | None = None
        self.model_: LocalRiskModel | None = None

    def _inner_mse(self, params: Dict[str, Any], X: np.ndarray, y: np.ndarray, folds) -> float:
        errors = []
        for tr_s, tr_e, te_s, te_e in folds:
            model = self.builder(params).fit(X[tr_s:tr_e], y[tr_s:tr_e])
            errors.append(y[te_s:te_e] - model.predict(X[te_s:te_e]))
        return float(np.mean(np.concatenate(errors) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TunedModel":
        if len(self.candidates) == 1:
            self.best_params_, self.scores_ = self.candidates[0], [np.nan]
        else:
            folds = walk_forward_splits(
                len(y), self.inner_train_window, self.inner_test_window, self.inner_test_window
            )
            self.scores_ = [self._inner_mse(p, X, y, folds) for p in self.candidates]
            self.best_params_ = self.candidates[int(np.argmin(self.scores_))]
        self.model_ = self.builder(self.best_params_).fit(X, y)
        return self

    def _check(self) -> LocalRiskModel:
        if self.model_ is None:
            raise RuntimeError("TunedModel must be fitted before prediction.")
        return self.model_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._check().predict(X)

    def predict_with_uncertainty(self, X: np.ndarray):
        return self._check().predict_with_uncertainty(X)

    @property
    def is_fitted(self) -> bool:
        return self.model_ is not None

    @property
    def model_name(self) -> str:
        return self._check().model_name if self.model_ is not None else "tuned"

    @property
    def hyperparameters(self) -> dict:
        return dict(self._check().hyperparameters) if self.model_ is not None else {}
