"""Nested walk-forward tuning tests (lrd.improved.tuning)."""

import numpy as np
import pytest

from lrd.core.base import LocalRiskModel
from lrd.core.evaluation.rolling import evaluate_walk_forward
from lrd.improved.models import RidgeRegression
from lrd.improved.tuning import TunedModel, expand_grid


class SpyModel(LocalRiskModel):
    """Records every training matrix it is fitted on."""

    seen: list = []

    def __init__(self, shift: float = 0.0) -> None:
        self.shift = shift
        self.mean_ = None

    def fit(self, X, y):
        SpyModel.seen.append(X.copy())
        self.mean_ = y.mean() + self.shift
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)

    @property
    def is_fitted(self):
        return self.mean_ is not None

    @property
    def model_name(self):
        return "spy"

    @property
    def hyperparameters(self):
        return {"shift": self.shift}


def test_expand_grid():
    assert expand_grid({}) == [{}]
    assert expand_grid({"a": [1, 2], "b": ["x"]}) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]


def test_selects_better_candidate():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = X @ np.array([2.0, -1.0]) + rng.normal(0, 0.1, 200)
    tuned = TunedModel(lambda p: RidgeRegression(**p), [{"alpha": 1000.0}, {"alpha": 0.0}], 120, 20)
    tuned.fit(X, y)
    assert tuned.best_params_ == {"alpha": 0.0}
    assert tuned.scores_[1] < tuned.scores_[0]
    assert tuned.model_.alpha == 0.0


def test_single_candidate_skips_inner_search():
    tuned = TunedModel(lambda p: RidgeRegression(**p), [{"alpha": 2.0}], 1000, 1000)
    tuned.fit(np.random.default_rng(0).normal(size=(30, 2)), np.zeros(30))
    assert tuned.best_params_ == {"alpha": 2.0}


def test_no_test_data_reaches_any_fit():
    """Training rows are negative, test rows positive: no fit may see a positive value."""
    n_train, n_test = 120, 30
    X = np.concatenate([-np.ones((n_train, 2)), np.ones((n_test, 2))])
    y = np.zeros(n_train + n_test)
    SpyModel.seen = []
    evaluate_walk_forward(
        lambda: TunedModel(lambda p: SpyModel(**p), [{"shift": 0.0}, {"shift": 1.0}], 60, 12),
        X, y, train_window=n_train, test_window=n_test,
    )
    assert len(SpyModel.seen) > 2  # inner folds for both candidates plus the refit
    assert all((seen < 0).all() for seen in SpyModel.seen)
    assert max(len(seen) for seen in SpyModel.seen) == n_train


def test_not_fitted():
    tuned = TunedModel(lambda p: RidgeRegression(**p), [{"alpha": 1.0}], 10, 5)
    assert not tuned.is_fitted
    with pytest.raises(RuntimeError):
        tuned.predict(np.zeros((1, 2)))


def test_empty_candidates():
    with pytest.raises(ValueError):
        TunedModel(lambda p: RidgeRegression(**p), [], 10, 5)
