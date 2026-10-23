"""Train-only preparation tests (lrd.improved.preprocessing)."""

import numpy as np
import pytest

from lrd.improved.models import RidgeRegression
from lrd.improved.preprocessing import (
    PreparedModel,
    Standardizer,
    Winsorizer,
    build_transformer,
)


class TestStandardizer:
    def test_uses_training_statistics_only(self):
        train = np.array([[0.0, 10.0], [2.0, 30.0]])
        test = np.array([[100.0, 100.0]])
        s = Standardizer().fit(train)
        np.testing.assert_allclose(s.mean_, [1.0, 20.0])
        np.testing.assert_allclose(s.transform(test), [[99.0, 8.0]])

    def test_constant_column_passes_through(self):
        out = Standardizer().fit_transform(np.array([[1.0, 5.0], [3.0, 5.0]]))
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out[:, 1], 0.0)

    def test_transform_before_fit(self):
        with pytest.raises(RuntimeError):
            Standardizer().transform(np.zeros((2, 2)))


class TestWinsorizer:
    def test_clips_test_data_with_training_thresholds(self):
        train = np.arange(101, dtype=float)[:, None]
        w = Winsorizer(0.05).fit(train)
        np.testing.assert_allclose(w.transform(np.array([[-50.0], [50.0], [500.0]])), [[5.0], [50.0], [95.0]])

    def test_invalid_quantile(self):
        with pytest.raises(ValueError):
            Winsorizer(0.6)


class TestBuildTransformer:
    def test_specs(self):
        assert isinstance(build_transformer("standardize"), Standardizer)
        w = build_transformer({"winsorize": 0.02})
        assert isinstance(w, Winsorizer) and w.quantile == 0.02

    @pytest.mark.parametrize("spec", ["scale", {"winsorize": 0.01, "x": 1}, 3])
    def test_invalid_specs(self, spec):
        with pytest.raises(ValueError):
            build_transformer(spec)


class TestPreparedModel:
    def test_preparation_relearned_on_each_fit(self):
        rng = np.random.default_rng(0)
        model = PreparedModel([Standardizer()], RidgeRegression(0.0))
        model.fit(rng.normal(0, 1, (50, 3)), rng.normal(size=50))
        first_mean = model.steps[0].mean_.copy()
        model.fit(rng.normal(5, 1, (50, 3)), rng.normal(size=50))
        assert not np.allclose(first_mean, model.steps[0].mean_)

    def test_predictions_match_manual_pipeline(self):
        rng = np.random.default_rng(1)
        X, y = rng.normal(size=(80, 4)), rng.normal(size=80)
        X_test = rng.normal(size=(10, 4))
        prepared = PreparedModel([Standardizer()], RidgeRegression(1.0)).fit(X, y)
        s = Standardizer().fit(X)
        manual = RidgeRegression(1.0).fit(s.transform(X), y)
        np.testing.assert_allclose(prepared.predict(X_test), manual.predict(s.transform(X_test)))

    def test_interface(self):
        model = PreparedModel([Standardizer()], RidgeRegression(2.0))
        model.fit(np.random.default_rng(2).normal(size=(30, 2)), np.zeros(30))
        assert model.is_fitted
        assert model.model_name == "ridge"
        assert model.hyperparameters == {"preprocessing": ["standardize"], "alpha": 2.0}
