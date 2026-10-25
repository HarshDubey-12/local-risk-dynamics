"""Forecast comparison statistic tests (lrd.core.evaluation.significance)."""

import numpy as np
import pytest

from lrd.core.evaluation.significance import clark_west, diebold_mariano, oos_r2


class TestClarkWest:
    def test_identical_errors(self):
        e = np.random.default_rng(0).normal(size=50)
        assert clark_west(e, e) == (0.0, 0.5)

    def test_better_model_is_significant(self):
        rng = np.random.default_rng(0)
        bench = rng.normal(0, 2.0, 500)
        model = rng.normal(0, 1.0, 500)
        t, p = clark_west(model, bench)
        assert t > 0 and p < 0.01

    def test_adjustment_favours_nested_model_relative_to_dm(self):
        """With pure estimation noise, DM penalises the larger model; CW corrects for it."""
        rng = np.random.default_rng(3)
        y = rng.normal(0, 1, 400)
        bench_err = y - 0.0
        model_err = y - rng.normal(0, 0.3, 400)  # model adds noise around the benchmark
        dm_t, _ = diebold_mariano(model_err, bench_err)
        cw_t, _ = clark_west(model_err, bench_err)
        assert cw_t > -dm_t

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            clark_west(np.zeros(3), np.zeros(4))


class TestOosR2:
    def test_equal_errors_give_zero(self):
        e = np.array([0.1, -0.2, 0.3])
        assert oos_r2(e, e) == 0.0

    def test_smaller_errors_are_positive(self):
        assert oos_r2(np.array([0.1, 0.1]), np.array([0.2, 0.2])) == pytest.approx(0.75)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            oos_r2(np.zeros(3), np.zeros(4))


class TestDieboldMariano:
    def test_identical_errors(self):
        e = np.random.default_rng(0).normal(size=100)
        assert diebold_mariano(e, e) == (0.0, 1.0)

    def test_clearly_better_model_is_significant(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 2.0, size=500)
        model = rng.normal(0, 1.0, size=500)
        t, p = diebold_mariano(model, ref)
        assert t < 0
        assert p < 0.01

    def test_p_value_matches_normal_distribution(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=200), rng.normal(size=200)
        t, p = diebold_mariano(a, b)
        from math import erfc, sqrt
        assert p == pytest.approx(erfc(abs(t) / sqrt(2)))
