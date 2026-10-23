"""Forecast comparison statistic tests (lrd.core.evaluation.significance)."""

import numpy as np
import pytest

from lrd.core.evaluation.significance import diebold_mariano, oos_r2


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
