"""Hyperparameter sweep tests (lrd.standard.experiments.sweep)."""

from dataclasses import replace

import pytest

from lrd.standard.config import ExperimentConfig
from lrd.standard.experiments.sweep import HyperparameterSweep

COMPARISON_CONFIG = "experiments/standard/config_comparison.yaml"


class TestHyperparameterSweep:
    def test_sweep_from_path(self):
        results = HyperparameterSweep(COMPARISON_CONFIG).sweep(
            "ridge_linear", {"alpha": [0.1, 1.0]}
        )
        # One row per combination: only the swept model is trained.
        assert len(results) == 2
        assert set(results["model"]) == {"ridge_linear"}
        assert list(results["hyperparams"]) == ["{'alpha': 0.1}", "{'alpha': 1.0}"]
        assert results["rmse"].nunique() == 2

    def test_sweep_does_not_mutate_base_config(self):
        sweep = HyperparameterSweep(COMPARISON_CONFIG)
        before = [replace(m) for m in sweep.base_config.models]
        sweep.sweep("ridge_linear", {"alpha": [5.0]})
        assert sweep.base_config.models == before

    def test_sweep_accepts_config_object(self):
        config = ExperimentConfig.from_yaml("experiments/standard/config_global.yaml")
        results = HyperparameterSweep(config).sweep("global_linear", {"fit_intercept": [True]})
        assert len(results) == 1

    def test_sweep_unknown_model(self):
        with pytest.raises(ValueError, match="not found"):
            HyperparameterSweep(COMPARISON_CONFIG).sweep("lwlr_typo", {"bandwidth": [0.1]})

    def test_sweep_invalid_param_is_rejected(self):
        with pytest.raises(ValueError, match="bandwidth"):
            HyperparameterSweep(COMPARISON_CONFIG).sweep("lwlr", {"bandwidth": [-1.0]})
