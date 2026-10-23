#!/usr/bin/env python
"""
Regression tests for the robustness fixes: MCLLR sampling stability,
kernel locality, the no-stochasticity ablation, config validation,
experiment tracking, hyperparameter sweeps and dataset path casing.
"""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ExperimentConfig, ModelCofig
from src.data.loader import load_fama_french
from src.data.preprocessing import time_series_split
from src.Experiments.sweep import HyperparameterSweep
from src.Experiments.tracking import ExperimentTracker
from src.models.factory import Modelfactory
from src.models.global_linear import GlobalLinearRegression
from src.models.lwlr import LocallyWeightedLinearRegression
from src.models.mcllr import MonteCarloLocalLinearRegression
from src.models.mcllr_ablations import MCLLRNoStochasticity

DATA_PATH = "data/raw/F-F_Research_Data_5_Factors_2x3.csv"
COMPARISON_CONFIG = "experiments/config_comparison.yaml"


@pytest.fixture
def sample_data():
    X, y, dates = load_fama_french(DATA_PATH)
    X_train, X_test, y_train, y_test, _, _ = time_series_split(X, y, dates, 0.8)
    return X_train, X_test, y_train, y_test


class TestDatasetPath:
    def test_data_directory_is_lowercase(self):
        # Exact-case check so the configs also resolve on case-sensitive filesystems.
        assert "data" in os.listdir(".")
        assert "Data" not in os.listdir(".")
        assert Path(DATA_PATH).exists()

    @pytest.mark.parametrize(
        "config_path", sorted(str(p) for p in Path("experiments").glob("config_*.yaml"))
    )
    def test_config_dataset_paths_exist(self, config_path):
        config = ExperimentConfig.from_yaml(config_path)
        assert Path(config.dataset_path).exists()


class TestMCLLRStability:
    @pytest.mark.parametrize("bandwidth", [0.02, 0.01, 0.005, 1e-4])
    def test_small_bandwidths_predict_all_test_points(self, sample_data, bandwidth):
        X_train, X_test, y_train, _ = sample_data
        model = MonteCarloLocalLinearRegression(
            bandwidth=bandwidth, n_simulations=3, subsample_size=80, random_state=0
        ).fit(X_train, y_train)
        y_pred, y_std = model.predict(X_test, return_std=True)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))

    def test_probabilities_are_valid_when_weights_underflow(self, sample_data):
        X_train, _, y_train, _ = sample_data
        model = MonteCarloLocalLinearRegression(
            bandwidth=1e-6, n_simulations=2, subsample_size=50, random_state=0
        ).fit(X_train, y_train)
        # A far-away query would underflow every unshifted Gaussian weight.
        weights = model._compute_kernel_weights(X_train[0] + 1.0)
        probabilities = model._sampling_probabilities(weights)
        assert np.isclose(probabilities.sum(), 1.0)
        assert np.count_nonzero(probabilities) >= model.subsample_size
        assert np.all(probabilities >= 0)

    def test_probabilities_unchanged_for_moderate_bandwidth(self, sample_data):
        X_train, X_test, y_train, _ = sample_data
        model = MonteCarloLocalLinearRegression(
            bandwidth=0.08, n_simulations=2, subsample_size=50, random_state=0
        ).fit(X_train, y_train)
        x0 = X_test[0]
        raw = np.exp(-np.sum((X_train - x0) ** 2, axis=1) / (2 * 0.08 ** 2))
        expected = raw / raw.sum()
        actual = model._sampling_probabilities(model._compute_kernel_weights(x0))
        np.testing.assert_allclose(actual, expected, rtol=1e-10)


class TestLocality:
    def test_comparison_config_bandwidths_induce_locality(self, sample_data):
        X_train, X_test, _, _ = sample_data
        config = ExperimentConfig.from_yaml(COMPARISON_CONFIG)
        bandwidths = {m.bandwidth for m in config.models if m.bandwidth is not None}
        assert bandwidths
        for bw in bandwidths:
            sq_dist = np.sum((X_train - X_test[0]) ** 2, axis=1)
            weights = np.exp(-sq_dist / (2 * bw ** 2))
            # With bandwidth=1.0 every weight was >0.95; locality needs real spread.
            assert np.median(weights) < 0.9
            assert weights.min() < 0.1

    def test_lwlr_differs_from_global_at_configured_bandwidth(self, sample_data):
        X_train, X_test, y_train, _ = sample_data
        global_pred = GlobalLinearRegression().fit(X_train, y_train).predict(X_test[:20])
        lwlr_pred = LocallyWeightedLinearRegression(bandwidth=0.08).fit(
            X_train, y_train
        ).predict(X_test[:20])
        assert not np.allclose(global_pred, lwlr_pred, rtol=1e-2)

    def test_no_stochasticity_ablation_depends_on_bandwidth(self, sample_data):
        X_train, X_test, y_train, _ = sample_data
        narrow = MCLLRNoStochasticity(bandwidth=0.03, subsample_size=80).fit(X_train, y_train)
        wide = MCLLRNoStochasticity(bandwidth=1.0, subsample_size=80).fit(X_train, y_train)
        assert not np.allclose(narrow.predict(X_test[:20]), wide.predict(X_test[:20]))

    def test_no_stochasticity_ablation_small_bandwidth_is_finite(self, sample_data):
        X_train, X_test, y_train, _ = sample_data
        model = MCLLRNoStochasticity(bandwidth=1e-4, subsample_size=80).fit(X_train, y_train)
        assert np.all(np.isfinite(model.predict(X_test)))


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs, missing",
        [
            ({"model_type": "mcllr", "n_simulations": 5, "subsample_size": 10}, "bandwidth"),
            ({"model_type": "mcllr_no_stochasticity", "subsample_size": 10}, "bandwidth"),
            ({"model_type": "lwlr"}, "bandwidth"),
            ({"model_type": "mc_linear", "subsample_size": 10}, "n_simulations"),
        ],
    )
    def test_missing_required_params(self, kwargs, missing):
        with pytest.raises(ValueError, match=missing):
            ModelCofig(**kwargs)

    def test_unknown_model_type(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            ModelCofig(model_type="foo")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"model_type": "lwlr", "bandwidth": 0.0},
            {"model_type": "mc_linear", "n_simulations": 0, "subsample_size": 10},
            {"model_type": "mc_linear", "n_simulations": 5, "subsample_size": 1},
            {"model_type": "ridge_linear", "alpha": -1.0},
        ],
    )
    def test_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            ModelCofig(**kwargs)

    def test_missing_experiment_keys(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("dataset: foo.csv\nmodel: lwlr\n")
        with pytest.raises(ValueError, match="missing required keys"):
            ExperimentConfig.from_yaml(path)

    def test_invalid_train_ratio(self):
        with pytest.raises(ValueError, match="train_ratio"):
            ExperimentConfig(
                dataset_path=DATA_PATH,
                train_ratio=1.5,
                models=[ModelCofig(model_type="global_linear")],
            )

    def test_experiment_seed_propagates_to_models(self):
        config = ExperimentConfig(
            dataset_path=DATA_PATH,
            train_ratio=0.8,
            random_state=7,
            models=[
                ModelCofig(model_type="mc_linear", n_simulations=5, subsample_size=10),
                ModelCofig(
                    model_type="mc_linear", n_simulations=5, subsample_size=10, random_state=3
                ),
            ],
        )
        assert [m.random_state for m in config.models] == [7, 3]

    @pytest.mark.parametrize(
        "config_path", sorted(str(p) for p in Path("experiments").glob("config_*.yaml"))
    )
    def test_all_shipped_configs_build_models(self, config_path):
        config = ExperimentConfig.from_yaml(config_path)
        for model_config in config.models:
            assert Modelfactory.create(model_config) is not None


class TestExperimentTracker:
    def test_creates_nested_log_dir(self, tmp_path):
        log_dir = tmp_path / "a" / "b" / "logs"
        ExperimentTracker(log_dir)
        assert log_dir.is_dir()

    @pytest.mark.parametrize("metadata", [None, {"dataset_hash": "abc123", "note": "x"}])
    def test_log_experiment(self, tmp_path, metadata):
        tracker = ExperimentTracker(tmp_path / "logs")
        results = pd.DataFrame([{"model": "global_linear", "rmse": 0.04}])
        exp_dir = Path(tracker.log_experiment(COMPARISON_CONFIG, results, metadata))

        assert (exp_dir / "config.yaml").exists()
        assert pd.read_csv(exp_dir / "result.csv").shape == (1, 2)
        meta = json.loads((exp_dir / "metadata.json").read_text())
        assert meta["config_file"] == COMPARISON_CONFIG
        assert "git_commit" in meta
        assert meta["dataset_hash"] == (metadata or {}).get("dataset_hash")

    def test_back_to_back_logs_do_not_collide(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "logs")
        results = pd.DataFrame([{"model": "m", "rmse": 1.0}])
        first = tracker.log_experiment(COMPARISON_CONFIG, results)
        second = tracker.log_experiment(COMPARISON_CONFIG, results)
        assert first != second


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
        config = ExperimentConfig.from_yaml("experiments/config_global.yaml")
        results = HyperparameterSweep(config).sweep("global_linear", {"fit_intercept": [True]})
        assert len(results) == 1

    def test_sweep_unknown_model(self):
        with pytest.raises(ValueError, match="not found"):
            HyperparameterSweep(COMPARISON_CONFIG).sweep("lwlr_typo", {"bandwidth": [0.1]})

    def test_sweep_invalid_param_is_rejected(self):
        with pytest.raises(ValueError, match="bandwidth"):
            HyperparameterSweep(COMPARISON_CONFIG).sweep("lwlr", {"bandwidth": [-1.0]})
