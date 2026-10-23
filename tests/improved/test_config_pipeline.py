"""Improved config validation and end-to-end pipeline tests."""

import numpy as np
import pandas as pd
import pytest

from lrd.core.data.loader import load_fama_french
from lrd.improved.config import ImprovedExperimentConfig, ImprovedModelConfig
from lrd.improved.pipeline import ImprovedPipeline
from lrd.improved.preprocessing import PreparedModel
from lrd.improved.tuning import TunedModel

DATA_PATH = "data/raw/F-F_Research_Data_5_Factors_2x3.csv"
DEFAULT_CONFIG = "experiments/improved/config_improved.yaml"


def small_config(**overrides):
    models = [
        ImprovedModelConfig("historical_mean", "historical_mean"),
        ImprovedModelConfig("ridge", "ridge", ["standardize"], grid={"alpha": [0.1, 10.0]}),
        ImprovedModelConfig(
            "mcllr_improved", "mcllr_improved", ["standardize"],
            params={"n_simulations": 8, "subsample_size": 40}, grid={"bandwidth": [1.0, 3.0]},
        ),
    ]
    settings = dict(dataset_path=DATA_PATH, models=models, train_window=120, test_window=120,
                    inner_train_window=60, inner_test_window=30)
    settings.update(overrides)
    return ImprovedExperimentConfig(**settings)


class TestModelConfig:
    def test_default_yaml_loads(self):
        config = ImprovedExperimentConfig.from_yaml(DEFAULT_CONFIG)
        assert [m.name for m in config.models] == ["historical_mean", "ridge", "mcllr_improved"]
        assert config.benchmark == "historical_mean"
        assert config.strict_validation is True

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(model_type="nope"), "Unknown model_type"),
            (dict(model_type="ridge", params={"beta": 1}), "unknown params"),
            (dict(model_type="ridge", grid={"alpha": [1]}, params={"alpha": 2}), "both params and grid"),
            (dict(model_type="ridge", grid={"alpha": []}), "non-empty list"),
            (dict(model_type="ridge", grid={"alpha": [-1.0]}), "non-negative"),
            (dict(model_type="ridge", preprocessing=["scale"]), "Unknown preprocessing"),
            (dict(model_type="mcllr_improved", params={"random_state": 1}), "unknown params"),
        ],
    )
    def test_invalid(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ImprovedModelConfig(name="m", **kwargs)


class TestExperimentConfig:
    def test_benchmark_must_exist(self):
        with pytest.raises(ValueError, match="benchmark"):
            small_config(benchmark="other")

    def test_unique_names(self):
        m = ImprovedModelConfig("historical_mean", "historical_mean")
        with pytest.raises(ValueError, match="unique"):
            small_config(models=[m, m])

    def test_inner_windows_must_fit(self):
        with pytest.raises(ValueError, match="inner_train_window"):
            small_config(inner_train_window=100, inner_test_window=30)

    def test_subsample_must_fit_inner_window(self):
        with pytest.raises(ValueError, match="subsample_size 40 exceeds"):
            small_config(inner_train_window=30, inner_test_window=10)

    def test_missing_yaml_keys(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("dataset_path: x.csv\n")
        with pytest.raises(ValueError, match="missing required keys"):
            ImprovedExperimentConfig.from_yaml(path)

    def test_build_model(self):
        config = small_config()
        bench, ridge, mcllr = (config.build_model(m) for m in config.models)
        assert isinstance(bench, PreparedModel) and not bench.steps
        assert isinstance(ridge, TunedModel) and len(ridge.candidates) == 2
        assert isinstance(mcllr, TunedModel)
        # the experiment seed is injected into stochastic models
        assert mcllr.builder({"bandwidth": 1.0, "n_simulations": 8, "subsample_size": 40}).model.random_state == 42


class TestPipeline:
    @pytest.fixture(scope="class")
    def results(self):
        return ImprovedPipeline(small_config()).run(log=lambda _: None)

    def test_summary(self, results):
        summary = results.summary.set_index("model")
        assert list(summary.index) == ["historical_mean", "ridge", "mcllr_improved"]
        assert summary.loc["historical_mean", "oos_r2_%"] == 0.0
        assert summary[["rmse", "coverage_95", "gaussian_nll"]].notna().all().all()
        assert (summary["coverage_95"] > 0.8).all()  # every improved model is roughly calibrated

    def test_all_models_share_test_months(self, results):
        n = len(load_fama_french(DATA_PATH)[1])
        n_folds = (n - 120) // 120
        assert len(results.predictions) == n_folds * 120
        assert results.folds.groupby("model")["test_start"].apply(tuple).nunique() == 1
        assert results.predictions["date"].is_monotonic_increasing

    def test_tuning_log(self, results):
        tuning = results.tuning
        assert set(tuning["model"]) == {"ridge", "mcllr_improved"}
        assert set(tuning.loc[tuning.model == "ridge", "alpha"]) <= {0.1, 10.0}
        assert tuning["inner_mse"].notna().all()

    def test_save(self, results, tmp_path):
        out = results.save(tmp_path / "improved")
        for name in ["summary.csv", "folds.csv", "predictions.csv", "tuning.csv", "validation_report.md"]:
            assert (out / name).exists()
        assert "PASSED" in (out / "validation_report.md").read_text(encoding="utf-8")
        assert len(pd.read_csv(out / "summary.csv")) == 3

    def test_strict_validation_stops_on_bad_data(self, tmp_path):
        bad = tmp_path / "bad.csv"
        lines = open(DATA_PATH).read().splitlines(keepends=True)
        header_end = next(i for i, l in enumerate(lines) if l.startswith(",Mkt-RF")) + 1
        bad.write_text("".join(lines[:header_end] + lines[header_end:header_end + 3] + lines[header_end:]))
        with pytest.raises(ValueError, match="failed validation"):
            ImprovedPipeline(small_config(dataset_path=str(bad))).run(log=lambda _: None)
