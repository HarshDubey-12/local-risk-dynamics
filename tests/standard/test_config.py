"""Standard experiment/model configuration validation tests."""

from pathlib import Path

import pytest

from lrd.standard.config import ExperimentConfig, ModelConfig
from lrd.standard.models.factory import ModelFactory

DATA_PATH = "data/raw/F-F_Research_Data_5_Factors_2x3.csv"


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
            ModelConfig(**kwargs)

    def test_unknown_model_type(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            ModelConfig(model_type="foo")

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
            ModelConfig(**kwargs)

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
                models=[ModelConfig(model_type="global_linear")],
            )

    def test_experiment_seed_propagates_to_models(self):
        config = ExperimentConfig(
            dataset_path=DATA_PATH,
            train_ratio=0.8,
            random_state=7,
            models=[
                ModelConfig(model_type="mc_linear", n_simulations=5, subsample_size=10),
                ModelConfig(
                    model_type="mc_linear", n_simulations=5, subsample_size=10, random_state=3
                ),
            ],
        )
        assert [m.random_state for m in config.models] == [7, 3]

    @pytest.mark.parametrize(
        "config_path", sorted(str(p) for p in Path("experiments/standard").glob("config_*.yaml"))
    )
    def test_all_shipped_configs_build_models(self, config_path):
        config = ExperimentConfig.from_yaml(config_path)
        for model_config in config.models:
            assert ModelFactory.create(model_config) is not None
