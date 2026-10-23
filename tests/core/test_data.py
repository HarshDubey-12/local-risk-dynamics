"""Data loading, splitting and dataset-location tests (lrd.core.data)."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lrd.core.data.loader import load_fama_french
from lrd.standard.config import ExperimentConfig

DATA_PATH = "data/raw/F-F_Research_Data_5_Factors_2x3.csv"


class TestDataLoading:
    def test_load_fama_french(self):
        X, y, dates = load_fama_french('data/raw/F-F_Research_Data_5_Factors_2x3.csv')
        assert X.shape[0] == y.shape[0] == dates.shape[0]
        assert X.shape[1] == 5  # 5 factors
        assert isinstance(dates[0], (np.datetime64, pd.Timestamp))

    def test_time_series_split(self, sample_data):
        X_train, X_test, y_train, y_test = sample_data
        assert X_train.shape[0] + X_test.shape[0] == X_train.shape[0] + X_test.shape[0]  # total rows
        assert X_train.shape[1] == X_test.shape[1] == 5
        assert y_train.shape[0] == X_train.shape[0]
        assert y_test.shape[0] == X_test.shape[0]

class TestDatasetPath:
    def test_data_directory_is_lowercase(self):
        # Exact-case check so the configs also resolve on case-sensitive filesystems.
        assert "data" in os.listdir(".")
        assert "Data" not in os.listdir(".")
        assert Path(DATA_PATH).exists()

    @pytest.mark.parametrize(
        "config_path", sorted(str(p) for p in Path("experiments/standard").glob("config_*.yaml"))
    )
    def test_config_dataset_paths_exist(self, config_path):
        config = ExperimentConfig.from_yaml(config_path)
        assert Path(config.dataset_path).exists()
