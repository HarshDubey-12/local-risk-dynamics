"""Experiment tracker tests (lrd.core.tracking)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from lrd.core.tracking import ExperimentTracker

COMPARISON_CONFIG = "experiments/standard/config_comparison.yaml"


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
