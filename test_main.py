#!/usr/bin/env python
"""Smoke tests for the end-to-end entry point (main.py)."""

import json
from pathlib import Path

import pandas as pd

import main
from src.Experiments.tracking import ExperimentTracker

FAST_MODELS = ["global_linear", "ridge_linear", "mc_linear"]


def test_hold_out_run_writes_outputs(tmp_path):
    out = tmp_path / "out"
    code = main.main(["--models", *FAST_MODELS, "--output-dir", str(out), "--no-track"])
    assert code == 0

    metrics = pd.read_csv(out / "metrics.csv")
    assert list(metrics["model"]) == FAST_MODELS
    assert metrics["rmse"].notna().all()

    predictions = pd.read_csv(out / "predictions.csv")
    assert len(predictions) == 164  # 20% hold-out of 816 months
    assert {"date", "y_true", "global_linear_pred", "mc_linear_pred", "mc_linear_std"} <= set(
        predictions.columns
    )
    assert "global_linear_std" not in predictions.columns

    figures = {p.name for p in (out / "figures").glob("*.png")}
    assert "comparison_rmse.png" in figures
    assert "mc_linear_uncertainty.png" in figures
    assert "global_linear_uncertainty.png" not in figures


def test_rolling_run_writes_fold_results(tmp_path):
    out = tmp_path / "out"
    code = main.main(
        [
            "--models", "global_linear", "mcllr",
            "--rolling", "--train-window", "600", "--test-window", "60",
            "--output-dir", str(out), "--no-plots", "--no-track",
        ]
    )
    assert code == 0
    folds = pd.read_csv(out / "rolling_folds.csv")
    # (816 - 600) // 60 = 3 folds per model
    assert folds.groupby("model")["fold_id"].nunique().to_dict() == {
        "global_linear": 3,
        "mcllr": 3,
    }
    assert (out / "rolling_summary.csv").exists()
    assert not (out / "figures").exists()


def test_run_is_logged(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(main, "ExperimentTracker", lambda: ExperimentTracker(log_dir))
    code = main.main(["--models", "global_linear", "--output-dir", str(tmp_path / "out"), "--no-plots"])
    assert code == 0

    (run_dir,) = log_dir.iterdir()
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["models"] == ["global_linear"]
    assert meta["dataset_hash"]


def test_unknown_model_fails_cleanly(tmp_path, capsys):
    code = main.main(["--models", "not_a_model", "--output-dir", str(tmp_path), "--no-track"])
    assert code == 1
    assert "not_a_model" in capsys.readouterr().err


def test_missing_config_fails_cleanly(tmp_path, capsys):
    code = main.main(["--config", str(tmp_path / "missing.yaml"), "--output-dir", str(tmp_path)])
    assert code == 1
    assert "Error" in capsys.readouterr().err


def test_relative_paths_resolve_from_caller_directory(tmp_path, monkeypatch):
    config = Path("experiments/config_global.yaml").resolve()
    monkeypatch.chdir(tmp_path)
    code = main.main(["--config", str(config), "--output-dir", "rel_out", "--no-plots", "--no-track"])
    assert code == 0
    assert (tmp_path / "rel_out" / "metrics.csv").exists()
