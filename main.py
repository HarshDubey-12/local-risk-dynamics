#!/usr/bin/env python
"""
End-to-end entry point for Local Risk Dynamics.

Runs the full pipeline from an experiment config:

    1. Load and validate the experiment config (optionally a subset of models).
    2. Load the Fama-French dataset and make a chronological train/test split.
    3. Fit every model and evaluate it on the hold-out period.
    4. Optionally run walk-forward (rolling) evaluation.
    5. Save metrics, predictions and figures, and log the run.

Examples
--------
    python main.py
    python main.py --models global_linear lwlr mcllr
    python main.py --rolling --train-window 240 --test-window 12
    python main.py --config experiments/config_mcllr.yaml --no-plots
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # Figures are written to disk; no display needed.

import pandas as pd

from src.config import ExperimentConfig
from src.Experiments.rolling_runner import RollingExperimentRunner
from src.Experiments.runner import ExperimentRunner
from src.Experiments.tracking import ExperimentTracker
from src.utils.plotting import (
    plot_model_comparison,
    plot_predictions_vs_true,
    plot_residuals,
    plot_time_series,
    plot_uncertainty,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = str(PROJECT_ROOT / "experiments" / "config_comparison.yaml")
SUMMARY_COLUMNS = ["model", "rmse", "mae", "r2", "coverage_95", "interval_width_95"]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Local Risk Dynamics models end to end."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Experiment YAML (default: experiments/config_comparison.yaml).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL_TYPE",
        help="Only run these model types from the config (default: all).",
    )
    parser.add_argument(
        "--output-dir", default="results", help="Directory for CSVs and figures (default: results)."
    )
    parser.add_argument(
        "--rolling", action="store_true", help="Also run walk-forward evaluation."
    )
    parser.add_argument(
        "--train-window", type=int, default=240, help="Walk-forward training window in months."
    )
    parser.add_argument(
        "--test-window", type=int, default=12, help="Walk-forward test window in months."
    )
    parser.add_argument(
        "--step", type=int, default=None, help="Walk-forward step (default: test window)."
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip writing figures.")
    parser.add_argument(
        "--no-track", action="store_true", help="Skip logging the run to experiments/logs."
    )
    return parser.parse_args(argv)


def load_config(config_path: str, model_types: Optional[List[str]]) -> ExperimentConfig:
    """Load the experiment config, keeping only the requested model types."""
    config = ExperimentConfig.from_yaml(config_path)
    if not model_types:
        return config

    available = [m.model_type for m in config.models]
    unknown = sorted(set(model_types) - set(available))
    if unknown:
        raise ValueError(
            f"Models {unknown} are not in {config_path}. Available: {available}"
        )
    selected = [m for m in config.models if m.model_type in model_types]
    return replace(config, models=selected)


def save_predictions(runner: ExperimentRunner, output_dir: Path) -> Path:
    """Write per-date ground truth, predictions and uncertainties for every model."""
    frame = pd.DataFrame({"date": runner.test_dates})
    frame["y_true"] = runner.eval_results[0].ground_truth
    for result in runner.eval_results:
        frame[f"{result.model_name}_pred"] = result.predictions
        if result.uncertainties is not None:
            frame[f"{result.model_name}_std"] = result.uncertainties

    path = output_dir / "predictions.csv"
    frame.to_csv(path, index=False)
    return path


def save_figures(runner: ExperimentRunner, figures_dir: Path) -> int:
    """Write diagnostic figures per model plus cross-model comparisons."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    n_figures = 0

    for result in runner.eval_results:
        name = result.model_name
        y_true, y_pred = result.ground_truth, result.predictions
        plot_predictions_vs_true(y_true, y_pred, save_path=str(figures_dir / f"{name}_pred_vs_true.png"))
        plot_residuals(y_true, y_pred, save_path=str(figures_dir / f"{name}_residuals.png"))
        n_figures += 2
        if result.uncertainties is not None:
            # plot_uncertainty draws the bars it is given; scale std to a 95% interval.
            plot_uncertainty(
                y_pred, 1.96 * result.uncertainties,
                save_path=str(figures_dir / f"{name}_uncertainty.png"),
            )
            n_figures += 1

    results_by_model = {r.model_name: r for r in runner.eval_results}
    for metric in ("rmse", "r2"):
        plot_model_comparison(
            results_by_model, metric, save_path=str(figures_dir / f"comparison_{metric}.png")
        )
        n_figures += 1

    plot_time_series(
        runner.eval_results[0].ground_truth,
        dates=runner.test_dates,
        title="Test period: next-month market excess return",
        save_path=str(figures_dir / "test_target.png"),
    )
    return n_figures + 1


def run_rolling(config: ExperimentConfig, args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    """Walk-forward evaluation; returns a model x metric table of fold means."""
    runner = RollingExperimentRunner(
        config, train_window=args.train_window, test_window=args.test_window, step=args.step
    )
    folds, summary = runner.run()
    folds.to_csv(output_dir / "rolling_folds.csv", index=False)
    summary.to_csv(output_dir / "rolling_summary.csv", index=False)
    print(f"  {folds['fold_id'].nunique()} folds per model")
    return summary.pivot(index="model", columns="metric", values="mean")


def print_table(frame: pd.DataFrame, columns: List[str]) -> None:
    present = [c for c in columns if c in frame.columns]
    print(frame[present].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # Resolve user paths against the caller's directory, then work from the
    # project root, where the configs' dataset paths are relative to.
    args.config = str(Path(args.config).resolve())
    output_dir = Path(args.output_dir).resolve()
    os.chdir(PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    try:
        print(f"[1/4] Loading config {args.config}")
        config = load_config(args.config, args.models)
        print(f"  Models: {', '.join(m.model_type for m in config.models)}")

        print("[2/4] Train/test evaluation")
        runner = ExperimentRunner(config)
        results = runner.run()
        results.to_csv(output_dir / "metrics.csv", index=False)
        save_predictions(runner, output_dir)

        rolling_table = None
        if args.rolling:
            print(
                f"[3/4] Walk-forward evaluation "
                f"(train={args.train_window}, test={args.test_window}, step={args.step or args.test_window})"
            )
            rolling_table = run_rolling(config, args, output_dir)
        else:
            print("[3/4] Walk-forward evaluation skipped (use --rolling)")

        print("[4/4] Saving outputs")
        if not args.no_plots:
            n_figures = save_figures(runner, output_dir / "figures")
            print(f"  {n_figures} figures -> {output_dir / 'figures'}")
        if not args.no_track:
            log_dir = ExperimentTracker().log_experiment(
                args.config,
                results,
                metadata={
                    "dataset_hash": runner.dataset.source_hash,
                    "models": [m.model_type for m in config.models],
                    "rolling": args.rolling,
                },
            )
            print(f"  Run logged -> {log_dir}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\n=== Hold-out results ===")
    print_table(results.sort_values("rmse"), SUMMARY_COLUMNS)
    if rolling_table is not None:
        print("\n=== Walk-forward results (mean over folds) ===")
        print_table(rolling_table.reset_index().sort_values("rmse"), SUMMARY_COLUMNS)

    print(f"\nDone in {time.perf_counter() - start:.1f}s. Outputs in {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
