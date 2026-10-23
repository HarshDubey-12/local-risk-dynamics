#!/usr/bin/env python
"""
End-to-end entry point for Local Risk Dynamics.

Standard system (``--system standard``, the default):

    1. Load and validate the experiment config (optionally a subset of models).
    2. Load the Fama-French dataset and make a chronological train/test split.
    3. Fit every model and evaluate it on the hold-out period.
    4. Optionally run walk-forward (rolling) evaluation.
    5. Save metrics, predictions and figures, and log the run.

Improved system (``--system improved``; outputs in <output-dir>/improved):

    1. Validate the raw data (missing/garbage/duplicate months, outliers flagged).
    2. Walk-forward folds shared by every model.
    3. Per model and fold: preparation and nested tuning on the training window,
       then fit and predict.
    4. Metrics, calibration and Diebold-Mariano tests against the benchmark.

Examples
--------
    python main.py
    python main.py --models global_linear lwlr mcllr
    python main.py --rolling --train-window 240 --test-window 12
    python main.py --config experiments/standard/config_mcllr.yaml --no-plots
    python main.py --system improved
    python main.py --system both --rolling
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

from lrd.standard.config import ExperimentConfig
from lrd.standard.experiments.rolling_runner import RollingExperimentRunner
from lrd.standard.experiments.runner import ExperimentRunner
from lrd.improved.config import ImprovedExperimentConfig
from lrd.improved.pipeline import ImprovedPipeline
from lrd.core.tracking import ExperimentTracker
from lrd.core.utils.plotting import (
    plot_model_comparison,
    plot_predictions_vs_true,
    plot_residuals,
    plot_time_series,
    plot_uncertainty,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = str(PROJECT_ROOT / "experiments" / "standard" / "config_comparison.yaml")
DEFAULT_IMPROVED_CONFIG = str(PROJECT_ROOT / "experiments" / "improved" / "config_improved.yaml")
SUMMARY_COLUMNS = ["model", "rmse", "mae", "r2", "coverage_95", "interval_width_95"]
IMPROVED_COLUMNS = ["model", "rmse", "mae", "oos_r2_%", "coverage_95", "interval_score_95",
                    "dm_p_vs_historical_mean"]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Local Risk Dynamics models end to end."
    )
    parser.add_argument(
        "--system",
        choices=["standard", "improved", "both"],
        default="standard",
        help="Which modelling system to run (default: standard).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Standard experiment YAML (default: experiments/standard/config_comparison.yaml).",
    )
    parser.add_argument(
        "--improved-config",
        default=DEFAULT_IMPROVED_CONFIG,
        help="Improved experiment YAML (default: experiments/improved/config_improved.yaml).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Only run these models: model types for standard, model names for improved "
        "(the improved benchmark is always kept). Default: all.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for outputs (default: results). Improved outputs go to <output-dir>/improved.",
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


def build_predictions_frame(runner: ExperimentRunner) -> pd.DataFrame:
    """Per-date ground truth, predictions and uncertainties for every model."""
    frame = pd.DataFrame({"date": runner.test_dates})
    frame["y_true"] = runner.eval_results[0].ground_truth
    for result in runner.eval_results:
        frame[f"{result.model_name}_pred"] = result.predictions
        if result.uncertainties is not None:
            frame[f"{result.model_name}_std"] = result.uncertainties
    return frame


def save_predictions(runner: ExperimentRunner, output_dir: Path) -> Path:
    """Write the predictions frame to ``output_dir/predictions.csv``."""
    path = output_dir / "predictions.csv"
    build_predictions_frame(runner).to_csv(path, index=False)
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


def run_standard(args: argparse.Namespace, output_dir: Path) -> List[tuple[str, pd.DataFrame, List[str]]]:
    """Standard system: hold-out (+ optional walk-forward). Returns tables to print."""
    print(f"[standard 1/4] Loading config {args.config}")
    config = load_config(args.config, args.models)
    print(f"  Models: {', '.join(m.model_type for m in config.models)}")

    print("[standard 2/4] Train/test evaluation")
    runner = ExperimentRunner(config)
    results = runner.run()
    results.to_csv(output_dir / "metrics.csv", index=False)
    save_predictions(runner, output_dir)

    tables = [("Standard: hold-out results", results.sort_values("rmse"), SUMMARY_COLUMNS)]
    if args.rolling:
        print(
            f"[standard 3/4] Walk-forward evaluation "
            f"(train={args.train_window}, test={args.test_window}, step={args.step or args.test_window})"
        )
        rolling_table = run_rolling(config, args, output_dir)
        tables.append((
            "Standard: walk-forward results (mean over folds)",
            rolling_table.reset_index().sort_values("rmse"),
            SUMMARY_COLUMNS,
        ))
    else:
        print("[standard 3/4] Walk-forward evaluation skipped (use --rolling)")

    print("[standard 4/4] Saving outputs")
    if not args.no_plots:
        n_figures = save_figures(runner, output_dir / "figures")
        print(f"  {n_figures} figures -> {output_dir / 'figures'}")
    if not args.no_track:
        log_dir = ExperimentTracker().log_experiment(
            args.config,
            results,
            metadata={
                "system": "standard",
                "dataset_hash": runner.dataset.source_hash,
                "models": [m.model_type for m in config.models],
                "rolling": args.rolling,
            },
        )
        print(f"  Run logged -> {log_dir}")
    return tables


def load_improved_config(config_path: str, names: Optional[List[str]]) -> ImprovedExperimentConfig:
    """Load the improved config, keeping the requested model names plus the benchmark."""
    config = ImprovedExperimentConfig.from_yaml(config_path)
    if not names:
        return config
    available = [m.name for m in config.models]
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Models {unknown} are not in {config_path}. Available: {available}")
    keep = set(names) | {config.benchmark}
    return replace(config, models=[m for m in config.models if m.name in keep])


def run_improved(args: argparse.Namespace, output_dir: Path) -> List[tuple[str, pd.DataFrame, List[str]]]:
    """Improved system: validated data, per-model preparation and nested tuning, walk-forward."""
    print(f"[improved 1/3] Loading config {args.improved_config}")
    config = load_improved_config(args.improved_config, args.models)
    print(f"  Models: {', '.join(m.name for m in config.models)} (benchmark: {config.benchmark})")

    print(f"[improved 2/3] Walk-forward evaluation (train={config.train_window}, test={config.test_window}; "
          f"nested tuning inner train={config.inner_train_window})")
    results = ImprovedPipeline(config).run()

    print("[improved 3/3] Saving outputs")
    results.save(output_dir)
    print(f"  summary, folds, predictions, tuning, validation report -> {output_dir}")
    if not args.no_track:
        log_dir = ExperimentTracker().log_experiment(
            args.improved_config,
            results.summary,
            metadata={
                "system": "improved",
                "dataset_hash": results.validation.source_hash,
                "models": [m.name for m in config.models],
            },
        )
        print(f"  Run logged -> {log_dir}")
    return [(
        f"Improved: walk-forward results (all test months; OOS R2 vs {config.benchmark})",
        results.summary.sort_values("rmse"),
        IMPROVED_COLUMNS,
    )]


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # Resolve user paths against the caller's directory, then work from the
    # project root, where the configs' dataset paths are relative to.
    args.config = str(Path(args.config).resolve())
    args.improved_config = str(Path(args.improved_config).resolve())
    output_dir = Path(args.output_dir).resolve()
    os.chdir(PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    tables = []
    try:
        if args.system in ("standard", "both"):
            tables += run_standard(args, output_dir)
        if args.system in ("improved", "both"):
            tables += run_improved(args, output_dir / "improved")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for title, frame, columns in tables:
        print(f"\n=== {title} ===")
        print_table(frame, columns)

    print(f"\nDone in {time.perf_counter() - start:.1f}s. Outputs in {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
