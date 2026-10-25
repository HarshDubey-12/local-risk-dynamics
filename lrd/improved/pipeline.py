"""
Improved pipeline: validate → shared folds → per-model prepare/tune/fit/predict
→ shared evaluation and significance tests.

Data loading, fold construction, metrics and significance tests all come
from ``lrd.core``, so results are directly comparable with the standard
system. Everything model-specific (preparation, tuning) happens inside each
model's ``fit`` on the training window of the current fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from lrd.core.data.loader import load_fama_french
from lrd.core.data.validation import ValidationReport, validate_fama_french
from lrd.core.evaluation import metrics
from lrd.core.evaluation.backtesting import economic_value, trailing_variance
from lrd.core.evaluation.rolling import evaluate_walk_forward, rolling_results_to_frame
from lrd.core.evaluation.significance import clark_west, diebold_mariano, oos_r2
from lrd.improved.config import ImprovedExperimentConfig
from lrd.improved.tuning import TunedModel


@dataclass
class ImprovedResults:
    """Everything one improved-pipeline run produces."""

    validation: ValidationReport
    summary: pd.DataFrame      # one row per model, metrics over all test months
    folds: pd.DataFrame        # per-fold metrics from the core evaluator
    predictions: pd.DataFrame  # date, y_true, <model>_pred, <model>_std
    tuning: pd.DataFrame       # hyperparameters chosen in each fold
    benchmark: str

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(output_dir / "summary.csv", index=False)
        self.folds.to_csv(output_dir / "folds.csv", index=False)
        self.predictions.to_csv(output_dir / "predictions.csv", index=False)
        self.tuning.to_csv(output_dir / "tuning.csv", index=False)
        (output_dir / "validation_report.md").write_text(self.validation.to_markdown(), encoding="utf-8")
        return output_dir


class ImprovedPipeline:
    """Run every model of an improved config through walk-forward evaluation."""

    def __init__(self, config: str | Path | ImprovedExperimentConfig) -> None:
        self.config = (
            config
            if isinstance(config, ImprovedExperimentConfig)
            else ImprovedExperimentConfig.from_yaml(config)
        )

    def run(self, log: Callable[[str], None] = print) -> ImprovedResults:
        cfg = self.config

        # 1. Validate the raw data once.
        report = validate_fama_french(cfg.dataset_path, outlier_threshold=cfg.outlier_threshold)
        log(f"  Data validation: {'passed' if report.ok else 'FAILED'} "
            f"({report.n_monthly_rows} months, {len(report.outliers)} extreme values flagged)")
        if cfg.strict_validation:
            report.raise_if_invalid()

        # 2. Load once; folds are shared by every model.
        X, y, dates = load_fama_french(cfg.dataset_path, target=cfg.target)

        fold_frames, tuning_rows = [], []
        predictions: pd.DataFrame | None = None
        errors: Dict[str, np.ndarray] = {}
        stds: Dict[str, np.ndarray] = {}

        # 3. Per model: preparation + tuning + fit + predict, inside each fold.
        for model_cfg in cfg.models:
            built: List = []

            def builder(model_cfg=model_cfg, built=built):
                model = cfg.build_model(model_cfg)
                built.append(model)
                return model

            folds = evaluate_walk_forward(
                model_builder=builder, X=X, y=y,
                train_window=cfg.train_window, test_window=cfg.test_window, step=cfg.step,
            )
            frame = rolling_results_to_frame(folds)
            frame["model"] = model_cfg.name
            fold_frames.append(frame)

            if predictions is None:
                test_idx = np.concatenate([np.arange(f.test_start, f.test_end) for f in folds])
                predictions = pd.DataFrame({"date": dates[test_idx], "y_true": y[test_idx]})
            pred = np.concatenate([f.result.predictions for f in folds])
            std = np.concatenate([f.result.uncertainties for f in folds])
            predictions[f"{model_cfg.name}_pred"] = pred
            predictions[f"{model_cfg.name}_std"] = std
            errors[model_cfg.name] = predictions["y_true"].to_numpy() - pred
            stds[model_cfg.name] = std

            for fold, model in zip(folds, built):
                if isinstance(model, TunedModel):
                    best_mse = float(np.nanmin(model.scores_)) if model.scores_ else np.nan
                    tuning_rows.append({
                        "model": model_cfg.name, "fold_id": fold.fold_id,
                        "test_start_date": dates[fold.test_start], **model.best_params_,
                        "inner_mse": best_mse,
                    })
            log(f"  {model_cfg.name}: {len(folds)} folds done")

        # 4. Shared evaluation over all test months, against the benchmark.
        y_true = predictions["y_true"].to_numpy()
        bench = errors[cfg.benchmark]
        bench_pred = y_true - bench
        variance = None
        if cfg.target == "return":  # market timing only makes sense for return forecasts
            variance = trailing_variance(pd.Series(X[:, 0], index=dates)).reindex(predictions["date"]).to_numpy()
        rows = []
        for model_cfg in cfg.models:
            name, e, s = model_cfg.name, errors[model_cfg.name], stds[model_cfg.name]
            pred = y_true - e
            t_stat, p_value = diebold_mariano(e, bench)
            cw_t, cw_p = clark_west(e, bench)
            row = {
                "model": name,
                "rmse": metrics.rmse(y_true, pred),
                "mae": metrics.mae(y_true, pred),
                "oos_r2_%": 100 * oos_r2(e, bench),
                "coverage_95": metrics.coverage_95(y_true, pred, s),
                "interval_width_95": metrics.interval_width_95(s),
                "gaussian_nll": metrics.gaussian_nll(y_true, pred, s),
                "interval_score_95": metrics.interval_score_95(y_true, pred, s),
                f"dm_t_vs_{cfg.benchmark}": t_stat,
                f"dm_p_vs_{cfg.benchmark}": p_value,
                f"cw_t_vs_{cfg.benchmark}": cw_t,
                f"cw_p_vs_{cfg.benchmark}": cw_p,
            }
            if variance is not None:
                econ = economic_value(pred, bench_pred, y_true, variance)
                row.update({"cer_gain_%": econ["cer_gain_%"], "sharpe": econ["sharpe"]})
            rows.append(row)

        return ImprovedResults(
            validation=report,
            summary=pd.DataFrame(rows),
            folds=pd.concat(fold_frames, ignore_index=True),
            predictions=predictions,
            tuning=pd.DataFrame(tuning_rows),
            benchmark=cfg.benchmark,
        )
