"""
Cross-system model comparison.

Runs standard and improved models through the *same* walk-forward folds and
summarises them against the historical-mean benchmark: accuracy, calibration
at several interval levels, Diebold-Mariano tests, sub-period results and
folds won. Used by the dashboard (app/) and scripts/build_app_results.py.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from lrd.core.base import LocalRiskModel
from lrd.core.data.loader import load_fama_french
from lrd.core.data.preprocessing import time_series_split
from lrd.core.evaluation import metrics
from lrd.core.evaluation.rolling import evaluate_walk_forward
from lrd.core.evaluation.significance import diebold_mariano, oos_r2
from lrd.improved.config import ImprovedExperimentConfig, ImprovedModelConfig
from lrd.improved.tuning import TunedModel
from lrd.standard.config import ExperimentConfig, ModelCofig
from lrd.standard.models.factory import Modelfactory

DATA_PATH = "data/raw/F-F_Research_Data_5_Factors_2x3.csv"
STANDARD_CONFIG = "experiments/standard/config_comparison.yaml"
IMPROVED_CONFIG = "experiments/improved/config_improved.yaml"
TRAIN_WINDOW, TEST_WINDOW = 240, 12
BENCHMARK_KEY = "improved:historical_mean"
COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)
SPLIT_DATE = pd.Timestamp("2005-01-01")


# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """How to build one model, independent of which system it comes from."""

    key: str
    system: str          # "standard" | "improved"
    model_type: str
    params: dict
    tuned: bool
    build: Callable[[], LocalRiskModel] = field(repr=False, compare=False)


def standard_spec(config: ModelCofig, key: Optional[str] = None) -> ModelSpec:
    params = {k: v for k, v in asdict(config).items() if v is not None and k != "model_type"}
    return ModelSpec(
        key=key or f"standard:{config.model_type}",
        system="standard",
        model_type=config.model_type,
        params=params,
        tuned=False,
        build=lambda: Modelfactory.create(config),
    )


def improved_spec(
    model_config: ImprovedModelConfig,
    experiment: ImprovedExperimentConfig,
    key: Optional[str] = None,
) -> ModelSpec:
    params = dict(model_config.params)
    if model_config.preprocessing:
        params["preprocessing"] = list(model_config.preprocessing)
    if model_config.grid:
        params["grid"] = dict(model_config.grid)
    return ModelSpec(
        key=key or f"improved:{model_config.name}",
        system="improved",
        model_type=model_config.model_type,
        params=params,
        tuned=bool(model_config.grid),
        build=lambda: experiment.build_model(model_config),
    )


def improved_experiment(models: List[ImprovedModelConfig], base: Optional[ImprovedExperimentConfig] = None
                        ) -> ImprovedExperimentConfig:
    """An improved experiment holding ``models`` (plus the benchmark), sharing ``base`` settings."""
    base = base or ImprovedExperimentConfig.from_yaml(IMPROVED_CONFIG)
    names = {m.name for m in models}
    benchmark = [] if base.benchmark in names else [ImprovedModelConfig(base.benchmark, "historical_mean")]
    return ImprovedExperimentConfig(
        dataset_path=base.dataset_path, models=benchmark + list(models), benchmark=base.benchmark,
        random_state=base.random_state, train_window=base.train_window, test_window=base.test_window,
        inner_train_window=base.inner_train_window, inner_test_window=base.inner_test_window,
    )


def default_specs(standard_config: str = STANDARD_CONFIG, improved_config: str = IMPROVED_CONFIG) -> List[ModelSpec]:
    """Every model of the default standard and improved configs."""
    standard = ExperimentConfig.from_yaml(standard_config)
    improved = ImprovedExperimentConfig.from_yaml(improved_config)
    return [standard_spec(m) for m in standard.models] + [improved_spec(m, improved) for m in improved.models]


# ---------------------------------------------------------------------------
# Running models
# ---------------------------------------------------------------------------

@dataclass
class ModelRun:
    """Out-of-sample forecasts of one model plus per-fold details."""

    key: str
    system: str
    model_type: str
    params: dict
    protocol: str                 # "walk_forward" | "holdout"
    dates: pd.DatetimeIndex
    y_true: np.ndarray
    pred: np.ndarray
    std: Optional[np.ndarray]     # None for models without uncertainty
    folds: pd.DataFrame           # fold_id, test_start_date, test_end_date, rmse, mae
    tuning: pd.DataFrame          # fold_id, test_start_date, chosen, inner_mse (tuned models)

    @property
    def errors(self) -> np.ndarray:
        return self.y_true - self.pred

    @property
    def has_uncertainty(self) -> bool:
        return self.std is not None


def _fold_frame(folds, dates) -> pd.DataFrame:
    return pd.DataFrame([
        {"fold_id": f.fold_id, "test_start_date": dates[f.test_start], "test_end_date": dates[f.test_end - 1],
         "rmse": f.result.metrics["rmse"], "mae": f.result.metrics["mae"]}
        for f in folds
    ])


def run_model(
    spec: ModelSpec, X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex,
    train_window: int = TRAIN_WINDOW, test_window: int = TEST_WINDOW,
) -> ModelRun:
    """Walk-forward evaluation on the shared folds."""
    built: List[LocalRiskModel] = []

    def builder() -> LocalRiskModel:
        model = spec.build()
        built.append(model)
        return model

    folds = evaluate_walk_forward(builder, X, y, train_window, test_window)
    idx = np.concatenate([np.arange(f.test_start, f.test_end) for f in folds])
    stds = [f.result.uncertainties for f in folds]
    tuning = [
        {"fold_id": f.fold_id, "test_start_date": dates[f.test_start], "chosen": dict(m.best_params_),
         "inner_mse": float(np.nanmin(m.scores_))}
        for f, m in zip(folds, built) if isinstance(m, TunedModel)
    ]
    return ModelRun(
        key=spec.key, system=spec.system, model_type=spec.model_type, params=spec.params,
        protocol="walk_forward", dates=dates[idx], y_true=y[idx],
        pred=np.concatenate([f.result.predictions for f in folds]),
        std=None if any(s is None for s in stds) else np.concatenate(stds),
        folds=_fold_frame(folds, dates),
        tuning=pd.DataFrame(tuning, columns=["fold_id", "test_start_date", "chosen", "inner_mse"]),
    )


def run_holdout(
    spec: ModelSpec, X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex, train_ratio: float = 0.8,
) -> ModelRun:
    """Single chronological train/test split (secondary view)."""
    X_tr, X_te, y_tr, y_te, _, d_te = time_series_split(X, y, dates, train_ratio)
    model = spec.build().fit(X_tr, y_tr)
    pred, std = model.predict_with_uncertainty(X_te)
    tuning = []
    if isinstance(model, TunedModel):
        tuning = [{"fold_id": 0, "test_start_date": d_te[0], "chosen": dict(model.best_params_),
                   "inner_mse": float(np.nanmin(model.scores_))}]
    folds = pd.DataFrame([{"fold_id": 0, "test_start_date": d_te[0], "test_end_date": d_te[-1],
                           "rmse": metrics.rmse(y_te, pred), "mae": metrics.mae(y_te, pred)}])
    return ModelRun(
        key=spec.key, system=spec.system, model_type=spec.model_type, params=spec.params,
        protocol="holdout", dates=d_te, y_true=y_te, pred=np.asarray(pred), std=None if std is None else np.asarray(std),
        folds=folds, tuning=pd.DataFrame(tuning, columns=["fold_id", "test_start_date", "chosen", "inner_mse"]),
    )


def compute_default_runs(progress: Callable[[str], None] = lambda _: None) -> List[ModelRun]:
    """Walk-forward runs of every default model, on identical folds."""
    X, y, dates = load_fama_french(DATA_PATH)
    runs = []
    for spec in default_specs():
        progress(spec.key)
        runs.append(run_model(spec, X, y, dates))
    return runs


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def coverage(run: ModelRun, level: float) -> float:
    """Share of outcomes inside the model's central ``level`` interval (NaN without uncertainty)."""
    if not run.has_uncertainty:
        return float("nan")
    z = NormalDist().inv_cdf(0.5 + level / 2)
    return float(np.mean(np.abs(run.errors) <= z * run.std))


def _check_aligned(run: ModelRun, benchmark: ModelRun) -> None:
    if len(run.dates) != len(benchmark.dates) or not (run.dates == benchmark.dates).all():
        raise ValueError(f"{run.key} and {benchmark.key} were not evaluated on the same months.")


def stress_mask(y_true: np.ndarray) -> np.ndarray:
    """Months whose return is more than two standard deviations from the mean."""
    return np.abs(y_true - y_true.mean()) > 2 * y_true.std()


def summarize(runs: List[ModelRun], benchmark: ModelRun) -> pd.DataFrame:
    """One row per run: accuracy, significance vs benchmark, calibration, sub-periods, folds won."""
    for run in runs:
        _check_aligned(run, benchmark)
    b = benchmark.errors
    pre = benchmark.dates < SPLIT_DATE
    stress = stress_mask(benchmark.y_true)

    fold_rmse = pd.DataFrame({r.key: r.folds["rmse"].to_numpy() for r in runs})
    wins = fold_rmse.idxmin(axis=1).value_counts() if len(runs) > 1 else pd.Series(dtype=int)

    rows = []
    for run in runs:
        e = run.errors
        t_stat, p_value = diebold_mariano(e, b)
        row = {
            "key": run.key, "system": run.system, "model_type": run.model_type,
            "rmse": metrics.rmse(run.y_true, run.pred),
            "mae": metrics.mae(run.y_true, run.pred),
            "oos_r2_%": 100 * oos_r2(e, b),
            "dm_t": t_stat, "dm_p": p_value,
            "r2_pre2005_%": 100 * oos_r2(e[pre], b[pre]) if pre.any() else np.nan,
            "r2_post2005_%": 100 * oos_r2(e[~pre], b[~pre]) if (~pre).any() else np.nan,
            "r2_stress_%": 100 * oos_r2(e[stress], b[stress]) if stress.any() else np.nan,
            "folds_won": int(wins.get(run.key, 0)),
        }
        for level in COVERAGE_LEVELS:
            row[f"coverage_{int(level * 100)}"] = coverage(run, level)
        if run.has_uncertainty:
            row["interval_score_95"] = metrics.interval_score_95(run.y_true, run.pred, run.std)
            row["gaussian_nll"] = metrics.gaussian_nll(run.y_true, run.pred, run.std)
        else:
            row["interval_score_95"] = row["gaussian_nll"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_dm(runs: List[ModelRun]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diebold-Mariano t-statistics and p-values for every ordered pair (row vs column).

    A negative t in cell (i, j) means model i has smaller squared errors than j.
    """
    keys = [r.key for r in runs]
    t = pd.DataFrame(0.0, index=keys, columns=keys)
    p = pd.DataFrame(1.0, index=keys, columns=keys)
    for i, a in enumerate(runs):
        for j, b in enumerate(runs):
            if i != j:
                _check_aligned(a, b)
                t.iloc[i, j], p.iloc[i, j] = diebold_mariano(a.errors, b.errors)
    return t, p


def cumulative_advantage(run: ModelRun, benchmark: ModelRun) -> pd.Series:
    """Running sum of (benchmark squared error − model squared error); rising = beating it."""
    _check_aligned(run, benchmark)
    return pd.Series(np.cumsum(benchmark.errors**2 - run.errors**2), index=run.dates)


# ---------------------------------------------------------------------------
# Persistence (precomputed results shipped with the app)
# ---------------------------------------------------------------------------

def data_hash(path: str | Path = DATA_PATH) -> str:
    """Content hash, independent of line endings (git may check out CRLF on Windows)."""
    return hashlib.md5(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:8]


def save_runs(runs: List[ModelRun], directory: str | Path, extra_meta: Optional[Dict] = None) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"key": r.key, "system": r.system, "model_type": r.model_type,
         "protocol": r.protocol, "params": json.dumps(r.params, default=str)}
        for r in runs
    ]).to_csv(directory / "models.csv", index=False)
    pd.concat([
        pd.DataFrame({"key": r.key, "date": r.dates, "y_true": r.y_true, "pred": r.pred,
                      "std": r.std if r.has_uncertainty else np.nan})
        for r in runs
    ]).to_csv(directory / "predictions.csv", index=False)
    pd.concat([r.folds.assign(key=r.key) for r in runs]).to_csv(directory / "folds.csv", index=False)
    tuning = [r.tuning.assign(key=r.key, chosen=r.tuning["chosen"].map(json.dumps)) for r in runs if len(r.tuning)]
    (pd.concat(tuning) if tuning else pd.DataFrame(columns=["key", "fold_id", "test_start_date", "chosen", "inner_mse"])
     ).to_csv(directory / "tuning.csv", index=False)
    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_hash": data_hash(),
        "train_window": TRAIN_WINDOW, "test_window": TEST_WINDOW,
        **(extra_meta or {}),
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))
    return directory


def load_runs(directory: str | Path) -> tuple[List[ModelRun], Dict]:
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    models = pd.read_csv(directory / "models.csv")
    preds = pd.read_csv(directory / "predictions.csv", parse_dates=["date"])
    folds = pd.read_csv(directory / "folds.csv", parse_dates=["test_start_date", "test_end_date"])
    tuning = pd.read_csv(directory / "tuning.csv", parse_dates=["test_start_date"])
    runs = []
    for m in models.itertuples(index=False):
        p = preds[preds["key"] == m.key]
        t = tuning[tuning["key"] == m.key].drop(columns="key").reset_index(drop=True)
        t["chosen"] = t["chosen"].map(json.loads)
        runs.append(ModelRun(
            key=m.key, system=m.system, model_type=m.model_type, params=json.loads(m.params),
            protocol=m.protocol, dates=pd.DatetimeIndex(p["date"]), y_true=p["y_true"].to_numpy(),
            pred=p["pred"].to_numpy(), std=None if p["std"].isna().all() else p["std"].to_numpy(),
            folds=folds[folds["key"] == m.key].drop(columns="key").reset_index(drop=True),
            tuning=t,
        ))
    return runs, meta
