"""Cached data access and model runs for the dashboard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.catalog import model_info
from lrd import comparison as cmp
from lrd.core.data.loader import load_fama_french
from lrd.core.data.validation import ValidationReport, validate_fama_french
from lrd.improved.config import ImprovedModelConfig
from lrd.standard.config import ModelCofig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "app" / "data" / "default_run"
SEED = 42


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data() -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    return load_fama_french(cmp.DATA_PATH)


@st.cache_data(show_spinner=False)
def raw_frame() -> pd.DataFrame:
    """Factors at month t (decimal) with the t+1 target."""
    X, y, dates = load_data()
    frame = pd.DataFrame(X, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA"], index=dates)
    frame["target (Mkt-RF t+1)"] = y
    return frame


@st.cache_data(show_spinner=False)
def validation(threshold: float = 5.0) -> ValidationReport:
    return validate_fama_french(cmp.DATA_PATH, outlier_threshold=threshold)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Running every model on the shared folds (about a minute, once)…")
def default_runs() -> dict[str, cmp.ModelRun]:
    """Precomputed default runs; recomputed if missing or built from different data."""
    try:
        runs, meta = cmp.load_runs(DEFAULT_RUN_DIR)
        if meta.get("data_hash") == cmp.data_hash() and meta.get("train_window") == cmp.TRAIN_WINDOW:
            return {r.key: r for r in runs}
    except FileNotFoundError:
        pass
    return {r.key: r for r in cmp.compute_default_runs()}


def benchmark(protocol: str = "walk_forward") -> cmp.ModelRun:
    if protocol == "walk_forward":
        return default_runs()[cmp.BENCHMARK_KEY]
    return custom_run("improved", "historical_mean", "{}", tuned=False, grid_json="{}", protocol="holdout")


def custom_key(system: str, model_type: str, params_json: str, tuned: bool, grid_json: str, protocol: str) -> str:
    digest = hashlib.md5(f"{params_json}|{tuned}|{grid_json}|{protocol}".encode()).hexdigest()[:6]
    return f"{system}:{model_type}#{digest}"


@st.cache_resource(show_spinner=False)
def custom_run(
    system: str, model_type: str, params_json: str, tuned: bool, grid_json: str, protocol: str,
) -> cmp.ModelRun:
    """Run one model with user-chosen settings (cached per settings)."""
    params, grid = json.loads(params_json), json.loads(grid_json)
    key = custom_key(system, model_type, params_json, tuned, grid_json, protocol)
    if system == "standard":
        spec = cmp.standard_spec(ModelCofig(model_type=model_type, random_state=SEED, **params), key=key)
    else:
        preprocessing = [] if model_type == "historical_mean" else ["standardize"]
        model_cfg = ImprovedModelConfig(name=model_type, model_type=model_type, preprocessing=preprocessing,
                                        params=params, grid=grid if tuned else {})
        spec = cmp.improved_spec(model_cfg, cmp.improved_experiment([model_cfg]), key=key)
    X, y, dates = load_data()
    if protocol == "holdout":
        return cmp.run_holdout(spec, X, y, dates)
    return cmp.run_model(spec, X, y, dates)


def remember(run: cmp.ModelRun, label: str) -> None:
    """Make a custom walk-forward run available on the comparison screen."""
    if run.protocol == "walk_forward":
        st.session_state.setdefault("custom_runs", {})[run.key] = (run, label)


def available_runs() -> tuple[dict[str, cmp.ModelRun], dict[str, str]]:
    """All walk-forward runs (defaults + custom) and their display labels."""
    runs = dict(default_runs())
    labels = {k: label_for(r) for k, r in runs.items()}
    for key, (run, label) in st.session_state.get("custom_runs", {}).items():
        runs[key], labels[key] = run, label
    return runs, labels


def label_for(run: cmp.ModelRun) -> str:
    base = model_info(run.system, run.model_type)["label"]
    return f"{base} [{run.system}]"


def run_label(system: str, model_type: str, params: dict, tuned: bool) -> str:
    shown = ", ".join(f"{k}={v}" for k, v in params.items())
    suffix = " · tuned per fold" if tuned else ""
    return f"{model_info(system, model_type)['label']} [{system}, custom: {shown or 'defaults'}{suffix}]"
