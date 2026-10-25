"""
Regenerate every table and figure used in the dissertation.

Writes to reports/:
    tables/main_results.{md,csv}     all default models on the shared walk-forward folds
    tables/seed_robustness.{md,csv}  stochastic models over several random seeds
    tables/window_sensitivity.{md,csv}  results for 120/240/360-month training windows
    tables/risk_target.{md,csv}      forecasting |next-month return| (volatility proxy)
    figures/*.png                    data overview, accuracy, cumulative advantage,
                                     calibration, economic value, windows, risk target
    README.md                        index with the main tables, data hash and git commit

Run from anywhere (about 10 minutes):
    python scripts/make_report.py
    python scripts/make_report.py --quick   # 2 seeds, no window sensitivity (for checks)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.catalog import MODELS  # noqa: E402  (plain dict of model labels; no Streamlit import)
from lrd import comparison as cmp  # noqa: E402
from lrd.core.data.loader import load_fama_french  # noqa: E402
from lrd.improved.pipeline import ImprovedPipeline  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports"
TABLES, FIGURES = REPORT_DIR / "tables", REPORT_DIR / "figures"
RISK_CONFIG = "experiments/improved/config_risk.yaml"
SEEDS = [0, 1, 2, 3, 4]
WINDOWS = [120, 240, 360]
STOCHASTIC = ["standard:mc_linear", "standard:mcllr", "standard:mcllr_no_geometry",
              "improved:mcllr_improved", "improved:mcllr_regime"]
HIGHLIGHT = ["standard:global_linear", "standard:ridge_linear", "standard:mcllr",
             "improved:ridge", "improved:mcllr_improved", "improved:mcllr_regime"]
COLORS = {"standard": "#f58518", "improved": "#4c78a8"}


def label(key: str) -> str:
    return MODELS[key.split("#")[0]]["label"]


def markdown_table(frame: pd.DataFrame, formats: dict) -> str:
    """Markdown table without optional dependencies; NaN shown as '–'."""
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append("–" if isinstance(v, float) and np.isnan(v) else (formats[c].format(v) if c in formats else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_table(name: str, frame: pd.DataFrame, formats: dict) -> str:
    frame.to_csv(TABLES / f"{name}.csv", index=False)
    text = markdown_table(frame, formats)
    (TABLES / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    return text


def git_commit() -> str:
    """Short HEAD hash, marked if the code had uncommitted changes (reports/ itself excluded)."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", ":!reports"], stderr=subprocess.DEVNULL).decode().strip()
        return f"{head} + uncommitted changes" if dirty else head
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def default_runs(log) -> dict:
    try:
        runs, meta = cmp.load_runs(PROJECT_ROOT / "app" / "data" / "default_run")
        if meta.get("data_hash") == cmp.data_hash() and {r.key for r in runs} == {s.key for s in cmp.default_specs()}:
            log("  using precomputed default runs (app/data/default_run)")
            return {r.key: r for r in runs}
    except FileNotFoundError:
        pass
    log("  computing default runs")
    return {r.key: r for r in cmp.compute_default_runs(progress=lambda k: log(f"    {k}"))}


MAIN_COLUMNS = {
    "label": "Model", "system": "System", "rmse": "RMSE", "oos_r2_%": "OOS R² (%)", "dm_p": "DM p",
    "cw_p": "CW p", "coverage_95": "95% cov.", "interval_score_95": "Int. score", "cer_gain_%": "CER gain (%/yr)",
    "sharpe": "Sharpe", "folds_won": "Folds won",
}
MAIN_FORMATS = {"RMSE": "{:.4f}", "OOS R² (%)": "{:+.2f}", "DM p": "{:.3f}", "CW p": "{:.3f}", "95% cov.": "{:.1%}",
                "Int. score": "{:.4f}", "CER gain (%/yr)": "{:+.2f}", "Sharpe": "{:.2f}", "Folds won": "{:d}"}


def main_results(runs: dict) -> pd.DataFrame:
    summary = cmp.summarize(list(runs.values()), runs[cmp.BENCHMARK_KEY])
    summary["label"] = summary["key"].map(label)
    return summary.sort_values("oos_r2_%", ascending=False).reset_index(drop=True)


def seed_robustness(log, seeds) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        log(f"  seed {seed}")
        runs = cmp.compute_default_runs(seed=seed, keys=STOCHASTIC + [cmp.BENCHMARK_KEY])
        by_key = {r.key: r for r in runs}
        bench = by_key[cmp.BENCHMARK_KEY]
        summary = cmp.summarize([by_key[k] for k in STOCHASTIC], bench)
        summary["seed"] = seed
        rows.append(summary)
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(TABLES / "seed_robustness_raw.csv", index=False)
    agg = raw.groupby("key").agg(
        r2_mean=("oos_r2_%", "mean"), r2_std=("oos_r2_%", "std"), r2_min=("oos_r2_%", "min"),
        r2_max=("oos_r2_%", "max"), cw_p_mean=("cw_p", "mean"), cov_mean=("coverage_95", "mean"),
        cer_mean=("cer_gain_%", "mean"),
    ).reindex(STOCHASTIC).reset_index()
    agg.insert(0, "Model", agg["key"].map(label))
    return agg.drop(columns="key")


def window_sensitivity(log, windows, default) -> pd.DataFrame:
    rows = []
    for window in windows:
        log(f"  training window {window}")
        runs = default if window == cmp.TRAIN_WINDOW else {
            r.key: r for r in cmp.compute_default_runs(train_window=window)}
        summary = cmp.summarize(list(runs.values()), runs[cmp.BENCHMARK_KEY])
        summary["window"] = window
        summary["test_months"] = len(runs[cmp.BENCHMARK_KEY].dates)
        rows.append(summary)
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(TABLES / "window_sensitivity_raw.csv", index=False)
    table = raw.pivot(index="key", columns="window", values="oos_r2_%")
    table.columns = [f"R² train={w} (%)" for w in table.columns]
    cw = raw.pivot(index="key", columns="window", values="cw_p")
    cw.columns = [f"CW p train={w}" for w in cw.columns]
    out = table.join(cw).reset_index()
    out.insert(0, "Model", out["key"].map(label))
    order = [s.key for s in cmp.default_specs()]
    return out.set_index("key").reindex(order).reset_index(drop=True)


def risk_target(log):
    log("  improved pipeline with target = |next-month return|")
    results = ImprovedPipeline(RISK_CONFIG).run(log=lambda m: log("  " + m.strip()))
    summary = results.summary.copy()
    keep = ["model", "rmse", "mae", "oos_r2_%", "dm_p_vs_historical_mean", "cw_p_vs_historical_mean",
            "coverage_95", "interval_score_95"]
    return summary[keep].rename(columns={
        "model": "Model", "rmse": "RMSE", "mae": "MAE", "oos_r2_%": "OOS R² (%)",
        "dm_p_vs_historical_mean": "DM p", "cw_p_vs_historical_mean": "CW p",
        "coverage_95": "95% cov.", "interval_score_95": "Int. score"}), results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)


def fig_data_overview() -> None:
    X, _, dates = load_fama_french(cmp.DATA_PATH)
    market = pd.Series(X[:, 0], index=dates)
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    a.bar(dates, market, width=25, color="#4c78a8")
    a.set_ylabel("Mkt-RF (monthly)")
    a.set_title("Market excess return and its volatility regimes")
    vol = market.rolling(12).std() * np.sqrt(12)
    b.plot(dates, vol, color="#e45756")
    b.fill_between(dates, 0, vol, where=vol > vol.quantile(0.8), color="#e45756", alpha=0.2)
    b.set_ylabel("12-month volatility (ann.)")
    save(fig, "fig1_data_overview.png")


def fig_accuracy(main: pd.DataFrame, seeds: pd.DataFrame | None) -> None:
    frame = main.iloc[::-1]
    err = np.zeros(len(frame))
    if seeds is not None:
        spread = dict(zip(seeds["Model"], seeds["r2_std"]))
        err = np.array([spread.get(l, 0.0) for l in frame["label"]])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(frame["label"], frame["oos_r2_%"], xerr=err, color=[COLORS[s] for s in frame["system"]], capsize=3)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("Out-of-sample R² vs historical mean (%)  [bars: ±1 sd across seeds]")
    ax.set_title("Forecast accuracy (orange = standard, blue = improved)")
    save(fig, "fig2_oos_r2.png")


def fig_cumulative(runs: dict) -> None:
    bench = runs[cmp.BENCHMARK_KEY]
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in HIGHLIGHT:
        adv = cmp.cumulative_advantage(runs[key], bench)
        ax.plot(adv.index, adv.values, label=label(key), linestyle="-" if key.startswith("improved") else "--")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Cumulative SSE advantage over historical mean")
    ax.set_title("When models beat the benchmark (rising = better)")
    ax.legend(fontsize=8)
    save(fig, "fig3_cumulative_advantage.png")


def fig_calibration(runs: dict) -> None:
    levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0.5, 0.99], [0.5, 0.99], color="gray", linestyle="--", label="perfect calibration")
    for key, run in runs.items():
        if run.has_uncertainty:
            ax.plot(levels, [cmp.coverage(run, lv) for lv in levels], marker="o", label=label(key))
    ax.set_xlabel("Nominal interval level")
    ax.set_ylabel("Actual coverage")
    ax.set_title("Calibration of predictive intervals")
    ax.legend(fontsize=7)
    save(fig, "fig4_calibration.png")


def fig_economic(main: pd.DataFrame) -> None:
    frame = main.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(frame["label"], frame["cer_gain_%"], color=[COLORS[s] for s in frame["system"]])
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("CER gain over historical-mean investor (% per year, γ = 3)")
    ax.set_title("Economic value of the forecasts")
    save(fig, "fig5_economic_value.png")


def fig_windows(windows: pd.DataFrame) -> None:
    cols = [c for c in windows.columns if c.startswith("R²")]
    keep = windows[windows["Model"].isin([label(k) for k in HIGHLIGHT])]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = [int(c.split("=")[1].split()[0]) for c in cols]
    for _, row in keep.iterrows():
        ax.plot(x, [row[c] for c in cols], marker="o", label=row["Model"])
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Training window (months)")
    ax.set_ylabel("OOS R² vs historical mean (%)")
    ax.set_title("Sensitivity to the training window")
    ax.legend(fontsize=8)
    save(fig, "fig6_window_sensitivity.png")


def fig_risk(results) -> None:
    preds = results.predictions
    bench_err = preds["y_true"] - preds["historical_mean_pred"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in results.summary["model"]:
        if name == "historical_mean":
            continue
        err = preds["y_true"] - preds[f"{name}_pred"]
        ax.plot(preds["date"], np.cumsum(bench_err**2 - err**2), label=name)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Cumulative SSE advantage over historical mean")
    ax.set_title("Risk target |next-month return|: when models beat the benchmark")
    ax.legend(fontsize=8)
    save(fig, "fig7_risk_target.png")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="2 seeds and no window sensitivity (fast check).")
    args = parser.parse_args()
    seeds = SEEDS[:2] if args.quick else SEEDS
    windows = [] if args.quick else WINDOWS

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    log = lambda m: print(m, flush=True)  # noqa: E731

    log("[1/5] Main results")
    runs = default_runs(log)
    main_df = main_results(runs)
    main_md = save_table("main_results", main_df[list(MAIN_COLUMNS)].rename(columns=MAIN_COLUMNS), MAIN_FORMATS)

    log(f"[2/5] Seed robustness ({len(seeds)} seeds)")
    seeds_df = seed_robustness(log, seeds)
    seeds_md = save_table("seed_robustness", seeds_df, {
        "r2_mean": "{:+.2f}", "r2_std": "{:.2f}", "r2_min": "{:+.2f}", "r2_max": "{:+.2f}",
        "cw_p_mean": "{:.3f}", "cov_mean": "{:.1%}", "cer_mean": "{:+.2f}"})

    windows_md = "_Skipped (--quick)._"
    windows_df = None
    if windows:
        log(f"[3/5] Window sensitivity {windows}")
        windows_df = window_sensitivity(log, windows, runs)
        windows_md = save_table("window_sensitivity", windows_df,
                                {c: ("{:+.2f}" if c.startswith("R²") else "{:.3f}") for c in windows_df.columns if c != "Model"})
    else:
        log("[3/5] Window sensitivity skipped")

    log("[4/5] Risk target")
    risk_df, risk_results = risk_target(log)
    risk_md = save_table("risk_target", risk_df, {"RMSE": "{:.4f}", "MAE": "{:.4f}", "OOS R² (%)": "{:+.2f}",
                                                 "DM p": "{:.3f}", "CW p": "{:.3f}", "95% cov.": "{:.1%}",
                                                 "Int. score": "{:.4f}"})

    log("[5/5] Figures")
    fig_data_overview()
    fig_accuracy(main_df, seeds_df)
    fig_cumulative(runs)
    fig_calibration(runs)
    fig_economic(main_df)
    if windows_df is not None:
        fig_windows(windows_df)
    fig_risk(risk_results)

    bench = runs[cmp.BENCHMARK_KEY]
    (REPORT_DIR / "README.md").write_text(f"""# Dissertation results

Generated by `python scripts/make_report.py` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC
from commit `{git_commit()}`, data hash `{cmp.data_hash()}`.

Protocol: walk-forward, {cmp.TRAIN_WINDOW}-month training window, {cmp.TEST_WINDOW}-month test window,
{len(bench.folds)} folds, {len(bench.dates)} test months ({bench.dates[0]:%b %Y} – {bench.dates[-1]:%b %Y}).
Benchmark: historical mean. OOS R² = 1 − SSE(model)/SSE(benchmark).
DM = Diebold–Mariano (two-sided); CW = Clark–West (one-sided, model better than the nested benchmark).
CER gain: annualised certainty-equivalent return gain of a mean-variance investor (γ = 3, weights in [0, 1.5],
trailing 60-month variance) over the same investor using the historical mean.

## Main results
{main_md}

## Seed robustness (stochastic models, seeds {seeds})
{seeds_md}

## Window sensitivity (OOS R² and Clark–West p per training window)
{windows_md}

## Risk target: forecasting |next-month market excess return|
{risk_md}

## Figures
""" + "\n".join(f"![{p.stem}](figures/{p.name})" for p in sorted(FIGURES.glob("*.png"))) + "\n", encoding="utf-8")
    log(f"Done in {time.perf_counter() - start:.0f}s -> {REPORT_DIR}")


if __name__ == "__main__":
    main()
