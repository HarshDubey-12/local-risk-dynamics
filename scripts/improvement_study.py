"""
Walk-forward study behind MODEL_ASSESSMENT.md.

Compares the repository's models against the historical-mean benchmark and
tests candidate MCLLR improvements (local ridge shrinkage, standardised
distances, larger neighbourhoods, predictive variance, recency locality).

Run from the project root (takes ~2 minutes):
    python scripts/improvement_study.py
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.data.loader import load_fama_french  # noqa: E402
from src.models.lwlr import LocallyWeightedLinearRegression  # noqa: E402
from src.models.mc_linear import MonteCarloLinearRegression  # noqa: E402
from src.models.mcllr import MonteCarloLocalLinearRegression  # noqa: E402

TRAIN_WINDOW, TEST_WINDOW = 240, 12
SPLIT_DATE = np.datetime64("2005-01-01")


def design(A: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(A), 1)), A])


def ridge_fit(A: np.ndarray, b: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """(Ridge) least squares with an unpenalised intercept; alpha=0 is OLS."""
    D = design(A)
    penalty = alpha * np.eye(D.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(D.T @ D + penalty + 1e-12 * np.eye(D.shape[1]), D.T @ b)


def improved_mcllr(
    X_train, y_train, X_query, rng, bandwidth, n_simulations=50, subsample_size=80,
    alpha=0.0, standardize=False, recency=None, predictive_std=False,
):
    """MCLLR with optional improvements.

    alpha           ridge penalty per local fit, scaled by subsample_size
    standardize     kernel distances on z-scored factors (bandwidth in std units)
    recency         sample by recency (half-life-like scale in months) instead of geometry
    predictive_std  add the local residual variance to the Monte Carlo spread
    """
    if standardize:
        mu, sd = X_train.mean(0), X_train.std(0)
        Z, Zq = (X_train - mu) / sd, (X_query - mu) / sd
    else:
        Z, Zq = X_train, X_query
    n = len(y_train)
    means, stds = [], []
    for z0, x0 in zip(Zq, X_query):
        if recency is not None:
            log_w = -(n - 1 - np.arange(n)) / recency
        else:
            sq_dist = ((Z - z0) ** 2).sum(1)
            log_w = -(sq_dist - sq_dist.min()) / (2 * bandwidth**2)
        p = np.exp(log_w - log_w.max())
        p /= p.sum()
        if np.count_nonzero(p) < subsample_size:
            p = np.maximum(p, 1e-12)
            p /= p.sum()

        preds, resid_var = [], []
        for _ in range(n_simulations):
            idx = rng.choice(n, subsample_size, replace=False, p=p)
            beta = ridge_fit(X_train[idx], y_train[idx], alpha * subsample_size)
            preds.append(design(x0[None])[0] @ beta)
            if predictive_std:
                resid = y_train[idx] - design(X_train[idx]) @ beta
                resid_var.append(resid.var(ddof=X_train.shape[1] + 1))
        preds = np.array(preds)
        means.append(preds.mean())
        stds.append(np.sqrt(preds.var() + (np.mean(resid_var) if predictive_std else 0.0)))
    return np.array(means), np.array(stds)


def point(pred):
    return pred, None


MODELS = {
    # --- benchmarks and repository models ---
    "Historical mean": lambda a, b, q, r: point(np.full(len(q), b.mean())),
    "Global OLS": lambda a, b, q, r: point(design(q) @ ridge_fit(a, b)),
    "Ridge (alpha=1, repo)": lambda a, b, q, r: point(design(q) @ ridge_fit(a, b, 1.0)),
    "Ridge (alpha=0.02n, tuned)": lambda a, b, q, r: point(design(q) @ ridge_fit(a, b, 0.02 * len(b))),
    "MC Linear (repo)": lambda a, b, q, r: MonteCarloLinearRegression(50, 80, random_state=0).fit(a, b).predict(q, return_std=True),
    "LWLR (repo, bw 0.08)": lambda a, b, q, r: point(LocallyWeightedLinearRegression(0.08).fit(a, b).predict(q)),
    "MCLLR (repo, bw 0.08, m 80)": lambda a, b, q, r: MonteCarloLocalLinearRegression(0.08, 50, 80, random_state=0).fit(a, b).predict(q, return_std=True),
    # --- improvement candidates ---
    "MCLLR, m=160": lambda a, b, q, r: improved_mcllr(a, b, q, r, 0.08, subsample_size=160),
    "MCLLR, standardised bw=2, m=160": lambda a, b, q, r: improved_mcllr(a, b, q, r, 2.0, subsample_size=160, standardize=True),
    "Bagged ridge (no locality)": lambda a, b, q, r: improved_mcllr(a, b, q, r, 1e6, subsample_size=160, standardize=True, alpha=0.05),
    "MCLLR + local ridge (bw=2)": lambda a, b, q, r: improved_mcllr(a, b, q, r, 2.0, subsample_size=160, standardize=True, alpha=0.05),
    "MCLLR + local ridge (bw=1)": lambda a, b, q, r: improved_mcllr(a, b, q, r, 1.0, subsample_size=160, standardize=True, alpha=0.05),
    "MCLLR + local ridge + predictive std": lambda a, b, q, r: improved_mcllr(a, b, q, r, 2.0, subsample_size=160, standardize=True, alpha=0.05, predictive_std=True),
    "Recency-local MC + ridge + predictive std": lambda a, b, q, r: improved_mcllr(a, b, q, r, None, subsample_size=120, alpha=0.05, recency=60, predictive_std=True),
}


def diebold_mariano(e_model: np.ndarray, e_ref: np.ndarray) -> tuple[float, float]:
    """DM test, squared loss, h=1. Negative t means the model beats the reference."""
    d = e_model**2 - e_ref**2
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(t), math.erfc(abs(t) / math.sqrt(2))  # two-sided normal p-value


def r2_vs(errors: np.ndarray, benchmark_errors: np.ndarray) -> float:
    return 100 * (1 - np.sum(errors**2) / np.sum(benchmark_errors**2))


def main() -> None:
    X, y, dates = load_fama_french(PROJECT_ROOT / "data/raw/F-F_Research_Data_5_Factors_2x3.csv")
    folds = [
        (s, s + TRAIN_WINDOW, s + TRAIN_WINDOW + TEST_WINDOW)
        for s in range(0, len(y) - TRAIN_WINDOW - TEST_WINDOW + 1, TEST_WINDOW)
    ]
    y_test = np.concatenate([y[te:end] for _, te, end in folds])
    test_dates = np.concatenate([dates[te:end] for _, te, end in folds])

    errors, rows = {}, []
    for name, model in MODELS.items():
        start, rng = time.perf_counter(), np.random.default_rng(0)
        preds, stds = [], []
        for s, te, end in folds:
            p, sd = model(X[s:te], y[s:te], X[te:end], rng)
            preds.append(p)
            stds.append(sd)
        e = y_test - np.concatenate(preds)
        errors[name] = e
        coverage = np.mean(np.abs(e) <= 1.96 * np.concatenate(stds)) if stds[0] is not None else np.nan
        rows.append({"model": name, "rmse": np.sqrt(np.mean(e**2)), "coverage_95": coverage,
                     "seconds": time.perf_counter() - start})
        print(f"  {name:<45} done", flush=True)

    res = pd.DataFrame(rows)
    hist = errors["Historical mean"]
    pre = test_dates < SPLIT_DATE
    worst = np.argmax(np.abs(y_test - y_test.mean()))
    keep = np.arange(len(y_test)) != worst

    res["r2_oos_%"] = [r2_vs(errors[m], hist) for m in res.model]
    res["r2_pre2005_%"] = [r2_vs(errors[m][pre], hist[pre]) for m in res.model]
    res["r2_post2005_%"] = [r2_vs(errors[m][~pre], hist[~pre]) for m in res.model]
    res["r2_ex_worst_%"] = [r2_vs(errors[m][keep], hist[keep]) for m in res.model]
    for ref, col in [("Ridge (alpha=1, repo)", "dm_vs_ridge"), ("Historical mean", "dm_vs_mean")]:
        tests = [diebold_mariano(errors[m], errors[ref]) for m in res.model]
        res[f"{col}_t"] = [t for t, _ in tests]
        res[f"{col}_p"] = [p for _, p in tests]

    print(f"\n{len(folds)} folds, {len(y_test)} test months "
          f"({pd.Timestamp(test_dates[0]):%Y-%m} to {pd.Timestamp(test_dates[-1]):%Y-%m}); "
          f"largest move excluded in r2_ex_worst: {pd.Timestamp(test_dates[worst]):%b %Y} ({y_test[worst]:.1%})")
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
