# Pipelines: Standard and Improved

The repository contains two modelling systems that share one core. This document
describes how each pipeline runs and where its code lives.

```
                         lrd/core  (shared, identical for both systems)
        ┌──────────────────────────────────────────────────────────────────┐
        │ data loading + validation → chronological split / walk-forward   │
        │ folds → model interface → metrics + significance tests → tracking │
        └──────────────────────────────────────────────────────────────────┘
                 ▲                                         ▲
     lrd/standard (original system)          lrd/improved (in development)
```

**Why a shared core?** A comparison between systems is only fair if both are tested
on the same months with the same metric definitions. Data handling, splitting and
evaluation therefore exist once, in `lrd/core`, and both systems call it.

---

## 1. Shared core (`lrd/core`)

| Step | Module | What it does |
|---|---|---|
| Load | `data/loader.py` | Reads the Fama–French CSV, keeps monthly `YYYYMM` rows, converts % to decimals, builds X (factors at *t*) and y (Mkt-RF at *t+1*), drops the last month |
| Cache / hash | `data/pipeline.py` | Wraps the data in a `Dataset` with an MD5 hash of the source file for reproducibility |
| Split | `data/preprocessing.py` | `time_series_split`: chronological train/test split, no shuffling |
| Walk-forward | `evaluation/rolling.py` | Fixed-length rolling train/test folds; refits a fresh model per fold |
| Model interface | `base.py` | `LocalRiskModel`: `fit`, `predict`, `predict_with_uncertainty`, `is_fitted`, `model_name`, `hyperparameters` |
| Evaluate | `evaluation/evaluator.py`, `metrics.py` | Accuracy metrics for all models; calibration metrics for models that return a standard deviation |
| Compare | `evaluation/significance.py` | Out-of-sample R² vs a benchmark, Diebold–Mariano test |
| Backtest | `evaluation/backtesting.py` | Sharpe, drawdown, Sortino, Calmar (needs fixes: see MODEL_ASSESSMENT §5.6) |
| Track | `tracking.py` | Archives config, results, dataset hash and git commit per run |
| Utilities | `utils/` | Kernels, sampling helpers, plotting |

---

## 2. Standard pipeline (`lrd/standard`)

The original system, kept as the reference implementation. Its exact pre-restructure
state is tagged `v1-standard`.

```
YAML config (experiments/standard/) → ExperimentConfig (validated)
  → ModelFactory → model.fit(train) → model.predict(test)
  → core Evaluator → metrics.csv / predictions.csv / figures / run log
```

| Component | Location |
|---|---|
| Config + validation | `lrd/standard/config.py` |
| Models: global OLS, ridge, LWLR, MC linear, MCLLR, 2 ablations | `lrd/standard/models/` |
| Hold-out runner | `lrd/standard/experiments/runner.py` |
| Walk-forward runner | `lrd/standard/experiments/rolling_runner.py` |
| Grid sweep / walk-forward tuning | `lrd/standard/experiments/sweep.py`, `tuning.py` |
| CLI | `main.py` |
| App | `app/streamlit_app.py` |

Characteristics: one preparation for all models (raw decimal returns), hyperparameters
fixed in the config, MCLLR uncertainty is epistemic only. Results and limitations are in
[MODEL_ASSESSMENT.md](MODEL_ASSESSMENT.md).

---

## 3. Improved pipeline (`lrd/improved`, in development)

Designed to fix the issues found in the assessment while keeping comparisons fair.

```
1. Data validation + analysis report         (once, shared core)
2. Shared chronological split / walk-forward folds   (once, shared core)
3. For each model, using TRAINING data only:
     a. model-specific preparation  (e.g. standardise factors, winsorise training features)
     b. hyperparameter tuning       (inner walk-forward on the training window)
     c. fit → predict on the test window
4. Shared evaluation: metrics, calibration, significance vs historical mean
5. Comparison: CLI report and Streamlit app (standard vs improved)
```

### Design rules

1. **One split for every model.** Model-specific preparation happens *after* the shared
   split, so all models are scored on identical months.
2. **Preparation is part of the model.** Each improved model learns its preparation
   (scaling statistics, clipping thresholds) inside `fit()` from the training window and
   only applies it in `predict()`. Walk-forward then relearns it in every fold, with no
   leakage from test data.
3. **Bias/variance is controlled by tuned hyperparameters** (bandwidth, ridge α,
   neighbourhood size), selected on training data only.
4. **Extreme months are kept.** Outliers are flagged in the data report, not deleted;
   crash months are real data and decide most model rankings.
5. **Metrics are never computed inside models.** The shared evaluator guarantees
   identical definitions across systems.

### Planned contents

| Module | Purpose | Source of evidence |
|---|---|---|
| `lrd/core/data/validation.py` | Missing values, junk rows, date gaps, outlier flags, summary report | — |
| `lrd/improved/preprocessing.py` | Train-only transformers (standardise, winsorise) | MODEL_ASSESSMENT §3 |
| `lrd/improved/models/mcllr.py` | MCLLR + local ridge + standardised distances + predictive variance | Validated: +0.35% OOS R², 94% coverage (§4) |
| `lrd/improved/tuning.py` | Nested walk-forward hyperparameter selection | §5.5 |
| `experiments/improved/*.yaml` | Configs for the improved system | — |
| `tests/improved/` | Tests for the above | — |
| `main.py --system` | Run standard, improved or both | — |

The validated changes are prototyped in `scripts/improvement_study.py`; the full roadmap
is in [MODEL_ASSESSMENT.md](MODEL_ASSESSMENT.md) §5.
