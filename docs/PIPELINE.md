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
     lrd/standard (original system)          lrd/improved (improved system)
```

**Why a shared core?** A comparison between systems is only fair if both are tested
on the same months with the same metric definitions. Data handling, splitting and
evaluation therefore exist once, in `lrd/core`, and both systems call it.

---

## 1. Shared core (`lrd/core`)

| Step | Module | What it does |
|---|---|---|
| Validate | `data/validation.py` | Checks the raw file for missing/garbage values, duplicate or missing months and ordering; flags extreme values (never removes them); writes a Markdown report |
| Load | `data/loader.py` | Reads the Fama–French CSV, keeps monthly `YYYYMM` rows, converts % to decimals, builds X (factors at *t*) and y (Mkt-RF at *t+1*), drops the last month; refuses duplicated or out-of-order months |
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
| Dashboard | `app/dashboard.py` (see [APP.md](APP.md)) |

Characteristics: one preparation for all models (raw decimal returns), hyperparameters
fixed in the config, MCLLR uncertainty is epistemic only. Results and limitations are in
[MODEL_ASSESSMENT.md](MODEL_ASSESSMENT.md).

---

## 3. Improved pipeline (`lrd/improved`)

Fixes the issues found in the assessment while keeping comparisons fair.

```
1. Data validation + report                     (once, lrd.core.data.validation)
2. Shared walk-forward folds                    (once, lrd.core.evaluation.rolling)
3. For each model, inside every fold, on TRAINING data only:
     a. model-specific preparation  (PreparedModel: standardise / winsorise)
     b. hyperparameter tuning       (TunedModel: inner walk-forward on the training window)
     c. fit → predict on the test window (point forecast + predictive std)
4. Shared evaluation: metrics, calibration, OOS R² and Diebold–Mariano vs the benchmark
5. Outputs: summary, folds, predictions, tuning log, validation report
```

Run it with:

```bash
python main.py --system improved            # outputs in results/improved/
python main.py --system both --rolling      # standard and improved in one run
```

### Design rules

1. **One split for every model.** Model-specific preparation happens *after* the shared
   split, so all models are scored on identical months.
2. **Preparation is part of the model.** `PreparedModel` learns its preparation (scaling
   statistics, clipping thresholds) inside `fit()` from the training window and only
   applies it in `predict()`. Walk-forward relearns it in every fold, with no leakage.
3. **Bias/variance is controlled by tuned hyperparameters** (bandwidth, ridge α).
   `TunedModel` selects them by an inner walk-forward on training data only.
4. **Extreme months are kept.** Outliers are flagged in the validation report, not
   deleted; crash months are real data and decide most model rankings.
5. **Metrics are never computed inside models.** The shared evaluator guarantees
   identical definitions across systems.
6. **Every model reports a predictive standard deviation**, so all are scored on
   calibration as well as accuracy.

### Components

| Module | Purpose |
|---|---|
| `lrd/core/data/validation.py` | Raw-data checks and Markdown report (shared) |
| `lrd/improved/preprocessing.py` | `Standardizer`, `Winsorizer` (train-only) and `PreparedModel` |
| `lrd/improved/models/benchmark.py` | `HistoricalMean`: the benchmark every model must beat |
| `lrd/improved/models/ridge.py` | `RidgeRegression` with residual-based predictive std |
| `lrd/improved/models/mcllr.py` | `ImprovedMCLLR`: local ridge, Gumbel top-k sampling, batched fits, predictive std |
| `lrd/improved/models/regime.py` | `RegimeMCLLR`: kernel on the market regime (trailing 12-month volatility and momentum) |
| `experiments/improved/config_risk.yaml` | Risk-forecasting experiment: target = absolute next-month return |
| `lrd/core/evaluation/backtesting.py` | Portfolio statistics (√12 annualisation) and market-timing economic value |
| `lrd/core/evaluation/significance.py` | OOS R², Diebold–Mariano and Clark–West tests |
| `lrd/improved/tuning.py` | `TunedModel`: nested walk-forward hyperparameter selection |
| `lrd/improved/config.py` | YAML schema, validation and model construction |
| `lrd/improved/pipeline.py` | `ImprovedPipeline` and `ImprovedResults` |
| `experiments/improved/config_improved.yaml` | Default improved experiment |
| `tests/improved/`, `tests/core/test_validation.py` | Tests, including leakage checks |

### Config format

```yaml
dataset_path: data/raw/F-F_Research_Data_5_Factors_2x3.csv
random_state: 42
benchmark: historical_mean          # OOS R², DM and CW tests are relative to this model
target: return                      # or abs_return (|next-month return|, a volatility proxy)
evaluation: {train_window: 240, test_window: 12}
tuning: {inner_train_window: 180, inner_test_window: 12}
validation: {strict: true, outlier_threshold: 5.0}
models:
  - name: mcllr_improved            # unique label used in outputs
    model_type: mcllr_improved      # historical_mean | ridge | mcllr_improved | mcllr_regime
    preprocessing: [standardize]    # or {winsorize: 0.01}
    params: {n_simulations: 50, subsample_size: 160}   # fixed
    grid: {bandwidth: [1, 2, 3], alpha: [1, 3, 10, 30]} # tuned per fold
```

Results are in [MODEL_ASSESSMENT.md](MODEL_ASSESSMENT.md) §5; remaining work is in §6.
