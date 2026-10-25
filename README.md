# Local Risk Dynamics

[![tests](https://github.com/HarshDubey-12/local-risk-dynamics/actions/workflows/tests.yml/badge.svg)](https://github.com/HarshDubey-12/local-risk-dynamics/actions/workflows/tests.yml)

**Deterministic vs stochastic locality in linear models for financial risk.**

Financial markets are non-stationary: the relationship between risk factors and returns changes
across regimes, while a global linear regression assumes it never changes. This project asks:

> Can regime-dependent financial behaviour be approximated by many small, **local** linear models
> instead of one global model — with locality defined deterministically (kernel distance) or
> stochastically (kernel-weighted sampling)?

The proposed model, **Monte Carlo Local Linear Regression (MCLLR)**, samples neighbourhoods of
similar months with kernel-weighted probabilities, fits a linear model on each, and aggregates the
forecasts into a prediction and an uncertainty estimate. It is compared with global, deterministic-local
and stochastic-global alternatives on next-month US market excess returns, using the five
Fama–French factors as predictors.

## Key findings

- **As originally specified, MCLLR fails:** it is significantly worse than a ridge baseline and the
  historical-mean benchmark (−6.6% out-of-sample R²), and its 95% intervals cover only ~37% of outcomes.
- **Why:** unregularised local fits on noisy, weakly predictive data; locality measured on factor
  returns rather than on the market regime; uncertainty that ignores return noise.
- **The improved system fixes the mechanics:** local ridge shrinkage, standardised distances,
  predictive variance and nested walk-forward tuning make MCLLR the best return-forecasting model
  tested (+0.36% OOS R², calibrated intervals, +1.2% a year for a market-timing investor) — but the
  edge is borderline (Clark–West p = 0.055) and disappears with 120- or 360-month windows.
- **Locality pays off for risk:** forecasting next-month volatility (|return|), the regime-state
  MCLLR is the best model and significantly beats the historical mean (+4.65% OOS R²,
  Diebold–Mariano p = 0.001, Clark–West p < 0.001).
- Full results, robustness checks (5 seeds, 3 training windows) and figures:
  [`reports/README.md`](reports/README.md); discussion: [`docs/MODEL_ASSESSMENT.md`](docs/MODEL_ASSESSMENT.md).

## Repository structure

```
local-risk-dynamics/
├── README.md                     this file
├── docs/
│   ├── MODEL_ASSESSMENT.md       results, diagnosis, improvements, conclusions
│   ├── PIPELINE.md               how the standard and improved pipelines work
│   ├── APP.md                    dashboard screens and design decisions
│   └── DATA.md                   data source, processing and citations
├── data/raw/                     Fama–French 5-factor monthly data (Kenneth R. French Data Library)
├── lrd/                          Python package
│   ├── core/                     shared: data loading + validation, splits, model interface,
│   │                             metrics, significance tests, economic evaluation, tracking
│   ├── standard/                 original system: 7 models, configs, runners
│   ├── improved/                 improved system: train-only preparation, nested tuning,
│   │                             improved and regime-state MCLLR, pipeline
│   └── comparison.py             runs both systems on shared folds; summaries and tests
├── experiments/                  YAML configs (standard/, improved/)
├── main.py                       end-to-end CLI
├── app/dashboard.py              interactive Streamlit dashboard (screens in app/views/)
├── scripts/                      make_report.py, build_app_results.py, improvement_study.py
├── reports/                      generated tables and figures for the dissertation
├── notebooks/                    four notebooks, one per analysis chapter
└── tests/                        core/, standard/, improved/, CLI, comparison and dashboard tests
```

The original system is preserved exactly at git tag `v1-standard`.

## Installation

Python 3.11–3.13.

```bash
git clone https://github.com/HarshDubey-12/local-risk-dynamics.git
cd local-risk-dynamics
pip install -r requirements.txt          # pinned runtime dependencies
pip install -r requirements-dev.txt      # + pytest and Jupyter
```

## Reproducing the results

```bash
python -m pytest                          # test suite
python scripts/make_report.py             # every table and figure in reports/ (~10 min)
python scripts/build_app_results.py       # precomputed results for the dashboard (~1 min)
python main.py --system both --rolling    # both systems end to end via the CLI
streamlit run app/dashboard.py            # interactive dashboard
```

`reports/README.md` records the commit and data hash each report was generated from.

## Models

| Model | System | Locality | Uncertainty |
|---|---|---|---|
| Global OLS | standard | none | none |
| Ridge (α = 1) | standard | none | none |
| LWLR | standard | Gaussian kernel weights (deterministic) | none |
| Monte Carlo Linear | standard | none (random subsamples) | spread across fits |
| **MCLLR** (proposed) | standard | kernel-weighted sampling | spread across neighbourhoods |
| MCLLR without geometry | standard (ablation) | uniform sampling | spread across subsamples |
| MCLLR without stochasticity | standard (ablation) | m nearest months, kernel-weighted | none |
| Historical mean | improved (benchmark) | — | training standard deviation |
| Ridge (tuned) | improved | none | residual standard deviation |
| **MCLLR (improved)** | improved | kernel-weighted sampling on standardised factors | predictive (spread + residual variance) |
| **MCLLR (regime state)** | improved | kernel on market regime: trailing volatility and momentum | predictive |

### How MCLLR works

For each query month x₀:

1. **Kernel weights** to every training month, wᵢ = exp(−‖xᵢ − x₀‖² / 2τ²), computed relative
   to the nearest month so they never underflow.
2. **Sampling probabilities** pᵢ = wᵢ / Σⱼ wⱼ.
3. **K neighbourhoods** of m months drawn without replacement with probabilities p.
4. **A linear fit per neighbourhood** — plain OLS in the original model, ridge in the improved one —
   and a prediction at x₀.
5. **Aggregation:** the mean of the K predictions is the forecast; their spread (plus, in the improved
   model, the local residual variance) is the uncertainty.

Locality is enforced by *sampling*, not by weighting the regression. Limiting cases:

- **τ → ∞:** all months equally likely, so MCLLR becomes per-query subsampled OLS (the
  *no-geometry* ablation, equivalent in expectation to Monte Carlo Linear).
- **m = n:** every neighbourhood is the whole training window, so every fit is global OLS,
  whatever τ is.
- **τ → 0:** the m nearest months are almost always chosen, which approaches the
  *no-stochasticity* ablation (without its kernel weighting).

## Evaluation methodology

- **Walk-forward:** 240-month training window, 12-month test window, refit every year — the same
  folds for every model of both systems.
- **Benchmark:** the training-window historical mean; **OOS R²** = 1 − SSE(model)/SSE(benchmark).
- **Significance:** Clark–West (appropriate for the nested benchmark, one-sided) and
  Diebold–Mariano (two-sided).
- **Calibration:** coverage of 50–99% intervals, interval score, Gaussian log-likelihood.
- **Economic value:** a mean-variance investor (γ = 3, weights in [0, 1.5]) times the market with
  each forecast; certainty-equivalent return gain and Sharpe ratio versus the historical-mean investor.
- **Robustness:** five random seeds for stochastic models; 120/240/360-month training windows;
  sub-periods before and after 2005 and stress months.
- **No look-ahead:** preparation (standardisation) and hyperparameter tuning use each fold's training
  window only; tests verify that no test month reaches any fit.

## Data

Monthly Fama/French 5 Factors (2x3), Kenneth R. French Data Library, July 1963 – November 2025.
Features: the five factors at month *t*; target: the market excess return at month *t + 1* (or its
absolute value for the risk-forecasting experiment). The file is validated before use; validation
found and removed a duplicated 68-month block in the originally committed copy. Source, processing
and citations: [`docs/DATA.md`](docs/DATA.md).

## Notebooks

| Notebook | Contents |
|---|---|
| `notebooks/01_data_analysis.ipynb` | validation, distributions, regimes, predictability |
| `notebooks/02_standard_system.ipynb` | standard models, ablations, calibration |
| `notebooks/03_improved_system.ipynb` | improved and regime-state models, tuning stability |
| `notebooks/04_robustness_and_economics.ipynb` | seeds, windows, economic value, risk target |

Notebooks read the precomputed results (`app/data/default_run/`) and the report tables
(`reports/tables/`); run the reproduction commands above first.

## Dashboard

`streamlit run app/dashboard.py` opens four screens: project overview, data analysis, each model
(with live settings) and a comparison of any models from both systems. See [`docs/APP.md`](docs/APP.md).
To publish it, deploy the repository on [Streamlit Community Cloud](https://share.streamlit.io) with
main file `app/dashboard.py`.

## Licence and citation

Code: MIT licence (`LICENSE`). The Fama–French data is not covered by the licence; cite it as
described in [`docs/DATA.md`](docs/DATA.md).
