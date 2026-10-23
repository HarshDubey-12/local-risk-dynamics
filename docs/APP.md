# Dashboard

An interactive Streamlit dashboard covering both modelling systems.

```bash
streamlit run app/dashboard.py          # http://localhost:8501
python scripts/build_app_results.py     # rebuild the precomputed default results (~40 s)
```

Deploy on [Streamlit Community Cloud](https://share.streamlit.io) with **main file
`app/dashboard.py`**; dependencies come from `requirements.txt`.

---

## Design decisions

| Decision | Choice |
|---|---|
| Evaluation | **Walk-forward on shared folds** (240-month train, 12-month test; 42 folds, 504 test months) for every model of both systems, so all numbers are directly comparable. A hold-out 80/20 split is available as a secondary view for standard models on the Models screen. |
| Speed | Default results are **precomputed** in `app/data/default_run/` and load instantly. They store the data file's hash; if the data changes, the dashboard recomputes them on first load (and `tests/test_comparison.py` fails until they are rebuilt). |
| Interactivity | Every model can be **re-run with custom settings** on the Models screen. Improved models can use fixed values or be **tuned per fold** over user-chosen candidates. Results are cached per setting, and custom walk-forward runs appear on the Comparison screen. |
| Benchmark | The training-window **historical mean**; OOS R² and Diebold–Mariano tests are relative to it. |

---

## Screens

### 1. Project overview (`app/views/overview.py`)
- **Objective:** research question, task and a headline result computed from the default runs.
- **Methodology:** pipeline diagram (shared core, standard and improved systems) and tabs on walk-forward
  evaluation (fold timeline), the benchmark and OOS R², the Diebold–Mariano test, nested tuning and calibration.
- **Models:** a card for each of the ten models with its family, locality, uncertainty and OOS R²; **Open model →**
  jumps to that model on screen 3.
- **Practices applied:** chronological evaluation, shared folds, train-only preparation, honest tuning, outliers kept,
  benchmark and significance, calibration, validated data.

### 2. Data analysis (`app/views/data_analysis.py`)
- **Data quality checks** from `lrd.core.data.validation`: status, rows, missing, garbage, duplicate and missing
  months, ordering; the data-correction note and the full validation report.
- Controls: factors and year range. Tabs:
  - **Time series & outliers** — adjustable robust z-score threshold; flagged months circled and listed.
  - **Distributions** — histogram vs normal curve; mean, std, skew, excess kurtosis, extremes.
  - **Correlations** — heatmap of factors and the next-month target.
  - **Volatility regimes** — rolling annualised volatility with high-volatility months shaded.
  - **Predictability** — factor vs next-month target scatter (r, R²) and target autocorrelation with noise band.
  - **Evaluation folds** — fold timeline for adjustable windows.

### 3. Models (`app/views/models.py`)
- Switch between all ten models (standard and improved).
- **Explanation:** system, family, locality, uncertainty, idea, algorithm steps, formula, hyperparameters,
  strengths, limitations, cost.
- **Results:** default (precomputed) or custom settings. Metric tiles (RMSE, OOS R² vs mean, DM p-value,
  95% coverage, interval score) and tabs:
  - **Forecasts** — forecast vs actual over time with the 95% band.
  - **Diagnostics** — forecast vs actual scatter, residuals over time and their distribution.
  - **Per fold** — fold RMSE against the benchmark.
  - **Calibration** — actual vs nominal coverage at 50–99%.
  - **Tuning** — hyperparameters chosen in each fold (tuned improved models).
  - **Locality explorer** — kernel weights, effective sample size and influential months for any test month and
    bandwidth (local models).
  - **Original vs improved** — head-to-head table and significance test (MCLLR pages).

### 4. Comparison (`app/views/comparison.py`)
- Select any default or custom runs from either system.
- **Leaderboard:** RMSE, OOS R², DM p-value, 95% coverage, interval score, Gaussian NLL, folds won.
- Tabs: **Accuracy** (ranked OOS R² and RMSE), **Significance** (pairwise Diebold–Mariano heatmap),
  **Over time** (cumulative squared-error advantage over the benchmark), **Calibration** (coverage curves),
  **Sub-periods** (before 2005, 2005 onwards, stress months), **Folds** (folds won, per-fold RMSE),
  **Standard vs improved** (MCLLR and Ridge head-to-head).

---

## Code layout

| File | Purpose |
|---|---|
| `app/dashboard.py` | Entry point: page config and navigation |
| `app/views/*.py` | One module per screen, each with `render()` |
| `app/catalog.py` | Model descriptions, practices and metric definitions |
| `app/engine.py` | Cached data, validation, default and custom runs |
| `app/charts.py` | Shared Altair charts and the highlighted table helper |
| `app/data/default_run/` | Precomputed default results (`scripts/build_app_results.py`) |
| `lrd/comparison.py` | Runs any model of either system on the shared folds; summaries, pairwise tests, persistence |
| `tests/test_dashboard.py`, `tests/test_comparison.py` | Screen smoke/interaction tests and engine tests |
