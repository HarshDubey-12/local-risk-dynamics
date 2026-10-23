Local Risk Dynamics

Deterministic vs Stochastic Locality in Linear Models for Financial Risk

Overview
Financial markets are inherently non-stationary: relationships between risk factors and returns evolve across time, regimes, and uncertainty conditions. Traditional global linear regression assumes constant factor sensitivities, which often leads to model mis-specification in real-world financial environments.

This project studies an alternative paradigm:

Can complex, regime-dependent financial risk behavior be approximated using ensembles of small, local linear models instead of a single global model?

We investigate this through a structured comparison of four linear modeling philosophies that differ in how they define:

- Locality
- Uncertainty

Core Research Idea
All models in this repository lie on a conceptual spectrum:

Stationary → Deterministic Local → Stochastic Global → Stochastic Local

The central hypothesis:

Non-linear financial dynamics can be effectively approximated using collections of locally valid linear models, where locality may be enforced deterministically (distance-based) or stochastically (sampling-based).

Quick Start
Installation

# Clone repository
git clone https://github.com/HarshDubey-12/local-risk-dynamics.git
cd local-risk-dynamics

# Install dependencies
pip install -r requirements.txt

Results and Model Assessment

As originally specified, the proposed MCLLR does not outperform the baselines: no model
significantly beats the historical-mean forecast, and MCLLR is significantly worse than
Ridge in walk-forward evaluation. The improved system (`lrd/improved`) adds data
validation, train-only standardisation, local ridge shrinkage, predictive variance and
nested walk-forward tuning; its MCLLR is the best model tested (+0.36% out-of-sample R²
against the historical mean, best interval score, 94% coverage of 95% intervals), though
the edge is not statistically significant. See
[docs/MODEL_ASSESSMENT.md](docs/MODEL_ASSESSMENT.md) for the full results, the diagnosis
and the remaining roadmap.

Data validation also found and removed a duplicated block of 68 months in the raw CSV;
all results use the corrected data (the original state is preserved in tag `v1-standard`).

Dashboard

```bash
streamlit run app/dashboard.py
```

Opens at http://localhost:8501 with four interactive screens covering both systems:
**Project overview** (objective, methodology, every model, practices), **Data analysis**
(validation checks and pre-analysis), **Models** (each model switchable, with live
settings, results and a locality explorer) and **Comparison** (any models from either
system on the shared walk-forward folds). See [docs/APP.md](docs/APP.md).

Default results are precomputed in `app/data/default_run/`; rebuild them with
`python scripts/build_app_results.py` after changing the data or default configs. To
share the dashboard via a public link, deploy the repository on
[Streamlit Community Cloud](https://share.streamlit.io) with `app/dashboard.py` as the
main file.

Run End to End

```bash
# Train and evaluate every model in experiments/standard/config_comparison.yaml
python main.py

# Add walk-forward evaluation (240-month train, 12-month test windows)
python main.py --rolling --train-window 240 --test-window 12

# Run a subset of models, or a different config
python main.py --models global_linear lwlr mcllr
python main.py --config experiments/standard/config_mcllr.yaml --no-plots

# Improved system (outputs in results/improved/), or both systems in one run
python main.py --system improved
python main.py --system both --rolling
```

Outputs go to `results/`: `metrics.csv`, `predictions.csv` (per-date predictions and
uncertainties), `rolling_folds.csv` / `rolling_summary.csv` with `--rolling`, and
diagnostic figures in `results/figures/`. The improved system writes `summary.csv`,
`folds.csv`, `predictions.csv`, `tuning.csv` (hyperparameters chosen per fold) and
`validation_report.md` to `results/improved/`. Each run is also archived under
`experiments/logs/` (disable with `--no-track`). See `python main.py --help`.

Run Experiments (notebooks)

# Launch notebook for exploratory data analysis
jupyter notebook notebooks/01_dataset_exploration.ipynb

# Run baseline global linear model
jupyter notebook notebooks/02_global_linear_baseline.ipynb

# Run LWLR deterministic locality experiments
jupyter notebook notebooks/03_lwlr_deterministic.ipynb

# Run Monte Carlo subsampled linear regression
jupyter notebook notebooks/04_mc_subsampled_linear.ipynb

# Run MCLLR (proposed) stochastic locality model
jupyter notebook notebooks/05_mcllr_proposed.ipynb

# Compare all model results
jupyter notebook notebooks/06_results_comparison.ipynb

Dataset
We use the Fama–French Five-Factor dataset, a canonical benchmark in asset pricing and quantitative finance.

Data location: data/raw/F-F_Research_Data_5_Factors_2x3.csv

Features (risk factors) at time t
- Market excess return (MKT-RF)
- Size factor (SMB)
- Value factor (HML)
- Profitability factor (RMW)
- Investment factor (CMA)

Prediction target
yt+1 = next-period market excess return

This formulation ensures:

- True forecasting (no look-ahead bias)
- Risk-adjusted interpretation
- Industry relevance in portfolio and factor modeling

Data Loading Example
```python
import pandas as pd
import numpy as np

# Load Fama-French data
df = pd.read_csv('data/raw/F-F_Research_Data_5_Factors_2x3.csv', skiprows=3)

# Parse date column
df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d')

# Convert returns to decimal form (divide by 100)
factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
df[factors] = df[factors] / 100

# Prepare features and targets
X = df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].values
y = df['Mkt-RF'].shift(-1).dropna().values  # next-period market return

print(f"Shape: X={X.shape}, y={y.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
```

Scientific Framing
This work is not merely a finance regression comparison. It is a study of deterministic vs stochastic locality in linear model ensembles for non-stationary environments. While finance provides a realistic testbed, the framework generalizes to energy demand forecasting, dynamic pricing, sensor modeling, and time-varying control systems.

Mathematical Foundation of Locality
From Global Linearity to Stochastic Locality
The repository’s models form a precise theoretical ladder:

Global Linear Regression → LWLR → Monte Carlo LR → MCLLR

Each step relaxes one structural assumption:

Model | Relaxes | Limitation Remaining
---|---:|---
Global LR | Stationarity | No locality
LWLR | Globality | Deterministic locality, no uncertainty
MC Linear | Determinism | No geometric locality
MCLLR | — | Stochastic locality

Thus MCLLR is a stochastic generalization of local regression, not merely an ensemble.

Kernel Similarity and Local Geometry
For a query state x0:

w_i(x0) = exp(-||x_i - x0||^2 / (2 tau^2))

These kernel weights encode geometric similarity and deterministic locality but are not probabilities.

Probability Normalization — The Critical Bridge
To enable sampling we normalize weights to obtain a discrete probability distribution over data indices:

p_i(x0) = w_i(x0) / sum_j w_j(x0)

This preserves relative similarity, ensures the probabilities sum to one, and converts deterministic geometry into probabilistic locality.

Stochastic Local Sampling
Fixed-Size Local Neighborhoods (canonical MCLLR)

S_k ~ Multinomial(m, p(x0))

Where m is the stochastic neighborhood size; each sample defines a plausible local regime realization and yields a local regression.

Random-Size Locality (extensions)
Bernoulli inclusion
Poisson sampling

These model uncertainty in regime width, not only membership.

Diversity-Aware and Continuous Extensions
Weighted sampling without replacement
Determinantal point processes
Kernel density / GP-style resampling

The MCLLR Estimator
For r Monte Carlo realizations:

y^(k)(x0) = x0^T beta^_{S_k}

Mean prediction:
y^(x0) = (1/r) sum_{k=1..r} y^(k)(x0)

Predictive variance:
sigma^2(x0) = (1/r) sum_k (y^(k)(x0) - y^(x0))^2

Thus MCLLR computes expectation and variance over stochastic local linear models.

Interpretation of Core Parameters
tau (bandwidth): geometric locality in LWLR
m (subsample size): stochastic locality scale (analogue of bandwidth)
r (MC repetitions): number of stochastic local worlds simulated (controls estimation stability)

Monte Carlo convergence: estimation error ∝ 1/r.

Regime Interpretation
MCLLR does not explicitly label regimes. Instead it models a distribution over locally valid linear behaviors; regime uncertainty emerges implicitly from probabilistic local membership and stochastic sampling.

Unified Theoretical Statement
Monte Carlo Local Linear Regression estimates predictions as expectations over locally fitted linear models drawn from a kernel-defined probability distribution, modeling non-stationary dynamics through stochastic locality rather than deterministic regime partitioning.

Model Hierarchy & Usage Examples
1) Global Linear Regression — Stationary Baseline
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

2) Locally Weighted Linear Regression (LWLR) — Deterministic Locality
```python
from lrd.standard.models.lwlr import LocallyWeightedLinearRegression

lwlr = LocallyWeightedLinearRegression(bandwidth=1.0)
lwlr.fit(X_train, y_train)
y_pred = lwlr.predict(X_test)
```

3) Monte Carlo Subsampled Linear Regression — Stochastic Global Ensemble
```python
from lrd.standard.models.mc_linear import MonteCarloLinearRegression

mc_model = MonteCarloLinearRegression(n_simulations=100, subsample_size=50)
mc_model.fit(X_train, y_train)
y_mean, y_std = mc_model.predict(X_test, return_std=True)
```

4) Monte Carlo Local Linear Regression (MCLLR) — Stochastic Locality (Proposed)
```python
from lrd.standard.models.mcllr import MonteCarloLocalLinearRegression

mcllr = MonteCarloLocalLinearRegression(
    bandwidth=1.0,
    n_simulations=50,
    subsample_size=30
)

mcllr.fit(X_train, y_train)
y_pred, y_unc = mcllr.predict(X_test, return_std=True)
```

Monte Carlo Local Linear Regression (MCLLR) — Design Architecture

The following section documents the core design philosophy of MCLLR, defining its structural choices before implementation. These decisions reflect a unified principle: **stochastic locality through kernel-weighted probabilistic sampling, with uncertainty realization via Monte Carlo aggregation**.

**Query-Dependent Locality**

*What:* The algorithm executes an outer loop over test points and an inner loop over stochastic simulations. Kernel weights depend on each query point x₀, making locality query-specific rather than pre-computed globally.

*Why:* In time-series financial data, different regimes have different local geometries. A fixed global weighting scheme cannot capture this. By recomputing weights for each prediction point, the model adapts its notion of "nearby" observations to the specific context of the query.

*Role:* This design ensures that MCLLR produces **personalized neighborhoods** for each x₀, enforcing that stochastic samples are drawn from geometrically relevant regions of the feature space. The outer loop structure is not a computational artifact but a deliberate design choice encoding query dependence.

**Squared Euclidean Distance (No Square Root)**

*What:* Kernel weight computation uses squared Euclidean distance: ||x_i - x₀||², not ||x_i - x₀||.

*Why:* Avoiding the square root operation achieves two objectives: (1) numerical stability in floating-point arithmetic when distances are very small, preventing underflow and precision loss; (2) computational efficiency during large-scale prediction, eliminating expensive square root evaluations for every distance pair.

*Role:* This choice preserves monotonic distance ordering (the ranking of nearby points remains unchanged) while improving numerical robustness. It is standard in kernel methods and aligns MCLLR with LWLR implementation practices in the literature.

**Gaussian Kernel for Weight Computation**

*What:* Weights are computed via the Gaussian (RBF) kernel:

w_i(x₀) = exp(−||x_i − x₀||²/(2τ²))

*Why:* The Gaussian kernel provides smooth, continuous locality: points near x₀ receive high weight, distant points receive exponentially decaying weight. This is theoretically justified for regression neighborhoods because it enforces smooth transitions between local regimes rather than sharp cutoffs (which would create discontinuities in predictions).

*Role:* This serves as the **geometric locality function**. The bandwidth parameter τ controls the spatial extent of locality. The kernel is deterministic and problem-independent; its output is treated as raw similarity, not probability.

**Normalization of Weights Only Before Sampling**

*What:* Weights are normalized to probabilities exclusively to enable sampling:

p_i = w_i / ∑_j w_j

Normalization does **not** occur before regression; it occurs only to define the probability distribution for drawing the stochastic neighborhood.

*Why:* Normalization serves one purpose: to ensure sampling probabilities sum to one. Inside each simulation, OLS regression uses original weights only if explicitly weighting is demanded; however, in MCLLR's core design, the key insight is that normalization is probabilistic, not geometric. This separation prevents double-counting locality.

*Role:* This normalization is the **critical bridge** between deterministic geometry and stochastic locality. It converts kernel similarities (unnormalized) into a discrete probability measure over observations, enabling principled sampling.

**Weighted Random Sampling Without Replacement**

*What:* Given probabilities p_i derived from normalized weights, MCLLR draws a fixed-size sample S_k of size m without replacement from the training set, where each observation has probability proportional to p_i.

*Why:* Sampling without replacement is preferred over multinomial sampling (with replacement) because: (1) it preserves observation **uniqueness** within a neighborhood, enforcing that each local regression uses distinct observations; (2) it avoids redundancy in the local feature space, maintaining geometric diversity; (3) it models a concrete "plausible local world" rather than abstract reweighting.

*Role:* This mechanism **realizes stochasticity** in locality selection. Each simulation k draws a different local subset, inducing variation in local linear models and ultimately in predictions. The randomness captures uncertainty in which observations best characterize the local regime.

**Plain OLS Inside Each Simulation**

*What:* Within each simulation, the selected subsample S_k is fit using ordinary least squares (OLS)—ordinary, unweighted regression—not weighted regression.

*Why:* Locality has already been enforced via the sampling step: by construction, S_k contains observations with high kernel weight relative to x₀. Applying weighted regression again within the subsample would **double-count locality**, giving disproportionate influence to nearby points even among nearby points. OLS inside the sampled neighborhood is sufficient and theoretically correct.

*Role:* This choice ensures **clean separation of concerns**: sampling enforces geometric locality; OLS fits the local surface. The combination avoids compound locality effects that would distort local linear approximation.

**Monte Carlo Aggregation (Mean and Standard Deviation)**

*What:* After fitting r independent local linear models via sampling and OLS, predictions aggregate as:

ŷ(x₀) = (1/r) ∑_{k=1}^r ŷ^{(k)}(x₀)

σ̂²(x₀) = (1/r) ∑_{k=1}^r (ŷ^{(k)}(x₀) − ŷ(x₀))²

*Why:* Monte Carlo integration produces both point estimates (mean) and uncertainty quantification (variance). The standard deviation reflects sampling variability across stochastic local models, encoding our epistemic uncertainty about which local regime applies to x₀. This is interpretable as a predictive distribution.

*Role:* This aggregation is the **uncertainty mechanism** of MCLLR. Variance does not reflect training noise (as in classical frequentist regression); it reflects regime uncertainty—the ensemble spread of plausible local linear behaviors. This is essential for risk-aware decision-making in finance.

**No Model-Switching Logic**

*What:* MCLLR does not contain hidden conditional logic that changes model behavior based on hyperparameters. Hyperparameters τ (bandwidth) and m (subsample size) directly scale locality geometry and stochastic neighborhood size, but do not trigger regime switches or algorithmic branches.

*Why:* This maintains interpretability and ensures **smooth theoretical degeneracy**. MCLLR must naturally reduce to simpler models as follows:

- Large τ (broad bandwidth) → all observations weighted nearly equally → sampling is nearly uniform → MCLLR → MC Linear Regression
  
- Large m (subsample size) → S_k approaches the full training set → local models become global models → MCLLR → LWLR
  
- Both large τ and large m → all observations uniformly sampled in large subsets → MCLLR → Global Linear Regression

These limiting behaviors must emerge from the same code path, not from explicit conditional branches. This guarantees theoretical consistency and prevents ad-hoc model selection.

*Role:* This design ensures MCLLR is a **unified, parameter-continuous framework** encompassing global and local approaches. It validates that MCLLR is a generalization, not a disjoint ensemble method.

**Epsilon-Based Numerical Stability Safeguard**

*What:* When computing normalized weights p_i = w_i / (∑_j w_j), if the sum ∑_j w_j is numerically very small (below machine epsilon), a small constant ε is added to prevent division by zero or catastrophic precision loss.

*Why:* In rare edge cases—such as prediction far from the training support where all weights become vanishingly small—floating-point computation may produce NaN or Inf. The epsilon safeguard ensures numerical stability without altering the model's structural logic or changing weights materially.

*Role:* This is a **robustness safeguard** that keeps the algorithm running correctly in numerical boundary conditions without compromising design philosophy. It is not a modeling choice but an engineering necessity.

**Clear Separation of Responsibilities**

The MCLLR architecture enforces a clean functional decomposition:

| Component | Function | Responsibility |
|---|---|---|
| Kernel | w_i(x₀) = exp(−\\||x_i − x₀\\||²/(2τ²)) | Geometric locality; deterministic similarity |
| Normalization | p_i = w_i / ∑_j w_j | Probability assignment; sampling distribution |
| Sampling | S_k ∼ p_i without replacement | Stochastic neighborhood realization |
| OLS | β̂_k = (S_k^T S_k)^{−1} S_k^T y_k | Local surface fitting within neighborhood |
| Aggregation | ŷ = mean, σ̂² = var over k | Uncertainty quantification; ensemble prediction |

This separation ensures each component has a single, well-defined purpose. Changes to one (e.g., kernel choice) do not cascade through the pipeline; design modifications remain localized and testable.

**Conceptual Summary**

Monte Carlo Local Linear Regression is a stochastic realization of local linear approximations whose uncertainty emerges from probabilistic neighborhood selection. By decoupling deterministic geometric locality (kernels) from stochastic membership (sampling), MCLLR captures both regime-dependent non-linearity and estimation uncertainty in a unified, theoretically coherent framework.

Experimental Goals
- Prediction accuracy across regimes
- Stability vs variance trade-offs
- Uncertainty calibration
- Robustness to non-stationarity

Validation Roadmap (Implemented)
The following checklist converts the research hypothesis into code-level validation steps.

- [x] Rolling non-stationary evaluation
- [x] Time-series hyperparameter tuning
- [x] MCLLR ablation analysis
- [x] Uncertainty quality metrics
- [x] Stronger regularized baseline

1) Rolling non-stationary evaluation
Theory
- A single fixed train/test split can hide regime sensitivity.
- Walk-forward folds test whether performance remains stable as regimes shift.

Programming
- Module: `lrd/core/evaluation/rolling.py`
- Key APIs:
  - `walk_forward_splits(...)`
  - `evaluate_walk_forward(...)`
  - `rolling_results_to_frame(...)`
  - `summarize_rolling_metrics(...)`

Example
```python
from lrd.core.evaluation.rolling import evaluate_walk_forward, rolling_results_to_frame

folds = evaluate_walk_forward(
    model_builder=lambda: model,
    X=X,
    y=y,
    train_window=240,
    test_window=24,
    step=24,
)
df_folds = rolling_results_to_frame(folds)
```

2) Time-series hyperparameter tuning
Theory
- Hyperparameters (`tau`, `m`, `r`) must be selected using chronological validation,
  not random splits, to avoid leakage and over-optimistic selection.

Programming
- Module: `lrd/standard/experiments/tuning.py`
- API: `tune_model_with_walk_forward(...)`

Example
```python
from lrd.standard.config import ModelCofig
from lrd.standard.experiments.tuning import tune_model_with_walk_forward

leaderboard, best = tune_model_with_walk_forward(
    base_model_config=ModelCofig(
        model_type="mcllr",
        bandwidth=1.0,
        n_simulations=50,
        subsample_size=80,
        fit_intercept=True,
        random_state=42,
    ),
    param_grid={
        "bandwidth": [0.5, 1.0, 2.0],
        "subsample_size": [50, 80, 120],
        "n_simulations": [30, 50],
    },
    X=X,
    y=y,
    train_window=240,
    test_window=24,
    objective_metric="rmse",
)
```

3) MCLLR ablation analysis
Theory
- To validate the claim "geometric + stochastic locality is necessary",
  each component must be removed and compared.

Programming
- Models added:
  - `mcllr_no_geometry` (stochastic sampling only)
  - `mcllr_no_stochasticity` (deterministic local subset only)
- File: `lrd/standard/models/mcllr_ablations.py`
- Factory support: `lrd/standard/models/factory.py`

Interpretation
- If full `mcllr` consistently outperforms both ablations in rolling tests,
  the core thesis is empirically supported.

4) Uncertainty quality metrics
Theory
- Uncertainty-aware modeling is not only "having a std output".
- It must be calibrated and decision-useful.

Programming
- Added metrics in `lrd/core/evaluation/metrics.py`:
  - `coverage_95`
  - `interval_width_95`
  - `gaussian_nll`
  - `interval_score_95`
- Integrated into `lrd/core/evaluation/evaluator.py` for all models that return uncertainty.

5) Stronger regularized baseline
Theory
- A stronger linear baseline prevents over-claiming MCLLR gains over weak comparators.
- Ridge is a robust baseline under multicollinearity and noisy factors.

Programming
- Model: `ridge_linear`
- File: `lrd/standard/models/ridge_linear.py`
- Config support in `lrd/standard/config.py` and `lrd/standard/models/factory.py`

6) End-to-end rolling experiment runner
Programming
- Module: `lrd/standard/experiments/rolling_runner.py`
- Runs all models from one config over walk-forward folds.

Example
```python
from lrd.standard.experiments.rolling_runner import RollingExperimentRunner

runner = RollingExperimentRunner(
    "experiments/standard/config_comparison.yaml",
    train_window=240,
    test_window=24,
    step=24,
)
fold_df, summary_df = runner.run()
```

7) Unified experiment config for claim testing
Programming
- File: `experiments/standard/config_comparison.yaml`
- Includes:
  - `global_linear`
  - `ridge_linear`
  - `lwlr`
  - `mc_linear`
  - `mcllr`
  - `mcllr_no_geometry`
  - `mcllr_no_stochasticity`

Practical criterion for claim acceptance
- Full MCLLR should:
  - win on rolling `rmse/mae` mean across folds,
  - maintain competitive variance (`std`) across folds,
  - show credible uncertainty quality (`coverage_95`, lower interval score/NLL),
  - outperform both ablations consistently.

Project Structure
```
local-risk-dynamics/
  README.md                  # overview (this file)
  docs/
    PIPELINE.md              # how the standard and improved pipelines work
    MODEL_ASSESSMENT.md      # results, diagnosis and improvement roadmap
    APP.md                   # dashboard screens and design decisions
  data/raw/                  # Fama-French data (shared by both systems)
  lrd/                       # Python package
    core/                    # SHARED: data loading/splitting, model interface,
                             #   metrics, evaluator, walk-forward folds,
                             #   significance tests, experiment tracking
    standard/                # original system: config, 7 models, runners
    improved/                # improved system: validation, train-only prep, nested tuning
  experiments/standard/      # YAML configs for the standard system
  experiments/improved/      # YAML config for the improved system
  main.py                    # end-to-end CLI
  app/dashboard.py           # interactive dashboard (screens in app/views/)
  lrd/comparison.py          # runs both systems on shared folds for comparison
  scripts/                   # improvement_study.py and utilities
  notebooks/                 # exploratory notebooks
  tests/                     # core/, standard/, improved/, CLI and app tests
  pyproject.toml             # package + pytest configuration
  requirements.txt
```

The original system is preserved exactly at the git tag `v1-standard`
(`git checkout v1-standard`). Both systems share `lrd/core`, so they are always compared
on the same months with the same metric definitions.

Repository Philosophy
Separates theory (README, docs/), engineering (lrd/), and evidence (notebooks, scripts,
the app), mirroring quantitative research repositories and ML paper implementations.

Project Status
- Standard system: implemented, tested and evaluated (see docs/MODEL_ASSESSMENT.md).
- Improved system: implemented in `lrd/improved/` (`python main.py --system improved`);
  results in docs/MODEL_ASSESSMENT.md §5, next steps in §6.

Running tests
```bash
python -m pytest
```

Contributing
```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Commit with meaningful messages
git commit -m "add: descriptive commit message"

# Push and open a pull request
git push origin feature/your-feature-name
```

License
This project is licensed under the MIT License. See LICENSE file for details.

Key Takeaway
Financial markets are locally linear but globally non-linear. Understanding risk may therefore require many small linear views of reality—combined through deterministic structure, stochastic sampling, or both.

If you'd like, I can next produce a NeurIPS/ICLR-style paper draft, a formal mathematical appendix, or implement the full MCLLR algorithm and notebooks.
