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

Run Experiments

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
from src.models.lwlr import LocallyWeightedLinearRegression

lwlr = LocallyWeightedLinearRegression(bandwidth=1.0)
lwlr.fit(X_train, y_train)
y_pred = lwlr.predict(X_test)
```

3) Monte Carlo Subsampled Linear Regression — Stochastic Global Ensemble
```python
from src.models.mc_linear import MonteCarloLinearRegression

mc_model = MonteCarloLinearRegression(n_simulations=100, subsample_size=50)
mc_model.fit(X_train, y_train)
y_mean, y_std = mc_model.predict(X_test, return_std=True)
```

4) Monte Carlo Local Linear Regression (MCLLR) — Stochastic Locality (Proposed)
```python
from src.models.mcllr import MonteCarloLocalLinearRegression

mcllr = MonteCarloLocalLinearRegression(
    bandwidth=1.0,
    n_simulations=50,
    subsample_size=30
)

mcllr.fit(X_train, y_train)
y_pred, y_unc = mcllr.predict(X_test, return_uncertainty=True)
```

Experimental Goals
- Prediction accuracy across regimes
- Stability vs variance trade-offs
- Uncertainty calibration
- Robustness to non-stationarity

Project Structure
```
local-risk-dynamics/
  data/
    raw/                    # Original Fama-French data
    processed/              # Preprocessed datasets
  notebooks/
    01_dataset_exploration.ipynb
    02_global_linear_baseline.ipynb
    03_lwlr_deterministic.ipynb
    04_mc_subsampled_linear.ipynb
    05_mcllr_proposed.ipynb
    06_results_comparison.ipynb
  src/
    data/
      loader.py
      preprocessing.py
    models/
      global_linear.py
      lwlr.py
      mc_linear.py
      mcllr.py
    optimization/
      sgd.py
      closed_form.py
    evaluation/
      metrics.py
      backtesting.py
    utils/
      kernels.py
      sampling.py
      plotting.py
  experiments/
    config_global.yaml
    config_lwlr.yaml
    config_mc.yaml
    config_mcllr.yaml
  figures/
  README.md
  requirements.txt
```

Repository Philosophy
Separates theory (README), engineering (src/), and evidence (notebooks/figures), mirroring quantitative research repositories and ML paper implementations.

Project Status
Current stage: complete theoretical foundation + structured research scaffold.

Next milestones:
- Full MCLLR implementation
- Empirical comparison
- Industry implications

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
