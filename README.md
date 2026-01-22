# Local Risk Dynamics
**Deterministic vs Stochastic Locality in Linear Models for Financial Risk**

## Overview

Financial markets are inherently non-stationary: relationships between risk factors and returns evolve across time, regimes, and uncertainty conditions. Traditional global linear regression assumes constant factor sensitivities, which often leads to model mis-specification in real-world financial environments.

This project studies an alternative paradigm:

> Can complex, regime-dependent financial risk behavior be approximated using ensembles of small, local linear models instead of a single global model?

We investigate this question through a structured comparison of four linear modeling philosophies that differ in how they define locality and uncertainty.

## Core Research Idea

All models in this repository lie on a conceptual spectrum:

**Stationary  Deterministic Local  Stochastic Global  Stochastic Local**

The project's central hypothesis is:

> Non-linear financial dynamics can be effectively approximated using collections of locally valid linear models, where locality may be enforced deterministically (distance-based) or stochastically (sampling-based).

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/HarshDubey-12/local-risk-dynamics.git
cd local-risk-dynamics

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

```bash
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
```

## Dataset

We use the **Fama–French Five-Factor dataset**, a canonical benchmark in asset pricing and quantitative finance.

**Data location:** `data/raw/F-F_Research_Data_5_Factors_2x3.csv`

### Features (risk factors) at time $t$

- Market excess return (MKT-RF)
- Size factor (SMB)
- Value factor (HML)
- Profitability factor (RMW)
- Investment factor (CMA)

### Prediction target

$$y_{t+1} = \text{next-period market excess return}$$

This formulation ensures:

- True forecasting (no look-ahead bias)
- Risk-adjusted interpretation
- Industry relevance in portfolio and factor modeling

### Data Loading Example

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

## Model Hierarchy

### 1. Global Linear Regression — Stationary Baseline

$$y_t = X_t \beta + \epsilon_t$$

**Assumptions**

- Constant factor sensitivities
- Single market regime
- Linear global structure

**Role in the project**

- Provides a stationary benchmark
- Quantifies mis-specification under regime shifts

**Usage Example**

```python
from sklearn.linear_model import LinearRegression

# Fit global model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
```

### 2. Locally Weighted Linear Regression (LWLR) — Deterministic Locality

LWLR fits a query-specific linear model using distance-based kernel weights:

$$\hat{\beta}(x_0) = (X^T W(x_0) X)^{-1} X^T W(x_0) y$$

**Gaussian kernel weights:**

$$w_i(x_0) = \exp\left(-\frac{\|x_i - x_0\|^2}{2\tau^2}\right)$$

**Key properties**

- Reduces bias via local linearization
- Sensitive to bandwidth selection
- Produces deterministic, regime-aware predictions

**Interpretation**

Financial factor sensitivities vary smoothly across similar market conditions.

**Usage Example**

```python
from src.models.lwlr import LocallyWeightedLinearRegression

# Initialize LWLR with bandwidth parameter
lwlr = LocallyWeightedLinearRegression(bandwidth=1.0)

# Fit on training data
lwlr.fit(X_train, y_train)

# Predict on test point (query-specific model)
y_pred = lwlr.predict(X_test)
```

### 3. Monte Carlo Subsampled Linear Regression — Stochastic Global Ensemble

**Procedure**

1. Randomly sample small subsets of historical data
2. Fit linear regression (via SGD) on each subset
3. Aggregate predictions across simulations

**Outputs**

- Mean prediction  expected return
- Prediction variance  uncertainty / risk

**Key distinction from LWLR**

- Locality arises from sampling, not geometry
- Captures uncertainty but lacks distance awareness

**Interpretation**

Market behavior is better represented as a distribution of plausible linear regimes rather than a single deterministic fit.

**Usage Example**

```python
from src.models.mc_linear import MonteCarloLinearRegression

# Initialize with 100 Monte Carlo samples
mc_model = MonteCarloLinearRegression(n_simulations=100, subsample_size=50)

# Fit ensemble
mc_model.fit(X_train, y_train)

# Get mean prediction and uncertainty
y_mean, y_std = mc_model.predict(X_test, return_std=True)
```

### 4. Monte Carlo Local Linear Regression (MCLLR) — Stochastic Locality (Proposed)

MCLLR combines:

- Kernel-defined neighborhoods (from LWLR)
- Monte Carlo sampling (from stochastic ensembles)
- Local linear regression aggregation

**For each query point:**

1. Define a local region via distance kernel
2. Randomly sample multiple neighborhoods within that region
3. Fit local linear models
4. Aggregate:
   - Mean prediction
   - Local uncertainty
   - Distance-aware weighting

**Conceptual contribution**

MCLLR converts deterministic bandwidth sensitivity into stochastic robustness by averaging across sampled local neighborhoods.

This produces:

- Regime-aware predictions
- Quantified uncertainty
- Improved stability under non-stationarity

**Usage Example**

```python
from src.models.mcllr import MonteCarloLocalLinearRegression

# Initialize MCLLR with both locality and stochasticity
mcllr = MonteCarloLocalLinearRegression(
    bandwidth=1.0,
    n_simulations=50,
    subsample_size=30
)

# Fit model
mcllr.fit(X_train, y_train)

# Predict with uncertainty quantification
y_pred, y_unc = mcllr.predict(X_test, return_uncertainty=True)
```

## Project Structure

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
       loader.py           # Data loading utilities
       preprocessing.py    # Preprocessing pipeline
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
 figures/                    # Output plots and results
 README.md
 requirements.txt
```

## Experimental Goals

The project evaluates:

- Prediction accuracy across regimes
- Stability vs variance trade-offs
- Uncertainty estimation quality
- Behavior under non-stationary financial dynamics

Rather than asking:

> Which model is best?

we ask:

> How different notions of locality influence risk modeling in non-stationary systems.

## Scientific Framing

This work is not merely a finance regression comparison.

It is a study of:

**Deterministic vs stochastic locality in linear model ensembles for non-stationary environments.**

While finance provides a realistic testbed, the framework generalizes to:

- Energy demand forecasting
- Dynamic pricing
- Sensor modeling
- Time-varying control systems

## Repository Philosophy

The repository is intentionally structured to separate:

- **Theory**  README / documentation
- **Engineering**  `src/`
- **Evidence**  notebooks / figures

This mirrors:

- Quantitative research repositories
- Machine learning paper implementations
- Production-grade experimentation frameworks

## Project Status

### Current stage

Structured research scaffold with planned modular implementation.

### Next milestones

- Reproducible data pipeline
- Stationary baseline validation
- Deterministic and stochastic locality experiments
- Full implementation of MCLLR
- Comparative analysis and industry implications

## Contributing

Contributions are welcome. To contribute:

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Commit with meaningful messages
git commit -m "add: descriptive commit message"

# Push and open a pull request
git push origin feature/your-feature-name
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Key Takeaway

**Financial markets are locally linear but globally non-linear.**

Understanding risk may therefore require:

> Not one global model, but many small linear views of reality—
> combined through deterministic structure, stochastic sampling, or both.
