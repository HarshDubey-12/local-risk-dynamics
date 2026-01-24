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

## Dataset

We use the **Fama–French Five-Factor dataset**, a canonical benchmark in asset pricing and quantitative finance.

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

## Key Takeaway

**Financial markets are locally linear but globally non-linear.**

Understanding risk may therefore require:

> Not one global model, but many small linear views of reality—
> combined through deterministic structure, stochastic sampling, or both.
