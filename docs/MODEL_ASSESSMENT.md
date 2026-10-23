# Model Assessment: Current State of MCLLR and Improvement Roadmap

This document records what the experiments in this repository show about the proposed
**Monte Carlo Local Linear Regression (MCLLR)** model, why it falls short of the
hypothesis, and what needs to change to make it outperform the existing baselines.

All numbers can be reproduced with:

```bash
python scripts/improvement_study.py     # walk-forward study (~2 minutes)
python main.py --rolling                # repository pipeline: hold-out + walk-forward
```

---

## TL;DR

| Question | Answer |
|---|---|
| Does MCLLR work as the theory proposes? | **No.** As implemented, it is **statistically significantly worse** than Ridge and the historical-mean benchmark. |
| Does any model predict monthly market returns? | **No.** The best baseline (Ridge) is statistically indistinguishable from forecasting the historical mean. |
| Are MCLLR's uncertainty estimates reliable? | **No.** Nominal 95% intervals cover only **37%** of outcomes. |
| Can it be fixed? | **Partially.** Adding ridge shrinkage inside each local fit and a predictive-variance term makes it the **best model tested** (+0.35% out-of-sample R², 94% coverage), but the edge is **not yet statistically significant**. |

---

## 1. Evaluation setup

| Item | Setting |
|---|---|
| Data | Fama–French 5 factors, monthly, Jul 1963 – Oct 2025 (816 months) |
| Task | Predict next month's market excess return (`Mkt-RF` at *t+1*) from the 5 factors at *t* |
| Protocol | Walk-forward: 240-month training window, 12-month test window, refit every year → **48 folds, 576 test months (Nov 1977 – Oct 2025)** |
| Benchmark | **Historical mean** of the training window (the standard benchmark in return predictability) |
| Main score | **Out-of-sample R² vs historical mean** = 1 − SSE(model) / SSE(mean). Positive ⇒ the model beats the mean. |
| Significance | **Diebold–Mariano test** (squared loss) on monthly forecast errors |
| Calibration | Fraction of outcomes inside the model's nominal 95% interval (target: 0.95) |

---

## 2. Current state of the models

### Walk-forward results (repository models, default configuration)

| Model | RMSE | OOS R² vs mean | 95% coverage | DM vs Ridge |
|---|---|---|---|---|
| Historical mean | 0.04510 | 0.00% | – | equal (p = 0.84) |
| **Ridge (α = 1)** | **0.04509** | **+0.05%** | – | – |
| MC Linear | 0.04571 | −2.72% | 38% | worse (p = 0.09) |
| Global OLS | 0.04580 | −3.12% | – | **worse (p = 0.03)** |
| **MCLLR (proposed)** | **0.04634** | **−5.56%** | **37%** | **worse (p = 0.01)** |
| LWLR | 0.04669 | −7.15% | – | **worse (p = 0.004)** |

The single 80/20 hold-out test (`python main.py`) gives the same ordering: every model has
negative R² against the test-period mean, Ridge is best, MCLLR ranks fifth of seven.

### Conclusions

1. **Returns are essentially unpredictable from contemporaneous factors.** No model
   significantly beats the historical mean. Ridge "wins" only because α = 1 is very strong
   shrinkage at this data scale (XᵀX has diagonal ≈ n·var ≈ 1), which pulls it almost
   all the way to the mean.
2. **Locality hurts point forecasts.** Performance degrades monotonically with locality:
   Global OLS > MCLLR > LWLR. The ranking holds before and after 2005 and with the
   largest move (Sep 1987, −23%) excluded, so it is not driven by one period or outlier.
3. **The uncertainty estimates are badly over-confident.** MC Linear and MCLLR cover only
   37–38% of outcomes with nominal 95% intervals.
4. **Neither MCLLR component adds value as built.** Removing geometry
   (`mcllr_no_geometry`) or stochasticity (`mcllr_no_stochasticity`) does not reveal a
   component that is helping: the full model is still significantly worse than Ridge.

---

## 3. Why MCLLR under-performs

| Cause | Explanation |
|---|---|
| **Variance dominates** | Each local fit estimates 6 coefficients from 80 noisy months with no regularisation. When the true signal is near zero, coefficient noise *is* the forecast error. Averaging 50 fits reduces but does not remove it. |
| **Locality in the wrong space** | "Similar months" are defined by similar *factor returns*, which are mostly noise. They are a poor proxy for the economic regime (volatility, valuations, rates) that the hypothesis is about. |
| **Weak signal to localise** | Locality pays off when there is a real relationship that *changes* across regimes. Here the relationship is close to zero in every regime. |
| **Epistemic-only uncertainty** | The reported standard deviation is the spread *between* Monte Carlo fits (parameter uncertainty). It ignores the month-to-month noise in returns, which is ~4% a month, so intervals are far too narrow. |
| **Bandwidth on raw scale** | Distances are computed on unscaled decimal returns, so factors with higher volatility (Mkt-RF) dominate the kernel. |

---

## 4. Improvement experiments

All candidates below were evaluated with the same 48-fold walk-forward protocol
(`scripts/improvement_study.py`). Local ridge uses penalty `alpha × subsample_size`;
standardised variants z-score the factors with training-window statistics before
computing kernel distances.

| Variant | OOS R² vs mean | 95% coverage | DM vs Ridge | Verdict |
|---|---|---|---|---|
| MCLLR (current) | −5.56% | 37% | worse, p = 0.01 | baseline |
| MCLLR, larger neighbourhood (m = 160) | −4.30% | 19% | worse, p = 0.03 | helps a little |
| MCLLR, standardised distances only | −5.87% | 18% | worse, p = 0.01 | no help alone |
| Global Ridge, tuned (α = 0.02·n) | +0.04% | – | equal, p = 0.95 | ≈ mean |
| Bagged ridge (same as below, **no locality**) | +0.05% | 8% | equal, p = 1.00 | ≈ mean |
| **MCLLR + local ridge** (standardised, bw = 2, m = 160) | **+0.35%** | 7% | better, p = 0.22 | **best accuracy** |
| MCLLR + local ridge (bw = 1) | +0.39% | 4% | better, p = 0.30 | similar |
| **MCLLR + local ridge + predictive std** | **+0.35%** | **94%** | better, p = 0.22 | **best overall** |
| Recency-local MC + ridge + predictive std | −0.05% | 94% | equal, p = 0.81 | time locality ≈ mean |

### What the experiments show

1. **Shrinkage inside each local fit is the decisive change.** It turns MCLLR from the
   second-worst model into the best one tested.
2. **Locality adds value once variance is controlled.** MCLLR + local ridge (+0.35%)
   beats the identical procedure without locality, bagged ridge (+0.05%), and tuned global
   ridge (+0.04%). The result is stable across ridge strengths (α ∈ [0.02, 0.1] → +0.34% to
   +0.36%) and is best with narrower kernels (bw = 1–2 standard deviations).
3. **Predictive variance fixes calibration.** Adding the local residual variance to the
   Monte Carlo spread raises 95% coverage from 7% to **94%**, without changing the point
   forecast. No baseline in the repository provides any uncertainty, so this is MCLLR's
   clearest advantage.
4. **The evidence is not yet conclusive.**
   - The improvement over Ridge is **not statistically significant** (p ≈ 0.15–0.30).
   - It is concentrated **before 2005** (+0.57%); after 2005 it is ≈ 0 (+0.06%).
   - The variants were chosen on the same test period, so results carry mild
     selection bias. They must be re-confirmed with nested tuning (see §5.5).

For context, Campbell & Thompson (2008) show that a monthly OOS R² of roughly 0.5% can
already be economically meaningful for a mean-variance investor. The improved model is in
that range, which makes the direction worth pursuing, but not yet proven.

---

## 5. Proposed changes

Ordered by priority. Items 5.1–5.2 are validated by the experiments above; the rest are
the next steps needed to make the improvement robust and significant.

### 5.1 Regularise the local fits *(validated: −5.56% → +0.35% OOS R²)*

- Add an `alpha` hyperparameter to `MonteCarloLocalLinearRegression` and solve each local
  fit as ridge regression with an unpenalised intercept, penalty `alpha × subsample_size`.
- Default `subsample_size` to ~160 instead of 80 (more data per local fit).
- Compute kernel distances on **standardised** factors (z-scores from the training window)
  so the bandwidth is in standard-deviation units and no factor dominates. Default
  bandwidth ≈ 1–2.
- Files: a new improved MCLLR in `lrd/improved/` (the original in
  `lrd/standard/models/mcllr.py` stays unchanged as the reference), with its configs
  in `experiments/improved/`.

### 5.2 Report predictive, not just epistemic, uncertainty *(validated: 37% → 94% coverage)*

- For each simulation, record the residual variance of the local fit; return
  `std = sqrt(var(Monte Carlo predictions) + mean(local residual variance))`.
- Apply the same fix to `MonteCarloLinearRegression`.
- Position MCLLR as the model that provides **calibrated, query-specific risk intervals**.

### 5.3 Define locality on regime (state) variables

The most promising untested change. Measure similarity on variables that describe the
market regime instead of on this month's factor returns, for example:

- realised volatility of the market over the past 1–3 months,
- VIX or another implied-volatility index,
- dividend yield / earnings yield, term spread, default spread,
- recent drawdown or 12-month momentum of the market.

Keep the regression on the 5 factors, but compute kernel weights from the state vector.
This matches the original hypothesis ("regime-dependent relationships") far better than
distances between noisy return vectors.

### 5.4 Shrink forecasts toward the historical mean

- Combine forecasts: `ŷ = w · ŷ_MCLLR + (1 − w) · ȳ_train`, with `w` chosen in-sample.
- Optionally apply the Campbell–Thompson restriction (truncate negative equity-premium
  forecasts at zero).
- Both are standard, cheap, and typically add out-of-sample R² for weak predictors.

### 5.5 Tune without look-ahead (nested walk-forward)

- Select `bandwidth`, `alpha` and `subsample_size` inside each fold using only that fold's
  training window (e.g. an inner walk-forward on the 240 training months), using the
  existing `lrd/standard/experiments/tuning.py`.
- Report the nested results as the headline numbers; this removes the selection bias
  noted in §4.

### 5.6 Use the right tests and economic metrics

- **Clark–West test** (add next to Diebold–Mariano in `lrd/core/evaluation/significance.py`) for comparing a model against the historical mean, which is a
  nested model (Diebold–Mariano is conservative in that case).
- **Economic value**: certainty-equivalent return gains of a mean-variance investor
  using the forecasts, and the Sharpe ratio of a market-timing strategy.
- **Calibration**: coverage at several levels (50%, 80%, 95%), interval score, and
  Gaussian NLL (already in `lrd/core/evaluation/metrics.py`).
- Fix `lrd/core/evaluation/backtesting.py` to annualise monthly data with √12 (it currently
  uses √252) and replace the error-based "pseudo-returns" with a real strategy return.

### 5.7 Consider forecasting risk instead of returns

The project is named *Local Risk Dynamics*. Next-month **volatility** (or factor betas,
or Value-at-Risk) is far more predictable than next-month returns and genuinely changes
with the regime. A locality-based model is much more likely to beat global baselines on
that target. The pipeline needs only a different target in `lrd/core/data/loader.py`.

### 5.8 Engineering for larger studies

- Vectorise the Monte Carlo loop (batch least squares over simulations) and parallelise
  across query points / folds (e.g. `joblib`). MCLLR is currently ~100× slower than the
  global models, which limits nested tuning.

---

## 6. Recommended framing for the write-up

> Stochastic locality on its own does **not** improve point forecasts of monthly market
> excess returns; as originally specified, MCLLR is significantly worse than a ridge
> baseline. With local shrinkage, MCLLR matches or slightly exceeds the strongest baseline
> (+0.35% out-of-sample R², not significant), and with a predictive-variance correction it
> is the only model that delivers calibrated, query-specific 95% intervals (94% coverage).
> Defining locality on economic state variables and targeting risk rather than returns are
> the most promising directions for a conclusive result.

---

## 7. Status checklist

| Change | Status |
|---|---|
| Bandwidth on correct scale, MCLLR sampling stability, valid ablations | ✅ Done (in repository) |
| Walk-forward evaluation, `main.py`, Streamlit comparison app | ✅ Done |
| Local ridge in MCLLR (§5.1) | 🧪 Validated in `scripts/improvement_study.py`; not yet in `lrd/improved/` |
| Predictive variance (§5.2) | 🧪 Validated in study script; not yet in `lrd/improved/` |
| State-variable locality (§5.3) | ⬜ Not started (needs additional data) |
| Shrinkage toward mean (§5.4) | ⬜ Not started |
| Nested tuning (§5.5) | ⬜ Not started |
| Clark–West, economic metrics, backtest fixes (§5.6) | ⬜ Not started |
| Risk (volatility) target (§5.7) | ⬜ Not started |
| Vectorised / parallel MCLLR (§5.8) | ⬜ Not started |
