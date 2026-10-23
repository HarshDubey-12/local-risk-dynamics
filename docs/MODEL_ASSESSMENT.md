# Model Assessment: Current State of MCLLR and Improvement Roadmap

This document records what the experiments in this repository show about the proposed
**Monte Carlo Local Linear Regression (MCLLR)** model, why it falls short of the
hypothesis, what the improved pipeline (`lrd/improved`) achieves, and what remains to be
done.

All numbers can be reproduced with:

```bash
python scripts/improvement_study.py     # walk-forward study of candidate fixes (~3 minutes)
python main.py --system improved        # improved pipeline with nested tuning (~40 seconds)
python main.py --rolling                # standard pipeline: hold-out + walk-forward
```

> **Data correction (this revision).** The data validation added in the improved
> pipeline found that the raw CSV contained a duplicated block: its first 73 lines were a
> copy of the file header plus 68 months (Jul 1963 – Feb 1969). The loader had silently
> used both copies, which double-counted those months and gave Feb 1969 a wrong target
> (Jul 1963). The duplicate has been removed and the loader now refuses duplicated or
> out-of-order months. All numbers below are on the corrected data. The conclusions are
> unchanged; the earlier figures are preserved in the `v1-standard` tag.

---

## TL;DR

| Question | Answer |
|---|---|
| Does MCLLR work as the theory proposes? | **No.** As originally specified it is **statistically significantly worse** than Ridge and the historical-mean benchmark. |
| Does any model predict monthly market returns? | **Barely.** No model significantly beats the historical mean. |
| Are the original MCLLR's uncertainty estimates reliable? | **No.** Nominal 95% intervals cover only **37%** of outcomes. |
| Does the improved MCLLR fix this? | **Partially.** With local ridge, standardised features and predictive variance, tuned honestly inside every fold, it is the **best model tested** (+0.36% out-of-sample R², best interval score and NLL, 94% coverage), but the edge is **not statistically significant** (p = 0.44). |

---

## 1. Evaluation setup

| Item | Setting |
|---|---|
| Data | Fama–French 5 factors, monthly, Jul 1963 – Oct 2025 (748 usable months after the t+1 shift) |
| Task | Predict next month's market excess return (`Mkt-RF` at *t+1*) from the 5 factors at *t* |
| Protocol | Walk-forward: 240-month training window, 12-month test window, refit every year → **42 folds, 504 test months (Jul 1983 – Jun 2025)** |
| Benchmark | **Historical mean** of the training window (the standard benchmark in return predictability) |
| Main score | **Out-of-sample R² vs historical mean** = 1 − SSE(model) / SSE(mean). Positive ⇒ the model beats the mean. |
| Significance | **Diebold–Mariano test** (squared loss) on monthly forecast errors |
| Calibration | Fraction of outcomes inside the model's nominal 95% interval (target: 0.95) |

---

## 2. Current state of the standard models

### Walk-forward results (standard system, default configuration)

| Model | RMSE | OOS R² vs mean | 95% coverage | DM vs Ridge |
|---|---|---|---|---|
| Historical mean | 0.0448 | 0.00% | – | equal (p = 0.98) |
| **Ridge (α = 1)** | **0.0448** | **+0.01%** | – | – |
| Global OLS | 0.0455 | −3.28% | – | **worse (p = 0.04)** |
| MC Linear | 0.0458 | −4.83% | 39% | **worse (p = 0.007)** |
| **MCLLR (proposed)** | **0.0462** | **−6.60%** | **37%** | **worse (p = 0.01)** |
| LWLR | 0.0464 | −7.38% | – | **worse (p = 0.005)** |

The single 80/20 hold-out test (`python main.py`) gives every model a negative R² against
the test-period mean; Ridge ranks first and MCLLR second of seven there, but the hold-out
covers only 150 months and the walk-forward ranking above is the more reliable one.

### Conclusions

1. **Returns are essentially unpredictable from contemporaneous factors.** No model
   significantly beats the historical mean. Ridge "wins" only because α = 1 is very strong
   shrinkage at this data scale (XᵀX has diagonal ≈ n·var ≈ 1), which pulls it almost
   all the way to the mean.
2. **Locality hurts point forecasts as originally specified.** Both local models (MCLLR,
   LWLR) are worse than global OLS, before and after 2005 and with the largest move
   (Sep 1987, −23%) excluded, so it is not driven by one period or outlier.
3. **The uncertainty estimates are badly over-confident.** MC Linear and MCLLR cover only
   37–39% of outcomes with nominal 95% intervals.
4. **Neither MCLLR component rescues it.** Removing geometry (`mcllr_no_geometry`) or
   stochasticity (`mcllr_no_stochasticity`) does not reveal a component that is helping:
   the full model is still significantly worse than Ridge.

---

## 3. Why the original MCLLR under-performs

| Cause | Explanation |
|---|---|
| **Variance dominates** | Each local fit estimates 6 coefficients from 80 noisy months with no regularisation. When the true signal is near zero, coefficient noise *is* the forecast error. Averaging 50 fits reduces but does not remove it. |
| **Locality in the wrong space** | "Similar months" are defined by similar *factor returns*, which are mostly noise. They are a poor proxy for the economic regime (volatility, valuations, rates) that the hypothesis is about. |
| **Weak signal to localise** | Locality pays off when there is a real relationship that *changes* across regimes. Here the relationship is close to zero in every regime. |
| **Epistemic-only uncertainty** | The reported standard deviation is the spread *between* Monte Carlo fits (parameter uncertainty). It ignores the month-to-month noise in returns, which is ~4% a month, so intervals are far too narrow. |
| **Bandwidth on raw scale** | Distances are computed on unscaled decimal returns, so factors with higher volatility (Mkt-RF) dominate the kernel. |

---

## 4. Improvement experiments

All candidates below were evaluated with the same 42-fold walk-forward protocol
(`scripts/improvement_study.py`). Local ridge uses penalty `alpha × subsample_size` on
raw-scale features; standardised variants z-score the factors with training-window
statistics before computing kernel distances. Hyperparameters here were chosen on the
same test period (see §5 for the bias-free version).

| Variant | OOS R² vs mean | 95% coverage | DM vs Ridge | Verdict |
|---|---|---|---|---|
| MCLLR (original) | −6.60% | 37% | worse, p = 0.01 | baseline |
| MCLLR, larger neighbourhood (m = 160) | −4.71% | 20% | worse, p = 0.03 | helps a little |
| MCLLR, standardised distances only | −5.75% | 19% | worse, p = 0.01 | no help alone |
| Global Ridge, tuned (α = 0.02·n) | +0.03% | – | equal, p = 0.92 | ≈ mean |
| Bagged ridge (same as below, **no locality**) | −0.02% | 9% | equal, p = 0.93 | ≈ mean |
| **MCLLR + local ridge** (standardised, bw = 2, m = 160) | **+0.30%** | 8% | better, p = 0.28 | **best accuracy** |
| MCLLR + local ridge (bw = 1) | +0.30% | 4% | better, p = 0.37 | similar |
| **MCLLR + local ridge + predictive std** | **+0.30%** | **94%** | better, p = 0.28 | **best overall** |
| Recency-local MC + ridge + predictive std | 0.00% | 94% | equal, p = 0.99 | time locality ≈ mean |

### What the experiments show

1. **Shrinkage inside each local fit is the decisive change.** It turns MCLLR from the
   second-worst model into the best one tested.
2. **Locality adds value once variance is controlled.** MCLLR + local ridge (+0.30%)
   beats the identical procedure without locality, bagged ridge (−0.02%), and tuned
   global ridge (+0.03%). It is positive both before 2005 (+0.46%) and after (+0.13%).
3. **Predictive variance fixes calibration.** Adding the local residual variance to the
   Monte Carlo spread raises 95% coverage from 8% to **94%** without changing the point
   forecast.
4. **These results carry selection bias.** The variants were chosen on the same test
   months they are scored on. §5 removes that bias.

---

## 5. Improved pipeline results (implemented in `lrd/improved`)

The validated changes are implemented as `ImprovedMCLLR` in the improved pipeline, which
adds data validation, train-only preparation (standardisation) and **nested walk-forward
tuning**: in every outer fold, bandwidth and ridge α are selected by an inner walk-forward
(180-month inner train, 12-month inner test) on that fold's training window only. No test
month influences any choice, so these are the headline numbers.

Configuration: `experiments/improved/config_improved.yaml` — Ridge tuned over
α ∈ {0.1, 1, 3, 10, 30}; ImprovedMCLLR with K = 50, m = 160, predictive std, tuned over
bandwidth ∈ {1, 2, 3} × α ∈ {1, 3, 10, 30}. Same 42 folds / 504 test months as above.

| Model | RMSE | OOS R² vs mean | DM vs mean | 95% coverage | Interval score ↓ | Gaussian NLL ↓ |
|---|---|---|---|---|---|---|
| Historical mean | 0.0448 | 0.00% | – | 94.1% | 0.2380 | −1.673 |
| Ridge (tuned) | 0.0447 | +0.25% | p = 0.68 | 94.4% | 0.2364 | −1.676 |
| **ImprovedMCLLR (tuned)** | **0.0447** | **+0.36%** | p = 0.44 | **94.3%** | **0.2345** | **−1.688** |

### Findings

1. **The improvement survives honest tuning.** ImprovedMCLLR remains the best model on
   accuracy and on both proper uncertainty scores (interval score, NLL), with calibrated
   95% intervals. Compared with the original MCLLR (−6.60%), the gap closed by the
   improvements is almost 7 percentage points of OOS R².
2. **It is still not statistically significant.** p = 0.44 against the historical mean;
   the evidence is consistent with a small real edge but cannot establish one.
3. **Hyperparameter choices are unstable across folds.** The inner search picks α = 1 in
   22 folds and α = 30 in 16, and bandwidth 3 in 23 folds versus 1 in 15. With a signal
   this weak, the inner validation error is nearly flat across candidates — another sign
   that the remaining gains must come from better locality information, not more tuning.

---

## 6. Proposed changes and status

Ordered by priority.

### 6.1 Regularise the local fits — ✅ implemented (`lrd/improved/models/mcllr.py`)

Local ridge with penalty `alpha × subsample_size`, larger neighbourhoods (m = 160),
kernel distances on standardised factors via the `standardize` preparation step.

### 6.2 Predictive, not just epistemic, uncertainty — ✅ implemented

`ImprovedMCLLR` returns `sqrt(var(Monte Carlo predictions) + mean(local residual
variance))`; `RidgeRegression` and `HistoricalMean` return residual / sample standard
deviations, so every improved model is scored on calibration.

### 6.3 Define locality on regime (state) variables — ⬜ next

The most promising remaining change. Measure similarity on variables that describe the
market regime instead of on this month's factor returns, for example:

- realised volatility of the market over the past 1–3 months,
- VIX or another implied-volatility index,
- dividend yield / earnings yield, term spread, default spread,
- recent drawdown or 12-month momentum of the market.

Keep the regression on the 5 factors, but compute kernel weights from the state vector.
Realised volatility and momentum can be built from the existing data; the others need an
additional data source.

### 6.4 Shrink forecasts toward the historical mean — ⬜

- Combine forecasts: `ŷ = w · ŷ_MCLLR + (1 − w) · ȳ_train`, with `w` tuned inside the fold.
- Optionally apply the Campbell–Thompson restriction (truncate negative equity-premium
  forecasts at zero).

### 6.5 Tune without look-ahead (nested walk-forward) — ✅ implemented (`lrd/improved/tuning.py`)

`TunedModel` runs an inner walk-forward inside every `fit`, so tuning never sees test data.
See §5 for the results.

### 6.6 Use the right tests and economic metrics — ⬜ partly done

- ✅ Diebold–Mariano and OOS R² in `lrd/core/evaluation/significance.py`.
- ⬜ **Clark–West test** for comparisons against the nested historical-mean model
  (add next to Diebold–Mariano, which is conservative in that case).
- ⬜ **Economic value**: certainty-equivalent gains of a mean-variance investor and the
  Sharpe ratio of a market-timing strategy.
- ⬜ Fix `lrd/core/evaluation/backtesting.py` to annualise monthly data with √12 (it
  currently uses √252) and replace the error-based "pseudo-returns" with a real strategy.

### 6.7 Consider forecasting risk instead of returns — ⬜

The project is named *Local Risk Dynamics*. Next-month **volatility** (or factor betas,
or Value-at-Risk) is far more predictable than next-month returns and genuinely changes
with the regime. A locality-based model is much more likely to beat global baselines on
that target. The pipeline needs only a different target in `lrd/core/data/loader.py`.

### 6.8 Engineering for larger studies — ✅ partly done

`ImprovedMCLLR` samples all neighbourhoods at once (Gumbel top-k) and solves the local
fits as one batched system: ~1 ms per query versus ~8 ms for the original (same machine,
50 simulations). Parallelising
across folds (e.g. `joblib`) remains open.

---

## 7. Recommended framing for the write-up

> Stochastic locality on its own does **not** improve point forecasts of monthly market
> excess returns; as originally specified, MCLLR is significantly worse than a ridge
> baseline (−6.6% out-of-sample R²). With local ridge shrinkage, standardised distances
> and predictive variance, and with hyperparameters tuned by nested walk-forward,
> MCLLR becomes the best model tested (+0.36% OOS R² against the historical mean, best
> interval score and predictive likelihood, 94% coverage of nominal 95% intervals), but
> the edge is not statistically significant. Defining locality on economic state
> variables and targeting risk rather than returns are the most promising directions for
> a conclusive result.

---

## 8. Status checklist

| Change | Status |
|---|---|
| Bandwidth on correct scale, MCLLR sampling stability, valid ablations | ✅ Done (standard system) |
| Walk-forward evaluation, `main.py`, Streamlit comparison app | ✅ Done |
| Raw-data validation; duplicated-month corruption found and fixed | ✅ Done (`lrd/core/data/validation.py`) |
| Local ridge, standardised distances (§6.1) | ✅ Done (`lrd/improved`) |
| Predictive variance (§6.2) | ✅ Done (`lrd/improved`) |
| Nested tuning (§6.5) | ✅ Done (`lrd/improved/tuning.py`) |
| State-variable locality (§6.3) | ⬜ Next |
| Shrinkage toward mean (§6.4) | ⬜ Not started |
| Clark–West, economic metrics, backtest fixes (§6.6) | ⬜ Not started |
| Risk (volatility) target (§6.7) | ⬜ Not started |
| Parallel fold evaluation (§6.8) | ⬜ Not started |
