# Model Assessment: Results, Diagnosis and Conclusions

This document records what the experiments show about the proposed **Monte Carlo Local
Linear Regression (MCLLR)**, why the original specification fails, what the improved system
achieves, how robust that is, and where locality genuinely pays off.

Every number below is reproduced by:

```bash
python scripts/make_report.py        # reports/: main results, seeds, windows, risk target, figures
python scripts/improvement_study.py  # §4: the experiments that motivated the improvements
```

> **Data correction.** Data validation found that the originally committed CSV began with a
> duplicated copy of its header and 68 months (Jul 1963 – Feb 1969), which the loader had used
> twice (giving Feb 1969 a wrong target). The block was removed and the loader now rejects
> duplicated or out-of-order months. All results use the corrected data; the original state is
> preserved in tag `v1-standard`. See `docs/DATA.md`.

---

## Summary

| Question | Answer |
|---|---|
| Does MCLLR work as originally proposed? | **No.** It is significantly worse than ridge (DM p = 0.01) with −6.6% OOS R², and its 95% intervals cover only 36% of outcomes. |
| Can it be fixed? | **Yes, mechanically.** With local ridge shrinkage, standardised distances, predictive variance and nested tuning it becomes the best return forecaster (+0.36% OOS R², calibrated intervals, +1.2%/yr economic value). |
| Is that edge real? | **Weak.** Clark–West p = 0.055 (just short of 5%), stable across random seeds, but **not robust to the training window** (negative at 120 and 360 months). |
| Does locality ever clearly help? | **Yes — for risk.** Forecasting next-month volatility (|return|), the regime-state MCLLR is the best model with **+4.65% OOS R², DM p = 0.001, CW p < 0.001**. |

---

## 1. Evaluation protocol

| Item | Setting |
|---|---|
| Data | Fama–French 5 factors, monthly, Jul 1963 – Oct 2025 (748 forecasting months) |
| Task | Predict the market excess return at *t+1* (or its absolute value, §7) from the five factors at *t* |
| Protocol | Walk-forward: 240-month training window, 12-month test window, yearly refit → **42 folds, 504 test months (Jul 1983 – Jun 2025)**, identical for every model |
| Benchmark | Historical mean of the training window |
| Accuracy | OOS R² = 1 − SSE(model)/SSE(benchmark) |
| Significance | **Clark–West** (one-sided; the appropriate test against the nested historical-mean benchmark) and Diebold–Mariano (two-sided) |
| Calibration | Coverage of 50–99% intervals, interval score, Gaussian NLL |
| Economic value | Mean-variance investor, γ = 3, weight = forecast / (γ σ²) in [0, 1.5], σ² = trailing 60-month variance; certainty-equivalent return (CER) gain and Sharpe ratio versus the same investor using the historical mean |
| Robustness | 5 random seeds for stochastic models; 120/240/360-month training windows |

---

## 2. Main results

Default configurations, seed 42 (`reports/tables/main_results.md`):

| Model | System | OOS R² | DM p | CW p | 95% cov. | Interval score ↓ | CER gain (%/yr) | Sharpe |
|---|---|---|---|---|---|---|---|---|
| **MCLLR (improved)** | improved | **+0.36%** | 0.442 | **0.055** | 94.2% | 0.2345 | +1.22 | 0.54 |
| Ridge (tuned) | improved | +0.25% | 0.678 | 0.088 | 94.4% | 0.2364 | +0.45 | 0.49 |
| MCLLR (regime state) | improved | +0.15% | 0.787 | 0.121 | 94.4% | **0.2324** | **+1.24** | 0.54 |
| Ridge (α = 1) | standard | +0.01% | 0.983 | 0.363 | – | – | +0.21 | 0.48 |
| Historical mean | benchmark | 0.00% | – | – | 94.0% | 0.2380 | 0.00 | 0.46 |
| MCLLR ablation: no geometry | standard | −2.62% | 0.135 | 0.439 | 36.5% | 0.7054 | −0.31 | 0.44 |
| Monte Carlo Linear | standard | −3.09% | 0.091 | 0.524 | 37.1% | 0.7056 | 0.00 | 0.46 |
| Global OLS | standard | −3.28% | 0.060 | 0.614 | – | – | −0.56 | 0.43 |
| **MCLLR (original)** | standard | **−6.58%** | **0.011** | 0.709 | **36.3%** | 0.7423 | −0.67 | 0.43 |
| LWLR | standard | −7.38% | 0.006 | 0.773 | – | – | +0.14 | 0.47 |
| MCLLR ablation: no stochasticity | standard | −21.51% | 0.000 | 0.828 | – | – | +0.26 | 0.48 |

DM and CW p-values are against the historical mean. For the standard models, the significant
DM p-values mean *significantly worse* than the benchmark.

### Conclusions on the standard system

1. **Returns are barely predictable from contemporaneous factors.** No standard model beats the
   historical mean; ridge with α = 1 only matches it because, at this data scale, α = 1 shrinks it
   almost all the way to the mean.
2. **Locality as originally specified hurts.** Both local models (MCLLR, LWLR) are worse than
   global OLS, and the deterministic-neighbourhood ablation is by far the worst model.
3. **The original uncertainty is badly over-confident:** nominal 95% intervals cover 36–37%.
4. **Neither ablation reveals a helpful component:** the full MCLLR is still significantly worse
   than the benchmark.

---

## 3. Why the original MCLLR under-performs

| Cause | Explanation |
|---|---|
| **Variance dominates** | Each local fit estimates 6 coefficients from 80 noisy months with no regularisation. With a near-zero signal, coefficient noise *is* the forecast error. |
| **Locality in the wrong space** | "Similar months" are defined by similar factor returns, which are mostly noise, rather than by the economic regime the hypothesis is about. |
| **Weak signal to localise** | Locality helps when a real relationship changes across regimes; for monthly returns the relationship is close to zero everywhere. |
| **Epistemic-only uncertainty** | The reported spread measures parameter uncertainty and ignores the ~4%-a-month noise in returns. |
| **Raw-scale distances** | Unscaled factors let the most volatile factor dominate the kernel. |

---

## 4. Improvement experiments

`scripts/improvement_study.py`, same 42 folds, seed 0. Hyperparameters here were chosen on the
same test months (selection bias, removed in §5).

| Variant | OOS R² | 95% coverage | DM vs Ridge | Verdict |
|---|---|---|---|---|
| MCLLR (original) | −6.60% | 37% | worse, p = 0.01 | baseline |
| Larger neighbourhood (m = 160) | −4.71% | 20% | worse, p = 0.03 | helps a little |
| Standardised distances only | −5.75% | 19% | worse, p = 0.01 | no help alone |
| Global ridge, tuned | +0.03% | – | equal, p = 0.92 | ≈ mean |
| Bagged ridge (no locality) | −0.02% | 9% | equal, p = 0.93 | ≈ mean |
| **MCLLR + local ridge** | **+0.30%** | 8% | better, p = 0.28 | best accuracy |
| **MCLLR + local ridge + predictive std** | **+0.30%** | **94%** | better, p = 0.28 | best overall |
| Recency-local MC + ridge + predictive std | 0.00% | 94% | equal, p = 0.99 | time locality ≈ mean |

Findings: shrinkage inside each local fit is the decisive change; locality adds value only once
variance is controlled (local ridge beats the same procedure without locality); adding residual
variance fixes calibration (8% → 94% coverage) without changing the point forecast.

---

## 5. Improved system: honest results

The improved system (`lrd/improved`) implements these changes with train-only standardisation
and **nested walk-forward tuning** (inner 180-month train / 12-month test inside each training
window), so no test month influences any choice. Its models report predictive standard
deviations, so every model is scored on calibration.

- **ImprovedMCLLR is the best return forecaster:** +0.36% OOS R², best Gaussian NLL, calibrated
  intervals (94.2%), and a **CER gain of +1.22% per year** for a market-timing investor
  (Sharpe 0.54 vs 0.46 for the benchmark investor).
- **Significance is borderline:** Clark–West p = 0.055 (Diebold–Mariano, which is conservative for
  nested models, gives p = 0.44). For monthly return forecasting, a CW p-value near 5% with a
  positive economic value is a respectable result, but not a conclusive one.
- **Tuning is unstable:** the inner search picks α = 1 in 22 folds and α = 30 in 16, and bandwidth
  3 in 23 folds versus 1 in 15 — the inner validation error is nearly flat, a sign of a weak signal.

---

## 6. Robustness

### 6.1 Random seeds (5 seeds, `reports/tables/seed_robustness.md`)

| Model | Mean OOS R² | SD | Range | Mean CW p | Mean coverage | Mean CER gain |
|---|---|---|---|---|---|---|
| Monte Carlo Linear | −3.55% | 0.96 | −4.83 … −2.49 | 0.594 | 38.3% | −0.46 |
| MCLLR (original) | −6.05% | 0.34 | −6.60 … −5.77 | 0.633 | 37.1% | −0.25 |
| MCLLR ablation: no geometry | −3.42% | 0.37 | −3.92 … −2.91 | 0.582 | 38.8% | −0.88 |
| **MCLLR (improved)** | **+0.25%** | **0.06** | **+0.16 … +0.34** | 0.089 | 94.3% | +1.00 |
| MCLLR (regime state) | +0.15% | 0.09 | +0.06 … +0.25 | 0.124 | 94.4% | +1.07 |

The ranking is not a seed artefact: the improved models are positive for every seed and far more
stable than the standard Monte Carlo models (whose results move by up to 2.3 points between
seeds). The seed-42 result (+0.36%) is at the favourable end of the improved MCLLR's range.

### 6.2 Training window (`reports/tables/window_sensitivity.md`)

| Model | 120 months | 240 months | 360 months |
|---|---|---|---|
| Ridge (tuned) | −1.23% | +0.25% | −0.14% |
| **MCLLR (improved)** | −0.51% | **+0.36%** | −0.13% |
| MCLLR (regime state) | −0.83% | +0.15% | −0.16% |
| MCLLR (original) | −5.93% | −6.58% | −6.16% |
| Global OLS | −6.97% | −3.28% | −3.15% |

**The positive return-forecasting result depends on the 240-month window.** With 120 or 360
months every improved model falls slightly below the historical mean (CW p ≈ 0.4). The robust
conclusions are the relative ones: the improvements close almost all of the gap between the
original MCLLR and the benchmark at every window, but beating the benchmark for returns is not
robust.

---

## 7. Where locality pays off: forecasting risk

Forecasting the **absolute** next-month market excess return (a simple volatility proxy;
`experiments/improved/config_risk.yaml`, same folds):

| Model | OOS R² | DM p | CW p | 95% coverage | Interval score ↓ |
|---|---|---|---|---|---|
| Historical mean | 0.00% | – | – | 95.0% | 0.1632 |
| Ridge (tuned) | +2.66% | 0.085 | < 0.001 | 96.0% | 0.1577 |
| MCLLR (improved) | +3.45% | 0.016 | < 0.001 | 94.6% | 0.1605 |
| **MCLLR (regime state)** | **+4.65%** | **0.001** | **< 0.001** | 95.0% | **0.1575** |

- Volatility clusters in regimes, so this is where the original hypothesis should hold — and it does.
  The **regime-state MCLLR**, whose neighbourhoods are months with similar trailing volatility and
  momentum, is the best model and **significantly beats the benchmark by both tests**.
- Locality adds over global shrinkage: both MCLLR variants have a higher OOS R² than tuned ridge,
  the regime variant by the widest margin (the models were not tested pairwise against ridge).
  A linear global model cannot exploit the *size* of this month's return (it is linear in the
  signed factors); the kernel can.
- The regime model's advantage builds steadily from the early 1990s, with a large jump in 2020
  (`reports/figures/fig7_risk_target.png`), so it is not driven by a single episode.

---

## 8. Conclusions

1. **Stochastic locality on noisy factor returns does not improve monthly return forecasts.** As
   proposed, MCLLR is significantly worse than simple benchmarks.
2. **The failure is diagnosable and largely fixable.** Local ridge shrinkage, standardised distances,
   predictive variance and honest tuning turn MCLLR into the best return forecaster tested, with
   calibrated uncertainty and positive economic value — but its edge over the historical mean is
   borderline (CW p ≈ 0.06) and not robust to the training window.
3. **Locality works where regimes matter.** For risk (volatility) forecasting, MCLLR with locality
   defined on the market regime is significantly better than the benchmark (+4.65% OOS R²,
   p ≤ 0.001) and more accurate than global ridge.

Suggested framing:

> Stochastic local linear models do not reliably beat the historical mean for monthly equity-premium
> forecasts; as originally specified, MCLLR is significantly worse. With local shrinkage, predictive
> variance and nested tuning it becomes the best return model tested (+0.36% OOS R², Clark–West
> p = 0.055, +1.2% annual certainty-equivalent gain), though this edge is sensitive to the
> estimation window. When the target is risk rather than return, MCLLR with locality defined on the
> market regime significantly outperforms the historical mean and has the highest accuracy of all
> models, ahead of global ridge (+4.65% OOS R², Diebold–Mariano p = 0.001), supporting the hypothesis
> that locality captures regime-dependent *risk* dynamics.

---

## 9. Limitations and further work

- **Return forecasts:** the positive result is window-dependent; a combination or shrinkage of the
  forecast toward the historical mean (Campbell–Thompson restrictions) could add robustness.
- **Risk target:** |r| is a noisy volatility proxy; daily data would allow realised volatility and
  standard volatility benchmarks (e.g. GARCH, HAR), which a full risk study should include.
- **State variables:** the regime state uses only trailing market volatility and momentum; valuation
  ratios, interest-rate spreads or implied volatility (VIX) need external data.
- **Single market and frequency:** US monthly factors only.
- **Multiple comparisons:** many models and variants were compared; the risk result (p ≤ 0.001)
  survives a Bonferroni correction for the four risk models, the return result does not.

---

## 10. Status checklist

| Item | Status |
|---|---|
| Correct bandwidth scale, MCLLR stability, valid ablations | ✅ |
| Walk-forward evaluation, CLI, dashboard | ✅ |
| Raw-data validation; duplicated-month corruption fixed | ✅ |
| Local ridge, standardised distances, predictive variance | ✅ `lrd/improved/models/mcllr.py` |
| Nested walk-forward tuning | ✅ `lrd/improved/tuning.py` |
| Regime-state locality | ✅ `lrd/improved/models/regime.py` |
| Risk (volatility) target | ✅ `experiments/improved/config_risk.yaml` |
| Clark–West test, economic value, corrected backtesting (√12) | ✅ `lrd/core/evaluation/` |
| Seed and window robustness | ✅ `scripts/make_report.py` |
| Pinned dependencies, CI, licence, data citation | ✅ |
| Forecast combination / Campbell–Thompson restrictions | ⬜ further work |
| Realised-volatility target from daily data, GARCH/HAR benchmarks | ⬜ further work |
