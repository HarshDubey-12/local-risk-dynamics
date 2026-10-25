| Model | System | RMSE | OOS R² (%) | DM p | CW p | 95% cov. | Int. score | CER gain (%/yr) | Sharpe | Folds won |
|---|---|---|---|---|---|---|---|---|---|---|
| MCLLR (improved) | improved | 0.0447 | +0.36 | 0.442 | 0.055 | 94.2% | 0.2345 | +1.22 | 0.54 | 3 |
| Ridge (tuned) | improved | 0.0447 | +0.25 | 0.678 | 0.088 | 94.4% | 0.2364 | +0.45 | 0.49 | 0 |
| MCLLR (regime state) | improved | 0.0447 | +0.15 | 0.787 | 0.121 | 94.4% | 0.2324 | +1.24 | 0.54 | 8 |
| Ridge (α = 1) | standard | 0.0448 | +0.01 | 0.983 | 0.363 | – | – | +0.21 | 0.48 | 2 |
| Historical mean (benchmark) | improved | 0.0448 | +0.00 | 1.000 | 0.500 | 94.0% | 0.2380 | +0.00 | 0.46 | 7 |
| MCLLR ablation: no geometry | standard | 0.0454 | -2.62 | 0.135 | 0.439 | 36.5% | 0.7054 | -0.31 | 0.44 | 4 |
| Monte Carlo Linear | standard | 0.0455 | -3.09 | 0.091 | 0.524 | 37.1% | 0.7056 | +0.00 | 0.46 | 2 |
| Global OLS | standard | 0.0455 | -3.28 | 0.060 | 0.614 | – | – | -0.56 | 0.43 | 1 |
| MCLLR (original) | standard | 0.0462 | -6.58 | 0.011 | 0.709 | 36.3% | 0.7423 | -0.67 | 0.43 | 4 |
| LWLR | standard | 0.0464 | -7.38 | 0.006 | 0.773 | – | – | +0.14 | 0.47 | 3 |
| MCLLR ablation: no stochasticity | standard | 0.0494 | -21.51 | 0.000 | 0.828 | – | – | +0.26 | 0.48 | 8 |
