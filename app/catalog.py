"""Dashboard content: model descriptions, practices and metric definitions.

Model entries are keyed by ``"<system>:<model_type>"``, matching the keys of
the default runs produced by ``lrd.comparison``.
"""

from __future__ import annotations

MODELS: dict[str, dict] = {
    # ------------------------------------------------------------- standard
    "standard:global_linear": {
        "label": "Global OLS",
        "family": "Stationary", "locality": "None", "uncertainty": "None", "local": False,
        "short": "One set of factor sensitivities for the whole training window.",
        "idea": "The control model: it assumes the link between this month's factors and next month's "
                "market return never changes, and fits one least-squares line through all training months.",
        "steps": ["Add an intercept column to the factor matrix.",
                  "Solve ordinary least squares with a pseudo-inverse (numerically stable).",
                  "Predict every test month with the same coefficients."],
        "formula": r"\hat\beta=(X^\top X)^{+}X^\top y,\qquad \hat y(x_0)=x_0^\top\hat\beta",
        "params": {"fit_intercept": "Include a constant term."},
        "pros": "Fast, stable and interpretable.",
        "cons": "Cannot adapt if relationships drift between regimes; noisy coefficients overfit.",
        "cost": "Fit O(n·p²) once; predict O(p) per month.",
    },
    "standard:ridge_linear": {
        "label": "Ridge (α = 1)",
        "family": "Stationary, regularised", "locality": "None", "uncertainty": "None", "local": False,
        "short": "Global OLS with an L2 penalty on raw-scale factors.",
        "idea": "Shrinks the slopes toward zero to trade a little bias for much less variance. On raw "
                "decimal returns XᵀX ≈ 1, so α = 1 is very strong shrinkage — close to forecasting the mean.",
        "steps": ["Add an (unpenalised) intercept.", "Solve the penalised normal equations.",
                  "Predict with the shrunken coefficients."],
        "formula": r"\hat\beta=(X^\top X+\alpha I')^{-1}X^\top y",
        "params": {"alpha": "Penalty strength (absolute, raw factor scale)."},
        "pros": "Robust to noise and correlated factors; the strongest standard baseline.",
        "cons": "Penalty depends on the data scale; no uncertainty estimate.",
        "cost": "Same as OLS.",
    },
    "standard:lwlr": {
        "label": "LWLR",
        "family": "Deterministic local", "locality": "Gaussian kernel weights", "uncertainty": "None", "local": True,
        "short": "Refits a kernel-weighted regression around every query month.",
        "idea": "Training months whose factor returns resemble the query month get large weights, so the "
                "coefficients can vary smoothly across market regimes.",
        "steps": ["fit() only stores the training data (lazy learner).",
                  "For each query x₀, weight every training month by a Gaussian kernel of its distance.",
                  "Solve weighted least squares and predict x₀."],
        "formula": r"w_i=\exp\!\Big(-\tfrac{\lVert x_i-x_0\rVert^2}{2\tau^2}\Big),\quad "
                   r"\hat\beta(x_0)=(X^\top WX)^{+}X^\top Wy",
        "params": {"bandwidth (τ)": "Neighbourhood size on raw decimal returns (≈0.03–0.2 is local; ≥1 is global)."},
        "pros": "Captures regime-dependent sensitivities without a parametric form.",
        "cons": "High variance: few effective observations per query, no shrinkage.",
        "cost": "One weighted solve per query: O(n·p²).",
    },
    "standard:mc_linear": {
        "label": "Monte Carlo Linear",
        "family": "Stochastic global", "locality": "None", "uncertainty": "Spread across subsample fits", "local": False,
        "short": "Many OLS fits on random subsamples; their spread is the uncertainty.",
        "idea": "Re-fitting OLS on random subsets of history shows how unstable the global coefficients are. "
                "The ensemble mean is the forecast, the ensemble spread the uncertainty.",
        "steps": ["Draw K random subsamples of m months (without replacement).",
                  "Fit OLS on each (once, at fit time).",
                  "Predict with every coefficient vector; report mean and standard deviation."],
        "formula": r"\hat y(x_0)=\tfrac1K\textstyle\sum_k x_0^\top\hat\beta_k,\quad \hat\sigma=\operatorname{sd}_k(x_0^\top\hat\beta_k)",
        "params": {"n_simulations (K)": "Number of subsamples.", "subsample_size (m)": "Months per subsample."},
        "pros": "Cheap estimate of parameter uncertainty.",
        "cons": "Spread ignores return noise, so 95% intervals cover far fewer than 95% of outcomes.",
        "cost": "Fit O(K·m·p²); predict O(K·p).",
    },
    "standard:mcllr": {
        "label": "MCLLR (original)",
        "family": "Stochastic local", "locality": "Kernel-weighted sampling", "uncertainty": "Spread across neighbourhoods", "local": True,
        "short": "The proposed model: samples kernel-weighted neighbourhoods and fits OLS on each.",
        "idea": "Locality is enforced by sampling rather than weighting: months similar to the query are more "
                "likely to be drawn into each neighbourhood. Repeating the draw gives a local forecast and a "
                "local uncertainty.",
        "steps": ["Compute Gaussian kernel weights to every training month (relative to the nearest month).",
                  "Normalise them into sampling probabilities.",
                  "K times: sample m months without replacement, fit plain OLS, predict x₀.",
                  "Report the mean and standard deviation of the K predictions."],
        "formula": r"p_i=\tfrac{w_i}{\sum_j w_j},\quad S_k\sim\text{Sample}(m;p),\quad \hat y=\tfrac1K\textstyle\sum_k x_0^\top\hat\beta_{S_k}",
        "params": {"bandwidth (τ)": "Concentration of the sampling distribution (raw scale).",
                   "n_simulations (K)": "Neighbourhoods per query.", "subsample_size (m)": "Months per neighbourhood."},
        "pros": "Local forecasts with a query-specific uncertainty estimate.",
        "cons": "Unregularised local fits are noisy; uncertainty is epistemic only. Significantly worse than Ridge.",
        "cost": "K regressions per query (~8 ms).",
    },
    "standard:mcllr_no_geometry": {
        "label": "MCLLR ablation: no geometry",
        "family": "Ablation (stochastic global)", "locality": "None (uniform sampling)", "uncertainty": "Spread across subsamples", "local": False,
        "short": "MCLLR with the kernel removed — every month equally likely.",
        "idea": "Isolates what the kernel contributes by sampling uniformly.",
        "steps": ["For each query, draw K uniform subsamples of m months.", "Fit OLS on each and predict.",
                  "Report mean and standard deviation."],
        "formula": r"S_k\sim\text{Uniform}(m)",
        "params": {"n_simulations (K)": "Subsamples per query.", "subsample_size (m)": "Months per subsample."},
        "pros": "Clean control for the effect of geometry.",
        "cons": "Equivalent in expectation to Monte Carlo Linear but slower.",
        "cost": "K regressions per query.",
    },
    "standard:mcllr_no_stochasticity": {
        "label": "MCLLR ablation: no stochasticity",
        "family": "Ablation (deterministic local)", "locality": "m nearest months, kernel-weighted", "uncertainty": "None", "local": True,
        "short": "Deterministic: kernel-weighted fit on the m nearest months.",
        "idea": "Isolates what Monte Carlo sampling contributes by replacing it with a fixed neighbourhood.",
        "steps": ["Keep the m months with the highest kernel weight.", "Fit kernel-weighted least squares.",
                  "Predict the query."],
        "formula": r"\hat\beta=(X_N^\top W_NX_N)^{+}X_N^\top W_Ny_N,\ N=m\text{ nearest}",
        "params": {"bandwidth (τ)": "Weighting inside the neighbourhood.", "subsample_size (m)": "Neighbourhood size."},
        "pros": "Deterministic and cheaper than MCLLR.",
        "cons": "Hard cut-off with few points overfits noise — the worst standard model.",
        "cost": "One sort and one solve per query.",
    },
    # ------------------------------------------------------------- improved
    "improved:historical_mean": {
        "label": "Historical mean (benchmark)",
        "family": "Benchmark", "locality": "None", "uncertainty": "Training standard deviation", "local": False,
        "short": "Forecasts the training-window average return — the bar every model must clear.",
        "idea": "The standard benchmark in return predictability. Out-of-sample R² and Diebold–Mariano tests "
                "of every model are measured against it.",
        "steps": ["Average the target over the training window.", "Use that mean as every forecast.",
                  "Use the training standard deviation as the uncertainty."],
        "formula": r"\hat y=\bar y_{\text{train}},\quad \hat\sigma=s_{\text{train}}",
        "params": {},
        "pros": "Impossible to overfit; surprisingly hard to beat for monthly returns.",
        "cons": "Ignores all factor information.",
        "cost": "Trivial.",
    },
    "improved:ridge": {
        "label": "Ridge (tuned)",
        "family": "Stationary, regularised", "locality": "None", "uncertainty": "Residual standard deviation", "local": False,
        "short": "Ridge on standardised factors, α chosen per fold by nested walk-forward.",
        "idea": "The best global baseline done properly: factors are standardised on the training window, the "
                "penalty is relative to sample size, and α is tuned inside each fold without seeing test data.",
        "steps": ["Standardise factors with training-window statistics.",
                  "Inner walk-forward on the training window picks α from the grid.",
                  "Refit with the chosen α; predict with residual-based uncertainty."],
        "formula": r"\hat\beta=(Z^\top Z+\alpha n I')^{-1}Z^\top y,\quad \hat\sigma=\sqrt{\tfrac{\lVert y-Z\hat\beta\rVert^2}{n-p}}",
        "params": {"alpha": "Penalty relative to sample size (tuned per fold)."},
        "pros": "Honest tuning, scale-free penalty, calibrated intervals.",
        "cons": "Still a single global relationship.",
        "cost": "Grid × inner folds solves per outer fold (milliseconds).",
    },
    "improved:mcllr_improved": {
        "label": "MCLLR (improved)",
        "family": "Stochastic local, regularised", "locality": "Kernel-weighted sampling (standardised)", "uncertainty": "Predictive (spread + residual variance)", "local": True,
        "short": "MCLLR with local ridge, standardised distances, predictive variance and nested tuning.",
        "idea": "Keeps MCLLR's stochastic locality but fixes why it failed: each local fit is a ridge "
                "regression, distances use standardised factors, the uncertainty adds local residual variance, "
                "and bandwidth/α are tuned per fold.",
        "steps": ["Standardise factors with training-window statistics.",
                  "For each query, compute kernel log-weights to every training month.",
                  "Draw all K neighbourhoods at once (Gumbel top-k sampling without replacement).",
                  "Solve the K local ridge regressions as one batched system.",
                  "Report the mean prediction and √(Monte Carlo variance + mean residual variance).",
                  "Bandwidth and α are chosen per fold by an inner walk-forward."],
        "formula": r"\hat\beta_{S_k}=(Z_{S_k}^\top Z_{S_k}+\alpha m I')^{-1}Z_{S_k}^\top y_{S_k},\quad "
                   r"\hat\sigma^2=\operatorname{var}_k(\hat y_k)+\overline{s^2_{\text{resid}}}",
        "params": {"bandwidth": "Kernel width in standard deviations (tuned).",
                   "alpha": "Local ridge penalty relative to m (tuned).",
                   "n_simulations (K)": "Neighbourhoods per query.", "subsample_size (m)": "Months per neighbourhood.",
                   "predictive_std": "Add residual variance to the uncertainty."},
        "pros": "Best model tested on accuracy and uncertainty scores; calibrated 95% intervals; ~1 ms per query.",
        "cons": "Edge over the benchmark is not statistically significant; tuned values vary between folds.",
        "cost": "K batched ridge solves per query.",
    },
}

SYSTEMS = {"standard": "Standard system", "improved": "Improved system"}


def model_info(system: str, model_type: str) -> dict:
    return MODELS[f"{system}:{model_type}"]


PRACTICES = [
    ("Chronological evaluation", "Folds never shuffle time: every forecast uses only earlier months."),
    ("One set of folds for all models", "Both systems are scored on identical test months, so differences come from the models."),
    ("Train-only preparation", "Scaling and clipping are learned on each fold's training window, never on test data."),
    ("Honest tuning", "Hyperparameters are chosen by an inner walk-forward inside the training window."),
    ("Outliers kept", "Crash months are flagged, not deleted — they decide most rankings."),
    ("Benchmark and significance", "Every model is compared with the historical mean using out-of-sample R² and a Diebold–Mariano test."),
    ("Calibration", "Uncertainty is checked: a 95% interval should contain about 95% of outcomes."),
    ("Validated data", "The raw file is checked for missing, garbage, duplicate and missing months before use."),
]

METRICS = {
    "rmse": ("RMSE", "Root mean squared forecast error (decimal return). Lower is better."),
    "mae": ("MAE", "Mean absolute forecast error. Lower is better."),
    "oos_r2_%": ("OOS R² vs mean", "1 − SSE(model)/SSE(historical mean), in %. Above 0 beats the benchmark."),
    "dm_p": ("DM p-value", "Diebold–Mariano test vs the benchmark; below 0.05 would be significant."),
    "coverage_95": ("95% coverage", "Share of outcomes inside the 95% interval. Target ≈ 95%."),
    "interval_score_95": ("Interval score", "Width plus penalties for misses (proper score). Lower is better."),
    "gaussian_nll": ("Gaussian NLL", "Predictive log-loss; punishes over-confidence. Lower is better."),
    "folds_won": ("Folds won", "Walk-forward folds where this model had the lowest RMSE."),
}
