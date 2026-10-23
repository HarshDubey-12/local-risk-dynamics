"""
Local Risk Dynamics - interactive explainer and model comparison.

Run locally:
    streamlit run app/streamlit_app.py

Pages
-----
Overview     research question, model spectrum, end-to-end pipeline
Pipeline     each step (data, split, fitting, metrics, walk-forward, outputs)
Models       how every model works, plus an interactive locality explorer
Comparison   run models with your own settings and compare the results
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)  # dataset/cache paths in the codebase are project-relative
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import build_predictions_frame  # noqa: E402
from lrd.standard.config import ExperimentConfig, ModelCofig  # noqa: E402
from lrd.core.data.loader import FACTOR_COLUMNS, load_fama_french  # noqa: E402
from lrd.core.evaluation.rolling import walk_forward_splits  # noqa: E402
from lrd.standard.experiments.rolling_runner import RollingExperimentRunner  # noqa: E402
from lrd.standard.experiments.runner import ExperimentRunner  # noqa: E402

DATA_PATH = "data/raw/F-F_Research_Data_5_Factors_2x3.csv"

# ---------------------------------------------------------------------------
# Model catalogue (single source for descriptions used across pages)
# ---------------------------------------------------------------------------

MODELS = {
    "global_linear": {
        "label": "Global Linear (OLS)",
        "family": "Stationary",
        "locality": "None",
        "uncertainty": "None",
        "summary": "One set of factor sensitivities for the whole history.",
        "idea": (
            "The control model. It assumes the relationship between today's factor "
            "returns and next month's market return never changes, and fits a single "
            "ordinary-least-squares line through all training months."
        ),
        "steps": [
            "Add an intercept column to the factor matrix X.",
            "Solve for the coefficients with the Moore-Penrose pseudo-inverse (numerically stable OLS).",
            "Predict every query month with the same coefficients.",
        ],
        "formula": r"\hat\beta = (X^\top X)^{+} X^\top y,\qquad \hat y(x_0) = x_0^\top \hat\beta",
        "params": {"fit_intercept": "Include a constant term (default true)."},
        "pros": "Fast, stable, interpretable; the benchmark every other model must beat.",
        "cons": "Cannot adapt if factor sensitivities drift across regimes.",
        "cost": "Fit O(n·p²) once; predict O(p) per query.",
    },
    "ridge_linear": {
        "label": "Ridge Linear",
        "family": "Stationary (regularised)",
        "locality": "None",
        "uncertainty": "None",
        "summary": "Global OLS with an L2 penalty that shrinks coefficients toward zero.",
        "idea": (
            "A regularised global baseline. Shrinking the slopes trades a little bias for "
            "less variance, which helps when the signal is weak - as it is for monthly "
            "return forecasts."
        ),
        "steps": [
            "Add an intercept column (the intercept is not penalised).",
            "Solve the penalised normal equations.",
            "Predict with the shrunken coefficients.",
        ],
        "formula": r"\hat\beta = (X^\top X + \alpha I')^{-1} X^\top y",
        "params": {"alpha": "Penalty strength; larger means more shrinkage (default 1.0)."},
        "pros": "Robust to noise and correlated factors.",
        "cons": (
            "Factor returns are small decimals, so XᵀX has diagonal ≈ n·var ≈ 1. "
            "alpha = 1 is therefore heavy shrinkage - close to predicting the training mean."
        ),
        "cost": "Same as OLS.",
    },
    "lwlr": {
        "label": "Locally Weighted LR (LWLR)",
        "family": "Deterministic local",
        "locality": "Gaussian kernel weights",
        "uncertainty": "None",
        "summary": "Refits a weighted regression around every query point.",
        "idea": (
            "Deterministic locality. Training months whose factor returns look similar to "
            "the query month get large weights, dissimilar months get small weights, so the "
            "coefficients can vary smoothly across market regimes."
        ),
        "steps": [
            "fit() only stores the training data (a lazy learner).",
            "For each query x₀ compute Gaussian weights from the squared distance to every training month.",
            "Solve weighted least squares with those weights.",
            "Predict x₀ with its own local coefficients.",
        ],
        "formula": (
            r"w_i = \exp\!\Big(-\frac{\lVert x_i - x_0\rVert^2}{2\tau^2}\Big),\qquad "
            r"\hat\beta(x_0) = (X^\top W X)^{+} X^\top W y"
        ),
        "params": {
            "bandwidth (τ)": "Neighbourhood size. Large τ → all weights ≈ 1 → identical to global OLS; "
            "small τ → only the nearest months matter (noisier).",
        },
        "pros": "Captures regime-dependent sensitivities without assuming a parametric form.",
        "cons": "A weighted solve per query; bandwidth must match the data scale.",
        "cost": "Predict O(n·p²) per query.",
    },
    "mc_linear": {
        "label": "Monte Carlo Linear",
        "family": "Stochastic global",
        "locality": "None",
        "uncertainty": "Spread across random subsamples",
        "summary": "Many OLS fits on random subsamples; the spread measures uncertainty.",
        "idea": (
            "Stochastic but global. Re-fitting OLS on many random subsets of history shows "
            "how unstable the global coefficients are; the ensemble mean is the forecast and "
            "the ensemble spread is the uncertainty."
        ),
        "steps": [
            "Draw K subsamples of m training months uniformly, without replacement.",
            "Fit OLS on each subsample (done once, at fit time).",
            "Predict the query with all K coefficient vectors.",
            "Return the mean as the forecast and the standard deviation as uncertainty.",
        ],
        "formula": (
            r"\hat y(x_0) = \frac{1}{K}\sum_{k=1}^{K} x_0^\top \hat\beta_k,\qquad "
            r"\hat\sigma(x_0) = \operatorname{sd}_k\big(x_0^\top \hat\beta_k\big)"
        ),
        "params": {
            "n_simulations (K)": "Number of subsamples.",
            "subsample_size (m)": "Months per subsample (1 < m ≤ n).",
        },
        "pros": "Cheap uncertainty estimate; reveals coefficient instability.",
        "cons": (
            "The spread only reflects estimation uncertainty, not the noise in returns, "
            "so 95% intervals cover far fewer than 95% of outcomes."
        ),
        "cost": "Fit O(K·m·p²); predict O(K·p) per query.",
    },
    "mcllr": {
        "label": "MCLLR (proposed)",
        "family": "Stochastic local",
        "locality": "Kernel-weighted sampling",
        "uncertainty": "Spread across sampled neighbourhoods",
        "summary": "Samples local neighbourhoods by kernel probability and fits OLS on each.",
        "idea": (
            "The proposed hybrid. Locality is enforced through *sampling* rather than "
            "regression weights: similar months are more likely to be drawn into each "
            "neighbourhood. Repeating the draw gives both a local forecast and a local "
            "uncertainty estimate."
        ),
        "steps": [
            "For each query x₀ compute Gaussian kernel weights to every training month "
            "(relative to the nearest month, so they never underflow).",
            "Normalise the weights into sampling probabilities p_i = w_i / Σw.",
            "For K simulations: sample m months without replacement using p, fit plain OLS, predict x₀.",
            "Return the mean and standard deviation of the K predictions.",
        ],
        "formula": (
            r"p_i(x_0) = \frac{w_i(x_0)}{\sum_j w_j(x_0)},\quad S_k \sim \text{Sample}(m; p),\quad "
            r"\hat y(x_0) = \frac{1}{K}\sum_k x_0^\top \hat\beta_{S_k}"
        ),
        "params": {
            "bandwidth (τ)": "How concentrated the sampling distribution is.",
            "n_simulations (K)": "Neighbourhoods drawn per query.",
            "subsample_size (m)": "Months per neighbourhood.",
        },
        "pros": "Local forecasts with a built-in, query-specific uncertainty estimate.",
        "cons": "Most expensive model: K regressions for every query point.",
        "cost": "Predict O(K·(n·p + m·p²)) per query.",
    },
    "mcllr_no_geometry": {
        "label": "MCLLR ablation: no geometry",
        "family": "Ablation (stochastic global)",
        "locality": "None (uniform sampling)",
        "uncertainty": "Spread across random subsamples",
        "summary": "MCLLR with the kernel removed - every month is equally likely.",
        "idea": (
            "Removes geometric locality from MCLLR by sampling uniformly. Comparing it "
            "with MCLLR isolates what the kernel contributes."
        ),
        "steps": [
            "For each query, draw K uniform subsamples of m months.",
            "Fit OLS on each and predict the query.",
            "Return mean and standard deviation.",
        ],
        "formula": r"S_k \sim \text{Uniform}(m),\qquad \hat y(x_0) = \frac{1}{K}\sum_k x_0^\top \hat\beta_{S_k}",
        "params": {"n_simulations (K)": "Subsamples per query.", "subsample_size (m)": "Months per subsample."},
        "pros": "Clean control for the effect of the kernel.",
        "cons": "Equivalent in expectation to Monte Carlo Linear, but resamples per query (slower).",
        "cost": "Predict O(K·m·p²) per query.",
    },
    "mcllr_no_stochasticity": {
        "label": "MCLLR ablation: no stochasticity",
        "family": "Ablation (deterministic local)",
        "locality": "m nearest months, kernel-weighted",
        "uncertainty": "None",
        "summary": "Deterministic: kernel-weighted fit on the m nearest months.",
        "idea": (
            "Removes the randomness from MCLLR. Each query uses its m nearest months, "
            "weighted by the same Gaussian kernel. Comparing it with MCLLR isolates what "
            "the Monte Carlo sampling contributes."
        ),
        "steps": [
            "For each query, compute kernel weights and keep the m highest-weighted months.",
            "Fit kernel-weighted least squares on that neighbourhood.",
            "Predict the query.",
        ],
        "formula": r"\hat\beta(x_0) = (X_N^\top W_N X_N)^{+} X_N^\top W_N y_N,\quad N = \text{m nearest}",
        "params": {"bandwidth (τ)": "Weighting within the neighbourhood.", "subsample_size (m)": "Neighbourhood size."},
        "pros": "Deterministic and cheaper than MCLLR.",
        "cons": "A hard neighbourhood cut-off with few points can overfit noise.",
        "cost": "Predict O(n log n + m·p²) per query.",
    },
}
MODEL_ORDER = list(MODELS)
STOCHASTIC = {"mc_linear", "mcllr", "mcllr_no_geometry"}

METRICS = [
    ("rmse", "Root mean squared error", r"\sqrt{\tfrac1n\sum (y-\hat y)^2}", "Lower is better. In return units (0.04 ≈ 4% a month)."),
    ("mae", "Mean absolute error", r"\tfrac1n\sum |y-\hat y|", "Lower is better; less sensitive to crash months than RMSE."),
    ("r2", "Out-of-sample R²", r"1 - \frac{\sum (y-\hat y)^2}{\sum (y-\bar y)^2}", "Above 0 means the model beats the test-period mean. Negative values are common for monthly returns."),
    ("explained_variance", "Explained variance", r"1 - \frac{\operatorname{Var}(y-\hat y)}{\operatorname{Var}(y)}", "Like R² but ignores a constant bias."),
    ("median_ae", "Median absolute error", r"\operatorname{median}|y-\hat y|", "Typical error in an ordinary month."),
    ("mape", "Mean absolute percentage error", r"\tfrac1n\sum \left|\frac{y-\hat y}{y}\right|", "Reported for completeness - unreliable because returns are often near zero."),
    ("coverage_95", "95% interval coverage", r"\tfrac1n\sum \mathbf 1\{|y-\hat y| \le 1.96\,\hat\sigma\}", "Stochastic models only. Should be ≈ 0.95 if the uncertainty is calibrated."),
    ("interval_width_95", "95% interval width", r"\tfrac1n\sum 2\cdot1.96\,\hat\sigma", "Stochastic models only. Narrower is better at equal coverage."),
    ("gaussian_nll", "Gaussian negative log-likelihood", r"\tfrac1n\sum \tfrac12\log(2\pi\hat\sigma^2)+\tfrac{(y-\hat y)^2}{2\hat\sigma^2}", "Stochastic models only. Lower is better; punishes over-confidence."),
    ("interval_score_95", "Interval score (95%)", r"\text{width} + \tfrac{2}{\alpha}\,\text{miss distance}", "Stochastic models only. Proper score: lower is better."),
]


# ---------------------------------------------------------------------------
# Cached data and experiment runs
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data() -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    return load_fama_french(DATA_PATH)


@st.cache_data(show_spinner=False)
def raw_preview(n_lines: int = 10) -> str:
    with open(DATA_PATH) as f:
        return "".join(f.readline() for _ in range(n_lines))


def build_config(models: tuple[str, ...], params: dict, train_ratio: float, seed: int) -> ExperimentConfig:
    model_configs = []
    for model_type in models:
        kwargs = {"model_type": model_type, "random_state": seed}
        if model_type in {"lwlr", "mcllr", "mcllr_no_stochasticity"}:
            kwargs["bandwidth"] = params["bandwidth"]
        if model_type in {"mc_linear", "mcllr", "mcllr_no_geometry"}:
            kwargs["n_simulations"] = params["n_simulations"]
        if model_type in {"mc_linear", "mcllr", "mcllr_no_geometry", "mcllr_no_stochasticity"}:
            kwargs["subsample_size"] = params["subsample_size"]
        if model_type == "ridge_linear":
            kwargs["alpha"] = params["alpha"]
        model_configs.append(ModelCofig(**kwargs))
    return ExperimentConfig(
        dataset_path=DATA_PATH, train_ratio=train_ratio, models=model_configs, random_state=seed
    )


@st.cache_data(show_spinner=False)
def run_holdout(models: tuple[str, ...], params: tuple, train_ratio: float, seed: int):
    runner = ExperimentRunner(build_config(models, dict(params), train_ratio, seed))
    metrics = runner.run()
    return metrics, build_predictions_frame(runner)


@st.cache_data(show_spinner=False)
def run_walk_forward(
    models: tuple[str, ...], params: tuple, seed: int, train_window: int, test_window: int, step: int
):
    config = build_config(models, dict(params), 0.8, seed)
    folds, _ = RollingExperimentRunner(config, train_window, test_window, step).run()
    _, _, dates = load_data()
    folds["test_end_date"] = dates[folds["test_end"].to_numpy() - 1]
    return folds


def label(model_type: str) -> str:
    return MODELS[model_type]["label"]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview() -> None:
    st.title("📈 Local Risk Dynamics")
    st.markdown(
        "#### Can regime-dependent financial risk be approximated by many small, *local* linear models instead of one global one?"
    )
    st.markdown(
        "Markets are non-stationary: the link between risk factors and returns changes "
        "over time. This project forecasts **next month's US market excess return** from "
        "this month's five **Fama-French factors** and compares seven linear models that "
        "differ in *how they define locality* and *whether they quantify uncertainty*."
    )

    st.subheader("The modelling spectrum")
    spectrum = st.columns(4)
    for col, (title, models, text) in zip(
        spectrum,
        [
            ("1 · Stationary", "Global OLS, Ridge", "One relationship for all time."),
            ("2 · Deterministic local", "LWLR", "Weight similar months more heavily."),
            ("3 · Stochastic global", "Monte Carlo Linear", "Random subsamples → uncertainty."),
            ("4 · Stochastic local", "MCLLR (proposed)", "Sample similar months → local forecast + uncertainty."),
        ],
    ):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(models)
            st.write(text)

    st.subheader("End-to-end pipeline")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR; bgcolor="transparent";
            node [shape=box, style="rounded,filled", fillcolor="#eef3fb", color="#4c78a8", fontname="Helvetica", fontsize=11];
            edge [color="#888888"];
            cfg  [label="1. Config\\nYAML / UI settings"];
            data [label="2. Data\\nFama-French 5 factors\\n% → decimals, target = t+1"];
            split[label="3. Split\\nchronological\\ntrain / test"];
            fit  [label="4. Fit models\\n7 linear variants"];
            eval [label="5. Evaluate\\naccuracy + calibration"];
            wf   [label="5b. Walk-forward\\nrolling refits"];
            out  [label="6. Outputs\\nCSV · figures · run log"];
            cfg -> data -> split -> fit -> eval -> out;
            split -> wf -> out;
        }
        """
    )
    st.caption("Each box is explained on the **Pipeline** page; each model on the **Models** page.")

    st.subheader("Headline result (default settings, hold-out test)")
    with st.spinner("Running the default experiment…"):
        metrics, _ = run_holdout(tuple(MODEL_ORDER), DEFAULT_PARAMS_TUPLE, 0.8, 42)
    best = metrics.sort_values("rmse").iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Lowest RMSE ({best['rmse']:.4f})", label(best["model"]))
    c2.metric("Models with R² > 0", f"{int((metrics['r2'] > 0).sum())} of {len(metrics)}")
    stochastic = metrics.dropna(subset=["coverage_95"])
    c3.metric("Best 95% interval coverage (target 95%)", f"{stochastic['coverage_95'].max():.0%}")
    st.info(
        "Monthly market returns are very hard to predict: no model beats the test-period "
        "mean (all R² < 0), heavy shrinkage (Ridge) does best, and adding locality does not "
        "improve point forecasts. The Monte Carlo uncertainty bands are too narrow because "
        "they capture estimation uncertainty only. Explore why on the **Comparison** page."
    )


def page_pipeline() -> None:
    st.title("🔧 Pipeline, step by step")
    X, y, dates = load_data()
    tabs = st.tabs(
        ["1 · Data", "2 · Train/test split", "3 · Model fitting", "4 · Evaluation metrics", "5 · Walk-forward", "6 · Outputs"]
    )

    with tabs[0]:
        st.markdown(
            "**Source:** Kenneth French's *5 Factors (2x3)* monthly file "
            f"(`{DATA_PATH}`). The loader (`lrd/core/data/loader.py`):"
        )
        st.markdown(
            "1. Skips the 4 metadata lines and reads the table.\n"
            "2. Keeps only `YYYYMM` rows - this drops the annual-factor section and copyright footer.\n"
            "3. Converts percentages to decimals (`-0.39` → `-0.0039`).\n"
            "4. Builds features **X** = the five factors in month *t* and target **y** = market excess return in month *t + 1*.\n"
            "5. Drops the last month, which has no next-month target - no look-ahead bias."
        )
        with st.expander("Raw file (first lines)"):
            st.code(raw_preview(), language="text")
        frame = pd.DataFrame(X, columns=FACTOR_COLUMNS)
        frame.insert(0, "month t", dates.strftime("%Y-%m"))
        frame["target y (Mkt-RF at t+1)"] = y
        st.markdown(f"**Model-ready data:** {len(y)} months, {dates[0]:%b %Y} → {dates[-1]:%b %Y}")
        st.dataframe(frame.head(6).style.format(precision=4), hide_index=True, use_container_width=True)
        st.caption("Notice the target in each row equals the next row's Mkt-RF.")

        factor = st.radio("Factor to plot", FACTOR_COLUMNS, horizontal=True)
        series = pd.DataFrame({"date": dates, "return": X[:, FACTOR_COLUMNS.index(factor)]})
        st.altair_chart(
            alt.Chart(series).mark_line(strokeWidth=1).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("return:Q", axis=alt.Axis(format="%"), title=f"{factor} monthly return"),
                tooltip=[alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("return:Q", format=".2%")],
            ).properties(height=260),
            use_container_width=True,
        )
        st.dataframe(
            pd.DataFrame(X, columns=FACTOR_COLUMNS).describe().T[["mean", "std", "min", "max"]].style.format("{:.2%}"),
            use_container_width=True,
        )
        st.caption(
            "Typical factor volatility is 2-4% a month. This scale matters: kernel bandwidths "
            "must be of a similar size (≈ 0.03-0.2) to create any locality."
        )

    with tabs[1]:
        st.markdown(
            "Time series must never be shuffled. `time_series_split` puts the first "
            "`train_ratio` of months in training and the rest in testing, so every "
            "forecast is made only from the past."
        )
        ratio = st.slider("train_ratio", 0.5, 0.95, 0.8, 0.05)
        split = int(np.floor(len(y) * ratio))
        c1, c2 = st.columns(2)
        c1.metric("Training months", split)
        c1.caption(f"{dates[0]:%b %Y} → {dates[split - 1]:%b %Y}")
        c2.metric("Test months", len(y) - split)
        c2.caption(f"{dates[split]:%b %Y} → {dates[-1]:%b %Y}")
        split_frame = pd.DataFrame(
            {"date": dates, "target": y, "set": np.where(np.arange(len(y)) < split, "train", "test")}
        )
        st.altair_chart(
            alt.Chart(split_frame).mark_line(strokeWidth=1).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("target:Q", axis=alt.Axis(format="%"), title="Next-month Mkt-RF"),
                color=alt.Color("set:N", scale=alt.Scale(domain=["train", "test"], range=["#4c78a8", "#f58518"])),
            ).properties(height=260),
            use_container_width=True,
        )

    with tabs[2]:
        st.markdown(
            "`ModelFactory` turns each config entry into a model object. Every model "
            "implements the same interface (`lrd/core/base.py`):"
        )
        st.code(
            "model.fit(X_train, y_train)                 # learn / store data\n"
            "y_pred = model.predict(X_test)              # point forecast\n"
            "y_pred, y_std = model.predict_with_uncertainty(X_test)  # std is None for deterministic models",
            language="python",
        )
        st.markdown(
            "- **Eager models** (Global OLS, Ridge, Monte Carlo Linear) do their work in `fit`.\n"
            "- **Lazy models** (LWLR, MCLLR and its ablations) only store the data in `fit` and "
            "solve a new local regression for every query in `predict`."
        )
        overview = pd.DataFrame(
            [
                {"Model": m["label"], "Family": m["family"], "Locality": m["locality"], "Uncertainty": m["uncertainty"]}
                for m in MODELS.values()
            ]
        )
        st.dataframe(overview, hide_index=True, use_container_width=True)
        st.caption("Open the **Models** page for each model's algorithm and maths.")

    with tabs[3]:
        st.markdown(
            "`Evaluator.evaluate` scores every model on the same test months. Accuracy "
            "metrics apply to all models; calibration metrics need a predicted standard "
            "deviation, so only the stochastic models get them."
        )
        for key, name, formula, meaning in METRICS:
            c1, c2, c3 = st.columns([1.2, 2, 2.8])
            c1.markdown(f"**{name}**  \n`{key}`")
            c2.latex(formula)
            c3.write(meaning)

    with tabs[4]:
        st.markdown(
            "A single split tests one period. **Walk-forward evaluation** "
            "(`lrd/core/evaluation/rolling.py`) slides a fixed training window through time, "
            "refits from scratch at every fold and tests on the months that follow - "
            "a realistic simulation of re-estimating a model every year."
        )
        c1, c2, c3 = st.columns(3)
        train_w = c1.number_input("train_window (months)", 60, 600, 240, 12)
        test_w = c2.number_input("test_window (months)", 6, 120, 12, 6)
        step = c3.number_input("step (months)", 6, 120, int(test_w), 6)
        try:
            folds = walk_forward_splits(len(y), int(train_w), int(test_w), int(step))
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.metric("Folds", len(folds))
            shown = folds[:12]
            rows = []
            for i, (tr_s, tr_e, te_s, te_e) in enumerate(shown):
                rows.append({"fold": i, "part": "train", "start": dates[tr_s], "end": dates[tr_e - 1]})
                rows.append({"fold": i, "part": "test", "start": dates[te_s], "end": dates[te_e - 1]})
            st.altair_chart(
                alt.Chart(pd.DataFrame(rows)).mark_bar(height=12).encode(
                    x=alt.X("start:T", title=None),
                    x2="end:T",
                    y=alt.Y("fold:O", title="Fold"),
                    color=alt.Color("part:N", scale=alt.Scale(domain=["train", "test"], range=["#4c78a8", "#f58518"])),
                    tooltip=["fold", "part", alt.Tooltip("start:T", format="%b %Y"), alt.Tooltip("end:T", format="%b %Y")],
                ).properties(height=30 * len(shown) + 40),
                use_container_width=True,
            )
            if len(folds) > len(shown):
                st.caption(f"Showing the first {len(shown)} of {len(folds)} folds.")

    with tabs[5]:
        st.markdown("The same pipeline runs from the command line through `main.py`:")
        st.code(
            "python main.py                       # hold-out evaluation of every model\n"
            "python main.py --rolling             # + walk-forward evaluation\n"
            "python main.py --models lwlr mcllr   # a subset of models",
            language="bash",
        )
        st.markdown(
            "| Output | Contents |\n|---|---|\n"
            "| `results/metrics.csv` | One row per model with every metric |\n"
            "| `results/predictions.csv` | Date, actual return, each model's prediction and std |\n"
            "| `results/rolling_folds.csv` | Metrics for every walk-forward fold |\n"
            "| `results/rolling_summary.csv` | Mean and std of each metric across folds |\n"
            "| `results/figures/` | Prediction, residual, uncertainty and comparison plots |\n"
            "| `experiments/logs/<timestamp>/` | Config copy, results, dataset hash and git commit |"
        )


def page_models() -> None:
    st.title("🧠 Models")
    choice = st.selectbox("Model", MODEL_ORDER, format_func=label)
    info = MODELS[choice]

    st.subheader(info["label"])
    for col, (name, value) in zip(
        st.columns(3),
        [("Family", info["family"]), ("Locality", info["locality"]), ("Uncertainty", info["uncertainty"])],
    ):
        with col.container(border=True):
            st.caption(name)
            st.markdown(f"**{value}**")
    st.markdown(info["idea"])

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**How it predicts**")
        st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(info["steps"], 1)))
        st.latex(info["formula"])
    with right:
        st.markdown("**Hyperparameters**")
        st.markdown("\n".join(f"- **{k}** - {v}" for k, v in info["params"].items()))
        st.markdown(f"**Strengths:** {info['pros']}")
        st.markdown(f"**Limitations:** {info['cons']}")
        st.markdown(f"**Cost:** {info['cost']}")

    if choice in {"lwlr", "mcllr", "mcllr_no_stochasticity"}:
        locality_explorer(choice)
    else:
        st.info("Select LWLR, MCLLR or the no-stochasticity ablation to explore kernel locality interactively.")


def locality_explorer(model_type: str) -> None:
    st.divider()
    st.subheader("🔍 Locality explorer")
    st.markdown(
        "Pick a test month and a bandwidth to see how the kernel weights the training "
        "months. **Effective sample size** (ESS = (Σw)² / Σw²) counts how many months "
        "really influence the local fit."
    )
    X, y, dates = load_data()
    split = int(len(y) * 0.8)
    X_train, train_dates = X[:split], dates[:split]

    c1, c2 = st.columns(2)
    query_idx = c1.select_slider(
        "Query (test) month", options=list(range(split, len(y))),
        value=split + 108 if split + 108 < len(y) else split,
        format_func=lambda i: f"{dates[i]:%b %Y}",
    )
    bandwidth = c2.select_slider(
        "Bandwidth τ", options=[0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.5, 1.0], value=0.08
    )

    sq_dist = np.sum((X_train - X[query_idx]) ** 2, axis=1)
    weights = np.exp(-(sq_dist - sq_dist.min()) / (2 * bandwidth**2))
    probabilities = weights / weights.sum()
    ess = weights.sum() ** 2 / np.sum(weights**2)

    m1, m2, m3 = st.columns(3)
    m1.metric("Effective sample size", f"{ess:.0f} of {split}")
    m2.metric("Weight of median month", f"{np.median(weights):.2f}")
    m3.metric(
        "Probability share of 80 nearest months", f"{np.sort(probabilities)[-80:].sum():.0%}",
        help="How concentrated MCLLR's sampling is: 80 is the default neighbourhood size.",
    )

    frame = pd.DataFrame(
        {"date": train_dates, "distance": np.sqrt(sq_dist), "weight": weights, "Mkt-RF": X_train[:, 0]}
    )
    a, b = st.columns(2)
    a.altair_chart(
        alt.Chart(frame).mark_circle(size=18, opacity=0.6).encode(
            x=alt.X("distance:Q", title="Distance to query month"),
            y=alt.Y("weight:Q", title="Kernel weight (nearest = 1)"),
            tooltip=[alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("weight:Q", format=".3f")],
        ).properties(height=280, title="Weight vs distance"),
        use_container_width=True,
    )
    b.altair_chart(
        alt.Chart(frame).mark_area(opacity=0.7, interpolate="step").encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("weight:Q", title="Kernel weight", stack=None, scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("weight:Q", format=".3f")],
        ).properties(height=280, title="Which training months matter"),
        use_container_width=True,
    )
    if ess > 0.9 * split:
        st.warning("Almost every month gets the same weight - at this bandwidth the model behaves like its global counterpart.")
    elif ess < 30:
        st.warning("Very few months carry weight - local fits will be noisy.")
    if model_type == "mcllr":
        st.caption("For MCLLR these weights become the probabilities used to sample each 80-month neighbourhood.")


DEFAULT_PARAMS = {"bandwidth": 0.08, "n_simulations": 50, "subsample_size": 80, "alpha": 1.0}
DEFAULT_PARAMS_TUPLE = tuple(sorted(DEFAULT_PARAMS.items()))


def page_comparison() -> None:
    st.title("⚖️ Model comparison")

    with st.sidebar.form("settings"):
        st.header("Experiment settings")
        models = st.multiselect("Models", MODEL_ORDER, default=MODEL_ORDER, format_func=label)
        mode = st.radio("Evaluation", ["Hold-out test", "Walk-forward"], horizontal=True)
        st.subheader("Hyperparameters")
        bandwidth = st.select_slider("Bandwidth τ", [0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.5, 1.0], value=0.08)
        n_sims = st.slider("Monte Carlo simulations K", 10, 200, 50, 10)
        subsample = st.slider("Subsample / neighbourhood size m", 20, 200, 80, 10)
        alpha = st.select_slider("Ridge α", [0.01, 0.1, 0.3, 1.0, 3.0, 10.0], value=1.0)
        seed = st.number_input("Random seed", 0, 10_000, 42)
        st.subheader("Data split")
        train_ratio = st.slider("Hold-out train_ratio", 0.5, 0.9, 0.8, 0.05)
        train_window = st.slider("Walk-forward train window", 120, 480, 240, 12)
        test_window = st.slider("Walk-forward test window", 6, 60, 12, 6)
        st.form_submit_button("Run comparison", type="primary", use_container_width=True)

    if not models:
        st.warning("Select at least one model.")
        return
    min_train = train_window if mode == "Walk-forward" else int(816 * train_ratio)
    if subsample > min_train:
        st.error(f"Subsample size m = {subsample} exceeds the {min_train} training months.")
        return

    models = tuple(m for m in MODEL_ORDER if m in models)
    params = tuple(sorted({"bandwidth": bandwidth, "n_simulations": n_sims, "subsample_size": subsample, "alpha": alpha}.items()))

    if mode == "Hold-out test":
        with st.spinner("Fitting and evaluating models…"):
            metrics, predictions = run_holdout(models, params, train_ratio, int(seed))
        holdout_view(metrics, predictions)
    else:
        with st.spinner("Running walk-forward evaluation (refits every fold)…"):
            folds = run_walk_forward(models, params, int(seed), train_window, test_window, test_window)
        walk_forward_view(folds)


def metric_table(frame: pd.DataFrame) -> None:
    cols = [c for c in ["rmse", "mae", "r2", "median_ae", "coverage_95", "interval_width_95", "gaussian_nll", "interval_score_95"] if c in frame]
    table = frame[["model", *cols]].copy()
    table["model"] = table["model"].map(label)
    lower_better = [c for c in cols if c not in {"r2", "coverage_95"}]
    best = "background-color: rgba(84, 162, 75, 0.35); font-weight: 600"  # readable in light and dark themes
    styled = (
        table.style.format({c: "{:.4f}" for c in cols}, na_rep="-")
        .highlight_min(subset=lower_better, props=best)
        .highlight_max(subset=[c for c in ["r2"] if c in cols], props=best)
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)
    st.caption("Green marks the best value per column. Coverage should be close to 0.95, not maximal.")


def metric_bars(frame: pd.DataFrame, value_col: str = "rmse", error_col: str | None = None) -> None:
    # Dot plot on a zoomed axis: model differences are ~1% of the metric, which
    # bars anchored at zero would hide.
    data = frame.copy()
    data["label"] = data["model"].map(label)
    if error_col:
        data["lo"] = data[value_col] - data[error_col]
        data["hi"] = data[value_col] + data[error_col]
    y = alt.Y("label:N", sort=alt.EncodingSortField(value_col, order="ascending"), title=None)
    dots = alt.Chart(data).mark_circle(size=160, opacity=1).encode(
        x=alt.X(f"{value_col}:Q", title=value_col, scale=alt.Scale(zero=False)),
        y=y,
        color=alt.Color("label:N", legend=None),
        tooltip=["label", alt.Tooltip(f"{value_col}:Q", format=".4f")],
    )
    chart = dots
    if error_col:
        chart = alt.Chart(data).mark_rule(strokeWidth=2, opacity=0.6).encode(
            x="lo:Q", x2="hi:Q", y=y, color=alt.Color("label:N", legend=None)
        ) + dots
    st.altair_chart(chart.properties(height=36 * len(data) + 50), use_container_width=True)


def holdout_view(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    models = list(metrics["model"])
    st.markdown(
        f"Test period **{predictions['date'].min():%b %Y} → {predictions['date'].max():%b %Y}** "
        f"({len(predictions)} months)."
    )
    tab_metrics, tab_time, tab_unc, tab_cum = st.tabs(
        ["📊 Metrics", "📈 Predictions over time", "🎯 Uncertainty", "📉 Cumulative performance"]
    )

    with tab_metrics:
        metric_table(metrics)
        candidates = ["rmse", "mae", "r2", "coverage_95", "interval_width_95", "gaussian_nll"]
        available = [m for m in candidates if m in metrics and metrics[m].notna().any()]
        chosen = st.radio("Metric", available, horizontal=True)
        metric_bars(metrics.dropna(subset=[chosen]), chosen)

    with tab_time:
        selected = st.multiselect("Models to plot", models, default=models[:3], format_func=label)
        long = predictions.melt(id_vars="date", value_vars=["y_true", *[f"{m}_pred" for m in selected]], var_name="series", value_name="return")
        long["series"] = long["series"].map(lambda s: "Actual" if s == "y_true" else label(s.removesuffix("_pred")))
        st.altair_chart(
            alt.Chart(long).mark_line().encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("return:Q", axis=alt.Axis(format="%"), title="Next-month Mkt-RF"),
                color=alt.Color("series:N", title=None),
                strokeWidth=alt.condition(alt.datum.series == "Actual", alt.value(1), alt.value(2.2)),
                tooltip=["series", alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("return:Q", format=".2%")],
            ).properties(height=360).interactive(bind_y=False),
            use_container_width=True,
        )
        st.caption("Forecasts hug zero while actual returns swing ±10%: the predictable part of monthly returns is tiny.")

        scatter_model = st.selectbox("Predicted vs actual for", models, format_func=label)
        sc = pd.DataFrame({"actual": predictions["y_true"], "predicted": predictions[f"{scatter_model}_pred"]})
        line = pd.DataFrame({"actual": [sc.actual.min(), sc.actual.max()], "predicted": [sc.actual.min(), sc.actual.max()]})
        st.altair_chart(
            alt.Chart(sc).mark_circle(opacity=0.6).encode(
                x=alt.X("actual:Q", axis=alt.Axis(format="%")), y=alt.Y("predicted:Q", axis=alt.Axis(format="%"))
            ) + alt.Chart(line).mark_line(strokeDash=[4, 4], color="gray").encode(x="actual:Q", y="predicted:Q"),
            use_container_width=True,
        )

    with tab_unc:
        stochastic = [m for m in models if m in STOCHASTIC]
        if not stochastic:
            st.info("Include Monte Carlo Linear, MCLLR or the no-geometry ablation to see uncertainty bands.")
        else:
            model = st.selectbox("Stochastic model", stochastic, format_func=label)
            band = pd.DataFrame({
                "date": predictions["date"],
                "actual": predictions["y_true"],
                "pred": predictions[f"{model}_pred"],
                "lo": predictions[f"{model}_pred"] - 1.96 * predictions[f"{model}_std"],
                "hi": predictions[f"{model}_pred"] + 1.96 * predictions[f"{model}_std"],
            })
            band["inside"] = np.where((band.actual >= band.lo) & (band.actual <= band.hi), "inside 95% band", "outside")
            base = alt.Chart(band).encode(x=alt.X("date:T", title=None))
            st.altair_chart(
                (
                    base.mark_area(opacity=0.25, color="#4c78a8").encode(y=alt.Y("lo:Q", axis=alt.Axis(format="%"), title="Return"), y2="hi:Q")
                    + base.mark_line(color="#4c78a8").encode(y="pred:Q")
                    + base.mark_circle(size=22).encode(
                        y="actual:Q",
                        color=alt.Color("inside:N", scale=alt.Scale(domain=["inside 95% band", "outside"], range=["#54a24b", "#e45756"]), title=None),
                        tooltip=[alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("actual:Q", format=".2%"), alt.Tooltip("pred:Q", format=".2%")],
                    )
                ).properties(height=360),
                use_container_width=True,
            )
            row = metrics.set_index("model").loc[model]
            c1, c2 = st.columns(2)
            c1.metric("Coverage of nominal 95% band", f"{row['coverage_95']:.0%}", f"{row['coverage_95'] - 0.95:+.0%} vs target")
            c2.metric("Average band width", f"{row['interval_width_95']:.2%}")
            st.caption(
                "The band reflects disagreement between Monte Carlo fits (estimation uncertainty) "
                "and ignores month-to-month return noise, so most outcomes fall outside it."
            )

    with tab_cum:
        benchmark = st.selectbox(
            "Benchmark", models, index=models.index("global_linear") if "global_linear" in models else 0, format_func=label
        )
        bench_err = (predictions["y_true"] - predictions[f"{benchmark}_pred"]) ** 2
        curves = []
        for m in models:
            if m == benchmark:
                continue
            err = (predictions["y_true"] - predictions[f"{m}_pred"]) ** 2
            curves.append(pd.DataFrame({"date": predictions["date"], "model": label(m), "cum": (bench_err - err).cumsum()}))
        if curves:
            st.altair_chart(
                alt.Chart(pd.concat(curves)).mark_line().encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("cum:Q", title=f"Cumulative SSE advantage over {label(benchmark)}"),
                    color=alt.Color("model:N", title=None),
                    tooltip=["model", alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("cum:Q", format=".5f")],
                ).properties(height=360)
                + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q"),
                use_container_width=True,
            )
            st.caption(
                "Rising lines mean the model is beating the benchmark during that period; "
                "jumps show the individual months (often crashes) that decide the ranking."
            )
        else:
            st.info("Select at least two models to compare against a benchmark.")


def walk_forward_view(folds: pd.DataFrame) -> None:
    models = [m for m in MODEL_ORDER if m in set(folds["model"])]
    n_folds = folds["fold_id"].nunique()
    st.markdown(
        f"**{n_folds} folds** per model, testing {folds['test_end_date'].min():%Y} → "
        f"{folds['test_end_date'].max():%Y}. Each fold refits every model from scratch."
    )
    numeric = folds.select_dtypes("number").columns.difference(["fold_id", "train_start", "train_end", "test_start", "test_end"])
    summary = folds.groupby("model")[list(numeric)].mean().reindex(models).reset_index()
    spread = folds.groupby("model")["rmse"].std().reindex(models).rename("rmse_std").reset_index()

    tab_metrics, tab_time, tab_dist = st.tabs(["📊 Mean metrics", "📈 RMSE by fold", "📦 Distribution"])
    with tab_metrics:
        metric_table(summary)
        st.markdown("**Mean RMSE across folds** (error bars: ±1 std across folds)")
        metric_bars(summary.merge(spread, on="model"), "rmse", "rmse_std")
        wins = folds.loc[folds.groupby("fold_id")["rmse"].idxmin(), "model"].value_counts().reindex(models, fill_value=0)
        st.markdown("**Folds won** (lowest RMSE in that fold)")
        st.bar_chart(wins.rename(index=label), horizontal=True)

    with tab_time:
        selected = st.multiselect("Models", models, default=models[:3], format_func=label, key="wf_models")
        data = folds[folds["model"].isin(selected)].assign(label=lambda d: d["model"].map(label))
        st.altair_chart(
            alt.Chart(data).mark_line(point=True).encode(
                x=alt.X("test_end_date:T", title="End of test window"),
                y=alt.Y("rmse:Q", title="Fold RMSE"),
                color=alt.Color("label:N", title=None),
                tooltip=["label", alt.Tooltip("test_end_date:T", format="%b %Y"), alt.Tooltip("rmse:Q", format=".4f")],
            ).properties(height=360).interactive(bind_y=False),
            use_container_width=True,
        )
        st.caption("Error spikes line up with crises (1987, 2000-02, 2008, 2020) for every model alike.")

    with tab_dist:
        st.altair_chart(
            alt.Chart(folds.assign(label=folds["model"].map(label))).mark_boxplot(extent="min-max").encode(
                x=alt.X("rmse:Q", title="Fold RMSE"), y=alt.Y("label:N", title=None), color=alt.Color("label:N", legend=None)
            ).properties(height=50 * len(models) + 40),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def run_app() -> None:
    st.set_page_config(page_title="Local Risk Dynamics", page_icon="📈", layout="wide")
    pages = [
        st.Page(page_overview, title="Overview", icon="🏠", default=True),
        st.Page(page_pipeline, title="Pipeline", icon="🔧"),
        st.Page(page_models, title="Models", icon="🧠"),
        st.Page(page_comparison, title="Comparison", icon="⚖️"),
    ]
    st.sidebar.caption("Local Risk Dynamics · Fama-French 5-factor forecasting")
    st.navigation(pages).run()


# `streamlit run` executes this file as __main__; importing it (e.g. in tests)
# only defines the pages.
if __name__ == "__main__":
    run_app()
