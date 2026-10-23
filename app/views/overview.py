"""Screen 1: project overview — objective, methodology, models and practices."""

from __future__ import annotations

import streamlit as st

from app import charts, engine
from app.catalog import MODELS, PRACTICES, SYSTEMS
from lrd import comparison as cmp

PIPELINE_DIAGRAM = """
digraph {
  rankdir=LR; bgcolor="transparent"; compound=true;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#4c78a8", fillcolor="#eef3fb"];
  edge [color="#888888"];
  subgraph cluster_core { label="Shared core (lrd.core)"; fontname="Helvetica"; fontsize=11; color="#bbbbbb"; style="rounded";
    validate [label="1. Validate raw data\\nmissing · garbage · duplicates · gaps"];
    load [label="2. Load\\nX = factors at t\\ny = Mkt-RF at t+1"];
    folds [label="3. Walk-forward folds\\n240-month train · 12-month test"];
    evaluate [label="6. Evaluate\\nRMSE · OOS R² · DM test · calibration"];
  }
  subgraph cluster_std { label="Standard system (lrd.standard)"; fontname="Helvetica"; fontsize=11; color="#bbbbbb"; style="rounded";
    s_fit [label="4. Fit on raw factors\\nfixed hyperparameters", fillcolor="#fdf0e2", color="#f58518"];
    s_pred [label="5. Predict test months", fillcolor="#fdf0e2", color="#f58518"];
  }
  subgraph cluster_imp { label="Improved system (lrd.improved)"; fontname="Helvetica"; fontsize=11; color="#bbbbbb"; style="rounded";
    i_prep [label="4a. Standardise\\n(training window only)", fillcolor="#e8f4e6", color="#54a24b"];
    i_tune [label="4b. Nested tuning\\ninner walk-forward", fillcolor="#e8f4e6", color="#54a24b"];
    i_pred [label="5. Predict + predictive std", fillcolor="#e8f4e6", color="#54a24b"];
  }
  validate -> load -> folds;
  folds -> s_fit -> s_pred -> evaluate;
  folds -> i_prep -> i_tune -> i_pred -> evaluate;
}
"""


def _open_model(key: str) -> None:
    st.session_state["model_key"] = key
    pages = st.session_state.get("pages")
    if pages:
        st.switch_page(pages["models"])


def render() -> None:
    st.title("📈 Local Risk Dynamics")
    st.markdown("##### Deterministic vs stochastic locality in linear models for financial risk")

    runs = engine.default_runs()
    summary = cmp.summarize(list(runs.values()), runs[cmp.BENCHMARK_KEY]).set_index("key")

    # ------------------------------------------------------------ objective
    st.header("Objective")
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            "Financial markets are **non-stationary**: the link between risk factors and returns changes "
            "across regimes. A single global regression assumes it never changes.\n\n"
            "**Research question:** can regime-dependent behaviour be captured by many small, **local** linear "
            "models instead of one global model — with locality defined deterministically (kernel distance) "
            "or stochastically (sampling)?\n\n"
            "**Task:** forecast next month's US market excess return from this month's five Fama–French "
            "factors, and quantify the uncertainty of each forecast."
        )
    with right:
        best = summary["oos_r2_%"].idxmax()
        with st.container(border=True):
            st.markdown("**Headline result** (shared walk-forward folds)")
            c1, c2 = st.columns(2)
            c1.metric("Best model", MODELS[best]["label"])
            c2.metric("OOS R² vs historical mean", f"{summary.loc[best, 'oos_r2_%']:+.2f}%")
            c3, c4 = st.columns(2)
            c3.metric("Original MCLLR", f"{summary.loc['standard:mcllr', 'oos_r2_%']:+.2f}%")
            c4.metric("Improved MCLLR coverage", f"{summary.loc['improved:mcllr_improved', 'coverage_95']:.0%}")
            st.caption(
                f"No model beats the benchmark significantly (best p = {summary.loc[best, 'dm_p']:.2f}). "
                "Explore why on the Data analysis and Comparison screens."
            )

    # ---------------------------------------------------------- methodology
    st.header("Methodology")
    st.graphviz_chart(PIPELINE_DIAGRAM, use_container_width=True)
    st.caption(
        "Both systems share data loading, folds and evaluation, so every model is scored on identical months "
        "with identical metric definitions. Only model preparation and fitting differ."
    )
    tabs = st.tabs(["Walk-forward evaluation", "Benchmark & OOS R²", "Significance", "Nested tuning", "Calibration"])
    X, y, dates = engine.load_data()
    with tabs[0]:
        st.markdown(
            "Each fold trains on 240 months and tests on the next 12, then slides forward a year — a realistic "
            f"simulation of refitting annually. That gives **{len(runs[cmp.BENCHMARK_KEY].folds)} folds** and "
            f"**{len(runs[cmp.BENCHMARK_KEY].dates)} test months** "
            f"({runs[cmp.BENCHMARK_KEY].dates[0]:%b %Y} – {runs[cmp.BENCHMARK_KEY].dates[-1]:%b %Y})."
        )
        st.altair_chart(charts.fold_timeline(dates, cmp.TRAIN_WINDOW, cmp.TEST_WINDOW), use_container_width=True)
    with tabs[1]:
        st.markdown("The **historical mean** of the training window is the standard benchmark for return forecasts.")
        st.latex(r"R^2_{\text{OOS}} = 1 - \frac{\sum_t (y_t - \hat y_t)^2}{\sum_t (y_t - \bar y_{\text{train}})^2}")
        st.markdown("Above 0 means the model's squared errors are smaller than the benchmark's. "
                    "For monthly returns, even +0.5% is considered economically meaningful.")
    with tabs[2]:
        st.markdown("The **Diebold–Mariano test** checks whether the difference in squared errors between two "
                    "models is larger than chance. p < 0.05 means the difference is statistically significant.")
        st.latex(r"d_t = e_{1,t}^2 - e_{2,t}^2,\qquad DM = \frac{\bar d}{\hat\sigma_d / \sqrt{T}} \sim N(0,1)")
    with tabs[3]:
        st.markdown("Improved models choose hyperparameters **inside each fold**: an inner walk-forward "
                    "(180-month train, 12-month test) on the training window compares candidate settings, then "
                    "the best one is refitted on the whole training window. No test month influences any choice.")
    with tabs[4]:
        st.markdown("A model's uncertainty is useful only if it is **calibrated**: a 95% interval should contain "
                    "about 95% of outcomes. Coverage, interval score and Gaussian log-loss measure this.")

    # ---------------------------------------------------------------- models
    st.header("Models")
    st.caption("Ten models across two systems. Open any model for its algorithm, live settings and results.")
    for system, title in SYSTEMS.items():
        st.subheader(title)
        keys = [k for k in MODELS if k.startswith(system + ":")]
        cols = st.columns(len(keys) if len(keys) <= 4 else 4)
        for i, key in enumerate(keys):
            info, row = MODELS[key], summary.loc[key]
            with cols[i % len(cols)].container(border=True):
                st.markdown(f"**{info['label']}**")
                st.caption(f"{info['family']} · locality: {info['locality']} · uncertainty: {info['uncertainty']}")
                st.write(info["short"])
                st.markdown(f"OOS R² vs mean: **{row['oos_r2_%']:+.2f}%**")
                if st.button("Open model →", key=f"open_{key}"):
                    _open_model(key)

    # ------------------------------------------------------------- practices
    st.header("Practices applied")
    cols = st.columns(4)
    for i, (title, text) in enumerate(PRACTICES):
        with cols[i % 4].container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)
