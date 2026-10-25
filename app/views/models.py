"""Screen 3: each model — explanation, live settings, results and metrics."""

from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app import charts, engine
from app.catalog import MODELS
from lrd import comparison as cmp
from lrd.core.evaluation.significance import diebold_mariano

# (name, widget, options, default) — widgets: "select", "slider" (min, max, step), "toggle"
STANDARD_SETTINGS = {
    "global_linear": [],
    "ridge_linear": [("alpha", "select", [0.01, 0.1, 0.3, 1.0, 3.0, 10.0], 1.0)],
    "lwlr": [("bandwidth", "select", [0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.5, 1.0], 0.08)],
    "mc_linear": [("n_simulations", "slider", (10, 200, 10), 50), ("subsample_size", "slider", (20, 240, 10), 80)],
    "mcllr": [("bandwidth", "select", [0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.5, 1.0], 0.08),
              ("n_simulations", "slider", (10, 200, 10), 50), ("subsample_size", "slider", (20, 240, 10), 80)],
    "mcllr_no_geometry": [("n_simulations", "slider", (10, 200, 10), 50), ("subsample_size", "slider", (20, 240, 10), 80)],
    "mcllr_no_stochasticity": [("bandwidth", "select", [0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.5, 1.0], 0.08),
                               ("subsample_size", "slider", (20, 240, 10), 80)],
}
IMPROVED_FIXED = {
    "historical_mean": [],
    "ridge": [("alpha", "select", [0.1, 1.0, 3.0, 10.0, 30.0, 100.0], 3.0)],
    "mcllr_improved": [("bandwidth", "select", [0.5, 1.0, 2.0, 3.0, 5.0], 2.0),
                       ("alpha", "select", [0.3, 1.0, 3.0, 10.0, 30.0, 100.0], 10.0),
                       ("n_simulations", "slider", (10, 200, 10), 50), ("subsample_size", "slider", (40, 240, 10), 160),
                       ("predictive_std", "toggle", None, True)],
    "mcllr_regime": [("bandwidth", "select", [0.25, 0.5, 1.0, 2.0, 3.0], 1.0),
                     ("alpha", "select", [0.3, 1.0, 3.0, 10.0, 30.0, 100.0], 10.0),
                     ("n_simulations", "slider", (10, 200, 10), 50), ("subsample_size", "slider", (40, 240, 10), 160),
                     ("state_window", "slider", (3, 36, 3), 12), ("predictive_std", "toggle", None, True)],
}
IMPROVED_GRID = {
    "ridge": {"alpha": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]},
    "mcllr_improved": {"bandwidth": [0.5, 1.0, 2.0, 3.0, 5.0], "alpha": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]},
    "mcllr_regime": {"bandwidth": [0.25, 0.5, 1.0, 2.0, 3.0], "alpha": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]},
}
LOCAL_SCALE = {"standard": "raw decimal returns", "improved": "standardised factors (σ units)"}


def _widget(name: str, kind: str, options, default, key: str):
    if kind == "select":
        return st.select_slider(name, options=options, value=default if default in options else options[0], key=key)
    if kind == "slider":
        lo, hi, step = options
        return st.slider(name, lo, hi, int(min(max(default, lo), hi)), step, key=key)
    return st.toggle(name, value=bool(default), key=key)


def _settings_form(key: str, info: dict, default_run: cmp.ModelRun):
    """Render the custom-settings form; returns (params, tuned, grid, protocol) or None until submitted."""
    system, model_type = default_run.system, default_run.model_type
    params, grid, tuned, protocol = {}, {}, False, "walk_forward"
    # Mode switches live outside the form: they change which inputs the form shows,
    # and widgets inside a form do not rerun the page until it is submitted.
    if system == "standard":
        evaluation = st.radio("Evaluation", ["Walk-forward (shared folds)", "Hold-out 80/20 (secondary view)"],
                              horizontal=True, key=f"proto_{key}")
        protocol = "holdout" if evaluation.startswith("Hold") else "walk_forward"
    elif model_type in IMPROVED_GRID:
        tuned = st.radio("Hyperparameters", ["Tuned per fold (nested walk-forward)", "Fixed values"],
                         horizontal=True, key=f"mode_{key}").startswith("Tuned")
    with st.form(f"settings_{key}"):
        if system == "standard":
            for name, kind, options, default in STANDARD_SETTINGS[model_type]:
                params[name] = _widget(name, kind, options, default_run.params.get(name, default), f"{key}_{name}")
        else:
            for name, kind, options, default in IMPROVED_FIXED[model_type]:
                if tuned and name in IMPROVED_GRID[model_type]:
                    defaults = default_run.params.get("grid", {}).get(name, IMPROVED_GRID[model_type][name][:3])
                    grid[name] = st.multiselect(f"{name} candidates", IMPROVED_GRID[model_type][name],
                                                default=[v for v in defaults if v in IMPROVED_GRID[model_type][name]],
                                                key=f"{key}_grid_{name}")
                else:
                    if tuned and name == "subsample_size":
                        options = (options[0], 180, options[2])  # must fit the 180-month inner window
                    params[name] = _widget(name, kind, options, default_run.params.get(name, default), f"{key}_{name}")
        if not (STANDARD_SETTINGS if system == "standard" else IMPROVED_FIXED)[model_type]:
            st.caption("This model has no hyperparameters.")
        submitted = st.form_submit_button("Run with these settings", type="primary")
    if not submitted:
        return None
    if tuned and any(not v for v in grid.values()):
        st.error("Choose at least one candidate for every tuned hyperparameter.")
        return None
    return params, tuned, grid, protocol


def _metric_tiles(run: cmp.ModelRun, bench: cmp.ModelRun) -> None:
    row = cmp.summarize([run], bench).iloc[0]
    cols = st.columns(6)
    cols[0].metric("RMSE", f"{row['rmse']:.4f}")
    cols[1].metric("OOS R² vs mean", f"{row['oos_r2_%']:+.2f}%")
    cols[2].metric("CW p-value", f"{row['cw_p']:.2f}",
                   help=f"Clark–West test vs the historical mean (one-sided); below 0.05 would be significant. "
                        f"Diebold–Mariano p = {row['dm_p']:.2f}.")
    cols[3].metric("95% coverage", "n/a" if np.isnan(row["coverage_95"]) else f"{row['coverage_95']:.0%}",
                   help="Target ≈ 95%. Not available for models without uncertainty.")
    cols[4].metric("Interval score", "n/a" if np.isnan(row["interval_score_95"]) else f"{row['interval_score_95']:.3f}",
                   help="Lower is better.")
    cols[5].metric("CER gain", f"{row['cer_gain_%']:+.2f}%/yr",
                   help="Extra certainty-equivalent return per year for a mean-variance investor (γ = 3) "
                        "timing the market with this model instead of the historical mean.")


def _per_fold(run: cmp.ModelRun, bench: cmp.ModelRun, label: str) -> None:
    frame = pd.concat([
        run.folds.assign(model=label), bench.folds.assign(model="Historical mean (benchmark)")
    ])
    st.altair_chart(alt.Chart(frame).mark_line(point=True).encode(
        x=alt.X("test_end_date:T", title="End of test window"), y=alt.Y("rmse:Q", title="Fold RMSE"),
        color=alt.Color("model:N", title=None),
        tooltip=["model", alt.Tooltip("test_end_date:T", format="%b %Y"), alt.Tooltip("rmse:Q", format=".4f")],
    ).properties(height=320).interactive(bind_y=False), use_container_width=True)
    beat = int((run.folds["rmse"].to_numpy() < bench.folds["rmse"].to_numpy()).sum())
    st.caption(f"Beats the benchmark in {beat} of {len(run.folds)} folds. Spikes are crisis years, for every model alike.")


def _calibration(run: cmp.ModelRun) -> None:
    if not run.has_uncertainty:
        st.info("This model gives point forecasts only, so calibration cannot be assessed.")
        return
    levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    frame = pd.DataFrame({"nominal": levels, "actual": [cmp.coverage(run, lv) for lv in levels]})
    diag = pd.DataFrame({"nominal": [0.5, 0.99], "actual": [0.5, 0.99]})
    st.altair_chart((
        alt.Chart(diag).mark_line(strokeDash=[4, 4], color="gray").encode(x="nominal:Q", y="actual:Q")
        + alt.Chart(frame).mark_line(point=True).encode(
            x=alt.X("nominal:Q", axis=charts.PCT, title="Nominal interval level", scale=alt.Scale(domain=[0.45, 1])),
            y=alt.Y("actual:Q", axis=charts.PCT, title="Actual coverage", scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip("nominal:Q", format=".0%"), alt.Tooltip("actual:Q", format=".1%")])
    ).properties(height=320, title="Calibration (on the dashed line = perfectly calibrated)"), use_container_width=True)
    st.caption("Below the line: intervals too narrow (over-confident). Above: too wide.")


def _tuning(run: cmp.ModelRun) -> None:
    rows = pd.DataFrame([{"test_start_date": d, **c} for d, c in zip(run.tuning["test_start_date"], run.tuning["chosen"])])
    tuned_names = [c for c in run.params.get("grid", {})] or [c for c in rows.columns if c != "test_start_date"]
    for name in tuned_names:
        if name in rows:
            st.altair_chart(alt.Chart(rows).mark_line(point=True, interpolate="step-after").encode(
                x=alt.X("test_start_date:T", title="Fold start"), y=alt.Y(f"{name}:Q", scale=alt.Scale(type="log")),
                tooltip=[alt.Tooltip("test_start_date:T", format="%b %Y"), name],
            ).properties(height=200, title=f"Chosen {name} per fold"), use_container_width=True)
    st.caption("Each fold picks its own values from its training window only. Large jumps mean the inner "
               "validation error is nearly flat — a sign of a weak signal.")


def _locality(run: cmp.ModelRun, bandwidth: float) -> None:
    X, y, dates = engine.load_data()
    test_idx = [dates.get_loc(d) for d in run.dates]
    c1, c2 = st.columns(2)
    q = c1.select_slider("Query (test) month", options=test_idx, value=test_idx[len(test_idx) // 2],
                         format_func=lambda i: f"{dates[i]:%b %Y}", key=f"loc_q_{run.key}")
    bw = c2.number_input(f"Bandwidth ({LOCAL_SCALE[run.system]})", min_value=0.005, max_value=10.0,
                         value=float(bandwidth), step=0.01 if run.system == "standard" else 0.25, key=f"loc_bw_{run.key}")
    lo = max(0, q - cmp.TRAIN_WINDOW)
    X_train, train_dates = X[lo:q], dates[lo:q]
    x0 = X[q]
    if run.system == "improved":
        mu, sd = X_train.mean(0), X_train.std(0)
        X_train, x0 = (X_train - mu) / sd, (x0 - mu) / sd
    sq = ((X_train - x0) ** 2).sum(1)
    w = np.exp(-(sq - sq.min()) / (2 * bw**2))
    ess = w.sum() ** 2 / (w**2).sum()
    m1, m2, m3 = st.columns(3)
    m1.metric("Effective sample size", f"{ess:.0f} of {len(w)}")
    m2.metric("Median weight", f"{np.median(w):.2f}")
    m3.metric("Weight share of nearest 25%", f"{np.sort(w)[-len(w) // 4:].sum() / w.sum():.0%}")
    frame = pd.DataFrame({"date": train_dates, "distance": np.sqrt(sq), "weight": w})
    a, b = st.columns(2)
    a.altair_chart(alt.Chart(frame).mark_circle(size=20, opacity=0.6).encode(
        x=alt.X("distance:Q", title="Distance to query"), y=alt.Y("weight:Q", title="Kernel weight (nearest = 1)"),
        tooltip=[alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("weight:Q", format=".3f")],
    ).properties(height=280, title="Weight vs distance"), use_container_width=True)
    b.altair_chart(alt.Chart(frame).mark_area(opacity=0.7, interpolate="step").encode(
        x=alt.X("date:T", title=None), y=alt.Y("weight:Q", scale=alt.Scale(domain=[0, 1]), stack=None),
        tooltip=[alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("weight:Q", format=".3f")],
    ).properties(height=280, title="Which training months matter"), use_container_width=True)
    if ess > 0.9 * len(w):
        st.warning("Almost every month has the same weight: at this bandwidth the model behaves globally.")
    elif ess < 30:
        st.warning("Very few months carry weight: local fits will be noisy without shrinkage.")


def _mcllr_head_to_head() -> None:
    runs = engine.default_runs()
    original, improved = runs["standard:mcllr"], runs["improved:mcllr_improved"]
    regime = runs["improved:mcllr_regime"]
    table = cmp.summarize([original, improved, regime], runs[cmp.BENCHMARK_KEY]).set_index("key")
    table.index = ["MCLLR (original)", "MCLLR (improved)", "MCLLR (regime state)"]
    cols = ["rmse", "oos_r2_%", "cw_p", "coverage_95", "interval_score_95", "cer_gain_%"]
    st.dataframe(charts.styled_table(
        table[cols], {"rmse": "{:.4f}", "oos_r2_%": "{:+.2f}%", "cw_p": "{:.3f}", "coverage_95": "{:.0%}",
                      "interval_score_95": "{:.3f}", "cer_gain_%": "{:+.2f}"},
        lower_better=["rmse", "interval_score_95"], higher_better=["oos_r2_%", "cer_gain_%"],
    ), use_container_width=True)
    t, p = diebold_mariano(improved.errors, original.errors)
    st.caption(f"Improved vs original MCLLR: Diebold–Mariano t = {t:.2f}, p = {p:.3f} "
               f"({'significant' if p < 0.05 else 'not significant'} at 5%).")


def render() -> None:
    st.title("🧠 Models")
    keys = list(MODELS)
    if st.session_state.get("model_key") not in keys:  # first visit, or set by the overview's "Open model"
        st.session_state["model_key"] = "improved:mcllr_improved"
    key = st.selectbox("Model", keys, key="model_key",
                       format_func=lambda k: f"{MODELS[k]['label']}  ·  {k.split(':')[0]} system")
    info = MODELS[key]
    default_run = engine.default_runs()[key]

    # ----------------------------------------------------------- explanation
    st.header(info["label"])
    for col, (name, value) in zip(st.columns(4), [("System", key.split(":")[0].title()), ("Family", info["family"]),
                                                  ("Locality", info["locality"]), ("Uncertainty", info["uncertainty"])]):
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
        if info["params"]:
            st.markdown("**Hyperparameters**")
            st.markdown("\n".join(f"- **{k}** — {v}" for k, v in info["params"].items()))
        st.markdown(f"**Strengths:** {info['pros']}")
        st.markdown(f"**Limitations:** {info['cons']}")
        st.markdown(f"**Cost:** {info['cost']}")

    # -------------------------------------------------------------- settings
    st.header("Results")
    source = st.radio("Show results for", ["Default settings (precomputed)", "Custom settings"], horizontal=True,
                      key=f"source_{key}")
    run, label = default_run, info["label"]
    if source.startswith("Custom"):
        choice = _settings_form(key, info, default_run)
        state_key = f"custom_choice_{key}"
        if choice is not None:
            st.session_state[state_key] = choice
        choice = st.session_state.get(state_key)
        if choice is None:
            st.info("Adjust the settings and press **Run with these settings**.")
            return
        params, tuned, grid, protocol = choice
        with st.spinner("Running walk-forward evaluation…" if protocol == "walk_forward" else "Fitting on the hold-out split…"):
            run = engine.custom_run(default_run.system, default_run.model_type, json.dumps(params, sort_keys=True),
                                    tuned, json.dumps(grid, sort_keys=True), protocol)
        label = engine.run_label(default_run.system, default_run.model_type, params, tuned)
        engine.remember(run, label)
        if protocol == "walk_forward":
            st.success("Run added to the Comparison screen.", icon="➕")
    else:
        st.caption("Default configuration from experiments/standard and experiments/improved, evaluated on the "
                   "shared walk-forward folds.")

    bench = engine.benchmark(run.protocol)
    period = f"{run.dates[0]:%b %Y} – {run.dates[-1]:%b %Y}, {len(run.dates)} months"
    st.markdown(f"**{'Walk-forward' if run.protocol == 'walk_forward' else 'Hold-out'} evaluation** · {period}")
    _metric_tiles(run, bench)

    tab_names = ["Forecasts", "Diagnostics", "Calibration"]
    if run.protocol == "walk_forward":
        tab_names.insert(2, "Per fold")
    if len(run.tuning) and run.protocol == "walk_forward":
        tab_names.append("Tuning")
    if info["local"]:
        tab_names.append("Locality explorer")
    if default_run.model_type in ("mcllr", "mcllr_improved", "mcllr_regime"):
        tab_names.append("Original vs improved")
    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    with tabs["Forecasts"]:
        title = info["label"] + (" (custom settings)" if source.startswith("Custom") else "")
        st.altair_chart(charts.predictions_over_time(run, title), use_container_width=True)
        st.caption("Dots are actual monthly returns; the line is the forecast"
                   + (" with its 95% band (green = inside)." if run.has_uncertainty else "."))
    with tabs["Diagnostics"]:
        st.altair_chart(charts.predicted_vs_actual(run), use_container_width=True)
        st.altair_chart(charts.residuals(run), use_container_width=True)
    if "Per fold" in tabs:
        with tabs["Per fold"]:
            _per_fold(run, bench, label)
    with tabs["Calibration"]:
        _calibration(run)
    if "Tuning" in tabs:
        with tabs["Tuning"]:
            _tuning(run)
    if "Locality explorer" in tabs:
        with tabs["Locality explorer"]:
            chosen = run.tuning["chosen"].iloc[-1] if len(run.tuning) else {}
            _locality(run, chosen.get("bandwidth", run.params.get("bandwidth", 2.0)))
    if "Original vs improved" in tabs:
        with tabs["Original vs improved"]:
            _mcllr_head_to_head()
