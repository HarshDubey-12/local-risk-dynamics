"""Screen 4: compare any models from either system on the shared folds."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app import charts, engine
from lrd import comparison as cmp

LEADERBOARD = {
    "label": "Model", "system": "System", "rmse": "RMSE", "oos_r2_%": "OOS R² vs mean (%)", "cw_p": "CW p",
    "dm_p": "DM p", "coverage_95": "95% coverage", "interval_score_95": "Interval score",
    "gaussian_nll": "Gaussian NLL", "cer_gain_%": "CER gain (%/yr)", "sharpe": "Sharpe", "folds_won": "Folds won",
}


def _leaderboard(summary: pd.DataFrame) -> None:
    table = summary[list(LEADERBOARD)].rename(columns=LEADERBOARD).sort_values("RMSE").reset_index(drop=True)
    st.dataframe(
        charts.styled_table(
            table,
            {"RMSE": "{:.4f}", "OOS R² vs mean (%)": "{:+.2f}", "CW p": "{:.3f}", "DM p": "{:.3f}",
             "95% coverage": "{:.1%}", "Interval score": "{:.4f}", "Gaussian NLL": "{:.3f}",
             "CER gain (%/yr)": "{:+.2f}", "Sharpe": "{:.2f}", "Folds won": "{:d}"},
            lower_better=["RMSE", "Interval score", "Gaussian NLL"],
            higher_better=["OOS R² vs mean (%)", "CER gain (%/yr)", "Sharpe", "Folds won"],
        ),
        hide_index=True, use_container_width=True,
    )
    st.caption("Green = best in column. 95% coverage should be close to 95%, not maximal. "
               "CW = Clark–West test against the nested historical-mean benchmark (one-sided, the appropriate test "
               "here); DM = Diebold–Mariano (two-sided). Values below 0.05 would be significant. CER gain: extra "
               "certainty-equivalent return per year for a mean-variance investor (γ = 3) timing the market with "
               "the model instead of the historical mean.")


def _significance_matrix(selected, labels) -> None:
    t, p = cmp.pairwise_dm(selected)
    names = {r.key: labels[r.key] for r in selected}
    frame = pd.DataFrame([
        {"row": names[i], "col": names[j], "t": t.loc[i, j], "p": p.loc[i, j],
         "text": "" if i == j else f"{p.loc[i, j]:.2f}" + ("*" if p.loc[i, j] < 0.05 else "")}
        for i in t.index for j in t.columns
    ])
    order = [names[r.key] for r in selected]
    heat = alt.Chart(frame).mark_rect().encode(
        x=alt.X("col:N", sort=order, title="… compared with", axis=alt.Axis(labelAngle=-35, labelLimit=180)),
        y=alt.Y("row:N", sort=order, title="Model", axis=alt.Axis(labelLimit=220)),
        color=alt.Color("t:Q", scale=alt.Scale(scheme="redblue", reverse=True, domainMid=0), title="DM t"),
        tooltip=["row", "col", alt.Tooltip("t:Q", format=".2f"), alt.Tooltip("p:Q", format=".3f")],
    )
    text = alt.Chart(frame).mark_text(fontSize=11).encode(
        x=alt.X("col:N", sort=order), y=alt.Y("row:N", sort=order), text="text:N")
    st.altair_chart((heat + text).properties(height=46 * len(order) + 60), use_container_width=True)
    st.caption("Blue: the row model has smaller squared errors than the column model; red: larger. Numbers are "
               "p-values (* = significant at 5%).")


def render() -> None:
    st.title("⚖️ Model comparison")
    runs, labels = engine.available_runs()
    defaults = list(engine.default_runs())
    custom = [k for k in runs if k not in defaults]
    st.markdown(
        "All models are evaluated on the **same walk-forward folds**, so differences come from the models alone. "
        "The historical mean is the benchmark for OOS R² and significance. Custom runs from the Models screen "
        "appear here automatically."
    )
    selected_keys = st.multiselect("Models to compare", list(runs), default=defaults + custom,
                                   format_func=lambda k: labels[k])
    if len(selected_keys) < 2:
        st.warning("Select at least two models.")
        return
    selected = [runs[k] for k in selected_keys]
    bench = runs[cmp.BENCHMARK_KEY]
    summary = cmp.summarize(selected, bench)
    summary["label"] = summary["key"].map(labels)

    _leaderboard(summary)

    tabs = st.tabs(["Accuracy", "Significance", "Over time", "Calibration", "Sub-periods", "Folds",
                    "Standard vs improved"])

    with tabs[0]:
        frame = summary.assign(significant=np.where(summary["dm_p"] < 0.05, "significant (p < 0.05)", "not significant"))
        st.altair_chart(charts.dot_plot(frame, "oos_r2_%", title="OOS R² vs historical mean (%)", fmt="+.2f", ref=0),
                        use_container_width=True)
        st.caption("Right of the dashed line = beats the benchmark. Colours mark the system.")
        st.altair_chart(charts.dot_plot(summary, "rmse", title="RMSE", fmt=".4f"), use_container_width=True)

    with tabs[1]:
        _significance_matrix(selected, labels)

    with tabs[2]:
        curves = pd.concat([
            pd.DataFrame({"date": r.dates, "model": labels[r.key], "advantage": cmp.cumulative_advantage(r, bench).to_numpy()})
            for r in selected if r.key != cmp.BENCHMARK_KEY
        ])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
        st.altair_chart((zero + alt.Chart(curves).mark_line().encode(
            x=alt.X("date:T", title=None), y=alt.Y("advantage:Q", title="Cumulative SSE advantage over the mean"),
            color=alt.Color("model:N", title=None),
            tooltip=["model", alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("advantage:Q", format=".5f")],
        )).properties(height=380).interactive(bind_y=False), use_container_width=True)
        st.caption("Rising = beating the historical mean during that period. Vertical jumps are single months "
                   "(often crashes) that move the whole ranking.")

    with tabs[3]:
        with_unc = [r for r in selected if r.has_uncertainty]
        if not with_unc:
            st.info("None of the selected models produce uncertainty estimates.")
        else:
            levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
            frame = pd.DataFrame([{"model": labels[r.key], "nominal": lv, "actual": cmp.coverage(r, lv)}
                                  for r in with_unc for lv in levels])
            diag = pd.DataFrame({"nominal": [0.5, 0.99], "actual": [0.5, 0.99]})
            st.altair_chart((
                alt.Chart(diag).mark_line(strokeDash=[4, 4], color="gray").encode(x="nominal:Q", y="actual:Q")
                + alt.Chart(frame).mark_line(point=True).encode(
                    x=alt.X("nominal:Q", axis=charts.PCT, title="Nominal interval level", scale=alt.Scale(domain=[0.45, 1])),
                    y=alt.Y("actual:Q", axis=charts.PCT, title="Actual coverage", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("model:N", title=None),
                    tooltip=["model", alt.Tooltip("nominal:Q", format=".0%"), alt.Tooltip("actual:Q", format=".1%")])
            ).properties(height=380), use_container_width=True)
            st.caption("On the dashed line = calibrated. Standard Monte Carlo models sit far below it "
                       "(over-confident); improved models track it.")
            skipped = [labels[r.key] for r in selected if not r.has_uncertainty]
            if skipped:
                st.caption(f"Point-forecast models not shown: {', '.join(skipped)}.")

    with tabs[4]:
        periods = summary.melt(id_vars=["label", "system"], value_vars=["r2_pre2005_%", "r2_post2005_%", "r2_stress_%"],
                               var_name="period", value_name="oos_r2")
        periods["period"] = periods["period"].map({"r2_pre2005_%": "Before 2005", "r2_post2005_%": "2005 onwards",
                                                   "r2_stress_%": "Stress months (|return| > 2σ)"})
        order = ["Before 2005", "2005 onwards", "Stress months (|return| > 2σ)"]
        st.altair_chart(alt.Chart(periods).mark_bar().encode(
            x=alt.X("oos_r2:Q", title="OOS R² vs mean (%)"), y=alt.Y("label:N", title=None, axis=alt.Axis(labelLimit=260)),
            yOffset=alt.YOffset("period:N", sort=order),
            color=alt.Color("period:N", title=None, sort=order),
            tooltip=["label", "period", alt.Tooltip("oos_r2:Q", format="+.2f")],
        ).properties(height=54 * len(summary) + 40), use_container_width=True)
        st.caption("A real improvement should hold in every period, not only in one.")

    with tabs[5]:
        st.altair_chart(charts.dot_plot(summary, "folds_won", title="Folds won (lowest RMSE among selected models)", fmt="d"),
                        use_container_width=True)
        show = st.multiselect("Per-fold RMSE for", selected_keys, default=selected_keys[:3],
                              format_func=lambda k: labels[k], key="fold_models")
        if show:
            frame = pd.concat([runs[k].folds.assign(model=labels[k]) for k in show])
            st.altair_chart(alt.Chart(frame).mark_line(point=True).encode(
                x=alt.X("test_end_date:T", title="End of test window"), y=alt.Y("rmse:Q", title="Fold RMSE"),
                color=alt.Color("model:N", title=None),
                tooltip=["model", alt.Tooltip("test_end_date:T", format="%b %Y"), alt.Tooltip("rmse:Q", format=".4f")],
            ).properties(height=320).interactive(bind_y=False), use_container_width=True)

    with tabs[6]:
        pairs = [("standard:mcllr", "improved:mcllr_improved", "MCLLR"),
                 ("standard:ridge_linear", "improved:ridge", "Ridge")]
        for std_key, imp_key, name in pairs:
            if std_key in runs and imp_key in runs:
                a, b = runs[std_key], runs[imp_key]
                t, p = cmp.diebold_mariano(b.errors, a.errors)
                pair = cmp.summarize([a, b], bench).set_index("key")
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{name}: OOS R² standard → improved",
                          f"{pair.loc[imp_key, 'oos_r2_%']:+.2f}%", f"{pair.loc[imp_key, 'oos_r2_%'] - pair.loc[std_key, 'oos_r2_%']:+.2f} pts")
                cov_a, cov_b = pair.loc[std_key, "coverage_95"], pair.loc[imp_key, "coverage_95"]
                c2.metric(f"{name}: 95% coverage", "n/a" if np.isnan(cov_b) else f"{cov_b:.0%}",
                          None if np.isnan(cov_a) or np.isnan(cov_b) else f"{(cov_b - cov_a) * 100:+.0f} pts")
                c3.metric(f"{name}: DM improved vs standard", f"p = {p:.3f}",
                          "improved better" if t < 0 else "standard better", delta_color="normal" if t < 0 else "inverse")
        st.caption("Same months, same folds: the difference is entirely due to the improved pipeline "
                   "(standardisation, local ridge, predictive variance and nested tuning).")
