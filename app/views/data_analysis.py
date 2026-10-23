"""Screen 2: data validation and interactive pre-analysis."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app import charts, engine

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
TARGET = "target (Mkt-RF t+1)"


def _quality(report) -> None:
    st.subheader("Data quality checks")
    checks = [
        ("Overall", "PASSED" if report.ok else "FAILED", report.ok),
        ("Monthly rows", f"{report.n_monthly_rows}", True),
        ("Missing values", f"{sum(report.missing.values())}", not sum(report.missing.values())),
        ("Garbage values", f"{sum(report.non_numeric.values())}", not sum(report.non_numeric.values())),
        ("Duplicate months", f"{len(report.duplicate_months)}", not report.duplicate_months),
        ("Missing months", f"{len(report.missing_months)}", not report.missing_months),
        ("Chronological", "yes" if report.is_sorted else "no", report.is_sorted),
    ]
    for col, (name, value, good) in zip(st.columns(len(checks)), checks):
        with col.container(border=True):
            st.caption(name)
            st.markdown(f"{'✅' if good else '❌'} **{value}**")
    st.caption(
        f"{report.date_range[0]} – {report.date_range[1]} · {report.n_dropped_rows} non-monthly rows "
        f"(notes, annual table) ignored by the loader · file hash `{report.source_hash}`."
    )
    st.info(
        "**Data correction.** These checks found that the original file began with a duplicated copy of its "
        "header and the first 68 months (Jul 1963 – Feb 1969). The loader used both copies, double-counting "
        "those months and giving Feb 1969 a wrong next-month target. The duplicate was removed and the loader "
        "now rejects duplicated or out-of-order months.",
        icon="🛠️",
    )
    with st.expander("Full validation report"):
        st.markdown(report.to_markdown())


def _robust_z(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = np.median(np.abs(series - median))
    return 0.6745 * (series - median) / mad if mad else series * 0


def render() -> None:
    st.title("🔎 Data analysis")
    st.markdown(
        "Kenneth French's **Fama–French 5-factor** monthly file. Features are this month's factor returns; the "
        "target is next month's market excess return. Everything below is computed live from the data."
    )
    _quality(engine.validation())

    frame = engine.raw_frame()
    st.subheader("Explore")
    c1, c2 = st.columns([2, 3])
    factors = c1.multiselect("Factors", FACTORS, default=["Mkt-RF", "HML"])
    years = c2.slider("Years", int(frame.index.year.min()), int(frame.index.year.max()),
                      (int(frame.index.year.min()), int(frame.index.year.max())))
    view = frame[(frame.index.year >= years[0]) & (frame.index.year <= years[1])]
    if not factors:
        st.warning("Select at least one factor.")
        return
    long = view[factors].reset_index(names="date").melt("date", var_name="factor", value_name="return")

    tabs = st.tabs(["Time series & outliers", "Distributions", "Correlations", "Volatility regimes",
                    "Predictability", "Evaluation folds"])

    with tabs[0]:
        threshold = st.slider("Flag months with |robust z-score| above", 3.0, 10.0, 5.0, 0.5,
                              help="Robust z uses the median and median absolute deviation, so outliers do not hide themselves.")
        flags = pd.concat([
            pd.DataFrame({"date": view.index, "factor": f, "return": view[f].to_numpy(), "z": _robust_z(frame[f]).loc[view.index].to_numpy()})
            for f in factors
        ])
        flags = flags[flags["z"].abs() > threshold]
        lines = alt.Chart(long).mark_line(strokeWidth=1).encode(
            x=alt.X("date:T", title=None), y=alt.Y("return:Q", axis=charts.PCT, title="Monthly return"),
            color=alt.Color("factor:N", title=None),
            tooltip=["factor", alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("return:Q", format=".2%")],
        )
        marks = alt.Chart(flags).mark_point(size=80, color=charts.BAD, filled=False, strokeWidth=2).encode(
            x="date:T", y="return:Q", tooltip=["factor", alt.Tooltip("date:T", format="%b %Y"),
                                               alt.Tooltip("return:Q", format=".2%"), alt.Tooltip("z:Q", format=".1f")])
        st.altair_chart((lines + marks).properties(height=360).interactive(bind_y=False), use_container_width=True)
        st.caption(f"{len(flags)} extreme month(s) circled. They are **kept** in the data: crashes are real and "
                   "decide most model rankings.")
        if len(flags):
            st.dataframe(flags.assign(date=flags["date"].dt.strftime("%Y-%m")).sort_values("z", key=np.abs, ascending=False)
                         .style.format({"return": "{:.2%}", "z": "{:+.1f}"}), hide_index=True, use_container_width=True)

    with tabs[1]:
        chosen = st.radio("Factor", factors, horizontal=True, key="dist_factor")
        series = view[chosen]
        hist = alt.Chart(pd.DataFrame({"return": series})).mark_bar(opacity=0.75).encode(
            x=alt.X("return:Q", bin=alt.Bin(maxbins=50), axis=charts.PCT), y=alt.Y("count():Q", title="Months"))
        grid = np.linspace(series.min(), series.max(), 200)
        bin_width = (series.max() - series.min()) / 50
        normal = pd.DataFrame({"return": grid, "months": len(series) * bin_width *
                               np.exp(-0.5 * ((grid - series.mean()) / series.std()) ** 2) / (series.std() * np.sqrt(2 * np.pi))})
        curve = alt.Chart(normal).mark_line(color=charts.BAD).encode(x="return:Q", y="months:Q")
        st.altair_chart((hist + curve).properties(height=320, title=f"{chosen}: histogram vs normal (red)"), use_container_width=True)
        stats = pd.DataFrame({
            "mean": view[FACTORS + [TARGET]].mean(), "std": view[FACTORS + [TARGET]].std(),
            "skew": view[FACTORS + [TARGET]].skew(), "excess kurtosis": view[FACTORS + [TARGET]].kurt(),
            "min": view[FACTORS + [TARGET]].min(), "max": view[FACTORS + [TARGET]].max(),
        })
        st.dataframe(stats.style.format({"mean": "{:.2%}", "std": "{:.2%}", "skew": "{:+.2f}",
                                         "excess kurtosis": "{:.1f}", "min": "{:.2%}", "max": "{:.2%}"}),
                     use_container_width=True)
        st.caption("Excess kurtosis well above 0 means fat tails: extreme months are far more common than a "
                   "normal distribution predicts.")

    with tabs[2]:
        corr = view[FACTORS + [TARGET]].corr().reset_index(names="row").melt("row", var_name="col", value_name="corr")
        heat = alt.Chart(corr).mark_rect().encode(
            x=alt.X("col:N", title=None, sort=FACTORS + [TARGET]), y=alt.Y("row:N", title=None, sort=FACTORS + [TARGET]),
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="corr"))
        text = alt.Chart(corr).mark_text(fontSize=12).encode(
            x=alt.X("col:N", sort=FACTORS + [TARGET]), y=alt.Y("row:N", sort=FACTORS + [TARGET]),
            text=alt.Text("corr:Q", format=".2f"))
        st.altair_chart((heat + text).properties(height=380), use_container_width=True)
        st.caption("Factors are correlated with each other, but the bottom row shows how weakly any of them is "
                   "related to next month's market return.")

    with tabs[3]:
        window = st.slider("Rolling window (months)", 6, 60, 12, 6)
        vol = (view[factors].rolling(window).std() * np.sqrt(12)).dropna()
        long_vol = vol.reset_index(names="date").melt("date", var_name="factor", value_name="vol")
        high = vol[factors[0]] > vol[factors[0]].quantile(0.8)
        regimes = pd.DataFrame({"date": vol.index[high]})
        shading = alt.Chart(regimes).mark_rule(color="#e45756", opacity=0.12, strokeWidth=4).encode(x="date:T")
        lines = alt.Chart(long_vol).mark_line().encode(
            x=alt.X("date:T", title=None), y=alt.Y("vol:Q", axis=charts.PCT, title="Annualised volatility"),
            color=alt.Color("factor:N", title=None),
            tooltip=["factor", alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("vol:Q", format=".1%")])
        st.altair_chart((shading + lines).properties(height=340).interactive(bind_y=False), use_container_width=True)
        st.caption(f"Shaded: months in the top 20% of {factors[0]} volatility. Volatility clusters into regimes — "
                   "the motivation for local models.")

    with tabs[4]:
        chosen = st.radio("Factor at month t", FACTORS, horizontal=True, key="pred_factor")
        scatter = pd.DataFrame({"factor": view[chosen], "target": view[TARGET]})
        r = scatter.corr().iloc[0, 1]
        points = alt.Chart(scatter).mark_circle(opacity=0.45).encode(
            x=alt.X("factor:Q", axis=charts.PCT, title=f"{chosen} at t"),
            y=alt.Y("target:Q", axis=charts.PCT, title="Mkt-RF at t+1"))
        fit = points.transform_regression("factor", "target").mark_line(color=charts.BAD)
        left, right = st.columns(2)
        left.altair_chart((points + fit).properties(height=320, title=f"r = {r:+.3f}, R² = {r * r:.2%}"),
                          use_container_width=True)
        target = view[TARGET]
        acf = pd.DataFrame({"lag": range(1, 13), "autocorrelation": [target.autocorr(k) for k in range(1, 13)]})
        bound = 1.96 / np.sqrt(len(target))
        bars = alt.Chart(acf).mark_bar().encode(x=alt.X("lag:O"), y=alt.Y("autocorrelation:Q"),
                                                color=alt.condition(f"abs(datum.autocorrelation) > {bound}",
                                                                    alt.value(charts.BAD), alt.value(charts.TRAIN_COLOR)))
        rules = alt.Chart(pd.DataFrame({"y": [bound, -bound]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
        right.altair_chart((bars + rules).properties(height=320, title="Target autocorrelation (dashed = 95% noise band)"),
                           use_container_width=True)
        st.caption("Single factors explain well under 1% of next month's return, and past returns barely predict "
                   "future ones. This weak signal is why no model beats the historical mean by much.")

    with tabs[5]:
        _, _, dates = engine.load_data()
        c1, c2 = st.columns(2)
        train = c1.slider("Training window (months)", 60, 480, 240, 12)
        test = c2.slider("Test window (months)", 6, 60, 12, 6)
        st.altair_chart(charts.fold_timeline(dates, train, test), use_container_width=True)
        st.caption("The models use 240/12, shared by both systems. Every fold refits from scratch on its own "
                   "training window; preparation and tuning never see the test months.")
