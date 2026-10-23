"""Reusable Altair charts for the dashboard."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from lrd import comparison as cmp
from lrd.core.evaluation.rolling import walk_forward_splits

TRAIN_COLOR, TEST_COLOR = "#4c78a8", "#f58518"
GOOD, BAD = "#54a24b", "#e45756"
PCT = alt.Axis(format="%")


BEST_CELL = "background-color: rgba(84, 162, 75, 0.35); font-weight: 600"  # readable in light and dark themes


def styled_table(numeric: pd.DataFrame, formats: dict, lower_better=(), higher_better=()):
    """Pre-formatted text table with the best value per column highlighted.

    Cells are formatted to strings first (missing values become "–"), because
    Streamlit renders NaN cells of a styled numeric frame as "None".
    """
    display = numeric.copy().astype(object)
    for col, fmt in formats.items():
        display[col] = [("–" if pd.isna(v) else fmt.format(v)) for v in numeric[col]]
    css = pd.DataFrame("", index=numeric.index, columns=numeric.columns)
    for cols, pick in ((lower_better, "idxmin"), (higher_better, "idxmax")):
        for col in cols:
            values = pd.to_numeric(numeric[col], errors="coerce")
            if values.notna().any():
                css.loc[getattr(values, pick)(), col] = BEST_CELL
    return display.style.apply(lambda _: css, axis=None)


def predictions_over_time(run: cmp.ModelRun, label: str, band: bool = True) -> alt.Chart:
    """Actual returns, the forecast line and (if available) its 95% band."""
    frame = pd.DataFrame({"date": run.dates, "actual": run.y_true, "forecast": run.pred})
    base = alt.Chart(frame).encode(x=alt.X("date:T", title=None))
    layers = []
    if band and run.has_uncertainty:
        frame["lo"], frame["hi"] = run.pred - 1.96 * run.std, run.pred + 1.96 * run.std
        frame["status"] = np.where((frame.actual >= frame.lo) & (frame.actual <= frame.hi), "inside 95% band", "outside")
        base = alt.Chart(frame).encode(x=alt.X("date:T", title=None))
        layers.append(base.mark_area(opacity=0.22, color=TRAIN_COLOR).encode(y=alt.Y("lo:Q", axis=PCT, title="Return"), y2="hi:Q"))
        points = base.mark_circle(size=16).encode(
            y="actual:Q",
            color=alt.Color("status:N", title=None, scale=alt.Scale(domain=["inside 95% band", "outside"], range=[GOOD, BAD])),
            tooltip=[alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("actual:Q", format=".2%"),
                     alt.Tooltip("forecast:Q", format=".2%")],
        )
    else:
        points = base.mark_circle(size=14, color="#9d9d9d").encode(
            y=alt.Y("actual:Q", axis=PCT, title="Return"),
            tooltip=[alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("actual:Q", format=".2%"),
                     alt.Tooltip("forecast:Q", format=".2%")],
        )
    line = base.mark_line(color=TRAIN_COLOR, strokeWidth=1.6).encode(y=alt.Y("forecast:Q", axis=PCT, title="Return"))
    return alt.layer(*layers, points, line).properties(height=340, title=f"{label}: forecast vs actual").interactive(bind_y=False)


def predicted_vs_actual(run: cmp.ModelRun) -> alt.Chart:
    frame = pd.DataFrame({"actual": run.y_true, "forecast": run.pred})
    lo, hi = frame.actual.min(), frame.actual.max()
    diag = pd.DataFrame({"actual": [lo, hi], "forecast": [lo, hi]})
    return (
        alt.Chart(frame).mark_circle(opacity=0.5).encode(x=alt.X("actual:Q", axis=PCT), y=alt.Y("forecast:Q", axis=PCT))
        + alt.Chart(diag).mark_line(strokeDash=[4, 4], color="gray").encode(x="actual:Q", y="forecast:Q")
    ).properties(height=300, title="Forecast vs actual (dashed = perfect)")


def residuals(run: cmp.ModelRun) -> alt.Chart:
    frame = pd.DataFrame({"date": run.dates, "residual": run.errors})
    over_time = alt.Chart(frame).mark_bar(width=1.2).encode(
        x=alt.X("date:T", title=None), y=alt.Y("residual:Q", axis=PCT),
        color=alt.condition(alt.datum.residual > 0, alt.value(TRAIN_COLOR), alt.value(BAD)),
    ).properties(height=220, title="Residuals over time")
    hist = alt.Chart(frame).mark_bar(opacity=0.8).encode(
        x=alt.X("residual:Q", bin=alt.Bin(maxbins=40), axis=PCT), y=alt.Y("count():Q", title="Months"),
    ).properties(height=220, title="Residual distribution")
    return alt.hconcat(over_time, hist)


def fold_timeline(dates: pd.DatetimeIndex, train_window: int, test_window: int, max_folds: int = 12) -> alt.Chart:
    folds = walk_forward_splits(len(dates), train_window, test_window, test_window)
    rows = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds[:max_folds]):
        rows.append({"fold": i, "part": "train", "start": dates[tr_s], "end": dates[tr_e - 1]})
        rows.append({"fold": i, "part": "test", "start": dates[te_s], "end": dates[te_e - 1]})
    return alt.Chart(pd.DataFrame(rows)).mark_bar(height=10).encode(
        x=alt.X("start:T", title=None), x2="end:T", y=alt.Y("fold:O", title="Fold"),
        color=alt.Color("part:N", title=None, scale=alt.Scale(domain=["train", "test"], range=[TRAIN_COLOR, TEST_COLOR])),
        tooltip=["fold", "part", alt.Tooltip("start:T", format="%b %Y"), alt.Tooltip("end:T", format="%b %Y")],
    ).properties(height=24 * min(len(folds), max_folds) + 40, title=f"First {min(len(folds), max_folds)} of {len(folds)} walk-forward folds")


def dot_plot(frame: pd.DataFrame, value: str, label: str = "label", color: str = "system",
             title: str = "", fmt: str = ".3f", ref: float | None = None) -> alt.Chart:
    """Ranked dot plot with an optional reference line (e.g. 0 = benchmark)."""
    y = alt.Y(f"{label}:N", sort=alt.EncodingSortField(value, order="descending"), title=None)
    dots = alt.Chart(frame).mark_circle(size=150, opacity=1).encode(
        x=alt.X(f"{value}:Q", title=title or value, scale=alt.Scale(zero=False)), y=y,
        color=alt.Color(f"{color}:N", title=None), tooltip=[label, alt.Tooltip(f"{value}:Q", format=fmt)],
    )
    chart = dots
    if ref is not None:
        chart = alt.Chart(pd.DataFrame({"x": [ref]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x:Q") + dots
    return chart.properties(height=34 * len(frame) + 40)
