"""
Data validation

Checks the raw Fama-French file before it is ingested and reports what the
loader would otherwise handle silently: rows it drops, cells it coerces to
NaN, missing or duplicated months, and extreme values.

Hard problems (missing values, garbage cells, duplicate or missing months,
unsorted dates) are reported as errors. Extreme values are only flagged:
crash months are real data and must stay in the sample.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lrd.core.data.loader import ALL_RETURN_COLUMNS


def _markdown_table(frame: pd.DataFrame, index: bool = False) -> str:
    """Render a small DataFrame as a Markdown table (no optional dependencies)."""
    if index:
        frame = frame.reset_index()
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    rows = [" | ".join(map(str, frame.columns)), " | ".join("---" for _ in frame.columns)]
    rows += [" | ".join(fmt(v) for v in row) for row in frame.itertuples(index=False)]
    return "\n".join(f"| {r} |" for r in rows)


class DataValidationError(ValueError):
    """Raised when a dataset fails validation and cannot be used."""


@dataclass
class ValidationReport:
    """Outcome of validating one raw dataset file."""

    source: str
    n_monthly_rows: int
    n_dropped_rows: int
    dropped_examples: list[str]
    missing: dict[str, int]
    non_numeric: dict[str, int]
    duplicate_months: list[str]
    missing_months: list[str]
    is_sorted: bool
    outliers: pd.DataFrame
    stats: pd.DataFrame
    outlier_threshold: float = 5.0
    date_range: tuple[str, str] = ("", "")
    source_hash: str = ""  # MD5 prefix, matching lrd.core.data.pipeline.Dataset
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.errors = []
        if sum(self.missing.values()):
            self.errors.append(f"Missing values: {self._nonzero(self.missing)}")
        if sum(self.non_numeric.values()):
            self.errors.append(f"Non-numeric (garbage) values: {self._nonzero(self.non_numeric)}")
        if self.duplicate_months:
            self.errors.append(f"Duplicate months: {', '.join(self.duplicate_months[:10])}")
        if self.missing_months:
            self.errors.append(
                f"{len(self.missing_months)} missing month(s): {', '.join(self.missing_months[:10])}"
            )
        if not self.is_sorted:
            self.errors.append("Months are not in chronological order.")

    @staticmethod
    def _nonzero(counts: dict[str, int]) -> dict[str, int]:
        return {k: v for k, v in counts.items() if v}

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise DataValidationError(
                f"{self.source} failed validation:\n- " + "\n- ".join(self.errors)
            )

    def to_markdown(self) -> str:
        """Human-readable report, written alongside pipeline outputs."""
        lines = [
            "# Data validation report",
            "",
            f"- Source: `{self.source}` (hash `{self.source_hash}`)",
            f"- Status: **{'PASSED' if self.ok else 'FAILED'}**",
            f"- Monthly rows: {self.n_monthly_rows} ({self.date_range[0]} to {self.date_range[1]})",
            f"- Non-monthly rows dropped by the loader: {self.n_dropped_rows}"
            + (f" (e.g. {', '.join(self.dropped_examples)})" if self.dropped_examples else ""),
            f"- Missing values: {sum(self.missing.values())}",
            f"- Non-numeric values: {sum(self.non_numeric.values())}",
            f"- Duplicate months: {len(self.duplicate_months)}",
            f"- Missing months: {len(self.missing_months)}",
            f"- Chronological order: {'yes' if self.is_sorted else 'NO'}",
            "",
        ]
        if self.errors:
            lines += ["## Errors", ""] + [f"- {e}" for e in self.errors] + [""]
        lines += [
            f"## Extreme values (|robust z| > {self.outlier_threshold:g}, flagged, not removed)",
            "",
        ]
        if self.outliers.empty:
            lines.append("None.")
        else:
            lines.append(_markdown_table(self.outliers))
        lines += ["", "## Summary statistics (decimal returns)", "", _markdown_table(self.stats.rename_axis("column"), index=True), ""]
        return "\n".join(lines)


def validate_fama_french(
    csv_path: str | Path,
    outlier_threshold: float = 5.0,
) -> ValidationReport:
    """Validate the raw Fama-French 5-factor CSV.

    Parameters
    ----------
    csv_path : str or Path
        Raw file in the Kenneth French library format.
    outlier_threshold : float, default=5.0
        Values with robust z-score (median/MAD based) above this are flagged.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fama-French file not found: {csv_path}")

    raw = pd.read_csv(csv_path, skiprows=4, skipinitialspace=True, dtype=str, keep_default_na=False)
    raw = raw.rename(columns={raw.columns[0]: "Date"})
    missing_cols = [c for c in ALL_RETURN_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise DataValidationError(f"{csv_path} is missing columns: {missing_cols}")

    first_col = raw["Date"].str.strip()
    is_monthly = first_col.str.match(r"^\d{6}$")
    dropped = first_col[~is_monthly & (first_col != "")]
    monthly = raw[is_monthly].copy()
    monthly["Date"] = first_col[is_monthly]

    missing, non_numeric, values = {}, {}, {}
    for col in ALL_RETURN_COLUMNS:
        text = monthly[col].str.strip()
        empty = text.isin(["", "NaN", "nan", "NA", "-99.99", "-999"])
        numeric = pd.to_numeric(text.where(~empty), errors="coerce")
        missing[col] = int(empty.sum())
        non_numeric[col] = int((numeric.isna() & ~empty).sum())
        values[col] = numeric / 100.0

    periods = pd.PeriodIndex(pd.to_datetime(monthly["Date"], format="%Y%m"), freq="M")
    duplicated = periods[periods.duplicated()].unique()
    expected = pd.period_range(periods.min(), periods.max(), freq="M") if len(periods) else periods
    gaps = expected.difference(periods)

    frame = pd.DataFrame(values)
    frame.insert(0, "month", periods.astype(str))
    outlier_rows = []
    for col in ALL_RETURN_COLUMNS:
        series = frame[col].dropna()
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            continue
        robust_z = 0.6745 * (series - median) / mad
        for idx in robust_z[np.abs(robust_z) > outlier_threshold].index:
            outlier_rows.append(
                {"month": frame.at[idx, "month"], "column": col,
                 "value": frame.at[idx, col], "robust_z": robust_z[idx]}
            )
    outliers = pd.DataFrame(outlier_rows, columns=["month", "column", "value", "robust_z"])
    if not outliers.empty:
        outliers = outliers.sort_values("robust_z", key=np.abs, ascending=False).reset_index(drop=True)

    return ValidationReport(
        source=str(csv_path),
        n_monthly_rows=len(monthly),
        n_dropped_rows=len(dropped),
        dropped_examples=list(dropped.head(3)),
        missing=missing,
        non_numeric=non_numeric,
        duplicate_months=[str(p) for p in duplicated],
        missing_months=[str(p) for p in gaps],
        is_sorted=bool(periods.is_monotonic_increasing),
        outliers=outliers,
        stats=frame[ALL_RETURN_COLUMNS].describe().T[["mean", "std", "min", "max"]],
        outlier_threshold=outlier_threshold,
        date_range=(str(periods.min()), str(periods.max())) if len(periods) else ("", ""),
        source_hash=hashlib.md5(csv_path.read_bytes()).hexdigest()[:8],
    )
