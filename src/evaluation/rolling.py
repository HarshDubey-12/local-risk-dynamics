from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from .evaluator import Evaluator, EvaluationResult


@dataclass
class RollingFoldResult:
    """Evaluation output for one chronological fold."""

    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    result: EvaluationResult


def walk_forward_splits(
    n_samples: int,
    train_window: int,
    test_window: int,
    step: int | None = None,
) -> List[tuple[int, int, int, int]]:
    """Return chronological (train_start, train_end, test_start, test_end) folds."""
    if train_window <= 0 or test_window <= 0:
        raise ValueError("train_window and test_window must be positive.")
    if train_window + test_window > n_samples:
        raise ValueError("train_window + test_window cannot exceed n_samples.")

    step = test_window if step is None else step
    if step <= 0:
        raise ValueError("step must be positive.")

    folds: List[tuple[int, int, int, int]] = []
    train_start = 0
    while True:
        train_end = train_start + train_window
        test_start = train_end
        test_end = test_start + test_window
        if test_end > n_samples:
            break
        folds.append((train_start, train_end, test_start, test_end))
        train_start += step

    if not folds:
        raise ValueError("No valid walk-forward folds for provided settings.")
    return folds


def evaluate_walk_forward(
    model_builder: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    train_window: int,
    test_window: int,
    step: int | None = None,
    extra_metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
) -> List[RollingFoldResult]:
    """Evaluate a model over rolling chronological folds."""
    folds = walk_forward_splits(
        n_samples=len(y),
        train_window=train_window,
        test_window=test_window,
        step=step,
    )

    results: List[RollingFoldResult] = []
    for fold_id, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        model = model_builder()
        model.fit(X[tr_s:tr_e], y[tr_s:tr_e])
        eval_result = Evaluator.evaluate(
            model=model,
            X_test=X[te_s:te_e],
            y_test=y[te_s:te_e],
            extra_metrics=extra_metrics,
        )
        results.append(
            RollingFoldResult(
                fold_id=fold_id,
                train_start=tr_s,
                train_end=tr_e,
                test_start=te_s,
                test_end=te_e,
                result=eval_result,
            )
        )
    return results


def rolling_results_to_frame(results: List[RollingFoldResult]) -> pd.DataFrame:
    """Flatten fold metrics into a DataFrame."""
    rows = []
    for fold in results:
        row = {
            "fold_id": fold.fold_id,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "model": fold.result.model_name,
        }
        row.update(fold.result.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_rolling_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return mean/std summary for all numeric metric columns."""
    metric_cols = [
        c
        for c in frame.columns
        if c not in {"fold_id", "train_start", "train_end", "test_start", "test_end", "model"}
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    summary = frame[metric_cols].agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]
    return summary
