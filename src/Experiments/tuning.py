from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import ModelCofig
from ..evaluation.rolling import (
    evaluate_walk_forward,
    rolling_results_to_frame,
    summarize_rolling_metrics,
)
from ..models.factory import Modelfactory


def _iter_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    names = list(param_grid.keys())
    values = [param_grid[k] for k in names]
    return [dict(zip(names, combo)) for combo in product(*values)]


def tune_model_with_walk_forward(
    base_model_config: ModelCofig,
    param_grid: Dict[str, List[Any]],
    X: np.ndarray,
    y: np.ndarray,
    train_window: int,
    test_window: int,
    objective_metric: str = "rmse",
    minimize: bool = True,
    step: int | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Grid-search model hyperparameters using walk-forward validation."""
    grid = _iter_grid(param_grid)
    if not grid:
        raise ValueError("param_grid produced no configurations.")

    rows: List[Dict[str, Any]] = []
    for params in grid:
        model_config = replace(base_model_config, **params)

        def builder():
            return Modelfactory.create(model_config)

        folds = evaluate_walk_forward(
            model_builder=builder,
            X=X,
            y=y,
            train_window=train_window,
            test_window=test_window,
            step=step,
        )
        fold_frame = rolling_results_to_frame(folds)
        summary = summarize_rolling_metrics(fold_frame)
        metric_row = summary[summary["metric"] == objective_metric]
        if metric_row.empty:
            raise ValueError(f"Metric '{objective_metric}' not found in fold summary.")

        rows.append(
            {
                "params": params,
                "objective_metric": objective_metric,
                "objective_mean": float(metric_row["mean"].iloc[0]),
                "objective_std": float(metric_row["std"].iloc[0]),
                "n_folds": int(len(folds)),
            }
        )

    leaderboard = pd.DataFrame(rows)
    leaderboard = leaderboard.sort_values(
        by=["objective_mean", "objective_std"],
        ascending=[minimize, True],
    ).reset_index(drop=True)
    best = leaderboard.iloc[0].to_dict()
    return leaderboard, best
