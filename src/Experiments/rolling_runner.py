from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..data.loader import load_fama_french
from ..evaluation.rolling import (
    evaluate_walk_forward,
    rolling_results_to_frame,
    summarize_rolling_metrics,
)
from ..models.factory import Modelfactory


class RollingExperimentRunner:
    """Run walk-forward evaluation for all models in an experiment config."""

    def __init__(
        self,
        config_path: str | Path,
        train_window: int,
        test_window: int,
        step: int | None = None,
    ) -> None:
        self.config = ExperimentConfig.from_yaml(config_path)
        self.train_window = train_window
        self.test_window = test_window
        self.step = step

    def run(
        self,
        extra_metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        X, y, _ = load_fama_french(self.config.dataset_path)

        all_fold_frames = []
        all_summary_frames = []
        for model_config in self.config.models:
            def builder():
                return Modelfactory.create(model_config)

            fold_results = evaluate_walk_forward(
                model_builder=builder,
                X=X,
                y=y,
                train_window=self.train_window,
                test_window=self.test_window,
                step=self.step,
                extra_metrics=extra_metrics,
            )
            fold_frame = rolling_results_to_frame(fold_results)
            summary_frame = summarize_rolling_metrics(fold_frame)
            summary_frame["model"] = model_config.model_type

            all_fold_frames.append(fold_frame)
            all_summary_frames.append(summary_frame)

        folds = pd.concat(all_fold_frames, ignore_index=True)
        summary = pd.concat(all_summary_frames, ignore_index=True)
        return folds, summary
