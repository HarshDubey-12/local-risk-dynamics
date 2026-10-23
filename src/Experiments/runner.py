from pathlib import Path 
from typing import List, Callable
import pandas as pd 
import numpy as np

from ..config import ExperimentConfig
from ..data.pipeline import DataPipeline
from ..data.preprocessing import time_series_split
from ..models.factory import Modelfactory
from ..evaluation.evaluator import Evaluator

class ExperimentRunner:
    """Orchestrate multi-model training and evaluation."""

    def __init__(self, config: str | Path | ExperimentConfig):
        self.config = (
            config
            if isinstance(config, ExperimentConfig)
            else ExperimentConfig.from_yaml(config)
        )
        self.pipeline = DataPipeline()
        self.results = {}
        # Populated by run(): full per-model outputs for plotting/export.
        self.dataset = None
        self.test_dates = None
        self.eval_results = []

    def run(
        self,
        extra_metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
    ) -> pd.DataFrame:
        """Execute full experiment.

        Parameters
        ----------
        extra_metrics : dict, optional
            Additional evaluation functions (y_true, y_pred) -> float. They
            will be forwarded to ``Evaluator.evaluate``.
        """

        # 1. Load data once
        print("Loading dataset...")
        dataset = self.pipeline.load_fama_french(self.config.dataset_path)
        print(f" {dataset.describe()}")

        # 2. Split data once 
        print("Splitting data...")
        X_train, X_test, y_train, y_test, _, dates_test = time_series_split(
            dataset.X, dataset.y, dataset.dates, self.config.train_ratio
        )
        self.dataset = dataset
        self.test_dates = dates_test
        print(f" Train: {len(y_train)}, Test: {len(y_test)}")

        #3. Train and evaluate each model
        results = []
        for model_config in self.config.models:
            print(f"\nTraining {model_config.model_type}...")

            model = Modelfactory.create(model_config)
            model.fit(X_train, y_train)

            eval_result = Evaluator.evaluate(
                model, X_test, y_test, extra_metrics=extra_metrics
            )
            results.append(eval_result)
            print(f" {eval_result}")

        #4. Complie results into Dataframe
        results_df = pd.DataFrame([
            {
                "model": r.model_name,
                **r.metrics,
            }
            for r in results 
        ])

        self.results = results_df
        self.eval_results = results
        return results_df
    
    def save_results(self, path: str | Path):
        """Persist results."""
        self.results.to_csv(path, index=False)
        print(f"Results saved to {path} ")
