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

    def __init__(self, config_path: str | Path):
        self.config = ExperimentConfig.from_yaml(config_path)
        self.pipeline = DataPipeline()
        self.results = {}

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
        X_train, X_test, y_train, y_test, _, _ = time_series_split(
            dataset.X, dataset.y, dataset.dates, self.config.train_ratio
        )
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
        return results_df
    
    def save_results(self, path: str | Path):
        """Persist results."""
        self.results.to_csv(path, index=False)
        print(f"Results saved to {path} ")

# Usage in notebook
runner = ExperimentRunner("experiments/config_comparison.yaml")
results_df = runner.run()
print(results_df)