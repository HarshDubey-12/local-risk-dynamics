from dataclasses import replace
from itertools import product
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd

from lrd.standard.config import ExperimentConfig
from lrd.standard.experiments.runner import ExperimentRunner

class HyperparameterSweep:
    """Grid search over model configurations."""

    def __init__(self, base_config: str | Path | ExperimentConfig):
        self.base_config = (
            base_config
            if isinstance(base_config, ExperimentConfig)
            else ExperimentConfig.from_yaml(base_config)
        )

    def sweep(
            self,
            model_type: str,
            param_grid: Dict[str, List[Any]]
    ) -> pd.DataFrame:
        """Run all combinations of hyperparameters for one model type.

        Only the swept model is trained for each combination; the base
        config is never modified.
        """

        matches = [m for m in self.base_config.models if m.model_type == model_type]
        if not matches:
            raise ValueError(f"Model type '{model_type}' not found in base config.")
        base_model = matches[0]

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        results = []
        for values in product(*param_values):
            config_dict = dict(zip(param_names, values))

            # Create variant config (replace re-runs validation on the new params)
            config = replace(
                self.base_config,
                models=[replace(base_model, **config_dict)],
            )

            result = ExperimentRunner(config).run()
            result["hyperparams"] = str(config_dict)
            results.append(result)

        return pd.concat(results, ignore_index=True)
