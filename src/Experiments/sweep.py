from itertools import product
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd

# Import the missing classes
from ..config import ExperimentConfig
from .runner import ExperimentRunner

class HyperparameterSweep:
    """Grid search over model congigurations."""

    def __init__(self, base_config_path: str | Path):
        self.base_config = ExperimentConfig.from_yaml(base_config_path)
    
    def sweep(
            self,
            model_type: str, 
            param_grid: Dict[str, List[Any]]
    ) -> pd.DataFrame:
        """Run all combinations of hyperparameters."""

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        results = []
        for values in product(*param_values):
            config_dict = dict(zip(param_names, values))

            # Create variant config
            config = self.base_config
            model_config = [m for m in config.models if m.model_type == model_type][0]
            for param, value in config_dict.items():
                setattr(model_config, param, value)

            runner = ExperimentRunner(self.base_config)
            runner.config = config
            result = runner.run()
            result["hyperparams"] = str(config_dict)
            results.append(result)

        return pd.concat(results, ignore_index=True)
