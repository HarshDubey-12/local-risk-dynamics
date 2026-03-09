from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any
import yaml 
from pathlib import Path

@dataclass
class ModelCofig:
    """Configuration for any model in the framework."""
    model_type: Literal[
        "global_linear",
        "lwlr",
        "mc_linear",
        "mcllr",
        "mcllr_no_geometry",
        "mcllr_no_stochasticity",
        "ridge_linear",
    ]
    fit_intercept: bool = True
    random_state: int | None = None
    # Model specific params
    bandwidth: float | None = None # LWLR, MCLLR
    n_simulations: int | None = None # MC, MCLLR
    subsample_size: int | None = None # MC, MCLLR
    alpha: float | None = None # Ridge baseline

    def __post_init__(self):
        # validate based on model_type 
        if self.model_type == "lwlr" and self.bandwidth is None:
            raise ValueError("Bandwidth is required for LWLR")
        if self.model_type in ["mc_linear","mcllr", "mcllr_no_geometry"] and self.n_simulations is None:
            raise ValueError(f"n_simulations is required for {self.model_type}")
        if self.model_type in ["mc_linear", "mcllr", "mcllr_no_geometry", "mcllr_no_stochasticity"] and self.subsample_size is None:
            raise ValueError(f"subsample_size is required for {self.model_type}")
        
    @classmethod
    def from_yaml(cls, path:str | Path) -> "ModelCofig":
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str | Path):
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)

@dataclass
class ExperimentConfig:
    """Experiment-level settings."""
    dataset_path: str
    train_ratio: float
    models: list[ModelCofig]
    random_state: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        config_dict["models"] = [
            ModelCofig(**m) for m in config_dict["models"]
        ]
        return cls(**config_dict)
