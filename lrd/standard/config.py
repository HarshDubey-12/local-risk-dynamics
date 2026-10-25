from dataclasses import dataclass, asdict, replace
from typing import Literal, Dict, Any, get_args
import yaml
from pathlib import Path

ModelType = Literal[
    "global_linear",
    "lwlr",
    "mc_linear",
    "mcllr",
    "mcllr_no_geometry",
    "mcllr_no_stochasticity",
    "ridge_linear",
]

# Hyperparameters each model type cannot run without.
_REQUIRED_PARAMS: Dict[str, tuple[str, ...]] = {
    "global_linear": (),
    "ridge_linear": (),
    "lwlr": ("bandwidth",),
    "mc_linear": ("n_simulations", "subsample_size"),
    "mcllr": ("bandwidth", "n_simulations", "subsample_size"),
    "mcllr_no_geometry": ("n_simulations", "subsample_size"),
    "mcllr_no_stochasticity": ("bandwidth", "subsample_size"),
}

@dataclass
class ModelConfig:
    """Configuration for any model in the framework."""
    model_type: ModelType
    fit_intercept: bool = True
    random_state: int | None = None
    # Model specific params
    bandwidth: float | None = None # LWLR, MCLLR
    n_simulations: int | None = None # MC, MCLLR
    subsample_size: int | None = None # MC, MCLLR
    alpha: float | None = None # Ridge baseline

    def __post_init__(self):
        # validate based on model_type
        if self.model_type not in get_args(ModelType):
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. "
                f"Expected one of: {', '.join(get_args(ModelType))}"
            )
        for param in _REQUIRED_PARAMS[self.model_type]:
            if getattr(self, param) is None:
                raise ValueError(f"{param} is required for {self.model_type}")

        if self.bandwidth is not None and self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")
        if self.n_simulations is not None and self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive.")
        if self.subsample_size is not None and self.subsample_size <= 1:
            raise ValueError("subsample_size must be greater than 1.")
        if self.alpha is not None and self.alpha < 0:
            raise ValueError("alpha must be non-negative.")

    @classmethod
    def from_yaml(cls, path:str | Path) -> "ModelConfig":
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
    models: list[ModelConfig]
    random_state: int = 42

    def __post_init__(self):
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1 (exclusive).")
        if not self.models:
            raise ValueError("Experiment config must define at least one model.")

        # Stochastic models without their own seed inherit the experiment seed,
        # so a run is reproducible from the config alone.
        self.models = [
            replace(m, random_state=self.random_state) if m.random_state is None else m
            for m in self.models
        ]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        missing = {"dataset_path", "train_ratio", "models"} - set(config_dict or {})
        if missing:
            raise ValueError(
                f"Experiment config {path} is missing required keys: {sorted(missing)}"
            )
        config_dict["models"] = [
            ModelConfig(**m) for m in config_dict["models"]
        ]
        return cls(**config_dict)
