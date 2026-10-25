"""
Configuration for the improved pipeline.

Example (experiments/improved/config_improved.yaml)::

    dataset_path: data/raw/F-F_Research_Data_5_Factors_2x3.csv
    random_state: 42
    benchmark: historical_mean
    evaluation: {train_window: 240, test_window: 12}
    tuning: {inner_train_window: 180, inner_test_window: 12}
    models:
      - name: ridge
        model_type: ridge
        preprocessing: [standardize]
        grid: {alpha: [0.1, 1.0, 10.0]}
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from lrd.core.base import LocalRiskModel
from lrd.core.data.loader import TARGETS
from lrd.improved.models import MODEL_REGISTRY, STOCHASTIC_MODELS
from lrd.improved.preprocessing import PreparedModel, build_transformer
from lrd.improved.tuning import TunedModel, expand_grid


@dataclass
class ImprovedModelConfig:
    """One model in the improved system.

    ``params`` are fixed hyperparameters; ``grid`` lists values tuned by
    nested walk-forward inside every fold. ``preprocessing`` steps are fitted
    on training data only.
    """

    name: str
    model_type: str
    preprocessing: List[Any] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    grid: Dict[str, List[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type '{self.model_type}' for '{self.name}'. "
                f"Expected one of: {sorted(MODEL_REGISTRY)}"
            )
        accepted = set(inspect.signature(MODEL_REGISTRY[self.model_type]).parameters) - {"random_state"}
        for section, keys in (("params", self.params), ("grid", self.grid)):
            unknown = set(keys) - accepted
            if unknown:
                raise ValueError(f"{self.name}: unknown {section} {sorted(unknown)}; accepted: {sorted(accepted)}")
        overlap = set(self.params) & set(self.grid)
        if overlap:
            raise ValueError(f"{self.name}: {sorted(overlap)} appear in both params and grid.")
        for key, values in self.grid.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"{self.name}: grid '{key}' must be a non-empty list.")
        for step in self.preprocessing:
            build_transformer(step)  # validates the step specification
        for params in self.candidates():  # every combination must be a valid model
            MODEL_REGISTRY[self.model_type](**params)

    def candidates(self) -> List[Dict[str, Any]]:
        return [{**self.params, **combo} for combo in expand_grid(self.grid)]

    def max_value(self, key: str) -> Any:
        values = [c[key] for c in self.candidates() if key in c]
        return max(values) if values else None


@dataclass
class ImprovedExperimentConfig:
    """Experiment settings for the improved pipeline."""

    dataset_path: str
    models: List[ImprovedModelConfig]
    benchmark: str = "historical_mean"
    target: str = "return"  # "return" or "abs_return" (volatility proxy); see lrd.core.data.loader
    random_state: int = 42
    train_window: int = 240
    test_window: int = 12
    step: int | None = None
    inner_train_window: int = 180
    inner_test_window: int = 12
    strict_validation: bool = True
    outlier_threshold: float = 5.0

    def __post_init__(self) -> None:
        names = [m.name for m in self.models]
        if not names:
            raise ValueError("At least one model is required.")
        if len(set(names)) != len(names):
            raise ValueError(f"Model names must be unique: {names}")
        if self.benchmark not in names:
            raise ValueError(f"benchmark '{self.benchmark}' must be one of the model names {names}.")
        if self.target not in TARGETS:
            raise ValueError(f"Unknown target '{self.target}'. Expected one of: {sorted(TARGETS)}")
        for label, value in [("train_window", self.train_window), ("test_window", self.test_window),
                             ("inner_train_window", self.inner_train_window),
                             ("inner_test_window", self.inner_test_window)]:
            if value <= 0:
                raise ValueError(f"{label} must be positive.")
        if any(m.grid for m in self.models) and \
                self.inner_train_window + self.inner_test_window > self.train_window:
            raise ValueError("inner_train_window + inner_test_window must fit inside train_window.")
        for m in self.models:
            largest = m.max_value("subsample_size")
            limit = self.inner_train_window if m.grid else self.train_window
            if largest is not None and largest > limit:
                raise ValueError(
                    f"{m.name}: subsample_size {largest} exceeds the {limit}-month "
                    f"{'inner ' if m.grid else ''}training window."
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ImprovedExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        missing = {"dataset_path", "models"} - set(raw)
        if missing:
            raise ValueError(f"Improved config {path} is missing required keys: {sorted(missing)}")
        evaluation = raw.pop("evaluation", {}) or {}
        tuning = raw.pop("tuning", {}) or {}
        validation = raw.pop("validation", {}) or {}
        models = [ImprovedModelConfig(**m) for m in raw.pop("models")]
        return cls(
            models=models,
            **raw,
            **evaluation,
            **tuning,
            **{("strict_validation" if k == "strict" else k): v for k, v in validation.items()},
        )

    def build_model(self, model_config: ImprovedModelConfig) -> LocalRiskModel:
        """A fresh, unfitted model: preparation + estimator, tuned if a grid is given."""
        cls_ = MODEL_REGISTRY[model_config.model_type]
        seed = {"random_state": self.random_state} if model_config.model_type in STOCHASTIC_MODELS else {}

        def make(params: Dict[str, Any]) -> LocalRiskModel:
            steps = [build_transformer(s) for s in model_config.preprocessing]
            return PreparedModel(steps, cls_(**params, **seed))

        candidates = model_config.candidates()
        if model_config.grid:
            return TunedModel(make, candidates, self.inner_train_window, self.inner_test_window)
        return make(candidates[0])
