"""Improved models. Each returns a predictive standard deviation."""

from lrd.improved.models.benchmark import HistoricalMean
from lrd.improved.models.mcllr import ImprovedMCLLR
from lrd.improved.models.ridge import RidgeRegression

# model_type (as used in experiments/improved/*.yaml) -> class
MODEL_REGISTRY = {
    "historical_mean": HistoricalMean,
    "ridge": RidgeRegression,
    "mcllr_improved": ImprovedMCLLR,
}

# Model types whose constructor takes a random_state.
STOCHASTIC_MODELS = {"mcllr_improved"}

__all__ = ["HistoricalMean", "ImprovedMCLLR", "RidgeRegression", "MODEL_REGISTRY", "STOCHASTIC_MODELS"]
