from typing import Type, Dict
from .base import LocalRiskModel
from .global_linear import GlobalLinearRegression
from .lwlr import LocallyWeightedLinearRegression
from .mc_linear import MonteCarloLinearRegression
from .mcllr import MonteCarloLocalLinearRegression
from ..config import ModelCofig

class Modelfactory:
    """Registry pattern for model instatntiation."""

    _registry: Dict[str, Type[LocalRiskModel]] = {
        "global_linear": GlobalLinearRegression,
        "lwlr": LocallyWeightedLinearRegression,
        "mc_linear": MonteCarloLinearRegression,
        "mcllr": MonteCarloLocalLinearRegression,
    }

    @classmethod
    def create(cls, config: ModelCofig) -> LocalRiskModel:
        """Instantiate model from config."""
        model_cls = cls._registry.get(config.modle_type)
        if model_cls is None:
            raise ValueError(f"Unknown model: {config.model_type }")
        
        # Build kwargs from config 
        kwargs = {
            "fit_intercept": config.fit_intercept,
            "random_state": config.random_state,
        }

        # Add model-specific params
        if config.model_type == "lwlr":
            kwargs["bandwidth"] = config.bandwidth
        elif config.model_type in ["mc_linear","mcllr"]:
            kwargs["n_simulations"] = config.n_simulations
            kwargs["subsample_size"] = config.subsample_size
            if config.model_type == "mcllr":
                kwargs["bandwidth"] = config.bandwidth
            
        return model_cls(**kwargs)
    
    @classmethod 
    def register(cls, name: str, model_cls: Type[LocalRiskModel]):
        """register new model type dynamically."""
        cls._registry[name] = model_cls

# Usage 
config = ModelCofig.from_yaml("experiments/config_mcllr.yaml")
model = Modelfactory.create(config)