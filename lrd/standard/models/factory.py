from typing import Type, Dict
from lrd.core.base import LocalRiskModel
from lrd.standard.models.global_linear import GlobalLinearRegression
from lrd.standard.models.lwlr import LocallyWeightedLinearRegression
from lrd.standard.models.mc_linear import MonteCarloLinearRegression
from lrd.standard.models.mcllr import MonteCarloLocalLinearRegression
from lrd.standard.models.mcllr_ablations import MCLLRNoGeometry, MCLLRNoStochasticity
from lrd.standard.models.ridge_linear import RidgeLinearRegression
from lrd.standard.config import ModelConfig

class ModelFactory:
    """Registry pattern for model instantiation."""

    _registry: Dict[str, Type[LocalRiskModel]] = {
        "global_linear": GlobalLinearRegression,
        "lwlr": LocallyWeightedLinearRegression,
        "mc_linear": MonteCarloLinearRegression,
        "mcllr": MonteCarloLocalLinearRegression,
        "mcllr_no_geometry": MCLLRNoGeometry,
        "mcllr_no_stochasticity": MCLLRNoStochasticity,
        "ridge_linear": RidgeLinearRegression,
    }

    @classmethod
    def create(cls, config: ModelConfig) -> LocalRiskModel:
        """Instantiate model from config."""
        model_cls = cls._registry.get(config.model_type)
        if model_cls is None:
            raise ValueError(f"Unknown model: {config.model_type }")
        
        # Build model-specific kwargs to avoid passing unsupported args.
        if config.model_type == "global_linear":
            kwargs = {
                "fit_intercept": config.fit_intercept,
            }
        elif config.model_type == "lwlr":
            kwargs = {
                "bandwidth": config.bandwidth,
                "fit_intercept": config.fit_intercept,
            }
        elif config.model_type == "mc_linear":
            kwargs = {
                "n_simulations": config.n_simulations,
                "subsample_size": config.subsample_size,
                "fit_intercept": config.fit_intercept,
                "random_state": config.random_state,
            }
        elif config.model_type == "mcllr":
            kwargs = {
                "bandwidth": config.bandwidth,
                "n_simulations": config.n_simulations,
                "subsample_size": config.subsample_size,
                "fit_intercept": config.fit_intercept,
                "random_state": config.random_state,
            }
        elif config.model_type == "mcllr_no_geometry":
            kwargs = {
                "n_simulations": config.n_simulations,
                "subsample_size": config.subsample_size,
                "fit_intercept": config.fit_intercept,
                "random_state": config.random_state,
            }
        elif config.model_type == "mcllr_no_stochasticity":
            kwargs = {
                "bandwidth": config.bandwidth,
                "subsample_size": config.subsample_size,
                "fit_intercept": config.fit_intercept,
            }
        elif config.model_type == "ridge_linear":
            kwargs = {
                "alpha": config.alpha if config.alpha is not None else 1.0,
                "fit_intercept": config.fit_intercept,
            }
        else:
            raise ValueError(f"Unknown model: {config.model_type}")
            
        return model_cls(**kwargs)
    
    @classmethod 
    def register(cls, name: str, model_cls: Type[LocalRiskModel]):
        """register new model type dynamically."""
        cls._registry[name] = model_cls
