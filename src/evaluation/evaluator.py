from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
import numpy as np
from ..evaluation import metrics as metrics_module
from ..models.base import LocalRiskModel
from ..data.pipeline import Dataset

@dataclass
class EvaluationResult:
    """Unified evaluation output."""
    model_name: str
    metrics: Dict[str,float]
    predictions: np.ndarray
    uncertainties: np.ndarray | None = None
    ground_truth: np.ndarray = None

    def __repr__(self):
        metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in self.metrics.items())
        return f"<EvaluationResult {self.model_name} | {metric_str}>"

class Evaluator:
    """Unified evaluation pipeline matching assumptions."""

    @staticmethod
    def evaluate(
        model: LocalRiskModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
        extra_metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    ) -> EvaluationResult:
        """Evaluate a model on the test set and return a full result object.

        Parameters
        ----------
        model : LocalRiskModel
            Fitted model implementing ``predict_with_uncertainty``.
        X_test : np.ndarray
            Feature matrix for testing.
        y_test : np.ndarray
            Ground truth targets.
        extra_metrics : dict, optional
            Additional metric functions taking (y_true, y_pred) and
            returning a float. They will be merged into the default set.
        """

        y_pred, y_unc = model.predict_with_uncertainty(X_test)

        # basic regression scores
        metrics: Dict[str, float] = {
            "rmse": metrics_module.rmse(y_test, y_pred),
            "mae": metrics_module.mae(y_test, y_pred),
            "mape": metrics_module.mape(y_test, y_pred),
            "r2": metrics_module.r2(y_test, y_pred),
            "explained_variance": metrics_module.explained_variance(y_test, y_pred),
            "median_ae": metrics_module.median_ae(y_test, y_pred),
        }

        # calibration for models that return uncertainty
        if y_unc is not None:
            metrics["coverage_95"] = metrics_module.coverage_95(y_test, y_pred, y_unc)
            metrics["avg_std"] = metrics_module.avg_std(y_unc)

        # compute any extras provided by caller
        if extra_metrics:
            for name, func in extra_metrics.items():
                try:
                    metrics[name] = func(y_test, y_pred)
                except Exception:
                    metrics[name] = np.nan

        return EvaluationResult(
            model_name=model.model_name,
            metrics=metrics,
            predictions=y_pred,
            uncertainties=y_unc,
            ground_truth=y_test,
        )
    
    @staticmethod
    def compare_models(
        models: List[LocalRiskModel],
        X_test: np.ndarray,
        y_test: np.ndarray,
        extra_metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    ) -> Dict[str, EvaluationResult]:
        """Evaluate all models on same test set.

        Parameters mirror ``evaluate``; ``extra_metrics`` are passed through.
        """
        return {
            model.model_name: Evaluator.evaluate(model, X_test, y_test, extra_metrics)
            for model in models
        }