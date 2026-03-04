from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

class Evaluator:
    """Unified evaluation pipeline matching assumptions."""

    @staticmethod
    def evaluate(
        model: LocalRiskModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> EvaluationResult:
        """Evaluate model on test set."""

        y_pred, y_unc = model.predict_with_uncertainty(X_test)

        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test,y_pred)),
            "mae": mean_absolute_error(y_test,y_pred),
            "r2": r2_score(y_test, y_pred),
        }

        #Calibration for uncertainty models 
        if y_unc is not None:
            metrics["coverage_95"] = np.mean(
                np.abs(y_test -y_pred) <= 1.96 * y_unc
            )
            metrics["avg_std"] = np.mean(y_unc)

        return EvaluationResult(
            model_name = model.model_name,
            metrics = metrics,
            predictions = y_pred,
            uncertainties = y_unc,
            ground_truth = y_test,
        )
    
    @staticmethod
    def compare_models(
        models: List[LocalRiskModel],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, EvaluationResult]:
        """Evaluate all models on same test set."""
        return {
            model.model_name: Evaluator.evaluate(model, X_test, y_test)
            for model in models
        }