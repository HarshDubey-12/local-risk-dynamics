"""
Plotting helpers

Common plotting utilities for figures used in notebooks and reports.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Any


def plot_predictions_vs_true(y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
    """
    Scatter plot of predictions vs true values with perfect fit line.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=50)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs True Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_uncertainty(y_pred: np.ndarray, y_std: np.ndarray, save_path: Optional[str] = None):
    """
    Error bar plot showing predictions with uncertainty.
    """
    plt.figure(figsize=(10, 6))
    indices = np.arange(len(y_pred))
    plt.errorbar(indices, y_pred, yerr=y_std, fmt='o', alpha=0.7, capsize=3, elinewidth=1)
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction')
    plt.title('Predictions with Uncertainty (95% CI)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_kernel_weights(distances: np.ndarray, weights: np.ndarray, kernel_name: str, save_path: Optional[str] = None):
    """
    Plot kernel weights as a function of distance.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(distances, weights, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Distance')
    plt.ylabel('Weight')
    plt.title(f'{kernel_name} Kernel Weights')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results: Dict[str, Any], metric: str = 'r2', save_path: Optional[str] = None):
    """
    Bar plot comparing models on a given metric.

    Args:
        results: Dict of model_name -> EvaluationResult object with .metrics dict
        metric: Metric key to plot (e.g., 'r2', 'rmse')
    """
    models = list(results.keys())
    values = [results[m].metrics.get(metric, np.nan) for m in models]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, alpha=0.8, edgecolor='k')
    plt.xlabel('Model')
    plt.ylabel(metric.upper().replace('_', ' '))
    plt.title(f'Model Comparison: {metric.upper().replace("_", " ")}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_time_series(y: np.ndarray, dates: Optional[np.ndarray] = None, title: str = 'Time Series', save_path: Optional[str] = None):
    """
    Plot time series data.
    """
    plt.figure(figsize=(12, 6))
    if dates is not None:
        plt.plot(dates, y, 'b-', linewidth=1)
        plt.xlabel('Date')
    else:
        plt.plot(y, 'b-', linewidth=1)
        plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
    """
    Plot residuals (true - pred) vs predicted values.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=50)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
