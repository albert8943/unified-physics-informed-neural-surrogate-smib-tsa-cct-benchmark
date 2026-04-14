"""
Utilities Module for PINN Training and Evaluation.

This module provides metrics, visualization, data utilities, stability checking,
and CCT estimation via binary search.
"""

from .cct_binary_search import estimate_cct_batch, estimate_cct_binary_search
from .data_utils import load_dataset, prepare_batch, save_dataset
from .metrics import (
    compute_cct_metrics,
    compute_metrics,
    compute_parameter_metrics,
    compute_trajectory_metrics,
)
from .stability_checker import (
    check_stability,
    check_stability_batch,
    stability_angle_metric_rad,
)
from .visualization import plot_comparison, plot_loss_curves, plot_trajectories

__all__ = [
    "compute_metrics",
    "compute_trajectory_metrics",
    "compute_cct_metrics",
    "compute_parameter_metrics",
    "plot_trajectories",
    "plot_loss_curves",
    "plot_comparison",
    "load_dataset",
    "save_dataset",
    "prepare_batch",
    "check_stability",
    "check_stability_batch",
    "stability_angle_metric_rad",
    "estimate_cct_binary_search",
    "estimate_cct_batch",
]
