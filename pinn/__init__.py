"""
Physics-Informed Neural Network (PINN) Framework for Power System Transient Stability Analysis.

This module implements PINN models for:
- Trajectory prediction (δ, ω) - Forward problem
- Parameter estimation (H, D) - Inverse problem
- Multimachine extension

Note:
    CCT estimation is now performed using binary search with the trajectory model.
    See `utils.cct_binary_search.estimate_cct_binary_search()` for the new approach.
    The CCTEstimationPINN class is deprecated but kept for reference.
"""

from .cct_estimation import CCTEstimationPINN  # Deprecated - use binary search instead
from .core import (
    PINN,
    LossWeightScheduler,
    NormalizedStateLoss,
    PhysicsInformedLoss,
    Scale,
    Standardise,
)
from .parameter_estimation import ParameterEstimationPINN
from .trajectory_prediction import TrajectoryPredictionPINN

__all__ = [
    "PINN",
    "PhysicsInformedLoss",
    "NormalizedStateLoss",
    "LossWeightScheduler",
    "Standardise",
    "Scale",
    "TrajectoryPredictionPINN",
    "CCTEstimationPINN",  # Deprecated - use utils.cct_binary_search instead
    "ParameterEstimationPINN",
]
