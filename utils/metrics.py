"""
Evaluation Metrics for PINN Models.

This module provides comprehensive metrics for trajectory prediction,
CCT estimation, and parameter estimation.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def wrap_angle_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Compute angle error with wrapping to handle periodic angles.

    Rotor angles are periodic (2π = 360°), so angles that differ by multiples
    of 2π are equivalent. This function computes the wrapped error, which is
    the minimum angular difference between pred and true.

    Parameters:
    -----------
    pred : np.ndarray
        Predicted angles (in radians)
    true : np.ndarray
        True angles (in radians)

    Returns:
    --------
    np.ndarray
        Wrapped angle errors (in radians, range [-π, π])
    """
    error = pred - true
    # Wrap to [-π, π] using atan2(sin, cos) trick
    wrapped_error = np.arctan2(np.sin(error), np.cos(error))
    return wrapped_error


def compute_trajectory_metrics(
    delta_pred: np.ndarray,
    omega_pred: np.ndarray,
    delta_true: np.ndarray,
    omega_true: np.ndarray,
    t: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute metrics for trajectory prediction.

    Parameters:
    -----------
    delta_pred : np.ndarray
        Predicted rotor angles
    omega_pred : np.ndarray
        Predicted rotor speeds
    delta_true : np.ndarray
        True rotor angles
    omega_true : np.ndarray
        True rotor speeds
    t : np.ndarray, optional
        Time array (required for first swing metrics)

    Returns:
    --------
    dict : Dictionary of metrics
    """
    metrics = {}

    # Delta metrics (with angle wrapping for periodic angles)
    # Compute wrapped error to handle angles near 0°/360° boundaries
    delta_error_wrapped = wrap_angle_error(delta_pred, delta_true)
    metrics["delta_rmse"] = np.sqrt(mean_squared_error(delta_true, delta_pred))
    metrics["delta_rmse_wrapped"] = np.sqrt(np.mean(delta_error_wrapped**2))
    metrics["delta_mae"] = mean_absolute_error(delta_true, delta_pred)
    metrics["delta_mae_wrapped"] = np.mean(np.abs(delta_error_wrapped))

    # Improved MAPE: Use symmetric MAPE (sMAPE) for better handling of near-zero values
    # sMAPE = mean(|pred - true| / (|pred| + |true| + eps)) * 100
    # This is more robust when values are close to zero
    delta_denominator = np.abs(delta_pred) + np.abs(delta_true) + 1e-8
    metrics["delta_mape"] = np.mean(np.abs(delta_true - delta_pred) / delta_denominator) * 100

    # Also compute traditional MAPE for values above threshold (for comparison)
    delta_threshold = np.max(np.abs(delta_true)) * 0.01  # 1% of max value
    valid_mask = np.abs(delta_true) > delta_threshold
    if np.any(valid_mask):
        metrics["delta_mape_above_threshold"] = (
            np.mean(
                np.abs(
                    (delta_true[valid_mask] - delta_pred[valid_mask])
                    / (delta_true[valid_mask] + 1e-8)
                )
            )
            * 100
        )
    else:
        metrics["delta_mape_above_threshold"] = 0.0

    metrics["delta_max_error"] = np.max(np.abs(delta_true - delta_pred))
    metrics["delta_max_error_wrapped"] = np.max(np.abs(delta_error_wrapped))
    metrics["delta_r2"] = r2_score(delta_true, delta_pred)

    # Use wrapped error for R2 if it's significantly better (indicates wrapping helped)
    if metrics["delta_rmse_wrapped"] < metrics["delta_rmse"] * 0.9:
        # Wrapping significantly improved error, use wrapped version for R2
        # Note: R2 with wrapped errors is approximate but more meaningful for periodic data
        delta_pred_wrapped = delta_true + delta_error_wrapped
        metrics["delta_r2_wrapped"] = r2_score(delta_true, delta_pred_wrapped)
    else:
        metrics["delta_r2_wrapped"] = metrics["delta_r2"]

    # Normalized RMSE (NRMSE) - useful for comparing across different scales
    delta_range = np.max(delta_true) - np.min(delta_true)
    if delta_range > 1e-8:
        metrics["delta_nrmse"] = metrics["delta_rmse"] / delta_range
    else:
        metrics["delta_nrmse"] = 0.0

    # Omega metrics
    metrics["omega_rmse"] = np.sqrt(mean_squared_error(omega_true, omega_pred))
    metrics["omega_mae"] = mean_absolute_error(omega_true, omega_pred)

    # Improved MAPE: Use symmetric MAPE (sMAPE) for better handling of near-zero values
    omega_denominator = np.abs(omega_pred) + np.abs(omega_true) + 1e-8
    metrics["omega_mape"] = np.mean(np.abs(omega_true - omega_pred) / omega_denominator) * 100

    # Also compute traditional MAPE for values above threshold (for comparison)
    omega_threshold = np.max(np.abs(omega_true)) * 0.01  # 1% of max value
    valid_mask = np.abs(omega_true) > omega_threshold
    if np.any(valid_mask):
        metrics["omega_mape_above_threshold"] = (
            np.mean(
                np.abs(
                    (omega_true[valid_mask] - omega_pred[valid_mask])
                    / (omega_true[valid_mask] + 1e-8)
                )
            )
            * 100
        )
    else:
        metrics["omega_mape_above_threshold"] = 0.0

    metrics["omega_max_error"] = np.max(np.abs(omega_true - omega_pred))
    metrics["omega_r2"] = r2_score(omega_true, omega_pred)

    # Normalized RMSE (NRMSE)
    omega_range = np.max(omega_true) - np.min(omega_true)
    if omega_range > 1e-8:
        metrics["omega_nrmse"] = metrics["omega_rmse"] / omega_range
    else:
        metrics["omega_nrmse"] = 0.0

    # Maximum relative error for omega (matching delta metric)
    # Use improved relative error calculation (symmetric, handles near-zero values)
    omega_max_pred = np.max(np.abs(omega_pred))
    omega_max_true = np.max(np.abs(omega_true))
    omega_max_denominator = np.abs(omega_max_pred) + np.abs(omega_max_true) + 1e-8
    metrics["omega_max_relative_error"] = (
        np.abs(omega_max_pred - omega_max_true) / omega_max_denominator * 100
    )

    # Combined metrics
    metrics["combined_rmse"] = np.sqrt(
        (metrics["delta_rmse"] ** 2 + metrics["omega_rmse"] ** 2) / 2
    )
    metrics["combined_mae"] = (metrics["delta_mae"] + metrics["omega_mae"]) / 2

    # Energy-based metrics (transient energy function)
    # Transient energy: E = 0.5*M*(omega-1)^2 + integral(Pm - Pe) ddelta
    # This is a simplified version - full implementation would require integration
    energy_pred = 0.5 * (omega_pred - 1.0) ** 2  # Simplified kinetic energy
    energy_true = 0.5 * (omega_true - 1.0) ** 2
    metrics["energy_rmse"] = np.sqrt(mean_squared_error(energy_true, energy_pred))
    metrics["energy_mae"] = mean_absolute_error(energy_true, energy_pred)

    # Stability margin accuracy
    # Maximum rotor angle is key indicator of stability
    delta_max_pred = np.max(np.abs(delta_pred))
    delta_max_true = np.max(np.abs(delta_true))
    metrics["delta_max_abs_error"] = np.abs(delta_max_pred - delta_max_true)
    # Use improved relative error calculation
    delta_max_denominator = np.abs(delta_max_pred) + np.abs(delta_max_true) + 1e-8
    metrics["delta_max_relative_error"] = (
        np.abs(delta_max_pred - delta_max_true) / delta_max_denominator * 100
    )

    # Stability classification accuracy
    # WARNING: This metric is computed incorrectly when trajectories from multiple scenarios
    # are concatenated. It computes max across ALL trajectories, not per-scenario.
    # For correct per-scenario stability classification, compute it in the evaluation script
    # by comparing predicted vs true stability for each scenario separately.
    #
    # This value is kept for backward compatibility but should NOT be used for reporting.
    # Use per-scenario stability comparisons instead.
    # Use same logic as determine_stability_180deg for consistency
    # Convert to degrees and check if max angle < 180 degrees
    delta_max_pred_deg = delta_max_pred * (180.0 / np.pi)
    delta_max_true_deg = delta_max_true * (180.0 / np.pi)
    stable_threshold_deg = 180.0
    stable_pred = delta_max_pred_deg < stable_threshold_deg
    stable_true = delta_max_true_deg < stable_threshold_deg
    metrics["stability_classification_accuracy"] = float(stable_pred == stable_true)

    # First swing prediction accuracy
    # Find first swing peak (maximum in first oscillation)
    if t is not None and len(delta_pred) > 10:
        # Approximate first swing as first 2 seconds or first local maximum
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        first_swing_idx = min(len(delta_pred) // 2, int(2.0 / dt) if dt > 0 else len(delta_pred))
        first_swing_pred = np.max(np.abs(delta_pred[:first_swing_idx]))
        first_swing_true = np.max(np.abs(delta_true[:first_swing_idx]))
        metrics["first_swing_error"] = np.abs(first_swing_pred - first_swing_true)
        # Use improved relative error calculation
        first_swing_denominator = np.abs(first_swing_pred) + np.abs(first_swing_true) + 1e-8
        metrics["first_swing_relative_error"] = (
            np.abs(first_swing_pred - first_swing_true) / first_swing_denominator * 100
        )

    return metrics


def compute_cct_metrics(cct_pred: np.ndarray, cct_true: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for CCT estimation.

    Parameters:
    -----------
    cct_pred : np.ndarray
        Predicted CCT values
    cct_true : np.ndarray
        True CCT values

    Returns:
    --------
    dict : Dictionary of metrics
    """
    metrics = {}

    # Absolute error (in seconds)
    metrics["cct_mae"] = mean_absolute_error(cct_true, cct_pred)
    metrics["cct_rmse"] = np.sqrt(mean_squared_error(cct_true, cct_pred))
    metrics["cct_max_error"] = np.max(np.abs(cct_true - cct_pred))

    # Relative error (percentage) - using improved MAPE
    # For CCT, values are typically > 0, so traditional MAPE is usually fine
    # But use symmetric MAPE for robustness
    cct_denominator = np.abs(cct_pred) + np.abs(cct_true) + 1e-8
    metrics["cct_mape"] = np.mean(np.abs(cct_true - cct_pred) / cct_denominator) * 100
    metrics["cct_max_relative_error"] = np.max(np.abs(cct_true - cct_pred) / cct_denominator) * 100

    # Error in milliseconds
    metrics["cct_mae_ms"] = metrics["cct_mae"] * 1000
    metrics["cct_rmse_ms"] = metrics["cct_rmse"] * 1000
    metrics["cct_max_error_ms"] = metrics["cct_max_error"] * 1000

    # R2 score
    metrics["cct_r2"] = r2_score(cct_true, cct_pred)

    # Classification accuracy (stable/unstable)
    # For CCT, we can classify based on whether predicted CCT is close to true CCT
    # within a tolerance (e.g., 5ms)
    tolerance_ms = 5.0
    tolerance_s = tolerance_ms / 1000.0
    correct_classifications = np.sum(np.abs(cct_true - cct_pred) < tolerance_s)
    metrics["cct_classification_accuracy"] = correct_classifications / len(cct_true)

    # Normalized RMSE
    cct_range = np.max(cct_true) - np.min(cct_true)
    if cct_range > 1e-8:
        metrics["cct_nrmse"] = metrics["cct_rmse"] / cct_range
    else:
        metrics["cct_nrmse"] = 0.0

    return metrics


def compute_parameter_metrics(
    H_pred: np.ndarray, D_pred: np.ndarray, H_true: np.ndarray, D_true: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for parameter estimation.

    Parameters:
    -----------
    H_pred : np.ndarray
        Predicted H values
    D_pred : np.ndarray
        Predicted D values
    H_true : np.ndarray
        True H values
    D_true : np.ndarray
        True D values

    Returns:
    --------
    dict : Dictionary of metrics
    """
    metrics = {}

    # H metrics
    metrics["H_mae"] = mean_absolute_error(H_true, H_pred)
    metrics["H_rmse"] = np.sqrt(mean_squared_error(H_true, H_pred))
    # Improved MAPE using symmetric MAPE
    H_denominator = np.abs(H_pred) + np.abs(H_true) + 1e-8
    metrics["H_mape"] = np.mean(np.abs(H_true - H_pred) / H_denominator) * 100
    metrics["H_max_error"] = np.max(np.abs(H_true - H_pred))
    metrics["H_max_relative_error"] = np.max(np.abs(H_true - H_pred) / H_denominator) * 100
    metrics["H_r2"] = r2_score(H_true, H_pred)

    # D metrics
    metrics["D_mae"] = mean_absolute_error(D_true, D_pred)
    metrics["D_rmse"] = np.sqrt(mean_squared_error(D_true, D_pred))
    # Improved MAPE using symmetric MAPE
    D_denominator = np.abs(D_pred) + np.abs(D_true) + 1e-8
    metrics["D_mape"] = np.mean(np.abs(D_true - D_pred) / D_denominator) * 100
    metrics["D_max_error"] = np.max(np.abs(D_true - D_pred))
    metrics["D_max_relative_error"] = np.max(np.abs(D_true - D_pred) / D_denominator) * 100
    metrics["D_r2"] = r2_score(D_true, D_pred)

    # Combined metrics
    metrics["combined_mape"] = (metrics["H_mape"] + metrics["D_mape"]) / 2
    metrics["combined_mae"] = (metrics["H_mae"] + metrics["D_mae"]) / 2

    return metrics


def compute_metrics(predictions: Dict, targets: Dict, task: str = "trajectory") -> Dict[str, float]:
    """
    Compute metrics for a specific task.

    Parameters:
    -----------
    predictions : dict
        Dictionary of predictions
    targets : dict
        Dictionary of targets
    task : str
        Task type: 'trajectory', 'cct', or 'parameter'

    Returns:
    --------
    dict : Dictionary of metrics
    """
    if task == "trajectory":
        return compute_trajectory_metrics(
            predictions["delta"], predictions["omega"], targets["delta"], targets["omega"]
        )
    elif task == "cct":
        return compute_cct_metrics(predictions["cct"], targets["cct"])
    elif task == "parameter":
        return compute_parameter_metrics(
            predictions["H"], predictions["D"], targets["H"], targets["D"]
        )
    else:
        raise ValueError("Unknown task: {task}")
