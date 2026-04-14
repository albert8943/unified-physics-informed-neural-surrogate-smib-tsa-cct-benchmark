"""
Diagnostic utilities for prediction error analysis.

This module provides functions to diagnose common issues with PINN predictions,
including scaler mismatches, angle wrapping, time alignment, and more.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def diagnose_prediction_issues(
    delta_pred: np.ndarray,
    delta_true: np.ndarray,
    omega_pred: np.ndarray,
    omega_true: np.ndarray,
    scenario_data: pd.DataFrame,
    scalers: Dict[str, StandardScaler],
    scenario_id: Optional[int] = None,
) -> List[str]:
    """
    Quick diagnostic for prediction issues.

    Parameters:
    -----------
    delta_pred : np.ndarray
        Predicted rotor angles
    delta_true : np.ndarray
        True rotor angles
    omega_pred : np.ndarray
        Predicted rotor speeds
    omega_true : np.ndarray
        True rotor speeds
    scenario_data : pd.DataFrame
        Scenario data (must include time, delta, omega columns)
    scalers : dict
        Dictionary of fitted scalers
    scenario_id : int, optional
        Scenario ID for logging

    Returns:
    --------
    list
        List of issue messages (empty if no issues found)
    """
    issues = []
    scenario_id_str = f"Scenario {scenario_id}" if scenario_id is not None else "Scenario"

    # 1. Check IC enforcement
    if len(delta_pred) > 0 and len(delta_true) > 0:
        delta0_true = scenario_data["delta"].iloc[0] if len(scenario_data) > 0 else None
        delta0_pred = delta_pred[0]
        if delta0_true is not None and abs(delta0_pred - delta0_true) > 0.01:
            issues.append(
                f"❌ {scenario_id_str}: IC not enforced: delta0_pred={delta0_pred:.4f}, "
                f"delta0_true={delta0_true:.4f}, diff={abs(delta0_pred - delta0_true):.4f}"
            )

        omega0_true = scenario_data["omega"].iloc[0] if len(scenario_data) > 0 else None
        omega0_pred = omega_pred[0]
        if omega0_true is not None and abs(omega0_pred - omega0_true) > 0.01:
            issues.append(
                f"❌ {scenario_id_str}: IC not enforced: omega0_pred={omega0_pred:.4f}, "
                f"omega0_true={omega0_true:.4f}, diff={abs(omega0_pred - omega0_true):.4f}"
            )

    # 2. Check scaler mismatch
    if "delta0" in scalers and "delta" in scalers:
        delta0_mean = scalers["delta0"].mean_[0]
        delta0_std = scalers["delta0"].scale_[0]
        delta_mean = scalers["delta"].mean_[0]
        delta_std = scalers["delta"].scale_[0]

        mean_diff_pct = abs(delta0_mean - delta_mean) / (abs(delta_mean) + 1e-8) * 100
        std_diff_pct = abs(delta0_std - delta_std) / (abs(delta_std) + 1e-8) * 100

        if mean_diff_pct > 10 or std_diff_pct > 10:
            issues.append(
                f"⚠️ {scenario_id_str}: Scaler mismatch: delta0 vs delta scalers have different"
                f"statistics"
            )
            issues.append(f"      delta0 scaler: mean={delta0_mean:.6f}, std={delta0_std:.6f}")
            issues.append(f"      delta scaler: mean={delta_mean:.6f}, std={delta_std:.6f}")
            issues.append(
                f"      Mean difference: {mean_diff_pct:.1f}%, Std difference: {std_diff_pct:.1f}%"
            )

    # 3. Check angle wrapping
    if len(delta_pred) > 0 and len(delta_true) > 0:
        max_delta = np.max(np.abs(delta_true))
        if max_delta > np.pi:
            from utils.metrics import wrap_angle_error

            wrapped_error = wrap_angle_error(delta_pred, delta_true)
            unwrapped_error = delta_pred - delta_true

            if np.std(wrapped_error) < np.std(unwrapped_error) * 0.9:
                issues.append(
                    f"💡 {scenario_id_str}: Angle wrapping might improve metrics "
                    f"(wrapped std: {np.std(wrapped_error):.4f}, "
                    f"unwrapped std: {np.std(unwrapped_error):.4f})"
                )

    # 4. Check extreme values
    if len(delta_pred) > 0:
        max_delta_abs = np.max(np.abs(delta_pred))
        if max_delta_abs > 20:  # 20 rad ≈ 1146°
            issues.append(
                f"⚠️  {scenario_id_str}: Extreme delta predictions: max={max_delta_abs:.2f} rad "
                f"({np.degrees(max_delta_abs):.1f}°)"
            )

    # 5. Check time alignment
    if len(delta_pred) != len(delta_true):
        issues.append(
            f"⚠️  {scenario_id_str}: Length mismatch: pred={len(delta_pred)}, true={len(delta_true)}"
        )

    # Check if time arrays match
    if "time" in scenario_data.columns and len(scenario_data) > 0:
        scenario_time = scenario_data["time"].values
        if len(delta_pred) > 0:
            pred_time = np.linspace(scenario_time[0], scenario_time[-1], len(delta_pred))
            if len(pred_time) == len(scenario_time):
                time_diff = np.abs(pred_time - scenario_time)
                max_time_diff = np.max(time_diff)
                if max_time_diff > 0.01:  # 10ms tolerance
                    issues.append(
                        f"⚠️  {scenario_id_str}: Time misalignment detected: "
                        f"max_diff={max_time_diff:.4f} s"
                    )

    # 6. Check for NaN or Inf values
    if len(delta_pred) > 0:
        if np.any(np.isnan(delta_pred)) or np.any(np.isinf(delta_pred)):
            issues.append(f"❌ {scenario_id_str}: NaN or Inf values in delta_pred")
        if np.any(np.isnan(omega_pred)) or np.any(np.isinf(omega_pred)):
            issues.append(f"❌ {scenario_id_str}: NaN or Inf values in omega_pred")

    return issues


def validate_scaler_statistics(
    scalers: Dict[str, StandardScaler],
    data: Optional[pd.DataFrame] = None,
) -> List[str]:
    """
    Validate scaler statistics and check for potential issues.

    Parameters:
    -----------
    scalers : dict
        Dictionary of fitted scalers
    data : pd.DataFrame, optional
        Data to compare against (if provided, checks if data is within scaler distribution)

    Returns:
    --------
    list
        List of validation messages
    """
    messages = []

    # Check for delta0 vs delta mismatch
    if "delta0" in scalers and "delta" in scalers:
        delta0_mean = scalers["delta0"].mean_[0]
        delta0_std = scalers["delta0"].scale_[0]
        delta_mean = scalers["delta"].mean_[0]
        delta_std = scalers["delta"].scale_[0]

        mean_diff_pct = abs(delta0_mean - delta_mean) / (abs(delta_mean) + 1e-8) * 100
        std_diff_pct = abs(delta0_std - delta_std) / (abs(delta_std) + 1e-8) * 100

        if mean_diff_pct > 10 or std_diff_pct > 10:
            messages.append(
                "⚠️  Scaler mismatch: delta0 vs delta scalers have different statistics"
            )
            messages.append(f"      delta0 scaler: mean={delta0_mean:.6f}, std={delta0_std:.6f}")
            messages.append(f"      delta scaler: mean={delta_mean:.6f}, std={delta_std:.6f}")
            messages.append(
                f"      Mean difference: {mean_diff_pct:.1f}%, Std difference: {std_diff_pct:.1f}%"
            )
            messages.append(
                "      This may cause systematic offsets in predictions. "
                "Consider using the same scaler for both input and output."
            )

    # Check if data is within scaler distribution
    if data is not None:
        for key, scaler in scalers.items():
            if key in data.columns:
                data_values = data[key].values
                scaler_mean = scaler.mean_[0]
                scaler_std = scaler.scale_[0]

                # Check if data mean is within 2 standard deviations
                data_mean = np.mean(data_values)
                data_std = np.std(data_values)

                if abs(data_mean - scaler_mean) > 2 * scaler_std:
                    messages.append(
                        f"⚠️  Data mean ({data_mean:.6f}) differs significantly from "
                        f"scaler mean ({scaler_mean:.6f}) for {key}"
                    )
                if abs(data_std - scaler_std) > 0.5 * scaler_std:
                    messages.append(
                        f"⚠️  Data std ({data_std:.6f}) differs from scaler std "
                        f"({scaler_std:.6f}) for {key}"
                    )

    return messages
