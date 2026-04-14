"""
Angle Filtering Utility for Trajectory Data.

This module provides functions to filter trajectory data by rotor angle limits
and update stability labels based on 180-degree criterion.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .stability_checker import stability_angle_metric_rad


def filter_trajectory_by_angle(
    data: pd.DataFrame,
    max_angle_deg: float = 360.0,
    scenario_id_col: str = "scenario_id",
    delta_col: str = "delta",
    stability_threshold_deg: float = 180.0,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Filter trajectories to keep only points where |delta| < max_angle_deg.

    For unstable trajectories, this keeps the early part (before extreme divergence).
    Also updates stability labels based on 180-degree criterion.

    Parameters:
    -----------
    data : pd.DataFrame
        Trajectory data with columns including delta and scenario_id
    max_angle_deg : float
        Maximum angle in degrees to keep in training data (default: 360.0)
    scenario_id_col : str
        Name of the column containing scenario IDs (default: "scenario_id")
    delta_col : str
        Name of the column containing rotor angles in radians (default: "delta")
    stability_threshold_deg : float
        Stability threshold in degrees (default: 180.0)
        Scenarios with max(|delta|) >= this are marked as unstable

    Returns:
    --------
    filtered_data : pd.DataFrame
        Filtered data with trajectories truncated at max_angle_deg
    statistics : dict
        Dictionary with filtering statistics:
        - original_scenarios: int
        - filtered_scenarios: int
        - removed_scenarios: list
        - original_points: int
        - filtered_points: int
        - truncated_scenarios: list
        - max_angle_before: float
        - max_angle_after: float
    """
    data = data.copy()

    # Convert delta to degrees for filtering
    max_angle_rad = np.deg2rad(max_angle_deg)
    stability_threshold_rad = np.deg2rad(stability_threshold_deg)

    # Statistics tracking
    original_scenarios = data[scenario_id_col].nunique()
    original_points = len(data)
    removed_scenarios = []
    truncated_scenarios = []
    max_angle_before = np.abs(data[delta_col]).max() * (180.0 / np.pi) if len(data) > 0 else 0.0

    # Group by scenario
    filtered_data_list = []

    for scenario_id in data[scenario_id_col].unique():
        scenario_data = data[data[scenario_id_col] == scenario_id].copy()

        # Keep only points where angle < max_angle_deg
        abs_delta = np.abs(scenario_data[delta_col])
        mask = abs_delta < max_angle_rad

        if mask.sum() > 0:
            # Keep this scenario (even if truncated)
            filtered_scenario = scenario_data[mask].copy()

            # Check if trajectory was truncated
            if len(filtered_scenario) < len(scenario_data):
                truncated_scenarios.append(scenario_id)

            # Stability label: preserve original if present (e.g. CCT-based from data generation).
            # Only compute from 180-degree criterion when is_stable was not provided.
            max_angle_in_scenario = np.abs(filtered_scenario[delta_col]).max()
            if "is_stable" in filtered_scenario.columns:
                # Keep original label (first row is constant per scenario)
                is_stable = filtered_scenario["is_stable"].iloc[0]
            else:
                is_stable = max_angle_in_scenario < stability_threshold_rad
                filtered_scenario["is_stable"] = is_stable

            # Add metadata about truncation
            filtered_scenario["truncated_at_360"] = len(filtered_scenario) < len(scenario_data)
            filtered_scenario["max_angle_deg"] = max_angle_in_scenario * (180.0 / np.pi)

            filtered_data_list.append(filtered_scenario)
        else:
            # No valid points - remove scenario entirely
            removed_scenarios.append(scenario_id)

    if len(filtered_data_list) > 0:
        filtered_df = pd.concat(filtered_data_list, ignore_index=True)
    else:
        # Return empty dataframe with same structure
        filtered_df = pd.DataFrame(columns=data.columns)

    # Calculate statistics
    filtered_scenarios = filtered_df[scenario_id_col].nunique() if len(filtered_df) > 0 else 0
    filtered_points = len(filtered_df)
    max_angle_after = (
        np.abs(filtered_df[delta_col]).max() * (180.0 / np.pi) if len(filtered_df) > 0 else 0.0
    )

    statistics = {
        "original_scenarios": original_scenarios,
        "filtered_scenarios": filtered_scenarios,
        "removed_scenarios": removed_scenarios,
        "original_points": original_points,
        "filtered_points": filtered_points,
        "truncated_scenarios": truncated_scenarios,
        "max_angle_before": max_angle_before,
        "max_angle_after": max_angle_after,
        "points_removed": original_points - filtered_points,
        "scenarios_removed": len(removed_scenarios),
        "scenarios_truncated": len(truncated_scenarios),
    }

    return filtered_df, statistics


def determine_stability_180deg(
    delta_trajectory: np.ndarray,
    threshold_deg: float = 180.0,
    *,
    stability_mode: str = "global_max",
    time: Optional[np.ndarray] = None,
    final_window_seconds: float = 0.25,
    persistence_window_seconds: float = 0.25,
    persistence_violation_fraction: float = 0.9,
) -> Tuple[bool, float]:
    """
    Determine stability using the same angle metrics as :func:`utils.stability_checker.check_stability`.

    Parameters:
    -----------
    delta_trajectory : np.ndarray
        Rotor angle trajectory (in radians)
    threshold_deg : float
        Stability threshold in degrees (default: 180.0)
    stability_mode : str
        ``global_max`` | ``terminal`` | ``final_window`` | ``persistence_fraction``.
    time : np.ndarray, optional
        Time (s), same length as ``delta_trajectory``; needed for ``final_window``
        when ``final_window_seconds`` > 0, and for ``persistence_fraction``.
    final_window_seconds : float
        Trailing window (s) for ``final_window`` mode.
    persistence_window_seconds : float
        Sliding window (s) for ``persistence_fraction``.
    persistence_violation_fraction : float
        Fraction of samples in a window with |δ| ≥ threshold for persistence.

    Returns:
    --------
    is_stable : bool
        True if stable, False if unstable
    angle_metric_deg : float
        The angle metric expressed in degrees for reporting; for ``global_max`` this
        is max |δ| over samples. For ``persistence_fraction`` this is ``0.0`` or
        ``inf`` (converted from the internal rad metric).
    """
    threshold_rad = float(np.deg2rad(threshold_deg))
    metric_rad = float(
        stability_angle_metric_rad(
            delta_trajectory,
            mode=stability_mode,
            time=time,
            final_window_seconds=final_window_seconds,
            persistence_window_seconds=persistence_window_seconds,
            persistence_violation_fraction=persistence_violation_fraction,
            violation_threshold_rad=threshold_rad,
        )
    )
    angle_metric_deg = metric_rad * (180.0 / np.pi)
    is_stable = metric_rad < threshold_rad
    return is_stable, float(angle_metric_deg)


def update_stability_labels(
    data: pd.DataFrame,
    threshold_deg: float = 180.0,
    scenario_id_col: str = "scenario_id",
    delta_col: str = "delta",
    *,
    stability_mode: str = "global_max",
    time_col: str = "time",
    final_window_seconds: float = 0.25,
    persistence_window_seconds: float = 0.25,
    persistence_violation_fraction: float = 0.9,
) -> pd.DataFrame:
    """
    Update stability labels based on 180-degree criterion.

    Parameters:
    -----------
    data : pd.DataFrame
        Trajectory data
    threshold_deg : float
        Stability threshold in degrees (default: 180.0)
    scenario_id_col : str
        Name of the column containing scenario IDs
    delta_col : str
        Name of the column containing rotor angles in radians
    stability_mode : str
        Passed to :func:`determine_stability_180deg`.
    time_col : str
        Column name for time (s); if missing, ``time`` is not passed (use
        ``global_max`` or ``terminal`` only).
    final_window_seconds : float
        Trailing window for ``final_window`` mode.
    persistence_window_seconds : float
        Sliding window for ``persistence_fraction``.
    persistence_violation_fraction : float
        Violation fraction for ``persistence_fraction``.

    Returns:
    --------
    data : pd.DataFrame
        Data with updated is_stable column
    """
    data = data.copy()

    # Group by scenario and compute angle metric
    stability_labels = []
    max_angles = []

    for scenario_id in data[scenario_id_col].unique():
        scenario_data = data[data[scenario_id_col] == scenario_id]
        t_arr = (
            scenario_data[time_col].values.astype(np.float64)
            if time_col in scenario_data.columns
            else None
        )
        is_stable, angle_metric_deg = determine_stability_180deg(
            scenario_data[delta_col].values,
            threshold_deg=threshold_deg,
            stability_mode=stability_mode,
            time=t_arr,
            final_window_seconds=final_window_seconds,
            persistence_window_seconds=persistence_window_seconds,
            persistence_violation_fraction=persistence_violation_fraction,
        )
        max_angles.append(angle_metric_deg)
        stability_labels.append(is_stable)

    # Create mapping
    stability_map = dict(zip(data[scenario_id_col].unique(), stability_labels))
    max_angle_map = dict(zip(data[scenario_id_col].unique(), max_angles))

    # Update data
    data["is_stable"] = data[scenario_id_col].map(stability_map)
    data["max_angle_deg"] = data[scenario_id_col].map(max_angle_map)

    return data
