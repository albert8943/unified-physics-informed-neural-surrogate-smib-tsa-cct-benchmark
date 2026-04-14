"""
Failure Case Analysis.

Identifies where/why PINN fails and analyzes failure patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def identify_failure_cases(
    predictions: Dict,
    targets: Dict,
    error_threshold: float = 0.5,
    top_n: int = 10,
) -> List[Dict]:
    """
    Identify failure cases (high error predictions).

    Parameters:
    -----------
    predictions : dict
        Dictionary with 'delta_pred' and 'omega_pred' arrays
    targets : dict
        Dictionary with 'delta_true' and 'omega_true' arrays
    error_threshold : float
        Error threshold for failure (RMSE)
    top_n : int
        Number of top failure cases to return

    Returns:
    --------
    failure_cases : list
        List of failure case dictionaries
    """
    delta_pred = predictions.get("delta_pred", np.array([]))
    omega_pred = predictions.get("omega_pred", np.array([]))
    delta_true = targets.get("delta_true", np.array([]))
    omega_true = targets.get("omega_true", np.array([]))

    # Compute errors
    delta_errors = np.abs(delta_pred - delta_true)
    omega_errors = np.abs(omega_pred - omega_true)
    combined_errors = np.sqrt(delta_errors**2 + omega_errors**2)

    # Find top N failures
    top_indices = np.argsort(combined_errors)[-top_n:][::-1]

    failure_cases = []
    for idx in top_indices:
        failure_cases.append(
            {
                "index": int(idx),
                "delta_error": float(delta_errors[idx]),
                "omega_error": float(omega_errors[idx]),
                "combined_error": float(combined_errors[idx]),
                "delta_pred": float(delta_pred[idx]),
                "omega_pred": float(omega_pred[idx]),
                "delta_true": float(delta_true[idx]),
                "omega_true": float(omega_true[idx]),
            }
        )

    return failure_cases


def analyze_failure_patterns(
    failure_cases: List[Dict],
    scenario_data: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Analyze patterns in failure cases.

    Parameters:
    -----------
    failure_cases : list
        List of failure case dictionaries
    scenario_data : pd.DataFrame, optional
        Scenario data with parameters

    Returns:
    --------
    analysis : dict
        Failure pattern analysis
    """
    analysis = {
        "n_failures": len(failure_cases),
        "avg_delta_error": np.mean([f["delta_error"] for f in failure_cases]),
        "avg_omega_error": np.mean([f["omega_error"] for f in failure_cases]),
        "max_delta_error": np.max([f["delta_error"] for f in failure_cases]),
        "max_omega_error": np.max([f["omega_error"] for f in failure_cases]),
    }

    # Analyze parameter ranges if scenario data available
    if scenario_data is not None and len(failure_cases) > 0:
        failure_indices = [f["index"] for f in failure_cases]
        failure_scenarios = scenario_data.iloc[failure_indices]

        analysis["parameter_ranges"] = {
            "H": {
                "min": float(failure_scenarios["H"].min()),
                "max": float(failure_scenarios["H"].max()),
                "mean": float(failure_scenarios["H"].mean()),
            },
            "D": {
                "min": float(failure_scenarios["D"].min()),
                "max": float(failure_scenarios["D"].max()),
                "mean": float(failure_scenarios["D"].mean()),
            },
            "Pm": {
                "min": float(failure_scenarios["Pm"].min()),
                "max": float(failure_scenarios["Pm"].max()),
                "mean": float(failure_scenarios["Pm"].mean()),
            },
        }

    return analysis


def analyze_near_instability_cases(
    predictions: Dict,
    targets: Dict,
    stability_threshold: float = np.pi,
) -> Dict:
    """
    Analyze cases near instability boundary.

    Parameters:
    -----------
    predictions : dict
        Predictions dictionary
    targets : dict
        Targets dictionary
    stability_threshold : float
        Stability threshold (max angle in radians)

    Returns:
    --------
    analysis : dict
        Near-instability analysis
    """
    delta_pred = predictions.get("delta_pred", np.array([]))
    delta_true = targets.get("delta_true", np.array([]))

    # Find near-instability cases (max angle close to threshold)
    max_angle_pred = np.max(np.abs(delta_pred))
    max_angle_true = np.max(np.abs(delta_true))

    near_instability = {
        "max_angle_pred": float(max_angle_pred),
        "max_angle_true": float(max_angle_true),
        "is_stable_pred": max_angle_pred < stability_threshold,
        "is_stable_true": max_angle_true < stability_threshold,
        "stability_match": (max_angle_pred < stability_threshold)
        == (max_angle_true < stability_threshold),
    }

    return near_instability
