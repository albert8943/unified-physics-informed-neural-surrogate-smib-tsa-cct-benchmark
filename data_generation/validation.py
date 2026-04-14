"""
Data Validation Utilities for PINN Training Data.

This module provides validation utilities for:
- Correlation analysis between parameters
- Coverage analysis of parameter space
- Stability boundary detection for CCT data
- Data quality assessment
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .sampling_strategies import correlation_analysis, validate_sample_quality


def detect_stability_boundary(
    data: pd.DataFrame, delta_threshold: float = np.pi, omega_threshold: float = 1.5
) -> pd.DataFrame:
    """
    Detect stability boundary in simulation data.

    Classifies each simulation as stable or unstable based on trajectory
    characteristics. Unstable cases typically have:
    - Rotor angle exceeding threshold (delta > pi)
    - Rotor speed deviating significantly (omega > threshold)

    Parameters:
    -----------
    data : pd.DataFrame
        Simulation data with columns: 'delta', 'omega', 'time', etc.
    delta_threshold : float
        Threshold for rotor angle to classify as unstable (radians)
    omega_threshold : float
        Threshold for rotor speed deviation to classify as unstable (pu)

    Returns:
    --------
    pd.DataFrame
        Data with added 'is_stable' column (boolean)
    """
    data = data.copy()

    # Group by scenario to analyze each simulation
    if "scenario_id" in data.columns:
        scenario_col = "scenario_id"
    elif "param_H" in data.columns and "param_D" in data.columns and "param_tc" in data.columns:
        # Create scenario ID from parameters
        data["scenario_id"] = (
            data["param_H"].astype(str)
            + "_"
            + data["param_D"].astype(str)
            + "_"
            + data["param_tc"].astype(str)
        )
        scenario_col = "scenario_id"
    else:
        raise ValueError(
            "Data must contain 'scenario_id' or parameter columns (param_H, param_D, param_tc)"
        )

    # Analyze each scenario
    is_stable_list = []
    scenario_ids = data[scenario_col].unique()

    for scenario_id in scenario_ids:
        scenario_data = data[data[scenario_col] == scenario_id].copy()

        # Check for instability indicators (stability from rotor angle only; omega commented out)
        max_delta = scenario_data["delta"].abs().max()
        # max_omega_dev = (scenario_data["omega"] - 1.0).abs().max()

        # Classify as stable if angle threshold is not exceeded
        is_stable = max_delta < delta_threshold
        is_stable_list.append((scenario_id, is_stable))

    # Create mapping
    stability_map = pd.DataFrame(is_stable_list, columns=[scenario_col, "is_stable"])

    # Merge back to original data
    data = data.merge(stability_map, on=scenario_col, how="left")

    return data


def estimate_cct_from_data(
    data: pd.DataFrame, H: float, D: float, fault_bus: Optional[int] = None, tolerance: float = 0.01
) -> Optional[float]:
    """
    Estimate Critical Clearing Time (CCT) from simulation data.

    Finds the maximum fault clearing time that results in a stable system
    by analyzing stability of simulations with different fault clearing times.

    Parameters:
    -----------
    data : pd.DataFrame
        Simulation data with stability labels
    H : float
        Inertia constant (seconds)
    D : float
        Damping coefficient (pu)
    fault_bus : int, optional
        Fault bus index (if multiple fault locations)
    tolerance : float
        Tolerance for CCT estimation (seconds)

    Returns:
    --------
    float or None
        Estimated CCT, or None if cannot be determined
    """
    # Filter data for specific H, D, and optionally fault_bus.
    filtered_data = data[(data["param_H"] == H) & (data["param_D"] == D)]

    if fault_bus is not None:
        filtered_data = filtered_data[filtered_data["param_fault_bus"] == fault_bus]

    if len(filtered_data) == 0:
        return None

    # Get unique fault clearing times and their stability
    if "is_stable" not in filtered_data.columns:
        # Detect stability if not already done
        filtered_data = detect_stability_boundary(filtered_data)

    # Group by fault clearing time
    tc_stability = filtered_data.groupby("param_tc")["is_stable"].first().reset_index()
    tc_stability = tc_stability.sort_values("param_tc")

    # Find transition from stable to unstable
    stable_tcs = tc_stability[tc_stability["is_stable"] is True]["param_tc"].values
    unstable_tcs = tc_stability[tc_stability["is_stable"] is False]["param_tc"].values

    if len(stable_tcs) == 0:
        # All cases unstable
        return None
    if len(unstable_tcs) == 0:
        # All cases stable - CCT is at least the maximum tc
        return tc_stability["param_tc"].max()

    # Find maximum stable tc and minimum unstable tc
    max_stable_tc = stable_tcs.max()
    min_unstable_tc = unstable_tcs.min()

    # CCT is between these values
    # Use linear interpolation if close, otherwise return max_stable_tc
    if min_unstable_tc - max_stable_tc < tolerance:
        # Close enough - return midpoint
        return (max_stable_tc + min_unstable_tc) / 2.0
    else:
        # Return maximum stable clearing time
        return max_stable_tc


def validate_cct_data_quality(
    data: pd.DataFrame, min_stable_ratio: float = 0.2, max_stable_ratio: float = 0.8
) -> Dict[str, any]:
    """
    Validate quality of CCT estimation data.

    Checks:
    - Balance between stable and unstable cases
    - Coverage of fault clearing times
    - Presence of stability boundaries

    Parameters:
    -----------
    data : pd.DataFrame
        Simulation data with stability labels
    min_stable_ratio : float
        Minimum ratio of stable cases (0.0 to 1.0)
    max_stable_ratio : float
        Maximum ratio of stable cases (0.0 to 1.0)

    Returns:
    --------
    dict
        Validation results and quality metrics
    """
    # Detect stability if not already done.
    if "is_stable" not in data.columns:
        data = detect_stability_boundary(data)

    # Calculate stability ratio
    total_cases = len(data.groupby(["param_H", "param_D", "param_tc"]).size())
    stable_cases = data.groupby(["param_H", "param_D", "param_tc"])["is_stable"].first().sum()
    stable_ratio = stable_cases / total_cases if total_cases > 0 else 0.0

    # Check balance
    balance_ok = min_stable_ratio <= stable_ratio <= max_stable_ratio

    # Check coverage of fault clearing times
    unique_tcs = data["param_tc"].nunique()
    tc_range = data["param_tc"].max() - data["param_tc"].min()
    tc_coverage = (
        unique_tcs / (tc_range / 0.01) if tc_range > 0 else 0.0
    )  # Normalize by 0.01s intervals

    # Check for stability boundaries (transitions from stable to unstable)
    boundaries_found = 0
    for H in data["param_H"].unique():
        for D in data["param_D"].unique():
            scenario_data = data[(data["param_H"] == H) & (data["param_D"] == D)]
            if len(scenario_data) > 1:
                stability_sequence = scenario_data.sort_values("param_tc")["is_stable"].values
                # Check for transitions
                transitions = np.sum(np.diff(stability_sequence.astype(int)) != 0)
                if transitions > 0:
                    boundaries_found += 1

    return {
        "stable_ratio": stable_ratio,
        "balance_ok": balance_ok,
        "unique_fault_clearing_times": unique_tcs,
        "fault_clearing_time_coverage": tc_coverage,
        "stability_boundaries_found": boundaries_found,
        "total_scenarios": len(data.groupby(["param_H", "param_D"]).size()),
        "validation_passed": balance_ok and boundaries_found > 0,
    }


def analyze_parameter_coverage(
    data: pd.DataFrame, parameter_columns: List[str] = ["param_H", "param_D"]
) -> Dict[str, any]:
    """
    Analyze coverage of parameter space in data.

    Parameters:
    -----------
    data : pd.DataFrame
        Simulation data
    parameter_columns : list
        List of parameter column names to analyze

    Returns:
    --------
    dict
        Coverage analysis results
    """
    results = {}

    for param_col in parameter_columns:
        if param_col not in data.columns:
            continue

        param_values = data[param_col].unique()
        param_min = data[param_col].min()
        param_max = data[param_col].max()
        param_range = param_max - param_min

        # Calculate coverage in quartiles
        q1 = param_min + 0.25 * param_range
        q2 = param_min + 0.50 * param_range
        q3 = param_min + 0.75 * param_range

        in_q1 = np.sum((data[param_col] >= param_min) & (data[param_col] < q1))
        in_q2 = np.sum((data[param_col] >= q1) & (data[param_col] < q2))
        in_q3 = np.sum((data[param_col] >= q2) & (data[param_col] < q3))
        in_q4 = np.sum((data[param_col] >= q3) & (data[param_col] <= param_max))

        quartile_counts = [in_q1, in_q2, in_q3, in_q4]
        expected_per_quartile = len(data) / 4
        coverage_score = (
            1.0 - np.std(quartile_counts) / expected_per_quartile
            if expected_per_quartile > 0
            else 0.0
        )

        results[param_col] = {
            "unique_values": len(param_values),
            "min": param_min,
            "max": param_max,
            "range": param_range,
            "quartile_counts": quartile_counts,
            "coverage_score": coverage_score,
            "mean": data[param_col].mean(),
            "std": data[param_col].std(),
        }

    return results


def validate_data_for_task(data: pd.DataFrame, task: str, **kwargs) -> Dict[str, any]:
    """
    Validate data quality for a specific task.

    Parameters:
    -----------
    data : pd.DataFrame
        Simulation data
    task : str
        Task type: 'trajectory', 'parameter_estimation', 'cct'
    **kwargs : dict
        Additional validation parameters

    Returns:
    --------
    dict
        Validation results
    """
    if task == "trajectory":
        # For trajectory prediction, check coverage and basic statistics
        coverage = analyze_parameter_coverage(data)
        return {
            "task": task,
            "coverage_analysis": coverage,
            "n_samples": len(data),
            "validation_passed": True,  # Trajectory prediction is more lenient
        }

    elif task == "parameter_estimation":
        # For parameter estimation, check H-D correlation
        if "param_H" in data.columns and "param_D" in data.columns:
            H_D_samples = data[["param_H", "param_D"]].drop_duplicates().values
            corr_info = correlation_analysis(H_D_samples, ["H", "D"])
            max_corr = corr_info["max_correlation"]

            target_max_corr = kwargs.get("max_correlation", 0.3)
            correlation_ok = abs(max_corr) <= target_max_corr

            return {
                "task": task,
                "correlation_analysis": corr_info,
                "max_correlation": max_corr,
                "correlation_ok": correlation_ok,
                "n_samples": len(H_D_samples),
                "validation_passed": correlation_ok,
            }
        else:
            return {
                "task": task,
                "error": "Missing param_H or param_D columns",
                "validation_passed": False,
            }

    elif task == "cct":
        # For CCT estimation, check stability balance and boundaries
        return validate_cct_data_quality(data, **kwargs)

    else:
        raise ValueError(f"Unknown task: {task}")


def generate_validation_report(
    data: pd.DataFrame, task: str, output_file: Optional[str] = None
) -> str:
    """
    Generate a comprehensive validation report for data.

    Parameters:
    -----------
    data : pd.DataFrame
        Simulation data
    task : str
        Task type
    output_file : str, optional
        Path to save report (if provided)

    Returns:
    --------
    str
        Validation report as string
    """
    # Perform validation.
    validation_results = validate_data_for_task(data, task)

    # Generate report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"Data Validation Report for Task: {task}")
    report_lines.append("=" * 60)
    report_lines.append("")

    report_lines.append(f"Total data points: {len(data)}")
    report_lines.append(f"Validation passed: {validation_results.get('validation_passed', False)}")
    report_lines.append("")

    if task == "parameter_estimation":
        report_lines.append("Parameter Estimation Validation:")
        max_correlation = validation_results.get("max_correlation", "N/A")
        if isinstance(max_correlation, (int, float)):
            report_lines.append(f"  Max H-D correlation: {max_correlation:.4f}")
        else:
            report_lines.append(f"  Max H-D correlation: {max_correlation}")
        report_lines.append(f"  Correlation OK: {validation_results.get('correlation_ok', False)}")
        report_lines.append(f"  Unique (H, D) pairs: {validation_results.get('n_samples', 'N/A')}")

    elif task == "cct":
        report_lines.append("CCT Estimation Validation:")
        stable_ratio = validation_results.get("stable_ratio", "N/A")
        if isinstance(stable_ratio, (int, float)):
            report_lines.append(f"  Stable ratio: {stable_ratio:.2%}")
        else:
            report_lines.append(f"  Stable ratio: {stable_ratio}")
        report_lines.append(f"  Balance OK: {validation_results.get('balance_ok', False)}")
        report_lines.append(
            f"  Stability boundaries found: "
            f"{validation_results.get('stability_boundaries_found', 'N/A')}"
        )
        report_lines.append(
            f"  Unique fault clearing times: "
            f"{validation_results.get('unique_fault_clearing_times', 'N/A')}"
        )

    elif task == "trajectory":
        report_lines.append("Trajectory Prediction Validation:")
        coverage = validation_results.get("coverage_analysis", {})
        for param, info in coverage.items():
            report_lines.append(f"  {param}:")
            report_lines.append(f"    Coverage score: {info.get('coverage_score', 0):.4f}")
            report_lines.append(f"    Unique values: {info.get('unique_values', 0)}")
            report_lines.append(f"    Range: [{info.get('min', 0):.2f}, {info.get('max', 0):.2f}]")

    report_lines.append("")
    report_lines.append("=" * 60)

    report = "\n".join(report_lines)

    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            f.write(report)

    return report


def generate_data_quality_report(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report.

    Provides detailed quality metrics including completeness, consistency,
    coverage, balance, correlation, boundary samples, and outlier detection.

    Parameters:
    -----------
    data_list : list
        List of data dictionaries (one per sample)

    Returns:
    --------
    report : dict
        Dictionary with quality metrics including:
        - completeness: Percentage of non-NaN values
        - consistency: Time step consistency score
        - coverage: Parameter space coverage
        - balance: Stability distribution ratio
        - correlation_H_D: Correlation between H and D parameters
        - boundary_samples: Count of samples near CCT
        - outlier_rate: Percentage of outliers detected
    """
    if len(data_list) == 0:
        return {"error": "No data provided"}

    report = {}

    # Completeness: % non-NaN
    total_values = 0
    nan_values = 0
    for data in data_list:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                total_values += value.size
                nan_values += np.sum(np.isnan(value))
            elif isinstance(value, (list, tuple)):
                arr = np.array(value)
                total_values += arr.size
                nan_values += np.sum(np.isnan(arr))

    completeness = 1.0 - (nan_values / total_values) if total_values > 0 else 0.0
    report["completeness"] = completeness

    # Consistency: Time step consistency
    dt_std_list = []
    for data in data_list:
        if "time" in data:
            time = np.array(data["time"])
            if len(time) > 1:
                dt = np.diff(time)
                if len(dt) > 0:
                    dt_mean = np.mean(dt)
                    if dt_mean > 0:
                        dt_std = np.std(dt)
                        dt_std_list.append(dt_std / dt_mean)

    consistency = 1.0 - np.mean(dt_std_list) if len(dt_std_list) > 0 else 1.0
    report["consistency"] = consistency

    # Coverage: Parameter space coverage
    # Extract parameter values (support both H/D and M/D naming)
    H_values = [d.get("H", d.get("M", 0)) for d in data_list if "H" in d or "M" in d]
    D_values = [d.get("D", 0) for d in data_list if "D" in d]
    Pm_values = [d.get("Pm", 0) for d in data_list if "Pm" in d]

    if len(H_values) > 0 and len(D_values) > 0:
        # Calculate unique combinations
        unique_combinations = len(set(zip(H_values, D_values)))
        total_possible = len(set(H_values)) * len(set(D_values))
        coverage = unique_combinations / total_possible if total_possible > 0 else 0.0
        report["coverage"] = coverage
    elif len(Pm_values) > 0 and len(H_values) > 0 and len(D_values) > 0:
        # Fallback to 3D coverage if Pm is available
        unique_combinations = len(set(zip(Pm_values, H_values, D_values)))
        total_possible = len(set(Pm_values)) * len(set(H_values)) * len(set(D_values))
        coverage = unique_combinations / total_possible if total_possible > 0 else 0.0
        report["coverage"] = coverage
    else:
        report["coverage"] = 0.0

    # Balance: Stability distribution
    stable_count = sum(1 for d in data_list if d.get("is_stable", False))
    total_count = len(data_list)
    balance = stable_count / total_count if total_count > 0 else 0.0
    report["balance"] = balance
    report["stable_count"] = stable_count
    report["unstable_count"] = total_count - stable_count

    # Correlation: H-D correlation (or M-D correlation)
    if len(H_values) > 1 and len(D_values) > 1:
        try:
            correlation = np.corrcoef(H_values, D_values)[0, 1]
            report["correlation_H_D"] = correlation
        except Exception:
            report["correlation_H_D"] = 0.0
    else:
        report["correlation_H_D"] = 0.0

    # Boundary samples: Count samples near CCT
    boundary_count = 0
    for data in data_list:
        if "cct" in data and "clearing_time" in data:
            cct = data.get("cct", 0)
            tc = data.get("clearing_time", 0)
            if abs(tc - cct) < 0.01:  # Within 10ms of CCT
                boundary_count += 1

    report["boundary_samples"] = boundary_count

    # Outlier detection (simplified)
    max_angles = [d.get("max_angle", 0) for d in data_list if "max_angle" in d]
    if len(max_angles) > 0:
        max_angles_arr = np.array(max_angles)
        q1 = np.percentile(max_angles_arr, 25)
        q3 = np.percentile(max_angles_arr, 75)
        iqr = q3 - q1
        outliers = np.sum((max_angles_arr < q1 - 1.5 * iqr) | (max_angles_arr > q3 + 1.5 * iqr))
        report["outlier_rate"] = outliers / len(max_angles_arr) if len(max_angles_arr) > 0 else 0.0
    else:
        report["outlier_rate"] = 0.0

    return report


def track_quality_incremental(
    new_sample: Dict[str, Any], quality_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update quality metrics incrementally as samples are generated.

    This function allows tracking data quality metrics during data generation
    without storing all samples in memory.

    Parameters:
    -----------
    new_sample : dict
        New sample data to incorporate
    quality_metrics : dict
        Current quality metrics (will be updated in-place)

    Returns:
    --------
    quality_metrics : dict
        Updated quality metrics dictionary with:
        - total_samples: Total number of samples processed
        - stable_count: Number of stable samples
        - nan_count: Total number of NaN values found
        - total_values: Total number of values processed
        - completeness: Percentage of non-NaN values
        - balance: Ratio of stable to total samples
    """
    # Initialize if needed
    if "total_samples" not in quality_metrics:
        quality_metrics["total_samples"] = 0
        quality_metrics["stable_count"] = 0
        quality_metrics["nan_count"] = 0
        quality_metrics["total_values"] = 0

    quality_metrics["total_samples"] += 1

    # Update stability count
    if new_sample.get("is_stable", False):
        quality_metrics["stable_count"] += 1

    # Update completeness
    for key, value in new_sample.items():
        if isinstance(value, np.ndarray):
            quality_metrics["total_values"] += value.size
            quality_metrics["nan_count"] += np.sum(np.isnan(value))
        elif isinstance(value, (list, tuple)):
            arr = np.array(value)
            quality_metrics["total_values"] += arr.size
            quality_metrics["nan_count"] += np.sum(np.isnan(arr))

    # Recalculate metrics
    if quality_metrics["total_values"] > 0:
        quality_metrics["completeness"] = 1.0 - (
            quality_metrics["nan_count"] / quality_metrics["total_values"]
        )

    if quality_metrics["total_samples"] > 0:
        quality_metrics["balance"] = (
            quality_metrics["stable_count"] / quality_metrics["total_samples"]
        )

    return quality_metrics
