"""
Statistical Summary Computation.

Computes mean, standard deviation, and confidence intervals for metrics
across multiple experimental runs.
"""

import numpy as np
from typing import Dict, List, Optional


def compute_statistical_summary(results: List[Dict], confidence_level: float = 0.95) -> Dict:
    """
    Compute statistical summary from multiple experimental runs.

    Parameters:
    -----------
    results : list
        List of result dictionaries, each containing 'metrics' key
    confidence_level : float
        Confidence level for intervals (default: 0.95 for 95% CI)

    Returns:
    --------
    summary : dict
        Dictionary with statistical summaries for each metric
    """
    if not results:
        return {}

    # Extract all metrics
    all_metrics = {}
    for result in results:
        # Extract PINN metrics (backward compatibility: check both "metrics" and "pinn.metrics")
        pinn_metrics = result.get("pinn", {}).get("metrics", {}) or result.get("metrics", {})
        for key, value in pinn_metrics.items():
            pinn_key = f"pinn.{key}" if not key.startswith("pinn.") else key
            if pinn_key not in all_metrics:
                all_metrics[pinn_key] = []
            # Handle nested metrics (e.g., metrics.rmse_delta)
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    full_key = f"{pinn_key}.{subkey}"
                    if full_key not in all_metrics:
                        all_metrics[full_key] = []
                    if isinstance(subvalue, (int, float)):
                        all_metrics[full_key].append(subvalue)
            elif isinstance(value, (int, float)):
                all_metrics[pinn_key].append(value)

        # Extract ML baseline metrics
        ml_baseline_results = result.get("ml_baseline", {})
        if ml_baseline_results:
            for model_type, model_info in ml_baseline_results.items():
                ml_metrics = model_info.get("metrics", {})
                for key, value in ml_metrics.items():
                    ml_key = f"ml_baseline.{model_type}.{key}"
                    if ml_key not in all_metrics:
                        all_metrics[ml_key] = []
                    # Handle nested metrics
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            full_key = f"{ml_key}.{subkey}"
                            if full_key not in all_metrics:
                                all_metrics[full_key] = []
                            if isinstance(subvalue, (int, float)):
                                all_metrics[full_key].append(subvalue)
                    elif isinstance(value, (int, float)):
                        all_metrics[ml_key].append(value)

    # Compute statistics for each metric
    summary = {}
    for metric_name, values in all_metrics.items():
        if len(values) == 0:
            continue

        values_array = np.array(values)

        # Filter out NaN and inf values before computing statistics
        valid_mask = np.isfinite(values_array)
        if not np.any(valid_mask):
            # All values are NaN/inf, skip this metric
            continue

        # Use only valid values
        values_array = values_array[valid_mask]
        n = len(values_array)

        # Basic statistics
        mean = np.mean(values_array)
        std = np.std(values_array, ddof=1) if n > 1 else 0.0  # Fix: std=0.0 when n=1 (not NaN)
        median = np.median(values_array)
        min_val = np.min(values_array)
        max_val = np.max(values_array)

        # Confidence interval using t-distribution
        # Handle edge case: n=1 (cannot compute CI with 0 degrees of freedom)
        if n == 1:
            # With only one sample, use point estimate (no CI)
            ci_lower = float(mean)
            ci_upper = float(mean)
            std = 0.0  # Fix: std should be 0.0 when n=1, not NaN
        else:
            try:
                from scipy import stats

                t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n - 1)
            except (ImportError, AttributeError):
                # Fallback to normal distribution if scipy not available
                # Use z-score instead of t-distribution
                # For 95% CI, z = 1.96; for 99% CI, z = 2.576
                if confidence_level == 0.95:
                    t_critical = 1.96
                elif confidence_level == 0.99:
                    t_critical = 2.576
                else:
                    # Approximate using normal distribution z-score
                    # For other confidence levels, use standard normal approximation
                    # z = norm.ppf((1 + confidence_level) / 2)
                    # Common values: 90% = 1.645, 95% = 1.96, 99% = 2.576
                    if confidence_level == 0.90:
                        t_critical = 1.645
                    else:
                        t_critical = 1.96  # Default to 95% CI z-score
            se = std / np.sqrt(n)  # Standard error
            ci_lower = mean - t_critical * se
            ci_upper = mean + t_critical * se

        # Ensure no NaN or inf in final values
        summary[metric_name] = {
            "mean": float(mean) if np.isfinite(mean) else 0.0,
            "std": float(std) if np.isfinite(std) else 0.0,
            "median": float(median) if np.isfinite(median) else 0.0,
            "min": float(min_val) if np.isfinite(min_val) else 0.0,
            "max": float(max_val) if np.isfinite(max_val) else 0.0,
            "n": n,
            "ci_lower": float(ci_lower) if np.isfinite(ci_lower) else float(mean),
            "ci_upper": float(ci_upper) if np.isfinite(ci_upper) else float(mean),
            "confidence_level": confidence_level,
        }

    # Also compute summary for common metrics if they exist
    common_metrics = {
        "delta_r2": ["pinn.delta_r2", "delta_r2", "metrics.r2_delta", "r2_delta"],
        "omega_r2": ["pinn.omega_r2", "omega_r2", "metrics.r2_omega", "r2_omega"],
        "delta_rmse": ["pinn.delta_rmse", "delta_rmse", "metrics.rmse_delta", "rmse_delta"],
        "omega_rmse": ["pinn.omega_rmse", "omega_rmse", "metrics.rmse_omega", "rmse_omega"],
        "delta_mae": ["pinn.delta_mae", "delta_mae", "metrics.mae_delta", "mae_delta"],
        "omega_mae": ["pinn.omega_mae", "omega_mae", "metrics.mae_omega", "mae_omega"],
    }

    # Create aliases for common metrics
    for alias, possible_keys in common_metrics.items():
        for key in possible_keys:
            if key in summary:
                summary[alias] = summary[key]
                break

    # Also create aliases for ML baseline common metrics
    ml_common_metrics = {
        "ml_baseline.delta_r2": ["ml_baseline.standard_nn.delta_r2"],
        "ml_baseline.omega_r2": ["ml_baseline.standard_nn.omega_r2"],
        "ml_baseline.delta_rmse": ["ml_baseline.standard_nn.delta_rmse"],
        "ml_baseline.omega_rmse": ["ml_baseline.standard_nn.omega_rmse"],
        "ml_baseline.delta_mae": ["ml_baseline.standard_nn.delta_mae"],
        "ml_baseline.omega_mae": ["ml_baseline.standard_nn.omega_mae"],
    }

    for alias, possible_keys in ml_common_metrics.items():
        for key in possible_keys:
            if key in summary:
                summary[alias] = summary[key]
                break

    return summary


def format_statistical_summary(summary: Dict) -> str:
    """
    Format statistical summary as a readable string.

    Parameters:
    -----------
    summary : dict
        Statistical summary dictionary

    Returns:
    --------
    formatted : str
        Formatted string representation
    """
    lines = []
    lines.append("=" * 80)
    lines.append("STATISTICAL SUMMARY")
    lines.append("=" * 80)

    for metric_name, stats in summary.items():
        if not isinstance(stats, dict) or "mean" not in stats:
            continue

        mean = stats["mean"]
        std = stats.get("std", 0.0)
        ci_lower = stats.get("ci_lower", mean)
        ci_upper = stats.get("ci_upper", mean)
        n = stats.get("n", 0)

        lines.append(f"\n{metric_name}:")
        lines.append(f"  Mean ± Std: {mean:.6f} ± {std:.6f}")
        lines.append(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
        lines.append(f"  Range: [{stats.get('min', 0):.6f}, {stats.get('max', 0):.6f}]")
        lines.append(f"  N: {n}")

    return "\n".join(lines)


def extract_metric_values(results: List[Dict], metric_name: str) -> Optional[np.ndarray]:
    """
    Extract values for a specific metric from results.

    Parameters:
    -----------
    results : list
        List of result dictionaries
    metric_name : str
        Name of metric to extract

    Returns:
    --------
    values : np.ndarray or None
        Array of metric values, or None if not found
    """
    values = []
    for result in results:
        # Fix: Check both "metrics" and "pinn.metrics" for backward compatibility
        metrics = result.get("pinn", {}).get("metrics", {}) or result.get("metrics", {})
        value = None

        # Try direct key
        if metric_name in metrics:
            value = metrics[metric_name]
        # Try nested key (e.g., metrics.r2_delta)
        elif "." in metric_name:
            parts = metric_name.split(".")
            current = metrics
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    current = None
                    break
            if current is not None:
                value = current
        # Try common variations
        else:
            variations = [
                metric_name,
                f"metrics.{metric_name}",
            ]
            # Handle metric name patterns: delta_r2 -> r2_delta, omega_r2 -> r2_omega, etc.
            if "_" in metric_name:
                parts = metric_name.split("_")
                if len(parts) == 2:
                    # Try reversed: delta_r2 -> r2_delta
                    reversed_name = f"{parts[1]}_{parts[0]}"
                    variations.extend(
                        [
                            reversed_name,
                            f"metrics.{reversed_name}",
                        ]
                    )

            for var in variations:
                if var in metrics:
                    value = metrics[var]
                    break

        if value is not None and isinstance(value, (int, float)):
            values.append(value)

    if len(values) == 0:
        return None

    return np.array(values)
