#!/usr/bin/env python
"""
Statistical Analysis of GENROU Validation Results.

Computes comprehensive statistics including confidence intervals,
distribution analysis, and outlier identification.

Usage:
    python scripts/validation/analyze_genrou_results.py \
        --results outputs/publication/genrou_validation/exp_YYYYMMDD_HHMMSS/results/genrou_validation_results.json \
        --output-dir outputs/publication/genrou_validation/exp_YYYYMMDD_HHMMSS/analysis
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Fix encoding for Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats


def compute_statistics(values: List[float]) -> Dict:
    """
    Compute comprehensive statistics for a list of values.

    Parameters:
    -----------
    values : list
        List of numeric values

    Returns:
    --------
    stats_dict : dict
        Dictionary of statistics
    """
    values = [v for v in values if not np.isnan(v) and not np.isinf(v)]

    if len(values) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "ci_95": [np.nan, np.nan],
        }

    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample standard deviation
    median = np.median(values)
    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)

    # 95% confidence interval
    if len(values) > 1:
        ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=stats.sem(values))
    else:
        ci = [mean, mean]

    return {
        "n": len(values),
        "mean": float(mean),
        "std": float(std),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "ci_95": [float(ci[0]), float(ci[1])],
    }


def identify_outliers(values: List[float], method: str = "iqr") -> List[int]:
    """
    Identify outliers using IQR method.

    Parameters:
    -----------
    values : list
        List of numeric values
    method : str
        Method for outlier detection ('iqr' or 'zscore')

    Returns:
    --------
    outlier_indices : list
        Indices of outliers
    """
    values = np.array([v for v in values if not np.isnan(v) and not np.isinf(v)])

    if len(values) == 0:
        return []

    if method == "iqr":
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = np.where((values < lower_bound) | (values > upper_bound))[0]
    elif method == "zscore":
        z_scores = np.abs(stats.zscore(values))
        outliers = np.where(z_scores > 3)[0]
    else:
        outliers = []

    return outliers.tolist()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze GENROU validation results")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to GENROU validation results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GENROU VALIDATION - STATISTICAL ANALYSIS")
    print("=" * 70)

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r") as f:
        results = json.load(f)

    print(f"✓ Loaded {len(results)} validation results")

    # Extract metrics
    delta_r2 = [r.get("delta_r2", np.nan) for r in results]
    omega_r2 = [r.get("omega_r2", np.nan) for r in results]
    delta_rmse = [r.get("delta_rmse", np.nan) for r in results]
    omega_rmse = [r.get("omega_rmse", np.nan) for r in results]
    delta_mae = [r.get("delta_mae", np.nan) for r in results]
    omega_mae = [r.get("omega_mae", np.nan) for r in results]

    # Compute statistics
    stats_delta_r2 = compute_statistics(delta_r2)
    stats_omega_r2 = compute_statistics(omega_r2)
    stats_delta_rmse = compute_statistics(delta_rmse)
    stats_omega_rmse = compute_statistics(omega_rmse)
    stats_delta_mae = compute_statistics(delta_mae)
    stats_omega_mae = compute_statistics(omega_mae)

    # Identify outliers
    outliers_delta_r2 = identify_outliers(delta_r2)
    outliers_omega_r2 = identify_outliers(omega_r2)

    # Create comprehensive statistics dictionary
    analysis = {
        "n_scenarios": len(results),
        "delta_r2": stats_delta_r2,
        "omega_r2": stats_omega_r2,
        "delta_rmse": stats_delta_rmse,
        "omega_rmse": stats_omega_rmse,
        "delta_mae": stats_delta_mae,
        "omega_mae": stats_omega_mae,
        "outliers": {
            "delta_r2_indices": outliers_delta_r2,
            "omega_r2_indices": outliers_omega_r2,
        },
    }

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_file = output_dir / "statistical_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"✓ Analysis saved to: {analysis_file}")

    # Create summary table
    summary_data = {
        "Metric": [
            "R² Delta",
            "R² Omega",
            "RMSE Delta (rad)",
            "RMSE Omega (pu)",
            "MAE Delta (rad)",
            "MAE Omega (pu)",
        ],
        "N": [
            stats_delta_r2["n"],
            stats_omega_r2["n"],
            stats_delta_rmse["n"],
            stats_omega_rmse["n"],
            stats_delta_mae["n"],
            stats_omega_mae["n"],
        ],
        "Mean": [
            f"{stats_delta_r2['mean']:.4f}",
            f"{stats_omega_r2['mean']:.4f}",
            f"{stats_delta_rmse['mean']:.4f}",
            f"{stats_omega_rmse['mean']:.4f}",
            f"{stats_delta_mae['mean']:.4f}",
            f"{stats_omega_mae['mean']:.4f}",
        ],
        "Std": [
            f"{stats_delta_r2['std']:.4f}",
            f"{stats_omega_r2['std']:.4f}",
            f"{stats_delta_rmse['std']:.4f}",
            f"{stats_omega_rmse['std']:.4f}",
            f"{stats_delta_mae['std']:.4f}",
            f"{stats_omega_mae['std']:.4f}",
        ],
        "95% CI": [
            f"[{stats_delta_r2['ci_95'][0]:.4f}, {stats_delta_r2['ci_95'][1]:.4f}]",
            f"[{stats_omega_r2['ci_95'][0]:.4f}, {stats_omega_r2['ci_95'][1]:.4f}]",
            f"[{stats_delta_rmse['ci_95'][0]:.4f}, {stats_delta_rmse['ci_95'][1]:.4f}]",
            f"[{stats_omega_rmse['ci_95'][0]:.4f}, {stats_omega_rmse['ci_95'][1]:.4f}]",
            f"[{stats_delta_mae['ci_95'][0]:.4f}, {stats_delta_mae['ci_95'][1]:.4f}]",
            f"[{stats_omega_mae['ci_95'][0]:.4f}, {stats_omega_mae['ci_95'][1]:.4f}]",
        ],
        "Min": [
            f"{stats_delta_r2['min']:.4f}",
            f"{stats_omega_r2['min']:.4f}",
            f"{stats_delta_rmse['min']:.4f}",
            f"{stats_omega_rmse['min']:.4f}",
            f"{stats_delta_mae['min']:.4f}",
            f"{stats_omega_mae['min']:.4f}",
        ],
        "Max": [
            f"{stats_delta_r2['max']:.4f}",
            f"{stats_omega_r2['max']:.4f}",
            f"{stats_delta_rmse['max']:.4f}",
            f"{stats_omega_rmse['max']:.4f}",
            f"{stats_delta_mae['max']:.4f}",
            f"{stats_omega_mae['max']:.4f}",
        ],
    }

    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / "statistical_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Summary table saved to: {summary_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Print outliers
    if outliers_delta_r2 or outliers_omega_r2:
        print("\n" + "=" * 70)
        print("OUTLIERS DETECTED")
        print("=" * 70)
        if outliers_delta_r2:
            print(f"Delta R² outliers (indices): {outliers_delta_r2}")
            print(f"  Values: {[delta_r2[i] for i in outliers_delta_r2]}")
        if outliers_omega_r2:
            print(f"Omega R² outliers (indices): {outliers_omega_r2}")
            print(f"  Values: {[omega_r2[i] for i in outliers_omega_r2]}")

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
