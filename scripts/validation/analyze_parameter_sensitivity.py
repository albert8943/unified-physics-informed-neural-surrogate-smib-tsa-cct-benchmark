#!/usr/bin/env python
"""
Parameter Sensitivity Analysis for GENROU Validation.

Analyzes how performance varies with system parameters (H, D, Pm, clearing time).

Usage:
    python scripts/validation/analyze_parameter_sensitivity.py \
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


def analyze_parameter_ranges(df: pd.DataFrame, param: str, ranges: List[tuple]) -> Dict:
    """
    Analyze performance across parameter ranges.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with results
    param : str
        Parameter name (e.g., 'H', 'D', 'Pm')
    ranges : list
        List of (min, max) tuples for ranges

    Returns:
    --------
    analysis : dict
        Analysis results per range
    """
    analysis = {}

    for i, (min_val, max_val) in enumerate(ranges):
        if min_val == -np.inf:
            subset = df[df[param] < max_val]
            range_name = f"{param} < {max_val:.2f}"
        elif max_val == np.inf:
            subset = df[df[param] >= min_val]
            range_name = f"{param} >= {min_val:.2f}"
        else:
            subset = df[(df[param] >= min_val) & (df[param] < max_val)]
            range_name = f"{min_val:.2f} <= {param} < {max_val:.2f}"

        if len(subset) == 0:
            continue

        analysis[range_name] = {
            "n": len(subset),
            "delta_r2_mean": float(subset["delta_r2"].mean()),
            "delta_r2_std": float(subset["delta_r2"].std()),
            "omega_r2_mean": float(subset["omega_r2"].mean()),
            "omega_r2_std": float(subset["omega_r2"].std()),
            "delta_rmse_mean": float(subset["delta_rmse"].mean()),
            "delta_rmse_std": float(subset["delta_rmse"].std()),
            "omega_rmse_mean": float(subset["omega_rmse"].mean()),
            "omega_rmse_std": float(subset["omega_rmse"].std()),
        }

    return analysis


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze parameter sensitivity")
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
    print("GENROU VALIDATION - PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r") as f:
        results = json.load(f)

    print(f"✓ Loaded {len(results)} validation results")

    # Convert to DataFrame
    data = []
    for r in results:
        scenario = r.get("scenario", {})
        data.append(
            {
                "scenario_id": scenario.get("scenario_id", ""),
                "H": scenario.get("H", np.nan),
                "D": scenario.get("D", np.nan),
                "Pm": scenario.get("Pm", np.nan),
                "delta0": scenario.get("delta0", np.nan),
                "omega0": scenario.get("omega0", np.nan),
                "tf": scenario.get("tf", np.nan),
                "tc": scenario.get("tc", np.nan),
                "delta_r2": r.get("delta_r2", np.nan),
                "omega_r2": r.get("omega_r2", np.nan),
                "delta_rmse": r.get("delta_rmse", np.nan),
                "omega_rmse": r.get("omega_rmse", np.nan),
                "delta_mae": r.get("delta_mae", np.nan),
                "omega_mae": r.get("omega_mae", np.nan),
            }
        )

    df = pd.DataFrame(data)

    # Compute clearing time duration
    df["clearing_time"] = df["tc"] - df["tf"]

    # Define parameter ranges
    h_ranges = [(-np.inf, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, np.inf)]
    d_ranges = [(-np.inf, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, np.inf)]
    pm_ranges = [(-np.inf, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, np.inf)]
    tc_ranges = [(-np.inf, 1.1), (1.1, 1.2), (1.2, 1.3), (1.3, 1.4), (1.4, np.inf)]

    # Analyze each parameter
    h_analysis = analyze_parameter_ranges(df, "H", h_ranges)
    d_analysis = analyze_parameter_ranges(df, "D", d_ranges)
    pm_analysis = analyze_parameter_ranges(df, "Pm", pm_ranges)
    tc_analysis = analyze_parameter_ranges(df, "clearing_time", tc_ranges)

    # Create comprehensive analysis
    sensitivity_analysis = {
        "H_inertia": h_analysis,
        "D_damping": d_analysis,
        "Pm_mechanical_power": pm_analysis,
        "tc_clearing_time": tc_analysis,
    }

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_file = output_dir / "parameter_sensitivity.json"
    with open(analysis_file, "w") as f:
        json.dump(sensitivity_analysis, f, indent=2, default=str)
    print(f"✓ Analysis saved to: {analysis_file}")

    # Create summary tables
    tables = {}

    # H (Inertia) table
    h_data = {
        "H Range (s)": list(h_analysis.keys()),
        "N": [h_analysis[k]["n"] for k in h_analysis.keys()],
        "Mean R² Delta": [
            f"{h_analysis[k]['delta_r2_mean']:.4f} ± {h_analysis[k]['delta_r2_std']:.4f}"
            for k in h_analysis.keys()
        ],
        "Mean RMSE Delta (rad)": [
            f"{h_analysis[k]['delta_rmse_mean']:.4f} ± {h_analysis[k]['delta_rmse_std']:.4f}"
            for k in h_analysis.keys()
        ],
    }
    tables["H_inertia"] = pd.DataFrame(h_data)

    # D (Damping) table
    d_data = {
        "D Range (pu)": list(d_analysis.keys()),
        "N": [d_analysis[k]["n"] for k in d_analysis.keys()],
        "Mean R² Delta": [
            f"{d_analysis[k]['delta_r2_mean']:.4f} ± {d_analysis[k]['delta_r2_std']:.4f}"
            for k in d_analysis.keys()
        ],
        "Mean RMSE Delta (rad)": [
            f"{d_analysis[k]['delta_rmse_mean']:.4f} ± {d_analysis[k]['delta_rmse_std']:.4f}"
            for k in d_analysis.keys()
        ],
    }
    tables["D_damping"] = pd.DataFrame(d_data)

    # Pm (Mechanical Power) table
    pm_data = {
        "Pm Range (pu)": list(pm_analysis.keys()),
        "N": [pm_analysis[k]["n"] for k in pm_analysis.keys()],
        "Mean R² Delta": [
            f"{pm_analysis[k]['delta_r2_mean']:.4f} ± {pm_analysis[k]['delta_r2_std']:.4f}"
            for k in pm_analysis.keys()
        ],
        "Mean RMSE Delta (rad)": [
            f"{pm_analysis[k]['delta_rmse_mean']:.4f} ± {pm_analysis[k]['delta_rmse_std']:.4f}"
            for k in pm_analysis.keys()
        ],
    }
    tables["Pm_mechanical_power"] = pd.DataFrame(pm_data)

    # Save tables
    for name, table in tables.items():
        table_file = output_dir / f"{name}_sensitivity.csv"
        table.to_csv(table_file, index=False)
        print(f"✓ {name} table saved to: {table_file}")

    # Print summaries
    print("\n" + "=" * 70)
    print("PARAMETER SENSITIVITY SUMMARY")
    print("=" * 70)

    print("\nH (Inertia Constant):")
    print(tables["H_inertia"].to_string(index=False))

    print("\nD (Damping Coefficient):")
    print(tables["D_damping"].to_string(index=False))

    print("\nPm (Mechanical Power):")
    print(tables["Pm_mechanical_power"].to_string(index=False))

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
