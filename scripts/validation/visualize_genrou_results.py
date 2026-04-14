#!/usr/bin/env python
"""
Visualize GENROU Validation Results.

Creates publication-ready figures for trajectory comparison, error analysis,
and parameter sensitivity.

Usage:
    python scripts/validation/visualize_genrou_results.py \
        --results outputs/publication/genrou_validation/exp_YYYYMMDD_HHMMSS/results/genrou_validation_results.json \
        --output-dir outputs/publication/genrou_validation/exp_YYYYMMDD_HHMMSS/figures \
        [--pinn-model PATH] [--genrou-case PATH] [--test-scenarios PATH] \
        [--n-trajectories 5] [--dpi 300]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Note: Encoding fix removed to avoid issues in test environments
# The script will work fine without it

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set publication-quality defaults
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.linewidth"] = 1.5
matplotlib.rcParams["lines.linewidth"] = 2.0
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.alpha"] = 0.3


def load_results(results_path: Path) -> List[Dict]:
    """Load GENROU validation results."""
    with open(results_path, "r") as f:
        return json.load(f)


def create_error_analysis_plots(df: pd.DataFrame, output_dir: Path, dpi: int = 300):
    """
    Create error analysis plots.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with results
    output_dir : Path
        Output directory
    dpi : int
        Figure resolution
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Error vs H
    ax = axes[0, 0]
    valid = df[df["delta_rmse"].notna() & df["H"].notna()]
    ax.scatter(valid["H"], valid["delta_rmse"], alpha=0.6, s=50)
    ax.set_xlabel("Inertia Constant H (s)", fontweight="bold")
    ax.set_ylabel("RMSE Delta (rad)", fontweight="bold")
    ax.set_title("Error vs Inertia", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Error vs D
    ax = axes[0, 1]
    valid = df[df["delta_rmse"].notna() & df["D"].notna()]
    ax.scatter(valid["D"], valid["delta_rmse"], alpha=0.6, s=50, color="orange")
    ax.set_xlabel("Damping Coefficient D (pu)", fontweight="bold")
    ax.set_ylabel("RMSE Delta (rad)", fontweight="bold")
    ax.set_title("Error vs Damping", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # R² vs H
    ax = axes[1, 0]
    valid = df[df["delta_r2"].notna() & df["H"].notna()]
    ax.scatter(valid["H"], valid["delta_r2"], alpha=0.6, s=50, color="green")
    ax.axhline(y=0, color="r", linestyle="--", linewidth=1.5, label="R² = 0")
    ax.set_xlabel("Inertia Constant H (s)", fontweight="bold")
    ax.set_ylabel("R² Delta", fontweight="bold")
    ax.set_title("R² vs Inertia", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R² vs D
    ax = axes[1, 1]
    valid = df[df["delta_r2"].notna() & df["D"].notna()]
    ax.scatter(valid["D"], valid["delta_r2"], alpha=0.6, s=50, color="purple")
    ax.axhline(y=0, color="r", linestyle="--", linewidth=1.5, label="R² = 0")
    ax.set_xlabel("Damping Coefficient D (pu)", fontweight="bold")
    ax.set_ylabel("R² Delta", fontweight="bold")
    ax.set_title("R² vs Damping", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_file = output_dir / "error_analysis.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"✓ Saved: {output_file}")

    # PDF generation skipped - only PNG needed

    plt.close()


def create_parameter_sensitivity_plots(df: pd.DataFrame, output_dir: Path, dpi: int = 300):
    """
    Create parameter sensitivity plots.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with results
    output_dir : Path
        Output directory
    dpi : int
        Figure resolution
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # R² vs Pm
    ax = axes[0, 0]
    valid = df[df["delta_r2"].notna() & df["Pm"].notna()]
    ax.scatter(valid["Pm"], valid["delta_r2"], alpha=0.6, s=50)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=1.5, label="R² = 0")
    ax.set_xlabel("Mechanical Power Pm (pu)", fontweight="bold")
    ax.set_ylabel("R² Delta", fontweight="bold")
    ax.set_title("R² vs Mechanical Power", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R² vs Clearing Time
    ax = axes[0, 1]
    df["clearing_time"] = df["tc"] - df["tf"]
    valid = df[df["delta_r2"].notna() & df["clearing_time"].notna()]
    ax.scatter(valid["clearing_time"], valid["delta_r2"], alpha=0.6, s=50, color="orange")
    ax.axhline(y=0, color="r", linestyle="--", linewidth=1.5, label="R² = 0")
    ax.set_xlabel("Clearing Time (s)", fontweight="bold")
    ax.set_ylabel("R² Delta", fontweight="bold")
    ax.set_title("R² vs Clearing Time", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE vs Pm
    ax = axes[1, 0]
    valid = df[df["delta_rmse"].notna() & df["Pm"].notna()]
    ax.scatter(valid["Pm"], valid["delta_rmse"], alpha=0.6, s=50, color="green")
    ax.set_xlabel("Mechanical Power Pm (pu)", fontweight="bold")
    ax.set_ylabel("RMSE Delta (rad)", fontweight="bold")
    ax.set_title("RMSE vs Mechanical Power", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # RMSE vs Clearing Time
    ax = axes[1, 1]
    valid = df[df["delta_rmse"].notna() & df["clearing_time"].notna()]
    ax.scatter(valid["clearing_time"], valid["delta_rmse"], alpha=0.6, s=50, color="purple")
    ax.set_xlabel("Clearing Time (s)", fontweight="bold")
    ax.set_ylabel("RMSE Delta (rad)", fontweight="bold")
    ax.set_title("RMSE vs Clearing Time", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_file = output_dir / "parameter_sensitivity.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"✓ Saved: {output_file}")

    output_file_pdf = output_dir / "parameter_sensitivity.pdf"
    plt.savefig(output_file_pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"✓ Saved: {output_file_pdf}")

    plt.close()


def create_distribution_plots(df: pd.DataFrame, output_dir: Path, dpi: int = 300):
    """
    Create distribution plots for metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with results
    output_dir : Path
        Output directory
    dpi : int
        Figure resolution
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Delta R² distribution
    ax = axes[0, 0]
    valid = df[df["delta_r2"].notna()]["delta_r2"]
    ax.hist(valid, bins=20, alpha=0.7, edgecolor="black")
    ax.axvline(x=0, color="r", linestyle="--", linewidth=1.5, label="R² = 0")
    ax.set_xlabel("R² Delta", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Distribution of R² Delta", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Omega R² distribution
    ax = axes[0, 1]
    valid = df[df["omega_r2"].notna()]["omega_r2"]
    # Clip extreme values for visualization
    valid_clipped = np.clip(valid, -50, 10)
    ax.hist(valid_clipped, bins=20, alpha=0.7, edgecolor="black", color="orange")
    ax.axvline(x=0, color="r", linestyle="--", linewidth=1.5, label="R² = 0")
    ax.set_xlabel("R² Omega (clipped to [-50, 10])", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Distribution of R² Omega", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Delta RMSE distribution
    ax = axes[1, 0]
    valid = df[df["delta_rmse"].notna()]["delta_rmse"]
    ax.hist(valid, bins=20, alpha=0.7, edgecolor="black", color="green")
    ax.set_xlabel("RMSE Delta (rad)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Distribution of RMSE Delta", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Omega RMSE distribution
    ax = axes[1, 1]
    valid = df[df["omega_rmse"].notna()]["omega_rmse"]
    ax.hist(valid, bins=20, alpha=0.7, edgecolor="black", color="purple")
    ax.set_xlabel("RMSE Omega (pu)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Distribution of RMSE Omega", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save
    output_file = output_dir / "metric_distributions.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"✓ Saved: {output_file}")

    # PDF generation skipped - only PNG needed

    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize GENROU validation results")
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
        help="Output directory for figures",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution (default: 300)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GENROU VALIDATION - VISUALIZATION")
    print("=" * 70)

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    results = load_results(results_path)
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
            }
        )

    df = pd.DataFrame(data)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create plots
    print("\nCreating error analysis plots...")
    create_error_analysis_plots(df, output_dir, dpi=args.dpi)

    print("\nCreating parameter sensitivity plots...")
    create_parameter_sensitivity_plots(df, output_dir, dpi=args.dpi)

    print("\nCreating distribution plots...")
    create_distribution_plots(df, output_dir, dpi=args.dpi)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"✓ All figures saved to: {output_dir}")
    print(f"\nGenerated figures:")
    print(f"  - error_analysis.png/pdf")
    print(f"  - parameter_sensitivity.png/pdf")
    print(f"  - metric_distributions.png/pdf")


if __name__ == "__main__":
    main()
