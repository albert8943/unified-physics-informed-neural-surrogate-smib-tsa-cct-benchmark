#!/usr/bin/env python
"""
Generate publication-quality hyperparameter ablation figure from delta weight sweep results.

Usage:
    python scripts/generate_hyperparameter_ablation_figure.py \
        --sweep-csv validation/delta_weight_sweep/sweep_analysis_results.csv \
        --output-dir outputs/publication/paper_figures
"""

import argparse
import sys
from pathlib import Path

# Set matplotlib backend before importing pyplot
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.utils import generate_timestamped_filename

# Set publication-quality plotting style
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("colorblind")
PUBLICATION_DPI = 300
FIGURE_SIZE_DOUBLE = (7.0, 5.0)  # Double column width (inches)


def load_sweep_data(csv_path: Path) -> pd.DataFrame:
    """Load and clean sweep analysis results."""
    df = pd.read_csv(csv_path)

    # Filter out rows with missing critical metrics
    df = df.dropna(subset=["pinn_r2_delta", "pinn_rmse_delta", "delta_weight", "omega_weight"])

    # Sort by delta_weight, then omega_weight for consistent plotting
    df = df.sort_values(["delta_weight", "omega_weight"]).reset_index(drop=True)

    return df


def generate_ablation_figure(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate publication-quality hyperparameter ablation figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Hyperparameter Ablation Study: Impact of Loss Weight Scaling",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Filter to delta=20.0 experiments (the chosen configuration)
    df_delta20 = df[df["delta_weight"] == 20.0].copy()
    df_delta20 = df_delta20.sort_values("omega_weight")

    # 1. R² Delta vs Omega Weight (for delta=20.0)
    ax = axes[0, 0]
    ax.plot(
        df_delta20["omega_weight"],
        df_delta20["pinn_r2_delta"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="steelblue",
        label="R² Delta",
    )
    ax.axvline(
        x=40.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Chosen (ω=40.0)"
    )
    ax.set_xlabel("Omega Weight (ω)", fontweight="bold", fontsize=11)
    ax.set_ylabel("R² Delta", fontweight="bold", fontsize=11)
    ax.set_title("(a) R² Delta vs Omega Weight (Δ=20.0)", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim([0, 110])
    ax.set_ylim([0.85, 0.93])

    # Add value annotations for key points
    for _, row in df_delta20.iterrows():
        if row["omega_weight"] in [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0]:
            ax.annotate(
                f'{row["pinn_r2_delta"]:.3f}',
                (row["omega_weight"], row["pinn_r2_delta"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

    # 2. RMSE Delta vs Omega Weight (for delta=20.0)
    ax = axes[0, 1]
    ax.plot(
        df_delta20["omega_weight"],
        df_delta20["pinn_rmse_delta"],
        marker="s",
        linewidth=2,
        markersize=8,
        color="coral",
        label="RMSE Delta",
    )
    ax.axvline(
        x=40.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Chosen (ω=40.0)"
    )
    ax.set_xlabel("Omega Weight (ω)", fontweight="bold", fontsize=11)
    ax.set_ylabel("RMSE Delta (rad)", fontweight="bold", fontsize=11)
    ax.set_title("(b) RMSE Delta vs Omega Weight (Δ=20.0)", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim([0, 110])
    ax.invert_yaxis()  # Lower is better for RMSE

    # Add value annotations
    for _, row in df_delta20.iterrows():
        if row["omega_weight"] in [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0]:
            ax.annotate(
                f'{row["pinn_rmse_delta"]:.3f}',
                (row["omega_weight"], row["pinn_rmse_delta"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

    # 3. R² Delta vs Delta Weight (grouped by omega weight)
    ax = axes[1, 0]
    omega_values = sorted(df["omega_weight"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(omega_values)))

    for omega, color in zip(omega_values, colors):
        df_subset = df[df["omega_weight"] == omega].sort_values("delta_weight")
        if len(df_subset) > 0:
            label = f"ω={omega:.1f}" if omega < 10 else f"ω={omega:.0f}"
            ax.plot(
                df_subset["delta_weight"],
                df_subset["pinn_r2_delta"],
                marker="o",
                linewidth=1.5,
                markersize=6,
                color=color,
                label=label,
                alpha=0.7,
            )

    ax.axvline(
        x=20.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Chosen (Δ=20.0)"
    )
    ax.set_xlabel("Delta Weight (Δ)", fontweight="bold", fontsize=11)
    ax.set_ylabel("R² Delta", fontweight="bold", fontsize=11)
    ax.set_title("(c) R² Delta vs Delta Weight (Multiple Omega)", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.set_xlim([0, 105])
    ax.set_ylim([0.70, 0.93])

    # 4. Heatmap: R² Delta as function of both weights
    ax = axes[1, 1]

    # Create pivot table for heatmap
    pivot_data = df.pivot_table(
        values="pinn_r2_delta", index="omega_weight", columns="delta_weight", aggfunc="mean"
    )

    # Sort for better visualization
    pivot_data = pivot_data.sort_index(ascending=True)
    pivot_data = pivot_data.sort_index(axis=1, ascending=True)

    im = ax.imshow(pivot_data.values, cmap="viridis", aspect="auto", vmin=0.70, vmax=0.93)

    # Set ticks
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels([f"{int(c)}" for c in pivot_data.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels([f"{int(i)}" for i in pivot_data.index])

    ax.set_xlabel("Delta Weight (Δ)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Omega Weight (ω)", fontweight="bold", fontsize=11)
    ax.set_title("(d) R² Delta Heatmap", fontweight="bold", fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("R² Delta", fontweight="bold", fontsize=10)

    # Mark chosen configuration
    if 20.0 in pivot_data.columns and 40.0 in pivot_data.index:
        delta_idx = list(pivot_data.columns).index(20.0)
        omega_idx = list(pivot_data.index).index(40.0)
        ax.plot(
            delta_idx, omega_idx, "r*", markersize=15, markeredgecolor="white", markeredgewidth=1.5
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    filename = generate_timestamped_filename("hyperparameter_ablation", "png")
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=PUBLICATION_DPI, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate hyperparameter ablation figure from delta weight sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sweep-csv",
        type=str,
        default="validation/delta_weight_sweep/sweep_analysis_results.csv",
        help="Path to sweep analysis results CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/publication/paper_figures",
        help="Output directory for figure",
    )

    args = parser.parse_args()

    # Load data
    csv_path = Path(args.sweep_csv)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        sys.exit(1)

    print(f"📂 Loading sweep data from: {csv_path}")
    df = load_sweep_data(csv_path)
    print(f"   Loaded {len(df)} experiments")

    # Generate figure
    output_dir = Path(args.output_dir)
    output_path = generate_ablation_figure(df, output_dir)

    print(f"\n✅ Hyperparameter ablation figure generated: {output_path}")
    print(f"   Figure shows impact of delta and omega weights on R² Delta and RMSE Delta")


if __name__ == "__main__":
    main()
