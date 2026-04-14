#!/usr/bin/env python
"""
Omega Weight Ablation Analysis Script.

Analyzes and compares results from omega weight ablation experiments.
Generates comparison tables and visualizations.

Usage:
    python scripts/analyze_omega_weight_ablation.py --experiments outputs/experiments/exp_*
    python scripts/analyze_omega_weight_ablation.py --experiments exp_20250101_120000 exp_20250101_140000
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.utils import generate_timestamped_filename, load_config


def load_experiment_results(experiment_dir: Path) -> dict:
    """Load results from an experiment directory."""
    results = {}

    # Load config to get omega_weight (from scale_to_norm)
    config_path = experiment_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        scale_to_norm = config.get("loss", {}).get("scale_to_norm", [1.0, 100.0])
        if isinstance(scale_to_norm, list) and len(scale_to_norm) >= 2:
            results["omega_weight"] = scale_to_norm[1]  # Omega weight is second element
        else:
            results["omega_weight"] = 100.0  # Default
        results["n_samples"] = config.get("data", {}).get("generation", {}).get("n_samples", None)
        results["epochs"] = config.get("training", {}).get("epochs", None)

    # Load metrics - check multiple locations
    metrics_data = None

    # First try: results/metrics_*.json
    metrics_dir = experiment_dir / "results"
    if metrics_dir.exists():
        metrics_files = list(metrics_dir.glob("metrics_*.json"))
        if metrics_files:
            metrics_path = max(metrics_files, key=lambda p: p.stat().st_mtime)
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)

    # Second try: experiment_summary.json (new format)
    if metrics_data is None:
        summary_path = experiment_dir / "experiment_summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
            if "results" in summary_data and "metrics" in summary_data["results"]:
                metrics_data = summary_data["results"]["metrics"]
            elif "metrics" in summary_data:
                metrics_data = summary_data["metrics"]

    if metrics_data:
        if "metrics" in metrics_data:
            results["metrics"] = metrics_data["metrics"]
        else:
            results["metrics"] = metrics_data

    # Load training history - check multiple locations
    history_data = None

    # First try: results/training_history_*.json
    if metrics_dir.exists():
        history_files = list(metrics_dir.glob("training_history_*.json"))
        if history_files:
            history_path = max(history_files, key=lambda p: p.stat().st_mtime)
            with open(history_path, "r") as f:
                history_data = json.load(f)

    # Second try: model/training_history_*.json
    if history_data is None:
        model_dir = experiment_dir / "model"
        if model_dir.exists():
            history_files = list(model_dir.glob("training_history_*.json"))
            if history_files:
                history_path = max(history_files, key=lambda p: p.stat().st_mtime)
                with open(history_path, "r") as f:
                    history_data = json.load(f)

    # Third try: experiment_summary.json
    if history_data is None:
        summary_path = experiment_dir / "experiment_summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
            if "training_history" in summary_data:
                history_data = summary_data["training_history"]

    if history_data:
        results["training_history"] = history_data

    results["experiment_id"] = experiment_dir.name
    return results


def compare_omega_weights(experiment_dirs: list, output_dir: Path = None):
    """Compare multiple omega weight experiments."""
    print("=" * 70)
    print("OMEGA WEIGHT ABLATION ANALYSIS")
    print("=" * 70)

    if len(experiment_dirs) < 2:
        print("[ERROR] Need at least 2 experiments to compare")
        return

    print(f"Comparing {len(experiment_dirs)} experiments...")

    # Load results from each experiment
    all_results = []
    for exp_dir in experiment_dirs:
        exp_dir = Path(exp_dir)
        if not exp_dir.exists():
            print(f"[WARNING] Experiment directory not found: {exp_dir}")
            continue

        results = load_experiment_results(exp_dir)
        all_results.append(results)

    if len(all_results) < 2:
        print("[ERROR] Not enough valid experiments to compare")
        return

    # Create comparison table
    comparison_data = []

    for results in all_results:
        exp_id = results.get("experiment_id", "unknown")
        metrics = results.get("metrics", {})
        training_history = results.get("training_history", {})

        row = {
            "experiment_id": exp_id,
            "omega_weight": results.get("omega_weight", None),
            "n_samples": results.get("n_samples", None),
            "epochs": results.get("epochs", None),
        }

        # Add metrics
        if metrics:
            if "metrics" in metrics and isinstance(metrics["metrics"], dict):
                actual_metrics = metrics["metrics"]
            else:
                actual_metrics = metrics

            row.update(
                {
                    "rmse_delta": actual_metrics.get("rmse_delta", None),
                    "rmse_omega": actual_metrics.get("rmse_omega", None),
                    "mae_delta": actual_metrics.get("mae_delta", None),
                    "mae_omega": actual_metrics.get("mae_omega", None),
                    "r2_delta": actual_metrics.get("r2_delta", None),
                    "r2_omega": actual_metrics.get("r2_omega", None),
                }
            )

        # Add training metrics
        if training_history:
            val_losses = training_history.get("val_losses", [])
            train_losses = training_history.get("train_losses", [])
            if val_losses:
                row["best_val_loss"] = min(val_losses)
                row["final_val_loss"] = val_losses[-1]
            if train_losses:
                row["final_train_loss"] = train_losses[-1]
            if val_losses and train_losses:
                row["train_val_gap"] = val_losses[-1] - train_losses[-1]

        comparison_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))

    # Save CSV
    if output_dir is None:
        output_dir = Path("outputs/ablation_studies")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = generate_timestamped_filename("omega_weight_ablation_comparison", "csv")
    csv_path = output_dir / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Comparison table saved: {csv_path}")

    # Find best configuration
    df_valid = df[
        (df["rmse_delta"].notna())
        & (df["rmse_delta"] > 0.0)
        & (df["r2_omega"].notna())
        & (df["epochs"] >= 50)  # At least 50 epochs
    ].copy()

    if len(df_valid) > 0:
        # Best is lowest RMSE Delta while maintaining good R² Omega
        # Prioritize experiments with R² Omega > 0.5
        df_valid["score"] = (
            -df_valid["rmse_delta"] * 0.7 - df_valid["rmse_omega"] * 0.3
        )  # Weighted score
        best_row = df_valid.loc[df_valid["score"].idxmax()]

        print("\n" + "=" * 70)
        print("BEST CONFIGURATION")
        print("=" * 70)
        print(f"Omega Weight: {best_row['omega_weight']}")
        print(f"RMSE Delta: {best_row['rmse_delta']:.6f} rad")
        print(f"RMSE Omega: {best_row['rmse_omega']:.6f} pu")
        print(f"R² Delta: {best_row['r2_delta']:.4f}")
        print(f"R² Omega: {best_row['r2_omega']:.4f}")
        print(f"Best Val Loss: {best_row.get('best_val_loss', 'N/A')}")
        print(f"n_samples: {best_row['n_samples']}")
        print(f"epochs: {best_row['epochs']}")
        print(f"Experiment: {best_row['experiment_id']}")
        print("=" * 70)

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_omega_comparison(df, all_results, output_dir)

    return df


def plot_omega_comparison(df: pd.DataFrame, results_list: list, output_dir: Path):
    """Generate comparison plots for omega weight ablation study."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to only show key ablation experiments
    # Only n_samples=30, epochs=300, valid RMSE
    df_clean = df[
        (df["n_samples"] == 30.0)
        & (df["epochs"] == 300)
        & (df["rmse_delta"].notna())
        & (df["rmse_delta"] > 0.0)
    ].copy()

    if len(df_clean) == 0:
        print("[WARNING] No complete data for plotting")
        return

    # Sort by omega_weight for consistent ordering
    df_clean = df_clean.sort_values("omega_weight").reset_index(drop=True)

    # Figure 1: Metrics comparison (2x3 grid - 6 subplots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: Delta metrics
    # RMSE Delta
    ax = axes[0, 0]
    df_plot = df_clean[df_clean["rmse_delta"].notna() & np.isfinite(df_clean["rmse_delta"])]
    x_pos = np.arange(len(df_plot))
    bars = ax.bar(x_pos, df_plot["rmse_delta"], alpha=0.7, color="steelblue")
    ax.set_xlabel("Omega Weight", fontweight="bold", fontsize=11)
    ax.set_ylabel("RMSE Delta (rad)", fontweight="bold", fontsize=11)
    ax.set_title("RMSE Delta vs Omega Weight", fontweight="bold", fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"ω={w:.0f}" for w in df_plot["omega_weight"]], rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    for i, (bar, val) in enumerate(zip(bars, df_plot["rmse_delta"])):
        if pd.notna(val) and np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # R² Delta
    ax = axes[0, 1]
    df_plot_r2 = df_clean[df_clean["r2_delta"].notna() & np.isfinite(df_clean["r2_delta"])]
    x_pos_r2 = np.arange(len(df_plot_r2))
    bars = ax.bar(x_pos_r2, df_plot_r2["r2_delta"], alpha=0.7, color="forestgreen")
    ax.set_xlabel("Omega Weight", fontweight="bold", fontsize=11)
    ax.set_ylabel("R² Delta", fontweight="bold", fontsize=11)
    ax.set_title("R² Delta vs Omega Weight", fontweight="bold", fontsize=12)
    ax.set_xticks(x_pos_r2)
    ax.set_xticklabels([f"ω={w:.0f}" for w in df_plot_r2["omega_weight"]], rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.0])
    for i, (bar, val) in enumerate(zip(bars, df_plot_r2["r2_delta"])):
        if pd.notna(val) and np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # MAE Delta
    ax = axes[0, 2]
    df_plot_mae = df_clean[df_clean["mae_delta"].notna() & np.isfinite(df_clean["mae_delta"])]
    x_pos_mae = np.arange(len(df_plot_mae))
    bars = ax.bar(x_pos_mae, df_plot_mae["mae_delta"], alpha=0.7, color="coral")
    ax.set_xlabel("Omega Weight", fontweight="bold", fontsize=11)
    ax.set_ylabel("MAE Delta (rad)", fontweight="bold", fontsize=11)
    ax.set_title("MAE Delta vs Omega Weight", fontweight="bold", fontsize=12)
    ax.set_xticks(x_pos_mae)
    ax.set_xticklabels([f"ω={w:.0f}" for w in df_plot_mae["omega_weight"]], rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    for i, (bar, val) in enumerate(zip(bars, df_plot_mae["mae_delta"])):
        if pd.notna(val) and np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Bottom row: Omega metrics (the focus of this ablation!)
    # RMSE Omega
    ax = axes[1, 0]
    df_plot_omega_rmse = df_clean[
        df_clean["rmse_omega"].notna() & np.isfinite(df_clean["rmse_omega"])
    ]
    x_pos_omega_rmse = np.arange(len(df_plot_omega_rmse))
    bars = ax.bar(x_pos_omega_rmse, df_plot_omega_rmse["rmse_omega"], alpha=0.7, color="orange")
    ax.set_xlabel("Omega Weight", fontweight="bold", fontsize=11)
    ax.set_ylabel("RMSE Omega (pu)", fontweight="bold", fontsize=11)
    ax.set_title("RMSE Omega vs Omega Weight (Focus)", fontweight="bold", fontsize=12)
    ax.set_xticks(x_pos_omega_rmse)
    ax.set_xticklabels(
        [f"ω={w:.0f}" for w in df_plot_omega_rmse["omega_weight"]], rotation=45, ha="right"
    )
    ax.grid(True, alpha=0.3, axis="y")
    for i, (bar, val) in enumerate(zip(bars, df_plot_omega_rmse["rmse_omega"])):
        if pd.notna(val) and np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0005,
                f"{val:.5f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # R² Omega (MOST IMPORTANT for this ablation!)
    ax = axes[1, 1]
    df_plot_omega_r2 = df_clean[df_clean["r2_omega"].notna() & np.isfinite(df_clean["r2_omega"])]
    x_pos_omega_r2 = np.arange(len(df_plot_omega_r2))
    bars = ax.bar(x_pos_omega_r2, df_plot_omega_r2["r2_omega"], alpha=0.7, color="gold")
    ax.set_xlabel("Omega Weight", fontweight="bold", fontsize=11)
    ax.set_ylabel("R² Omega", fontweight="bold", fontsize=11)
    ax.set_title("R² Omega vs Omega Weight (KEY METRIC)", fontweight="bold", fontsize=12)
    ax.set_xticks(x_pos_omega_r2)
    ax.set_xticklabels(
        [f"ω={w:.0f}" for w in df_plot_omega_r2["omega_weight"]], rotation=45, ha="right"
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.0])
    for i, (bar, val) in enumerate(zip(bars, df_plot_omega_r2["r2_omega"])):
        if pd.notna(val) and np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Train-Val Gap
    ax = axes[1, 2]
    if "train_val_gap" in df_clean.columns:
        df_plot_gap = df_clean[
            df_clean["train_val_gap"].notna() & np.isfinite(df_clean["train_val_gap"])
        ]
        x_pos_gap = np.arange(len(df_plot_gap))
        bars = ax.bar(x_pos_gap, df_plot_gap["train_val_gap"], alpha=0.7, color="purple")
        ax.set_xlabel("Omega Weight", fontweight="bold", fontsize=11)
        ax.set_ylabel("Train-Val Gap", fontweight="bold", fontsize=11)
        ax.set_title("Overfitting Indicator (Train-Val Gap)", fontweight="bold", fontsize=12)
        ax.set_xticks(x_pos_gap)
        ax.set_xticklabels(
            [f"ω={w:.0f}" for w in df_plot_gap["omega_weight"]], rotation=45, ha="right"
        )
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
        for i, (bar, val) in enumerate(zip(bars, df_plot_gap["train_val_gap"])):
            if pd.notna(val) and np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(val) * 0.1,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    else:
        ax.text(0.5, 0.5, "Train-Val Gap\nNot Available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    # Save figure
    filename = generate_timestamped_filename("omega_weight_ablation_comparison", "png")
    fig_path = output_dir / filename
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Comparison plot saved: {fig_path}")
    plt.close()

    # Figure 2: Training curves comparison
    results_with_history = [r for r in results_list if "training_history" in r]
    if len(results_with_history) > 0:
        # Filter to key experiments (n_samples=30, epochs=300)
        key_results = []
        for res in results_with_history:
            exp_id = res.get("experiment_id", "")
            matching_row = df_clean[df_clean["experiment_id"] == exp_id]
            if len(matching_row) > 0:
                key_results.append(res)

        if len(key_results) == 0:
            key_results = results_with_history  # Fallback to all

        print(f"[OK] Plotting training curves for {len(key_results)} experiments")

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Loss curves
        ax = axes[0]
        for res in key_results:
            history = res["training_history"]
            omega_weight = res.get("omega_weight", 100.0)

            val_losses = history.get("val_losses", [])
            train_losses = history.get("train_losses", [])

            if val_losses and train_losses:
                epochs = list(range(1, len(val_losses) + 1))
                if len(epochs) == len(val_losses) and len(epochs) == len(train_losses):
                    ax.plot(
                        epochs,
                        train_losses,
                        label=f"Train (ω={omega_weight:.0f})",
                        alpha=0.7,
                        linestyle="--",
                    )
                    ax.plot(
                        epochs,
                        val_losses,
                        label=f"Val (ω={omega_weight:.0f})",
                        alpha=0.7,
                        linewidth=2,
                    )

        ax.set_xlabel("Epoch", fontweight="bold", fontsize=11)
        ax.set_ylabel("Loss", fontweight="bold", fontsize=11)
        ax.set_title("Training and Validation Loss Curves", fontweight="bold", fontsize=12)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        # Omega-specific: R² Omega vs Epochs (if available in history)
        # Or show loss components
        ax = axes[1]
        for res in key_results:
            history = res["training_history"]
            omega_weight = res.get("omega_weight", 100.0)

            # Try to plot data loss and physics loss separately if available
            data_losses = history.get("train_data_losses", [])
            physics_losses = history.get("train_physics_losses", [])

            if data_losses:
                epochs = list(range(1, len(data_losses) + 1))
                if len(epochs) == len(data_losses):
                    ax.plot(
                        epochs,
                        data_losses,
                        label=f"Data Loss (ω={omega_weight:.0f})",
                        alpha=0.7,
                        linestyle="--",
                    )

            if physics_losses:
                epochs = list(range(1, len(physics_losses) + 1))
                if len(epochs) == len(physics_losses):
                    ax.plot(
                        epochs,
                        physics_losses,
                        label=f"Physics Loss (ω={omega_weight:.0f})",
                        alpha=0.7,
                        linewidth=2,
                    )

        if len(ax.lines) == 0:
            # Fallback: show validation loss evolution
            for res in key_results:
                history = res["training_history"]
                omega_weight = res.get("omega_weight", 100.0)
                val_losses = history.get("val_losses", [])
                if val_losses:
                    epochs = list(range(1, len(val_losses) + 1))
                    if len(epochs) == len(val_losses):
                        ax.plot(
                            epochs,
                            val_losses,
                            label=f"Val Loss (ω={omega_weight:.0f})",
                            alpha=0.7,
                            linewidth=2,
                        )

        ax.set_xlabel("Epoch", fontweight="bold", fontsize=11)
        ax.set_ylabel("Loss", fontweight="bold", fontsize=11)
        ax.set_title("Loss Components Evolution", fontweight="bold", fontsize=12)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()

        # Save figure
        filename = generate_timestamped_filename("omega_weight_ablation_training_curves", "png")
        fig_path = output_dir / filename
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Training curves plot saved: {fig_path}")
        plt.close()

    # Figure 3: Trade-off plot (R² Omega vs RMSE Delta)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    df_plot_tradeoff = df_clean[
        (df_clean["r2_omega"].notna())
        & (df_clean["rmse_delta"].notna())
        & np.isfinite(df_clean["r2_omega"])
        & np.isfinite(df_clean["rmse_delta"])
    ]

    if len(df_plot_tradeoff) > 0:
        scatter = ax.scatter(
            df_plot_tradeoff["rmse_delta"],
            df_plot_tradeoff["r2_omega"],
            s=200,
            alpha=0.6,
            c=df_plot_tradeoff["omega_weight"],
            cmap="viridis",
            edgecolors="black",
            linewidths=1.5,
        )

        # Add labels for each point
        for _, row in df_plot_tradeoff.iterrows():
            ax.annotate(
                f"ω={row['omega_weight']:.0f}",
                (row["rmse_delta"], row["r2_omega"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlabel("RMSE Delta (rad)", fontweight="bold", fontsize=12)
        ax.set_ylabel("R² Omega", fontweight="bold", fontsize=12)
        ax.set_title(
            "Trade-off: R² Omega vs RMSE Delta\n(Higher R² Omega + Lower RMSE Delta = Better)",
            fontweight="bold",
            fontsize=13,
        )
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Omega Weight", fontweight="bold", fontsize=11)

        plt.tight_layout()

        filename = generate_timestamped_filename("omega_weight_ablation_tradeoff", "png")
        fig_path = output_dir / filename
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Trade-off plot saved: {fig_path}")
        plt.close()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze omega weight ablation study results")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        required=True,
        help="Experiment directories (can use glob patterns)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save comparison results"
    )

    args = parser.parse_args()

    # Expand glob patterns
    experiment_dirs = []
    for pattern in args.experiments:
        if "*" in pattern or "?" in pattern:
            matches = glob.glob(pattern)
            experiment_dirs.extend(matches)
        else:
            experiment_dirs.append(pattern)

    # Remove duplicates and sort
    experiment_dirs = sorted(list(set(experiment_dirs)))

    if len(experiment_dirs) == 0:
        print("[ERROR] No experiment directories found")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    compare_omega_weights(experiment_dirs, output_dir)


if __name__ == "__main__":
    main()
