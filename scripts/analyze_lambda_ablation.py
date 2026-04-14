#!/usr/bin/env python
"""
Lambda Physics Ablation Analysis Script.

Analyzes and compares results from lambda_physics ablation experiments.
Generates comparison tables and visualizations.

Usage:
    python scripts/analyze_lambda_ablation.py --experiments outputs/experiments/exp_*
    python scripts/analyze_lambda_ablation.py --experiments exp_20250101_120000 exp_20250101_140000 --plot
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

    # Load config to get lambda_physics
    config_path = experiment_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        results["lambda_physics"] = config.get("loss", {}).get("lambda_physics", None)
        results["use_fixed_lambda"] = config.get("loss", {}).get("use_fixed_lambda", False)
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
            if "results" in summary_data and "training_history" in summary_data["results"]:
                history_data = summary_data["results"]["training_history"]

    # Fourth try: glob search
    if history_data is None:
        history_files = list(experiment_dir.glob("**/training_history_*.json"))
        if history_files:
            history_path = max(history_files, key=lambda p: p.stat().st_mtime)
            with open(history_path, "r") as f:
                history_data = json.load(f)

    if history_data:
        results["training_history"] = history_data

    results["experiment_id"] = experiment_dir.name
    results["experiment_dir"] = experiment_dir

    return results


def create_comparison_table(results_list: list) -> pd.DataFrame:
    """Create comparison table from experiment results."""
    rows = []

    for res in results_list:
        metrics = res.get("metrics", {})
        lambda_p = res.get("lambda_physics", "unknown")
        is_fixed = res.get("use_fixed_lambda", False)

        # Determine lambda type
        if is_fixed:
            lambda_type = f"Fixed ({lambda_p})"
        else:
            lambda_type = "Adaptive"

        row = {
            "experiment_id": res["experiment_id"],
            "lambda_type": lambda_type,
            "lambda_physics": lambda_p if is_fixed else "adaptive",
            "n_samples": res.get("n_samples", None),
            "epochs": res.get("epochs", None),
            "rmse_delta": metrics.get("rmse_delta", None),
            "rmse_omega": metrics.get("rmse_omega", None),
            "mae_delta": metrics.get("mae_delta", None),
            "mae_omega": metrics.get("mae_omega", None),
            "r2_delta": metrics.get("r2_delta", None),
            "r2_omega": metrics.get("r2_omega", None),
        }

        # Add training history info if available
        history = res.get("training_history", {})
        if history:
            val_losses = history.get("val_losses", [])
            train_losses = history.get("train_losses", [])
            if val_losses:
                row["best_val_loss"] = min(val_losses)
                row["final_val_loss"] = val_losses[-1]
                row["final_train_loss"] = train_losses[-1] if train_losses else None
                row["train_val_gap"] = (
                    (row["final_val_loss"] - row["final_train_loss"])
                    if row["final_train_loss"]
                    else None
                )

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_lambda_comparison(df: pd.DataFrame, results_list: list, output_dir: Path):
    """Generate comparison plots for lambda ablation study."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to only show key ablation experiments (same as training curves)
    # Only n_samples=30, epochs=300, valid RMSE
    df_clean = df[
        (df["n_samples"] == 30.0)
        & (df["epochs"] == 300)
        & (df["rmse_delta"].notna())
        & (df["rmse_delta"] > 0.0)
    ].copy()

    if len(df_clean) == 0:
        print("⚠️  No complete data for plotting")
        return

    # Group by configuration and select best (lowest RMSE) for each
    # Same logic as training curves filtering
    key_configs = {}
    for _, row in df_clean.iterrows():
        lambda_p = row["lambda_physics"]
        lambda_type = row["lambda_type"]

        # Create configuration key
        if "Fixed" in lambda_type:
            config_key = f"Fixed_{lambda_p}"
        else:
            # For adaptive, check if improved by looking at experiment results
            exp_id = row["experiment_id"]
            res = next((r for r in results_list if r.get("experiment_id") == exp_id), None)
            is_improved = False
            if res:
                history = res.get("training_history", {})
                lambda_evolution = history.get("lambda_physics", [])
                if len(lambda_evolution) > 100:
                    epoch_30_idx = min(30, len(lambda_evolution) - 1)
                    epoch_50_idx = min(50, len(lambda_evolution) - 1)
                    epoch_70_idx = min(70, len(lambda_evolution) - 1)
                    epoch_100_idx = min(100, len(lambda_evolution) - 1)

                    val_30 = (
                        lambda_evolution[epoch_30_idx]
                        if epoch_30_idx < len(lambda_evolution)
                        else 0.0
                    )
                    val_50 = (
                        lambda_evolution[epoch_50_idx]
                        if epoch_50_idx < len(lambda_evolution)
                        else 0.0
                    )
                    val_70 = (
                        lambda_evolution[epoch_70_idx]
                        if epoch_70_idx < len(lambda_evolution)
                        else 0.0
                    )
                    val_100 = (
                        lambda_evolution[epoch_100_idx]
                        if epoch_100_idx < len(lambda_evolution)
                        else 0.0
                    )

                    has_gradual_increase = (
                        val_30 < 0.3
                        and val_50 > val_30
                        and val_50 < 0.8
                        and val_70 > val_50
                        and val_70 < 0.95
                        and val_100 > 0.8
                    )
                    has_sudden_jump = abs(val_50 - val_30) > 0.7

                    if has_gradual_increase and not has_sudden_jump:
                        is_improved = True

            config_key = "Adaptive (Improved)" if is_improved else "Adaptive (Old)"

        # Keep best (lowest RMSE) for each configuration
        rmse = row["rmse_delta"]
        if config_key not in key_configs or rmse < key_configs[config_key]["rmse"]:
            key_configs[config_key] = {"row": row, "rmse": rmse}

    # Create filtered dataframe with only key experiments
    key_rows = [v["row"] for v in key_configs.values()]
    df_clean = pd.DataFrame(key_rows).reset_index(drop=True)

    if len(df_clean) == 0:
        print("⚠️  No key ablation experiments found for plotting")
        return

    # Create display labels for x-axis
    def get_display_label(row):
        lambda_type = row["lambda_type"]
        if "Fixed" in lambda_type:
            lambda_p = row["lambda_physics"]
            return f"Fixed λ={lambda_p}"
        else:
            # Check if improved
            exp_id = row["experiment_id"]
            res = next((r for r in results_list if r.get("experiment_id") == exp_id), None)
            is_improved = False
            if res:
                history = res.get("training_history", {})
                lambda_evolution = history.get("lambda_physics", [])
                if len(lambda_evolution) > 100:
                    epoch_30_idx = min(30, len(lambda_evolution) - 1)
                    epoch_50_idx = min(50, len(lambda_evolution) - 1)
                    epoch_70_idx = min(70, len(lambda_evolution) - 1)
                    epoch_100_idx = min(100, len(lambda_evolution) - 1)

                    val_30 = (
                        lambda_evolution[epoch_30_idx]
                        if epoch_30_idx < len(lambda_evolution)
                        else 0.0
                    )
                    val_50 = (
                        lambda_evolution[epoch_50_idx]
                        if epoch_50_idx < len(lambda_evolution)
                        else 0.0
                    )
                    val_70 = (
                        lambda_evolution[epoch_70_idx]
                        if epoch_70_idx < len(lambda_evolution)
                        else 0.0
                    )
                    val_100 = (
                        lambda_evolution[epoch_100_idx]
                        if epoch_100_idx < len(lambda_evolution)
                        else 0.0
                    )

                    has_gradual_increase = (
                        val_30 < 0.3
                        and val_50 > val_30
                        and val_50 < 0.8
                        and val_70 > val_50
                        and val_70 < 0.95
                        and val_100 > 0.8
                    )
                    has_sudden_jump = abs(val_50 - val_30) > 0.7

                    if has_gradual_increase and not has_sudden_jump:
                        is_improved = True
            return "Adaptive (Improved)" if is_improved else "Adaptive (Old)"

    df_clean["display_label"] = df_clean.apply(get_display_label, axis=1)

    # Sort: Fixed 0.1, 0.5, 1.0, then Adaptive (Old), then Adaptive (Improved)
    def get_sort_key(row):
        label = row["display_label"]
        if "Fixed" in label:
            lambda_val = float(label.split("=")[1])
            return (0, lambda_val)  # Fixed first, sorted by lambda value
        elif "Improved" in label:
            return (2, 0)  # Improved last
        else:
            return (1, 0)  # Old adaptive in middle

    df_clean["sort_key"] = df_clean.apply(get_sort_key, axis=1)
    df_clean = df_clean.sort_values("sort_key").reset_index(drop=True)
    df_clean = df_clean.drop(columns=["sort_key"])

    # Figure 1: RMSE and R² comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RMSE Delta
    ax = axes[0, 0]
    # Filter out NaN/infinite values for plotting
    df_plot = df_clean.copy()
    df_plot = df_plot[df_plot["rmse_delta"].notna() & np.isfinite(df_plot["rmse_delta"])]
    x_pos = np.arange(len(df_plot))
    bars = ax.bar(x_pos, df_plot["rmse_delta"], alpha=0.7, color="steelblue")
    ax.set_xlabel("Lambda Physics Value", fontweight="bold", fontsize=11)
    ax.set_ylabel("RMSE Delta (rad)", fontweight="bold", fontsize=11)
    ax.set_title("RMSE Delta vs Lambda Physics", fontweight="bold", fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_plot["display_label"], rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df_plot["rmse_delta"])):
        if pd.notna(val) and np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
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
    ax.set_xlabel("Lambda Physics Value", fontweight="bold", fontsize=11)
    ax.set_ylabel("R² Delta", fontweight="bold", fontsize=11)
    ax.set_title("R² Delta vs Lambda Physics", fontweight="bold", fontsize=12)
    ax.set_xticks(x_pos_r2)
    ax.set_xticklabels(df_plot_r2["lambda_type"], rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1])
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df_plot_r2["r2_delta"])):
        if pd.notna(val) and np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Validation Loss
    if "best_val_loss" in df_clean.columns:
        ax = axes[1, 0]
        df_plot_val = df_clean[
            df_clean["best_val_loss"].notna() & np.isfinite(df_clean["best_val_loss"])
        ]
        x_pos_val = np.arange(len(df_plot_val))
        bars = ax.bar(x_pos_val, df_plot_val["best_val_loss"], alpha=0.7, color="coral")
        ax.set_xlabel("Lambda Physics Value", fontweight="bold", fontsize=11)
        ax.set_ylabel("Best Validation Loss", fontweight="bold", fontsize=11)
        ax.set_title("Best Validation Loss vs Lambda Physics", fontweight="bold", fontsize=12)
        ax.set_xticks(x_pos_val)
        ax.set_xticklabels(df_plot_val["lambda_type"], rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, df_plot_val["best_val_loss"])):
            if pd.notna(val) and np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(val) * 0.02,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    # Train-Val Gap
    if "train_val_gap" in df_clean.columns:
        ax = axes[1, 1]
        df_plot_gap = df_clean[
            df_clean["train_val_gap"].notna() & np.isfinite(df_clean["train_val_gap"])
        ]
        x_pos_gap = np.arange(len(df_plot_gap))
        bars = ax.bar(x_pos_gap, df_plot_gap["train_val_gap"], alpha=0.7, color="purple")
        ax.set_xlabel("Lambda Physics Value", fontweight="bold", fontsize=11)
        ax.set_ylabel("Train-Val Gap", fontweight="bold", fontsize=11)
        ax.set_title("Overfitting Indicator (Train-Val Gap)", fontweight="bold", fontsize=12)
        ax.set_xticks(x_pos_gap)
        ax.set_xticklabels(df_plot_gap["lambda_type"], rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
        # Add value labels on bars
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

    plt.tight_layout()

    # Save figure
    filename = generate_timestamped_filename("lambda_ablation_comparison", "png")
    fig_path = output_dir / filename
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"✓ Comparison plot saved: {fig_path}")
    plt.close()

    # Figure 2: Training curves comparison (if history available)
    results_with_history = [r for r in results_list if "training_history" in r]
    if len(results_with_history) > 0:
        # Filter to only show key ablation experiments (n_samples=30, epochs=300)
        # Group by configuration and select best (lowest RMSE) for each
        key_configs = {}
        exp_id_to_res = {r["experiment_id"]: r for r in results_with_history}

        # Filter dataframe to ablation study experiments
        df_ablation = df[
            (df["n_samples"] == 30.0)
            & (df["epochs"] == 300)
            & (df["rmse_delta"].notna())
            & (df["rmse_delta"] > 0.0)
        ].copy()

        # Group by configuration and select best (lowest RMSE) for each
        for _, row in df_ablation.iterrows():
            exp_id = row["experiment_id"]
            if exp_id in exp_id_to_res:
                res = exp_id_to_res[exp_id]
                lambda_p = res.get("lambda_physics", "unknown")
                is_fixed = res.get("use_fixed_lambda", False)

                # Create configuration key
                if is_fixed:
                    config_key = f"Fixed_{lambda_p}"
                else:
                    # Distinguish between old and improved adaptive by checking lambda evolution
                    history = res.get("training_history", {})
                    lambda_evolution = history.get("lambda_physics", [])

                    # Check if it's improved adaptive (gradual increase) vs old (sudden jump)
                    # Improved: gradual increase over epochs 30-100
                    # Old: sudden jump at epoch 30
                    is_improved = False
                    if len(lambda_evolution) > 100:
                        # Check if there's a gradual increase between epochs 30-100
                        # Look for smooth transition (not sudden jump)
                        epoch_30_idx = min(30, len(lambda_evolution) - 1)
                        epoch_50_idx = min(50, len(lambda_evolution) - 1)
                        epoch_70_idx = min(70, len(lambda_evolution) - 1)
                        epoch_100_idx = min(100, len(lambda_evolution) - 1)

                        val_30 = (
                            lambda_evolution[epoch_30_idx]
                            if epoch_30_idx < len(lambda_evolution)
                            else 0.0
                        )
                        val_50 = (
                            lambda_evolution[epoch_50_idx]
                            if epoch_50_idx < len(lambda_evolution)
                            else 0.0
                        )
                        val_70 = (
                            lambda_evolution[epoch_70_idx]
                            if epoch_70_idx < len(lambda_evolution)
                            else 0.0
                        )
                        val_100 = (
                            lambda_evolution[epoch_100_idx]
                            if epoch_100_idx < len(lambda_evolution)
                            else 0.0
                        )

                        # Improved adaptive: gradual increase (values increase smoothly)
                        # Old adaptive: sudden jump (val_30 ≈ 0, then jumps to ~1.0 immediately)
                        # Check if there's a gradual progression
                        has_gradual_increase = (
                            val_30 < 0.3  # Still low at epoch 30
                            and val_50 > val_30
                            and val_50 < 0.8  # Intermediate at epoch 50
                            and val_70 > val_50
                            and val_70 < 0.95  # Still increasing at epoch 70
                            and val_100 > 0.8  # Near target at epoch 100
                        )

                        # Also check: old adaptive jumps immediately (val_30 to val_50 is huge jump)
                        has_sudden_jump = (
                            abs(val_50 - val_30) > 0.7
                        )  # Jump of >0.7 between epochs 30-50

                        if has_gradual_increase and not has_sudden_jump:
                            is_improved = True

                    if is_improved:
                        config_key = "Adaptive (Improved)"
                    else:
                        config_key = "Adaptive (Old)"

                # Keep best (lowest RMSE) for each configuration
                # But always include improved adaptive if it exists, even if slightly worse
                rmse = row["rmse_delta"]
                if config_key not in key_configs:
                    key_configs[config_key] = {"res": res, "rmse": rmse}
                elif config_key == "Adaptive (Improved)":
                    # Always include improved adaptive for comparison
                    key_configs[config_key] = {"res": res, "rmse": rmse}
                elif rmse < key_configs[config_key]["rmse"]:
                    # For other configs, keep best
                    key_configs[config_key] = {"res": res, "rmse": rmse}

        # Extract filtered results
        filtered_results = [v["res"] for v in key_configs.values()]

        if len(filtered_results) == 0:
            # Fallback: use all results if no key experiments found
            print(
                "⚠️  No ablation study experiments found (n_samples=30, epochs=300), showing all experiments"
            )
            filtered_results = results_with_history
        else:
            print(
                f"✓ Filtered to {len(filtered_results)} key ablation experiments (from"
                f"{len(results_with_history)} total)"
            )

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Loss curves
        ax = axes[0]
        for res in filtered_results:
            history = res["training_history"]
            lambda_p = res.get("lambda_physics", "unknown")
            is_fixed = res.get("use_fixed_lambda", False)

            # Determine label based on configuration
            if is_fixed:
                label = f"λ={lambda_p}"
            else:
                # Check if it's improved adaptive
                lambda_evolution = history.get("lambda_physics", [])
                is_improved = False
                if len(lambda_evolution) > 100:
                    epoch_30_idx = min(30, len(lambda_evolution) - 1)
                    epoch_50_idx = min(50, len(lambda_evolution) - 1)
                    epoch_70_idx = min(70, len(lambda_evolution) - 1)
                    epoch_100_idx = min(100, len(lambda_evolution) - 1)

                    val_30 = (
                        lambda_evolution[epoch_30_idx]
                        if epoch_30_idx < len(lambda_evolution)
                        else 0.0
                    )
                    val_50 = (
                        lambda_evolution[epoch_50_idx]
                        if epoch_50_idx < len(lambda_evolution)
                        else 0.0
                    )
                    val_70 = (
                        lambda_evolution[epoch_70_idx]
                        if epoch_70_idx < len(lambda_evolution)
                        else 0.0
                    )
                    val_100 = (
                        lambda_evolution[epoch_100_idx]
                        if epoch_100_idx < len(lambda_evolution)
                        else 0.0
                    )

                    has_gradual_increase = (
                        val_30 < 0.3
                        and val_50 > val_30
                        and val_50 < 0.8
                        and val_70 > val_50
                        and val_70 < 0.95
                        and val_100 > 0.8
                    )
                    has_sudden_jump = abs(val_50 - val_30) > 0.7

                    if has_gradual_increase and not has_sudden_jump:
                        is_improved = True

                label = "Adaptive (Improved)" if is_improved else "Adaptive (Old)"

            val_losses = history.get("val_losses", [])
            train_losses = history.get("train_losses", [])

            # Determine epochs - use the longest loss array to determine length
            max_len = max(len(val_losses), len(train_losses))
            if max_len == 0:
                continue  # Skip if no loss data

            epochs_list = history.get("epochs", [])
            if isinstance(epochs_list, range):
                epochs_list = list(epochs_list)

            # Create epochs if missing or wrong length
            if not epochs_list or len(epochs_list) != max_len:
                epochs_list = list(range(1, max_len + 1))

            # Plot validation losses if available and lengths match
            if val_losses and len(val_losses) == len(epochs_list):
                ax.plot(
                    epochs_list, val_losses, label=f"{label} (val)", linestyle="--", linewidth=2
                )

            # Plot training losses if available and lengths match
            if train_losses and len(train_losses) == len(epochs_list):
                ax.plot(
                    epochs_list,
                    train_losses,
                    label=f"{label} (train)",
                    linestyle="-",
                    linewidth=2,
                    alpha=0.7,
                )

        ax.set_xlabel("Epoch", fontweight="bold", fontsize=11)
        ax.set_ylabel("Loss", fontweight="bold", fontsize=11)
        ax.set_title("Training and Validation Loss Curves", fontweight="bold", fontsize=12)
        ax.legend(fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        # Lambda evolution (for adaptive)
        ax = axes[1]
        for res in filtered_results:
            history = res["training_history"]
            lambda_p = res.get("lambda_physics", "unknown")
            is_fixed = res.get("use_fixed_lambda", False)

            if not is_fixed and "lambda_physics" in history:
                lambda_values = history["lambda_physics"]
                if not lambda_values:
                    continue

                epochs_list = history.get("epochs", [])
                if isinstance(epochs_list, range):
                    epochs_list = list(epochs_list)

                # Ensure epochs and lambda_values have same length
                if not epochs_list or len(epochs_list) != len(lambda_values):
                    epochs_list = list(range(1, len(lambda_values) + 1))

                # Determine label
                is_improved = False
                if len(lambda_values) > 100:
                    epoch_30_idx = min(30, len(lambda_values) - 1)
                    epoch_50_idx = min(50, len(lambda_values) - 1)
                    epoch_70_idx = min(70, len(lambda_values) - 1)
                    epoch_100_idx = min(100, len(lambda_values) - 1)

                    val_30 = (
                        lambda_values[epoch_30_idx] if epoch_30_idx < len(lambda_values) else 0.0
                    )
                    val_50 = (
                        lambda_values[epoch_50_idx] if epoch_50_idx < len(lambda_values) else 0.0
                    )
                    val_70 = (
                        lambda_values[epoch_70_idx] if epoch_70_idx < len(lambda_values) else 0.0
                    )
                    val_100 = (
                        lambda_values[epoch_100_idx] if epoch_100_idx < len(lambda_values) else 0.0
                    )

                    has_gradual_increase = (
                        val_30 < 0.3
                        and val_50 > val_30
                        and val_50 < 0.8
                        and val_70 > val_50
                        and val_70 < 0.95
                        and val_100 > 0.8
                    )
                    has_sudden_jump = abs(val_50 - val_30) > 0.7

                    if has_gradual_increase and not has_sudden_jump:
                        is_improved = True

                lambda_label = (
                    "Adaptive (Improved) λ_physics" if is_improved else "Adaptive (Old) λ_physics"
                )

                if len(epochs_list) == len(lambda_values):
                    ax.plot(
                        epochs_list,
                        lambda_values,
                        label=lambda_label,
                        linewidth=2,
                        marker="o",
                        markersize=3,
                    )
            elif is_fixed:
                ax.axhline(y=lambda_p, label=f"Fixed λ={lambda_p}", linewidth=2, linestyle="--")

        ax.set_xlabel("Epoch", fontweight="bold", fontsize=11)
        ax.set_ylabel("Lambda Physics", fontweight="bold", fontsize=11)
        ax.set_title("Lambda Physics Evolution During Training", fontweight="bold", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = generate_timestamped_filename("lambda_ablation_training_curves", "png")
        fig_path = output_dir / filename
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ Training curves plot saved: {fig_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze lambda_physics ablation study results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        required=True,
        help="Experiment directories or glob patterns",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ablation_studies",
        help="Output directory for results (default: outputs/ablation_studies)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots",
    )

    args = parser.parse_args()

    # Expand glob patterns and find experiment directories
    experiment_dirs = []
    for pattern in args.experiments:
        if "*" in pattern:
            matches = glob.glob(str(PROJECT_ROOT / pattern))
            experiment_dirs.extend([Path(m) for m in matches if Path(m).is_dir()])
        else:
            exp_dir = PROJECT_ROOT / pattern
            if exp_dir.exists() and exp_dir.is_dir():
                experiment_dirs.append(exp_dir)

    if len(experiment_dirs) == 0:
        print("❌ No experiment directories found")
        sys.exit(1)

    print(f"Found {len(experiment_dirs)} experiment directories")

    # Load results
    results_list = []
    for exp_dir in experiment_dirs:
        try:
            results = load_experiment_results(exp_dir)
            if "metrics" in results:
                results_list.append(results)
                print(f"✓ Loaded: {exp_dir.name}")
            else:
                print(f"⚠️  Skipped {exp_dir.name} (no metrics found)")
        except Exception as e:
            print(f"⚠️  Error loading {exp_dir.name}: {e}")

    if len(results_list) == 0:
        print("❌ No valid results found")
        sys.exit(1)

    # Create comparison table
    df = create_comparison_table(results_list)

    # Save table
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = generate_timestamped_filename("lambda_ablation_comparison", "csv")
    csv_path = output_dir / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Comparison table saved: {csv_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("LAMBDA ABLATION STUDY RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))

    # Find best configuration (filter out invalid experiments)
    if "rmse_delta" in df.columns:
        # Filter out invalid experiments:
        # - RMSE = 0.0 or NaN (incomplete/failed experiments)
        # - Very few epochs (< 50) or NaN epochs
        # - NaN samples (incomplete data)
        valid_df = df.copy()

        # Remove rows with invalid RMSE
        valid_df = valid_df[(valid_df["rmse_delta"] > 0.0) & (valid_df["rmse_delta"].notna())]

        # Remove rows with very few epochs (likely incomplete)
        if "epochs" in valid_df.columns:
            valid_df = valid_df[(valid_df["epochs"].notna()) & (valid_df["epochs"] >= 50)]

        # Remove rows with NaN samples (incomplete data)
        if "n_samples" in valid_df.columns:
            valid_df = valid_df[valid_df["n_samples"].notna()]

        if len(valid_df) > 0:
            # Find best among valid experiments
            best_idx = valid_df["rmse_delta"].idxmin()
            best_row = valid_df.loc[best_idx]
            print("\n" + "=" * 70)
            print("BEST CONFIGURATION")
            print("=" * 70)
            print(f"Lambda Type: {best_row['lambda_type']}")
            if "lambda_physics" in best_row.index and pd.notna(best_row["lambda_physics"]):
                if best_row["lambda_type"] == "Adaptive":
                    print(f"Lambda Physics: adaptive")
                else:
                    print(f"Lambda Physics: {best_row['lambda_physics']}")
            print(f"RMSE Delta: {best_row['rmse_delta']:.6f} rad")
            if "r2_delta" in best_row.index and pd.notna(best_row["r2_delta"]):
                print(f"R² Delta: {best_row['r2_delta']:.4f}")
            if "best_val_loss" in best_row.index and pd.notna(best_row["best_val_loss"]):
                print(f"Best Val Loss: {best_row['best_val_loss']:.2f}")
            if "n_samples" in best_row.index and pd.notna(best_row["n_samples"]):
                print(f"n_samples: {best_row['n_samples']}")
            if "epochs" in best_row.index and pd.notna(best_row["epochs"]):
                print(f"epochs: {int(best_row['epochs'])}")
            print(f"Experiment: {best_row['experiment_id']}")
        else:
            print("\n" + "=" * 70)
            print("BEST CONFIGURATION")
            print("=" * 70)
            print("⚠️  No valid experiments found (all filtered out)")
            print("   Check data quality - experiments may be incomplete")

    # Generate plots
    if args.plot:
        print("\n" + "=" * 70)
        print("GENERATING PLOTS")
        print("=" * 70)
        plot_lambda_comparison(df, results_list, output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
