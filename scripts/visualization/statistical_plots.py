"""
Statistical Visualization Plots.

Generates box plots, confidence interval plots, and distribution plots
for statistical validation results.
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from scripts.analysis.statistical_summary import compute_statistical_summary, extract_metric_values


def generate_statistical_plots(results: List[Dict], output_dir: Path) -> None:
    """
    Generate all statistical plots.

    Parameters:
    -----------
    results : list
        List of result dictionaries
    output_dir : Path
        Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute summary
    summary = compute_statistical_summary(results)

    # Generate box plots for key metrics
    generate_box_plots(results, output_dir)

    # Generate confidence interval plots
    generate_ci_plots(summary, output_dir)

    # Generate distribution plots
    generate_distribution_plots(results, output_dir)


def generate_box_plots(results: List[Dict], output_dir: Path) -> None:
    """
    Generate box plots for key metrics.

    Parameters:
    -----------
    results : list
        List of result dictionaries
    output_dir : Path
        Directory to save plots
    """
    # Key metrics to plot
    metrics_to_plot = [
        ("delta_r2", "R² Delta"),
        ("omega_r2", "R² Omega"),
        ("delta_rmse", "RMSE Delta (rad)"),
        ("omega_rmse", "RMSE Omega (pu)"),
        ("delta_mae", "MAE Delta (rad)"),
        ("omega_mae", "MAE Omega (pu)"),
    ]

    # Extract values for each metric
    data_to_plot = {}
    labels = []
    for metric_key, metric_label in metrics_to_plot:
        values = extract_metric_values(results, metric_key)
        if values is not None and len(values) > 0:
            # Fix: Filter out NaN/inf values before plotting
            valid_values = values[np.isfinite(values)]
            if len(valid_values) > 0:
                data_to_plot[metric_label] = valid_values
                labels.append(metric_label)

    if len(data_to_plot) == 0:
        print("Warning: No metrics found for box plots")
        return

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (metric_label, values) in enumerate(data_to_plot.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        bp = ax.boxplot(
            [values],
            labels=[metric_label],
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )

        # Style the box plot
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)

        ax.set_ylabel("Value")
        ax.set_title(f"{metric_label} Distribution")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(data_to_plot), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "box_plots.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved box plots to: {output_dir / 'box_plots.png'}")


def generate_ci_plots(summary: Dict, output_dir: Path) -> None:
    """
    Generate confidence interval plots.

    Parameters:
    -----------
    summary : dict
        Statistical summary dictionary
    output_dir : Path
        Directory to save plots
    """
    # Key metrics to plot
    metrics_to_plot = [
        ("delta_r2", "R² Delta"),
        ("omega_r2", "R² Omega"),
        ("delta_rmse", "RMSE Delta (rad)"),
        ("omega_rmse", "RMSE Omega (pu)"),
    ]

    # Extract data
    metric_names = []
    means = []
    ci_lowers = []
    ci_uppers = []

    for metric_key, metric_label in metrics_to_plot:
        if metric_key in summary:
            stats = summary[metric_key]
            if isinstance(stats, dict) and "mean" in stats:
                # Fix: Filter out NaN/inf values before plotting
                mean = stats["mean"]
                ci_lower = stats.get("ci_lower", mean)
                ci_upper = stats.get("ci_upper", mean)
                # Skip if any value is NaN or inf
                if np.isfinite(mean) and np.isfinite(ci_lower) and np.isfinite(ci_upper):
                    metric_names.append(metric_label)
                    means.append(mean)
                    ci_lowers.append(ci_lower)
                    ci_uppers.append(ci_upper)

    if len(metric_names) == 0:
        print("Warning: No metrics found for CI plots")
        return

    # Create figure with better styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Statistical Validation Results (5 runs, 95% CI)", fontsize=14, fontweight="bold", y=1.02
    )

    # Plot 1: R² metrics
    r2_metrics = [m for m in metric_names if "R²" in m]
    if r2_metrics:
        r2_means = [means[metric_names.index(m)] for m in r2_metrics]
        r2_lowers = [ci_lowers[metric_names.index(m)] for m in r2_metrics]
        r2_uppers = [ci_uppers[metric_names.index(m)] for m in r2_metrics]

        x_pos = np.arange(len(r2_metrics))
        colors = ["#2E86AB", "#A23B72"]  # Blue and purple for better contrast

        bars = ax1.bar(
            x_pos,
            r2_means,
            width=0.6,
            color=colors[: len(r2_metrics)],
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
            label="Mean",
        )

        # Add error bars on top of bars
        yerr_lower = np.array(r2_means) - np.array(r2_lowers)
        yerr_upper = np.array(r2_uppers) - np.array(r2_means)
        ax1.errorbar(
            x_pos,
            r2_means,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            color="black",
            capsize=8,
            capthick=2,
            elinewidth=2,
            label="95% CI",
        )

        # Add value labels on bars (simplified)
        for i, (mean, lower, upper) in enumerate(zip(r2_means, r2_lowers, r2_uppers)):
            # Show mean on bar, CI range above
            ax1.text(
                i,
                mean / 2,  # Middle of bar
                f"{mean:.3f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )
            ax1.text(
                i,
                mean + yerr_upper[i] + 0.015,
                f"±{yerr_upper[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                style="italic",
            )

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(r2_metrics, fontsize=12, fontweight="bold")
        ax1.set_ylabel("R² Score", fontsize=12, fontweight="bold")
        ax1.set_title(
            "R² Metrics with 95% Confidence Intervals", fontsize=13, fontweight="bold", pad=20
        )
        ax1.grid(True, alpha=0.3, linestyle="--", axis="y", zorder=0)
        ax1.set_ylim([0, max(r2_uppers) * 1.12])  # Dynamic y-limit based on data
        ax1.axhline(y=0, color="black", linewidth=0.8, zorder=1)
        # Remove legend - values are on bars
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

    # Plot 2: RMSE metrics
    rmse_metrics = [m for m in metric_names if "RMSE" in m]
    if rmse_metrics:
        rmse_means = [means[metric_names.index(m)] for m in rmse_metrics]
        rmse_lowers = [ci_lowers[metric_names.index(m)] for m in rmse_metrics]
        rmse_uppers = [ci_uppers[metric_names.index(m)] for m in rmse_metrics]

        x_pos = np.arange(len(rmse_metrics))
        colors = ["#F18F01", "#C73E1D"]  # Orange and red for better contrast

        bars = ax2.bar(
            x_pos,
            rmse_means,
            width=0.6,
            color=colors[: len(rmse_metrics)],
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
            label="Mean",
        )

        # Add error bars on top of bars
        yerr_lower = np.array(rmse_means) - np.array(rmse_lowers)
        yerr_upper = np.array(rmse_uppers) - np.array(rmse_means)
        ax2.errorbar(
            x_pos,
            rmse_means,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            color="black",
            capsize=8,
            capthick=2,
            elinewidth=2,
            label="95% CI",
        )

        # Add value labels on bars (simplified)
        for i, (mean, lower, upper) in enumerate(zip(rmse_means, rmse_lowers, rmse_uppers)):
            # Show mean on bar, CI range above
            ax2.text(
                i,
                mean / 2,  # Middle of bar
                f"{mean:.4f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )
            ax2.text(
                i,
                mean + yerr_upper[i] + max(rmse_means) * 0.03,
                f"±{yerr_upper[i]:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                style="italic",
            )

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(rmse_metrics, fontsize=12, fontweight="bold")
        ax2.set_ylabel("RMSE", fontsize=12, fontweight="bold")
        ax2.set_title(
            "RMSE Metrics with 95% Confidence Intervals", fontsize=13, fontweight="bold", pad=20
        )
        ax2.grid(True, alpha=0.3, linestyle="--", axis="y", zorder=0)
        ax2.set_ylim([0, max(rmse_uppers) * 1.15])  # Dynamic y-limit based on data
        ax2.axhline(y=0, color="black", linewidth=0.8, zorder=1)
        # Remove legend - values are on bars
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "confidence_intervals.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved CI plots to: {output_dir / 'confidence_intervals.png'}")


def generate_distribution_plots(results: List[Dict], output_dir: Path) -> None:
    """
    Generate distribution plots for key metrics.

    Parameters:
    -----------
    results : list
        List of result dictionaries
    output_dir : Path
        Directory to save plots
    """
    # Key metrics to plot
    metrics_to_plot = [
        ("delta_r2", "R² Delta"),
        ("omega_r2", "R² Omega"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        values = extract_metric_values(results, metric_key)
        if values is None or len(values) == 0:
            continue

        # Fix: Filter out NaN/inf values before plotting
        valid_values = values[np.isfinite(values)]
        if len(valid_values) == 0:
            continue

        ax = axes[idx]
        ax.hist(valid_values, bins=min(10, len(valid_values)), edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(valid_values), color="red", linestyle="--", linewidth=2, label="Mean")
        ax.axvline(
            np.median(valid_values), color="green", linestyle="--", linewidth=2, label="Median"
        )
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{metric_label} Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved distribution plots to: {output_dir / 'distributions.png'}")
