"""
Pareto Frontier Visualization.

Creates Pareto frontier plots showing the trade-off between Delta and Omega performance.
"""

import json
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None


def load_experiment_config(experiment_dir: Path) -> Optional[Dict]:
    """
    Load experiment configuration details.

    Parameters:
    -----------
    experiment_dir : Path
        Experiment directory path

    Returns:
    --------
    config : dict or None
        Configuration dictionary with n_samples, epochs, architecture, or None if not found
    """
    # Try experiment_summary.json first
    summary_path = experiment_dir / "experiment_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                data = json.load(f)
                config = data.get("config", {})

                # Extract key details
                model_config = config.get("model", {})
                hidden_dims = model_config.get("hidden_dims", [])
                architecture = "x".join(map(str, hidden_dims)) if hidden_dims else "unknown"

                training_config = config.get("training", {})
                epochs = training_config.get("epochs", 0)

                data_config = config.get("data", {}).get("generation", {})
                n_samples = data_config.get("n_samples", 0)

                return {
                    "n_samples": n_samples,
                    "epochs": epochs,
                    "architecture": architecture,
                }
        except Exception:
            pass

    # Try config.yaml as fallback
    config_path = experiment_dir / "config.yaml"
    if config_path.exists() and yaml is not None:
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

                model_config = config.get("model", {})
                hidden_dims = model_config.get("hidden_dims", [])
                architecture = "x".join(map(str, hidden_dims)) if hidden_dims else "unknown"

                training_config = config.get("training", {})
                epochs = training_config.get("epochs", 0)

                data_config = config.get("data", {}).get("generation", {})
                n_samples = data_config.get("n_samples", 0)

                return {
                    "n_samples": n_samples,
                    "epochs": epochs,
                    "architecture": architecture,
                }
        except Exception:
            pass

    return None


def load_experiment_metrics(experiment_dir: Path) -> Optional[Dict]:
    """
    Load metrics from an experiment directory.

    Parameters:
    -----------
    experiment_dir : Path
        Experiment directory path

    Returns:
    --------
    metrics : dict or None
        Metrics dictionary, or None if not found
    """
    # Try results/metrics_*.json first (most reliable)
    metrics_dir = experiment_dir / "results"
    if metrics_dir.exists():
        metrics_files = list(metrics_dir.glob("metrics_*.json"))
        if metrics_files:
            metrics_path = max(metrics_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(metrics_path, "r") as f:
                    data = json.load(f)
                    if "metrics" in data:
                        return data["metrics"]
                    return data
            except Exception as e:
                print(f"Warning: Could not load {metrics_path}: {e}")

    # Try experiment_summary.json
    summary_path = experiment_dir / "experiment_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                data = json.load(f)
                # Try results.metrics first (newer format)
                results = data.get("results", {})
                if isinstance(results, dict):
                    metrics = results.get("metrics", {})
                    if metrics and isinstance(metrics, dict):
                        return metrics
                # Try direct metrics key (older format)
                metrics = data.get("metrics", {})
                if isinstance(metrics, dict):
                    # Handle nested metrics.metrics structure
                    if "metrics" in metrics and isinstance(metrics["metrics"], dict):
                        return metrics["metrics"]
                    return metrics
        except Exception as e:
            print(f"Warning: Could not load {summary_path}: {e}")

    return None


def identify_pareto_optimal(points: np.ndarray) -> np.ndarray:
    """
    Identify Pareto-optimal points.

    Parameters:
    -----------
    points : np.ndarray
        Array of shape (n_points, 2) with [delta_r2, omega_r2] values

    Returns:
    --------
    pareto_mask : np.ndarray
        Boolean array indicating which points are Pareto-optimal
    """
    n = len(points)
    pareto_mask = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Point j dominates point i if:
            # - j has higher or equal delta_r2 AND higher or equal omega_r2
            # - AND at least one is strictly higher
            if (
                points[j, 0] >= points[i, 0]
                and points[j, 1] >= points[i, 1]
                and (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])
            ):
                pareto_mask[i] = False
                break

    return pareto_mask


def generate_pareto_frontier_plot(
    experiment_dirs: List[Path],
    output_path: Path,
    title: str = "Pareto Frontier: Delta vs Omega Performance",
) -> None:
    """
    Generate Pareto frontier plot from multiple experiments.

    Parameters:
    -----------
    experiment_dirs : list
        List of experiment directory paths
    output_path : Path
        Path to save the plot
    title : str
        Plot title
    """
    # Collect metrics and config from all experiments
    delta_r2_values = []
    omega_r2_values = []
    experiment_ids = []
    experiment_configs = {}  # Store config for each experiment

    for exp_dir in experiment_dirs:
        metrics = load_experiment_metrics(exp_dir)
        if metrics is None:
            continue

        delta_r2 = metrics.get("r2_delta") or metrics.get("delta_r2")
        omega_r2 = metrics.get("r2_omega") or metrics.get("omega_r2")

        if delta_r2 is not None and omega_r2 is not None:
            delta_r2_values.append(delta_r2)
            omega_r2_values.append(omega_r2)
            experiment_ids.append(exp_dir.name)

            # Load config details
            config = load_experiment_config(exp_dir)
            if config:
                experiment_configs[exp_dir.name] = config

    if len(delta_r2_values) == 0:
        print("Warning: No valid metrics found for Pareto frontier plot")
        return

    # Convert to numpy arrays
    points = np.array([[d, o] for d, o in zip(delta_r2_values, omega_r2_values)])

    # Identify Pareto-optimal points
    pareto_mask = identify_pareto_optimal(points)
    pareto_points = points[pareto_mask]
    pareto_experiment_ids = [
        experiment_ids[i] for i in range(len(experiment_ids)) if pareto_mask[i]
    ]
    non_pareto_points = points[~pareto_mask]

    # Sort Pareto points for plotting
    if len(pareto_points) > 0:
        sort_idx = np.argsort(pareto_points[:, 0])
        pareto_points = pareto_points[sort_idx]
        pareto_experiment_ids = [pareto_experiment_ids[i] for i in sort_idx]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot non-Pareto points
    if len(non_pareto_points) > 0:
        ax.scatter(
            non_pareto_points[:, 0],
            non_pareto_points[:, 1],
            c="lightgray",
            alpha=0.5,
            s=50,
            label="Non-Pareto",
            edgecolors="black",
            linewidths=0.5,
        )

    # Plot Pareto-optimal points
    if len(pareto_points) > 0:
        ax.scatter(
            pareto_points[:, 0],
            pareto_points[:, 1],
            c="red",
            s=100,
            label="Pareto-Optimal",
            edgecolors="darkred",
            linewidths=2,
            zorder=5,
        )

        # Draw Pareto frontier line
        ax.plot(
            pareto_points[:, 0],
            pareto_points[:, 1],
            "r--",
            linewidth=2,
            alpha=0.7,
            label="Pareto Frontier",
            zorder=4,
        )

        # Add annotations for Pareto-optimal points
        for i, (point, exp_id) in enumerate(zip(pareto_points, pareto_experiment_ids)):
            config = experiment_configs.get(exp_id, {})
            n_samples = config.get("n_samples", "?")
            epochs = config.get("epochs", "?")
            arch = config.get("architecture", "?")

            # Create label text
            label = f"n={n_samples}\nep={epochs}\narch={arch}"

            # Position annotation to avoid overlap
            # Alternate between top-right and bottom-left
            if i % 2 == 0:
                # Top-right
                xytext = (10, 10)
                ha = "left"
                va = "bottom"
            else:
                # Bottom-left
                xytext = (-10, -10)
                ha = "right"
                va = "top"

            # Add annotation with arrow
            ax.annotate(
                label,
                xy=(point[0], point[1]),
                xytext=xytext,
                textcoords="offset points",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="yellow",
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=1,
                ),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="black", lw=1),
                ha=ha,
                va=va,
                zorder=6,
            )

    # Plot all points
    ax.scatter(
        delta_r2_values,
        omega_r2_values,
        c="blue",
        s=30,
        alpha=0.6,
        label="All Experiments",
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )

    ax.set_xlabel("R² Delta", fontsize=12, fontweight="bold")
    ax.set_ylabel("R² Omega", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])

    # Add quadrant labels
    ax.axhline(y=0.6, color="green", linestyle=":", alpha=0.5, linewidth=1)
    ax.axvline(x=0.85, color="green", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(0.02, 0.62, "Ω > 0.6", fontsize=10, color="green", alpha=0.7)
    ax.text(0.87, 0.02, "Δ > 0.85", fontsize=10, color="green", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved Pareto frontier plot to: {output_path}")
    print(f"  Total experiments: {len(delta_r2_values)}")
    print(f"  Pareto-optimal: {np.sum(pareto_mask)}")


def generate_pareto_from_experiment_log(
    log_file: Path, output_path: Path, title: str = "Pareto Frontier: Delta vs Omega Performance"
) -> None:
    """
    Generate Pareto frontier plot from experiment log CSV.

    Parameters:
    -----------
    log_file : Path
        Path to experiment_log.csv
    output_path : Path
        Path to save the plot
    title : str
        Plot title
    """
    import pandas as pd

    if not log_file.exists():
        print(f"Warning: Log file not found: {log_file}")
        return

    df = pd.read_csv(log_file)

    # Extract R² values (may need to load from experiment directories)
    delta_r2_values = []
    omega_r2_values = []

    base_dir = log_file.parent
    for _, row in df.iterrows():
        exp_id = row.get("experiment_id", "")
        exp_dir = base_dir / exp_id

        metrics = load_experiment_metrics(exp_dir)
        if metrics is None:
            continue

        delta_r2 = metrics.get("r2_delta") or metrics.get("delta_r2")
        omega_r2 = metrics.get("r2_omega") or metrics.get("omega_r2")

        if delta_r2 is not None and omega_r2 is not None:
            delta_r2_values.append(delta_r2)
            omega_r2_values.append(omega_r2)

    if len(delta_r2_values) == 0:
        print("Warning: No valid metrics found in experiment log")
        return

    # Create temporary experiment dirs list (just for plotting)
    experiment_dirs = [base_dir / f"exp_{i}" for i in range(len(delta_r2_values))]
    generate_pareto_frontier_plot(experiment_dirs, output_path, title)
