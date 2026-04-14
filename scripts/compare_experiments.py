#!/usr/bin/env python
"""
Experiment Comparison Script.

Compare multiple experiment results and generate comparison reports.

Usage:
    python scripts/compare_experiments.py --experiments outputs/experiments/exp_*
    python scripts/compare_experiments.py --experiments exp_20250101_120000 exp_20250101_140000
    python scripts/compare_experiments.py --metric val_loss
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.experiment_tracker import ExperimentTracker
from scripts.core.utils import load_config, save_json, generate_timestamped_filename


def load_experiment_results(experiment_dir: Path) -> dict:
    """
    Load results from an experiment directory.

    Parameters:
    -----------
    experiment_dir : Path
        Experiment directory path

    Returns:
    --------
    results : dict
        Experiment results
    """
    results = {}

    # Load experiment summary
    summary_path = experiment_dir / "experiment_summary.json"
    if summary_path.exists():
        import json

        with open(summary_path, "r") as f:
            results["summary"] = json.load(f)

    # Load metrics (support both timestamped and non-timestamped filenames)
    metrics_dir = experiment_dir / "results"
    if metrics_dir.exists():
        # Try timestamped filename first (most recent)
        metrics_files = list(metrics_dir.glob("metrics_*.json"))
        if metrics_files:
            # Use most recent metrics file
            metrics_path = max(metrics_files, key=lambda p: p.stat().st_mtime)
        else:
            # Fallback to non-timestamped
            metrics_path = metrics_dir / "metrics.json"

        if metrics_path.exists():
            import json

            with open(metrics_path, "r") as f:
                results["metrics"] = json.load(f)

    # Load training history (support both timestamped and non-timestamped filenames)
    model_dir = experiment_dir / "model"
    if model_dir.exists():
        # Try timestamped filename first (most recent)
        history_files = list(model_dir.glob("training_history_*.json"))
        if history_files:
            # Use most recent history file
            history_path = max(history_files, key=lambda p: p.stat().st_mtime)
        else:
            # Fallback to non-timestamped
            history_path = model_dir / "training_history.json"

        if history_path.exists():
            import json

            with open(history_path, "r") as f:
                results["training_history"] = json.load(f)

    # Load config
    config_path = experiment_dir / "config.yaml"
    if config_path.exists():
        results["config"] = load_config(config_path)

    return results


def compare_experiments(experiment_dirs: list, output_dir: Path = None):
    """
    Compare multiple experiments.

    Parameters:
    -----------
    experiment_dirs : list
        List of experiment directory paths
    output_dir : Path, optional
        Directory to save comparison results
    """
    print("=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)

    if len(experiment_dirs) < 2:
        print("❌ Need at least 2 experiments to compare")
        return

    print(f"Comparing {len(experiment_dirs)} experiments...")

    # Load results from each experiment
    all_results = []
    for exp_dir in experiment_dirs:
        exp_dir = Path(exp_dir)
        if not exp_dir.exists():
            print(f"⚠️  Experiment directory not found: {exp_dir}")
            continue

        exp_id = exp_dir.name
        results = load_experiment_results(exp_dir)
        results["experiment_id"] = exp_id
        all_results.append(results)

    if len(all_results) < 2:
        print("❌ Not enough valid experiments to compare")
        return

    # Create comparison table
    comparison_data = []

    for results in all_results:
        exp_id = results.get("experiment_id", "unknown")
        config = results.get("config", {})
        metrics = results.get("metrics", {})
        training_history = results.get("training_history", {})

        # Extract key information
        model_config = config.get("model", {})
        training_config = config.get("training", {})

        row = {
            "experiment_id": exp_id,
            "hidden_dims": str(model_config.get("hidden_dims", [])),
            "epochs": training_config.get("epochs", 0),
            "learning_rate": training_config.get("learning_rate", 0),
            "batch_size": training_config.get("batch_size", 0),
        }

        # Add metrics (handle nested structure if present)
        if metrics:
            # Check if metrics are nested under "metrics" key
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
            if val_losses:
                row["best_val_loss"] = min(val_losses)
                row["final_val_loss"] = val_losses[-1]

        comparison_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))

    # Save comparison
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV with timestamp
        csv_filename = generate_timestamped_filename("experiment_comparison", "csv")
        csv_path = output_dir / csv_filename
        df.to_csv(csv_path, index=False)
        print(f"\n[OK] Comparison saved to: {csv_path}")

        # Save as JSON with timestamp
        json_filename = generate_timestamped_filename("experiment_comparison", "json")
        json_path = output_dir / json_filename
        save_json(df.to_dict("records"), json_path)
        print(f"[OK] Comparison saved to: {json_path}")

    return df


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare multiple experiment results")
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
    parser.add_argument(
        "--metric", type=str, default=None, help="Specific metric to compare (optional)"
    )

    args = parser.parse_args()

    # Expand glob patterns
    experiment_dirs = []
    for pattern in args.experiments:
        # Check if it's a glob pattern
        if "*" in pattern or "?" in pattern:
            matches = glob.glob(pattern)
            experiment_dirs.extend(matches)
        else:
            experiment_dirs.append(pattern)

    # Remove duplicates and sort
    experiment_dirs = sorted(list(set(experiment_dirs)))

    if len(experiment_dirs) == 0:
        print("❌ No experiment directories found")
        return

    # Compare experiments
    df = compare_experiments(experiment_dirs, args.output_dir)

    # If specific metric requested, show detailed comparison
    if args.metric and df is not None:
        if args.metric in df.columns:
            print(f"\n{'=' * 70}")
            print(f"METRIC COMPARISON: {args.metric}")
            print("=" * 70)
            metric_df = df[["experiment_id", args.metric]].sort_values(args.metric)
            print(metric_df.to_string(index=False))
        else:
            print(f"⚠️  Metric '{args.metric}' not found in results")
            print(f"Available metrics: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()
