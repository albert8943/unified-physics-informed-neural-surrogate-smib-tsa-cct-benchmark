#!/usr/bin/env python
"""
Create Comprehensive Comparison Table

Compares experiments from different days with n_samples, epochs, architecture, and metrics.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


def extract_experiment_data(exp_dir: Path) -> Optional[Dict]:
    """Extract all relevant data from an experiment directory."""
    summary_file = exp_dir / "experiment_summary.json"
    if not summary_file.exists():
        return None

    try:
        with open(summary_file, "r") as f:
            data = json.load(f)

        config = data.get("config", {})
        metrics = data.get("metrics", {})
        training_history = data.get("training_history", {})

        # Extract model architecture
        model_config = config.get("model", {})
        hidden_dims = model_config.get("hidden_dims", [])
        architecture = "x".join(map(str, hidden_dims)) if hidden_dims else "unknown"

        # Extract training config
        training_config = config.get("training", {})
        epochs = training_config.get("epochs", 0)
        actual_epochs = len(training_history.get("val_losses", [])) if training_history else epochs

        # Extract data config
        data_config = config.get("data", {}).get("generation", {})
        n_samples = data_config.get("n_samples", 0)

        # Extract loss config
        loss_config = config.get("loss", {})
        lambda_physics = loss_config.get("lambda_physics", 0)
        use_fixed_lambda = loss_config.get("use_fixed_lambda", True)
        lambda_type = "Fixed" if use_fixed_lambda else "Adaptive"
        omega_weight = (
            loss_config.get("scale_to_norm", [1.0, 100.0])[1]
            if loss_config.get("scale_to_norm")
            else 100.0
        )

        # Extract metrics (handle nested structure)
        if "metrics" in metrics and isinstance(metrics["metrics"], dict):
            actual_metrics = metrics["metrics"]
        else:
            actual_metrics = metrics

        # Extract training metrics
        val_losses = training_history.get("val_losses", []) if training_history else []
        train_losses = training_history.get("train_losses", []) if training_history else []
        best_val_loss = min(val_losses) if val_losses else None
        final_val_loss = val_losses[-1] if val_losses else None
        final_train_loss = train_losses[-1] if train_losses else None
        train_val_gap = (
            (final_val_loss - final_train_loss) if (final_val_loss and final_train_loss) else None
        )

        # Get experiment date
        exp_date = datetime.fromtimestamp(exp_dir.stat().st_mtime).date()

        return {
            "experiment_id": exp_dir.name,
            "date": exp_date.strftime("%Y-%m-%d"),
            "n_samples": n_samples,
            "epochs": epochs,
            "actual_epochs": actual_epochs,
            "architecture": architecture,
            "lambda_physics": lambda_physics,
            "lambda_type": lambda_type,
            "omega_weight": omega_weight,
            "rmse_delta": actual_metrics.get("rmse_delta"),
            "rmse_omega": actual_metrics.get("rmse_omega"),
            "mae_delta": actual_metrics.get("mae_delta"),
            "mae_omega": actual_metrics.get("mae_omega"),
            "r2_delta": actual_metrics.get("r2_delta"),
            "r2_omega": actual_metrics.get("r2_omega"),
            "best_val_loss": best_val_loss,
            "final_val_loss": final_val_loss,
            "final_train_loss": final_train_loss,
            "train_val_gap": train_val_gap,
        }
    except Exception as e:
        print(f"Error processing {exp_dir.name}: {e}")
        return None


def create_comparison_table(
    experiment_dirs: List[Path], output_file: Optional[Path] = None
) -> pd.DataFrame:
    """Create comprehensive comparison table from experiments."""
    rows = []

    for exp_dir in experiment_dirs:
        data = extract_experiment_data(exp_dir)
        if data:
            rows.append(data)

    df = pd.DataFrame(rows)

    # Sort by date, then n_samples, then epochs
    if not df.empty:
        df = df.sort_values(["date", "n_samples", "epochs"])

    # Save to file if specified
    if output_file and not df.empty:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\n✅ Comparison table saved to: {output_file}")

    return df


def main():
    """Main function to create comparison table."""
    import argparse

    parser = argparse.ArgumentParser(description="Create comprehensive experiment comparison table")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments",
        help="Directory containing experiment folders",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="outputs/comprehensive_comparison_table.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--filter-date",
        type=str,
        nargs="+",
        help="Filter by date(s) (YYYY-MM-DD format)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Error: Output directory not found: {output_dir}")
        return

    # Find all experiments
    experiment_dirs = [
        d for d in output_dir.glob("exp_*") if (d / "experiment_summary.json").exists()
    ]

    # Filter by date if specified
    if args.filter_date:
        filter_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in args.filter_date]
        experiment_dirs = [
            d
            for d in experiment_dirs
            if datetime.fromtimestamp(d.stat().st_mtime).date() in filter_dates
        ]

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EXPERIMENT COMPARISON")
    print(f"{'='*80}")
    print(f"Found {len(experiment_dirs)} experiments")

    # Create comparison table
    df = create_comparison_table(experiment_dirs, Path(args.output_file))

    if df.empty:
        print("❌ No valid experiments found")
        return

    # Print formatted table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}\n")

    # Select key columns for display
    display_cols = [
        "experiment_id",
        "date",
        "n_samples",
        "epochs",
        "actual_epochs",
        "architecture",
        "lambda_physics",
        "lambda_type",
        "omega_weight",
        "rmse_delta",
        "rmse_omega",
        "r2_delta",
        "r2_omega",
        "best_val_loss",
    ]

    # Filter to existing columns
    display_cols = [col for col in display_cols if col in df.columns]

    # Format for display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 20)

    print(df[display_cols].to_string(index=False))

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    if "n_samples" in df.columns:
        print("By n_samples:")
        for n_samples in sorted(df["n_samples"].unique()):
            subset = df[df["n_samples"] == n_samples]
            if "rmse_delta" in subset.columns:
                best = (
                    subset.loc[subset["rmse_delta"].idxmin()]
                    if not subset["rmse_delta"].isna().all()
                    else None
                )
                if best is not None:
                    print(
                        f"  n_samples={n_samples}: Best RMSE Delta = {best['rmse_delta']:.4f} "
                        f"(epochs={best['epochs']}, exp={best['experiment_id']})"
                    )

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
