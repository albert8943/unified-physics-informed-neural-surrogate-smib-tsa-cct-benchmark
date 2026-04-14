#!/usr/bin/env python
"""Generate comprehensive comparison table for experiments."""
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_experiment_info(exp_dir: Path):
    """Extract experiment information."""
    summary_file = exp_dir / "experiment_summary.json"
    if not summary_file.exists():
        return None

    try:
        with open(summary_file, "r") as f:
            data = json.load(f)

        config = data.get("config", {})
        metrics = data.get("metrics", {})
        training_history = data.get("training_history", {})

        # Model architecture
        model_config = config.get("model", {})
        hidden_dims = model_config.get("hidden_dims", [])
        architecture = "x".join(map(str, hidden_dims)) if hidden_dims else "unknown"

        # Training config
        training_config = config.get("training", {})
        epochs = training_config.get("epochs", 0)
        actual_epochs = len(training_history.get("val_losses", [])) if training_history else epochs

        # Data config
        data_config = config.get("data", {}).get("generation", {})
        n_samples = data_config.get("n_samples", 0)

        # Loss config
        loss_config = config.get("loss", {})
        lambda_physics = loss_config.get("lambda_physics", 0)
        use_fixed_lambda = loss_config.get("use_fixed_lambda", True)
        lambda_type = "Fixed" if use_fixed_lambda else "Adaptive"
        scale_to_norm = loss_config.get("scale_to_norm", [1.0, 100.0])
        omega_weight = scale_to_norm[1] if len(scale_to_norm) > 1 else 100.0

        # Metrics
        if "metrics" in metrics and isinstance(metrics["metrics"], dict):
            actual_metrics = metrics["metrics"]
        else:
            actual_metrics = metrics

        # Training metrics
        val_losses = training_history.get("val_losses", []) if training_history else []
        train_losses = training_history.get("train_losses", []) if training_history else []
        best_val_loss = min(val_losses) if val_losses else None

        # Date
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
        }
    except Exception as e:
        print(f"Error processing {exp_dir.name}: {e}", file=sys.stderr)
        return None


def main():
    """Main function."""
    output_dir = Path("outputs/experiments")

    # Find all experiments
    experiment_dirs = [
        d for d in output_dir.glob("exp_*") if (d / "experiment_summary.json").exists()
    ]

    print(f"Found {len(experiment_dirs)} experiments")

    # Extract data
    rows = []
    for exp_dir in experiment_dirs:
        info = extract_experiment_info(exp_dir)
        if info:
            rows.append(info)

    if not rows:
        print("No valid experiments found")
        return

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Sort by date, n_samples, epochs
    df = df.sort_values(["date", "n_samples", "epochs"])

    # Filter for yesterday and today
    yesterday = datetime(2024, 12, 8).date()
    today = datetime(2024, 12, 9).date()
    df_filtered = df[df["date"].isin([yesterday.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")])]

    # Save CSV
    output_file = Path("outputs/comprehensive_comparison_table.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_file, index=False)
    print(f"\n✅ Saved comparison table to: {output_file}")

    # Display table
    print(f"\n{'='*120}")
    print("COMPREHENSIVE EXPERIMENT COMPARISON TABLE")
    print(f"{'='*120}\n")

    # Select columns for display
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

    display_cols = [col for col in display_cols if col in df_filtered.columns]

    # Format display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 25)

    print(df_filtered[display_cols].to_string(index=False))

    # Summary by n_samples
    print(f"\n{'='*120}")
    print("SUMMARY BY n_samples")
    print(f"{'='*120}\n")

    for n_samples in sorted(df_filtered["n_samples"].unique()):
        subset = df_filtered[df_filtered["n_samples"] == n_samples]
        print(f"\nn_samples = {n_samples}:")
        print(f"  Total experiments: {len(subset)}")
        if "rmse_delta" in subset.columns and not subset["rmse_delta"].isna().all():
            best = subset.loc[subset["rmse_delta"].idxmin()]
            print(
                f"Best RMSE Delta: {best['rmse_delta']:.4f} (epochs={best['epochs']},"
                f"exp={best['experiment_id']})"
            )
        if "r2_omega" in subset.columns and not subset["r2_omega"].isna().all():
            best_omega = subset.loc[subset["r2_omega"].idxmax()]
            print(
                f"Best R² Omega: {best_omega['r2_omega']:.4f} (epochs={best_omega['epochs']},"
                f"exp={best_omega['experiment_id']})"
            )


if __name__ == "__main__":
    main()
