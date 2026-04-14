#!/usr/bin/env python
"""
Find the best experiment that has preprocessed data available.

Usage:
    python scripts/find_best_experiment_data.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_best_experiment_with_data():
    """Find the best experiment that has preprocessed data."""
    # Load comparison results
    comparison_file = (
        PROJECT_ROOT / "outputs" / "comparisons" / "experiment_comparison_20251208_120403.csv"
    )

    if not comparison_file.exists():
        print("❌ Comparison file not found. Run comparison first.")
        return None

    df = pd.read_csv(comparison_file)
    df_sorted = df.sort_values("rmse_delta")  # Best first

    base = PROJECT_ROOT / "outputs" / "experiments"

    print("=" * 70)
    print("FINDING BEST EXPERIMENT WITH PREPROCESSED DATA")
    print("=" * 70)
    print("\nChecking experiments (sorted by RMSE, best first):\n")

    for idx, row in df_sorted.iterrows():
        exp_id = row["experiment_id"]
        preprocessed_dir = base / exp_id / "data" / "preprocessed"

        # Check if preprocessed directory exists and has CSV files
        has_data = False
        csv_files = []
        if preprocessed_dir.exists():
            csv_files = list(preprocessed_dir.glob("train_data_*.csv"))
            has_data = len(csv_files) > 0

        status = "✓ HAS DATA" if has_data else "✗ No data"
        print(
            f"{exp_id}: RMSE={row['rmse_delta']:.4f}, R²={row['r2_delta']:.4f}, "
            f"Epochs={row['epochs']}, {status}"
        )

        if has_data:
            print(f"  → Found {len(csv_files)} train data file(s)")
            print(f"  → Data directory: {preprocessed_dir}")
            print("\n" + "=" * 70)
            print("✓ BEST EXPERIMENT WITH DATA FOUND!")
            print("=" * 70)
            print(f"\nExperiment: {exp_id}")
            print(f"RMSE Delta: {row['rmse_delta']:.6f} rad")
            print(f"R² Delta: {row['r2_delta']:.4f}")
            print(f"Epochs: {int(row['epochs'])}")

            # Get n_samples from config
            config_path = base / exp_id / "config.yaml"
            n_samples = None
            if config_path.exists():
                import yaml

                with open(config_path) as f:
                    config = yaml.safe_load(f)
                n_samples = config.get("data", {}).get("generation", {}).get("n_samples")

            print(f"n_samples: {n_samples}")
            print(f"\nData directory: {preprocessed_dir}")

            print("\n" + "=" * 70)
            print("COMMAND TO RUN LAMBDA ABLATION STUDY")
            print("=" * 70)
            print("\nCopy and paste this command:\n")
            print(
                f"python scripts/run_lambda_ablation.py \\\n"
                f"  --config configs/experiments/hyperparameter_tuning.yaml \\\n"
                f"  --skip-data-generation \\\n"
                f"  --data-dir {preprocessed_dir} \\\n"
                f"  --n-samples {n_samples} \\\n"
                f"  --epochs {int(row['epochs'])} \\\n"
                f"  --lambda-values 0.1 0.5 1.0 \\\n"
                f"  --include-adaptive"
            )
            print("\n" + "=" * 70)

            return {
                "experiment_id": exp_id,
                "data_dir": str(preprocessed_dir),
                "n_samples": n_samples,
                "epochs": int(row["epochs"]),
                "rmse_delta": row["rmse_delta"],
                "r2_delta": row["r2_delta"],
            }

    print("\n❌ No experiments with preprocessed data found!")
    return None


if __name__ == "__main__":
    find_best_experiment_with_data()
