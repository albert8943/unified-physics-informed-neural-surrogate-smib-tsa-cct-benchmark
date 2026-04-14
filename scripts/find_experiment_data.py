#!/usr/bin/env python
"""
Find experiments with reusable data for lambda ablation study.

Usage:
    python scripts/find_experiment_data.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def find_experiments_with_data():
    """Find all experiments that have data directories with CSV files."""
    experiments_dir = PROJECT_ROOT / "outputs" / "experiments"

    if not experiments_dir.exists():
        print("❌ No experiments directory found")
        return []

    experiments_with_data = []

    for exp_dir in sorted(experiments_dir.glob("exp_*"), reverse=True):
        if not exp_dir.is_dir():
            continue

        data_dir = exp_dir / "data"
        if not data_dir.exists():
            continue

        # Check for CSV files
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            continue

        # Load config to get n_samples and epochs
        config_path = exp_dir / "config.yaml"
        n_samples = None
        epochs = None
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                n_samples = config.get("data", {}).get("generation", {}).get("n_samples")
                epochs = config.get("training", {}).get("epochs")
            except Exception:
                pass

        experiments_with_data.append(
            {
                "experiment_id": exp_dir.name,
                "data_dir": str(data_dir),
                "n_samples": n_samples,
                "epochs": epochs,
                "csv_files": [f.name for f in csv_files],
            }
        )

    return experiments_with_data


def main():
    print("=" * 70)
    print("FINDING EXPERIMENTS WITH REUSABLE DATA")
    print("=" * 70)

    experiments = find_experiments_with_data()

    if not experiments:
        print("\n❌ No experiments with data directories found")
        print("\nTo generate data for lambda ablation study:")
        print("  python scripts/run_lambda_ablation.py \\")
        print("    --config configs/experiments/hyperparameter_tuning.yaml \\")
        print("    --n-samples 20 --epochs 300 \\")
        print("    --lambda-values 0.1 0.5 1.0 --include-adaptive")
        return

    print(f"\n✓ Found {len(experiments)} experiment(s) with data:\n")

    print(f"{'Experiment ID':<25} {'n_samples':<12} {'epochs':<8} {'Data Dir':<40}")
    print("-" * 90)

    for exp in experiments:
        exp_id = exp["experiment_id"]
        n_samples = exp["n_samples"] if exp["n_samples"] else "?"
        epochs = exp["epochs"] if exp["epochs"] else "?"
        data_dir = exp["data_dir"]

        print(f"{exp_id:<25} {str(n_samples):<12} {str(epochs):<8} {data_dir:<40}")

    # Recommend best one (prefer n_samples=20, epochs=300)
    print("\n" + "=" * 70)
    print("RECOMMENDED FOR LAMBDA ABLATION STUDY")
    print("=" * 70)

    # Find best match (n_samples=20, epochs=300)
    best_match = None
    for exp in experiments:
        if exp["n_samples"] == 20 and exp["epochs"] == 300:
            best_match = exp
            break

    if not best_match:
        # Fall back to any with n_samples=20
        for exp in experiments:
            if exp["n_samples"] == 20:
                best_match = exp
                break

    if not best_match:
        # Use first one
        best_match = experiments[0]

    print(f"\n✓ Recommended: {best_match['experiment_id']}")
    print(f"  n_samples: {best_match['n_samples']}")
    print(f"  epochs: {best_match['epochs']}")
    print(f"  Data directory: {best_match['data_dir']}")

    print("\n" + "=" * 70)
    print("COMMAND TO RUN LAMBDA ABLATION STUDY")
    print("=" * 70)
    print("\nCopy and paste this command:\n")
    print(f"python scripts/run_lambda_ablation.py \\")
    print(f"  --config configs/experiments/hyperparameter_tuning.yaml \\")
    print(f"  --skip-data-generation \\")
    print(f"  --data-dir {best_match['data_dir']} \\")
    print(f"  --n-samples {best_match['n_samples']} \\")
    print(f"  --epochs {best_match['epochs']} \\")
    print(f"  --lambda-values 0.1 0.5 1.0 \\")
    print(f"  --include-adaptive")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
