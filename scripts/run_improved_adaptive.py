#!/usr/bin/env python
"""
Run only the improved adaptive lambda experiment.

This script runs just the adaptive experiment with the new improvements:
- Loss normalization (Option 4)
- Gradual increase (step-by-step)

Usage:
    python scripts/run_improved_adaptive.py --config configs/experiments/hyperparameter_tuning.yaml --skip-data-generation --data-dir outputs/experiments/exp_20251208_111254/data/preprocessed
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.utils import load_config


def run_adaptive_experiment(
    config_path: Path,
    data_dir: Path = None,
    n_samples: int = 30,
    epochs: int = 300,
):
    """
    Run only the improved adaptive lambda experiment.

    Parameters:
    -----------
    config_path : Path
        Path to base config file
    data_dir : Path, optional
        Data directory to reuse (skip data generation)
    n_samples : int
        Number of samples for data generation
    epochs : int
        Number of training epochs
    """
    print("=" * 70)
    print("IMPROVED ADAPTIVE LAMBDA EXPERIMENT")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"n_samples: {n_samples}")
    print(f"epochs: {epochs}")
    print("Using improved adaptive scheduler:")
    print("  - Loss normalization: Enabled (Option 4)")
    print("  - Gradual increase: Enabled (70 epochs)")
    if data_dir:
        print(f"Reusing data from: {data_dir}")
    print("=" * 70)

    # Load base config
    config = load_config(config_path)

    # Modify config for adaptive experiment
    config["data"]["generation"]["n_samples"] = n_samples
    config["training"]["epochs"] = epochs

    # Set up for adaptive (not fixed)
    config["loss"]["lambda_physics"] = 0.1  # Base value, scheduler will scale
    config["loss"].pop("use_fixed_lambda", None)  # Remove if exists, use default (False)

    # Enable improvements (these are defaults, but explicit for clarity)
    config["training"]["adaptive_gradual_epochs"] = 70
    config["loss"]["normalize_losses_for_adaptive"] = True

    # Create temporary config file
    temp_config_dir = PROJECT_ROOT / "configs" / "temp_ablation"
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_config_dir / "improved_adaptive_lambda.yaml"

    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nCreated temporary config: {temp_config_path}")

    # Build command
    cmd = [
        sys.executable,
        "scripts/run_experiment.py",
        "--config",
        str(temp_config_path),
    ]

    if data_dir:
        cmd.extend(["--skip-data-generation", "--data-dir", str(data_dir)])

    print(f"Command: {' '.join(cmd)}")
    print("\n" + "=" * 70)

    # Run experiment
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True,
        )
        print("\n" + "=" * 70)
        print("✓ Improved adaptive experiment completed successfully!")
        print("=" * 70)
        print(f"\nExperiment directory: Check outputs/experiments/exp_*")
        print("\nTo compare with previous fixed lambda experiments:")
        print("  python scripts/analyze_lambda_ablation.py \\")
        print("    --experiments outputs/experiments/exp_20251208_132827 \\")
        print("    --experiments outputs/experiments/exp_20251208_133918 \\")
        print("    --experiments outputs/experiments/exp_20251208_134925 \\")
        print("    --experiments outputs/experiments/exp_* \\")
        print("    --plot")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment failed")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run improved adaptive lambda experiment only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base config file",
    )
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Skip data generation and reuse existing data",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory to reuse (required if --skip-data-generation)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Number of samples for data generation (default: 30)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (default: 300)",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    data_dir = Path(args.data_dir) if args.data_dir else None
    if args.skip_data_generation and not data_dir:
        print("❌ --data-dir required when --skip-data-generation is used")
        sys.exit(1)

    success = run_adaptive_experiment(
        config_path,
        data_dir=data_dir,
        n_samples=args.n_samples,
        epochs=args.epochs,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
