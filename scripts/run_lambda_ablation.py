#!/usr/bin/env python
"""
Lambda Physics Ablation Study Script.

Tests different lambda_physics values to find optimal physics constraint weighting.
Compares fixed values (0.1, 0.5, 1.0) and adaptive scheduling.

Usage:
    python scripts/run_lambda_ablation.py --config configs/experiments/hyperparameter_tuning.yaml
    python scripts/run_lambda_ablation.py --config configs/experiments/hyperparameter_tuning.yaml --skip-data-generation --data-dir outputs/experiments/exp_20251208_111254/data
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.utils import load_config, generate_timestamped_filename


def run_experiment(
    config_path: Path,
    lambda_physics: float,
    use_adaptive: bool,
    data_dir: Path = None,
    n_samples: int = 20,
    epochs: int = 300,
):
    """
    Run a single experiment with specified lambda_physics value.

    Parameters:
    -----------
    config_path : Path
        Path to base config file
    lambda_physics : float
        Physics loss weight (ignored if use_adaptive=True)
    use_adaptive : bool
        Whether to use adaptive scheduling
    data_dir : Path, optional
        Data directory to reuse (skip data generation)
    n_samples : int
        Number of samples for data generation
    epochs : int
        Number of training epochs
    """
    print(f"\n{'='*70}")
    print(f"Running experiment: lambda_physics={lambda_physics}, adaptive={use_adaptive}")
    print(f"{'='*70}")

    # Load base config
    config = load_config(config_path)

    # Modify config for this experiment
    config["data"]["generation"]["n_samples"] = n_samples
    config["training"]["epochs"] = epochs

    if use_adaptive:
        # For adaptive, set base lambda_physics to 0.1 (will be scaled by scheduler)
        config["loss"]["lambda_physics"] = 0.1
        config["loss"].pop("use_fixed_lambda", None)  # Remove if exists, use default (False)
        experiment_name = f"lambda_adaptive"
    else:
        # For fixed lambda, set use_fixed_lambda=true
        config["loss"]["lambda_physics"] = lambda_physics
        config["loss"]["use_fixed_lambda"] = True
        experiment_name = f"lambda_{lambda_physics:.1f}"

    # Create temporary config file
    temp_config_dir = PROJECT_ROOT / "configs" / "temp_ablation"
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_config_dir / f"lambda_ablation_{experiment_name}.yaml"

    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created temporary config: {temp_config_path}")

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

    # Run experiment
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True,
        )
        print(f"✓ Experiment completed: {experiment_name}")

        # Clean up temp config (optional, keep for debugging)
        # temp_config_path.unlink()

        return True, experiment_name
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed: {experiment_name}")
        print(f"Error: {e}")
        return False, experiment_name


def main():
    parser = argparse.ArgumentParser(
        description="Run lambda_physics ablation study",
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
        default=20,
        help="Number of samples for data generation (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (default: 300)",
    )
    parser.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
        help="Lambda physics values to test (default: 0.1 0.5 1.0)",
    )
    parser.add_argument(
        "--include-adaptive",
        action="store_true",
        help="Include adaptive scheduling test",
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

    # Load config to get n_samples and epochs if not provided
    config = load_config(config_path)
    n_samples = args.n_samples or config.get("data", {}).get("generation", {}).get("n_samples", 20)
    epochs = args.epochs or config.get("training", {}).get("epochs", 300)

    print("=" * 70)
    print("LAMBDA PHYSICS ABLATION STUDY")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"n_samples: {n_samples}")
    print(f"epochs: {epochs}")
    print(f"Lambda values to test: {args.lambda_values}")
    if args.include_adaptive:
        print("Including adaptive scheduling test")
    if data_dir:
        print(f"Reusing data from: {data_dir}")
    print("=" * 70)

    # Run experiments
    results = []
    experiments = []

    # Fixed lambda values
    for lambda_p in args.lambda_values:
        success, exp_name = run_experiment(
            config_path,
            lambda_p,
            use_adaptive=False,
            data_dir=data_dir,
            n_samples=n_samples,
            epochs=epochs,
        )
        experiments.append(
            {"name": exp_name, "lambda_physics": lambda_p, "adaptive": False, "success": success}
        )
        results.append((exp_name, success))

    # Adaptive scheduling
    if args.include_adaptive:
        success, exp_name = run_experiment(
            config_path,
            0.1,
            use_adaptive=True,  # Base value, scheduler scales it
            data_dir=data_dir,
            n_samples=n_samples,
            epochs=epochs,
        )
        experiments.append(
            {"name": exp_name, "lambda_physics": "adaptive", "adaptive": True, "success": success}
        )
        results.append((exp_name, success))

    # Summary
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    for exp_name, success in results:
        status = "✓ SUCCESS" if success else "❌ FAILED"
        print(f"{exp_name:30s} {status}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Compare results using:")
    print("   python scripts/analyze_lambda_ablation.py --experiments outputs/experiments/exp_*")
    print("\n2. Generate comparison figures:")
    print(
        "   python scripts/analyze_lambda_ablation.py --experiments outputs/experiments/exp_* --plot"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
