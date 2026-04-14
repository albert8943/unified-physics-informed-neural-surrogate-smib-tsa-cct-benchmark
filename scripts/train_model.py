#!/usr/bin/env python
"""
Model Training Script

Train PINN models on generated data with flexible configuration options.

Usage:
    # With existing data (directory or file)
    python scripts/train_model.py --data-path data/generated/quick_test/parameter_sweep_data_*.csv --epochs 100
    python scripts/train_model.py --data-dir data/generated/quick_test --epochs 100

    # With custom config
    python scripts/train_model.py --config configs/training/trajectory_config.yaml --data-path data/...
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import main as run_experiment_main
from scripts.core.utils import load_config, save_config


def create_default_config(output_path: Path, **overrides):
    """
    Create default configuration with optional overrides.

    Parameters:
    -----------
    output_path : Path
        Path to save config file
    **overrides : dict
        Configuration overrides
    """
    default_config = {
        "reproducibility": {
            "random_seed": 42,
        },
        "data": {
            "task": "trajectory",
            "generation": {
                "case_file": "smib/SMIB.json",
                "parameter_ranges": {
                    "H": [2.0, 10.0, 3],  # Reduced for quick training
                    "D": [0.5, 3.0, 3],
                    # NEW: Load variation (recommended)
                    "load": [0.6, 0.9, 3],  # Active power load (p0): [min, max, num_points]
                    "load_q": null,  # Reactive power (q0): null = fixed at 0
                    "use_load_variation": True,  # Enable load variation mode
                    # OLD: Pm variation (alternative - commented out)
                    # "Pm": [0.6, 0.8, 2],
                },
                "sampling_strategy": "full_factorial",
                "use_cct_based_sampling": False,  # Faster without CCT finding
                "fault": {
                    "start_time": 1.0,
                    "bus": 3,
                    "reactance": 0.0001,
                },
                "simulation_time": 5.0,
                "time_step": 0.001,
                "clearing_times": [1.15, 1.20, 1.25],
            },
            "validation": {
                "physics_validation": True,
                "strict_validation": False,
            },
        },
        "model": {
            # Input method: "reactance" (11 dims) or "pe_direct" (7 dims)
            # Reactance-based: [t, δ₀, ω₀, H, D, Pm, Xprefault, Xfault, Xpostfault, tf, tc]
            # Pe-based: [t, δ₀, ω₀, H, D, Pm, Pe(t)]
            "input_method": "reactance",  # "reactance" or "pe_direct"
            "input_dim": 11,  # Used for reactance-based (7 for pe_direct is auto-set)
            "hidden_dims": [64, 64, 64],
            "output_dim": 2,
            "activation": "tanh",
            "use_residual": False,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 50,
            "batch_size": None,  # None = adaptive (calculated from dataset size)
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "device": "auto",
        },
        "loss": {
            "lambda_data": 1.0,
            "lambda_physics": 0.1,
            "lambda_ic": 10.0,
        },
        "evaluation": {
            "run_baselines": False,
            "metrics": ["rmse", "mae"],
        },
    }

    # Apply overrides
    if "epochs" in overrides:
        default_config["training"]["epochs"] = overrides["epochs"]
    if "batch_size" in overrides:
        default_config["training"]["batch_size"] = overrides["batch_size"]
    if "learning_rate" in overrides:
        default_config["training"]["learning_rate"] = overrides["learning_rate"]
    if "input_method" in overrides:
        default_config["model"]["input_method"] = overrides["input_method"]
    if "data_dir" in overrides:
        # If data_dir is provided, skip data generation
        pass  # Will be handled by --skip-data-generation flag

    save_config(default_config, output_path)
    return default_config


def main():
    """Model training main function."""
    parser = argparse.ArgumentParser(
        description="Train PINN model on generated data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with data file (reactance-based, default)
  python scripts/train_model.py --data-path data/generated/quick_test/parameter_sweep_data_*.csv
  
  # Pe-based approach (7 input dimensions)
  python scripts/train_model.py --data-path data/.../file.csv --input-method pe_direct
  
  # Custom training parameters
  python scripts/train_model.py --data-path data/.../file.csv --epochs 200 --batch-size 16
  
  # With data directory
  python scripts/train_model.py --data-dir data/generated/quick_test --epochs 100
        """,
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (scenarios per batch). If not specified, auto-calculated from dataset (default: auto)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with existing data (skips data generation)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Direct path to CSV data file (alternative to --data-dir, skips data generation)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments",
        help="Output directory (default: outputs/experiments)",
    )
    parser.add_argument(
        "--input-method",
        type=str,
        choices=["reactance", "pe_direct"],
        default="reactance",
        help="Input method: 'reactance' (11 dims) or 'pe_direct' (7 dims) (default: reactance)",
    )

    args = parser.parse_args()

    # Create temporary config file
    config_path = Path("configs/experiments/quick_train_temp.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Batch size: None means adaptive (will be calculated in training.py)
    batch_size = args.batch_size if args.batch_size is not None else None

    create_default_config(
        config_path,
        epochs=args.epochs,
        batch_size=batch_size,  # None = adaptive, will be calculated
        learning_rate=args.learning_rate,
        input_method=args.input_method,
        data_dir=args.data_dir,
    )

    print("=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    if batch_size is None:
        print("Batch size: auto (will be calculated from dataset size)")
    else:
        print(f"Batch size: {batch_size} (user-specified)")
    print(f"Learning rate: {args.learning_rate}")
    input_dims = 7 if args.input_method == "pe_direct" else 11
    print(f"Input method: {args.input_method} ({input_dims} dimensions)")
    if args.data_dir:
        print(f"Using existing data: {args.data_dir}")
    print()

    # Prepare arguments for run_experiment
    import sys

    original_argv = sys.argv
    sys.argv = [
        "scripts/run_experiment.py",
        "--config",
        str(config_path),
        "--output-dir",
        args.output_dir,
    ]

    # Handle data input: either --data-dir (directory) or --data-path (file)
    if args.data_path:
        # If data-path is provided, use it directly
        data_path = Path(args.data_path)
        if not data_path.exists():
            print(f"❌ Error: Data file not found: {data_path}")
            sys.exit(1)
        # For run_experiment, we need to pass the directory containing the file
        # and the filename pattern, or modify run_experiment to accept direct file path
        # For now, use data-dir approach but point to parent directory
        data_dir = data_path.parent
        sys.argv.extend(["--skip-data-generation", "--data-dir", str(data_dir)])
        print(f"Using data file: {data_path}")
    elif args.data_dir:
        sys.argv.extend(["--skip-data-generation", "--data-dir", args.data_dir])

    try:
        run_experiment_main()
    finally:
        sys.argv = original_argv
        # Clean up temp config
        if config_path.exists():
            config_path.unlink()


if __name__ == "__main__":
    main()
