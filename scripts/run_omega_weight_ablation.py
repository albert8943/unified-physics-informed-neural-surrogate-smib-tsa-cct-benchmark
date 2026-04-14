#!/usr/bin/env python
"""
Omega Weight Ablation Study Script.

Tests different omega weights in normalized loss to find optimal value for omega predictions.
Compares omega weights: 100 (baseline), 200, 300, 500, etc.

Usage:
    python scripts/run_omega_weight_ablation.py --config configs/experiments/hyperparameter_tuning.yaml
    python scripts/run_omega_weight_ablation.py --config configs/experiments/hyperparameter_tuning.yaml --skip-data-generation --data-dir outputs/experiments/exp_20251208_133918/data
"""

import argparse
import os
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
    omega_weight: float,
    data_dir: Path = None,
    n_samples: int = 30,
    epochs: int = 300,
):
    """
    Run a single experiment with specified omega weight.

    Parameters:
    -----------
    config_path : Path
        Base configuration file path
    omega_weight : float
        Omega weight in scale_to_norm [delta_weight, omega_weight]
    data_dir : Path, optional
        Data directory to reuse (if skip-data-generation)
    n_samples : int
        Number of samples (for logging)
    epochs : int
        Number of epochs (for logging)

    Returns:
    --------
    str : Experiment ID if successful, None otherwise
    """
    # Load base config
    config = load_config(config_path)

    # Update omega weight in loss config
    if "loss" not in config:
        config["loss"] = {}

    config["loss"]["use_normalized_loss"] = True
    config["loss"]["scale_to_norm"] = [1.0, omega_weight]  # [delta, omega]

    # Create temporary config file
    temp_config = Path(
        "configs/temp_ablation/omega_weight_ablation_omega_{}.yaml".format(omega_weight)
    )
    temp_config.parent.mkdir(parents=True, exist_ok=True)

    with open(temp_config, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Build command with -u flag for unbuffered output
    cmd = [
        sys.executable,
        "-u",  # Unbuffered output
        "scripts/run_experiment.py",
        "--config",
        str(temp_config),
    ]

    if data_dir:
        cmd.extend(["--skip-data-generation", "--data-dir", str(data_dir)])

    print(f"\n{'='*70}", flush=True)
    print(
        f"Running experiment: omega_weight={omega_weight}, n_samples={n_samples}, epochs={epochs}",
        flush=True,
    )
    print(f"{'='*70}", flush=True)

    try:
        # Get list of existing experiments before running
        base_experiments_dir = Path("outputs/experiments")
        existing_experiments = (
            set(base_experiments_dir.glob("exp_*")) if base_experiments_dir.exists() else set()
        )

        # Set environment to ensure unbuffered output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # Use Popen for real-time streaming on Windows
        # This ensures output appears immediately, not just on interrupt
        process = subprocess.Popen(
            cmd,
            stdout=None,  # Inherit stdout (shows in terminal)
            stderr=None,  # Inherit stderr (shows in terminal)
            env=env,
            cwd=PROJECT_ROOT,
            bufsize=0,  # Unbuffered
        )

        # Wait for process to complete, but output streams directly to terminal
        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

        # Find the new experiment directory (most recent one not in our list)
        if base_experiments_dir.exists():
            new_experiments = set(base_experiments_dir.glob("exp_*")) - existing_experiments
            if new_experiments:
                # Get the most recently created one
                experiment_id = max(new_experiments, key=lambda p: p.stat().st_mtime).name
                print(f"\n[OK] Experiment completed: {experiment_id} (omega_weight={omega_weight})")
                return experiment_id

        # Fallback if we can't find it
        print(f"\n[OK] Experiment completed: omega_weight={omega_weight}")
        return f"exp_omega_{omega_weight}"
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Experiment failed: omega_weight={omega_weight}")
        print(f"Exit code: {e.returncode}")
        return None
    finally:
        # Clean up temp config
        if temp_config.exists():
            temp_config.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Run omega weight ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test omega weights: 100, 200, 300, 500
  python scripts/run_omega_weight_ablation.py --config configs/experiments/hyperparameter_tuning.yaml --omega-weights 100 200 300 500

  # Use existing data (faster)
  python scripts/run_omega_weight_ablation.py --config configs/experiments/hyperparameter_tuning.yaml --skip-data-generation --data-dir outputs/experiments/exp_20251208_133918/data/preprocessed --omega-weights 200 300 500

  # Default: Test 100, 200, 300, 500
  python scripts/run_omega_weight_ablation.py --config configs/experiments/hyperparameter_tuning.yaml --skip-data-generation --data-dir outputs/experiments/exp_20251208_133918/data/preprocessed
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/hyperparameter_tuning.yaml",
        help="Base configuration file",
    )
    parser.add_argument(
        "--omega-weights",
        type=float,
        nargs="+",
        default=[100.0, 200.0, 300.0, 500.0],
        help="Omega weights to test (default: 100 200 300 500)",
    )
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Skip data generation (use existing data)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (optional if --skip-data-generation, will auto-find if not provided)",
    )
    parser.add_argument(
        "--data-dir-base",
        type=str,
        default="outputs/experiments",
        help="Base directory to search for experiment data (default: outputs/experiments)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="n_samples value (for logging, extracted from config if not provided)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="epochs value (for logging, extracted from config if not provided)",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return

    # Load config to get n_samples and epochs
    config = load_config(config_path)
    n_samples = args.n_samples or config.get("data", {}).get("generation", {}).get("n_samples", 30)
    epochs = args.epochs or config.get("training", {}).get("epochs", 300)

    data_dir = None
    if args.skip_data_generation:
        if args.data_dir:
            data_dir = Path(args.data_dir)
            if not data_dir.exists():
                print(f"[ERROR] Data directory not found: {data_dir}")
                # Try to auto-find
                print("  Attempting to auto-find data directory...")
                data_dir = None
        else:
            print("[WARNING] --data-dir not provided, attempting to auto-find...")

        # Auto-find data directory if not provided or not found
        if data_dir is None:
            from scripts.run_hyperparameter_sweep import find_experiment_data_dir

            base_dir = Path(args.data_dir_base)
            found_data_dir = find_experiment_data_dir(base_dir, n_samples)
            if found_data_dir:
                # Check if it's a preprocessed directory or needs to point to preprocessed
                preprocessed_dir = found_data_dir / "preprocessed"
                if preprocessed_dir.exists():
                    data_dir = preprocessed_dir
                    print(f"[OK] Auto-found preprocessed data directory: {data_dir}")
                else:
                    data_dir = found_data_dir
                    print(f"[OK] Auto-found data directory: {data_dir}")
            else:
                print(f"[ERROR] Could not find data directory for n_samples={n_samples}")
                print(f"   Searched in: {base_dir}")
                print(f"\nOptions:")
                print(f"   1. Generate new data (remove --skip-data-generation)")
                print(f"   2. Specify --data-dir with correct path")
                print(f"   3. Check if data exists in: data/generated/ or data/processed/")
                return

    print("=" * 70)
    print("OMEGA WEIGHT ABLATION STUDY")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Omega weights to test: {args.omega_weights}")
    print(f"n_samples: {n_samples}")
    print(f"epochs: {epochs}")
    if data_dir:
        print(f"Data directory: {data_dir}")
    print("=" * 70)

    # Run experiments
    results = []
    for omega_weight in args.omega_weights:
        experiment_id = run_experiment(
            config_path=config_path,
            omega_weight=omega_weight,
            data_dir=data_dir,
            n_samples=n_samples,
            epochs=epochs,
        )
        results.append(
            {
                "omega_weight": omega_weight,
                "experiment_id": experiment_id,
                "status": "SUCCESS" if experiment_id else "FAILED",
            }
        )

    # Summary
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'SUCCESS')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'FAILED')}")
    print("\nResults:")
    for r in results:
        status_icon = "[OK]" if r["status"] == "SUCCESS" else "[ERROR]"
        print(
            f"  {status_icon} omega_weight={r['omega_weight']:6.1f}: {r['experiment_id'] or 'FAILED'}"
        )

    # Save results summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(
        "outputs/ablation_studies/omega_weight_ablation_summary_{}.txt".format(timestamp)
    )
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, "w") as f:
        f.write("OMEGA WEIGHT ABLATION STUDY SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"n_samples: {n_samples}\n")
        f.write(f"epochs: {epochs}\n")
        f.write("\nResults:\n")
        for r in results:
            f.write(f"  omega_weight={r['omega_weight']:6.1f}: {r['experiment_id'] or 'FAILED'}\n")

    print(f"\n[OK] Summary saved to: {summary_file}")

    # Automatically run analysis if experiments completed
    successful_experiments = [r["experiment_id"] for r in results if r["status"] == "SUCCESS"]
    if len(successful_experiments) >= 2:
        print("\n" + "=" * 70)
        print("RUNNING AUTOMATIC ANALYSIS")
        print("=" * 70)
        try:
            output_dir = Path("outputs/ablation_studies")
            analysis_cmd = [
                sys.executable,
                "scripts/analyze_omega_weight_ablation.py",
                "--experiments",
            ] + [f"outputs/experiments/{exp_id}" for exp_id in successful_experiments]
            analysis_cmd.append("--output-dir")
            analysis_cmd.append(str(output_dir))

            subprocess.run(analysis_cmd, check=True)
            print("\n[OK] Analysis complete! Check outputs/ablation_studies/ for figures")
        except Exception as e:
            print(f"\n[WARNING] Automatic analysis failed: {e}")
            print("   You can run analysis manually:")
            print(
                f"python scripts/analyze_omega_weight_ablation.py --experiments {' '.join(successful_experiments)}"
            )

    print("\nNext steps:")
    print("  1. View comparison figures in: outputs/ablation_studies/")
    print("  2. Analyze omega R² improvement vs delta RMSE trade-off")
    print("  3. Select optimal omega weight based on results")


if __name__ == "__main__":
    main()
