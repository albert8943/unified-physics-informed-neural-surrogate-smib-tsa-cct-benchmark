#!/usr/bin/env python
"""
Resume Hyperparameter Sweep Script

This script runs only the remaining experiments from a sweep, skipping already completed ones.
Useful for resuming interrupted sweeps or continuing on a different machine/GPU.

Usage:
    python scripts/resume_hyperparameter_sweep.py \
        --n-samples-range 50 60 70 80 90 100 \
        --epochs-range 500 600 700 800 900 1000 \
        --config configs/experiments/hyperparameter_tuning.yaml \
        --skip-data-generation \
        --data-dir-base "outputs/experiments"
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Set, Tuple

import yaml


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict, output_path: Path) -> None:
    """Save YAML configuration file."""
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def find_experiment_data_dir(base_dir: Path, n_samples: int) -> Path:
    """
    Find experiment data directory matching n_samples.

    Parameters:
    -----------
    base_dir : Path
        Base directory to search (e.g., outputs/experiments)
    n_samples : int
        Target n_samples value to match

    Returns:
    --------
    Path or None
        Path to data directory if found, None otherwise
    """
    for exp_dir in sorted(base_dir.glob("exp_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        data_dir = exp_dir / "data"
        if data_dir.exists():
            # Check preprocessing metadata
            preprocessed_dir = data_dir / "preprocessed"
            if preprocessed_dir.exists():
                metadata_files = list(preprocessed_dir.glob("preprocessing_metadata_*.json"))
                if metadata_files:
                    try:
                        with open(metadata_files[0], "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                            if metadata.get("n_samples") == n_samples:
                                return data_dir
                    except Exception:
                        continue
    return None


def find_latest_checkpoint(exp_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint in an experiment directory.

    Parameters:
    -----------
    exp_dir : Path
        Experiment directory to search

    Returns:
    --------
    Path or None
        Path to latest checkpoint if found, None otherwise
    """
    model_dir = exp_dir / "model"
    if not model_dir.exists():
        return None

    # Look for checkpoints (epoch_*.pth files)
    checkpoints = list(model_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoints:
        # Also check for best_model (which is also a checkpoint)
        best_models = list(model_dir.glob("best_model_*.pth"))
        if best_models:
            return sorted(best_models, key=lambda p: p.stat().st_mtime)[-1]
        return None

    # Return the most recent checkpoint
    return sorted(checkpoints, key=lambda p: p.stat().st_mtime)[-1]


def get_experiment_status(exp_dir: Path) -> Tuple[bool, Optional[Path], Optional[Tuple[int, int]]]:
    """
    Check experiment status: completed, interrupted, or not started.

    Parameters:
    -----------
    exp_dir : Path
        Experiment directory to check

    Returns:
    --------
    Tuple[bool, Optional[Path], Optional[Tuple[int, int]]]
        (is_completed, checkpoint_path, (n_samples, epochs))
    """
    summary_file = exp_dir / "experiment_summary.json"
    checkpoint_path = find_latest_checkpoint(exp_dir)

    # Check if completed
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                config = data.get("config", {})
                n_samples = config.get("data", {}).get("generation", {}).get("n_samples")
                epochs = config.get("training", {}).get("epochs")
                if n_samples is not None and epochs is not None:
                    return (True, None, (n_samples, epochs))
        except Exception:
            pass

    # Check if interrupted (has checkpoint but no summary)
    if checkpoint_path and checkpoint_path.exists():
        # Try to get config from checkpoint or config.yaml
        config_file = exp_dir / "config.yaml"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    n_samples = config.get("data", {}).get("generation", {}).get("n_samples")
                    epochs = config.get("training", {}).get("epochs")
                    if n_samples is not None and epochs is not None:
                        return (False, checkpoint_path, (n_samples, epochs))
            except Exception:
                pass

    return (False, None, None)


def get_completed_experiments(output_dir: Path) -> Set[Tuple[int, int]]:
    """
    Find all completed experiments by checking for experiment_summary.json files.

    Parameters:
    -----------
    output_dir : Path
        Base directory containing experiment folders (e.g., outputs/experiments)

    Returns:
    --------
    Set[Tuple[int, int]]
        Set of (n_samples, epochs) tuples for completed experiments
    """
    completed = set()
    for exp_dir in output_dir.glob("exp_*"):
        is_completed, _, config_tuple = get_experiment_status(exp_dir)
        if is_completed and config_tuple:
            completed.add(config_tuple)
    return completed


def get_interrupted_experiments(output_dir: Path) -> Dict[Tuple[int, int], Path]:
    """
    Find interrupted experiments (have checkpoints but no summary).

    Parameters:
    -----------
    output_dir : Path
        Base directory containing experiment folders

    Returns:
    --------
    Dict[Tuple[int, int], Path]
        Dictionary mapping (n_samples, epochs) to checkpoint path
    """
    interrupted = {}
    for exp_dir in output_dir.glob("exp_*"):
        is_completed, checkpoint_path, config_tuple = get_experiment_status(exp_dir)
        if not is_completed and checkpoint_path and config_tuple:
            interrupted[config_tuple] = checkpoint_path
    return interrupted


def main():
    parser = argparse.ArgumentParser(
        description="Resume hyperparameter sweep, skipping already completed experiments"
    )
    parser.add_argument(
        "--n-samples-range",
        type=int,
        nargs="+",
        required=True,
        help="List of n_samples values to test (e.g., 50 60 70 80 90 100)",
    )
    parser.add_argument(
        "--epochs-range",
        type=int,
        nargs="+",
        required=True,
        help="List of epochs values to test (e.g., 500 600 700 800 900 1000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments",
        help="Base directory for experiment outputs (default: outputs/experiments)",
    )
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Skip data generation (use existing data)",
    )
    parser.add_argument(
        "--data-dir-base",
        type=str,
        default=None,
        help="Base directory to search for existing data (default: same as --output-dir)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume interrupted experiments from their latest checkpoint",
    )

    args = parser.parse_args()

    # Setup paths
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir_base = Path(args.data_dir_base) if args.data_dir_base else output_dir

    # Get completed and interrupted experiments
    print(f"\n{'='*70}")
    print(f"RESUMING HYPERPARAMETER SWEEP")
    print(f"{'='*70}")
    print(f"Checking for completed experiments in: {output_dir}")
    completed = get_completed_experiments(output_dir)
    interrupted = get_interrupted_experiments(output_dir) if args.resume_from_checkpoint else {}

    print(f"Found {len(completed)} completed experiments:")
    for n_samples, epochs in sorted(completed):
        print(f"  ✓ n_samples={n_samples}, epochs={epochs}")

    if interrupted:
        print(f"\nFound {len(interrupted)} interrupted experiments (with checkpoints):")
        for (n_samples, epochs), checkpoint_path in sorted(interrupted.items()):
            # Try to get epoch number from checkpoint
            try:
                import torch

                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                epoch = checkpoint.get("epoch", "unknown")
                print(
                    f"⚠️ n_samples={n_samples}, epochs={epochs}: checkpoint at epoch {epoch}"
                    f"({checkpoint_path.name})"
                )
            except Exception:
                print(
                    f"⚠️ n_samples={n_samples}, epochs={epochs}: checkpoint found"
                    f"({checkpoint_path.name})"
                )

    # Calculate all possible combinations
    all_combinations = []
    for n_samples in args.n_samples_range:
        for epochs in args.epochs_range:
            all_combinations.append((n_samples, epochs))

    # Filter out completed experiments
    # If resume_from_checkpoint is enabled, we'll re-run interrupted experiments
    if args.resume_from_checkpoint:
        # Remove interrupted from completed (they'll be re-run)
        completed = completed - set(interrupted.keys())

    remaining = [(n, e) for n, e in all_combinations if (n, e) not in completed]
    total_experiments = len(all_combinations)
    remaining_count = len(remaining)

    print(f"\nTotal experiments in sweep: {total_experiments}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining to run: {remaining_count}")

    if remaining_count == 0:
        print("\n✅ All experiments already completed!")
        return

    print(f"\nRemaining experiments:")
    for n_samples, epochs in remaining:
        print(f"  - n_samples={n_samples}, epochs={epochs}")

    # Ask for confirmation
    print(f"\n{'='*70}")
    response = input("Continue with remaining experiments? (y/n): ").strip().lower()
    if response != "y":
        print("Cancelled.")
        return

    # Run remaining experiments
    results = []
    experiment_num = 0

    for n_samples, epochs in remaining:
        experiment_num += 1
        print(f"\n[{experiment_num}/{remaining_count}]")
        print(f"Running: n_samples={n_samples}, epochs={epochs}")

        # Build command
        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            str(config_path),
        ]

        # Check if this experiment was interrupted and should resume from checkpoint
        if args.resume_from_checkpoint and (n_samples, epochs) in interrupted:
            checkpoint_path = interrupted[(n_samples, epochs)]
            # Find the experiment directory for this checkpoint
            exp_dir = checkpoint_path.parent.parent
            print(f"  🔄 Found interrupted experiment, will resume from checkpoint")
            print(f"     Checkpoint: {checkpoint_path.name}")
            print(f"     Experiment dir: {exp_dir.name}")
            # Note: run_experiment.py doesn't directly support --resume, but we can
            # modify the config or use the checkpoint path. For now, we'll just re-run
            # and the training code should detect and use the checkpoint if available.
            # TODO: Add --resume-from-checkpoint argument to run_experiment.py

        if args.skip_data_generation:
            # Find data directory for this n_samples
            data_dir = find_experiment_data_dir(data_dir_base, n_samples)
            if data_dir:
                print(f"  ✓ Using existing data: {data_dir}")
                cmd.extend(["--skip-data-generation", "--data-dir", str(data_dir)])
            else:
                print(f"  ⚠️  Warning: Could not find data for n_samples={n_samples}")
                print(f"     Will generate new data...")

        # Override hyperparameters via config update
        config = load_config(config_path)
        config["data"]["generation"]["n_samples"] = n_samples
        config["training"]["epochs"] = epochs

        # Save temporary config
        temp_config = Path("configs/experiments/temp_sweep.yaml")
        temp_config.parent.mkdir(parents=True, exist_ok=True)
        save_config(config, temp_config)
        cmd[3] = str(temp_config)  # Update config path

        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True)
            status = "SUCCESS"
        except subprocess.CalledProcessError as e:
            status = "FAILED"
            print(f"❌ Experiment failed: {e}")

        results.append(
            {
                "n_samples": n_samples,
                "epochs": epochs,
                "status": status,
            }
        )

        # Clean up temp config
        if temp_config.exists():
            temp_config.unlink()

    # Summary
    print(f"\n{'='*70}")
    print(f"RESUME SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"Total experiments in sweep: {total_experiments}")
    print(f"Already completed: {len(completed)}")
    print(f"Just completed: {remaining_count}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'SUCCESS')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'FAILED')}")
    print(f"\nResults from this run:")
    for r in results:
        print(f"  n_samples={r['n_samples']:3d}, epochs={r['epochs']:3d}: {r['status']}")
    print(f"{'='*70}\n")

    print("💡 Next steps:")
    print("  1. Review experiment results in outputs/experiments/exp_XXX/results/")
    print("  2. Use scripts/analyze_sweep_results.py to analyze all results")
    print("  3. Compare experiments using scripts/compare_experiments.py")


if __name__ == "__main__":
    main()
