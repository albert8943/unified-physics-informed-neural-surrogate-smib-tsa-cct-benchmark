#!/usr/bin/env python
"""
Hyperparameter Sweep Script for Complete Experiments (PINN + ML Baselines + Comparison)

This script runs multiple complete experiments with different combinations of:
- n_samples: Number of unique parameter combinations
- epochs: Number of training epochs

Each experiment includes:
1. Data generation/reuse
2. Data analysis (ANDES)
3. PINN training
4. ML baseline training
5. Model evaluation
6. Model comparison

Usage:
    # Overnight sweep with data reuse
    python scripts/run_hyperparameter_sweep.py \
        --config configs/experiments/hyperparameter_tuning.yaml \
        --n-samples-range 10 20 30 50 \
        --epochs-range 100 200 300 \
        --skip-data-generation \
        --data-dir-base outputs/complete_experiments

    # Quick mode (skip optional steps for faster runs)
    python scripts/run_hyperparameter_sweep.py \
        --config configs/experiments/hyperparameter_tuning.yaml \
        --n-samples-range 20 30 \
        --epochs-range 100 200 \
        --skip-data-generation \
        --quick-mode \
        --data-dir-base outputs/complete_experiments
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict, output_path: Path) -> None:
    """Save YAML configuration file."""
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def find_experiment_data_dir(base_dir: Path, n_samples: int) -> Optional[Path]:
    """
    Find experiment data directory matching n_samples.

    Enhanced to work with complete_experiments structure:
    - Checks both outputs/experiments/ and outputs/complete_experiments/
    - Looks for preprocessed data in data/processed/ subdirectories
    - Prioritizes preprocessed train/test/val splits
    - Also checks common repository (data/processed/ and data/common/) for data files
    - Handles raw CSV files in data/common/ by matching n_samples in filename

    Parameters:
    -----------
    base_dir : Path
        Base directory to search (e.g., outputs/complete_experiments, data/common, data/processed)
    n_samples : int
        Target n_samples value to match

    Returns:
    --------
    Path or None
        Path to data directory if found, None otherwise
    """
    # Special handling for common repository directories
    if "common" in str(base_dir) or "processed" in str(base_dir):
        # Check for raw CSV files with n_samples in filename
        if base_dir.exists():
            # Look for files matching n_samples pattern
            csv_files = list(base_dir.glob(f"*{n_samples}*.csv"))
            if csv_files:
                # Return the directory (script will use the CSV file directly)
                return base_dir

            # Also check for processed data directories
            for exp_dir in sorted(
                base_dir.glob("exp_*"), key=lambda p: p.stat().st_mtime, reverse=True
            ):
                train_files = list(exp_dir.glob("train_data_*.csv"))
                if train_files:
                    # Try to infer n_samples from metadata or config
                    metadata_files = list(exp_dir.glob("*metadata*.json"))
                    for metadata_file in metadata_files:
                        try:
                            import json

                            metadata = json.load(open(metadata_file))
                            if metadata.get("n_samples") == n_samples:
                                return exp_dir
                        except Exception:
                            continue
                    # If no metadata, return first match (user should verify)
                    return exp_dir

        # Also check data/processed if base_dir is data/common
        if "common" in str(base_dir):
            processed_dir = base_dir.parent / "processed"
            if processed_dir.exists():
                for exp_dir in sorted(
                    processed_dir.glob("exp_*"), key=lambda p: p.stat().st_mtime, reverse=True
                ):
                    train_files = list(exp_dir.glob("train_data_*.csv"))
                    if train_files:
                        metadata_files = list(exp_dir.glob("*metadata*.json"))
                        for metadata_file in metadata_files:
                            try:
                                import json

                                metadata = json.load(open(metadata_file))
                                if metadata.get("n_samples") == n_samples:
                                    return exp_dir
                            except Exception:
                                continue

    # Search in base_dir and also check outputs/experiments as fallback
    search_dirs = [base_dir]
    if "complete_experiments" in str(base_dir):
        # Also check outputs/experiments
        experiments_dir = base_dir.parent / "experiments"
        if experiments_dir.exists():
            search_dirs.append(experiments_dir)

    # Priority 1: Check experiment directories
    for search_dir in search_dirs:
        for exp_dir in sorted(
            search_dir.glob("exp_*"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            config_file = exp_dir / "config.yaml"
            if not config_file.exists():
                continue

            try:
                exp_config = load_config(config_file)
                exp_n_samples = exp_config.get("data", {}).get("generation", {}).get("n_samples")
                if exp_n_samples == n_samples:
                    # Priority 1a: Check for preprocessed data (preferred for complete experiments)
                    processed_dir = exp_dir / "data" / "processed"
                    if processed_dir.exists():
                        train_files = list(processed_dir.glob("train_data_*.csv"))
                        if train_files:
                            return processed_dir

                    # Priority 1b: Check for raw data in data/ directory
                    data_dir = exp_dir / "data"
                    if data_dir.exists():
                        csv_files = (
                            list(data_dir.glob("parameter_sweep_data_*.csv"))
                            + list(data_dir.glob("trajectory_data_*.csv"))
                            + list(data_dir.glob("train_data_*.csv"))
                        )
                        if csv_files:
                            return data_dir
            except Exception:
                continue

    # Priority 2: Check common repository (data/processed/) for preprocessed data
    # This is where data is saved when use_common_repository=True (default behavior)
    common_processed_dir = Path("data/processed")
    if common_processed_dir.exists():
        # Search for experiment directories with matching n_samples in config
        for exp_dir in sorted(
            common_processed_dir.glob("exp_*"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            # Try to find config in the original experiment directory
            # The exp_id in data/processed should match an experiment in outputs/
            exp_id = exp_dir.name
            for search_dir in search_dirs:
                orig_exp_dir = search_dir / exp_id
                config_file = orig_exp_dir / "config.yaml"
                if config_file.exists():
                    try:
                        exp_config = load_config(config_file)
                        exp_n_samples = (
                            exp_config.get("data", {}).get("generation", {}).get("n_samples")
                        )
                        if exp_n_samples == n_samples:
                            # Check if preprocessed data exists
                            train_files = list(exp_dir.glob("train_data_*.csv"))
                            if train_files:
                                return exp_dir
                    except Exception:
                        continue
            # Also check if we can infer n_samples from the directory name or files
            # Some experiments might have metadata files
            train_files = list(exp_dir.glob("train_data_*.csv"))
            if train_files:
                # Try to load preprocessing metadata if available
                metadata_files = list(exp_dir.glob("preprocessing_metadata_*.json"))
                if metadata_files:
                    try:
                        import json

                        metadata = json.load(open(metadata_files[0]))
                        if metadata.get("n_samples") == n_samples:
                            return exp_dir
                    except Exception:
                        pass

    return None


def get_test_set_hash(data_dir: Path) -> Optional[str]:
    """
    Get a hash/identifier for the test set to ensure consistency across experiments.

    Parameters:
    -----------
    data_dir : Path
        Data directory containing preprocessed splits

    Returns:
    --------
    str or None
        Test set identifier (filename or hash) if found, None otherwise
    """
    if not data_dir.exists():
        return None

    # Look for test_data_*.csv file
    test_files = list(data_dir.glob("test_data_*.csv"))
    if test_files:
        # Use the most recent test file
        test_file = max(test_files, key=lambda p: p.stat().st_mtime)
        return test_file.name

    return None


def extract_experiment_metrics(experiment_dir: Path) -> Optional[Dict]:
    """
    Extract key metrics from an experiment directory.

    Parameters:
    -----------
    experiment_dir : Path
        Path to experiment directory

    Returns:
    --------
    dict or None
        Dictionary with extracted metrics, or None if not found
    """
    metrics = {}

    # Try to load experiment summary
    summary_path = experiment_dir / "experiment_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
                metrics["experiment_id"] = summary.get("experiment_id")
                metrics["timestamp"] = summary.get("timestamp")

                # Extract PINN metrics
                pinn_eval = summary.get("pinn", {}).get("evaluation", {})
                if pinn_eval:
                    metrics["pinn_rmse_delta"] = pinn_eval.get("metrics", {}).get("rmse_delta")
                    metrics["pinn_rmse_omega"] = pinn_eval.get("metrics", {}).get("rmse_omega")
                    metrics["pinn_r2_delta"] = pinn_eval.get("metrics", {}).get("r2_delta")
                    metrics["pinn_r2_omega"] = pinn_eval.get("metrics", {}).get("r2_omega")

                # Extract ML baseline metrics (first model type)
                ml_baseline = summary.get("ml_baseline", {})
                if ml_baseline:
                    first_model = next(iter(ml_baseline.values()), {})
                    ml_eval = first_model.get("evaluation", {})
                    if ml_eval:
                        metrics["ml_rmse_delta"] = ml_eval.get("metrics", {}).get("rmse_delta")
                        metrics["ml_rmse_omega"] = ml_eval.get("metrics", {}).get("rmse_omega")
                        metrics["ml_r2_delta"] = ml_eval.get("metrics", {}).get("r2_delta")
                        metrics["ml_r2_omega"] = ml_eval.get("metrics", {}).get("r2_omega")

                # Extract training info
                pinn_training = summary.get("pinn", {}).get("training", {})
                if pinn_training:
                    metrics["training_time"] = pinn_training.get("training_time")

        except Exception as e:
            print(f"  ⚠️  Warning: Could not extract metrics from {summary_path}: {e}")

    return metrics if metrics else None


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep for complete experiments (PINN + ML Baselines + Comparison)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Overnight sweep with data reuse (RECOMMENDED)
  python scripts/run_hyperparameter_sweep.py \
      --config configs/experiments/hyperparameter_tuning.yaml \
      --n-samples-range 10 20 30 50 \
      --epochs-range 100 200 300 \
      --skip-data-generation \
      --data-dir-base outputs/complete_experiments

  # Quick mode (skip optional steps for faster runs)
  python scripts/run_hyperparameter_sweep.py \
      --config configs/experiments/hyperparameter_tuning.yaml \
      --n-samples-range 20 30 \
      --epochs-range 100 200 \
      --skip-data-generation \
      --quick-mode \
      --data-dir-base outputs/complete_experiments

  # Full sweep with all features
  python scripts/run_hyperparameter_sweep.py \
      --config configs/experiments/hyperparameter_tuning.yaml \
      --n-samples-range 10 20 30 \
      --epochs-range 100 200 300 \
      --output-dir outputs/hyperparameter_sweep
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/hyperparameter_tuning.yaml",
        help="Base configuration file (default: hyperparameter_tuning.yaml)",
    )
    parser.add_argument(
        "--n-samples-range",
        type=int,
        nargs="+",
        required=True,
        help="List of n_samples values to test (e.g., 10 20 30)",
    )
    parser.add_argument(
        "--epochs-range",
        type=int,
        nargs="+",
        required=True,
        help="List of epochs values to test (e.g., 50 100 200)",
    )
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Skip data generation (use existing data)",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of data even if it exists in common repository",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (required if --skip-data-generation). Can be a single directory or a mapping like '20:path/to/data20,30:path/to/data30'",
    )
    parser.add_argument(
        "--data-dir-base",
        type=str,
        default=None,
        help="Base directory to search for experiment data. Script will auto-find data for each n_samples value in outputs/complete_experiments/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/complete_experiments",
        help="Base output directory for experiments (default: outputs/complete_experiments)",
    )
    parser.add_argument(
        "--skip-data-analysis",
        action="store_true",
        help="Skip data analysis (ANDES) for faster runs",
    )
    parser.add_argument(
        "--skip-ml-baseline-training",
        action="store_true",
        help="Skip ML baseline training (PINN only)",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip model comparison",
    )
    parser.add_argument(
        "--quick-mode",
        action="store_true",
        help="Quick mode: skip data analysis, ML baseline training, and comparison (PINN only, fastest)",
    )

    args = parser.parse_args()

    # Apply quick-mode flags
    if args.quick_mode:
        args.skip_data_analysis = True
        args.skip_ml_baseline_training = True
        args.skip_comparison = True

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    # Parse data directory mapping if provided
    data_dir_map = {}
    if args.data_dir:
        if ":" in args.data_dir:
            # Parse mapping format: "20:path1,30:path2"
            for mapping in args.data_dir.split(","):
                if ":" in mapping:
                    n_samples_str, data_path = mapping.split(":", 1)
                    try:
                        n_samples_val = int(n_samples_str.strip())
                        data_dir_map[n_samples_val] = Path(data_path.strip())
                    except ValueError:
                        print(f"⚠️  Warning: Invalid n_samples in mapping: {n_samples_str}")
                else:
                    # Single directory for all
                    data_dir_map[None] = Path(args.data_dir.strip())
        else:
            # Single directory for all experiments
            data_dir_map[None] = Path(args.data_dir)

    if args.skip_data_generation and not args.data_dir and not args.data_dir_base:
        print("❌ --data-dir or --data-dir-base required when --skip-data-generation is used")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track test set consistency (CRITICAL for fair comparison)
    test_set_hash = None
    test_set_warning_shown = False

    # Run all combinations
    results = []
    total_experiments = len(args.n_samples_range) * len(args.epochs_range)
    start_time = time.time()

    # Progress tracking file
    progress_file = output_dir / "sweep_progress.json"
    progress_data = {
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_experiments": total_experiments,
        "completed": 0,
        "failed": 0,
        "experiments": [],
    }

    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SWEEP (Complete Experiments)")
    print(f"{'='*70}")
    print(f"n_samples values: {args.n_samples_range}")
    print(f"epochs values: {args.epochs_range}")
    print(f"Total experiments: {total_experiments}")
    print(f"Output directory: {output_dir}")
    if args.quick_mode:
        print(f"Mode: QUICK (skipping data analysis, ML baseline, comparison)")
    print(f"{'='*70}\n")

    experiment_num = 0
    for n_samples in args.n_samples_range:
        for epochs in args.epochs_range:
            experiment_num += 1
            exp_start_time = time.time()
            print(f"\n[{experiment_num}/{total_experiments}]")

            # Build command for run_complete_experiment.py
            cmd = [
                sys.executable,
                "scripts/run_complete_experiment.py",
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
            ]

            # Add skip flags
            if args.skip_data_analysis:
                cmd.append("--skip-data-analysis")
            if args.skip_ml_baseline_training:
                cmd.append("--skip-ml-baseline-training")
            if args.skip_comparison:
                cmd.append("--skip-comparison")
            if args.force_regenerate:
                cmd.append("--force-regenerate")

            if args.skip_data_generation:
                # Determine which data directory to use for this n_samples
                data_dir = None

                # Priority 1: Explicit mapping for this n_samples
                if n_samples in data_dir_map:
                    data_dir = data_dir_map[n_samples]
                # Priority 2: Default mapping (for all n_samples)
                elif None in data_dir_map:
                    data_dir = data_dir_map[None]
                # Priority 3: Auto-find from base directory
                elif args.data_dir_base:
                    base_dir = Path(args.data_dir_base)
                    data_dir = find_experiment_data_dir(base_dir, n_samples)
                    if data_dir:
                        print(f"  ✓ Auto-found data for n_samples={n_samples}: {data_dir}")
                    else:
                        print(
                            f"  ⚠️  Warning: Could not find data directory for n_samples={n_samples}"
                        )
                        print(f"     Searched in: {base_dir}")

                if data_dir:
                    # Check if data_dir contains raw CSV files (from data/common)
                    csv_files = list(data_dir.glob(f"*{n_samples}*.csv"))
                    if csv_files and "common" in str(data_dir):
                        # Use the most recent matching CSV file
                        data_file = max(csv_files, key=lambda p: p.stat().st_mtime)
                        cmd.extend(["--skip-data-generation", "--data-path", str(data_file)])
                        print(f"  ✓ Using raw data file: {data_file.name}")
                    else:
                        # Directory with processed splits or experiment data
                        cmd.extend(["--skip-data-generation", "--data-dir", str(data_dir)])

                    # Verify test set consistency (CRITICAL) - only for directories with processed data
                    if not (csv_files and "common" in str(data_dir)):
                        current_test_hash = get_test_set_hash(data_dir)
                        if test_set_hash is None:
                            test_set_hash = current_test_hash
                            if test_set_hash:
                                print(f"  ✓ Test set identified: {test_set_hash}")
                        elif current_test_hash and current_test_hash != test_set_hash:
                            if not test_set_warning_shown:
                                print(
                                    f"  ⚠️  WARNING: Different test set detected! ({current_test_hash} vs {test_set_hash})"
                                )
                                print(
                                    f"     This may affect fair comparison. Consider using the same data directory for all experiments."
                                )
                                test_set_warning_shown = True
                else:
                    print(f"  ❌ No data directory found for n_samples={n_samples}")
                    status = "FAILED"
                    error_msg = "Data directory not found"
                    exp_duration = time.time() - exp_start_time
                    results.append(
                        {
                            "n_samples": n_samples,
                            "epochs": epochs,
                            "status": status,
                            "error": error_msg,
                            "duration_seconds": exp_duration,
                        }
                    )
                    progress_data["failed"] += 1
                    progress_data["experiments"].append(
                        {
                            "n_samples": n_samples,
                            "epochs": epochs,
                            "status": status,
                            "error": error_msg,
                        }
                    )
                    # Save progress
                    with open(progress_file, "w") as f:
                        json.dump(progress_data, f, indent=2)
                    continue

            # Override hyperparameters via config update
            config = load_config(config_path)
            config["data"]["generation"]["n_samples"] = n_samples
            config["training"]["epochs"] = epochs

            # Save temporary config
            temp_config = Path("configs/experiments/temp_sweep.yaml")
            temp_config.parent.mkdir(parents=True, exist_ok=True)
            save_config(config, temp_config)
            cmd[3] = str(temp_config)  # Update config path

            print(f"Running: n_samples={n_samples}, epochs={epochs}")
            print(f"Command: {' '.join(cmd)}")

            status = "FAILED"
            error_msg = None
            experiment_id = None

            try:
                # Use Popen to stream output in real-time
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,  # Line buffered
                    universal_newlines=True,
                )

                # Stream output in real-time
                output_lines = []
                for line in iter(process.stdout.readline, ""):
                    if not line:
                        break
                    # Print immediately for real-time display
                    print(line, end="", flush=True)
                    output_lines.append(line)

                    # Try to extract experiment ID from output
                    if experiment_id is None:
                        if "Experiment ID:" in line or "exp_" in line:
                            match = re.search(r"exp_\d{8}_\d{6}", line)
                            if match:
                                experiment_id = match.group(0)

                # Wait for process to complete
                return_code = process.wait()

                if return_code == 0:
                    status = "SUCCESS"
                else:
                    status = "FAILED"
                    error_msg = f"Process exited with code {return_code}"

            except subprocess.CalledProcessError as e:
                status = "FAILED"
                error_msg = f"Subprocess error: {str(e)}"
            except Exception as e:
                status = "FAILED"
                error_msg = f"Unexpected error: {str(e)}"
                print(f"❌ Experiment failed: {e}")
            except Exception as e:
                status = "FAILED"
                error_msg = f"Unexpected error: {str(e)}"
                print(f"❌ Experiment failed: {e}")

            exp_duration = time.time() - exp_start_time

            # Extract metrics if experiment succeeded
            metrics = None
            if status == "SUCCESS" and experiment_id:
                exp_dir = output_dir / experiment_id
                if exp_dir.exists():
                    metrics = extract_experiment_metrics(exp_dir)

            result_entry = {
                "n_samples": n_samples,
                "epochs": epochs,
                "status": status,
                "duration_seconds": exp_duration,
            }
            if error_msg:
                result_entry["error"] = error_msg
            if experiment_id:
                result_entry["experiment_id"] = experiment_id
            if metrics:
                result_entry["metrics"] = metrics

            results.append(result_entry)

            # Update progress
            if status == "SUCCESS":
                progress_data["completed"] += 1
            else:
                progress_data["failed"] += 1

            progress_data["experiments"].append(
                {
                    "n_samples": n_samples,
                    "epochs": epochs,
                    "status": status,
                    "experiment_id": experiment_id,
                    "duration_seconds": exp_duration,
                }
            )

            # Calculate and print estimated time remaining
            elapsed_time = time.time() - start_time
            completed_count = progress_data["completed"] + progress_data["failed"]
            if completed_count > 0:
                avg_time_per_exp = elapsed_time / completed_count
                remaining_exps = total_experiments - completed_count
                estimated_remaining = avg_time_per_exp * remaining_exps
                print(
                    f"  Duration: {exp_duration:.1f}s | Est. remaining: {estimated_remaining/60:.1f} min"
                )

            # Save progress after each experiment
            with open(progress_file, "w") as f:
                json.dump(progress_data, f, indent=2)

            # Clean up temp config
            if temp_config.exists():
                temp_config.unlink()

    # Generate sweep summary
    total_duration = time.time() - start_time
    successful = sum(1 for r in results if r["status"] == "SUCCESS")
    failed = sum(1 for r in results if r["status"] == "FAILED")

    # Extract best performing combinations
    successful_results = [r for r in results if r["status"] == "SUCCESS" and "metrics" in r]
    best_results = {}
    if successful_results:
        # Find best by R² Delta (primary metric)
        best_by_r2_delta = max(
            successful_results,
            key=lambda x: x.get("metrics", {}).get("pinn_r2_delta") or -float("inf"),
        )
        best_results["best_r2_delta"] = {
            "n_samples": best_by_r2_delta["n_samples"],
            "epochs": best_by_r2_delta["epochs"],
            "experiment_id": best_by_r2_delta.get("experiment_id"),
            "r2_delta": best_by_r2_delta.get("metrics", {}).get("pinn_r2_delta"),
        }

        # Find best by RMSE Delta (lower is better)
        best_by_rmse_delta = min(
            successful_results,
            key=lambda x: x.get("metrics", {}).get("pinn_rmse_delta") or float("inf"),
        )
        best_results["best_rmse_delta"] = {
            "n_samples": best_by_rmse_delta["n_samples"],
            "epochs": best_by_rmse_delta["epochs"],
            "experiment_id": best_by_rmse_delta.get("experiment_id"),
            "rmse_delta": best_by_rmse_delta.get("metrics", {}).get("pinn_rmse_delta"),
        }

    # Create sweep summary
    sweep_summary = {
        "sweep_metadata": {
            "start_time": progress_data["start_time"],
            "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_duration_seconds": total_duration,
            "config_path": str(config_path),
            "n_samples_range": args.n_samples_range,
            "epochs_range": args.epochs_range,
            "total_experiments": total_experiments,
            "successful": successful,
            "failed": failed,
            "output_dir": str(output_dir),
            "test_set_hash": test_set_hash,
        },
        "experiments": results,
        "best_performing": best_results,
    }

    # Save sweep summary
    summary_file = output_dir / "sweep_summary.json"
    with open(summary_file, "w") as f:
        json.dump(sweep_summary, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total duration: {total_duration/60:.1f} minutes")
    print(f"\nResults:")
    for r in results:
        status_icon = "✓" if r["status"] == "SUCCESS" else "✗"
        exp_id = r.get("experiment_id", "N/A")
        print(
            f"  {status_icon} n_samples={r['n_samples']:3d}, epochs={r['epochs']:3d}: {r['status']:7s} ({exp_id})"
        )

    if best_results:
        print(f"\nBest Performing Combinations:")
        if "best_r2_delta" in best_results:
            best = best_results["best_r2_delta"]
            print(
                f"  Best R² Delta: n_samples={best['n_samples']}, epochs={best['epochs']} (R²={best.get('r2_delta', 'N/A')})"
            )
        if "best_rmse_delta" in best_results:
            best = best_results["best_rmse_delta"]
            print(
                f"  Best RMSE Delta: n_samples={best['n_samples']}, epochs={best['epochs']} (RMSE={best.get('rmse_delta', 'N/A')})"
            )

    print(f"\nSummary saved to: {summary_file}")
    print(f"Progress file: {progress_file}")
    print(f"{'='*70}\n")

    print("💡 Next steps:")
    print("  1. Review sweep summary: cat sweep_summary.json")
    print("  2. Compare all experiment results:")
    print(f"     python scripts/compare_experiments.py --experiments {output_dir}/exp_*")
    print("  3. Analyze best performing combinations from sweep_summary.json")


if __name__ == "__main__":
    main()
