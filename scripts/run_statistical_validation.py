#!/usr/bin/env python
"""
Statistical Validation Script.

Runs multiple experiments with different random seeds to assess reproducibility
and compute confidence intervals for metrics.

Usage:
    python scripts/run_statistical_validation.py \
        --config configs/experiments/baseline.yaml \
        --seeds 0 1 2 3 4 \
        --output-dir outputs/statistical_validation
"""

import argparse
import sys
import io
from pathlib import Path
from typing import Dict, List, Optional

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import time
import subprocess
import random
import numpy as np
from scripts.core.data_generation import generate_training_data
from scripts.core.evaluation import evaluate_model
from scripts.core.experiment_tracker import ExperimentTracker
from scripts.core.training import train_model
from scripts.core.utils import (
    create_experiment_directory,
    generate_experiment_id,
    generate_timestamped_filename,
    load_config,
    save_config,
    validate_config,
    save_json,
    load_json,
)
from data_generation.preprocessing import preprocess_data, split_dataset
from evaluation.baselines.ml_baselines import MLBaselineTrainer


def run_single_experiment(
    config: Dict,
    data_path: Path,
    output_dir: Path,
    seed: int,
    experiment_id: Optional[str] = None,
    test_data_path: Optional[Path] = None,
) -> Dict:
    """
    Run a single experiment with a specific seed.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    data_path : Path
        Path to training data
    output_dir : Path
        Base output directory
    seed : int
        Random seed
    experiment_id : str, optional
        Experiment ID (if None, generates new one)

    Returns:
    --------
    results : dict
        Dictionary with experiment results
    """
    if experiment_id is None:
        experiment_id = generate_experiment_id()

    print(f"\n{'='*70}")
    print(f"EXPERIMENT WITH SEED {seed}")
    print(f"{'='*70}")
    print(f"Experiment ID: {experiment_id}")

    # Create experiment directory
    dirs = create_experiment_directory(output_dir, experiment_id)
    print(f"Output directory: {dirs['root']}")

    # Save config with seed annotation
    config_with_seed = config.copy()
    config_with_seed["seed"] = seed
    save_config(config_with_seed, dirs["root"] / "config.yaml")

    # Preprocess data if needed
    # Ensure train_data_path is always a Path object for consistency
    train_data_path = Path(data_path)
    eval_test_data_path = Path(test_data_path) if test_data_path is not None else None

    # Check if data needs preprocessing
    if config.get("preprocessing", {}).get("enabled", False):
        print("\nPreprocessing data...")
        preprocessed_data = preprocess_data(
            data_path,
            config.get("preprocessing", {}),
            output_dir=dirs["root"] / "data",
        )
        train_data, val_data, test_data = split_dataset(
            preprocessed_data,
            config.get("preprocessing", {}),
            output_dir=dirs["root"] / "data",
        )
        # Ensure paths are Path objects
        train_data_path = Path(train_data)
        eval_test_data_path = Path(test_data) if test_data is not None else None

    # If test data not provided, try to find it in same directory as train data
    if eval_test_data_path is None:
        train_dir = train_data_path.parent
        # Look for test_data_*.csv in same directory
        test_files = list(train_dir.glob("test_data_*.csv"))
        if test_files:
            eval_test_data_path = max(test_files, key=lambda p: p.stat().st_mtime)
            print(f"✓ Found test data in same directory: {eval_test_data_path.name}")
        else:
            print("⚠️  Test data not found, will use training data for evaluation (not ideal)")
            eval_test_data_path = train_data_path

    # Train model
    print(f"\nTraining model with seed {seed}...")
    model_path, training_history = train_model(
        config=config,
        data_path=train_data_path,
        output_dir=dirs["root"],
        seed=seed,
    )

    # CRITICAL FIX: Use experiment directory checkpoint (has scalers) instead of common repository
    # The common repository checkpoint only has model weights, not scalers.
    # This causes evaluation to fit scalers from test data, leading to normalization mismatch.
    # Solution: Find the experiment directory checkpoint which has full checkpoint with scalers.
    # Check multiple possible locations (statistical validation saves directly to root, complete experiments use pinn/model)
    possible_checkpoint_dirs = [
        dirs["root"],  # Direct location (statistical validation)
        dirs["root"] / "pinn" / "model",  # Subdirectory (complete experiments)
        dirs["root"] / "model",  # Alternative location
    ]

    exp_checkpoint_found = False
    for checkpoint_dir in possible_checkpoint_dirs:
        if checkpoint_dir.exists():
            exp_checkpoints = list(checkpoint_dir.glob("best_model_*.pth"))
            if exp_checkpoints:
                # Use the most recent checkpoint from experiment directory (has scalers)
                exp_checkpoint_path = max(exp_checkpoints, key=lambda p: p.stat().st_mtime)
                print(
                    f"✓ Using experiment directory checkpoint (has scalers): {exp_checkpoint_path.name}"
                )
                print(f"  Location: {exp_checkpoint_path.parent.name}/")
                print(f"  (Instead of common repository checkpoint which lacks scalers)")
                model_path = exp_checkpoint_path
                exp_checkpoint_found = True
                break

    if not exp_checkpoint_found:
        print(
            f"⚠️  Warning: No experiment directory checkpoint found, using common repository path"
        )
        print(f"  Path: {model_path}")
        print(f"  This may cause scaler mismatch if common repository checkpoint lacks scalers")
        print(f"  Searched in: {[str(d) for d in possible_checkpoint_dirs]}")

    # Evaluate PINN model
    print(f"\nEvaluating PINN model...")
    print(f"  Using test data: {eval_test_data_path}")
    print(f"  Using model: {model_path}")
    pinn_evaluation_results = evaluate_model(
        config=config,
        model_path=model_path,
        test_data_path=eval_test_data_path,  # Use proper test data
        output_dir=dirs["root"],
    )

    # Extract PINN metrics
    pinn_metrics = pinn_evaluation_results.get("metrics", {}) if pinn_evaluation_results else {}

    # Train and evaluate ML baseline if enabled
    ml_baseline_results = {}
    run_baselines = config.get("evaluation", {}).get("run_baselines", False)

    if run_baselines:
        print(f"\n{'='*70}")
        print("ML BASELINE TRAINING AND EVALUATION")
        print(f"{'='*70}")

        # Determine which models to train (default: standard_nn)
        models_to_train = ["standard_nn"]  # Default for statistical validation

        # Get training config
        train_config = config.get("training", {})
        model_config = config.get("model", {})
        input_method = model_config.get("input_method", "pe_direct")
        epochs = train_config.get("epochs", 400)
        learning_rate = float(train_config.get("learning_rate", 1e-3))
        weight_decay = float(train_config.get("weight_decay", 1e-5))
        early_stopping_patience = train_config.get("early_stopping_patience", None)

        # Get dropout from PINN config for fair comparison
        pinn_dropout = float(model_config.get("dropout", 0.0))

        # Get loss config
        loss_config = config.get("loss", {})
        lambda_ic = float(loss_config.get("lambda_ic", 10.0))
        scale_to_norm = loss_config.get("scale_to_norm", [1.0, 100.0])
        # Validate scale_to_norm is a list with 2 elements
        if not isinstance(scale_to_norm, list) or len(scale_to_norm) != 2:
            print(f"  ⚠️  Warning: scale_to_norm is not a valid list, using default [1.0, 100.0]")
            scale_to_norm = [1.0, 100.0]

        for model_type in models_to_train:
            print(f"\nTraining {model_type}...")

            # Create model directory
            ml_baseline_dir = dirs["root"] / "ml_baseline" / model_type
            model_dir = ml_baseline_dir / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Create trainer
            trainer = MLBaselineTrainer(
                model_type=model_type,
                model_config={
                    "hidden_dims": (
                        model_config.get("hidden_dims", [256, 256, 128, 128])
                        if model_type == "standard_nn"
                        else None
                    ),
                    "hidden_size": 128 if model_type == "lstm" else None,
                    "num_layers": 2 if model_type == "lstm" else None,
                    "activation": model_config.get("activation", "tanh"),
                    "dropout": pinn_dropout,  # Match PINN for fair comparison
                },
            )

            # Set random seed for reproducibility (same as PINN) - BEFORE data preparation
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            # Prepare data
            print(f"  Preparing data...")
            val_data_path = None
            # train_data_path is already a Path object, so we can use it directly
            if "train_data_" in train_data_path.name:
                val_data_path = train_data_path.parent / train_data_path.name.replace(
                    "train_data_", "val_data_"
                )
                if not val_data_path.exists():
                    val_data_path = None

            train_loader, val_loader, scalers = trainer.prepare_data(
                data_path=train_data_path,
                input_method=input_method,
                val_data_path=val_data_path if val_data_path and val_data_path.exists() else None,
            )

            # Train model
            print(f"  Training {model_type} with seed {seed}...")
            start_time = time.time()
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                early_stopping_patience=early_stopping_patience,
                lambda_ic=lambda_ic,
                scale_to_norm=scale_to_norm,
            )
            training_time = time.time() - start_time

            # Save model
            model_filename = generate_timestamped_filename("best_model", "pth")
            model_path_ml = model_dir / model_filename

            checkpoint_data = {
                "model_state_dict": trainer.model.state_dict(),
                "model_type": model_type,
                "model_config": trainer.model_config,
                "scalers": scalers,
                "input_method": input_method,
                "training_history": history,
            }

            torch.save(checkpoint_data, model_path_ml)
            torch.save(checkpoint_data, model_dir / "model.pth")  # Compatibility

            print(f"  ✓ {model_type} training complete: {training_time:.1f}s")

            # Evaluate ML baseline
            print(f"  Evaluating {model_type}...")
            eval_dir = ml_baseline_dir / "evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)

            # Use evaluate_ml_baseline.py script
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_ml_baseline.py"),
                "--model-path",
                str(model_path_ml),
                "--test-data",
                str(eval_test_data_path),
                "--test-split-path",
                str(eval_test_data_path),
                "--output-dir",
                str(eval_dir),
            ]

            # Fix: Specify UTF-8 encoding to avoid UnicodeDecodeError on Windows
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",  # Replace invalid characters instead of raising error
            )

            if result.returncode != 0:
                print(f"  ⚠️  ML baseline evaluation returned error code {result.returncode}")
                if result.stderr:
                    print(f"  Error output: {result.stderr}")
                if result.stdout:
                    print(f"  Standard output: {result.stdout}")
                ml_metrics = {}
            else:
                # Load evaluation results
                metrics_file = eval_dir / "metrics.json"
                if metrics_file.exists():
                    ml_metrics = load_json(metrics_file)
                    print(f"  ✓ {model_type} evaluation complete")
                else:
                    print(f"  ⚠️  Metrics file not found: {metrics_file}")
                    if result.stdout:
                        print(f"  Subprocess output: {result.stdout}")
                    ml_metrics = {}

            ml_baseline_results[model_type] = {
                "model_path": str(model_path_ml),
                "metrics": ml_metrics,
                "training_history": history,
                "training_time": training_time,
            }
    else:
        print(f"\n⚠️  ML baseline training/evaluation skipped (run_baselines=False in config)")

    # Build results dictionary
    results = {
        "experiment_id": experiment_id,
        "seed": seed,
        "metrics": pinn_metrics,  # PINN metrics (for backward compatibility)
        "pinn": {
            "metrics": pinn_metrics,
            "training_history": training_history or {},
            "model_path": str(model_path),
        },
        "experiment_dir": str(dirs["root"]),
    }

    # Add ML baseline results if available
    if ml_baseline_results:
        results["ml_baseline"] = ml_baseline_results

    return results


def main():
    """Main statistical validation workflow."""
    parser = argparse.ArgumentParser(description="Run statistical validation with multiple seeds")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to training data CSV file (if not generating)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds to use (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/statistical_validation",
        help="Output directory for statistical validation results",
    )
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Skip data generation (use existing data)",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        help="Path to test data CSV file (if using preprocessed splits)",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    print("=" * 70)
    print("STATISTICAL VALIDATION")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {args.output_dir}")

    config = load_config(config_path)
    try:
        validate_config(config)
    except ValueError as e:
        print(f"[ERROR] Invalid configuration: {e}")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle data generation/loading
    # Use existing data if --data-path is provided (with or without --skip-data-generation)
    if args.data_path:
        # Handle wildcard patterns
        data_path_str = args.data_path
        if "*" in data_path_str:
            # Expand wildcard pattern using Path.glob() for better Windows support
            if not Path(data_path_str).is_absolute():
                # Convert relative path to absolute
                data_path_pattern = PROJECT_ROOT / data_path_str
            else:
                data_path_pattern = Path(data_path_str)

            # Split into directory and pattern
            pattern_dir = data_path_pattern.parent
            pattern_name = data_path_pattern.name

            if not pattern_dir.exists():
                print(f"[ERROR] Directory not found: {pattern_dir}")
                print(f"  Searched for pattern: {args.data_path}")
                sys.exit(1)

            # Use Path.glob() which handles Windows paths better
            matching_files = list(pattern_dir.glob(pattern_name))
            if not matching_files:
                print(f"[ERROR] No files found matching pattern: {args.data_path}")
                print(f"  Searched in: {pattern_dir}")
                print(f"  Pattern: {pattern_name}")
                print(f"\n  Solutions:")
                print(f"  1. Use --skip-data-generation to auto-find latest preprocessed data:")
                print(
                    f"python scripts/run_statistical_validation.py --config <config>"
                    f"--skip-data-generation --seeds 0 1 2 3 4"
                )
                print(f"  2. Check if experiment has data directory:")
                print(f"     Check: {pattern_dir.parent.parent}/data/processed/")
                print(f"  3. Use a different experiment that has preprocessed data")
                sys.exit(1)
            # Use the latest matching file (by modification time)
            data_path = max(matching_files, key=lambda p: p.stat().st_mtime)
            print(
                f"Found {len(matching_files)} file(s) matching pattern, using latest: {data_path}"
            )
        else:
            data_path = Path(args.data_path)
            if not data_path.is_absolute():
                data_path = PROJECT_ROOT / data_path
            if not data_path.exists():
                print(f"[ERROR] Data file not found: {data_path}")
                print(f"\n  Tip: Use --skip-data-generation to auto-find latest preprocessed data")
                sys.exit(1)
        print(f"\nUsing existing data: {data_path}")
    elif args.skip_data_generation:
        # If --skip-data-generation but no --data-path, find latest preprocessed data
        print("\nFinding latest preprocessed data...")
        experiments_dir = PROJECT_ROOT / "outputs" / "experiments"
        if not experiments_dir.exists():
            print(
                "[ERROR] No experiments directory found. Please provide --data-path or generate data."
            )
            sys.exit(1)

        # Find latest experiment with preprocessed data
        latest_data_path = None
        for exp_dir in sorted(
            experiments_dir.glob("exp_*"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            preprocessed_dir = exp_dir / "data" / "preprocessed"
            if preprocessed_dir.exists():
                train_files = list(preprocessed_dir.glob("train_data_*.csv"))
                if train_files:
                    latest_data_path = max(train_files, key=lambda p: p.stat().st_mtime)
                    print(f"Found latest preprocessed data: {latest_data_path}")
                    print(f"  From experiment: {exp_dir.name}")
                    break

        if latest_data_path is None:
            print(
                "[ERROR] No preprocessed data found. Please provide --data-path or generate data."
            )
            sys.exit(1)

        data_path = latest_data_path
    else:
        # Generate data
        print("\nGenerating training data...")
        data_path, validation_results = generate_training_data(
            config=config,
            output_dir=output_dir / "data",
        )
        print(f"Generated data: {data_path}")

    # Get test data path if provided
    test_data_path = None
    if args.test_data_path:
        test_data_path_str = args.test_data_path
        if "*" in test_data_path_str:
            # Handle wildcard pattern
            if not Path(test_data_path_str).is_absolute():
                test_data_path_pattern = PROJECT_ROOT / test_data_path_str
            else:
                test_data_path_pattern = Path(test_data_path_str)
            pattern_dir = test_data_path_pattern.parent
            pattern_name = test_data_path_pattern.name
            matching_files = list(pattern_dir.glob(pattern_name))
            if matching_files:
                test_data_path = max(matching_files, key=lambda p: p.stat().st_mtime)
                print(f"Using test data: {test_data_path}")
            else:
                print(f"⚠️  Test data pattern not found: {test_data_path_str}")
        else:
            test_data_path = Path(test_data_path_str)
            if not test_data_path.is_absolute():
                test_data_path = PROJECT_ROOT / test_data_path
            if test_data_path.exists():
                print(f"Using test data: {test_data_path}")
            else:
                print(f"⚠️  Test data file not found: {test_data_path}")

    # Run experiments with different seeds
    all_results = []
    experiment_ids = []

    for seed in args.seeds:
        experiment_id = generate_experiment_id()
        experiment_ids.append(experiment_id)

        try:
            results = run_single_experiment(
                config=config,
                data_path=data_path,
                output_dir=output_dir / "experiments",
                seed=seed,
                experiment_id=experiment_id,
                test_data_path=test_data_path,
            )
            all_results.append(results)
            print(f"\n✓ Completed experiment with seed {seed}")
        except Exception as e:
            print(f"\n✗ Failed experiment with seed {seed}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Check if we have any results
    if not all_results:
        print("\n⚠️  WARNING: No successful experiments completed!")
        print("  Cannot generate statistical summary or plots.")
        print(f"  Check error messages above for details.")
        sys.exit(1)

    # Save raw results
    # Save as JSON first (preserves nested structure)
    results_json_file = output_dir / "raw_results.json"
    import json

    with open(results_json_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Saved raw results (JSON) to: {results_json_file}")

    # Also save flattened CSV for simple metrics (PINN metrics only, for backward compatibility)
    # Note: Nested structures (ml_baseline, training_history) are not included in CSV
    flattened_results = []
    for result in all_results:
        flat_result = {
            "experiment_id": result.get("experiment_id"),
            "seed": result.get("seed"),
            "experiment_dir": result.get("experiment_dir"),
        }
        # Add PINN metrics (flatten top-level metrics for backward compatibility)
        pinn_metrics = result.get("pinn", {}).get("metrics", {}) or result.get("metrics", {})
        for key, value in pinn_metrics.items():
            if isinstance(value, (int, float)):
                flat_result[f"pinn_{key}"] = value
        flattened_results.append(flat_result)

    if flattened_results:
        results_df = pd.DataFrame(flattened_results)
        results_file = output_dir / "raw_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"✓ Saved raw results (CSV, flattened) to: {results_file}")
        print(f"  Note: Full results with nested structures are in raw_results.json")

    # Compute statistical summary
    print("\n" + "=" * 70)
    print("COMPUTING STATISTICAL SUMMARY")
    print("=" * 70)

    from scripts.analysis.statistical_summary import compute_statistical_summary

    summary = compute_statistical_summary(all_results)

    # Check if summary is empty (all experiments failed or had no metrics)
    if not summary:
        print("⚠️  WARNING: Statistical summary is empty!")
        print("  This may indicate that all experiments failed or produced no metrics.")
        print("  Check error messages above for details.")

    summary_file = output_dir / "statistical_summary.json"
    import json

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved statistical summary to: {summary_file}")

    # Generate visualizations
    print("\nGenerating statistical plots...")
    from scripts.visualization.statistical_plots import generate_statistical_plots

    plots_dir = output_dir / "figures"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generate_statistical_plots(all_results, plots_dir)
    print(f"✓ Saved plots to: {plots_dir}")

    # Print summary
    print("\n" + "=" * 70)
    print("STATISTICAL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nNumber of successful runs: {len(all_results)}/{len(args.seeds)}")

    # Print PINN metrics summary
    print(f"\n{'='*70}")
    print("PINN METRICS SUMMARY")
    print(f"{'='*70}")
    pinn_metrics_printed = False
    for metric_name, stats in summary.items():
        if isinstance(stats, dict) and "mean" in stats:
            # Filter for PINN metrics (exclude ML baseline metrics)
            if not metric_name.startswith("ml_baseline."):
                mean = stats["mean"]
                std = stats.get("std", 0.0)
                # Fix: Use CI from summary (already computed correctly), don't recalculate
                ci_lower = stats.get("ci_lower")
                ci_upper = stats.get("ci_upper")
                # Fallback only if CI not in summary (shouldn't happen, but safe)
                if ci_lower is None or ci_upper is None:
                    # Only recalculate if missing (shouldn't happen)
                    n = stats.get("n", 1)
                    if n > 1:
                        se = std / np.sqrt(n)
                        ci_lower = mean - 1.96 * se
                        ci_upper = mean + 1.96 * se
                    else:
                        ci_lower = mean
                        ci_upper = mean
                print(f"  {metric_name}:")
                print(f"    Mean: {mean:.6f}")
                print(f"    Std:  {std:.6f}")
                print(f"    95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
                print(f"    N: {stats.get('n', 0)}")
                pinn_metrics_printed = True

    if not pinn_metrics_printed:
        print("  (No PINN metrics found)")

    # Print ML baseline metrics summary if available
    run_baselines = config.get("evaluation", {}).get("run_baselines", False)
    if run_baselines:
        print(f"\n{'='*70}")
        print("ML BASELINE METRICS SUMMARY")
        print(f"{'='*70}")
        ml_metrics_printed = False
        for metric_name, stats in summary.items():
            if isinstance(stats, dict) and "mean" in stats:
                # Filter for ML baseline metrics
                if metric_name.startswith("ml_baseline."):
                    mean = stats["mean"]
                    std = stats.get("std", 0.0)
                    # Fix: Use CI from summary (already computed correctly), don't recalculate
                    ci_lower = stats.get("ci_lower")
                    ci_upper = stats.get("ci_upper")
                    # Fallback only if CI not in summary (shouldn't happen, but safe)
                    if ci_lower is None or ci_upper is None:
                        # Only recalculate if missing (shouldn't happen)
                        n = stats.get("n", 1)
                        if n > 1:
                            se = std / np.sqrt(n)
                            ci_lower = mean - 1.96 * se
                            ci_upper = mean + 1.96 * se
                        else:
                            ci_lower = mean
                            ci_upper = mean
                    print(f"  {metric_name}:")
                    print(f"    Mean: {mean:.6f}")
                    print(f"    Std:  {std:.6f}")
                    print(f"    95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
                    print(f"    N: {stats.get('n', 0)}")
                    ml_metrics_printed = True

        if not ml_metrics_printed:
            print("  (No ML baseline metrics found)")

    print(f"\n✓ Statistical validation complete!")
    print(f"  Results: {output_dir}")


if __name__ == "__main__":
    main()
