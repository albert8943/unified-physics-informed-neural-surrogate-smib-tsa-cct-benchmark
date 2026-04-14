#!/usr/bin/env python
"""
Main Experiment Workflow Script.

This script orchestrates the full pipeline:
1. Load configuration
2. Generate/load training data
3. Analyze data (optional, generates statistics and figures)
4. Preprocess and split data (optional, creates train/val/test sets)
5. Train model
6. Evaluate on test set
7. Save all results with experiment ID

Usage:
    # Full workflow
    python scripts/run_experiment.py --config configs/experiments/baseline.yaml
    
    # Use existing data (from directory - searches common repository first)
    python scripts/run_experiment.py --config configs/experiments/baseline.yaml \
        --skip-data-generation --data-dir data/common
    
    # Use existing data (direct file path)
    python scripts/run_experiment.py --config configs/experiments/baseline.yaml \
        --skip-data-generation --data-path data/common/trajectory_data_1000_H2-10_D0.5-3_abc12345_20251205_170908.csv
    
    # Use existing model (skip training)
    python scripts/run_experiment.py --config configs/experiments/baseline.yaml \
        --skip-training --model-path outputs/experiments/exp_20251205_170908/model/best_model_20251205_170908.pth
    
    # Use existing data and model (only evaluation)
    python scripts/run_experiment.py --config configs/experiments/baseline.yaml \
        --skip-data-generation --data-path data/preprocessed/quick_test/test_data_20251205_170908.csv \
        --skip-training --model-path outputs/models/common/trajectory/model_abc12345_20251205_170908.pth
    
    # Override epochs from command line
    python scripts/run_experiment.py --config configs/experiments/baseline.yaml \
        --skip-data-generation --data-path data/common/trajectory_data_1000_H2-10_D0.5-3_abc12345_20251205_170908.csv \
        --epochs 100
"""

import argparse
import sys
import io
from pathlib import Path

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path (must be before imports)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports after path setup
import pandas as pd  # noqa: E402
from scripts.core.data_generation import generate_training_data  # noqa: E402
from scripts.core.evaluation import evaluate_model  # noqa: E402
from scripts.core.experiment_tracker import ExperimentTracker  # noqa: E402
from scripts.core.training import train_model  # noqa: E402
from scripts.core.utils import (  # noqa: E402
    create_experiment_directory,
    generate_experiment_id,
    generate_timestamped_filename,
    load_config,
    save_config,
    validate_config,
)
from data_generation.preprocessing import preprocess_data, split_dataset  # noqa: E402


def _handle_data_analysis(data_path: Path, output_dir: Path) -> None:
    """Handle data analysis step."""
    print("\n" + "=" * 70)
    print("STEP 1B: DATA ANALYSIS")
    print("=" * 70)

    # Set matplotlib backend before importing analysis functions (important for headless environments)
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend

    # Import analysis functions
    from scripts.analyze_data import (
        compute_cct_correlations,
        compute_dataset_statistics,
        compute_trajectory_statistics,
        generate_cct_figures,
        generate_parameter_space_figures,
        generate_summary_report,
        generate_trajectory_figures,
        load_data,
    )

    analysis_dir = output_dir / "analysis"
    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data file: {data_path}")
    print(f"Output directory: {analysis_dir}")

    # Load data
    print("\nLoading data...")
    df = load_data(data_path)

    # Compute statistics
    print("\nComputing statistics...")
    stats_dict = compute_dataset_statistics(df)
    traj_stats_df = compute_trajectory_statistics(df)
    cct_correlations = compute_cct_correlations(df)

    # Generate figures
    print("\nGenerating figures...")
    print(f"   Output directory: {figures_dir}")
    print(f"   Directory exists: {figures_dir.exists()}")
    if not figures_dir.exists():
        print(f"   [WARNING]  Creating directory: {figures_dir}")
        figures_dir.mkdir(parents=True, exist_ok=True)

    param_figures = generate_parameter_space_figures(df, figures_dir, ["png"])
    traj_figures = generate_trajectory_figures(df, figures_dir, ["png"])
    cct_figures = generate_cct_figures(df, figures_dir, ["png"])

    # Generate summary report
    report_path = generate_summary_report(stats_dict, traj_stats_df, cct_correlations, analysis_dir)

    # Save trajectory statistics
    if len(traj_stats_df) > 0:
        traj_stats_filename = generate_timestamped_filename("trajectory_statistics", "csv")
        traj_stats_path = analysis_dir / traj_stats_filename
        traj_stats_df.to_csv(traj_stats_path, index=False)
        print(f"  [OK] Trajectory statistics: {traj_stats_path.name}")

    # Verify figures were actually saved
    saved_figures = list(figures_dir.glob("*.png"))
    print(f"\n[OK] Analysis complete")
    print(f"  - Figures: {len(param_figures) + len(traj_figures) + len(cct_figures)} total")
    print(f"  - Actually saved: {len(saved_figures)} PNG files in {figures_dir}")
    if len(saved_figures) > 0:
        print(f"  - Saved files:")
        for fig in saved_figures:
            print(f"    • {fig.name}")
    else:
        print(f"  [WARNING]  WARNING: No PNG files found in {figures_dir}")
        print(f"     This may indicate a save failure. Check for errors above.")
    print(f"  - Report: {report_path.name}")


def _handle_data_preprocessing(
    data_path: Path, output_dir: Path, config: dict
) -> tuple[Path, Path, Path]:
    """Handle data preprocessing and splitting step."""
    print("\n" + "=" * 70)
    print("STEP 1C: DATA PREPROCESSING & SPLITTING")
    print("=" * 70)

    # Get preprocessing config
    data_config = config.get("data", {})
    preprocess_config = data_config.get("preprocessing", {})

    train_ratio = preprocess_config.get("train_ratio", 0.7)
    val_ratio = preprocess_config.get("val_ratio", 0.15)
    test_ratio = preprocess_config.get("test_ratio", 0.15)
    random_state = config.get("reproducibility", {}).get("random_seed", 42)
    stratify_by = preprocess_config.get("stratify_by", None)

    # Angle filtering settings
    filter_angles = preprocess_config.get("filter_angles", False)
    max_angle_deg = preprocess_config.get("max_angle_deg", 360.0)
    stability_threshold_deg = preprocess_config.get("stability_threshold_deg", 180.0)

    print(f"Input: {data_path}")
    print(f"Train ratio: {train_ratio}")
    print(f"Val ratio: {val_ratio}")
    print(f"Test ratio: {test_ratio}")
    print(f"Random state: {random_state}")
    if filter_angles:
        print(
            f"Angle filtering: ENABLED (max: {max_angle_deg}°, stability threshold:"
            f"{stability_threshold_deg}°)"
        )

    # Load data
    print("\nLoading data...")
    data = pd.read_csv(data_path)
    print(f"[OK] Loaded {len(data):,} rows, {len(data.columns)} columns")
    print(f"  Unique scenarios: {data['scenario_id'].nunique()}")

    # Apply angle filtering if enabled, then split dataset
    if filter_angles:
        print("\nApplying angle filtering...")
        preprocess_result = preprocess_data(
            data=data,
            normalize=False,  # Normalization happens in training
            apply_feature_engineering=False,  # Feature engineering happens in training
            split=True,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify_by=stratify_by,
            random_state=random_state,
            filter_angles=True,
            max_angle_deg=max_angle_deg,
            stability_threshold_deg=stability_threshold_deg,
        )
        train_data = preprocess_result["train_data"]
        val_data = preprocess_result["val_data"]
        test_data = preprocess_result["test_data"]
        filter_stats = preprocess_result.get("filter_stats")
        if filter_stats:
            print(f"\n[INFO] Angle filtering statistics:")
            print(
                f"Points: {filter_stats.get('original_points', 0):,} →"
                f"{filter_stats.get('filtered_points', 0):,}"
            )
            print(
                f"Scenarios: {filter_stats.get('original_scenarios', 0)} →"
                f"{filter_stats.get('filtered_scenarios', 0)}"
            )
            print(
                f"Max angle: {filter_stats.get('max_angle_before', 0):.1f}° →"
                f"{filter_stats.get('max_angle_after', 0):.1f}°"
            )
    else:
        print("\nSplitting dataset...")
        train_data, val_data, test_data = split_dataset(
            data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify_by=stratify_by,
            random_state=random_state,
        )

    print(f"  [OK] Training set: {len(train_data):,} rows ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  [OK] Validation set: {len(val_data):,} rows ({len(val_data)/len(data)*100:.1f}%)")
    print(f"  [OK] Test set: {len(test_data):,} rows ({len(test_data)/len(data)*100:.1f}%)")

    # Save splits
    print("\nSaving preprocessed data...")
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    train_filename = generate_timestamped_filename("train_data", "csv")
    val_filename = generate_timestamped_filename("val_data", "csv")
    test_filename = generate_timestamped_filename("test_data", "csv")

    train_path = output_dir / train_filename
    val_path = output_dir / val_filename
    test_path = output_dir / test_filename

    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"  [OK] Training data: {train_path.name}")
    print(f"  [OK] Validation data: {val_path.name}")
    print(f"  [OK] Test data: {test_path.name}")

    # Save metadata
    from scripts.core.utils import save_json

    metadata = {
        "input_file": str(data_path),
        "splitting": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "stratify_by": stratify_by,
            "random_state": random_state,
        },
        "statistics": {
            "total_rows": len(data),
            "train_rows": len(train_data),
            "val_rows": len(val_data),
            "test_rows": len(test_data),
            "train_scenarios": train_data["scenario_id"].nunique(),
            "val_scenarios": val_data["scenario_id"].nunique(),
            "test_scenarios": test_data["scenario_id"].nunique(),
        },
        "output_files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }

    metadata_filename = generate_timestamped_filename("preprocessing_metadata", "json")
    metadata_path = output_dir / metadata_filename
    save_json(metadata, metadata_path)
    print(f"  [OK] Metadata: {metadata_path.name}")

    return train_path, val_path, test_path


def _handle_data_generation(args, config, dirs):
    """Handle data generation step."""
    if not args.skip_data_generation:
        print("\n" + "=" * 70)
        print("STEP 1: DATA GENERATION")
        print("=" * 70)

        # Check if common repository should be used
        use_common_repository = config.get("use_common_repository", True)
        force_regenerate = getattr(args, "force_regenerate", False)

        if use_common_repository:
            # Use common repository (data will be saved there)
            data_dir = dirs["root"] / "data"  # Still create dir for compatibility
            data_path, validation_results = generate_training_data(
                config=config,
                output_dir=data_dir,
                validate_physics=True,
                skip_if_exists=False,
                use_common_repository=True,
                force_regenerate=force_regenerate,
            )
        else:
            # Legacy: use experiment-specific directory
            data_dir = dirs["root"] / "data"
            data_path, validation_results = generate_training_data(
                config=config,
                output_dir=data_dir,
                validate_physics=True,
                skip_if_exists=False,
                use_common_repository=False,
            )

        # Data analysis (optional, can be skipped)
        if not args.skip_data_analysis:
            _handle_data_analysis(data_path, dirs["root"])

        # Data preprocessing (optional, can be skipped)
        if not args.skip_preprocessing:
            processed_dir = dirs["root"] / "data" / "processed"
            train_path, val_path, test_path = _handle_data_preprocessing(
                data_path, processed_dir, config
            )
            # Return train path for training, test path for evaluation
            return train_path, validation_results, test_path
        else:
            # Return original data path (training will do its own split)
            return data_path, validation_results, None
    else:
        # If direct data path is provided, use it
        if args.data_path:
            data_path = Path(args.data_path)
            if not data_path.exists():
                print(f"[ERROR] Data file not found: {data_path}")
                sys.exit(1)

            print(f"[OK] Using existing data file: {data_path}")

            # Data analysis (optional, can be skipped)
            if not args.skip_data_analysis:
                _handle_data_analysis(data_path, dirs["root"])

            # Check if it's a preprocessed train file
            if "train_data_" in data_path.name:
                # Try to find corresponding test file in same directory
                test_files = list(data_path.parent.glob("test_data_*.csv"))
                test_path = sorted(test_files)[-1] if test_files else None
                if test_path:
                    print(f"[OK] Found test data: {test_path.name}")
                return data_path, None, test_path

            # If preprocessing is enabled, preprocess the data
            if not args.skip_preprocessing:
                processed_dir = dirs["root"] / "data" / "processed"
                train_path, val_path, test_path = _handle_data_preprocessing(
                    data_path, processed_dir, config
                )
                return train_path, None, test_path
            else:
                return data_path, None, None

        # Otherwise, search in data directory
        if args.data_dir:
            data_dir = Path(args.data_dir)
            task = config.get("data", {}).get("task", "trajectory")

            # First try to find preprocessed train_data_*.csv (for training, use train set)
            train_files = list(data_dir.glob("train_data_*.csv"))
            if train_files:
                # Use the latest train file
                train_path = sorted(train_files)[-1]
                print(f"[OK] Using preprocessed training data: {train_path}")
                # Try to find corresponding test file
                test_files = list(data_dir.glob("test_data_*.csv"))
                test_path = sorted(test_files)[-1] if test_files else None
                if test_path:
                    print(f"[OK] Found test data: {test_path.name}")
                return train_path, None, test_path

            # Try to find parameter_sweep_data_*.csv (our generated data format)
            csv_files = list(data_dir.glob("parameter_sweep_data_*.csv"))
            if csv_files:
                # Use the latest file
                data_path = sorted(csv_files)[-1]
                print(f"[OK] Using existing data: {data_path}")

                # Data analysis (optional, can be skipped)
                if not args.skip_data_analysis:
                    _handle_data_analysis(data_path, dirs["root"])

                # If preprocessing is enabled, preprocess the data
                if not args.skip_preprocessing:
                    processed_dir = dirs["root"] / "data" / "processed"
                    train_path, val_path, test_path = _handle_data_preprocessing(
                        data_path, processed_dir, config
                    )
                    return train_path, None, test_path
                else:
                    return data_path, None, None

            # Fallback to task_data.csv format
            data_path = data_dir / f"{task}_data.csv"
            if data_path.exists():
                print(f"[OK] Using existing data: {data_path}")

                # Data analysis (optional, can be skipped)
                if not args.skip_data_analysis:
                    _handle_data_analysis(data_path, dirs["root"])

                # If preprocessing is enabled, preprocess the data
                if not args.skip_preprocessing:
                    processed_dir = dirs["root"] / "data" / "processed"
                    train_path, val_path, test_path = _handle_data_preprocessing(
                        data_path, processed_dir, config
                    )
                    return train_path, None, test_path
                else:
                    return data_path, None, None

            print(f"[ERROR] Data file not found in: {data_dir}")
            print(f"   Looked for:")
            print(f"     - parameter_sweep_data_*.csv (generated data)")
            print(f"     - train_data_*.csv (preprocessed training data)")
            print(f"     - {task}_data.csv (legacy format)")
            print(f"\n   Available files:")
            all_csv = list(data_dir.glob("*.csv"))
            if all_csv:
                for f in sorted(all_csv, key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
                    print(f"     - {f.name}")
            else:
                print(f"     (no CSV files found)")
            sys.exit(1)
        else:
            print("[ERROR] --data-dir or --data-path required when --skip-data-generation is used")
            sys.exit(1)


def _handle_training(args, config, data_path, dirs):
    """Handle training step."""
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("STEP 2: MODEL TRAINING")
        print("=" * 70)

        # Override epochs if provided via command line
        if args.epochs is not None:
            if "training" not in config:
                config["training"] = {}
            config["training"]["epochs"] = args.epochs
            print(f"[OK] Overriding epochs: {args.epochs} (from command line)")

        seed = config.get("reproducibility", {}).get("random_seed", None)
        use_common_repository = config.get("use_common_repository", True)
        force_retrain = getattr(args, "force_retrain", False)

        return train_model(
            config=config,
            data_path=data_path,
            output_dir=dirs["model"],
            seed=seed,
            use_common_repository=use_common_repository,
            force_retrain=force_retrain,
        )
    else:
        if args.model_path:
            model_path = Path(args.model_path)
            # Resolve to absolute path
            if not model_path.is_absolute():
                model_path = model_path.resolve()
            if not model_path.exists():
                print(f"[WARNING]  Model file not found: {model_path}")
                print(f"   Resolved path: {model_path.resolve()}")
                # Try to find similar model files in the same directory
                if model_path.parent.exists():
                    similar_files = list(model_path.parent.glob("best_model_*.pth"))
                    if similar_files:
                        # Try to find the actual best model by checking validation loss in checkpoints
                        best_file = None
                        best_val_loss = float("inf")

                        print(f"   Searching for best model in {len(similar_files)} model files...")
                        for f in similar_files:
                            try:
                                import torch

                                checkpoint = torch.load(f, map_location="cpu", weights_only=False)
                                val_loss = checkpoint.get(
                                    "best_val_loss", checkpoint.get("val_loss", float("inf"))
                                )
                                if val_loss < best_val_loss:
                                    best_val_loss = val_loss
                                    best_file = f
                            except Exception:
                                # If we can't load the checkpoint, skip it
                                continue

                        if best_file:
                            print(
                                f"[OK] Auto-selected best model (lowest validation loss:"
                                f"{best_val_loss:.6f}): {best_file.name}"
                            )
                            model_path = best_file
                        else:
                            # Fallback: use most recent file
                            sorted_files = sorted(
                                similar_files, key=lambda p: p.stat().st_mtime, reverse=True
                            )
                            print(
                                f"[WARNING] Could not determine best model, using most recent:"
                                f"{sorted_files[0].name}"
                            )
                            model_path = sorted_files[0]
                    else:
                        print(f"[ERROR] No model files found in: {model_path.parent}")
                        sys.exit(1)
                else:
                    print(f"[ERROR] Model directory not found: {model_path.parent}")
                    sys.exit(1)
            else:
                print(f"[OK] Using specified model: {model_path}")
            print(f"[OK] Using existing model: {model_path}")
            return model_path, None
        else:
            # If evaluation is also skipped, model path is not required
            if args.skip_evaluation:
                print("[WARNING]  Skipping training and evaluation - model path not required")
                return None, None
            else:
                print(
                    "[ERROR] --model-path required when --skip-training is used (unless --skip-evaluation is also used)"
                )
                sys.exit(1)


def _handle_evaluation(args, config, model_path, train_data_path, test_data_path, dirs):
    """Handle evaluation step."""
    if not args.skip_evaluation:
        print("\n" + "=" * 70)
        print("STEP 3: MODEL EVALUATION")
        print("=" * 70)

        # Use separate test set if available, otherwise use train data path
        eval_data_path = test_data_path if test_data_path is not None else train_data_path
        if test_data_path is None:
            print(
                "[WARNING]  Warning: No separate test set available, using training data for evaluation"
            )
            print("   (Consider running with --skip-preprocessing=false to get proper test set)")

        return evaluate_model(
            config=config,
            model_path=model_path,
            test_data_path=eval_data_path,
            output_dir=dirs["results"],
        )
    else:
        print("Skipping evaluation (--skip-evaluation)")
        return None


def main():
    """Main experiment workflow."""
    parser = argparse.ArgumentParser(description="Run complete PINN experiment workflow")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment configuration YAML file"
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
        "--skip-data-analysis",
        action="store_true",
        help="Skip data analysis step (default: run analysis after generation)",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip data preprocessing/splitting (training will do internal split)",
    )
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip training (use existing model)"
    )
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with existing data (if skipping generation). "
        "Script will search for parameter_sweep_data_*.csv or train_data_*.csv",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Direct path to data file (if skipping generation). "
        "Overrides --data-dir if both are provided.",
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to existing model (if skipping training)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config file setting)",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force data regeneration even if data exists in common repository",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force model retraining even if model exists in common repository",
    )

    args = parser.parse_args()

    # Load and validate configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    print("=" * 70)
    print("PINN EXPERIMENT WORKFLOW")
    print("=" * 70)
    print(f"Config: {config_path}")

    config = load_config(config_path)

    try:
        validate_config(config)
    except ValueError as e:
        print(f"[ERROR] Invalid configuration: {e}")
        sys.exit(1)

    # Setup experiment
    experiment_id = generate_experiment_id()
    print(f"Experiment ID: {experiment_id}")

    base_dir = Path(args.output_dir)
    dirs = create_experiment_directory(base_dir, experiment_id)
    print(f"Output directory: {dirs['root']}")

    save_config(config, dirs["root"] / "config.yaml")
    tracker = ExperimentTracker(base_dir)

    # Run pipeline steps
    result = _handle_data_generation(args, config, dirs)
    if isinstance(result, tuple) and len(result) == 3:
        # Preprocessing was done, got train_path, validation_results, test_path
        train_data_path, validation_results, test_data_path = result
    else:
        # No preprocessing, got data_path, validation_results
        train_data_path, validation_results = result
        test_data_path = None

    model_path, training_history = _handle_training(args, config, train_data_path, dirs)
    evaluation_results = _handle_evaluation(
        args, config, model_path, train_data_path, test_data_path, dirs
    )

    # Save results
    print("\n" + "=" * 70)
    print("STEP 4: SAVING EXPERIMENT METADATA")
    print("=" * 70)

    results = {
        "training_history": training_history or {},
        "metrics": evaluation_results.get("metrics", {}) if evaluation_results else {},
        "validation": validation_results or {},
    }

    tracker.save_experiment_metadata(
        experiment_id=experiment_id,
        config=config,
        results=results,
        experiment_dir=dirs["root"],
    )

    tracker.log_experiment(
        experiment_id=experiment_id,
        config=config,
        results=results,
        config_path=config_path,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Experiment ID: {experiment_id}")
    print(f"Results saved to: {dirs['root']}")
    print("\nTo view experiment log:")
    print(f"  python scripts/compare_experiments.py --experiments {dirs['root']}")


if __name__ == "__main__":
    main()
