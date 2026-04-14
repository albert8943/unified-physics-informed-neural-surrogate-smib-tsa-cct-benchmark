#!/usr/bin/env python
"""
Train ML Baseline Models.

Trains standard NN and LSTM models without physics constraints for comparison.

Usage:
    python scripts/train_ml_baselines.py \
        --data-path data/generated/quick_test/parameter_sweep_data.csv \
        --output-dir outputs/ml_baselines \
        --models standard_nn lstm
"""

import argparse
import sys
import io
from pathlib import Path
from typing import List

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments
from evaluation.baselines.ml_baselines import MLBaselineTrainer

# Import utils first to avoid circular import
from scripts.core.utils import generate_experiment_id


def convert_to_serializable(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def main():
    """Main ML baseline training workflow."""
    parser = argparse.ArgumentParser(description="Train ML baseline models")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ml_baselines",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["standard_nn", "lstm"],
        choices=["standard_nn", "lstm"],
        help="Models to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--input-method",
        type=str,
        default="pe_direct",  # Default to pe_direct (9 dimensions: Pe + tf, tc)
        choices=["reactance", "pe_direct", "pe_direct_7"],
        help="Input method: 'pe_direct' (9 dims), 'pe_direct_7' (7 dims), or 'reactance' (11 dims)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for regularization (default: 0.2, recommended: 0.2-0.5)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization) for optimizer (default: 1e-4, recommended: 1e-4 to 1e-3)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs to wait before stopping, default: 20, use 0 to disable)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ML BASELINE TRAINING")
    print("=" * 70)
    print(f"Data: {args.data_path}")
    print(f"Models: {args.models}")
    print(f"Output: {args.output_dir}")

    # Handle wildcard patterns in data path (must be done before config creation)
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
            sys.exit(1)
        # Use the latest matching file (by modification time)
        data_path = max(matching_files, key=lambda p: p.stat().st_mtime)
        print(
            f"Found {len(matching_files)} file(s) matching pattern, using latest: {data_path.name}"
        )
    else:
        data_path = Path(args.data_path)
        if not data_path.exists():
            print(f"[ERROR] Data file not found: {data_path}")
            sys.exit(1)

    # Create timestamped experiment directory (after data_path is resolved)
    experiment_id = generate_experiment_id()  # Returns "exp_YYYYMMDD_HHMMSS"
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Create experiment folder: ml_baselines/exp_YYYYMMDD_HHMMSS/
    experiment_dir = base_output_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment ID: {experiment_id}")
    print(f"Experiment directory: {experiment_dir}")

    # Save experiment config (now data_path is defined)
    config = {
        "experiment_id": experiment_id,
        "data_path": str(data_path),
        "models": args.models,
        "epochs": args.epochs,
        "input_method": args.input_method,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "early_stopping_patience": (
            args.early_stopping_patience if args.early_stopping_patience > 0 else None
        ),
    }
    config_file = experiment_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    output_dir = experiment_dir  # Use experiment directory for all outputs

    # Train each model
    results = {}

    for model_type in args.models:
        print("\n" + "=" * 70)
        print(f"TRAINING {model_type.upper()}")
        print("=" * 70)

        # Create trainer
        trainer = MLBaselineTrainer(
            model_type=model_type,
            model_config={
                "hidden_dims": [256, 256, 128, 128] if model_type == "standard_nn" else None,
                "hidden_size": 128 if model_type == "lstm" else None,
                "num_layers": 2 if model_type == "lstm" else None,
                "activation": "tanh",
                "dropout": args.dropout,  # Use configurable dropout (default: 0.2)
            },
        )

        # Prepare data
        print("\nPreparing data...")
        train_loader, val_loader, scalers = trainer.prepare_data(
            data_path=data_path,
            input_method=args.input_method,
        )

        # Train model with regularization
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=1e-3,  # Default learning rate
            weight_decay=args.weight_decay,  # Use configurable weight decay (default: 1e-4)
            early_stopping_patience=(
                args.early_stopping_patience if args.early_stopping_patience > 0 else None
            ),  # Early stopping
        )

        # Evaluate on validation set
        print("\nEvaluating model...")
        metrics = trainer.evaluate(val_loader)

        # Get predictions for visualization
        print("\nCollecting predictions for visualization...")
        trainer.model.eval()
        all_delta_pred = []
        all_omega_pred = []
        all_delta_true = []
        all_omega_true = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(trainer.device)
                pred = trainer.model(batch_X)

                all_delta_pred.extend(pred[:, 0].cpu().numpy())
                all_omega_pred.extend(pred[:, 1].cpu().numpy())
                all_delta_true.extend(batch_y[:, 0].cpu().numpy())
                all_omega_true.extend(batch_y[:, 1].cpu().numpy())

        # Save model (best model is already loaded in trainer.model at this point)
        model_dir = output_dir / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pth"
        torch.save(
            {
                "model_state_dict": trainer.model.state_dict(),
                "model_type": model_type,
                "model_config": trainer.model_config,
                "scalers": scalers,
                "input_method": args.input_method,
                "training_history": history,  # Save training history in checkpoint
            },
            model_path,
        )

        # Save training history as separate JSON file (like PINN)
        from scripts.core.utils import generate_timestamped_filename

        history_filename = generate_timestamped_filename("training_history", "json")
        history_path = model_dir / history_filename
        with open(history_path, "w") as f:
            json.dump(convert_to_serializable(history), f, indent=2)
        print(f"  Training history: {history_path.name}")

        # Save configuration (similar to PINN's config.yaml)
        from scripts.core.utils import save_config

        # ML baseline loss weights: lambda_data=1.0 (implicit, MSE loss), lambda_ic=10.0 (matches PINN)
        # Note: ML baseline doesn't have physics loss (no lambda_physics)
        ml_config = {
            "model_type": model_type,
            "model_config": trainer.model_config,
            "input_method": args.input_method,
            "training": {
                "epochs": args.epochs,
                "learning_rate": 1e-3,  # Default learning rate
                "weight_decay": args.weight_decay,  # Configurable weight decay
                "early_stopping_patience": (
                    args.early_stopping_patience if args.early_stopping_patience > 0 else None
                ),  # Early stopping
            },
            "loss": {
                "lambda_data": 1.0,  # Implicit weight for data loss (MSE)
                "lambda_ic": 10.0,  # Initial condition loss weight (matches PINN)
                "lambda_physics": None,  # ML baseline doesn't use physics loss
                "loss_function": "MSE",
                "total_loss": "data_loss + lambda_ic * ic_loss",
            },
            "data": {
                "data_path": str(data_path),
            },
            "experiment_id": experiment_id,
        }
        config_file = model_dir / "config.yaml"
        save_config(ml_config, config_file)
        print(f"  Config: {config_file.name}")

        # Save results
        results[model_type] = {
            "metrics": metrics,
            "training_history": history,
            "model_path": str(model_path),
        }

        # Save metrics
        metrics_file = model_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(convert_to_serializable(metrics), f, indent=2)

        # Generate figures
        print("\n" + "=" * 70)
        print("GENERATING FIGURES")
        print("=" * 70)

        figures_dir = model_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Create config dict for figure generation (update with model-specific info)
        figure_config = {
            **config,  # Include experiment config
            "model_type": model_type,
        }

        # Generate training figures (lazy import to avoid circular dependency)
        # Save directly to figures_dir (flattened structure - matching PINN)
        try:
            print("\nGenerating training figures...")
            from visualization.publication_figures import generate_training_figures

            training_figures = generate_training_figures(
                training_history=history,
                config=figure_config,
                output_dir=figures_dir,  # Save directly to figures/ (no training/ subdirectory)
                figure_formats=["png"],
                dpi=300,
            )
            print(f"  ✓ Generated {len(training_figures)} training figure(s)")
        except Exception as e:
            print(f"  ⚠️  Could not generate training figures: {e}")
            import traceback

            traceback.print_exc()
            training_figures = {}

        # Generate evaluation figures (lazy import to avoid circular dependency)
        try:
            print("\nGenerating evaluation figures...")
            from visualization.publication_figures import generate_evaluation_figures

            evaluation_results = {
                "metrics": metrics,
                "predictions": {
                    "delta": np.array(all_delta_pred),
                    "omega": np.array(all_omega_pred),
                },
                "targets": {
                    "delta": np.array(all_delta_true),
                    "omega": np.array(all_omega_true),
                },
            }
            eval_figures = generate_evaluation_figures(
                evaluation_results=evaluation_results,
                config=figure_config,
                output_dir=figures_dir,  # Save directly to figures/ (no evaluation/ subdirectory)
                figure_formats=["png"],
                dpi=300,
            )
            print(f"  ✓ Generated {len(eval_figures)} evaluation figure(s)")
        except Exception as e:
            print(f"  ⚠️  Could not generate evaluation figures: {e}")
            import traceback

            traceback.print_exc()
            eval_figures = {}

        # Save figure paths
        figure_paths = {**training_figures, **eval_figures}
        if figure_paths:
            figure_index_path = figures_dir / "figure_index.json"
            with open(figure_index_path, "w") as f:
                json.dump({k: str(v) for k, v in figure_paths.items()}, f, indent=2)
            print(f"\n  ✓ Figure index saved to: {figure_index_path}")
            print(f"  ✓ Total figures: {len(figure_paths)}")
            print(f"  ✓ Figures directory: {figures_dir}")

        print(f"\n✓ {model_type} training complete")
        print(f"  Model saved to: {model_path}")
        print(f"  Metrics:")
        print(f"    R² Delta: {metrics.get('delta_r2', 0):.4f}")
        print(f"    R² Omega: {metrics.get('omega_r2', 0):.4f}")
        print(f"    RMSE Delta: {metrics.get('delta_rmse', 0):.4f}")
        print(f"    RMSE Omega: {metrics.get('omega_rmse', 0):.4f}")

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    print("\n" + "=" * 70)
    print("ML BASELINE TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nExperiment ID: {experiment_id}")
    print(f"Results saved to: {experiment_dir}")
    print(f"  Config: {config_file}")
    print(f"  Summary: {summary_file}")
    print(f"\nDirectory structure:")
    print(f"  {experiment_dir}/")
    for model_type in args.models:
        print(f"    {model_type}/")
        print(f"      model.pth")
        print(f"      metrics.json")
        print(f"      figures/")
        print(f"        training/")
        print(f"        evaluation/")


if __name__ == "__main__":
    main()
