#!/usr/bin/env python
"""
Unified Training Script: Train PINN and StandardNN Together.

This script trains both PINN and StandardNN models on the same data,
ensuring fair comparison with identical training conditions.

Usage:
    python scripts/train_unified_models.py \
        --data-path data/common/full_trajectory_data_30_*.csv \
        --output-dir outputs/unified_training \
        --epochs 200 \
        --input-method pe_direct
"""

import argparse
import json
import sys
import io
from pathlib import Path
from typing import Dict, Optional

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

from evaluation.baselines.ml_baselines import MLBaselineTrainer
from scripts.core.utils import generate_experiment_id, create_experiment_directory
from scripts.train_trajectory_pe_input import main as train_pinn
from scripts.train_ml_baselines import main as train_ml_baseline


def train_pinn_model(
    data_path: Path,
    output_dir: Path,
    epochs: int,
    input_method: str,
    config: Dict,
) -> Dict:
    """
    Train PINN model.

    Returns:
    --------
    dict : Training results with model path and metrics
    """
    print("\n" + "=" * 70)
    print("TRAINING PINN MODEL")
    print("=" * 70)

    # Import PINN training function
    import sys
    from scripts.train_trajectory_pe_input import main as train_pinn_main

    # Create PINN output directory
    pinn_dir = output_dir / "pinn"
    pinn_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for PINN training
    # Note: This is a simplified approach - you may need to adjust based on actual PINN training script
    print(f"Training PINN model...")
    print(f"  Data: {data_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Input method: {input_method}")
    print(f"  Output: {pinn_dir}")

    # For now, we'll call the existing training script
    # In practice, you might want to refactor to call training functions directly
    print("\n⚠️  Note: PINN training will be done separately.")
    print("   Please run PINN training manually or integrate PINN training function here.")
    print("   This ensures both models use the same data split.")

    return {
        "model_type": "pinn",
        "status": "pending",
        "output_dir": str(pinn_dir),
    }


def train_standard_nn_model(
    data_path: Path,
    output_dir: Path,
    epochs: int,
    input_method: str,
    config: Dict,
) -> Dict:
    """
    Train StandardNN model.

    Returns:
    --------
    dict : Training results with model path and metrics
    """
    print("\n" + "=" * 70)
    print("TRAINING STANDARDNN MODEL")
    print("=" * 70)

    # Create StandardNN trainer
    trainer = MLBaselineTrainer(
        model_type="standard_nn",
        model_config={
            "hidden_dims": [256, 256, 128, 128],
            "activation": "tanh",
            "dropout": 0.0,
        },
    )

    # Prepare data
    print("\nPreparing data...")
    train_loader, val_loader, scalers = trainer.prepare_data(
        data_path=data_path,
        input_method=input_method,
    )

    # Train model
    print(f"\nTraining StandardNN for {epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
    )

    # Evaluate
    print("\nEvaluating StandardNN...")
    metrics = trainer.evaluate(val_loader)

    # Save model
    model_dir = output_dir / "standard_nn"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "model_type": "standard_nn",
            "model_config": trainer.model_config,
            "scalers": scalers,
            "input_method": input_method,
        },
        model_path,
    )

    # Save metrics
    metrics_file = model_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(
            {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in metrics.items()
            },
            f,
            indent=2,
        )

    print(f"\n✓ StandardNN training complete")
    print(f"  Model: {model_path}")
    print(f"  Metrics: {metrics_file}")
    print(f"  R² Delta: {metrics.get('delta_r2', 0):.4f}")
    print(f"  R² Omega: {metrics.get('omega_r2', 0):.4f}")

    return {
        "model_type": "standard_nn",
        "status": "completed",
        "model_path": str(model_path),
        "metrics": metrics,
        "training_history": history,
    }


def main():
    """Main unified training workflow."""
    parser = argparse.ArgumentParser(
        description="Train PINN and StandardNN models together for fair comparison"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/unified_training",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--input-method",
        type=str,
        default="pe_direct",
        choices=["reactance", "pe_direct"],
        help="Input method",
    )
    parser.add_argument(
        "--skip-pinn",
        action="store_true",
        help="Skip PINN training (only train StandardNN)",
    )
    parser.add_argument(
        "--skip-standard-nn",
        action="store_true",
        help="Skip StandardNN training (only train PINN)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("UNIFIED MODEL TRAINING")
    print("=" * 70)
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Input method: {args.input_method}")

    # Handle wildcard patterns in data path
    data_path_str = args.data_path
    if "*" in data_path_str:
        import glob

        matching_files = glob.glob(data_path_str)
        if not matching_files:
            raise FileNotFoundError(f"No files match pattern: {data_path_str}")
        data_path = Path(matching_files[0])
        print(f"Using data file: {data_path}")
    else:
        data_path = Path(data_path_str)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Create experiment directory
    experiment_id = generate_experiment_id()
    experiment_dir = Path(args.output_dir) / f"exp_{experiment_id}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "experiment_id": experiment_id,
        "data_path": str(data_path),
        "epochs": args.epochs,
        "input_method": args.input_method,
        "skip_pinn": args.skip_pinn,
        "skip_standard_nn": args.skip_standard_nn,
    }
    config_file = experiment_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nExperiment ID: {experiment_id}")
    print(f"Experiment directory: {experiment_dir}")

    # Train models
    results = {}

    # Train StandardNN
    if not args.skip_standard_nn:
        results["standard_nn"] = train_standard_nn_model(
            data_path=data_path,
            output_dir=experiment_dir,
            epochs=args.epochs,
            input_method=args.input_method,
            config=config,
        )

    # Train PINN
    if not args.skip_pinn:
        results["pinn"] = train_pinn_model(
            data_path=data_path,
            output_dir=experiment_dir,
            epochs=args.epochs,
            input_method=args.input_method,
            config=config,
        )

    # Save summary
    summary_file = experiment_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                k: {kk: vv for kk, vv in v.items() if kk != "training_history"}
                for k, v in results.items()
            },
            f,
            indent=2,
            default=str,
        )

    print("\n" + "=" * 70)
    print("UNIFIED TRAINING COMPLETE")
    print("=" * 70)
    print(f"Experiment ID: {experiment_id}")
    print(f"Results saved to: {experiment_dir}")
    print(f"  Config: {config_file}")
    print(f"  Summary: {summary_file}")

    if not args.skip_standard_nn:
        print(f"\nStandardNN Model:")
        print(f"  {results['standard_nn']['model_path']}")

    if not args.skip_pinn:
        print(f"\nPINN Model:")
        print(f"  {results['pinn']['output_dir']}")
        print(f"  ⚠️  Note: PINN training needs to be completed separately")

    print("\nNext steps:")
    print("1. Complete PINN training if skipped")
    print("2. Use scripts/compare_models.py to compare both models")
    print("3. Use scripts/evaluate_model.py with --include-standard-nn for unified plots")


if __name__ == "__main__":
    main()
