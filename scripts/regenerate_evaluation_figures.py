#!/usr/bin/env python
"""
Regenerate Evaluation Figures for ML Baseline Model.

This script reloads a trained ML baseline model, regenerates predictions,
and recreates evaluation figures with updated visualization code (e.g., fixed labels).

Usage:
    python scripts/regenerate_evaluation_figures.py \
        --model-path outputs/ml_baselines/exp_20251215_160632/standard_nn/model.pth \
        --data-path data/common/full_trajectory_data_30_*.csv \
        --output-dir outputs/ml_baselines/exp_20251215_160632/standard_nn/figures/evaluation
"""

import argparse
import json
import sys
import io
from pathlib import Path
from typing import Dict, Optional, Tuple

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from evaluation.baselines.ml_baselines import MLBaselineTrainer, StandardNN, LSTMModel
from utils.metrics import compute_trajectory_metrics
from visualization.publication_figures import generate_evaluation_figures


def load_model_and_scalers(
    model_path: Path,
    device: str = "auto",
) -> Tuple[torch.nn.Module, Dict, str]:
    """
    Load ML baseline model and scalers from checkpoint.

    Parameters:
    -----------
    model_path : Path
        Path to model checkpoint (.pth file)
    device : str
        Device to use ("auto", "cpu", "cuda")

    Returns:
    --------
    model : torch.nn.Module
        Loaded model
    scalers : dict
        Dictionary of scalers for input features
    input_method : str
        Input method used ("reactance" or "pe_direct")
    """
    print(f"Loading model from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    model_type = checkpoint.get("model_type", "standard_nn")
    model_config = checkpoint.get("model_config", {})
    scalers = checkpoint.get("scalers", {})
    input_method = checkpoint.get("input_method", "reactance")

    # Handle device
    if device == "auto":
        device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_torch = torch.device(device)

    print(f"  Model type: {model_type}")
    print(f"  Input method: {input_method}")
    print(f"  Device: {device_torch}")

    # Build model
    if model_type == "standard_nn":
        # Determine input dimension based on input method
        if input_method == "pe_direct_7":
            input_dim = 7
        elif input_method == "pe_direct":
            input_dim = 9  # [t, delta0, omega0, H, D, Pload, Pe, tf, tc]
        else:  # reactance
            input_dim = 11  # [t, delta0, omega0, H, D, Pm, X, E, V, theta, Pe]

        model = StandardNN(
            input_dim=input_dim,
            hidden_dims=model_config.get("hidden_dims", [256, 256, 128, 128]),
            output_dim=2,
            activation=model_config.get("activation", "tanh"),
            dropout=model_config.get("dropout", 0.0),
        )
    elif model_type == "lstm":
        model = LSTMModel(
            input_dim=model_config.get("input_dim", 9),
            hidden_size=model_config.get("hidden_size", 128),
            num_layers=model_config.get("num_layers", 2),
            output_dim=2,
            dropout=model_config.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device_torch)
    model.eval()

    print(f"  ✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    return model, scalers, input_method, device_torch


def prepare_data_for_prediction(
    data: pd.DataFrame,
    scalers: Dict,
    input_method: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare data for model prediction.

    Parameters:
    -----------
    data : pd.DataFrame
        Data with columns: t, delta0, omega0, H, D, Pm, Pe, (and optionally X, E, V, theta)
    scalers : dict
        Dictionary of scalers for normalization
    input_method : str
        Input method ("reactance" or "pe_direct")

    Returns:
    --------
    X : torch.Tensor
        Input features (normalized)
    y : torch.Tensor
        Target values (delta, omega)
    """
    # Extract features based on input method
    if input_method == "pe_direct_7":
        feature_cols = ["time", "delta0", "omega0", "H", "D", "Pm", "Pe"]
    elif input_method == "pe_direct":
        feature_cols = ["time", "delta0", "omega0", "H", "D", "Pm", "Pe", "tf", "tc"]
    else:  # reactance
        feature_cols = [
            "t",
            "delta0",
            "omega0",
            "H",
            "D",
            "Pm",
            "X",
            "E",
            "V",
            "theta",
            "Pe",
        ]

    # Extract targets
    target_cols = ["delta", "omega"]

    # Prepare features
    X = data[feature_cols].values.astype(np.float32)
    y = data[target_cols].values.astype(np.float32)

    # Normalize features
    for i, col in enumerate(feature_cols):
        if col in scalers:
            X[:, i] = scalers[col].transform(X[:, i].reshape(-1, 1)).flatten()

    # Convert to tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    return X_tensor, y_tensor


def main():
    """Main regeneration workflow."""
    parser = argparse.ArgumentParser(
        description="Regenerate evaluation figures for ML baseline model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to data file (used for validation set)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for figures",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data splitting (default: 42)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for prediction (default: 32)",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("REGENERATING EVALUATION FIGURES")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")

    # Load model
    model, scalers, input_method, device = load_model_and_scalers(model_path)

    # Load data
    print(f"\nLoading data from: {data_path}")
    if "*" in str(data_path):
        # Handle wildcard pattern
        import glob

        matching_files = glob.glob(str(data_path))
        if not matching_files:
            raise FileNotFoundError(f"No files match pattern: {data_path}")
        data_path = Path(matching_files[0])
        print(f"  Using: {data_path}")

    data = pd.read_csv(data_path)
    print(f"  ✓ Loaded {len(data):,} rows")

    # Split data (use same split as training: 15% for validation)
    from sklearn.model_selection import train_test_split

    scenarios = data["scenario_id"].unique()
    train_scenarios, val_scenarios = train_test_split(
        scenarios, test_size=args.val_split, random_state=args.seed
    )
    val_data = data[data["scenario_id"].isin(val_scenarios)]
    print(f"  ✓ Validation set: {len(val_scenarios)} scenarios, {len(val_data):,} rows")

    # Prepare data
    print("\nPreparing data for prediction...")
    X, y = prepare_data_for_prediction(val_data, scalers, input_method)

    # Generate predictions
    print("Generating predictions...")
    model.eval()
    all_pred = []
    all_true = []

    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            pred = model(batch_X)
            all_pred.append(pred.cpu().numpy())
            all_true.append(batch_y.numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    delta_pred = all_pred[:, 0]
    omega_pred = all_pred[:, 1]
    delta_true = all_true[:, 0]
    omega_true = all_true[:, 1]

    print(f"  ✓ Generated {len(delta_pred):,} predictions")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_trajectory_metrics(
        delta_pred=delta_pred,
        omega_pred=omega_pred,
        delta_true=delta_true,
        omega_true=omega_true,
    )

    print(f"  R² Delta: {metrics.get('delta_r2', 0):.4f}")
    print(f"  R² Omega: {metrics.get('omega_r2', 0):.4f}")
    print(f"  RMSE Delta: {metrics.get('delta_rmse', 0):.4f}")
    print(f"  RMSE Omega: {metrics.get('omega_rmse', 0):.4f}")

    # Prepare evaluation results
    evaluation_results = {
        "metrics": metrics,
        "predictions": {
            "delta": delta_pred,
            "omega": omega_pred,
        },
        "targets": {
            "delta": delta_true,
            "omega": omega_true,
        },
    }

    # Load config if available
    config = {}
    config_file = model_path.parent.parent / "config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)

    # Load checkpoint again to get model type
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Add model info to config
    config["model_type"] = checkpoint.get("model_type", "standard_nn")
    config["input_method"] = input_method

    # Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        figure_paths = generate_evaluation_figures(
            evaluation_results=evaluation_results,
            config=config,
            output_dir=output_dir,
            figure_formats=["png"],
            dpi=300,
        )

        print(f"\n✓ Generated {len(figure_paths)} figure(s)")
        for name, path in figure_paths.items():
            print(f"  ✓ {name}: {path.name}")

    except Exception as e:
        print(f"\n❌ Error generating figures: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("REGENERATION COMPLETE")
    print("=" * 70)
    print(f"Figures saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
