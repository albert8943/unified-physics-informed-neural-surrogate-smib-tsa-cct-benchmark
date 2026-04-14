"""
Training Script for CCT Estimation PINN.

.. deprecated:: 2.0
    This script is deprecated. CCT estimation is now performed using binary search
    with the trajectory model. See utils.cct_binary_search.estimate_cct_binary_search()
    for the new approach.

    To estimate CCT:
    1. Train a trajectory prediction model using train_trajectory.py
    2. Use utils.cct_binary_search.estimate_cct_binary_search() with the trained model

This script is kept for reference and backward compatibility.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

# Issue deprecation warning
warnings.warn(
    "train_cct.py is deprecated. CCT estimation is now performed using binary search "
    "with the trajectory model. Train a trajectory model and use "
    "utils.cct_binary_search.estimate_cct_binary_search() instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Add parent directory to path

from pinn.cct_estimation import CCTEstimationPINN
from utils.metrics import compute_cct_metrics
from utils.visualization import plot_comparison


def train_epoch(
    model: nn.Module,
    dataloader: Any,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        features, targets = batch
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(features)

        # Compute loss
        loss = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(
    model: nn.Module, dataloader: Any, criterion: nn.Module, device: torch.device
) -> Tuple[float, List[Any], List[Any]]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            features, targets = batch
            features = features.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(features)

            # Compute loss
            loss = criterion(predictions, targets)
            total_loss += loss.item()

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Compute metrics
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    metrics = compute_cct_metrics(predictions.flatten(), targets.flatten())

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train CCT Estimation PINN (DEPRECATED)",
        epilog="Note: This script is deprecated. Use binary search with trajectory model instead. "
        "See utils.cct_binary_search.estimate_cct_binary_search() for the new approach.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/training/cct_config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory containing training data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save outputs"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")

    args = parser.parse_args()

    # Print deprecation notice
    print("=" * 60)
    print("DEPRECATION WARNING")
    print("=" * 60)
    print("This script is deprecated. CCT estimation is now performed using")
    print("binary search with the trajectory model.")
    print("\nTo estimate CCT:")
    print("1. Train a trajectory prediction model: python training/train_trajectory.py")
    print("2. Use binary search: from utils.cct_binary_search import estimate_cct_binary_search")
    print("\nSee examples/cct_pinn_example.py for usage examples.")
    print("=" * 60)
    print()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            "model": {
                "input_dim": 8,
                "hidden_dims": [64, 64, 64],
                "activation": "tanh",
                "dropout": 0.0,
            },
            "training": {
                "batch_size": 32,
                "num_epochs": 100,
                "learning_rate": 1e-3,
                "weight_decay": 1e-5,
            },
        }

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data = pd.read_csv(os.path.join(args.data_dir, "train_data.csv"))
    val_data = pd.read_csv(os.path.join(args.data_dir, "val_data.csv"))

    # Create model
    model = CCTEstimationPINN(
        input_dim=config["model"]["input_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        activation=config["model"]["activation"],
        dropout=config["model"]["dropout"],
    ).to(args.device)

    # Create optimizer and criterion
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.MSELoss()

    # Training loop
    train_losses = []
    val_losses = []

    num_epochs = config["training"]["num_epochs"]

    print("Starting training for {num_epochs} epochs...")
    print("Device: {args.device}")

    for epoch in range(num_epochs):
        print("\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, None, optimizer, criterion, args.device)
        train_losses.append(train_loss)

        # Validate
        val_loss, metrics = validate(model, None, criterion, args.device)
        val_losses.append(val_loss)

        print("Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print("CCT MAE: {metrics['cct_mae']:.6f} s ({metrics['cct_mae_ms']:.2f} ms)")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / "checkpoint_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print("\nFinal model saved: {final_model_path}")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
