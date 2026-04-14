"""
Training Script for Parameter Estimation PINN.

This script trains the PINN model for parameter estimation (H, D).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from pinn.parameter_estimation import ParameterEstimationPINN
from utils.metrics import compute_parameter_metrics
from utils.visualization import plot_comparison

# Add parent directory to path


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
    metrics = compute_parameter_metrics(
        predictions[:, 0], predictions[:, 1], targets[:, 0], targets[:, 1]
    )

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Parameter Estimation PINN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/parameter_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory containing training data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save outputs"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            "model": {
                "input_dim": 2,
                "sequence_length": 100,
                "hidden_dims": [128, 128, 64],
                "activation": "tanh",
                "dropout": 0.0,
                "use_lstm": True,
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
    model = ParameterEstimationPINN(
        input_dim=config["model"]["input_dim"],
        sequence_length=config["model"]["sequence_length"],
        hidden_dims=config["model"]["hidden_dims"],
        activation=config["model"]["activation"],
        dropout=config["model"]["dropout"],
        use_lstm=config["model"]["use_lstm"],
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
        print("H MAPE: {metrics['H_mape']:.2f}%, D MAPE: {metrics['D_mape']:.2f}%")

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
