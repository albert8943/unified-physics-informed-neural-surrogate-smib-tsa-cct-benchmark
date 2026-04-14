"""
Training Script for Trajectory Prediction PINN.

This script trains the PINN model for trajectory prediction (δ, ω).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

from pinn.trajectory_prediction import TrajectoryPredictionPINN
from utils.cct_binary_search import estimate_cct_binary_search
from utils.data_utils import TrajectoryDataset, create_dataloader, load_dataset
from utils.metrics import compute_cct_metrics, compute_trajectory_metrics
from utils.visualization import plot_loss_curves, plot_trajectories

# Add parent directory to path


def train_epoch(
    model: Any, dataloader: Any, optimizer: optim.Optimizer, device: torch.device, loss_fn: Any
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {"data": 0.0, "physics": 0.0, "ic": 0.0, "total": 0.0}

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Extract batch data
        # This is a simplified version - actual implementation would need
        # proper batch preparation with collocation points, etc.

        optimizer.zero_grad()

        # Forward pass and loss computation
        # Note: This is a placeholder - actual implementation would be more complex
        # and would include collocation points, physics loss, etc.

        # For now, use a simple MSE loss
        # In practice, this should use the PhysicsInformedLoss
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss, loss_components


def validate(model: Any, dataloader: Any, device: torch.device) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Validation logic
            loss = torch.tensor(0.0, device=device)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train Trajectory Prediction PINN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/trajectory_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory containing training data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save outputs"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument(
        "--validate_cct",
        action="store_true",
        help="Validate CCT estimation using binary search after training",
    )

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
                "input_dim": 10,
                "hidden_dims": [64, 64, 64, 64],
                "activation": "tanh",
                "dropout": 0.0,
            },
            "training": {
                "batch_size": 32,
                "num_epochs": 100,
                "learning_rate": 1e-3,
                "weight_decay": 1e-5,
            },
            "loss": {"lambda_data": 1.0, "lambda_physics": 0.1, "lambda_ic": 10.0},
        }

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data = load_dataset(os.path.join(args.data_dir, "train_data.csv"))
    val_data = load_dataset(os.path.join(args.data_dir, "val_data.csv"))

    # Create datasets and dataloaders
    # Note: This is simplified - actual implementation would need proper
    # feature/target column selection
    feature_columns = [
        "time",
        "delta0",
        "omega0",
        "M",
        "D",
        "Xprefault",
        "Xfault",
        "Xpostfault",
        "tf",
        "tc",
    ]
    target_columns = ["delta", "omega"]

    train_dataset = TrajectoryDataset(train_data, feature_columns, target_columns)
    val_dataset = TrajectoryDataset(val_data, feature_columns, target_columns)

    train_loader = create_dataloader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )

    # Create model
    model = TrajectoryPredictionPINN(
        input_dim=config["model"]["input_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        activation=config["model"]["activation"],
        dropout=config["model"]["dropout"],
    ).to(args.device)

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Training loop
    train_losses = []
    val_losses = []

    num_epochs = config["training"]["num_epochs"]

    print("Starting training for {num_epochs} epochs...")
    print("Device: {args.device}")
    print("Model parameters: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(num_epochs):
        print("\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, loss_components = train_epoch(
            model, train_loader, optimizer, args.device, model.loss_fn
        )
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, args.device)
        val_losses.append(val_loss)

        print("Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

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
            print("Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print("\nFinal model saved: {final_model_path}")

    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, save_path=output_dir / "loss_curves.png", show=False)

    # Optional CCT validation using binary search
    if args.validate_cct:
        print("\n" + "=" * 60)
        print("Validating CCT estimation using binary search...")
        print("=" * 60)

        # Example CCT validation (using sample parameters from validation data)
        if len(val_data) > 0:
            # Get sample parameters from validation data
            sample = val_data.iloc[0]

            # Create time evaluation points
            t_eval = np.linspace(0, 5.0, 500)

            try:
                cct_estimate, info = estimate_cct_binary_search(
                    trajectory_model=model,
                    delta0=float(sample.get("delta0", 0.5)),
                    omega0=float(sample.get("omega0", 1.0)),
                    H=float(sample.get("param_H", 5.0)),
                    D=float(sample.get("param_D", 1.0)),
                    Xprefault=float(sample.get("Xprefault", 0.5)),
                    Xfault=float(sample.get("Xfault", 0.0001)),
                    Xpostfault=float(sample.get("Xpostfault", 0.5)),
                    tf=float(sample.get("tf", 0.1)),
                    t_eval=t_eval,
                    device=args.device,
                    verbose=True,
                )

                print("\nCCT Validation Results:")
                print("  Estimated CCT: {cct_estimate:.4f} s")
                print("  Converged: {info['converged']}")
                print("  Iterations: {info['iterations']}")

            except Exception as e:
                print(f"  CCT validation failed: {e}")
                print(
                    "  Note: This is expected if validation data doesn't "
                    "contain required parameters"
                )
        else:
            print("  No validation data available for CCT validation")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
