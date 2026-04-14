"""
Training Script for Trajectory Prediction PINN with Pe(t) as Input.

This script trains a PINN model using Pe(t) directly from ANDES as input,
eliminating the need to compute electrical power from reactances.

Usage:
    python training/train_trajectory_pe_input.py --data-dir data --output-dir outputs/models
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pinn.core import AdaptiveLossWeightScheduler, PhysicsInformedLoss
from pinn.trajectory_prediction import TrajectoryPredictionPINN_PeInput
from utils.normalization import (
    PhysicsNormalizer,
    set_model_standardization_to_identity,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train Trajectory Prediction PINN with Pe(t) Input"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/models", help="Directory to save trained model"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size (auto-detect if None)"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")

    args = parser.parse_args()

    # Setup
    device = setup_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PINN Trajectory Prediction Training (Pe(t) Input Approach)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Load and prepare data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train_loader, val_loader, scalers = load_data_with_pe_input(args.data_dir, device=device)

    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")

    # Initialize model
    print("\n" + "=" * 70)
    print("PHASE 1: MODEL INITIALIZATION")
    print("=" * 70)

    model = TrajectoryPredictionPINN_PeInput(
        input_dim=7,  # [t, δ₀, ω₀, H, D, Pm, Pe(t)]
        hidden_dims=[64, 64, 64, 64],
        activation="tanh",
        use_residual=True,
        dropout=0.0,
        use_standardization=True,
    ).to(device)

    # CRITICAL: Set model standardization to identity
    set_model_standardization_to_identity(model, 7, 2, device)

    print(f"✓ Model initialized (input_dim=7 for Pe(t) input)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize loss with physics normalizer
    print("\n" + "=" * 70)
    print("PHASE 2: LOSS FUNCTION SETUP")
    print("=" * 70)

    physics_normalizer = PhysicsNormalizer(scalers, device=device)

    loss_fn = PhysicsInformedLoss(
        lambda_data=1.0,
        lambda_physics=0.0,  # Controlled by adaptive scheduler
        lambda_ic=10.0,
        fn=60.0,
        physics_normalizer=physics_normalizer,
        use_normalized_loss=True,
        scale_to_norm=torch.tensor([[1.0, 100.0]]),
        use_pe_direct=True,  # Use Pe(t) directly from ANDES
    )

    print("✓ Physics-informed loss initialized")
    print("  Data weight: 1.0")
    print("  Physics weight: Adaptive (scheduler controlled)")
    print("  IC weight: 10.0")
    print("  Pe direct mode: Enabled (uses Pe(t) from ANDES)")

    # Setup adaptive scheduler
    print("\n" + "=" * 70)
    print("PHASE 3: TRAINING STRATEGY")
    print("=" * 70)

    adaptive_scheduler = AdaptiveLossWeightScheduler(
        initial_ratio=0.0,
        final_ratio=0.5,
        warmup_epochs=30,
    )

    print("✓ Adaptive loss weight scheduler initialized")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    # Auto-detect batch size
    if args.batch_size is None:
        args.batch_size = 64 if device.type == "cuda" else 16

    print(f"✓ Optimizer: Adam (lr={args.lr})")
    print(f"✓ Batch size: {args.batch_size}")

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    start_time = time.time()
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Training step
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            adaptive_scheduler,
            physics_normalizer,
            epoch,
            device,
        )

        # Validation step
        val_loss, val_metrics = validate_epoch(
            model, val_loader, loss_fn, physics_normalizer, device
        )

        # Learning rate scheduling
        scheduler_lr.step(val_loss)

        # Diagnostics
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print_diagnostics(epoch, train_metrics, val_metrics, optimizer)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scalers, epoch, val_loss, output_dir / "best_model.pth"
            )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")


def setup_device(device_str: str) -> torch.device:
    """Setup device (cuda/cpu/auto)."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    return device


def load_data_with_pe_input(
    data_path: Path, device: str = "cpu"
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Load data with Pe(t) as input.

    Returns:
    --------
    train_loader, val_loader, scalers
    """
    data_path = Path(data_path)
    if data_path.is_dir():
        # Look for CSV file in directory
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        data_file = csv_files[0]  # Use first CSV file
    else:
        data_file = data_path

    # Load data
    data = pd.read_csv(data_file)

    # Verify Pe column exists
    if "Pe" not in data.columns:
        raise ValueError(
            "Pe column not found in data. Use use_pe_as_input=True during data generation."
        )

    # Extract features and targets
    # Input: [t, delta0, omega0, H, D, Pm, Pe(t)]
    # Target: [delta, omega]

    # Group by scenario for proper train/val split
    scenario_ids = data["scenario_id"].unique()
    np.random.seed(42)
    np.random.shuffle(scenario_ids)
    split_idx = int(0.8 * len(scenario_ids))
    train_scenarios = scenario_ids[:split_idx]
    val_scenarios = scenario_ids[split_idx:]

    train_data = data[data["scenario_id"].isin(train_scenarios)]
    val_data = data[data["scenario_id"].isin(val_scenarios)]

    # Create scalers
    scalers = {}
    input_cols = ["time", "delta0", "omega0", "H", "D", "Pm", "Pe"]

    for col in input_cols:
        if col in train_data.columns:
            scaler = StandardScaler()
            # Use .values to convert to numpy array to avoid feature name warnings
            # For Pe, normalize all time points
            if col == "Pe":
                Pe_all = train_data[col].values
                scaler.fit(Pe_all.reshape(-1, 1))
            else:
                scaler.fit(train_data[col].values.reshape(-1, 1))
            scalers[col] = scaler

    # Normalize and prepare tensors
    def prepare_tensors(df, scalers):
        inputs = []
        for col in input_cols:
            if col in df.columns:
                values = df[col].values
                if col == "Pe":
                    values = values.reshape(-1, 1)
                normalized = scalers[col].transform(
                    values.reshape(-1, 1) if col != "Pe" else values.reshape(-1, 1)
                )
                inputs.append(torch.tensor(normalized, dtype=torch.float32))

        # Stack inputs: [batch, 7]
        X = torch.stack(inputs, dim=1).squeeze(-1)

        # Targets
        delta = torch.tensor(df["delta"].values, dtype=torch.float32)
        omega = torch.tensor(df["omega"].values, dtype=torch.float32)
        y = torch.stack([delta, omega], dim=1)

        return X, y

    X_train, y_train = prepare_tensors(train_data, scalers)
    X_val, y_val = prepare_tensors(val_data, scalers)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, scalers


def train_epoch(
    model,
    train_loader,
    loss_fn,
    optimizer,
    adaptive_scheduler,
    physics_normalizer,
    epoch,
    device,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_physics_loss = 0.0
    total_ic_loss = 0.0
    n_batches = 0

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Extract inputs: [t, delta0, omega0, H, D, Pm, Pe]
        t = X_batch[:, 0]
        delta0 = X_batch[:, 1]
        omega0 = X_batch[:, 2]
        H = X_batch[:, 3]
        D = X_batch[:, 4]
        Pm = X_batch[:, 5]
        Pe = X_batch[:, 6]

        # Forward pass
        output = model(X_batch)
        delta_pred = output[:, 0]
        omega_pred = output[:, 1]

        # Compute loss
        # For Pe input, we need to pass Pe_from_andes to loss
        M = 2.0 * H  # Convert H to M

        loss_dict = loss_fn(
            t_data=t,
            delta_pred=delta_pred,
            omega_pred=omega_pred,
            delta_obs=y_batch[:, 0],
            omega_obs=y_batch[:, 1],
            t_ic=t[0:1] if len(t) > 0 else t,
            delta0=delta0,
            omega0=omega0,
            M=M,
            D=D,
            Pm=Pm,
            Pe_from_andes=Pe,  # Pass Pe directly
            use_pe_direct=True,
        )

        # Update physics weight
        if adaptive_scheduler is not None:
            lambda_physics = adaptive_scheduler.compute_weight(
                loss_dict["data"], loss_dict.get("physics", torch.tensor(0.0)), epoch
            )
            loss_fn.lambda_physics = lambda_physics

        loss = (
            loss_dict["data"]
            + loss_fn.lambda_physics * loss_dict.get("physics", torch.tensor(0.0))
            + loss_fn.lambda_ic * loss_dict.get("ic", torch.tensor(0.0))
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_data_loss += loss_dict["data"].item()
        total_physics_loss += loss_dict.get("physics", torch.tensor(0.0)).item()
        total_ic_loss += loss_dict.get("ic", torch.tensor(0.0)).item()
        n_batches += 1

    metrics = {
        "loss": total_loss / n_batches,
        "data_loss": total_data_loss / n_batches,
        "physics_loss": total_physics_loss / n_batches,
        "ic_loss": total_ic_loss / n_batches,
    }

    return metrics["loss"], metrics


def validate_epoch(model, val_loader, loss_fn, physics_normalizer, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_data_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Extract inputs
            t = X_batch[:, 0]
            delta0 = X_batch[:, 1]
            omega0 = X_batch[:, 2]
            H = X_batch[:, 3]
            D = X_batch[:, 4]
            Pm = X_batch[:, 5]
            Pe = X_batch[:, 6]

            # Forward pass
            output = model(X_batch)
            delta_pred = output[:, 0]
            omega_pred = output[:, 1]

            # Compute loss
            M = 2.0 * H

            loss_dict = loss_fn(
                t_data=t,
                delta_pred=delta_pred,
                omega_pred=omega_pred,
                delta_obs=y_batch[:, 0],
                omega_obs=y_batch[:, 1],
                t_ic=t[0:1] if len(t) > 0 else t,
                delta0=delta0,
                omega0=omega0,
                M=M,
                D=D,
                Pm=Pm,
                Pe_from_andes=Pe,
                use_pe_direct=True,
            )

            loss = (
                loss_dict["data"]
                + loss_fn.lambda_physics * loss_dict.get("physics", torch.tensor(0.0))
                + loss_fn.lambda_ic * loss_dict.get("ic", torch.tensor(0.0))
            )

            total_loss += loss.item()
            total_data_loss += loss_dict["data"].item()
            n_batches += 1

    metrics = {
        "loss": total_loss / n_batches,
        "data_loss": total_data_loss / n_batches,
    }

    return metrics["loss"], metrics


def print_diagnostics(epoch, train_metrics, val_metrics, optimizer):
    """Print training diagnostics."""
    print(f"\nEpoch {epoch}:")
    print(f"  Train Loss: {train_metrics['loss']:.6f}")
    print(f"    Data: {train_metrics['data_loss']:.6f}")
    print(f"    Physics: {train_metrics.get('physics_loss', 0.0):.6f}")
    print(f"    IC: {train_metrics.get('ic_loss', 0.0):.6f}")
    print(f"  Val Loss: {val_metrics['loss']:.6f}")
    print(f"    Data: {val_metrics['data_loss']:.6f}")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")


def save_checkpoint(model, optimizer, scalers, epoch, val_loss, path):
    """Save model checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "scalers": scalers,
        },
        path,
    )


if __name__ == "__main__":
    main()
