"""
Training Script for Multi-Machine PINN with Pe(t) as Input.

This script trains a multi-machine PINN model using Pe_i(t) directly from ANDES
as input for each machine.

Usage:
    python training/train_multimachine_pe_input.py --data-dir data --output-dir outputs/models
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
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
from pinn.multimachine import MultimachinePINN
from utils.normalization import set_model_standardization_to_identity


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Machine PINN with Pe(t) Input")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/models", help="Directory to save trained model"
    )
    parser.add_argument("--num-machines", type=int, required=True, help="Number of machines")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cuda/cpu/auto). Use cuda for much faster training.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (e.g. kundur_2area.yaml); overrides epochs, lr, batch_size, loss from training/loss sections",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest_checkpoint.pth or best_model.pth in output-dir if present",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap training set size (rows) for faster runs. Example: 100000. Omit for full data.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Cap validation set size (rows) for faster runs. Example: 20000. Omit for full data.",
    )

    args = parser.parse_args()

    # Optional: load config to align with SMIB / publication settings
    config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            t_cfg = config.get("training", {})
            l_cfg = config.get("loss", {})
            m_cfg = config.get("model", {})
            if t_cfg:
                if "epochs" in t_cfg:
                    args.epochs = int(t_cfg["epochs"])
                if "learning_rate" in t_cfg:
                    args.lr = float(t_cfg["learning_rate"])
                if "batch_size" in t_cfg and t_cfg["batch_size"] is not None:
                    args.batch_size = int(t_cfg["batch_size"])
            if l_cfg and "scale_to_norm" in l_cfg:
                args._scale_to_norm = l_cfg["scale_to_norm"]
            if m_cfg and "hidden_dims" in m_cfg:
                args._hidden_dims = m_cfg["hidden_dims"]
            print(f"[OK] Loaded config from {config_path}")
        else:
            print(f"Warning: --config {args.config} not found, using CLI defaults")

    # Setup
    device = setup_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional: load checkpoint for resume (need scalers before data load)
    resume_ckpt = None
    if args.resume:
        resume_ckpt = load_checkpoint_for_resume(output_dir, device)
        if resume_ckpt is not None:
            print(f"[OK] Resume checkpoint found (epoch {resume_ckpt.get('epoch', -1) + 1} next)")
        else:
            print("No checkpoint found in output-dir; starting from scratch.")

    print("=" * 70)
    print("Multi-Machine PINN Training (Pe(t) Input Approach)")
    print("=" * 70)
    print(f"Number of machines: {args.num_machines}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Load and prepare data (use checkpoint scalers when resuming)
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    resume_scalers = resume_ckpt["scalers"] if resume_ckpt and "scalers" in resume_ckpt else None
    train_loader, val_loader, scalers = load_multimachine_data_with_pe(
        args.data_dir,
        args.num_machines,
        device=device,
        batch_size=args.batch_size,
        resume_scalers=resume_scalers,
        max_train_samples=getattr(args, "max_train_samples", None),
        max_val_samples=getattr(args, "max_val_samples", None),
    )

    print(f"[OK] Training batches: {len(train_loader)}")
    print(f"[OK] Validation batches: {len(val_loader)}")

    # Initialize model
    print("\n" + "=" * 70)
    print("PHASE 1: MODEL INITIALIZATION")
    print("=" * 70)

    hidden_dims = getattr(args, "_hidden_dims", None) or [128, 128, 128, 64]
    model = MultimachinePINN(
        num_machines=args.num_machines,
        input_dim_per_machine=9,  # [t, δ₀, ω₀, H, D, Pm_i, Pe(t), tf, tc]
        hidden_dims=hidden_dims,
        activation="tanh",
        dropout=0.0,
        use_coi=True,
        use_pe_as_input=True,  # Use Pe(t) as input
    ).to(device)

    print(f"[OK] Model initialized (input_dim_per_machine=9: t, d0, w0, H, D, Pm_i, Pe, tf, tc)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize loss
    print("\n" + "=" * 70)
    print("PHASE 2: LOSS FUNCTION SETUP")
    print("=" * 70)

    scale_to_norm = getattr(args, "_scale_to_norm", None)
    if scale_to_norm is None:
        scale_to_norm = [[20.0, 40.0]]
    scale_tensor = torch.tensor(scale_to_norm, dtype=torch.float32)
    if scale_tensor.dim() == 1:
        scale_tensor = scale_tensor.unsqueeze(0)
    loss_fn = PhysicsInformedLoss(
        lambda_data=1.0,
        lambda_physics=0.0,  # Controlled by adaptive scheduler; not yet used in loop
        lambda_ic=10.0,
        fn=60.0,
        use_normalized_loss=True,
        scale_to_norm=scale_tensor,
        use_pe_direct=True,  # Use Pe(t) directly from ANDES
    )

    print("[OK] Physics-informed loss initialized")
    print("  Pe direct mode: Enabled")

    # Setup adaptive scheduler
    adaptive_scheduler = AdaptiveLossWeightScheduler(
        initial_ratio=0.0,
        final_ratio=0.5,
        warmup_epochs=30,
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    print(f"[OK] Optimizer: Adam (lr={args.lr})")
    print(f"[OK] Batch size: {args.batch_size}")

    # Resume state from checkpoint if present
    start_epoch = 0
    best_val_loss = float("inf")
    if resume_ckpt is not None:
        try:
            model.load_state_dict(resume_ckpt["model_state_dict"], strict=True)
            if "optimizer_state_dict" in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in resume_ckpt and hasattr(scheduler_lr, "load_state_dict"):
                scheduler_lr.load_state_dict(resume_ckpt["scheduler_state_dict"])
            start_epoch = resume_ckpt.get("epoch", -1) + 1
            best_val_loss = resume_ckpt.get("val_loss", float("inf"))
            print(
                f"[OK] Resuming from epoch {start_epoch} (best_val_loss so far: {best_val_loss:.6f})"
            )
        except Exception as e:
            print(f"Warning: could not restore training state: {e}; starting from epoch 0")

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Training step
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            adaptive_scheduler,
            epoch,
            device,
        )

        # Validation step
        val_loss, val_metrics = validate_epoch(model, val_loader, loss_fn, device)

        # Learning rate scheduling
        scheduler_lr.step(val_loss)

        # Diagnostics
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print_diagnostics(epoch, train_metrics, val_metrics, optimizer)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler_lr,
                scalers,
                epoch,
                val_loss,
                output_dir / "best_model.pth",
            )
        # Periodic latest checkpoint for resume (every 5 epochs)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                model,
                optimizer,
                scheduler_lr,
                scalers,
                epoch,
                best_val_loss,
                output_dir / "latest_checkpoint.pth",
            )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")


def load_checkpoint_for_resume(output_dir: Path, device: torch.device) -> Optional[Dict[str, Any]]:
    """Load checkpoint for resume. Prefer latest_checkpoint.pth, else best_model.pth. Returns None if not found."""
    output_dir = Path(output_dir)
    for name in ("latest_checkpoint.pth", "best_model.pth"):
        path = output_dir / name
        if path.exists():
            try:
                ckpt = torch.load(path, map_location=device, weights_only=False)
                return ckpt
            except Exception as e:
                print(f"Warning: could not load {path}: {e}")
    return None


def setup_device(device_str: str) -> torch.device:
    """Setup device (cuda/cpu/auto)."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    return device


def load_multimachine_data_with_pe(
    data_path: Path,
    num_machines: int,
    device: str = "cpu",
    batch_size: int = 32,
    resume_scalers: Optional[Dict[str, Any]] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Load multi-machine data with Pe_i(t) for each machine.

    Returns:
    --------
    train_loader, val_loader, scalers
    """
    data_path = Path(data_path)
    if data_path.is_dir():
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        data_file = csv_files[0]
    else:
        data_file = data_path

    # Load data
    data = pd.read_csv(data_file)

    # Verify Pe columns exist for each machine
    for i in range(num_machines):
        pe_col = f"Pe_{i}" if f"Pe_{i}" in data.columns else "Pe"
        if pe_col not in data.columns and i == 0:
            raise ValueError(
                f"Pe column not found for machine {i}. Use use_pe_as_input=True during data generation."
            )

    # Group by scenario for proper train/val split
    scenario_ids = data["scenario_id"].unique()
    np.random.seed(42)
    np.random.shuffle(scenario_ids)
    split_idx = int(0.8 * len(scenario_ids))
    train_scenarios = scenario_ids[:split_idx]
    val_scenarios = scenario_ids[split_idx:]

    train_data = data[data["scenario_id"].isin(train_scenarios)]
    val_data = data[data["scenario_id"].isin(val_scenarios)]

    if max_train_samples is not None and len(train_data) > max_train_samples:
        train_data = train_data.sample(n=max_train_samples, random_state=42)
        print(f"  Capped train rows to {len(train_data)} (--max-train-samples)")
    if max_val_samples is not None and len(val_data) > max_val_samples:
        val_data = val_data.sample(n=max_val_samples, random_state=42)
        print(f"  Capped val rows to {len(val_data)} (--max-val-samples)")

    if resume_scalers is not None:
        scalers = resume_scalers
    else:
        # REVERT: Use separate scalers for delta0/omega0 vs delta/omega (like SMIB training code)
        # December experiments achieved R² Delta = 0.881 using separate scalers.
        # Separate scalers create beneficial structure: tight initial condition space → wide trajectory space
        # This makes learning easier than using same scalers (wide → wide).

        # Create scalers
        scalers = {}

        # Fit delta0/omega0 scalers on initial conditions only (scenario_data)
        scenario_data = train_data.groupby("scenario_id").first().reset_index()

        # Initial conditions (fitted on scenario_data - one value per scenario)
        if "delta0" in scenario_data.columns:
            scalers["delta0"] = StandardScaler().fit(scenario_data["delta0"].values.reshape(-1, 1))
        if "omega0" in scenario_data.columns:
            scalers["omega0"] = StandardScaler().fit(scenario_data["omega0"].values.reshape(-1, 1))

        # Fit delta/omega scalers on full trajectories (sample_data for efficiency)
        sample_data = train_data.sample(min(10000, len(train_data)))

        # For multimachine, we need to handle per-machine delta/omega
        # Check if data has per-machine columns (delta_0, delta_1, etc.) or single columns
        has_per_machine_cols = any(f"delta_{i}" in train_data.columns for i in range(num_machines))

        if has_per_machine_cols:
            # Per-machine delta/omega columns (delta_0, delta_1, omega_0, omega_1, etc.)
            for i in range(num_machines):
                delta_col = f"delta_{i}"
                omega_col = f"omega_{i}"
                if delta_col in sample_data.columns:
                    scalers[f"delta_{i}"] = StandardScaler().fit(
                        sample_data[delta_col].values.reshape(-1, 1)
                    )
                if omega_col in sample_data.columns:
                    scalers[f"omega_{i}"] = StandardScaler().fit(
                        sample_data[omega_col].values.reshape(-1, 1)
                    )
        else:
            # Single delta/omega columns (assume same for all machines or COI-referenced)
            if "delta" in sample_data.columns:
                scalers["delta"] = StandardScaler().fit(sample_data["delta"].values.reshape(-1, 1))
            if "omega" in sample_data.columns:
                scalers["omega"] = StandardScaler().fit(sample_data["omega"].values.reshape(-1, 1))

        # Time scaler (fitted on full trajectories)
        if "time" in sample_data.columns:
            scalers["time"] = StandardScaler().fit(sample_data["time"].values.reshape(-1, 1))

        # Parameter scalers (fitted on scenario_data - one value per scenario)
        param_cols = ["H", "D", "Pm", "param_H", "param_D", "param_Pm"]
        for col in param_cols:
            if col in scenario_data.columns:
                scaler = StandardScaler()
                scaler.fit(scenario_data[col].values.reshape(-1, 1))
                scalers[col] = scaler
                # Also create alias without param_ prefix for convenience
                if col.startswith("param_"):
                    base_col = col.replace("param_", "")
                    if base_col not in scalers:
                        scalers[base_col] = scaler

        # Per-machine Pm scalers when Pm_0..Pm_{n-1} exist (for 9-dim input with individual Pm)
        has_per_machine_pm = any(f"Pm_{i}" in scenario_data.columns for i in range(num_machines))
        if has_per_machine_pm:
            for i in range(num_machines):
                pm_col = f"Pm_{i}"
                if pm_col in scenario_data.columns:
                    scaler = StandardScaler()
                    scaler.fit(scenario_data[pm_col].values.reshape(-1, 1))
                    scalers[pm_col] = scaler

        # Fault times tf, tc (scenario-level; for 9-dim input steady-state context)
        tc_col = "param_tc" if "param_tc" in scenario_data.columns else "tc"
        if "tf" in scenario_data.columns:
            scalers["tf"] = StandardScaler().fit(scenario_data["tf"].values.reshape(-1, 1))
        else:
            scalers["tf"] = StandardScaler().fit(np.array([[1.0]]))
        if tc_col in scenario_data.columns:
            scalers["tc"] = StandardScaler().fit(scenario_data[tc_col].values.reshape(-1, 1))
        else:
            scalers["tc"] = StandardScaler().fit(np.array([[1.2]]))

        # Normalize Pe for each machine (fitted on full trajectories)
        for i in range(num_machines):
            pe_col = f"Pe_{i}" if f"Pe_{i}" in train_data.columns else "Pe"
            if pe_col in sample_data.columns:
                scaler = StandardScaler()
                Pe_all = sample_data[pe_col].values
                scaler.fit(Pe_all.reshape(-1, 1))
                scalers[f"Pe_{i}"] = scaler
            elif "Pe" in sample_data.columns and i == 0:
                # Fallback: use single Pe column for all machines
                scaler = StandardScaler()
                Pe_all = sample_data["Pe"].values
                scaler.fit(Pe_all.reshape(-1, 1))
                scalers["Pe"] = scaler
                # Use same scaler for all machines
                for j in range(num_machines):
                    scalers[f"Pe_{j}"] = scaler

    # Prepare tensors for multimachine data
    # Structure: [batch_size, num_machines, 9] for inputs: [t, δ₀, ω₀, H, D, Pm_i, Pe_i, tf, tc]
    # Structure: [batch_size, num_machines, 2] for outputs (delta, omega)
    INPUT_DIM = 9

    from utils.normalization import normalize_value, normalize_array

    def prepare_multimachine_batch(data_subset, scalers, num_machines, device):
        """Prepare a batch of multimachine data (9 dims per machine)."""
        scenarios = data_subset["scenario_id"].unique()
        batch_inputs = []
        batch_outputs = []
        tc_col = "param_tc" if "param_tc" in data_subset.columns else "tc"

        for scenario_id in scenarios:
            scenario_data = data_subset[data_subset["scenario_id"] == scenario_id].sort_values(
                "time"
            )
            if len(scenario_data) == 0:
                continue

            # Get initial conditions and scenario-level tf, tc (from first row)
            row0 = scenario_data.iloc[0]
            tf_raw = float(row0.get("tf", 1.0))
            tc_raw = float(row0.get(tc_col, row0.get("tc", 1.2)))
            tf_norm = normalize_value(tf_raw, scalers["tf"]) if "tf" in scalers else 0.0
            tc_norm = normalize_value(tc_raw, scalers["tc"]) if "tc" in scalers else 0.0

            # Check data format: per-machine columns (delta_0, delta_1) or single columns (delta)
            has_per_machine = any(
                f"delta_{i}" in scenario_data.columns for i in range(num_machines)
            )

            # Prepare inputs for each time point
            for _, row in scenario_data.iterrows():
                machine_inputs = []
                machine_outputs = []

                t_raw = float(row["time"])
                t_norm = normalize_value(t_raw, scalers["time"])

                for i in range(num_machines):
                    # Get initial conditions (scenario-level, same for all time points)
                    if has_per_machine:
                        delta0_raw = float(row0.get(f"delta0_{i}", row0.get("delta0", 0.0)))
                        omega0_raw = float(row0.get(f"omega0_{i}", row0.get("omega0", 1.0)))
                        H_raw = float(row0.get(f"H_{i}", row0.get("param_H", row0.get("H", 3.0))))
                        D_raw = float(row0.get(f"D_{i}", row0.get("param_D", row0.get("D", 1.0))))
                        Pm_raw = float(
                            row0.get(f"Pm_{i}", row0.get("param_Pm", row0.get("Pm", 0.7)))
                        )
                        Pe_raw = float(row.get(f"Pe_{i}", row.get("Pe", 0.0)))
                        delta_raw = float(row.get(f"delta_{i}", row.get("delta", 0.0)))
                        omega_raw = float(row.get(f"omega_{i}", row.get("omega", 1.0)))
                    else:
                        # Single columns (assume same for all machines or COI-referenced)
                        delta0_raw = float(row0.get("delta0", 0.0))
                        omega0_raw = float(row0.get("omega0", 1.0))
                        H_raw = float(row0.get("param_H", row0.get("H", 3.0)))
                        D_raw = float(row0.get("param_D", row0.get("D", 1.0)))
                        Pm_raw = float(row0.get("param_Pm", row0.get("Pm", 0.7)))
                        Pe_raw = float(row.get(f"Pe_{i}", row.get("Pe", 0.0)))
                        delta_raw = float(row.get("delta", 0.0))
                        omega_raw = float(row.get("omega", 1.0))

                    # Normalize inputs using delta0/omega0 scalers (for initial conditions)
                    delta0_norm = normalize_value(delta0_raw, scalers["delta0"])
                    omega0_norm = normalize_value(omega0_raw, scalers["omega0"])
                    H_norm = normalize_value(H_raw, scalers.get("H", scalers.get("param_H")))
                    D_norm = normalize_value(D_raw, scalers.get("D", scalers.get("param_D")))
                    Pm_scaler = scalers.get(f"Pm_{i}", scalers.get("Pm", scalers.get("param_Pm")))
                    Pm_norm = normalize_value(Pm_raw, Pm_scaler) if Pm_scaler is not None else 0.0

                    # Normalize Pe using per-machine scaler or fallback
                    Pe_scaler = scalers.get(f"Pe_{i}", scalers.get("Pe"))
                    if Pe_scaler is not None:
                        Pe_norm = normalize_value(Pe_raw, Pe_scaler)
                    else:
                        Pe_norm = 0.0

                    # Input: [t, δ₀, ω₀, H, D, Pm_i, Pe(t), tf, tc] — 9 dims (same as SMIB)
                    machine_input = [
                        t_norm,
                        delta0_norm,
                        omega0_norm,
                        H_norm,
                        D_norm,
                        Pm_norm,
                        Pe_norm,
                        tf_norm,
                        tc_norm,
                    ]
                    machine_inputs.append(machine_input)

                    # Normalize outputs using delta/omega scalers (for full trajectories)
                    if has_per_machine:
                        delta_scaler = scalers.get(f"delta_{i}", scalers.get("delta"))
                        omega_scaler = scalers.get(f"omega_{i}", scalers.get("omega"))
                    else:
                        delta_scaler = scalers.get("delta")
                        omega_scaler = scalers.get("omega")

                    if delta_scaler is not None:
                        delta_norm = normalize_value(delta_raw, delta_scaler)
                    else:
                        delta_norm = 0.0

                    if omega_scaler is not None:
                        omega_norm = normalize_value(omega_raw, omega_scaler)
                    else:
                        omega_norm = 0.0

                    # Output: [δ(t), ω(t)]
                    machine_output = [delta_norm, omega_norm]
                    machine_outputs.append(machine_output)

                # Stack: [num_machines, input_dim] and [num_machines, 2]
                batch_inputs.append(machine_inputs)
                batch_outputs.append(machine_outputs)

        if len(batch_inputs) == 0:
            # Return empty tensors if no data (9 dims per machine)
            return torch.empty(0, num_machines, INPUT_DIM, device=device), torch.empty(
                0, num_machines, 2, device=device
            )

        # Convert to tensors: [batch_size, num_machines, input_dim] and [batch_size, num_machines, 2]
        X = torch.tensor(batch_inputs, dtype=torch.float32, device=device)
        y = torch.tensor(batch_outputs, dtype=torch.float32, device=device)

        return X, y

    # Prepare training and validation data
    X_train, y_train = prepare_multimachine_batch(train_data, scalers, num_machines, device)
    X_val, y_val = prepare_multimachine_batch(val_data, scalers, num_machines, device)

    if len(X_train) == 0:
        raise ValueError("No training data prepared. Check data format and columns.")

    print(f"[OK] Prepared {len(X_train)} training samples")
    print(f"[OK] Prepared {len(X_val)} validation samples")
    print(f"  Input shape: {X_train.shape} (batch, machines, input_dim)")
    print(f"  Output shape: {y_train.shape} (batch, machines, 2)")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    use_cuda = str(device).startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=0,
    )

    return train_loader, val_loader, scalers


def train_epoch(
    model,
    train_loader,
    loss_fn,
    optimizer,
    adaptive_scheduler,
    epoch,
    device,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        delta_pred, omega_pred = model(X_batch)

        # Compute loss: data MSE only. PhysicsInformedLoss (physics + IC + steady-state for t < tf)
        # is not yet wired here—see docs/guides/MULTIMACHINE_GAPS_AND_FINDINGS.md.
        loss = torch.nn.functional.mse_loss(
            delta_pred, y_batch[:, :, 0]
        ) + torch.nn.functional.mse_loss(omega_pred, y_batch[:, :, 1])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    metrics = {"loss": total_loss / n_batches}

    return metrics["loss"], metrics


def validate_epoch(model, val_loader, loss_fn, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            delta_pred, omega_pred = model(X_batch)

            # Compute loss (data MSE only; physics/IC/steady-state not used—see MULTIMACHINE_GAPS_AND_FINDINGS.md)
            loss = torch.nn.functional.mse_loss(
                delta_pred, y_batch[:, :, 0]
            ) + torch.nn.functional.mse_loss(omega_pred, y_batch[:, :, 1])

            total_loss += loss.item()
            n_batches += 1

    metrics = {"loss": total_loss / n_batches}

    return metrics["loss"], metrics


def print_diagnostics(epoch, train_metrics, val_metrics, optimizer):
    """Print training diagnostics."""
    print(f"\nEpoch {epoch}:")
    print(f"  Train Loss: {train_metrics['loss']:.6f}")
    print(f"  Val Loss: {val_metrics['loss']:.6f}")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")


def save_checkpoint(model, optimizer, scheduler, scalers, epoch, val_loss, path):
    """Save model checkpoint (for best model and for resume)."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "scalers": scalers,
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


if __name__ == "__main__":
    main()
