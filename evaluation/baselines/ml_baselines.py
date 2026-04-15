"""
Standard ML Baselines (No Physics Constraints).

Feedforward (MLP) baseline without physics constraints, for comparison with PINN.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import compute_trajectory_metrics


class StandardNN(nn.Module):
    """
    Standard feedforward neural network without physics constraints.

    Same architecture as PINN but trained only on data loss (no physics loss).
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dims: list = [256, 256, 128, 128],
        output_dim: int = 2,
        activation: str = "tanh",
        dropout: float = 0.0,
    ):
        """
        Initialize standard NN.

        Parameters:
        -----------
        input_dim : int
            Input dimension
        hidden_dims : list
            Hidden layer dimensions
        output_dim : int
            Output dimension (2 for delta, omega)
        activation : str
            Activation function
        dropout : float
            Dropout rate
        """
        super(StandardNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class MLBaselineTrainer:
    """
    Trainer for ML baseline models.
    """

    def __init__(
        self,
        model_type: str = "standard_nn",
        model_config: Optional[Dict] = None,
        device: str = "auto",
    ):
        """
        Initialize ML baseline trainer.

        Parameters:
        -----------
        model_type : str
            Model type: ``"standard_nn"`` (feedforward baseline)
        model_config : dict, optional
            Model configuration
        device : str
            Device to use
        """
        self.model_type = model_type
        self.model_config = model_config or {}
        if self.model_type != "standard_nn":
            raise ValueError(
                f"Unsupported ML baseline model_type {self.model_type!r}; only 'standard_nn' is supported."
            )
        # Handle device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.scalers = {}
        self.input_method = None  # Will be set in prepare_data()

    def prepare_data(
        self,
        data_path: Path,
        input_method: str = "reactance",
        test_size: float = 0.15,
        seed: int = 42,
        val_data_path: Optional[Path] = None,
        scale_to_norm: Optional[List[float]] = None,
        unstable_weight: float = 1.0,
        use_fixed_target_scale: bool = False,
    ) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        Prepare data for training.

        Parameters:
        -----------
        data_path : Path
            Path to training data CSV (or preprocessed train_data file)
        input_method : str
            Input method: "reactance" or "pe_direct"
        test_size : float
            Validation split size (only used if val_data_path not provided)
        seed : int
            Random seed (only used if val_data_path not provided)
        val_data_path : Path, optional
            Path to preprocessed validation data CSV (if provided, uses this instead of splitting)
        scale_to_norm : List[float], optional
            [delta_scale, omega_scale] for fixed target scaling (match PINN). Used when use_fixed_target_scale=True.
        unstable_weight : float
            Loss weight for rows from unstable scenarios (1.0 = no upweight). Fair: balances gradient toward unstable cases.
        use_fixed_target_scale : bool
            If True, normalize targets (and delta0/omega0 inputs) by scale_to_norm instead of StandardScaler (match PINN).

        Returns:
        --------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        scalers : dict
            Fitted scalers
        """
        # Check if this is a preprocessed train file and val file exists
        if "train_data_" in str(data_path.name) and val_data_path is None:
            # Look for corresponding val file in same directory
            val_path = data_path.parent / data_path.name.replace("train_data_", "val_data_")
            if val_path.exists():
                val_data_path = val_path
                print(f"✓ Detected preprocessed data: using train/val files directly")

        # Use preprocessed splits if available
        test_data = None
        if val_data_path and val_data_path.exists():
            print("\nLoading data...")
            train_data = pd.read_csv(data_path)
            val_data = pd.read_csv(val_data_path)
            print(f"✓ Loaded training data: {len(train_data):,} rows")
            print(f"✓ Loaded validation data: {len(val_data):,} rows")

            # Try to load test data if it exists in the same directory (for complete verification)
            if "train_data_" in data_path.name:
                test_path = data_path.parent / data_path.name.replace("train_data_", "test_data_")
                if test_path.exists():
                    try:
                        test_data = pd.read_csv(test_path)
                        print(f"✓ Detected test data file: {test_path.name}")
                    except Exception:
                        pass

            split_method = "preprocessed files"
        else:
            # Load data and split by scenario
            print("\nLoading data...")
            data = pd.read_csv(data_path)
            print(f"✓ Loaded data: {len(data):,} rows")

            # Split by scenario
            scenarios = data["scenario_id"].unique()
            train_scenarios, val_scenarios = train_test_split(
                scenarios, test_size=test_size, random_state=seed
            )

            train_data = data[data["scenario_id"].isin(train_scenarios)]
            val_data = data[data["scenario_id"].isin(val_scenarios)]

            print(f"✓ Training scenarios: {len(train_scenarios)}")
            print(f"✓ Validation scenarios: {len(val_scenarios)}")
            split_method = "on-the-fly split"

        # Determine input columns based on method
        if input_method == "pe_direct":
            # 9 inputs: [t, delta0, omega0, H, D, Pm, Pe, tf, tc] for steady-state context
            input_cols = ["time", "delta0", "omega0", "H", "D", "Pm", "Pe", "tf", "tc"]
            input_dim = 9
        elif input_method == "pe_direct_7":
            # 7 inputs: [t, delta0, omega0, H, D, Pm, Pe] (legacy setup that gave best ML R² in exp_20260113_103716)
            input_cols = ["time", "delta0", "omega0", "H", "D", "Pm", "Pe"]
            input_dim = 7
        else:
            input_cols = [
                "time",
                "delta0",
                "omega0",
                "H",
                "D",
                "Pm",
                "Xprefault",
                "Xfault",
                "Xpostfault",
                "tf",
                "tc",
            ]
            input_dim = 11

        print(f"✓ Input method: {input_method} ({input_dim} dimensions)")
        print(f"✓ Input columns: {', '.join(input_cols)}")

        # Store input_method for later use (e.g., in training summary)
        self.input_method = input_method

        # CRITICAL: Ensure canonical column names so we always have 9 (pe_direct) or 11 (reactance) inputs.
        # Some datasets use param_H/param_D/param_Pm/param_tc only; ML baseline expects H, D, Pm, tc.
        for df in [train_data, val_data]:
            if "H" not in df.columns and "param_H" in df.columns:
                df["H"] = df["param_H"]
            if "D" not in df.columns and "param_D" in df.columns:
                df["D"] = df["param_D"]
            if "Pm" not in df.columns and "param_Pm" in df.columns:
                df["Pm"] = df["param_Pm"]
            if "tc" not in df.columns and "param_tc" in df.columns:
                df["tc"] = df["param_tc"]

        missing = [c for c in input_cols if c not in train_data.columns]
        if missing:
            raise ValueError(
                f"ML baseline requires input columns {input_cols}. Missing in data: {missing}. "
                "Ensure data has H/D/Pm/tc (or param_H/param_D/param_Pm/param_tc)."
            )

        # Display comprehensive data split verification
        print("\n" + "=" * 70)
        print("DATA SPLIT VERIFICATION")
        print("=" * 70)

        train_scenarios = train_data["scenario_id"].unique()
        val_scenarios = val_data["scenario_id"].unique()
        train_scenario_set = set(train_scenarios)
        val_scenario_set = set(val_scenarios)

        # Include test scenarios if available
        test_scenario_set = set()
        if test_data is not None:
            test_scenarios = test_data["scenario_id"].unique()
            test_scenario_set = set(test_scenarios)

        # Check for overlaps
        train_val_overlap = train_scenario_set & val_scenario_set
        train_test_overlap = (
            train_scenario_set & test_scenario_set if test_data is not None else set()
        )
        val_test_overlap = val_scenario_set & test_scenario_set if test_data is not None else set()
        total_unique = len(train_scenario_set | val_scenario_set | test_scenario_set)

        print(f"Split Method: {split_method}")
        print(f"\nTraining Set:")
        print(f"  Scenarios: {len(train_scenarios)}")
        print(f"  Rows: {len(train_data):,}")
        if len(train_scenarios) > 0:
            print(f"  Scenario ID range: {min(train_scenarios)} - {max(train_scenarios)}")
        else:
            print(f"  Scenario ID range: N/A (empty)")

        print(f"\nValidation Set:")
        print(f"  Scenarios: {len(val_scenarios)}")
        print(f"  Rows: {len(val_data):,}")
        if len(val_scenarios) > 0:
            print(f"  Scenario ID range: {min(val_scenarios)} - {max(val_scenarios)}")
        else:
            print(f"  Scenario ID range: N/A (empty)")

        if test_data is not None:
            print(f"\nTest Set:")
            print(f"  Scenarios: {len(test_scenarios)}")
            print(f"  Rows: {len(test_data):,}")
            if len(test_scenarios) > 0:
                print(f"  Scenario ID range: {min(test_scenarios)} - {max(test_scenarios)}")
            else:
                print(f"  Scenario ID range: N/A (empty)")

        print(f"\nSplit Verification:")
        print(f"  Total unique scenarios: {total_unique}")

        # Check all overlaps
        all_overlaps = train_val_overlap | train_test_overlap | val_test_overlap
        if len(all_overlaps) == 0:
            print(f"  Overlapping scenarios: 0 ✓ (No overlap - correct)")
        else:
            print(f"  ⚠️  WARNING: Overlapping scenarios detected!")
            if train_val_overlap:
                print(f"     Train/Val overlap: {len(train_val_overlap)} scenarios")
            if train_test_overlap:
                print(f"     Train/Test overlap: {len(train_test_overlap)} scenarios")
            if val_test_overlap:
                print(f"     Val/Test overlap: {len(val_test_overlap)} scenarios")

        # Calculate split ratios (including test if available)
        if total_unique > 0:
            train_ratio = len(train_scenarios) / total_unique
            val_ratio = len(val_scenarios) / total_unique
            if test_data is not None:
                test_ratio = len(test_scenarios) / total_unique
                print(
                    f"  Split ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}"
                )
            else:
                print(
                    f"  Split ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%} (Test set not found)"
                )
                if total_unique < 146:  # Expected total for this dataset
                    print(
                        f"  ⚠️  Note: Expected ~146 total scenarios, found {total_unique}. Missing scenarios likely in test set."
                    )

        print("=" * 70)

        # Fixed target scale (match PINN) or StandardScaler for targets
        use_fixed = use_fixed_target_scale and scale_to_norm is not None and len(scale_to_norm) >= 2
        if use_fixed:
            scale_delta, scale_omega = float(scale_to_norm[0]), float(scale_to_norm[1])
            print(
                f"\n✓ Using fixed target scale (match PINN): delta/{scale_delta}, omega/{scale_omega}"
            )

        # Fit scalers on training data
        print("\nFitting scalers...")
        scalers = {}
        for col in input_cols:
            if col in train_data.columns:
                if use_fixed and col in ("delta0", "omega0"):
                    # delta0/omega0 use same fixed scale as targets (no StandardScaler)
                    continue
                scaler = StandardScaler()
                scaler.fit(train_data[col].values.reshape(-1, 1))
                scalers[col] = scaler

        if use_fixed:
            scalers["delta_fixed_scale"] = scale_delta
            scalers["omega_fixed_scale"] = scale_omega
        else:
            # CRITICAL: Also fit scalers for target variables (delta, omega)
            for target_col in ["delta", "omega"]:
                if target_col in train_data.columns:
                    scaler = StandardScaler()
                    scaler.fit(train_data[target_col].values.reshape(-1, 1))
                    scalers[target_col] = scaler
                    print(
                        f"  ✓ Fitted scaler for target: {target_col} (mean={scaler.mean_[0]:.6f}, std={scaler.scale_[0]:.6f})"
                    )

        print(f"✓ Fitted {len(scalers)} scalers (including target scalers)")

        def prepare_tensors(df, scalers_dict, use_fixed_scale=False, scale_to_norm_vals=None):
            inputs = []
            for col in input_cols:
                if col not in df.columns:
                    continue
                values = df[col].values.reshape(-1, 1)
                if use_fixed_scale and scale_to_norm_vals is not None and col == "delta0":
                    normalized = values / scale_to_norm_vals[0]
                elif use_fixed_scale and scale_to_norm_vals is not None and col == "omega0":
                    normalized = values / scale_to_norm_vals[1]
                else:
                    normalized = scalers_dict[col].transform(values)
                inputs.append(torch.tensor(normalized, dtype=torch.float32))

            X = torch.stack(inputs, dim=1).squeeze(-1)

            delta_raw = df["delta"].values.reshape(-1, 1)
            omega_raw = df["omega"].values.reshape(-1, 1)
            if use_fixed_scale and scale_to_norm_vals is not None:
                delta_normalized = delta_raw / scale_to_norm_vals[0]
                omega_normalized = omega_raw / scale_to_norm_vals[1]
            else:
                delta_normalized = scalers_dict["delta"].transform(delta_raw)
                omega_normalized = scalers_dict["omega"].transform(omega_raw)

            delta = torch.tensor(delta_normalized, dtype=torch.float32).squeeze(-1)
            omega = torch.tensor(omega_normalized, dtype=torch.float32).squeeze(-1)
            y = torch.stack([delta, omega], dim=1)
            return X, y

        scale_vals = [scale_delta, scale_omega] if use_fixed else None
        X_train, y_train = prepare_tensors(
            train_data, scalers, use_fixed_scale=use_fixed, scale_to_norm_vals=scale_vals
        )
        X_val, y_val = prepare_tensors(
            val_data, scalers, use_fixed_scale=use_fixed, scale_to_norm_vals=scale_vals
        )

        # Ensure we got the expected input dimension (catches wrong column order or missing cols)
        if X_train.shape[1] != input_dim:
            raise ValueError(
                f"ML baseline input dimension mismatch: expected {input_dim} (input_cols order), "
                f"got {X_train.shape[1]}. Check that all of {input_cols} are present and used in order."
            )

        # Per-row sample weights for unstable-scenario upweighting (fair: no physics, just loss balance)
        sample_weights_train = None
        if unstable_weight != 1.0 and "is_stable" in train_data.columns:
            scenario_stable = train_data.groupby("scenario_id")["is_stable"].first()
            train_data["_row_unstable"] = train_data["scenario_id"].map(
                lambda s: not scenario_stable.get(s, True)
            )
            w = np.where(train_data["_row_unstable"].values, float(unstable_weight), 1.0)
            sample_weights_train = torch.tensor(w, dtype=torch.float32)
            n_unstable = train_data["_row_unstable"].sum()
            print(
                f"\n✓ Unstable-scenario upweighting: weight={unstable_weight} for {n_unstable} rows from unstable scenarios"
            )

        # Adaptive batch size calculation for fair comparison with PINN
        # CRITICAL: PINN batches by scenarios (~6-8 batches/epoch)
        # ML baseline batches by rows, so we need to match effective gradient updates
        # Strategy: Scale batch size based on scenarios to match PINN's update frequency
        num_train_samples = len(X_train)
        num_train_scenarios = len(train_scenarios)

        # PINN typically uses 6-8 batches per epoch (target: 7)
        # For fair comparison, we should target similar number of batches per epoch
        # Since ML baseline processes rows (not scenarios), we calculate:
        # - If PINN has N scenarios and uses batch_size=16, it gets ~N/16 batches
        # - For ML baseline with M samples, to get similar batches: batch_size ≈ M/(N/16)
        # - But we allow 1.5-2x more batches for row-based training efficiency

        if num_train_scenarios > 0:
            # Calculate what PINN's batches per epoch would be
            # (assuming PINN uses adaptive batch size targeting ~7 batches)
            pinn_target_batches = 7
            pinn_batch_size_estimate = max(8, num_train_scenarios // pinn_target_batches)
            pinn_batches_per_epoch = num_train_scenarios / pinn_batch_size_estimate

            # For ML baseline, target 1.5-2x PINN's batches (accounting for row-based nature)
            # This gives us more gradient updates but keeps it reasonable
            target_batches = max(10, min(20, int(pinn_batches_per_epoch * 1.5)))
        else:
            # Fallback if scenarios not available
            target_batches = 15

        calculated_batch_size = num_train_samples // target_batches

        # Apply minimum and maximum constraints
        if num_train_samples < 1000:
            batch_size = max(16, calculated_batch_size)
        elif num_train_samples < 10000:
            batch_size = max(32, calculated_batch_size)
        elif num_train_samples < 100000:
            batch_size = max(64, calculated_batch_size)
        else:
            batch_size = max(128, calculated_batch_size)

        # Cap at reasonable maximum (increased to allow better matching with PINN)
        # Note: Large batch sizes are acceptable for row-based training
        # We allow up to 20,000 to match PINN's gradient update frequency
        batch_size = min(20000, batch_size)

        batches_per_epoch = num_train_samples / batch_size

        # Calculate comparison info for fair comparison reporting
        if num_train_scenarios > 0:
            pinn_batches_estimate = num_train_scenarios / max(8, num_train_scenarios // 7)
            ratio = batches_per_epoch / pinn_batches_estimate if pinn_batches_estimate > 0 else 0
        else:
            pinn_batches_estimate = 0
            ratio = 0

        print(
            f"\n✓ Adaptive batch size: {batch_size} (calculated from {num_train_samples:,} samples)"
        )
        print(f"  → {batches_per_epoch:.1f} batches per epoch")
        if num_train_scenarios > 0:
            print(
                f"  → PINN equivalent: ~{pinn_batches_estimate:.1f} batches/epoch ({num_train_scenarios} scenarios)"
            )
            print(f"  → Ratio: {ratio:.1f}x (target: 1.5-2x for fair comparison)")
        print(f"  Note: ML baseline batches by rows (not scenarios like PINN)")

        # Create data loaders (train may include sample weights for unstable upweighting)
        if sample_weights_train is not None:
            train_dataset = TensorDataset(X_train, y_train, sample_weights_train)
        else:
            train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Store scalers in trainer for later use (e.g., in evaluate())
        self.scalers = scalers

        return train_loader, val_loader, scalers

    def build_model(self, input_dim: int) -> nn.Module:
        """
        Build the model.

        Parameters:
        -----------
        input_dim : int
            Input dimension

        Returns:
        --------
        model : nn.Module
            Built model
        """
        if self.model_type == "standard_nn":
            model = StandardNN(
                input_dim=input_dim,
                hidden_dims=self.model_config.get("hidden_dims", [256, 256, 128, 128]),
                output_dim=2,
                activation=self.model_config.get("activation", "tanh"),
                dropout=self.model_config.get("dropout", 0.0),
            )
        else:
            raise ValueError(
                f"Unknown model type: {self.model_type!r}. "
                "Only 'standard_nn' is supported for ML baselines in this release."
            )

        return model.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 400,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        early_stopping_patience: Optional[int] = None,
        lambda_ic: float = 10.0,
        scale_to_norm: Optional[List[float]] = None,
        pre_fault_weight: float = 1.0,
        lambda_steady_state: float = 0.0,
        use_ic_over_prefault: bool = False,
        two_phase_training: bool = False,
        phase1_epochs: int = 0,
        phase2_pre_fault_weight: Optional[float] = None,
    ) -> Dict:
        """
        Train the model.

        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        epochs : int
            Number of epochs
        learning_rate : float
            Learning rate
        weight_decay : float
            Weight decay
        early_stopping_patience : int, optional
            Early stopping patience (None to disable)
        lambda_ic : float
            Weight for initial condition loss (default: 10.0 to match PINN)
        scale_to_norm : List[float], optional
            Scaling factors for normalized loss [delta_weight, omega_weight]
            (default: [1.0, 100.0] to match PINN)
        pre_fault_weight : float
            Loss weight for rows with t < tf (1.0 = no weighting). Used for fair comparison.
        lambda_steady_state : float
            Weight for steady-state auxiliary loss (pred vs delta0, omega0) on pre-fault rows.
        use_ic_over_prefault : bool
            If True, apply IC loss to all pre-fault rows (t < tf); else only at t ≈ 0.
        two_phase_training : bool
            If True, phase 1 uses only pre-fault data loss (lock IC); then full loss. Fair: no physics.
        phase1_epochs : int
            Number of epochs for phase 1 when two_phase_training is True.
        phase2_pre_fault_weight : float, optional
            If set, use this weight for pre-fault rows in phase 2 instead of pre_fault_weight.
            Lower (e.g. 2.0) gives more gradient to post-fault/unstable. Fair: same data, rebalance only.

        Returns:
        --------
        history : dict
            Training history
        """
        # Build model
        print("\nInitializing model...")
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch[0].shape[-1]
        self.model = self.build_model(input_dim)
        print(f"✓ Model initialized")
        print(f"  Input dimension: {input_dim}")
        print(f"  Output dimension: 2 (delta, omega)")
        print(f"  Architecture: {self.model_type}")
        if hasattr(self.model, "hidden_dims"):
            print(f"  Hidden dimensions: {self.model.hidden_dims}")
        elif hasattr(self.model, "hidden_size"):
            print(f"  Hidden size: {self.model.hidden_size}, Layers: {self.model.num_layers}")

        # Setup optimizer and loss
        print("\nSetting up optimizer and loss function...")
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Learning rate scheduler (matching PINN for fair comparison)
        scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )

        # Use weighted loss matching PINN's NormalizedStateLoss approach
        # PINN uses scale_to_norm = [1.0, 100.0] to weight omega errors 100x more
        # This balances delta (~0.1-5.0 rad) and omega (~0.001-0.01 pu) contributions
        use_weighted_loss = True  # Match PINN's normalized state loss

        # Get scale_to_norm from parameter (default matches PINN)
        if scale_to_norm is None:
            scale_to_norm = [1.0, 100.0]
        omega_weight = scale_to_norm[1]  # Extract omega weight from scale_to_norm

        # Initial condition loss weight (for fair comparison with PINN)
        # Use same lambda_ic as PINN for fair comparison (default: 10.0)
        # Note: Using same value as PINN ensures fair comparison - physics loss compensation removed

        print(f"✓ Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"✓ Loss function: Weighted MSE (matching PINN's NormalizedStateLoss)")
        print(f"✓ Loss weights:")
        print(f"    Data weight: 1.0")
        print(f"    Delta weight: 1.0")
        print(f"    Omega weight: {omega_weight:.1f} (matching PINN)")
        print(f"    IC weight: {lambda_ic:.1f}")
        print(f"    Physics weight: None (ML baseline doesn't use physics loss)")
        if pre_fault_weight != 1.0 or lambda_steady_state > 0 or use_ic_over_prefault:
            print(
                f"    Fair-comparison options: pre_fault_weight={pre_fault_weight}, lambda_steady_state={lambda_steady_state}, use_ic_over_prefault={use_ic_over_prefault}"
            )
        if two_phase_training and phase1_epochs > 0:
            print(
                f"    Two-phase training: phase1_epochs={phase1_epochs} (pre-fault data loss only), then full loss"
            )
        if phase2_pre_fault_weight is not None and two_phase_training and phase1_epochs > 0:
            print(
                f"    Phase 2 pre_fault_weight: {phase2_pre_fault_weight} (rebalance toward post-fault)"
            )

        # Training history
        history = {
            "train_losses": [],
            "val_losses": [],
            "epochs": [],
        }

        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = 0
        # Early stopping patience (matching PINN default: None = disabled, or configurable)
        # Use provided early_stopping_patience, or None to disable
        patience = early_stopping_patience  # Can be None (disabled) or int (number of epochs)
        patience_counter = 0

        # Time tracking
        training_start_time = time.time()
        epoch_times = []

        # Enhanced output similar to PINN
        print("\n" + "=" * 70)
        print(f"TRAINING {self.model_type.upper()} MODEL")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model type: {self.model_type}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Loss function: Weighted MSE (matching PINN)")
        print(
            f"Loss weights: lambda_data=1.0, delta=1.0, omega={omega_weight:.1f}, lambda_ic={lambda_ic:.1f} (matches PINN for fair comparison), lambda_physics=None"
        )
        print(f"Input method: {getattr(self, 'input_method', 'N/A')}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Training batches per epoch: {len(train_loader)}")
        print(f"Validation batches per epoch: {len(val_loader)}")
        print("=" * 70)
        sys.stdout.flush()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            phase1 = two_phase_training and epoch < phase1_epochs

            # Training
            self.model.train()
            train_loss = 0.0
            train_data_loss = 0.0
            train_ic_loss = 0.0
            for batch in train_loader:
                if len(batch) == 3:
                    batch_X, batch_y, batch_w = batch
                    batch_w = batch_w.to(self.device)
                else:
                    batch_X, batch_y = batch
                    batch_w = None

                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch_X)

                t_values = batch_X[:, 0]
                # tf column: index 7 for 9-dim (pe_direct), 9 for 11-dim (reactance)
                has_tf = batch_X.shape[1] >= 9
                tf_col = 7 if batch_X.shape[1] == 9 else (9 if batch_X.shape[1] >= 11 else None)
                if has_tf and tf_col is not None:
                    pre_fault_mask = t_values < batch_X[:, tf_col]
                else:
                    pre_fault_mask = torch.zeros(
                        batch_X.shape[0], dtype=torch.bool, device=batch_X.device
                    )

                # Pre-fault weight: in phase 2 can use lower value to rebalance toward post-fault (fair)
                effective_pf_weight = pre_fault_weight
                if not phase1 and phase2_pre_fault_weight is not None:
                    effective_pf_weight = phase2_pre_fault_weight

                # Sample weights: pre-fault emphasis and optional unstable-scenario upweight
                w = torch.where(
                    pre_fault_mask,
                    torch.tensor(effective_pf_weight, device=batch_X.device, dtype=batch_X.dtype),
                    torch.tensor(1.0, device=batch_X.device, dtype=batch_X.dtype),
                )
                if batch_w is not None:
                    w = w * batch_w

                # Data loss: Weighted MSE matching PINN's NormalizedStateLoss
                if use_weighted_loss:
                    delta_pred = pred[:, 0]
                    omega_pred = pred[:, 1]
                    delta_true = batch_y[:, 0]
                    omega_true = batch_y[:, 1]
                    delta_error = delta_pred - delta_true
                    omega_error = omega_pred - omega_true
                    if phase1:
                        # Phase 1: only pre-fault rows contribute to data loss (lock IC)
                        if pre_fault_mask.any():
                            n_pf = pre_fault_mask.sum().float().clamp(min=1)
                            data_loss = (
                                w[pre_fault_mask] * delta_error[pre_fault_mask] ** 2
                            ).sum() / n_pf + omega_weight * (
                                w[pre_fault_mask] * omega_error[pre_fault_mask] ** 2
                            ).sum() / n_pf
                        else:
                            data_loss = torch.tensor(0.0, device=self.device)
                    elif pre_fault_weight != 1.0 or batch_w is not None:
                        data_loss = (w * delta_error**2).mean() + omega_weight * (
                            w * omega_error**2
                        ).mean()
                    else:
                        data_loss = torch.mean(delta_error**2) + omega_weight * torch.mean(
                            omega_error**2
                        )
                else:
                    data_loss = nn.MSELoss()(pred, batch_y)

                # Steady-state auxiliary loss: pred = (delta0, omega0) for t < tf
                if lambda_steady_state > 0 and has_tf and pre_fault_mask.any():
                    delta0_batch = batch_X[pre_fault_mask, 1]
                    omega0_batch = batch_X[pre_fault_mask, 2]
                    ss_delta = ((pred[pre_fault_mask, 0] - delta0_batch) ** 2).mean()
                    ss_omega = ((pred[pre_fault_mask, 1] - omega0_batch) ** 2).mean()
                    ss_loss = ss_delta + omega_weight * ss_omega
                else:
                    ss_loss = torch.tensor(0.0, device=self.device)

                # Initial condition loss: either over all pre-fault rows or only at t ≈ 0
                if use_ic_over_prefault and has_tf and pre_fault_mask.any():
                    delta0_batch = batch_X[pre_fault_mask, 1]
                    omega0_batch = batch_X[pre_fault_mask, 2]
                    ic_loss_delta = ((pred[pre_fault_mask, 0] - delta0_batch) ** 2).mean()
                    ic_loss_omega = ((pred[pre_fault_mask, 1] - omega0_batch) ** 2).mean()
                    ic_loss = ic_loss_delta + omega_weight * ic_loss_omega
                else:
                    min_t = t_values.min()
                    tolerance = 1e-6
                    ic_mask = (t_values - min_t).abs() < tolerance
                    if ic_mask.sum() > 0:
                        delta0_batch = batch_X[ic_mask, 1]
                        omega0_batch = batch_X[ic_mask, 2]
                        ic_loss_delta = torch.mean((pred[ic_mask, 0] - delta0_batch) ** 2)
                        ic_loss_omega = torch.mean((pred[ic_mask, 1] - omega0_batch) ** 2)
                        ic_loss = ic_loss_delta + omega_weight * ic_loss_omega
                    else:
                        ic_loss = torch.tensor(0.0, device=self.device)

                # Total loss: data + steady-state + IC
                loss = data_loss + lambda_steady_state * ss_loss + lambda_ic * ic_loss

                loss.backward()

                # Gradient clipping (matching PINN for fair comparison)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                train_data_loss += data_loss.item()
                train_ic_loss += ic_loss.item()

            train_loss /= len(train_loader)
            train_data_loss /= len(train_loader)
            train_ic_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_data_loss = 0.0
            val_ic_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    pred = self.model(batch_X)

                    t_values = batch_X[:, 0]
                    has_tf = batch_X.shape[1] >= 9
                    tf_col = 7 if batch_X.shape[1] == 9 else (9 if batch_X.shape[1] >= 11 else None)
                    if has_tf and tf_col is not None:
                        pre_fault_mask = t_values < batch_X[:, tf_col]
                    else:
                        pre_fault_mask = torch.zeros(
                            batch_X.shape[0], dtype=torch.bool, device=batch_X.device
                        )

                    effective_pf_weight_val = (
                        pre_fault_weight
                        if (phase1 or phase2_pre_fault_weight is None)
                        else phase2_pre_fault_weight
                    )
                    w = torch.where(
                        pre_fault_mask,
                        torch.tensor(
                            effective_pf_weight_val, device=batch_X.device, dtype=batch_X.dtype
                        ),
                        torch.tensor(1.0, device=batch_X.device, dtype=batch_X.dtype),
                    )

                    if use_weighted_loss:
                        delta_pred = pred[:, 0]
                        omega_pred = pred[:, 1]
                        delta_true = batch_y[:, 0]
                        omega_true = batch_y[:, 1]
                        delta_error = delta_pred - delta_true
                        omega_error = omega_pred - omega_true
                        if (
                            effective_pf_weight_val != 1.0 or pre_fault_weight != 1.0
                        ) and pre_fault_mask.any():
                            data_loss = (w * delta_error**2).mean() + omega_weight * (
                                w * omega_error**2
                            ).mean()
                        else:
                            data_loss = torch.mean(delta_error**2) + omega_weight * torch.mean(
                                omega_error**2
                            )
                    else:
                        data_loss = nn.MSELoss()(pred, batch_y)

                    if lambda_steady_state > 0 and has_tf and pre_fault_mask.any():
                        delta0_batch = batch_X[pre_fault_mask, 1]
                        omega0_batch = batch_X[pre_fault_mask, 2]
                        ss_delta = ((pred[pre_fault_mask, 0] - delta0_batch) ** 2).mean()
                        ss_omega = ((pred[pre_fault_mask, 1] - omega0_batch) ** 2).mean()
                        ss_loss = ss_delta + omega_weight * ss_omega
                    else:
                        ss_loss = torch.tensor(0.0, device=self.device)

                    if use_ic_over_prefault and has_tf and pre_fault_mask.any():
                        delta0_batch = batch_X[pre_fault_mask, 1]
                        omega0_batch = batch_X[pre_fault_mask, 2]
                        ic_loss_delta = ((pred[pre_fault_mask, 0] - delta0_batch) ** 2).mean()
                        ic_loss_omega = ((pred[pre_fault_mask, 1] - omega0_batch) ** 2).mean()
                        ic_loss = ic_loss_delta + omega_weight * ic_loss_omega
                    else:
                        min_t = t_values.min()
                        tolerance = 1e-6
                        ic_mask = (t_values - min_t).abs() < tolerance
                        if ic_mask.sum() > 0:
                            delta0_batch = batch_X[ic_mask, 1]
                            omega0_batch = batch_X[ic_mask, 2]
                            ic_loss_delta = torch.mean((pred[ic_mask, 0] - delta0_batch) ** 2)
                            ic_loss_omega = torch.mean((pred[ic_mask, 1] - omega0_batch) ** 2)
                            ic_loss = ic_loss_delta + omega_weight * ic_loss_omega
                        else:
                            ic_loss = torch.tensor(0.0, device=self.device)

                    loss = data_loss + lambda_steady_state * ss_loss + lambda_ic * ic_loss

                    val_loss += loss.item()
                    val_data_loss += data_loss.item()
                    val_ic_loss += ic_loss.item()

            val_loss /= len(val_loader)
            val_data_loss /= len(val_loader)
            val_ic_loss /= len(val_loader)

            # Update learning rate scheduler (matching PINN - uses validation loss)
            scheduler_lr.step(val_loss)

            history["train_losses"].append(train_loss)
            history["val_losses"].append(val_loss)
            history["epochs"].append(epoch + 1)

            # Calculate epoch time
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_time)

            # Calculate elapsed and remaining time
            elapsed_time = epoch_end_time - training_start_time
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = epochs - (epoch + 1)
            estimated_remaining_time = avg_epoch_time * remaining_epochs

            # Format time strings
            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.1f}s"
                elif seconds < 3600:
                    return f"{seconds / 60:.1f}m"
                else:
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    return f"{hours}h{minutes}m"

            elapsed_str = format_time(elapsed_time)
            remaining_str = format_time(estimated_remaining_time)
            epoch_time_str = format_time(epoch_time)

            # Early stopping and best model tracking
            if val_loss < best_val_loss:
                old_best_val_loss = best_val_loss
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_model_state = self.model.state_dict().copy()  # Save best model
                patience_counter = 0

                # Print improvement message (similar to PINN training)
                if epoch > 0:
                    improvement = old_best_val_loss - val_loss
                    print(
                        f"  ✅ New best val loss: {old_best_val_loss:.6f} → {val_loss:.6f} (improved by {improvement:.6f})"
                    )
                else:
                    print(f"  ✅ Initial best val loss: {val_loss:.6f}")
            else:
                patience_counter += 1

            # Print progress every 10 epochs or on first/last epoch
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                best_str = f" (Best: {best_val_loss:.6f})" if best_val_loss < float("inf") else ""
                print(
                    f"  Epoch {epoch + 1}/{epochs}: Train Loss={train_loss:.6f} (Data={train_data_loss:.6f}, IC={train_ic_loss:.6f}), "
                    f"Val Loss={val_loss:.6f} (Data={val_data_loss:.6f}, IC={val_ic_loss:.6f}){best_str}"
                )
                print(
                    f"    ⏱️  Time: Epoch={epoch_time_str} | Elapsed={elapsed_str} | Remaining≈{remaining_str}"
                )
                sys.stdout.flush()

            # Early stopping (only if patience is set, matching PINN behavior)
            if patience is not None and patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                print(f"\n" + "=" * 70)
                print("TRAINING COMPLETE")
                print("=" * 70)
                print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch + 1})")
                print(f"Final training loss: {train_loss:.6f}")
                print(f"Total epochs: {epochs}")
                if epoch_times:
                    total_time = sum(epoch_times)
                    avg_time = total_time / len(epoch_times)
                    print(f"Training Time Summary:")
                    print(f"  Total time: {int(total_time // 60)}m {int(total_time % 60)}s")
                    print(f"  Average time per epoch: {int(avg_time)}s")
                print("=" * 70)
                sys.stdout.flush()
                # Restore best model before returning
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                break

        # Ensure best model is loaded at the end
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Print training complete summary (if not already printed by early stopping)
        if patience is None or patience_counter < patience:
            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
            if history["train_losses"]:
                print(f"Final training loss: {history['train_losses'][-1]:.6f}")
            print(f"Total epochs: {len(history['epochs'])}")
            if epoch_times:
                total_time = sum(epoch_times)
                avg_time = total_time / len(epoch_times)
                print(f"Training Time Summary:")
                print(f"  Total time: {int(total_time // 60)}m {int(total_time % 60)}s")
                print(f"  Average time per epoch: {int(avg_time)}s")
            print("=" * 70)

        return history

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate the model.

        Parameters:
        -----------
        test_loader : DataLoader
            Test data loader

        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        all_delta_pred = []
        all_omega_pred = []
        all_delta_true = []
        all_omega_true = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                pred = self.model(batch_X)  # Model outputs normalized predictions

                # Denormalize predictions and targets for metric computation
                # (metrics should be computed on original scale)
                delta_pred_norm = pred[:, 0].cpu().numpy()
                omega_pred_norm = pred[:, 1].cpu().numpy()
                delta_true_norm = batch_y[:, 0].cpu().numpy()
                omega_true_norm = batch_y[:, 1].cpu().numpy()

                # Denormalize using scalers (StandardScaler or fixed scale to match PINN)
                if "delta_fixed_scale" in self.scalers and "omega_fixed_scale" in self.scalers:
                    delta_pred = delta_pred_norm * self.scalers["delta_fixed_scale"]
                    omega_pred = omega_pred_norm * self.scalers["omega_fixed_scale"]
                    delta_true = delta_true_norm * self.scalers["delta_fixed_scale"]
                    omega_true = omega_true_norm * self.scalers["omega_fixed_scale"]
                elif "delta" in self.scalers and "omega" in self.scalers:
                    delta_pred = (
                        self.scalers["delta"]
                        .inverse_transform(delta_pred_norm.reshape(-1, 1))
                        .flatten()
                    )
                    omega_pred = (
                        self.scalers["omega"]
                        .inverse_transform(omega_pred_norm.reshape(-1, 1))
                        .flatten()
                    )
                    delta_true = (
                        self.scalers["delta"]
                        .inverse_transform(delta_true_norm.reshape(-1, 1))
                        .flatten()
                    )
                    omega_true = (
                        self.scalers["omega"]
                        .inverse_transform(omega_true_norm.reshape(-1, 1))
                        .flatten()
                    )
                else:
                    delta_pred = delta_pred_norm
                    omega_pred = omega_pred_norm
                    delta_true = delta_true_norm
                    omega_true = omega_true_norm

                all_delta_pred.extend(delta_pred)
                all_omega_pred.extend(omega_pred)
                all_delta_true.extend(delta_true)
                all_omega_true.extend(omega_true)

        # Compute metrics
        metrics = compute_trajectory_metrics(
            delta_pred=np.array(all_delta_pred),
            omega_pred=np.array(all_omega_pred),
            delta_true=np.array(all_delta_true),
            omega_true=np.array(all_omega_true),
        )

        return metrics
