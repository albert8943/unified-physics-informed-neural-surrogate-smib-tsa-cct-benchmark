"""
Training Workflow Module.

This module handles model training with checkpointing and returns training history.
"""

import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pinn.core import AdaptiveLossWeightScheduler, PhysicsInformedLoss
from pinn.trajectory_prediction import (
    TrajectoryPredictionPINN,
    TrajectoryPredictionPINN_PeInput,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.data_utils import generate_collocation_batch
from utils.normalization import (
    PhysicsNormalizer,
    denormalize_array,
    denormalize_tensor,
    denormalize_value,
    normalize_array,
    normalize_value,
    set_model_standardization_to_identity,
)

from .common_repository import (
    find_data_by_fingerprint,
    find_model_by_config_hash,
    save_model_to_common,
)
from .fingerprinting import compute_data_fingerprint, compute_model_config_hash
from .utils import generate_timestamped_filename, load_json, save_json
from datetime import datetime


def train_model(
    config: Dict,
    data_path: Path,
    output_dir: Path,
    seed: Optional[int] = None,
    resume_from: Optional[Path] = None,
    use_common_repository: bool = True,
    force_retrain: bool = False,
) -> Tuple[Path, Dict]:
    """
    Train PINN model based on configuration.

    Parameters:
    -----------
    config : dict
        Configuration dictionary with training parameters
    data_path : Path
        Path to training data CSV file
    output_dir : Path
        Directory to save trained model and checkpoints (used if use_common_repository=False)
    seed : int, optional
        Random seed for reproducibility
    resume_from : Path, optional
        Path to checkpoint to resume from
    use_common_repository : bool
        If True, save model to common repository (default: True)
    force_retrain : bool
        If True, force retraining even if model exists in common repository (default: False)

    Returns:
    --------
    model_path : Path
        Path to best trained model (common repository path if use_common_repository=True)
    training_history : dict
        Training history with losses and metrics
    """
    # Determine task type
    task = config.get("data", {}).get("task", "trajectory")

    # Check common repository for existing model if enabled
    if use_common_repository and not force_retrain:
        config_hash = compute_model_config_hash(config)
        # Try to get data fingerprint from data path metadata
        data_fingerprint = None
        if data_path.parent.name == "common":
            # Data is in common repository, try to extract fingerprint from metadata
            try:
                metadata_path = data_path.with_suffix(".json").with_name(
                    data_path.stem + "_metadata.json"
                )
                if metadata_path.exists():
                    metadata = load_json(metadata_path)
                    data_fingerprint = metadata.get("data_fingerprint")
            except Exception:
                pass

        # If we have data fingerprint, check for model with same config and data
        if data_fingerprint:
            existing_model = find_model_by_config_hash(config_hash, task)
            if existing_model and existing_model.exists():
                # Verify it was trained on the same data
                try:
                    model_metadata_path = existing_model.with_suffix(".json").with_name(
                        existing_model.stem + "_metadata.json"
                    )
                    if model_metadata_path.exists():
                        model_metadata = load_json(model_metadata_path)
                        if model_metadata.get("data_fingerprint") == data_fingerprint:
                            print(
                                f"✓ Found existing model in common repository:"
                                f"{existing_model.name}"
                            )
                            print("  Reusing existing model (use --force-retrain to override)")
                            # Load training history if available
                            training_history = model_metadata.get("training_history", {})
                            return existing_model, training_history
                except Exception:
                    pass

    # Fallback to output_dir if common repository not used
    if not use_common_repository:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # Get training configuration
    train_config = config.get("training", {})
    model_config = config.get("model", {})
    loss_config = config.get("loss", {})

    # Setup device
    device_str = train_config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Output: {output_dir}")

    # Load and prepare data
    print("\nLoading data...")
    train_data, val_data, scalers = _load_and_prepare_data(data_path, seed=seed)

    # Try to load test data if it exists in the same directory (for complete verification)
    test_data = None
    if "train_data_" in data_path.name:
        test_path = data_path.parent / data_path.name.replace("train_data_", "test_data_")
        if test_path.exists():
            try:
                test_data = pd.read_csv(test_path)
                print(f"✓ Detected test data file: {test_path.name}")
            except Exception:
                pass

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
    train_test_overlap = train_scenario_set & test_scenario_set if test_data is not None else set()
    val_test_overlap = val_scenario_set & test_scenario_set if test_data is not None else set()
    total_unique = len(train_scenario_set | val_scenario_set | test_scenario_set)

    print(f"Training Set:")
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
                    f"⚠️ Note: Expected ~146 total scenarios, found {total_unique}. Missing"
                    f"scenarios likely in test set."
                )

    print("=" * 70)

    # Initialize model
    print("\nInitializing model...")

    # Check if using Pe(t) as input - check multiple config locations
    # Priority: model.use_pe_as_input > data.generation.use_pe_as_input > model.input_method > top-level input_method
    use_pe_as_input = model_config.get("use_pe_as_input", False) or config.get("data", {}).get(
        "generation", {}
    ).get("use_pe_as_input", False)
    input_method = (
        model_config.get("input_method")
        or config.get("input_method")
        or "reactance"  # Default fallback
    )

    use_pe_path = input_method in ("pe_direct", "pe_direct_7") or use_pe_as_input

    if use_pe_path:
        if input_method == "pe_direct_7":
            input_method = "pe_direct_7"
            pe_input_dim = 7
        else:
            if input_method != "pe_direct":
                input_method = "pe_direct"
            pe_input_dim = int(model_config.get("input_dim", 9))
            if pe_input_dim not in (7, 9):
                pe_input_dim = 9
        model = TrajectoryPredictionPINN_PeInput(
            input_dim=pe_input_dim,
            hidden_dims=model_config.get("hidden_dims", [64, 64, 64, 64]),
            activation=model_config.get("activation", "tanh"),
            use_residual=model_config.get("use_residual", False),
            dropout=model_config.get("dropout", 0.0),
            use_standardization=model_config.get("use_standardization", True),
        ).to(device)
        print(f"✓ Using Pe(t) input model (input_dim={pe_input_dim}, input_method={input_method})")
        input_dim_actual = pe_input_dim
    else:
        # Use reactance-based model
        input_method = "reactance"  # Ensure consistency
        model = TrajectoryPredictionPINN(
            input_dim=model_config.get("input_dim", 11),
            hidden_dims=model_config.get("hidden_dims", [64, 64, 64, 64]),
            activation=model_config.get("activation", "tanh"),
            use_residual=model_config.get("use_residual", False),
            dropout=model_config.get("dropout", 0.0),
            use_standardization=model_config.get("use_standardization", True),
        ).to(device)
        input_dim_actual = model_config.get("input_dim", 11)
        print(f"✓ Using reactance-based model (input_dim={input_dim_actual})")

    # Ensure input_method is in model_config for checkpoint saving
    model_config["input_method"] = input_method
    model_config["use_pe_as_input"] = input_method in ("pe_direct", "pe_direct_7")
    model_config["input_dim"] = input_dim_actual

    # Set model standardization to identity (critical fix)
    # Output dimension is fixed at 2 (delta, omega)
    set_model_standardization_to_identity(model, input_dim_actual, 2, str(device))

    print(f"✓ Model initialized")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup loss function
    print("\nSetting up loss function...")
    physics_normalizer = PhysicsNormalizer(scalers, device=str(device))

    # Get normalized loss configuration
    use_normalized_loss = loss_config.get("use_normalized_loss", True)  # Default: enabled
    scale_to_norm = loss_config.get("scale_to_norm", [1.0, 100.0])  # [delta, omega]
    if isinstance(scale_to_norm, list):
        scale_to_norm = torch.tensor([scale_to_norm], device=device)

    # Check if using Pe direct mode
    use_pe_direct = loss_config.get("use_pe_direct", False)
    if input_method in ("pe_direct", "pe_direct_7") or use_pe_as_input:
        use_pe_direct = True

    loss_fn = PhysicsInformedLoss(
        lambda_data=loss_config.get("lambda_data", 1.0),
        lambda_physics=loss_config.get("lambda_physics", 0.0),  # Controlled by scheduler
        lambda_ic=loss_config.get("lambda_ic", 10.0),
        lambda_steady_state=loss_config.get("lambda_steady_state", 0.0),
        fn=60.0,  # System frequency
        physics_normalizer=physics_normalizer,
        use_normalized_loss=use_normalized_loss,
        scale_to_norm=scale_to_norm,
        use_pe_direct=use_pe_direct,  # Use Pe(t) directly from ANDES
    )

    print(f"✓ Loss function initialized")
    print(f"  Data weight: {loss_config.get('lambda_data', 1.0)}")
    print(f"  Physics weight: Adaptive (scheduler controlled)")
    print(f"  IC weight: {loss_config.get('lambda_ic', 10.0)}")
    print(f"  Steady-state weight: {loss_config.get('lambda_steady_state', 0.0)}")
    print(f"  Normalized state loss: {'Enabled' if use_normalized_loss else 'Disabled'}")
    if use_normalized_loss:
        scale_vals = (
            scale_to_norm.cpu().numpy()[0]
            if isinstance(scale_to_norm, torch.Tensor)
            else scale_to_norm
        )
        print(f"    Scaling factors: delta={scale_vals[0]:.1f}, omega={scale_vals[1]:.1f}")
    print(f"  Pe direct mode: {'Enabled' if use_pe_direct else 'Disabled'}")

    # Setup adaptive scheduler
    # Check if using fixed lambda (no adaptive scheduling)
    use_fixed_lambda = loss_config.get("use_fixed_lambda", False)
    if use_fixed_lambda:
        # For fixed lambda, set ratio to always be 1.0
        adaptive_scheduler = AdaptiveLossWeightScheduler(
            initial_ratio=1.0,
            final_ratio=1.0,
            warmup_epochs=0,  # No warmup for fixed lambda
        )
        print("✓ Using fixed lambda_physics (no adaptive scheduling)")
    else:
        # Default adaptive scheduling with gradual increase and loss normalization
        gradual_epochs = train_config.get(
            "adaptive_gradual_epochs", 70
        )  # Default: 70 epochs for gradual increase
        normalize_losses = loss_config.get(
            "normalize_losses_for_adaptive", True
        )  # Default: True (Option 4)
        adaptive_scheduler = AdaptiveLossWeightScheduler(
            initial_ratio=0.0,
            final_ratio=0.5,
            warmup_epochs=30,
            gradual_increase_epochs=gradual_epochs,
            normalize_losses=normalize_losses,
        )
        print("✓ Using adaptive lambda_physics scheduling")
        print(f"  Gradual increase over {gradual_epochs} epochs")
        print(f"  Loss normalization: {'Enabled' if normalize_losses else 'Disabled'}")

    # Setup optimizer
    learning_rate = float(train_config.get("learning_rate", 1e-3))
    weight_decay = float(train_config.get("weight_decay", 1e-5))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    # Get training parameters
    epochs = train_config.get("epochs", 100)
    batch_size = train_config.get("batch_size", None)
    early_stopping_patience = train_config.get("early_stopping_patience", None)  # None = disabled
    max_training_angle_degrees = train_config.get(
        "max_training_angle_degrees", 720.0
    )  # Default: 720°
    lambda_angle = train_config.get("lambda_angle", 0.1)  # Weight for angle penalty

    # Adaptive batch size calculation if not specified
    if batch_size is None:
        num_scenarios = len(train_data["scenario_id"].unique())
        # Target: 6-8 batches per epoch for optimal learning
        target_batches = 7  # Middle of 6-8 range
        calculated_batch_size = num_scenarios // target_batches

        # Apply minimum batch size constraints based on dataset size
        if num_scenarios < 50:
            batch_size = max(4, calculated_batch_size)
        elif num_scenarios < 100:
            batch_size = max(8, calculated_batch_size)
        elif num_scenarios < 200:
            batch_size = max(16, calculated_batch_size)
        else:
            batch_size = max(24, calculated_batch_size)

        # Cap at maximum reasonable batch size
        batch_size = min(32, batch_size)

        batches_per_epoch = num_scenarios / batch_size
        print(f"✓ Adaptive batch size: {batch_size} (calculated from {num_scenarios} scenarios)")
        print(f"  → {batches_per_epoch:.1f} batches per epoch")
    else:
        num_scenarios = len(train_data["scenario_id"].unique())
        batches_per_epoch = num_scenarios / batch_size
        print(f"✓ Batch size: {batch_size} (user-specified)")
        print(f"  → {batches_per_epoch:.1f} batches per epoch from {num_scenarios} scenarios")

    print(f"✓ Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"✓ Epochs: {epochs}")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_path = None  # Track the actual best model path
    training_history: Dict = {
        "train_losses": [],
        "val_losses": [],
        "train_data_losses": [],
        "train_physics_losses": [],
        "train_ic_losses": [],
        "val_data_losses": [],
        "val_physics_losses": [],
        "val_ic_losses": [],
        "learning_rates": [],
        "lambda_physics": [],  # Track lambda_physics over epochs
        "epochs": [],
    }

    if resume_from and resume_from.exists():
        print(f"\nResuming from checkpoint: {resume_from}")
        # weights_only=False needed for PyTorch 2.6+ when checkpoint contains sklearn scalers
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        training_history = checkpoint.get("training_history", training_history)

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    start_time = time.time()
    epoch_times = []  # Track time per epoch for ETA calculation

    # Helper function to format time
    def format_time(seconds):
        """Format seconds into hours, minutes, seconds"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        # Update lambda_physics weight before training
        if use_fixed_lambda:
            # For fixed lambda, always use the config value directly
            lambda_physics = loss_config.get("lambda_physics", 0.1)
            current_ratio = 1.0  # Always 1.0 for fixed lambda
        else:
            # For adaptive scheduling, compute lambda based on loss magnitudes
            if epoch > 0 and len(training_history.get("train_data_losses", [])) > 0:
                prev_data_loss = torch.tensor(
                    training_history["train_data_losses"][-1], device=device
                )
                prev_physics_loss = torch.tensor(
                    training_history["train_physics_losses"][-1], device=device
                )
                lambda_physics = adaptive_scheduler.compute_weight(
                    prev_data_loss, prev_physics_loss, epoch
                )
                current_ratio = adaptive_scheduler.get_current_ratio()
            else:
                adaptive_scheduler.current_epoch = epoch
                current_ratio = adaptive_scheduler.get_current_ratio()
                base_physics_weight = loss_config.get("lambda_physics", 0.1)
                lambda_physics = base_physics_weight * current_ratio

        loss_fn.lambda_physics = lambda_physics

        # Track lambda_physics value
        training_history["lambda_physics"].append(float(lambda_physics))

        # Training step
        train_loss, train_metrics = _train_epoch(
            model,
            train_data,
            loss_fn,
            optimizer,
            adaptive_scheduler,
            scalers,
            physics_normalizer,
            epoch,
            batch_size,
            device,
            lambda_physics,
            max_training_angle_degrees,
            lambda_angle,
        )

        # Validation step
        val_loss, val_metrics = _validate_epoch(
            model, val_data, loss_fn, scalers, physics_normalizer, batch_size, device
        )

        # Learning rate scheduling
        scheduler_lr.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        training_history["epochs"].append(epoch + 1)  # Epoch numbers start from 1
        training_history["train_losses"].append(float(train_loss))
        training_history["val_losses"].append(float(val_loss))
        training_history["train_data_losses"].append(train_metrics.get("data_loss", 0.0))
        training_history["train_physics_losses"].append(train_metrics.get("physics_loss", 0.0))
        training_history["train_ic_losses"].append(train_metrics.get("ic_loss", 0.0))
        training_history["val_data_losses"].append(val_metrics.get("data_loss", 0.0))
        training_history["val_physics_losses"].append(val_metrics.get("physics_loss", 0.0))
        training_history["val_ic_losses"].append(val_metrics.get("ic_loss", 0.0))
        training_history["learning_rates"].append(current_lr)

        # Calculate epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)

        # Calculate elapsed and remaining time
        elapsed_time = epoch_end_time - start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else epoch_time
        remaining_epochs = epochs - (epoch + 1)
        estimated_remaining_time = avg_epoch_time * remaining_epochs

        # Format time strings
        elapsed_str = format_time(elapsed_time)
        remaining_str = format_time(estimated_remaining_time)
        epoch_time_str = format_time(epoch_time)

        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"\nEpoch {epoch + 1}/{epochs}:")
            print(
                f"  Train Loss: {train_loss:.6f} "
                f"(Data: {train_metrics.get('data_loss', 0):.6f}, "
                f"Physics: {train_metrics.get('physics_loss', 0):.6f}, "
                f"IC: {train_metrics.get('ic_loss', 0):.6f})"
            )
            print(
                f"  Val Loss: {val_loss:.6f} "
                f"(Data: {val_metrics.get('data_loss', 0):.6f}, "
                f"Physics: {val_metrics.get('physics_loss', 0):.6f}, "
                f"IC: {val_metrics.get('ic_loss', 0):.6f})"
            )
            print(
                f"LR: {current_lr:.2e}, Lambda Physics: {lambda_physics:.6f}, Physics Ratio:"
                f"{current_ratio:.4f}"
            )
            print(
                f"⏱️ Time: Epoch={epoch_time_str} | Elapsed={elapsed_str} |"
                f"Remaining≈{remaining_str}"
            )
            sys.stdout.flush()  # Ensure output is displayed immediately

        # Save best model
        if val_loss < best_val_loss:
            old_best_val_loss = best_val_loss
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            epochs_no_improve = 0

            # Delete previous best model if it exists (to avoid accumulation)
            if best_model_path is not None and best_model_path.exists():
                try:
                    best_model_path.unlink()
                except Exception as e:
                    print(f"  ⚠️  Could not remove previous best model {best_model_path.name}: {e}")

            # Use timestamped filename for best model
            model_filename = generate_timestamped_filename("best_model", "pth")
            model_path = output_dir / model_filename
            best_model_path = model_path  # Track the actual best model path
            _save_checkpoint(
                model,
                optimizer,
                scalers,
                epoch,
                val_loss,
                best_val_loss,
                training_history,
                model_path,
                input_method=input_method,
                model_config=model_config,
            )

            # Print improvement message
            if epoch > 0:
                improvement = old_best_val_loss - val_loss
                print(
                    f"✅ New best val loss: {old_best_val_loss:.6f} → {val_loss:.6f} (improved by"
                    f"{improvement:.6f})"
                )
            else:
                print(f"  ✅ Initial best val loss: {val_loss:.6f}")
        else:
            # Track no improvement
            epochs_no_improve += 1
            patience_counter += 1

            # Print warning every 10 epochs of no improvement
            if epochs_no_improve % 10 == 0 and epochs_no_improve > 0:
                print(
                    f"⚠️ No improvement for {epochs_no_improve} epochs (best: {best_val_loss:.6f}"
                    f"at epoch {best_epoch})"
                )

            # Early stopping
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch + 1}")
                print(f"   Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
                break

        # Save periodic checkpoint
        if epoch % 20 == 0 or epoch == epochs - 1:
            # Use timestamped filename for checkpoints
            checkpoint_filename = generate_timestamped_filename(
                "checkpoint", "pth", prefix=f"epoch_{epoch:04d}"
            )
            checkpoint_path = output_dir / checkpoint_filename
            _save_checkpoint(
                model,
                optimizer,
                scalers,
                epoch,
                val_loss,
                best_val_loss,
                training_history,
                checkpoint_path,
                input_method=input_method,
                model_config=model_config,
            )

    elapsed = time.time() - start_time
    total_time_str = format_time(elapsed)
    avg_epoch_time_str = format_time(sum(epoch_times) / len(epoch_times) if epoch_times else 0)

    # Generate final timestamped filename for training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = output_dir / f"training_history_{timestamp}.json"

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(
        f"Final training loss: {training_history['train_losses'][-1]:.6f}"
        if training_history["train_losses"]
        else "N/A"
    )
    print(f"Total epochs: {len(training_history['train_losses'])}")
    print(f"Training Time Summary:")
    print(f"  Total time: {total_time_str}")
    print(f"  Average time per epoch: {avg_epoch_time_str}")

    # Use the actual best model path (saved during training)
    if best_model_path is None:
        # Fallback: if no model was saved (shouldn't happen), create a path
        best_model_path = output_dir / f"best_model_{timestamp}.pth"
        print(f"⚠️  Warning: No best model was saved during training!")

    print(f"Model saved to: {best_model_path}")

    # Save training history with timestamp
    save_json(training_history, history_path)

    # Save to common repository if enabled
    if use_common_repository:
        # Load the best model state
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model_state = checkpoint["model_state_dict"]

        # Get data fingerprint
        data_fingerprint = None
        if data_path.parent.name == "common":
            # Data is in common repository
            try:
                metadata_path = data_path.with_suffix(".json").with_name(
                    data_path.stem + "_metadata.json"
                )
                if metadata_path.exists():
                    metadata = load_json(metadata_path)
                    data_fingerprint = metadata.get("data_fingerprint")
            except Exception:
                pass

        # If we couldn't get fingerprint from metadata, compute it from config
        if data_fingerprint is None:
            data_fingerprint = compute_data_fingerprint(config)

        # Prepare metrics from training history
        metrics = {
            "best_val_loss": best_val_loss,
            "final_train_loss": (
                training_history["train_losses"][-1] if training_history["train_losses"] else None
            ),
            "final_val_loss": (
                training_history["val_losses"][-1] if training_history["val_losses"] else None
            ),
            "total_epochs": len(training_history["train_losses"]),
            "best_epoch": best_epoch,
        }

        # Save to common repository
        common_model_path, model_metadata = save_model_to_common(
            model_state=model_state,
            task=task,
            config=config,
            data_fingerprint=data_fingerprint,
            data_path=data_path,
            metrics=metrics,
            metadata={"training_history": training_history},
        )

        print(f"\n✓ Model also saved to common repository: {common_model_path.name}")
        print(f"  Config hash: {model_metadata.get('model_config_hash', 'N/A')[:16]}...")
        print(f"  Data fingerprint: {data_fingerprint[:16]}...")

        # Return common repository path
        return common_model_path, training_history

    # Verify: Print confirmation that we're returning the best model path
    if best_model_path:
        print(f"\n[VERIFY] PINN best model saved:")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Model path: {best_model_path.name}")

    return best_model_path, training_history


def _load_and_prepare_data(
    data_path: Path, seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load and prepare training data."""
    # Check if this is a preprocessed train file
    if "train_data_" in data_path.name:
        # Look for corresponding val file in same directory
        val_path = data_path.parent / data_path.name.replace("train_data_", "val_data_")

        if val_path.exists():
            # Use preprocessed splits directly
            print(f"✓ Detected preprocessed data: using train/val files directly")
            train_data = pd.read_csv(data_path)
            val_data = pd.read_csv(val_path)

            # Fit scalers on training data
            scalers = _fit_scalers(train_data)

            return train_data, val_data, scalers
        else:
            print(f"⚠️  Warning: train_data file found but val_data file not found: {val_path}")
            print(f"   Will split the train_data file by scenario instead")

    # Normal case: load data and split by scenario
    data = pd.read_csv(data_path)

    # Split by scenario
    scenarios = data["scenario_id"].unique()

    # Handle edge case: only 1 scenario
    if len(scenarios) == 1:
        print(
            f"⚠️  Warning: Only 1 scenario found. Using all data for training, no validation split."
        )
        print(f"   Consider generating more data or using preprocessed splits.")
        train_data = data
        val_data = data.iloc[:0].copy()  # Empty dataframe with same columns
    else:
        train_scenarios, val_scenarios = train_test_split(
            scenarios, test_size=0.15, random_state=seed or 42
        )
        train_data = data[data["scenario_id"].isin(train_scenarios)]
        val_data = data[data["scenario_id"].isin(val_scenarios)]

    # Fit scalers on training data
    scalers = _fit_scalers(train_data)

    return train_data, val_data, scalers


def _fit_scalers(train_data: pd.DataFrame) -> Dict:
    """Fit sklearn scalers on training data."""
    scalers = {}

    # Sample subset for efficiency
    sample_scenarios = train_data["scenario_id"].unique()[
        : min(100, len(train_data["scenario_id"].unique()))
    ]
    sample_data = train_data[train_data["scenario_id"].isin(sample_scenarios)]

    # Time, states
    scalers["time"] = StandardScaler().fit(sample_data["time"].values.reshape(-1, 1))
    scalers["delta"] = StandardScaler().fit(sample_data["delta"].values.reshape(-1, 1))
    scalers["omega"] = StandardScaler().fit(sample_data["omega"].values.reshape(-1, 1))

    # Pe (electrical power) - if available
    if "Pe" in sample_data.columns:
        scalers["Pe"] = StandardScaler().fit(sample_data["Pe"].values.reshape(-1, 1))

    # Parameters (scenario-level) - handle both param_* and direct column names
    scenario_data = train_data.groupby("scenario_id").first().reset_index()

    # Try param_* columns first, fall back to direct column names
    H_col = "param_H" if "param_H" in scenario_data.columns else "H"
    D_col = "param_D" if "param_D" in scenario_data.columns else "D"
    Pm_col = "param_Pm" if "param_Pm" in scenario_data.columns else "Pm"
    tc_col = "param_tc" if "param_tc" in scenario_data.columns else "tc"
    # Alpha column (for unified load variation) - try multiple possible column names
    # NEW: Prefer "alpha" (unified approach), fallback to "load"/"Pload" (backward compatibility)
    alpha_col = None
    if "alpha" in scenario_data.columns:
        alpha_col = "alpha"
    elif "load" in scenario_data.columns:
        alpha_col = "load"
    elif "Pload" in scenario_data.columns:
        alpha_col = "Pload"
    elif "param_load" in scenario_data.columns:
        alpha_col = "param_load"

    scalers["H"] = StandardScaler().fit(scenario_data[H_col].values.reshape(-1, 1))
    scalers["D"] = StandardScaler().fit(scenario_data[D_col].values.reshape(-1, 1))
    scalers["Pm"] = StandardScaler().fit(scenario_data[Pm_col].values.reshape(-1, 1))
    # Create alpha scaler if alpha column exists (for unified load variation mode)
    if alpha_col is not None:
        scalers["alpha"] = StandardScaler().fit(scenario_data[alpha_col].values.reshape(-1, 1))
        # Also create "load" and "Pload" keys for backward compatibility
        scalers["load"] = scalers["alpha"]
        scalers["Pload"] = scalers["alpha"]
    # Make reactance columns optional (may not be present in all datasets)
    if "Xprefault" in scenario_data.columns:
        scalers["Xprefault"] = StandardScaler().fit(
            scenario_data["Xprefault"].values.reshape(-1, 1)
        )
    if "Xfault" in scenario_data.columns:
        scalers["Xfault"] = StandardScaler().fit(scenario_data["Xfault"].values.reshape(-1, 1))
    if "Xpostfault" in scenario_data.columns:
        scalers["Xpostfault"] = StandardScaler().fit(
            scenario_data["Xpostfault"].values.reshape(-1, 1)
        )
    scalers["tf"] = StandardScaler().fit(scenario_data["tf"].values.reshape(-1, 1))
    scalers["tc"] = StandardScaler().fit(scenario_data[tc_col].values.reshape(-1, 1))

    # REVERT: Use separate scalers for delta0/omega0 (like December experiments)
    # December experiments achieved R² Delta = 0.881 using separate scalers.
    # Separate scalers create beneficial structure: tight initial condition space → wide trajectory space
    # This makes learning easier than using same scalers (wide → wide).
    # Fit delta0/omega0 scalers on initial conditions only (scenario_data)
    scalers["delta0"] = StandardScaler().fit(scenario_data["delta0"].values.reshape(-1, 1))
    scalers["omega0"] = StandardScaler().fit(scenario_data["omega0"].values.reshape(-1, 1))

    return scalers


def _extract_and_normalize_scenario_data(
    scenario_data: pd.DataFrame, scalers: Dict, device: torch.device
) -> Dict:
    """
    Extract and normalize scenario data for training.

    Returns normalized tensors ready for model input.
    """
    scenario_data = scenario_data.sort_values("time")

    # Extract raw values
    t_raw = scenario_data["time"].values.astype(np.float32)
    delta_obs_raw = scenario_data["delta"].values.astype(np.float32)
    omega_obs_raw = scenario_data["omega"].values.astype(np.float32)
    # Pe (electrical power) - if available
    pe_obs_raw = None
    if "Pe" in scenario_data.columns:
        pe_obs_raw = scenario_data["Pe"].values.astype(np.float32)

    row = scenario_data.iloc[0]
    delta0_raw = float(row.get("delta0", delta_obs_raw[0]))
    omega0_raw = float(row.get("omega0", omega_obs_raw[0]))
    H_raw = float(row.get("param_H", row.get("H", 5.0)))
    D_raw = float(row.get("param_D", row.get("D", 1.0)))
    Pm_raw = float(row.get("param_Pm", row.get("Pm", 0.8)))  # For physics loss
    # For model input: use load if available (load variation mode), otherwise fallback to Pm
    # NEW: Prefer alpha (unified approach), fallback to load/Pload (backward compatibility)
    alpha_raw = float(
        row.get("alpha", row.get("load", row.get("Pload", row.get("param_load", Pm_raw))))
    )
    Xprefault_raw = float(row.get("Xprefault", 0.5))
    Xfault_raw = float(row.get("Xfault", 0.0001))
    Xpostfault_raw = float(row.get("Xpostfault", 0.5))
    tf_raw = float(row.get("tf", 1.0))
    tc_raw = float(row.get("tc", row.get("param_tc", 1.2)))

    # Normalize
    t_data = torch.tensor(
        normalize_array(t_raw, scalers["time"]), dtype=torch.float32, device=device
    )
    delta_obs = torch.tensor(
        normalize_array(delta_obs_raw, scalers["delta"]),
        dtype=torch.float32,
        device=device,
    )
    omega_obs = torch.tensor(
        normalize_array(omega_obs_raw, scalers["omega"]),
        dtype=torch.float32,
        device=device,
    )

    # Normalize Pe if available
    pe_obs = None
    if pe_obs_raw is not None and "Pe" in scalers:
        pe_obs = torch.tensor(
            normalize_array(pe_obs_raw, scalers["Pe"]),
            dtype=torch.float32,
            device=device,
        )

    # REVERT: Use delta0/omega0 scalers for input normalization (like December experiments)
    # December experiments used separate scalers: delta0_scaler for input, delta_scaler for output
    # This creates beneficial structure: tight initial condition space → wide trajectory space
    # Input normalization uses delta0/omega0 scalers (fitted on initial conditions only)
    delta0_input = torch.tensor(
        [normalize_value(delta0_raw, scalers["delta0"])],
        dtype=torch.float32,
        device=device,
    )
    omega0_input = torch.tensor(
        [normalize_value(omega0_raw, scalers["omega0"])],
        dtype=torch.float32,
        device=device,
    )

    # For IC loss, normalize initial conditions with delta/omega scalers for comparison
    # The model predictions are in delta/omega normalized space, so we need to normalize
    # the observed initial conditions with delta/omega scalers (not delta0/omega0) for IC loss comparison
    delta0_for_ic = torch.tensor(
        [normalize_value(delta0_raw, scalers["delta"])],
        dtype=torch.float32,
        device=device,
    )
    omega0_for_ic = torch.tensor(
        [normalize_value(omega0_raw, scalers["omega"])],
        dtype=torch.float32,
        device=device,
    )

    H = torch.tensor([normalize_value(H_raw, scalers["H"])], dtype=torch.float32, device=device)
    D = torch.tensor([normalize_value(D_raw, scalers["D"])], dtype=torch.float32, device=device)
    Pm = torch.tensor(
        [normalize_value(Pm_raw, scalers["Pm"])], dtype=torch.float32, device=device
    )  # For physics loss
    # For model input: use alpha scaler if available, otherwise use Pm scaler
    alpha_scaler = scalers.get("alpha", scalers.get("load", scalers.get("Pload", scalers["Pm"])))
    alpha = torch.tensor(
        [normalize_value(alpha_raw, alpha_scaler)], dtype=torch.float32, device=device
    )
    # Make reactance tensors optional
    Xprefault = None
    Xfault = None
    Xpostfault = None
    if "Xprefault" in scalers:
        Xprefault = torch.tensor(
            [normalize_value(Xprefault_raw, scalers["Xprefault"])],
            dtype=torch.float32,
            device=device,
        )
    if "Xfault" in scalers:
        Xfault = torch.tensor(
            [normalize_value(Xfault_raw, scalers["Xfault"])],
            dtype=torch.float32,
            device=device,
        )
    if "Xpostfault" in scalers:
        Xpostfault = torch.tensor(
            [normalize_value(Xpostfault_raw, scalers["Xpostfault"])],
            dtype=torch.float32,
            device=device,
        )
    tf = torch.tensor([normalize_value(tf_raw, scalers["tf"])], dtype=torch.float32, device=device)
    tc = torch.tensor([normalize_value(tc_raw, scalers["tc"])], dtype=torch.float32, device=device)

    result = {
        "t_data": t_data,
        "delta_obs": delta_obs,
        "omega_obs": omega_obs,
        "delta0": delta0_input,
        "omega0": omega0_input,
        "delta0_for_ic": delta0_for_ic,
        "omega0_for_ic": omega0_for_ic,
        "H": H,
        "D": D,
        "Pm": Pm,  # For physics loss
        "alpha": alpha,  # For model input (unified approach)
        "Pload": alpha,  # Backward compatibility alias
        "tf": tf,
        "tc": tc,
    }

    # Add reactance tensors only if they exist
    if Xprefault is not None:
        result["Xprefault"] = Xprefault
    if Xfault is not None:
        result["Xfault"] = Xfault
    if Xpostfault is not None:
        result["Xpostfault"] = Xpostfault

    # Add Pe if available
    if pe_obs is not None:
        result["Pe"] = pe_obs

    return result


def _prepare_physics_loss_inputs(
    delta_colloc: torch.Tensor,
    omega_colloc: torch.Tensor,
    t_colloc: torch.Tensor,
    H: torch.Tensor,
    D: torch.Tensor,
    Pm: torch.Tensor,
    Xprefault: Optional[torch.Tensor],
    Xfault: Optional[torch.Tensor],
    Xpostfault: Optional[torch.Tensor],
    tf: torch.Tensor,
    tc: torch.Tensor,
    scalers: Dict,
    device: torch.device,
) -> Dict:
    """
    Denormalize values for physics loss computation and compute scaling factors.

    NOTE: All denormalization operations preserve gradients for backpropagation.
    """
    # Denormalize collocation predictions
    delta_colloc_phys = denormalize_tensor(delta_colloc, scalers["delta"], device=str(device))
    omega_colloc_phys = denormalize_tensor(omega_colloc, scalers["omega"], device=str(device))
    t_colloc_phys = denormalize_tensor(t_colloc, scalers["time"], device=str(device))

    # Denormalize parameters
    H_phys = denormalize_tensor(H, scalers["H"], device=str(device))
    M_phys = 2.0 * H_phys  # M = 2*H for 60 Hz
    D_phys = denormalize_tensor(D, scalers["D"], device=str(device))
    Pm_phys = denormalize_tensor(Pm, scalers["Pm"], device=str(device))

    # Denormalize reactance parameters only if they exist
    Xprefault_phys = None
    Xfault_phys = None
    Xpostfault_phys = None
    if Xprefault is not None and "Xprefault" in scalers:
        Xprefault_phys = denormalize_tensor(Xprefault, scalers["Xprefault"], device=str(device))
    if Xfault is not None and "Xfault" in scalers:
        Xfault_phys = denormalize_tensor(Xfault, scalers["Xfault"], device=str(device))
    if Xpostfault is not None and "Xpostfault" in scalers:
        Xpostfault_phys = denormalize_tensor(Xpostfault, scalers["Xpostfault"], device=str(device))

    tf_phys = denormalize_tensor(tf, scalers["tf"], device=str(device))
    tc_phys = denormalize_tensor(tc, scalers["tc"], device=str(device))

    # Compute derivative scaling factors
    std_time = torch.tensor(scalers["time"].scale_[0], dtype=torch.float32, device=device)
    std_delta = torch.tensor(scalers["delta"].scale_[0], dtype=torch.float32, device=device)
    time_scale = std_delta / std_time
    time_scale_sq = time_scale / std_time

    result = {
        "delta_colloc_phys": delta_colloc_phys,
        "omega_colloc_phys": omega_colloc_phys,
        "t_colloc_phys": t_colloc_phys,
        "M_phys": M_phys,
        "D_phys": D_phys,
        "Pm_phys": Pm_phys,
        "tf_phys": tf_phys,
        "tc_phys": tc_phys,
        "time_scale": time_scale,
        "time_scale_sq": time_scale_sq,
    }

    # Add reactance values only if they exist
    if Xprefault_phys is not None:
        result["Xprefault_phys"] = Xprefault_phys
    if Xfault_phys is not None:
        result["Xfault_phys"] = Xfault_phys
    if Xpostfault_phys is not None:
        result["Xpostfault_phys"] = Xpostfault_phys

    return result


def _train_epoch(
    model,
    train_data,
    loss_fn,
    optimizer,
    adaptive_scheduler,
    scalers,
    physics_normalizer,
    epoch,
    batch_size,
    device,
    lambda_physics,
    max_training_angle_degrees,
    lambda_angle,
):
    """Train for one epoch with full normalization procedures."""
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_physics_loss = 0.0
    total_ic_loss = 0.0
    n_batches = 0

    # Get scenarios and shuffle
    scenarios = train_data["scenario_id"].unique()
    np.random.shuffle(scenarios)

    # Limit scenarios for efficiency (can be adjusted)
    max_scenarios = min(100, len(scenarios))
    scenarios_to_use = scenarios[:max_scenarios]

    # Collocation points per scenario
    n_colloc_per_scenario = 200

    # Process in batches
    for batch_idx in range(0, len(scenarios_to_use), batch_size):
        batch_scenario_ids = scenarios_to_use[batch_idx : batch_idx + batch_size]

        optimizer.zero_grad()
        batch_total_loss = 0.0
        batch_data_loss = 0.0
        batch_physics_loss = 0.0
        batch_ic_loss = 0.0
        batch_count = 0

        # Process each scenario in the batch
        for scenario_id in batch_scenario_ids:
            scenario_data = train_data[train_data["scenario_id"] == scenario_id].copy()

            if len(scenario_data) < 10:
                continue

            scenario_data = scenario_data.sort_values("time")

            # Extract and normalize scenario data
            normalized_data = _extract_and_normalize_scenario_data(scenario_data, scalers, device)
            t_data = normalized_data["t_data"]
            delta_obs = normalized_data["delta_obs"]
            omega_obs = normalized_data["omega_obs"]
            delta0 = normalized_data["delta0"]
            omega0 = normalized_data["omega0"]
            delta0_for_ic = normalized_data["delta0_for_ic"]
            omega0_for_ic = normalized_data["omega0_for_ic"]
            H = normalized_data["H"]
            D = normalized_data["D"]
            Pm = normalized_data["Pm"]  # For physics loss
            # Extract alpha for Pe-based model (unified approach)
            # alpha is always in normalized_data (created in _extract_and_normalize_scenario_data)
            alpha = normalized_data["alpha"]
            Xprefault = normalized_data.get("Xprefault", None)
            Xfault = normalized_data.get("Xfault", None)
            Xpostfault = normalized_data.get("Xpostfault", None)
            tf = normalized_data["tf"]
            tc = normalized_data["tc"]

            # Extract Pe if available (for Pe-based model)
            Pe = normalized_data.get("Pe", None)

            # Predict at data points - handle both model types
            if isinstance(model, TrajectoryPredictionPINN_PeInput):
                # Pe-based model requires Pe(t) input
                if Pe is None:
                    raise ValueError(
                        "Pe column not found in data. Pe-based model requires Pe(t) measurements."
                    )
                delta_pred, omega_pred = model.predict_trajectory(
                    t=t_data,
                    delta0=delta0,
                    omega0=omega0,
                    H=H,
                    D=D,
                    alpha=alpha,  # Use alpha (unified approach) instead of Pload
                    Pe=Pe,
                    tf=normalized_data["tf"],
                    tc=normalized_data["tc"],
                )
            else:
                # Reactance-based model - provide defaults if reactance values are missing
                if Xprefault is None:
                    if "Xprefault" in scalers:
                        Xprefault = torch.tensor(
                            [normalize_value(0.5, scalers["Xprefault"])],
                            dtype=torch.float32,
                            device=device,
                        )
                    else:
                        Xprefault = torch.tensor([0.5], dtype=torch.float32, device=device)
                if Xfault is None:
                    if "Xfault" in scalers:
                        Xfault = torch.tensor(
                            [normalize_value(0.0001, scalers["Xfault"])],
                            dtype=torch.float32,
                            device=device,
                        )
                    else:
                        Xfault = torch.tensor([0.0001], dtype=torch.float32, device=device)
                if Xpostfault is None:
                    if "Xpostfault" in scalers:
                        Xpostfault = torch.tensor(
                            [normalize_value(0.5, scalers["Xpostfault"])],
                            dtype=torch.float32,
                            device=device,
                        )
                    else:
                        Xpostfault = torch.tensor([0.5], dtype=torch.float32, device=device)
                delta_pred, omega_pred = model.predict_trajectory(
                    t=t_data,
                    delta0=delta0,
                    omega0=omega0,
                    H=H,
                    D=D,
                    Pm=Pm,
                    Xprefault=Xprefault,
                    Xfault=Xfault,
                    Xpostfault=Xpostfault,
                    tf=tf,
                    tc=tc,
                )

            # Generate collocation points (in raw time, then normalize)
            t_data_raw = denormalize_array(t_data.cpu().numpy(), scalers["time"])
            tc_raw = denormalize_value(tc.item(), scalers["tc"])
            t_colloc_np_raw = generate_collocation_batch(
                t_data_raw,
                n_colloc=n_colloc_per_scenario,
                strategy="uniform",
                fault_clearing_time=tc_raw,
                seed=42 + epoch,  # Different points each epoch
            )
            t_colloc_np = normalize_array(t_colloc_np_raw, scalers["time"])
            t_colloc = torch.tensor(
                t_colloc_np, dtype=torch.float32, device=device, requires_grad=True
            )

            # Predict at collocation points - handle both model types
            if isinstance(model, TrajectoryPredictionPINN_PeInput):
                # For Pe-based model, interpolate Pe at collocation points
                if Pe is None:
                    raise ValueError("Pe required for Pe-based model at collocation points")

                # Interpolate Pe at collocation points
                # Denormalize times for interpolation
                t_colloc_raw = denormalize_array(t_colloc_np, scalers["time"])
                Pe_raw = denormalize_array(Pe.cpu().numpy(), scalers["Pe"])

                # Interpolate Pe at collocation points
                from scipy.interpolate import interp1d

                pe_interp = interp1d(
                    t_data_raw.flatten(),
                    Pe_raw.flatten(),
                    kind="linear",
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                Pe_colloc_raw = pe_interp(t_colloc_raw.flatten())
                Pe_colloc = torch.tensor(
                    normalize_array(Pe_colloc_raw, scalers["Pe"]),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=True,
                )

                delta_colloc, omega_colloc = model.predict_trajectory(
                    t=t_colloc,
                    delta0=delta0,
                    omega0=omega0,
                    H=H,
                    D=D,
                    alpha=alpha,  # Use alpha (unified approach) instead of Pload
                    Pe=Pe_colloc,
                    tf=normalized_data["tf"],
                    tc=normalized_data["tc"],
                )
            else:
                # Reactance-based model
                delta_colloc, omega_colloc = model.predict_trajectory(
                    t=t_colloc,
                    delta0=delta0,
                    omega0=omega0,
                    H=H,
                    D=D,
                    Pm=Pm,
                    Xprefault=Xprefault,
                    Xfault=Xfault,
                    Xpostfault=Xpostfault,
                    tf=tf,
                    tc=tc,
                )

            # Prepare physics loss inputs (denormalize for physics equation)
            if isinstance(model, TrajectoryPredictionPINN_PeInput):
                # For Pe-based model, prepare physics inputs without reactances
                physics_inputs = {
                    "delta_colloc_phys": denormalize_tensor(
                        delta_colloc, scalers["delta"], device=str(device)
                    ),
                    "omega_colloc_phys": denormalize_tensor(
                        omega_colloc, scalers["omega"], device=str(device)
                    ),
                    "t_colloc_phys": denormalize_tensor(
                        t_colloc, scalers["time"], device=str(device)
                    ),
                    "M_phys": denormalize_tensor(H * 2.0, scalers["H"], device=str(device))
                    * 2.0,  # M = 2*H
                    "D_phys": denormalize_tensor(D, scalers["D"], device=str(device)),
                    "Pm_phys": denormalize_tensor(Pm, scalers["Pm"], device=str(device)),
                }
                # Compute derivative scaling factors
                std_time = torch.tensor(
                    scalers["time"].scale_[0], dtype=torch.float32, device=device
                )
                std_delta = torch.tensor(
                    scalers["delta"].scale_[0], dtype=torch.float32, device=device
                )
                physics_inputs["time_scale"] = std_delta / std_time
                physics_inputs["time_scale_sq"] = physics_inputs["time_scale"] / std_time
            else:
                # Reactance-based model - use existing function
                physics_inputs = _prepare_physics_loss_inputs(
                    delta_colloc,
                    omega_colloc,
                    t_colloc,
                    H,
                    D,
                    Pm,
                    Xprefault,
                    Xfault,
                    Xpostfault,
                    tf,
                    tc,
                    scalers,
                    device,
                )

            # Compute loss using PhysicsInformedLoss
            loss_kwargs = {
                "t_data": t_data,  # Normalized
                "delta_pred": delta_pred,  # Normalized
                "omega_pred": omega_pred,  # Normalized
                "delta_obs": delta_obs,  # Normalized
                "omega_obs": omega_obs,  # Normalized
                "t_colloc": t_colloc,  # Normalized (for derivative computation)
                "delta_colloc": delta_colloc,  # Normalized
                "omega_colloc": omega_colloc,  # Normalized
                "t_ic": t_data[0:1],
                "delta0": delta0_for_ic,  # For IC loss (normalized with output scaler)
                "omega0": omega0_for_ic,  # For IC loss (normalized with output scaler)
                "M": physics_inputs["M_phys"],  # Denormalized
                "D": physics_inputs["D_phys"],  # Denormalized
                "Pm": physics_inputs["Pm_phys"],  # Denormalized
                "delta_colloc_phys": physics_inputs["delta_colloc_phys"],  # Denormalized
                "omega_colloc_phys": physics_inputs["omega_colloc_phys"],  # Denormalized
                "t_colloc_phys": physics_inputs["t_colloc_phys"],  # Denormalized
                "time_scale": physics_inputs["time_scale"],  # Derivative scaling
                "time_scale_sq": physics_inputs["time_scale_sq"],  # Second derivative scaling
            }

            # Add Pe or reactance parameters based on model type
            if isinstance(model, TrajectoryPredictionPINN_PeInput):
                # For Pe-based model, pass Pe directly
                Pe_colloc_phys = denormalize_tensor(Pe_colloc, scalers["Pe"], device=str(device))
                loss_kwargs["Pe_from_andes"] = Pe_colloc_phys
                loss_kwargs["use_pe_direct"] = True
                loss_kwargs["tf_norm"] = tf  # Normalized fault start (for steady-state loss)
            else:
                # Reactance-based model
                if "Xprefault_phys" in physics_inputs:
                    loss_kwargs["Xprefault"] = physics_inputs["Xprefault_phys"]
                if "Xfault_phys" in physics_inputs:
                    loss_kwargs["Xfault"] = physics_inputs["Xfault_phys"]
                if "Xpostfault_phys" in physics_inputs:
                    loss_kwargs["Xpostfault"] = physics_inputs["Xpostfault_phys"]
                loss_kwargs["tf"] = physics_inputs["tf_phys"]
                loss_kwargs["tc"] = physics_inputs["tc_phys"]
                loss_kwargs["tf_norm"] = tf  # Normalized fault start (for steady-state loss)

            losses = loss_fn(**loss_kwargs)

            # Get base loss components
            data_loss_val = losses.get("data", torch.tensor(0.0, device=device))
            physics_loss_val = losses.get("physics", torch.tensor(0.0, device=device))
            ic_loss_val = losses.get("ic", torch.tensor(0.0, device=device))
            steady_state_loss_val = losses.get("steady_state", torch.tensor(0.0, device=device))

            # Add angle penalty (from notebook)
            MAX_TRAINING_ANGLE_RAD = max_training_angle_degrees * (np.pi / 180.0)
            delta_pred_abs = torch.abs(delta_pred)
            exceeds_limit = delta_pred_abs > MAX_TRAINING_ANGLE_RAD

            if exceeds_limit.any():
                # Penalty: mean squared excess beyond limit
                angle_penalty = torch.mean(
                    (delta_pred_abs[exceeds_limit] - MAX_TRAINING_ANGLE_RAD) ** 2
                )
            else:
                angle_penalty = torch.tensor(0.0, device=device)

            # Compute total loss with angle penalty
            lambda_data_val = loss_fn.lambda_data
            lambda_physics_val = lambda_physics
            lambda_ic_val = loss_fn.lambda_ic
            lambda_steady_state_val = getattr(loss_fn, "lambda_steady_state", 0.0)

            total_loss_val = (
                lambda_data_val * data_loss_val
                + lambda_physics_val * physics_loss_val
                + lambda_ic_val * ic_loss_val
                + lambda_steady_state_val * steady_state_loss_val
                + lambda_angle * angle_penalty
            )

            batch_total_loss += total_loss_val
            batch_data_loss += data_loss_val
            batch_physics_loss += physics_loss_val
            batch_ic_loss += ic_loss_val
            batch_count += 1

        # End scenario loop

        if batch_count == 0:
            continue

        # Average loss over batch
        avg_batch_loss = batch_total_loss / batch_count

        # Backward pass
        avg_batch_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Record batch metrics
        total_loss += avg_batch_loss.item()
        total_data_loss += (batch_data_loss / batch_count).item()
        total_physics_loss += (batch_physics_loss / batch_count).item()
        total_ic_loss += (batch_ic_loss / batch_count).item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    metrics = {
        "loss": avg_loss,
        "data_loss": total_data_loss / max(n_batches, 1),
        "physics_loss": total_physics_loss / max(n_batches, 1),
        "ic_loss": total_ic_loss / max(n_batches, 1),
    }

    return avg_loss, metrics


def _validate_epoch(model, val_data, loss_fn, scalers, physics_normalizer, batch_size, device):
    """Validate for one epoch with full normalization procedures."""
    model.eval()
    total_loss = 0.0
    total_data_loss = 0.0
    total_physics_loss = 0.0
    total_ic_loss = 0.0
    n_batches = 0

    # Note: We need gradients for derivative computation in physics loss,
    # but we don't want to backpropagate. So we use torch.enable_grad()
    # but don't call backward().
    with torch.enable_grad():
        scenarios = val_data["scenario_id"].unique()
        max_scenarios = min(50, len(scenarios))
        n_colloc_per_scenario = 200

        for scenario_id in scenarios[:max_scenarios]:
            scenario_data = val_data[val_data["scenario_id"] == scenario_id].copy()

            if len(scenario_data) < 10:
                continue

            scenario_data = scenario_data.sort_values("time")

            # Extract and normalize scenario data
            normalized_data = _extract_and_normalize_scenario_data(scenario_data, scalers, device)
            t_data = normalized_data["t_data"]
            delta_obs = normalized_data["delta_obs"]
            omega_obs = normalized_data["omega_obs"]
            delta0 = normalized_data["delta0"]
            omega0 = normalized_data["omega0"]
            delta0_for_ic = normalized_data["delta0_for_ic"]
            omega0_for_ic = normalized_data["omega0_for_ic"]
            H = normalized_data["H"]
            D = normalized_data["D"]
            Pm = normalized_data["Pm"]  # For physics loss
            # Extract alpha for Pe-based model (unified approach)
            # alpha is always in normalized_data (created in _extract_and_normalize_scenario_data)
            alpha = normalized_data["alpha"]
            Pload = normalized_data.get(
                "Pload", Pm
            )  # For model input (fallback to Pm if not available)
            Xprefault = normalized_data.get("Xprefault", None)
            Xfault = normalized_data.get("Xfault", None)
            Xpostfault = normalized_data.get("Xpostfault", None)
            tf = normalized_data["tf"]
            tc = normalized_data["tc"]

            # Extract Pe if available (for Pe-based model)
            Pe = normalized_data.get("Pe", None)

            # Predict at data points - handle both model types
            if isinstance(model, TrajectoryPredictionPINN_PeInput):
                # Pe-based model requires Pe(t) input
                if Pe is None:
                    continue  # Skip if Pe not available
                delta_pred, omega_pred = model.predict_trajectory(
                    t=t_data,
                    delta0=delta0,
                    omega0=omega0,
                    H=H,
                    D=D,
                    alpha=alpha,  # Use alpha (unified approach) instead of Pload
                    Pe=Pe,
                    tf=normalized_data["tf"],
                    tc=normalized_data["tc"],
                )
            else:
                # Reactance-based model - provide defaults if reactance values are missing
                if Xprefault is None:
                    if "Xprefault" in scalers:
                        Xprefault = torch.tensor(
                            [normalize_value(0.5, scalers["Xprefault"])],
                            dtype=torch.float32,
                            device=device,
                        )
                    else:
                        Xprefault = torch.tensor([0.5], dtype=torch.float32, device=device)
                if Xfault is None:
                    if "Xfault" in scalers:
                        Xfault = torch.tensor(
                            [normalize_value(0.0001, scalers["Xfault"])],
                            dtype=torch.float32,
                            device=device,
                        )
                    else:
                        Xfault = torch.tensor([0.0001], dtype=torch.float32, device=device)
                if Xpostfault is None:
                    if "Xpostfault" in scalers:
                        Xpostfault = torch.tensor(
                            [normalize_value(0.5, scalers["Xpostfault"])],
                            dtype=torch.float32,
                            device=device,
                        )
                    else:
                        Xpostfault = torch.tensor([0.5], dtype=torch.float32, device=device)
                delta_pred, omega_pred = model.predict_trajectory(
                    t=t_data,
                    delta0=delta0,
                    omega0=omega0,
                    H=H,
                    D=D,
                    Pm=Pm,
                    Xprefault=Xprefault,
                    Xfault=Xfault,
                    Xpostfault=Xpostfault,
                    tf=tf,
                    tc=tc,
                )

            # Generate collocation points
            t_data_raw = denormalize_array(t_data.cpu().numpy(), scalers["time"])
            tc_raw = denormalize_value(tc.item(), scalers["tc"])
            t_colloc_np_raw = generate_collocation_batch(
                t_data_raw,
                n_colloc=n_colloc_per_scenario,
                strategy="uniform",
                fault_clearing_time=tc_raw,
                seed=42,
            )
            t_colloc_np = normalize_array(t_colloc_np_raw, scalers["time"])
            t_colloc = torch.tensor(
                t_colloc_np, dtype=torch.float32, device=device, requires_grad=True
            )

            # Predict at collocation points - handle both model types
            if isinstance(model, TrajectoryPredictionPINN_PeInput):
                # For Pe-based model, interpolate Pe at collocation points
                if Pe is None:
                    continue  # Skip if Pe not available

                # Interpolate Pe at collocation points
                t_colloc_raw = denormalize_array(t_colloc_np, scalers["time"])
                Pe_raw = denormalize_array(Pe.cpu().numpy(), scalers["Pe"])

                # Interpolate Pe at collocation points
                from scipy.interpolate import interp1d

                pe_interp = interp1d(
                    t_data_raw.flatten(),
                    Pe_raw.flatten(),
                    kind="linear",
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                Pe_colloc_raw = pe_interp(t_colloc_raw.flatten())
                Pe_colloc = torch.tensor(
                    normalize_array(Pe_colloc_raw, scalers["Pe"]),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=True,
                )

                delta_colloc, omega_colloc = model.predict_trajectory(
                    t=t_colloc,
                    delta0=delta0,
                    omega0=omega0,
                    H=H,
                    D=D,
                    alpha=alpha,  # Use alpha (unified approach) instead of Pload
                    Pe=Pe_colloc,
                    tf=normalized_data["tf"],
                    tc=normalized_data["tc"],
                )
            else:
                # Reactance-based model
                delta_colloc, omega_colloc = model.predict_trajectory(
                    t=t_colloc,
                    delta0=delta0,
                    omega0=omega0,
                    H=H,
                    D=D,
                    Pm=Pm,
                    Xprefault=Xprefault,
                    Xfault=Xfault,
                    Xpostfault=Xpostfault,
                    tf=tf,
                    tc=tc,
                )

            # Prepare physics loss inputs
            if isinstance(model, TrajectoryPredictionPINN_PeInput):
                # For Pe-based model, prepare physics inputs without reactances
                physics_inputs = {
                    "delta_colloc_phys": denormalize_tensor(
                        delta_colloc, scalers["delta"], device=str(device)
                    ),
                    "omega_colloc_phys": denormalize_tensor(
                        omega_colloc, scalers["omega"], device=str(device)
                    ),
                    "t_colloc_phys": denormalize_tensor(
                        t_colloc, scalers["time"], device=str(device)
                    ),
                    "M_phys": denormalize_tensor(H * 2.0, scalers["H"], device=str(device))
                    * 2.0,  # M = 2*H
                    "D_phys": denormalize_tensor(D, scalers["D"], device=str(device)),
                    "Pm_phys": denormalize_tensor(Pm, scalers["Pm"], device=str(device)),
                }
                # Compute derivative scaling factors
                std_time = torch.tensor(
                    scalers["time"].scale_[0], dtype=torch.float32, device=device
                )
                std_delta = torch.tensor(
                    scalers["delta"].scale_[0], dtype=torch.float32, device=device
                )
                physics_inputs["time_scale"] = std_delta / std_time
                physics_inputs["time_scale_sq"] = physics_inputs["time_scale"] / std_time
            else:
                # Reactance-based model - use existing function
                physics_inputs = _prepare_physics_loss_inputs(
                    delta_colloc,
                    omega_colloc,
                    t_colloc,
                    H,
                    D,
                    Pm,
                    Xprefault,
                    Xfault,
                    Xpostfault,
                    tf,
                    tc,
                    scalers,
                    device,
                )

            # Compute loss
            loss_kwargs = {
                "t_data": t_data,
                "delta_pred": delta_pred,
                "omega_pred": omega_pred,
                "delta_obs": delta_obs,
                "omega_obs": omega_obs,
                "t_colloc": t_colloc,
                "delta_colloc": delta_colloc,
                "omega_colloc": omega_colloc,
                "t_ic": t_data[0:1],
                "delta0": delta0_for_ic,
                "omega0": omega0_for_ic,
                "M": physics_inputs["M_phys"],
                "D": physics_inputs["D_phys"],
                "Pm": physics_inputs["Pm_phys"],
                "delta_colloc_phys": physics_inputs["delta_colloc_phys"],
                "omega_colloc_phys": physics_inputs["omega_colloc_phys"],
                "t_colloc_phys": physics_inputs["t_colloc_phys"],
                "time_scale": physics_inputs["time_scale"],
                "time_scale_sq": physics_inputs["time_scale_sq"],
            }

            # Add Pe or reactance parameters based on model type
            if isinstance(model, TrajectoryPredictionPINN_PeInput):
                # For Pe-based model, pass Pe directly
                Pe_colloc_phys = denormalize_tensor(Pe_colloc, scalers["Pe"], device=str(device))
                loss_kwargs["Pe_from_andes"] = Pe_colloc_phys
                loss_kwargs["use_pe_direct"] = True
                loss_kwargs["tf_norm"] = tf  # Normalized fault start (for steady-state loss)
            else:
                # Reactance-based model
                if "Xprefault_phys" in physics_inputs:
                    loss_kwargs["Xprefault"] = physics_inputs["Xprefault_phys"]
                if "Xfault_phys" in physics_inputs:
                    loss_kwargs["Xfault"] = physics_inputs["Xfault_phys"]
                if "Xpostfault_phys" in physics_inputs:
                    loss_kwargs["Xpostfault"] = physics_inputs["Xpostfault_phys"]
                loss_kwargs["tf"] = physics_inputs["tf_phys"]
                loss_kwargs["tc"] = physics_inputs["tc_phys"]
                loss_kwargs["tf_norm"] = tf  # Normalized fault start (for steady-state loss)

            losses = loss_fn(**loss_kwargs)

            loss = losses.get("total", losses.get("data", torch.tensor(0.0, device=device)))
            total_loss += loss.item()
            total_data_loss += losses.get("data", torch.tensor(0.0, device=device)).item()
            total_physics_loss += losses.get("physics", torch.tensor(0.0, device=device)).item()
            total_ic_loss += losses.get("ic", torch.tensor(0.0, device=device)).item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    metrics = {
        "loss": avg_loss,
        "data_loss": total_data_loss / max(n_batches, 1),
        "physics_loss": total_physics_loss / max(n_batches, 1),
        "ic_loss": total_ic_loss / max(n_batches, 1),
    }

    return avg_loss, metrics


def _save_checkpoint(
    model,
    optimizer,
    scalers,
    epoch,
    val_loss,
    best_val_loss,
    training_history,
    path,
    input_method=None,
    model_config=None,
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "training_history": training_history,
        "scalers": scalers,  # Note: This may not serialize well, may need to save separately
    }
    # CRITICAL FIX: Save input_method and model_config for proper model loading during evaluation
    if input_method is not None:
        checkpoint["input_method"] = input_method
    if model_config is not None:
        checkpoint["model_config"] = model_config
    torch.save(checkpoint, path)
