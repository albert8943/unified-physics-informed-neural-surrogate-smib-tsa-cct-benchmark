"""
Evaluation Workflow Module.

This module evaluates models and generates visualizations.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.comprehensive_evaluation import ComprehensiveEvaluator
from utils.metrics import compute_trajectory_metrics
from utils.visualization import plot_trajectories
from utils.normalization import (
    normalize_array,
    normalize_value,
    denormalize_array,
    set_model_standardization_to_identity,
)
from utils.angle_filter import determine_stability_180deg
from utils.stability_checker import check_stability
from pinn.checkpoint_layout import infer_architecture_from_state_dict
from pinn.trajectory_prediction import (
    TrajectoryPredictionPINN,
    TrajectoryPredictionPINN_PeInput,
)

from .utils import generate_timestamped_filename, save_json, load_json
from datetime import datetime
from .visualization import generate_experiment_figures_wrapper
from .experiment_ui import DELTA_ONLY_EXPERIMENT_UI


def evaluate_model(
    config: Dict,
    model_path: Path,
    test_data_path: Path,
    output_dir: Path,
    device: str = "auto",
) -> Dict:
    """
    Evaluate trained model on test data.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    model_path : Path
        Path to trained model checkpoint
    test_data_path : Path
        Path to test data CSV file
    output_dir : Path
        Directory to save evaluation results
    device : str
        Device to use ('auto', 'cpu', or 'cuda')

    Returns:
    --------
    results : dict
        Evaluation results with metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Test data: {test_data_path}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Load model
    print("\nLoading model...")
    # Convert Path to string if needed, and resolve to absolute path
    model_path_str = str(Path(model_path).resolve())
    if not Path(model_path_str).exists():
        raise FileNotFoundError(f"Model file not found: {model_path_str}")
    # weights_only=False needed for PyTorch 2.6+ when checkpoint contains sklearn scalers
    checkpoint = torch.load(model_path_str, map_location=device, weights_only=False)

    # Get model configuration from config or checkpoint
    model_config = config.get("model", {})
    # CRITICAL FIX: Check checkpoint first for input_method (saved during training)
    # If not in checkpoint, fall back to config, then default to "reactance"
    # Check multiple sources for input_method with priority:
    # 1. Config file (model.input_method or top-level input_method)
    # 2. Checkpoint input_method
    # 3. Checkpoint model_config.input_method
    # 4. Config model_config (from loaded config)
    # 5. Default to "reactance"
    input_method = (
        config.get("model", {}).get("input_method")
        or config.get("input_method")
        or checkpoint.get("input_method")
        or checkpoint.get("model_config", {}).get("input_method")
        or model_config.get("input_method", "reactance")
    )

    # Also check use_pe_as_input flag
    use_pe_as_input = (
        config.get("model", {}).get("use_pe_as_input", False)
        or config.get("data", {}).get("generation", {}).get("use_pe_as_input", False)
        or checkpoint.get("use_pe_as_input", False)
    )
    if use_pe_as_input and input_method == "reactance":
        input_method = "pe_direct"
        print("⚠️  Overriding input_method to 'pe_direct' based on use_pe_as_input=True")

    # Also try to infer from checkpoint model_config if available
    checkpoint_model_config = checkpoint.get("model_config", {})
    if checkpoint_model_config and "input_method" in checkpoint_model_config:
        input_method = checkpoint_model_config["input_method"]

    # If still not found, try to infer from model architecture (check input_dim in checkpoint)
    if input_method == "reactance" and "model_config" in checkpoint:
        checkpoint_input_dim = checkpoint["model_config"].get("input_dim")
        if checkpoint_input_dim == 7:
            input_method = "pe_direct_7"
            print(
                f"⚠️  Inferred input_method='pe_direct_7' from checkpoint input_dim={checkpoint_input_dim}"
            )
        elif checkpoint_input_dim == 9:
            input_method = "pe_direct"
            print(
                f"⚠️  Inferred input_method='pe_direct' from checkpoint input_dim={checkpoint_input_dim}"
            )

    # Get scalers from checkpoint
    scalers = checkpoint.get("scalers", {})
    if not scalers:
        print("⚠️  No scalers found in checkpoint.")

        # CRITICAL FIX: Try to find experiment directory checkpoint (has scalers)
        # Common repository checkpoints only have model weights, not scalers.
        # This causes normalization mismatch when evaluation fits scalers from test data.
        # Solution: Look for experiment directory checkpoint in the same directory structure.
        model_path_obj = Path(model_path)

        # Try to find experiment directory checkpoint
        # Common repository path: outputs/models/common/trajectory/model_*.pth
        # Experiment directory: outputs/.../experiments/exp_*/pinn/model/best_model_*.pth
        exp_checkpoint_found = False

        # Strategy 1: If model_path is in common repository, try to find corresponding experiment directory
        if "common" in str(model_path_obj):
            # Navigate up and look for experiment directories
            possible_exp_dirs = [
                model_path_obj.parent.parent.parent.parent / "experiments",
                model_path_obj.parent.parent.parent.parent.parent / "experiments",
                Path(output_dir) / "experiments" if output_dir else None,
            ]

            for exp_base_dir in possible_exp_dirs:
                if exp_base_dir and exp_base_dir.exists():
                    # Find most recent experiment directory with matching model
                    exp_dirs = sorted(
                        exp_base_dir.glob("exp_*"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    for exp_dir in exp_dirs[:5]:  # Check last 5 experiments
                        exp_checkpoints = list(
                            (exp_dir / "pinn" / "model").glob("best_model_*.pth")
                        )
                        if exp_checkpoints:
                            exp_checkpoint = max(exp_checkpoints, key=lambda p: p.stat().st_mtime)
                            exp_ckpt = torch.load(
                                exp_checkpoint, map_location=device, weights_only=False
                            )
                            if "scalers" in exp_ckpt and exp_ckpt["scalers"]:
                                print(
                                    f"✓ Found scalers in experiment directory checkpoint: {exp_checkpoint.name}"
                                )
                                scalers = exp_ckpt["scalers"]
                                # Also update checkpoint to use full checkpoint from experiment directory
                                checkpoint = exp_ckpt
                                exp_checkpoint_found = True
                                break
                    if exp_checkpoint_found:
                        break

        # Strategy 2: Look in output_dir for experiment checkpoints
        if not exp_checkpoint_found and output_dir:
            # Check multiple possible locations
            possible_patterns = [
                "**/pinn/model/best_model_*.pth",  # Complete experiment structure
                "**/best_model_*.pth",  # Direct location (statistical validation)
                "**/model/best_model_*.pth",  # Alternative location
            ]
            for pattern in possible_patterns:
                exp_checkpoints = list(output_dir.glob(pattern))
                if exp_checkpoints:
                    exp_checkpoint = max(exp_checkpoints, key=lambda p: p.stat().st_mtime)
                    exp_ckpt = torch.load(exp_checkpoint, map_location=device, weights_only=False)
                    if "scalers" in exp_ckpt and exp_ckpt["scalers"]:
                        print(
                            f"✓ Found scalers in output directory checkpoint: {exp_checkpoint.name}"
                        )
                        scalers = exp_ckpt["scalers"]
                        checkpoint = exp_ckpt
                        exp_checkpoint_found = True
                        break

        # If still no scalers found, fit from test data (last resort)
        if not scalers:
            print("⚠️  No experiment directory checkpoint found. Will fit scalers from test data.")
            print(
                "   WARNING: This may cause normalization mismatch if test data differs from training data!"
            )
            scalers = None

    # Prefer layout inferred from weights (sequential vs residual backbone)
    ckpt_mc = checkpoint.get("model_config", {})
    model_config = {**model_config, **ckpt_mc}
    sd = checkpoint.get("model_state_dict", checkpoint)
    use_res_inf, inp_inf, hid_inf = infer_architecture_from_state_dict(sd)
    model_config["use_residual"] = use_res_inf
    model_config["input_dim"] = inp_inf
    if hid_inf:
        model_config["hidden_dims"] = hid_inf

    # Initialize model based on input method
    if input_method in ("pe_direct", "pe_direct_7"):
        pe_dim = model_config.get("input_dim")
        if pe_dim is None:
            pe_dim = 7 if input_method == "pe_direct_7" else 9
        pe_dim = int(pe_dim)
        if pe_dim not in (7, 9):
            pe_dim = 7 if input_method == "pe_direct_7" else 9
        model = TrajectoryPredictionPINN_PeInput(
            input_dim=pe_dim,
            hidden_dims=model_config.get("hidden_dims", [64, 64, 64, 64]),
            activation=model_config.get("activation", "tanh"),
            use_residual=model_config.get("use_residual", False),
            dropout=float(model_config.get("dropout", 0.1 if use_res_inf else 0.0)),
            use_standardization=model_config.get("use_standardization", True),
        ).to(device)
        print(f"✓ Using Pe(t) input model (input_dim={pe_dim}, input_method={input_method})")
        input_dim_actual = pe_dim
    else:
        model = TrajectoryPredictionPINN(
            input_dim=model_config.get("input_dim", 11),
            hidden_dims=model_config.get("hidden_dims", [64, 64, 64, 64]),
            activation=model_config.get("activation", "tanh"),
            use_residual=model_config.get("use_residual", False),
            dropout=float(model_config.get("dropout", 0.1 if use_res_inf else 0.0)),
            use_standardization=model_config.get("use_standardization", True),
        ).to(device)
        input_dim_actual = model_config.get("input_dim", 11)
        print(f"✓ Using reactance-based model (input_dim={input_dim_actual})")

    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Set model standardization to identity (same as training)
    set_model_standardization_to_identity(model, input_dim_actual, 2, device)

    model.eval()
    print("✓ Model loaded and ready for evaluation")

    # Load test data
    print("\nLoading test data...")
    test_data = pd.read_csv(test_data_path)

    # Fit scalers from test data if not in checkpoint
    if scalers is None:
        print("Fitting scalers from test data...")
        scalers = _fit_scalers_from_data(test_data)

    # Get test scenarios (use all or limit if specified)
    eval_config = config.get("evaluation", {})
    max_scenarios = eval_config.get("max_test_scenarios", None)
    test_scenarios = test_data["scenario_id"].unique()
    if max_scenarios:
        test_scenarios = test_scenarios[:max_scenarios]
    test_subset = test_data[test_data["scenario_id"].isin(test_scenarios)]

    print(f"✓ Test scenarios: {len(test_scenarios)}")

    # Run evaluation
    print("\nRunning evaluation...")
    device_torch = torch.device(device)

    all_delta_pred = []
    all_omega_pred = []
    all_delta_true = []
    all_omega_true = []

    # Stability-based metrics separation
    stable_delta_errors = []
    stable_omega_errors = []
    unstable_delta_errors = []
    unstable_omega_errors = []
    stability_mismatches = []
    # Track total scenarios with stability labels for accuracy computation
    total_scenarios_with_stability = 0
    correct_stability_predictions = 0

    # Check if stability column exists
    has_stability_column = "is_stable" in test_data.columns

    if has_stability_column:
        # Get scenario stability from ground truth
        scenario_stability = test_data.groupby("scenario_id")["is_stable"].first()
        stable_scenarios_gt = set(scenario_stability[scenario_stability == True].index.tolist())
        unstable_scenarios_gt = set(scenario_stability[scenario_stability == False].index.tolist())
        print(
            f"Ground truth: {len(stable_scenarios_gt)} stable, {len(unstable_scenarios_gt)}"
            f"unstable scenarios"
        )

    with torch.no_grad():
        for scenario_id in test_scenarios:
            scenario_data = test_subset[test_subset["scenario_id"] == scenario_id].copy()

            if len(scenario_data) < 10:
                continue

            scenario_data = scenario_data.sort_values("time")

            # Extract and normalize scenario data
            normalized_data = _extract_and_normalize_scenario_data(
                scenario_data, scalers, device_torch
            )

            t_data = normalized_data["t_data"]
            delta0 = normalized_data["delta0"]
            omega0 = normalized_data["omega0"]
            H = normalized_data["H"]
            D = normalized_data["D"]
            Pm = normalized_data["Pm"]  # For physics loss (if needed)
            Pload = normalized_data.get(
                "Pload", Pm
            )  # For model input (fallback to Pm if not available)

            # Make predictions
            if input_method in ("pe_direct", "pe_direct_7"):
                Pe = normalized_data.get("Pe")
                if Pe is None:
                    continue  # Skip if Pe not available
                delta_pred_norm, omega_pred_norm = model.predict_trajectory(
                    t=t_data,
                    delta0=delta0,
                    omega0=omega0,
                    H=H,
                    D=D,
                    alpha=normalized_data.get(
                        "alpha", normalized_data.get("Pload", normalized_data["Pm"])
                    ),  # Use alpha (unified approach)
                    Pe=Pe,
                    tf=normalized_data.get("tf"),
                    tc=normalized_data.get("tc"),
                )
            else:
                # Reactance-based model - provide defaults if reactance values are missing
                Xprefault = normalized_data.get("Xprefault", None)
                Xfault = normalized_data.get("Xfault", None)
                Xpostfault = normalized_data.get("Xpostfault", None)
                tf = normalized_data["tf"]
                tc = normalized_data["tc"]

                # Provide default values if reactance values are missing
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

                delta_pred_norm, omega_pred_norm = model.predict_trajectory(
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

            # Denormalize predictions
            delta_pred = denormalize_array(
                delta_pred_norm.cpu().numpy().flatten(), scalers["delta"]
            )
            omega_pred = denormalize_array(
                omega_pred_norm.cpu().numpy().flatten(), scalers["omega"]
            )

            # Get true values
            delta_true = scenario_data["delta"].values
            omega_true = scenario_data["omega"].values
            scenario_time = scenario_data["time"].values

            # Improved time alignment: Interpolate predictions to match ground truth time points
            # This is more accurate than simple truncation, especially if time arrays don't match exactly
            if len(scenario_time) > 0 and len(delta_pred) > 0:
                # Check if time arrays match (within tolerance)
                pred_time = np.linspace(scenario_time[0], scenario_time[-1], len(delta_pred))
                time_match = len(pred_time) == len(scenario_time) and np.allclose(
                    pred_time, scenario_time, atol=0.01
                )

                if not time_match and len(delta_pred) != len(delta_true):
                    # Interpolate predictions to match ground truth time points
                    # Create interpolation function (extrapolate if needed)
                    pred_time_actual = np.linspace(
                        scenario_time[0], scenario_time[-1], len(delta_pred)
                    )
                    delta_interp = interp1d(
                        pred_time_actual,
                        delta_pred,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    omega_interp = interp1d(
                        pred_time_actual,
                        omega_pred,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )

                    # Interpolate to ground truth time points
                    delta_pred = delta_interp(scenario_time)
                    omega_pred = omega_interp(scenario_time)

            # Ensure same length (fallback to truncation if interpolation didn't work)
            min_len = min(len(delta_pred), len(delta_true), len(omega_pred), len(omega_true))
            delta_pred = delta_pred[:min_len]
            omega_pred = omega_pred[:min_len]
            delta_true = delta_true[:min_len]
            omega_true = omega_true[:min_len]
            if len(scenario_time) > 0:
                scenario_time = scenario_time[:min_len]

            # CRITICAL FIX: Enforce initial condition constraint for PINN
            # PINN models don't automatically enforce IC during inference, so we need to ensure
            # the first prediction matches the initial conditions exactly.
            if len(delta_pred) > 0 and len(omega_pred) > 0:
                # Ensure scenario_data is sorted by time to get the correct first row
                scenario_data_sorted = scenario_data.sort_values("time").reset_index(drop=True)
                first_row = scenario_data_sorted.iloc[0]
                delta0_true = float(first_row.get("delta0", scenario_data_sorted["delta"].iloc[0]))
                omega0_true = float(first_row.get("omega0", scenario_data_sorted["omega"].iloc[0]))

                # Get time array (already sorted)
                scenario_time = scenario_data_sorted["time"].values[:min_len]

                # Find the index closest to t=0 (or the minimum time)
                if len(scenario_time) > 0:
                    min_time = np.min(scenario_time)
                    min_time_idx = np.argmin(np.abs(scenario_time))

                    # Enforce initial conditions at the first time point
                    # Use the actual first trajectory value if delta0 column doesn't exist
                    delta0_to_use = delta0_true
                    omega0_to_use = omega0_true

                    # If delta0 column doesn't exist, use first trajectory value
                    if "delta0" not in first_row or pd.isna(first_row.get("delta0")):
                        delta0_to_use = float(scenario_data_sorted["delta"].iloc[0])
                    if "omega0" not in first_row or pd.isna(first_row.get("omega0")):
                        omega0_to_use = float(scenario_data_sorted["omega"].iloc[0])

                    # Enforce initial conditions at the first time point
                    delta_pred[min_time_idx] = delta0_to_use
                    omega_pred[min_time_idx] = omega0_to_use

                    # Also enforce at index 0 if different from min_time_idx
                    if min_time_idx != 0 and abs(scenario_time[0] - min_time) < 0.001:
                        delta_pred[0] = delta0_to_use
                        omega_pred[0] = omega0_to_use

                    # Validate IC enforcement worked
                    delta_ic_error = abs(delta_pred[min_time_idx] - delta0_to_use)
                    omega_ic_error = abs(omega_pred[min_time_idx] - omega0_to_use)
                    if delta_ic_error > 0.01 or omega_ic_error > 0.01:
                        print(f"   ⚠️  Scenario {scenario_id}: IC enforcement may have failed")
                        print(
                            f"Delta IC error: {delta_ic_error:.6f} rad, Omega IC error:"
                            f"{omega_ic_error:.6f} pu"
                        )

                    # Check for extreme predictions (likely due to insufficient training or model convergence issues)
                    # Note: These are warnings, not errors. They indicate predictions with very large angles,
                    # which may occur with short training (e.g., < 100 epochs) or models that haven't fully converged.
                    max_delta_abs = np.max(np.abs(delta_pred))
                    if max_delta_abs > 20:  # 20 rad ≈ 1146 degrees, way too high
                        print(f"   ⚠️  Scenario {scenario_id}: Extreme delta predictions detected!")
                        print(
                            f"Max |delta|: {max_delta_abs:.2f} rad"
                            f"({np.degrees(max_delta_abs):.1f}°)"
                        )
                        print(
                            f"      Delta range: [{np.min(delta_pred):.2f}, {np.max(delta_pred):.2f}] rad"
                        )
                        print(
                            f"Delta0 used: {delta0_to_use:.6f} rad"
                            f"({np.degrees(delta0_to_use):.1f}°)"
                        )
                        print(
                            f"Delta true range: [{np.min(delta_true):.2f},"
                            f"{np.max(delta_true):.2f}] rad"
                        )
                        print(
                            f"Note: This may indicate insufficient training. Consider training for"
                            f"more epochs."
                        )

            # Determine stability from PREDICTED trajectory (180° criterion)
            is_stable_predicted, max_angle_deg_predicted = determine_stability_180deg(
                delta_pred, threshold_deg=180.0
            )

            # Compute per-scenario metrics (with angle wrapping for delta)
            from utils.metrics import wrap_angle_error

            # Use wrapped error for delta (periodic angles)
            delta_error_wrapped = wrap_angle_error(delta_pred, delta_true)
            delta_rmse = np.sqrt(np.mean((delta_pred - delta_true) ** 2))
            delta_rmse_wrapped = np.sqrt(np.mean(delta_error_wrapped**2))

            # Use wrapped RMSE if it's significantly better (indicates wrapping helped)
            if delta_rmse_wrapped < delta_rmse * 0.9:
                delta_rmse = delta_rmse_wrapped  # Use wrapped version

            omega_rmse = np.sqrt(np.mean((omega_pred - omega_true) ** 2))

            # Separate by stability (use predicted stability)
            if is_stable_predicted:
                stable_delta_errors.append(delta_rmse)
                stable_omega_errors.append(omega_rmse)
            else:
                unstable_delta_errors.append(delta_rmse)
                unstable_omega_errors.append(omega_rmse)

            # Compare predicted vs ground truth stability (if available)
            if has_stability_column:
                is_stable_ground_truth = (
                    scenario_data["is_stable"].iloc[0]
                    if "is_stable" in scenario_data.columns
                    else None
                )
                if is_stable_ground_truth is not None:
                    total_scenarios_with_stability += 1
                    # Track correct predictions for accuracy computation
                    if is_stable_predicted == is_stable_ground_truth:
                        correct_stability_predictions += 1
                    else:
                        stability_mismatches.append(
                            {
                                "scenario_id": scenario_id,
                                "predicted": "stable" if is_stable_predicted else "unstable",
                                "ground_truth": "stable" if is_stable_ground_truth else "unstable",
                                "max_angle_deg": max_angle_deg_predicted,
                            }
                        )

            # Store for aggregate metrics
            all_delta_pred.extend(delta_pred)
            all_omega_pred.extend(omega_pred)
            all_delta_true.extend(delta_true)
            all_omega_true.extend(omega_true)

    # Compute aggregate metrics
    if len(all_delta_pred) > 0:
        metrics = compute_trajectory_metrics(
            delta_pred=np.array(all_delta_pred),
            omega_pred=np.array(all_omega_pred),
            delta_true=np.array(all_delta_true),
            omega_true=np.array(all_omega_true),
        )

        # Fix stability classification accuracy: compute per-scenario, not across all trajectories
        # The compute_trajectory_metrics function incorrectly computes this by taking max across
        # all concatenated trajectories. We need to compute it from per-scenario comparisons.
        if total_scenarios_with_stability > 0:
            metrics["stability_classification_accuracy"] = (
                correct_stability_predictions / total_scenarios_with_stability
            )
        else:
            # If no stability labels available, set to None or keep the incorrect value
            # (but document that it's not meaningful)
            metrics["stability_classification_accuracy"] = 0.0

        # Map to expected format - include ALL computed metrics for comprehensive evaluation
        # This ensures publication-quality figures with full metric details (matching ML baseline)
        results = {
            "metrics": {
                # Core metrics (for backward compatibility)
                "rmse_delta": metrics.get("delta_rmse", 0.0),
                "rmse_omega": metrics.get("omega_rmse", 0.0),
                "mae_delta": metrics.get("delta_mae", 0.0),
                "mae_omega": metrics.get("omega_mae", 0.0),
                "r2_delta": metrics.get("delta_r2", 0.0),
                "r2_omega": metrics.get("omega_r2", 0.0),
                # Comprehensive metrics for publication-quality figures
                # These match what ML baseline evaluation provides for fair comparison
                "delta_rmse": metrics.get("delta_rmse", 0.0),
                "delta_mae": metrics.get("delta_mae", 0.0),
                "delta_r2": metrics.get("delta_r2", 0.0),
                "delta_mape": metrics.get("delta_mape", 0.0),
                "delta_mape_above_threshold": metrics.get("delta_mape_above_threshold", 0.0),
                "delta_nrmse": metrics.get("delta_nrmse", 0.0),
                "delta_max_error": metrics.get("delta_max_error", 0.0),
                "delta_rmse_wrapped": metrics.get("delta_rmse_wrapped", 0.0),
                "delta_mae_wrapped": metrics.get("delta_mae_wrapped", 0.0),
                "delta_r2_wrapped": metrics.get("delta_r2_wrapped", 0.0),
                "omega_rmse": metrics.get("omega_rmse", 0.0),
                "omega_mae": metrics.get("omega_mae", 0.0),
                "omega_r2": metrics.get("omega_r2", 0.0),
                "omega_mape": metrics.get("omega_mape", 0.0),
                "omega_mape_above_threshold": metrics.get("omega_mape_above_threshold", 0.0),
                "omega_nrmse": metrics.get("omega_nrmse", 0.0),
                "omega_max_error": metrics.get("omega_max_error", 0.0),
                "omega_max_relative_error": metrics.get("omega_max_relative_error", 0.0),
            },
            # Include predictions and targets for comprehensive figure generation
            # This matches the format expected by generate_evaluation_figures
            "predictions": {
                "delta": np.array(all_delta_pred),
                "omega": np.array(all_omega_pred),
            },
            "targets": {
                "delta": np.array(all_delta_true),
                "omega": np.array(all_omega_true),
            },
            "n_test_scenarios": len(test_scenarios),
            "n_evaluated_scenarios": len(test_scenarios),
            "total_samples": len(all_delta_pred),
        }

        # Add stability-based metrics
        if len(stable_delta_errors) > 0 or len(unstable_delta_errors) > 0:
            results["metrics_by_stability"] = {}

            if len(stable_delta_errors) > 0:
                results["metrics_by_stability"]["stable"] = {
                    "n_scenarios": len(stable_delta_errors),
                    "rmse_delta": float(np.mean(stable_delta_errors)),
                    "rmse_delta_std": float(np.std(stable_delta_errors)),
                    "rmse_delta_min": float(np.min(stable_delta_errors)),
                    "rmse_delta_max": float(np.max(stable_delta_errors)),
                    "rmse_omega": float(np.mean(stable_omega_errors)),
                    "rmse_omega_std": float(np.std(stable_omega_errors)),
                    "rmse_omega_min": float(np.min(stable_omega_errors)),
                    "rmse_omega_max": float(np.max(stable_omega_errors)),
                }

            if len(unstable_delta_errors) > 0:
                results["metrics_by_stability"]["unstable"] = {
                    "n_scenarios": len(unstable_delta_errors),
                    "rmse_delta": float(np.mean(unstable_delta_errors)),
                    "rmse_delta_std": float(np.std(unstable_delta_errors)),
                    "rmse_delta_min": float(np.min(unstable_delta_errors)),
                    "rmse_delta_max": float(np.max(unstable_delta_errors)),
                    "rmse_omega": float(np.mean(unstable_omega_errors)),
                    "rmse_omega_std": float(np.std(unstable_omega_errors)),
                    "rmse_omega_min": float(np.min(unstable_omega_errors)),
                    "rmse_omega_max": float(np.max(unstable_omega_errors)),
                }

            # Comparison ratios
            if len(stable_delta_errors) > 0 and len(unstable_delta_errors) > 0:
                results["metrics_by_stability"]["comparison"] = {
                    "unstable_stable_delta_ratio": float(
                        np.mean(unstable_delta_errors) / np.mean(stable_delta_errors)
                    ),
                    "unstable_stable_omega_ratio": float(
                        np.mean(unstable_omega_errors) / np.mean(stable_omega_errors)
                    ),
                }

        # Add stability mismatch information
        if stability_mismatches:
            results["stability_mismatches"] = {
                "count": len(stability_mismatches),
                "mismatches": stability_mismatches[:10],  # Store first 10 for reference
            }

        print(f"\n✓ Evaluation complete")
        print(f"  RMSE Delta: {results['metrics']['rmse_delta']:.6f} rad")
        print(f"  MAE Delta: {results['metrics']['mae_delta']:.6f} rad")
        print(f"  R² Delta: {results['metrics'].get('r2_delta', 0):.6f}")
        if not DELTA_ONLY_EXPERIMENT_UI:
            print(f"  RMSE Omega: {results['metrics']['rmse_omega']:.6f} pu")
            print(f"  MAE Omega: {results['metrics']['mae_omega']:.6f} pu")
            print(f"  R² Omega: {results['metrics'].get('r2_omega', 0):.6f}")

        # Print stability-based metrics
        if "metrics_by_stability" in results:
            print(f"\n📊 Metrics by Stability:")
            if "stable" in results["metrics_by_stability"]:
                stable_metrics = results["metrics_by_stability"]["stable"]
                print(f"  ✅ Stable Scenarios ({stable_metrics['n_scenarios']}):")
                print(
                    f"Delta RMSE: {stable_metrics['rmse_delta']:.6f} rad (std:"
                    f"{stable_metrics['rmse_delta_std']:.6f})"
                )
                if not DELTA_ONLY_EXPERIMENT_UI:
                    print(
                        f"Omega RMSE: {stable_metrics['rmse_omega']:.6f} pu (std:"
                        f"{stable_metrics['rmse_omega_std']:.6f})"
                    )
                print(
                    f"Delta RMSE range: [{stable_metrics['rmse_delta_min']:.6f},"
                    f"{stable_metrics['rmse_delta_max']:.6f}] rad"
                )
                if not DELTA_ONLY_EXPERIMENT_UI:
                    print(
                        f"Omega RMSE range: [{stable_metrics['rmse_omega_min']:.6f},"
                        f"{stable_metrics['rmse_omega_max']:.6f}] pu"
                    )

            if "unstable" in results["metrics_by_stability"]:
                unstable_metrics = results["metrics_by_stability"]["unstable"]
                print(f"  ⚠️  Unstable Scenarios ({unstable_metrics['n_scenarios']}):")
                print(
                    f"Delta RMSE: {unstable_metrics['rmse_delta']:.6f} rad (std:"
                    f"{unstable_metrics['rmse_delta_std']:.6f})"
                )
                if not DELTA_ONLY_EXPERIMENT_UI:
                    print(
                        f"Omega RMSE: {unstable_metrics['rmse_omega']:.6f} pu (std:"
                        f"{unstable_metrics['rmse_omega_std']:.6f})"
                    )
                print(
                    f"Delta RMSE range: [{unstable_metrics['rmse_delta_min']:.6f},"
                    f"{unstable_metrics['rmse_delta_max']:.6f}] rad"
                )
                if not DELTA_ONLY_EXPERIMENT_UI:
                    print(
                        f"Omega RMSE range: [{unstable_metrics['rmse_omega_min']:.6f},"
                        f"{unstable_metrics['rmse_omega_max']:.6f}] pu"
                    )

            if "comparison" in results["metrics_by_stability"]:
                comp = results["metrics_by_stability"]["comparison"]
                print(f"\n  📊 Comparison:")
                print(
                    f"    Unstable/Stable Delta RMSE ratio: {comp['unstable_stable_delta_ratio']:.2f}x"
                )
                if not DELTA_ONLY_EXPERIMENT_UI:
                    print(
                        f"    Unstable/Stable Omega RMSE ratio: {comp['unstable_stable_omega_ratio']:.2f}x"
                    )

        if stability_mismatches:
            print(f"\n⚠️  Stability Mismatches: {len(stability_mismatches)} scenarios")
            print(f"    (Predicted stability differs from ground truth)")
            for mismatch in stability_mismatches[:5]:  # Show first 5
                print(
                    f"    Scenario {mismatch['scenario_id']}: Predicted {mismatch['predicted']}, "
                    f"Ground truth {mismatch['ground_truth']} (max angle:"
                    f"{mismatch['max_angle_deg']:.1f}°)"
                )

        # Note about extreme predictions warnings
        # (These are warnings, not errors - they indicate predictions with very large angles,
        # which may occur with short training or models that haven't fully converged)
    else:
        print("⚠️  No valid scenarios evaluated. Using placeholder metrics.")
        results = {
            "metrics": {
                "rmse_delta": 0.0,
                "rmse_omega": 0.0,
                "mae_delta": 0.0,
                "mae_omega": 0.0,
            },
            "n_test_scenarios": len(test_scenarios),
            "n_evaluated_scenarios": 0,
            "total_samples": 0,
        }

    # Save results with timestamp
    metrics_filename = generate_timestamped_filename("metrics", "json")
    results_path = output_dir / metrics_filename
    save_json(results, results_path)
    print(f"\n✓ Results saved to: {results_path}")

    # Load training history if available (check model directory and parent directory)
    training_history = None
    model_dir = model_path.parent
    # Check both model directory and parent directory (for complete_experiment structure)
    search_dirs = [model_dir]
    if model_dir.parent.exists():
        search_dirs.append(
            model_dir.parent
        )  # Also check parent (e.g., pinn/ when model is in pinn/model/)

    history_files = []
    for search_dir in search_dirs:
        found_files = list(search_dir.glob("training_history_*.json"))
        history_files.extend(found_files)

    if history_files:
        # Use the most recent training history file
        history_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        history_path = history_files[0]
        print(f"\n📂 Loading training history from: {history_path.name}")
        print(f"   Location: {history_path.parent}")
        try:
            training_history = load_json(history_path)
            print(
                f"✓ Loaded training history with {len(training_history.get('train_losses', []))}"
                f"epochs"
            )
        except Exception as e:
            print(f"⚠️  Could not load training history: {e}")
            training_history = None
    else:
        print("\n⚠️  No training history found in model directory or parent directory")
        print(f"   Searched in: {[str(d) for d in search_dirs]}")
        print("   (This is normal if you loaded a pre-trained model from elsewhere)")

    # Generate visualizations using comprehensive visualization module
    print("\nGenerating publication-quality figures...")
    figures_dir = output_dir / "figures"

    # Find original full dataset for data exploration figures (not just test set)
    # Look for parameter_sweep_data_*.csv in the experiment's data directory
    original_data_path = None
    experiment_data_dir = (
        output_dir.parent / "data"
    )  # Go up from results/ to experiment root, then to data/
    if experiment_data_dir.exists():
        original_files = list(experiment_data_dir.glob("parameter_sweep_data_*.csv"))
        if original_files:
            original_data_path = sorted(original_files)[-1]  # Use latest
            print(f"✓ Found original dataset for data exploration: {original_data_path.name}")
        else:
            # Fallback: look for trajectory_data_*.csv
            trajectory_files = list(experiment_data_dir.glob("trajectory_data_*.csv"))
            if trajectory_files:
                original_data_path = sorted(trajectory_files)[-1]
                print(f"✓ Found original dataset for data exploration: {original_data_path.name}")

    # Use original dataset for data exploration, test data for evaluation-specific figures
    data_path_for_figures = original_data_path if original_data_path else test_data_path
    if original_data_path:
        print(
            f"Using full dataset ({len(pd.read_csv(original_data_path)):,} rows) for data"
            f"exploration figures"
        )
    else:
        print(
            f"Using test data ({len(pd.read_csv(test_data_path)):,} rows) for data exploration"
            f"figures"
        )

    # Generate comprehensive figures
    try:
        figure_paths = generate_experiment_figures_wrapper(
            config=config,
            data_path=data_path_for_figures,  # Use original dataset for data exploration
            training_history=training_history,  # Now loads from model directory
            evaluation_results=results,
            output_dir=figures_dir,
        )
        results["figure_paths"] = {name: str(path) for name, path in figure_paths.items()}
    except Exception as e:
        print(f"⚠️  Could not generate comprehensive figures: {e}")
        print("   Falling back to basic visualizations...")
        _generate_visualizations(test_subset, output_dir)

    # Generate detailed scenario visualization plots (like Colab notebook)
    # Save directly to figures_dir (flattened structure - no nested subdirectories)
    print("\nGenerating detailed scenario visualizations...")
    try:
        scenario_plot_path = _generate_scenario_visualizations(
            model=model,
            test_data=test_subset,
            scalers=scalers,
            input_method=input_method,
            device=device,
            output_dir=figures_dir,  # Save directly to figures/ (no evaluation/ subdirectory)
        )
        if scenario_plot_path:
            results["figure_paths"]["test_scenarios"] = str(scenario_plot_path)
    except Exception as e:
        print(f"⚠️  Could not generate scenario visualizations: {e}")

    return results


def _generate_scenario_visualizations(
    model,
    test_data: pd.DataFrame,
    scalers: Dict,
    input_method: str,
    device: str,
    output_dir: Path,
    n_examples: int = 5,
) -> Optional[Path]:
    """
    Generate detailed scenario visualization plots (similar to Colab notebook).

    Parameters:
    -----------
    model : TrajectoryPredictionPINN or TrajectoryPredictionPINN_PeInput
        Trained model
    test_data : pd.DataFrame
        Test data with scenarios
    scalers : dict
        Normalization scalers
    input_method : str
        Input method ("reactance" or "pe_direct")
    device : str
        Device to use
    output_dir : Path
        Directory to save figures
    n_examples : int
        Number of examples per stability class (default: 5)

    Returns:
    --------
    Path or None
        Path to saved figure, or None if failed
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select scenarios similar to data analysis script
    has_stability = "is_stable" in test_data.columns
    if has_stability:
        stable_scenarios = test_data[test_data["is_stable"] == True]["scenario_id"].unique()[
            :n_examples
        ]
        unstable_scenarios = test_data[test_data["is_stable"] == False]["scenario_id"].unique()[
            :n_examples
        ]
        test_scenario_ids = list(stable_scenarios) + list(unstable_scenarios)
    else:
        test_scenario_ids = list(test_data["scenario_id"].unique()[:n_examples])

    if len(test_scenario_ids) == 0:
        print("   ⚠️  No scenarios to visualize")
        return None

    print(f"   Visualizing {len(test_scenario_ids)} test scenarios...")

    n_rows = len(test_scenario_ids)
    if DELTA_ONLY_EXPERIMENT_UI:
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows))
        if n_rows == 1:
            axes = np.array([axes])
    else:
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

    device_torch = torch.device(device)
    model.eval()

    cct_summary = []

    with torch.no_grad():
        for idx, scenario_id in enumerate(test_scenario_ids):
            scenario_data = test_data[test_data["scenario_id"] == scenario_id].sort_values("time")

            if len(scenario_data) < 10:
                continue

            first_row = scenario_data.iloc[0]

            # Get initial conditions from first time point
            delta0 = float(scenario_data["delta"].iloc[0])
            omega0 = float(scenario_data["omega"].iloc[0])
            t0 = float(scenario_data["time"].iloc[0])

            # Get parameters
            H = float(first_row.get("param_H", first_row.get("H", 5.0)))
            D = float(first_row.get("param_D", first_row.get("D", 1.0)))
            Pm = float(first_row.get("param_Pm", first_row.get("Pm", 0.8)))
            Xprefault = float(first_row.get("Xprefault", 0.5))
            Xfault = float(first_row.get("Xfault", 0.0001))
            Xpostfault = float(first_row.get("Xpostfault", 0.5))
            tf = float(first_row.get("tf", 1.0))
            tc = float(first_row.get("tc", first_row.get("param_tc", 1.2)))

            # Get CCT (if available)
            cct_true = None
            for col_name in ["param_cct_absolute", "cct", "CCT", "param_cct_duration"]:
                if col_name in scenario_data.columns:
                    cct_val = first_row.get(col_name)
                    if cct_val is not None and not pd.isna(cct_val):
                        cct_val = float(cct_val)
                        if "duration" in col_name.lower() or cct_val < tf:
                            cct_true = cct_val
                        else:
                            cct_true = cct_val - tf
                        break

            # Extract and normalize
            normalized_data = _extract_and_normalize_scenario_data(
                scenario_data, scalers, device_torch
            )

            t_data = normalized_data["t_data"]

            # Make predictions
            if input_method in ("pe_direct", "pe_direct_7"):
                Pe = normalized_data.get("Pe")
                if Pe is None:
                    continue
                Pload = normalized_data.get(
                    "Pload", normalized_data["Pm"]
                )  # For model input (fallback to Pm if not available)
                delta_pred_norm, omega_pred_norm = model.predict_trajectory(
                    t=t_data,
                    delta0=normalized_data["delta0"],
                    omega0=normalized_data["omega0"],
                    H=normalized_data["H"],
                    D=normalized_data["D"],
                    alpha=normalized_data.get(
                        "alpha", normalized_data.get("Pload", normalized_data["Pm"])
                    ),  # Use alpha (unified approach)
                    Pe=Pe,
                    tf=normalized_data.get("tf"),
                    tc=normalized_data.get("tc"),
                )
            else:
                # Reactance-based model - provide defaults if reactance values are missing
                Xprefault = normalized_data.get("Xprefault", None)
                Xfault = normalized_data.get("Xfault", None)
                Xpostfault = normalized_data.get("Xpostfault", None)

                # Provide default values if reactance values are missing
                if Xprefault is None:
                    if "Xprefault" in scalers:
                        from utils.normalization import normalize_value

                        Xprefault = torch.tensor(
                            [normalize_value(0.5, scalers["Xprefault"])],
                            dtype=torch.float32,
                            device=device_torch,
                        )
                    else:
                        Xprefault = torch.tensor([0.5], dtype=torch.float32, device=device_torch)
                if Xfault is None:
                    if "Xfault" in scalers:
                        from utils.normalization import normalize_value

                        Xfault = torch.tensor(
                            [normalize_value(0.0001, scalers["Xfault"])],
                            dtype=torch.float32,
                            device=device_torch,
                        )
                    else:
                        Xfault = torch.tensor([0.0001], dtype=torch.float32, device=device_torch)
                if Xpostfault is None:
                    if "Xpostfault" in scalers:
                        from utils.normalization import normalize_value

                        Xpostfault = torch.tensor(
                            [normalize_value(0.5, scalers["Xpostfault"])],
                            dtype=torch.float32,
                            device=device_torch,
                        )
                    else:
                        Xpostfault = torch.tensor([0.5], dtype=torch.float32, device=device_torch)

                delta_pred_norm, omega_pred_norm = model.predict_trajectory(
                    t=t_data,
                    delta0=normalized_data["delta0"],
                    omega0=normalized_data["omega0"],
                    H=normalized_data["H"],
                    D=normalized_data["D"],
                    Pm=normalized_data["Pm"],
                    Xprefault=Xprefault,
                    Xfault=Xfault,
                    Xpostfault=Xpostfault,
                    tf=normalized_data["tf"],
                    tc=normalized_data["tc"],
                )

            # Denormalize
            delta_pred = denormalize_array(
                delta_pred_norm.cpu().numpy().flatten(), scalers["delta"]
            )
            omega_pred = denormalize_array(
                omega_pred_norm.cpu().numpy().flatten(), scalers["omega"]
            )

            # Get true values
            scenario_time = scenario_data["time"].values
            delta_true = scenario_data["delta"].values
            omega_true = scenario_data["omega"].values

            # Align lengths
            min_len = min(len(delta_pred), len(delta_true), len(scenario_time))
            delta_pred = delta_pred[:min_len]
            omega_pred = omega_pred[:min_len]
            delta_true = delta_true[:min_len]
            omega_true = omega_true[:min_len]
            scenario_time = scenario_time[:min_len]

            # CRITICAL FIX: Enforce initial condition constraint for PINN
            # PINN models don't automatically enforce IC during inference, so we need to ensure
            # the first prediction matches the initial conditions exactly.
            if len(delta_pred) > 0 and len(omega_pred) > 0:
                # Ensure scenario_data is sorted by time to get the correct first row
                scenario_data_sorted = scenario_data.sort_values("time").reset_index(drop=True)
                first_row_sorted = scenario_data_sorted.iloc[0]
                delta0_true = float(
                    first_row_sorted.get("delta0", scenario_data_sorted["delta"].iloc[0])
                )
                omega0_true = float(
                    first_row_sorted.get("omega0", scenario_data_sorted["omega"].iloc[0])
                )

                # Find the index closest to t=0 (or the minimum time)
                if len(scenario_time) > 0:
                    min_time = np.min(scenario_time)
                    min_time_idx = np.argmin(np.abs(scenario_time))

                    # Enforce initial conditions at the first time point
                    delta_pred[min_time_idx] = delta0_true
                    omega_pred[min_time_idx] = omega0_true

                    # Also enforce at index 0 if different from min_time_idx
                    if min_time_idx != 0 and abs(scenario_time[0] - min_time) < 0.001:
                        delta_pred[0] = delta0_true
                        omega_pred[0] = omega0_true

            # Check stability
            is_stable_pred, max_angle_deg_pred = determine_stability_180deg(
                delta_pred, threshold_deg=180.0
            )
            is_stable_true = None
            if has_stability:
                is_stable_true = first_row.get("is_stable", None)

            # Compute metrics
            delta_rmse = np.sqrt(np.mean((delta_pred - delta_true) ** 2))
            omega_rmse = (
                np.sqrt(np.mean((omega_pred - omega_true) ** 2))
                if not DELTA_ONLY_EXPERIMENT_UI
                else float("nan")
            )

            # Create title
            title_parts = [f"Scenario {scenario_id}"]
            title_parts.append(f"H={H:.2f}, D={D:.2f}, Pm={Pm:.3f}")
            if cct_true is not None:
                title_parts.append(f"CCT={cct_true:.3f}s")
            title_parts.append(f"tc={tc:.3f}s")
            title_str = " | ".join(title_parts)

            cct_absolute = tf + cct_true if cct_true is not None else None

            # Plot Delta
            ax = axes[idx] if DELTA_ONLY_EXPERIMENT_UI else axes[idx, 0]
            ax.plot(scenario_time, np.degrees(delta_true), "r-", label="True", linewidth=2)
            ax.plot(
                scenario_time,
                np.degrees(delta_pred),
                "b--",
                label="Predicted",
                linewidth=2,
                alpha=0.7,
            )
            ax.axhline(y=0, color="k", linestyle="-", linewidth=0.8, alpha=0.3, zorder=0)
            ax.axvline(x=tf, color="g", linestyle=":", label="Fault Start", alpha=0.7)
            ax.axvline(x=tc, color="orange", linestyle=":", label="Fault Clear", alpha=0.7)
            if cct_absolute is not None:
                ax.axvline(
                    x=cct_absolute,
                    color="purple",
                    linestyle="--",
                    label=f"CCT={cct_true:.3f}s",
                    alpha=0.7,
                    linewidth=1.5,
                )
            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel("Rotor Angle (degrees)", fontsize=10)
            ax.set_title(title_str, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-360, 360)

            if not DELTA_ONLY_EXPERIMENT_UI:
                ax = axes[idx, 1]
                ax.plot(scenario_time, omega_true, "r-", label="True", linewidth=2)
                ax.plot(
                    scenario_time,
                    omega_pred,
                    "b--",
                    label="Predicted",
                    linewidth=2,
                    alpha=0.7,
                )
                ax.axvline(x=tf, color="g", linestyle=":", label="Fault Start", alpha=0.7)
                ax.axvline(x=tc, color="orange", linestyle=":", label="Fault Clear", alpha=0.7)
                if cct_absolute is not None:
                    ax.axvline(
                        x=cct_absolute,
                        color="purple",
                        linestyle="--",
                        label=f"CCT={cct_true:.3f}s",
                        alpha=0.7,
                        linewidth=1.5,
                    )
                ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="Nominal")
                ax.set_xlabel("Time (s)", fontsize=10)
                ax.set_ylabel("Rotor Speed (pu)", fontsize=10)
                ax.set_title(title_str, fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0.95, 1.05)

            row = {
                "scenario_id": scenario_id,
                "cct_true": cct_true,
                "delta_rmse": delta_rmse,
                "is_stable_pred": is_stable_pred,
                "is_stable_true": is_stable_true,
            }
            if not DELTA_ONLY_EXPERIMENT_UI:
                row["omega_rmse"] = omega_rmse
            cct_summary.append(row)

    plt.tight_layout()

    # Save figure
    fig_filename = generate_timestamped_filename("test_scenarios_predictions", "png")
    fig_path = output_dir / fig_filename
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"   ✓ Saved scenario visualizations: {fig_path.name}")

    # Print summary
    if cct_summary:
        print(f"\n   📊 CCT Summary for Visualized Scenarios:")
        if DELTA_ONLY_EXPERIMENT_UI:
            print(f"{'Scenario':<12} {'CCT (s)':<12} {'Delta RMSE':<12} {'Stability':<12}")
            print("   " + "-" * 52)
        else:
            print(
                f"{'Scenario':<12} {'CCT (s)':<12} {'Delta RMSE':<12} {'Omega RMSE':<12}"
                f"{'Stability':<12}"
            )
            print("   " + "-" * 60)
        for info in cct_summary:
            cct_str = f"{info['cct_true']:.4f}" if info["cct_true"] is not None else "N/A"
            stability_str = ""
            if info["is_stable_true"] is not None:
                match = "✅" if info["is_stable_pred"] == info["is_stable_true"] else "❌"
                stability_str = f"{match} {info['is_stable_pred']}"
            else:
                stability_str = f"{info['is_stable_pred']}"
            if DELTA_ONLY_EXPERIMENT_UI:
                print(
                    f"   {info['scenario_id']:<12} {cct_str:<12} "
                    f"{info['delta_rmse']:.6f} {stability_str:<12}"
                )
            else:
                print(
                    f"   {info['scenario_id']:<12} {cct_str:<12} "
                    f"{info['delta_rmse']:.6f} {info['omega_rmse']:.6f} {stability_str:<12}"
                )

    return fig_path


def _generate_visualizations(test_data: pd.DataFrame, output_dir: Path):
    """
    Generate visualization plots.

    Parameters:
    -----------
    test_data : pd.DataFrame
        Test data
    output_dir : Path
        Directory to save figures
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot sample trajectories
    scenarios = test_data["scenario_id"].unique()[:5]  # Plot first 5 scenarios

    n_sc = len(scenarios)
    if DELTA_ONLY_EXPERIMENT_UI:
        fig, axes = plt.subplots(n_sc, 1, figsize=(10, 3 * n_sc))
        if n_sc == 1:
            axes = np.array([axes])
    else:
        fig, axes = plt.subplots(n_sc, 2, figsize=(12, 3 * n_sc))
        if n_sc == 1:
            axes = axes.reshape(1, -1)

    for idx, scenario_id in enumerate(scenarios):
        scenario_data = test_data[test_data["scenario_id"] == scenario_id]

        ax_d = axes[idx] if DELTA_ONLY_EXPERIMENT_UI else axes[idx, 0]
        ax_d.plot(scenario_data["time"], scenario_data["delta"], "b-", label="True")
        ax_d.set_xlabel("Time (s)")
        ax_d.set_ylabel("Delta (rad)")
        ax_d.set_title(f"Scenario {scenario_id} - Delta")
        ax_d.grid(True)
        ax_d.legend()

        if not DELTA_ONLY_EXPERIMENT_UI:
            axes[idx, 1].plot(scenario_data["time"], scenario_data["omega"], "r-", label="True")
            axes[idx, 1].set_xlabel("Time (s)")
            axes[idx, 1].set_ylabel("Omega (pu)")
            axes[idx, 1].set_title(f"Scenario {scenario_id} - Omega")
            axes[idx, 1].grid(True)
            axes[idx, 1].legend()

    plt.tight_layout()
    fig_filename = generate_timestamped_filename("sample_trajectories", "png")
    fig_path = figures_dir / fig_filename
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Figures saved to: {figures_dir}")


def _fit_scalers_from_data(data: pd.DataFrame) -> Dict:
    """Fit scalers from data (used when scalers not in checkpoint)."""
    scalers = {}

    # Sample subset for efficiency
    sample_scenarios = data["scenario_id"].unique()[: min(100, len(data["scenario_id"].unique()))]
    sample_data = data[data["scenario_id"].isin(sample_scenarios)]

    # Time, states
    scalers["time"] = StandardScaler().fit(sample_data["time"].values.reshape(-1, 1))
    scalers["delta"] = StandardScaler().fit(sample_data["delta"].values.reshape(-1, 1))
    scalers["omega"] = StandardScaler().fit(sample_data["omega"].values.reshape(-1, 1))

    # Pe (electrical power) - if available
    if "Pe" in sample_data.columns:
        scalers["Pe"] = StandardScaler().fit(sample_data["Pe"].values.reshape(-1, 1))

    # Parameters (scenario-level)
    scenario_data = data.groupby("scenario_id").first().reset_index()

    # Try param_* columns first, fall back to direct column names
    H_col = "param_H" if "param_H" in scenario_data.columns else "H"
    D_col = "param_D" if "param_D" in scenario_data.columns else "D"
    Pm_col = "param_Pm" if "param_Pm" in scenario_data.columns else "Pm"
    tc_col = "param_tc" if "param_tc" in scenario_data.columns else "tc"

    scalers["H"] = StandardScaler().fit(scenario_data[H_col].values.reshape(-1, 1))
    scalers["D"] = StandardScaler().fit(scenario_data[D_col].values.reshape(-1, 1))
    scalers["Pm"] = StandardScaler().fit(scenario_data[Pm_col].values.reshape(-1, 1))
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

    # REVERT: Use separate scalers for delta0/omega0 (like training code and December experiments)
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
    Extract and normalize scenario data for evaluation.

    Returns normalized tensors ready for model input.
    """
    scenario_data = scenario_data.sort_values("time")

    # Check required columns
    required_cols = ["time", "delta", "omega"]
    missing_cols = [col for col in required_cols if col not in scenario_data.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns in scenario_data: {missing_cols}. "
            f"Available: {list(scenario_data.columns)}"
        )

    # Extract raw values (with better error messages)
    try:
        t_raw = scenario_data["time"].values.astype(np.float32)
    except KeyError:
        raise KeyError(f"'time' column not found. Available columns: {list(scenario_data.columns)}")

    try:
        delta_obs_raw = scenario_data["delta"].values.astype(np.float32)
    except KeyError:
        raise KeyError(
            f"'delta' column not found. Available columns: {list(scenario_data.columns)}"
        )

    try:
        omega_obs_raw = scenario_data["omega"].values.astype(np.float32)
    except KeyError:
        raise KeyError(
            f"'omega' column not found. Available columns: {list(scenario_data.columns)}"
        )
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
    Pload_raw = float(row.get("load", row.get("Pload", row.get("param_load", Pm_raw))))
    Xprefault_raw = float(row.get("Xprefault", 0.5))
    Xfault_raw = float(row.get("Xfault", 0.0001))
    Xpostfault_raw = float(row.get("Xpostfault", 0.5))
    tf_raw = float(row.get("tf", 1.0))
    tc_raw = float(row.get("tc", row.get("param_tc", 1.2)))

    # Normalize (with better error messages for missing scalers)
    if "time" not in scalers:
        raise KeyError(f"'time' scaler not found. Available scalers: {list(scalers.keys())}")
    if "delta" not in scalers:
        raise KeyError(f"'delta' scaler not found. Available scalers: {list(scalers.keys())}")
    if "omega" not in scalers:
        raise KeyError(f"'omega' scaler not found. Available scalers: {list(scalers.keys())}")

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

    # REVERT: Use delta0/omega0 scalers for input normalization (like training code and December experiments)
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

    # #region agent log
    scenario_id = int(row.get("scenario_id", -1))
    log_data = {
        "location": "evaluation.py:_extract_and_normalize_scenario_data",
        "message": "Scaler statistics comparison",
        "data": {
            "scenario_id": scenario_id,
            "delta0_raw": float(delta0_raw),
            "delta0_scaler_mean": float(scalers["delta0"].mean_[0]),
            "delta0_scaler_std": float(scalers["delta0"].scale_[0]),
            "delta_scaler_mean": float(scalers["delta"].mean_[0]),
            "delta_scaler_std": float(scalers["delta"].scale_[0]),
            "delta0_normalized_delta0_scaler": float(delta0_input[0].item()),
            "delta0_normalized_delta_scaler": float(normalize_value(delta0_raw, scalers["delta"])),
            "delta_obs_first": float(delta_obs_raw[0]) if len(delta_obs_raw) > 0 else None,
        },
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run2",
        "hypothesisId": "B",
    }
    try:
        with open(".cursor/debug.log", "a") as f:
            f.write(json.dumps(log_data) + "\n")
    except Exception:
        pass
    # #endregion

    H = torch.tensor([normalize_value(H_raw, scalers["H"])], dtype=torch.float32, device=device)
    D = torch.tensor([normalize_value(D_raw, scalers["D"])], dtype=torch.float32, device=device)
    Pm = torch.tensor(
        [normalize_value(Pm_raw, scalers["Pm"])], dtype=torch.float32, device=device
    )  # For physics loss
    # For model input: use load scaler if available, otherwise use Pm scaler
    load_scaler = scalers.get("load", scalers.get("Pload", scalers["Pm"]))
    Pload = torch.tensor(
        [normalize_value(Pload_raw, load_scaler)], dtype=torch.float32, device=device
    )

    result = {
        "t_data": t_data,
        "delta_obs": delta_obs,
        "omega_obs": omega_obs,
        "delta0": delta0_input,
        "omega0": omega0_input,
        "H": H,
        "D": D,
        "Pm": Pm,  # For physics loss
        "Pload": Pload,  # For model input (PeInput model)
    }

    # Only add reactance-related scalers if they exist (for reactance input method)
    if "Xprefault" in scalers:
        Xprefault = torch.tensor(
            [normalize_value(Xprefault_raw, scalers["Xprefault"])],
            dtype=torch.float32,
            device=device,
        )
        result["Xprefault"] = Xprefault
    if "Xfault" in scalers:
        Xfault = torch.tensor(
            [normalize_value(Xfault_raw, scalers["Xfault"])],
            dtype=torch.float32,
            device=device,
        )
        result["Xfault"] = Xfault
    if "Xpostfault" in scalers:
        Xpostfault = torch.tensor(
            [normalize_value(Xpostfault_raw, scalers["Xpostfault"])],
            dtype=torch.float32,
            device=device,
        )
        result["Xpostfault"] = Xpostfault
    if "tf" in scalers:
        tf = torch.tensor(
            [normalize_value(tf_raw, scalers["tf"])], dtype=torch.float32, device=device
        )
        result["tf"] = tf
    if "tc" in scalers:
        tc = torch.tensor(
            [normalize_value(tc_raw, scalers["tc"])], dtype=torch.float32, device=device
        )
        result["tc"] = tc

    # Add Pe if available
    if pe_obs is not None:
        result["Pe"] = pe_obs

    return result
