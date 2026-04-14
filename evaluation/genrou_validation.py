"""
GENROU Model Validation.

Validates PINN trained on GENCLS against GENROU (detailed generator model).
This is standard practice for publication: train on GENCLS, validate on GENROU.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False
    print("Warning: ANDES not available. GENROU validation will not work.")

from pinn.checkpoint_layout import infer_architecture_from_state_dict
from pinn.trajectory_prediction import (
    TrajectoryPredictionPINN,
    TrajectoryPredictionPINN_PeInput,
)
from scripts.core.utils import load_config
from utils.metrics import compute_trajectory_metrics, r2_score
from utils.normalization import set_model_standardization_to_identity
from evaluation.genrou_simulation import run_genrou_trajectory


def prepare_pinn_inputs(
    t: np.ndarray,
    delta0: float,
    omega0: float,
    H: float,
    D: float,
    Pm: float,
    Pe_t: np.ndarray,
    input_method: str = "pe_direct",
    scalers: dict = None,
) -> torch.Tensor:
    """
    Prepare PINN inputs from scenario parameters.

    Normalizes inputs using training scalers if provided.

    Parameters:
    -----------
    t : np.ndarray
        Time array
    delta0 : float
        Initial rotor angle
    omega0 : float
        Initial rotor speed
    H : float
        Inertia constant
    D : float
        Damping coefficient
    Pm : float
        Mechanical power
    Pe_t : np.ndarray
        Electrical power time series
    input_method : str
        Input method: "pe_direct" or "reactance"
    scalers : dict, optional
        Dictionary of scalers for normalization. If None, inputs are not normalized.

    Returns:
    --------
    inputs : torch.Tensor
        PINN input tensor (normalized if scalers provided)
    """
    from utils.normalization import normalize_value, normalize_array

    n_points = len(t)

    if input_method in ("pe_direct", "pe_direct_7"):
        # 7-D Pe-direct layout: [t, delta0, omega0, H, D, Pm, Pe(t)] (matches pe_direct_7 / ML)
        inputs = np.zeros((n_points, 7))

        # Normalize if scalers provided
        if scalers:
            inputs[:, 0] = normalize_array(t, scalers.get("time"))
            inputs[:, 1] = normalize_value(delta0, scalers.get("delta0", scalers.get("delta")))
            inputs[:, 2] = normalize_value(omega0, scalers.get("omega0", scalers.get("omega")))
            inputs[:, 3] = normalize_value(H, scalers.get("H"))
            inputs[:, 4] = normalize_value(D, scalers.get("D"))
            inputs[:, 5] = normalize_value(Pm, scalers.get("Pm"))
            pe_normalized = (
                Pe_t[:n_points]
                if len(Pe_t) >= n_points
                else np.pad(Pe_t, (0, n_points - len(Pe_t)), mode="edge")
            )
            if "Pe" in scalers:
                inputs[:, 6] = normalize_array(pe_normalized, scalers["Pe"])
            else:
                inputs[:, 6] = pe_normalized  # Use raw if no Pe scaler
        else:
            # No normalization
            inputs[:, 0] = t
            inputs[:, 1] = delta0
            inputs[:, 2] = omega0
            inputs[:, 3] = H
            inputs[:, 4] = D
            inputs[:, 5] = Pm
            inputs[:, 6] = (
                Pe_t[:n_points]
                if len(Pe_t) >= n_points
                else np.pad(Pe_t, (0, n_points - len(Pe_t)), mode="edge")
            )
    else:
        # Reactance-based input (would need reactance extraction from GENROU)
        # For now, use default values
        inputs = np.zeros((n_points, 11))
        if scalers:
            inputs[:, 0] = normalize_array(t, scalers.get("time"))
            inputs[:, 1] = normalize_value(delta0, scalers.get("delta0", scalers.get("delta")))
            inputs[:, 2] = normalize_value(omega0, scalers.get("omega0", scalers.get("omega")))
            inputs[:, 3] = normalize_value(H, scalers.get("H"))
            inputs[:, 4] = normalize_value(D, scalers.get("D"))
            inputs[:, 5] = normalize_value(Pm, scalers.get("Pm"))
            inputs[:, 6] = normalize_value(0.5, scalers.get("Xprefault", lambda x: x))
            inputs[:, 7] = normalize_value(0.0001, scalers.get("Xfault", lambda x: x))
            inputs[:, 8] = normalize_value(0.5, scalers.get("Xpostfault", lambda x: x))
            inputs[:, 9] = normalize_value(1.0, scalers.get("tf"))
            inputs[:, 10] = normalize_value(1.2, scalers.get("tc"))
        else:
            inputs[:, 0] = t
            inputs[:, 1] = delta0
            inputs[:, 2] = omega0
            inputs[:, 3] = H
            inputs[:, 4] = D
            inputs[:, 5] = Pm
            inputs[:, 6] = 0.5  # Xprefault (default)
            inputs[:, 7] = 0.0001  # Xfault (default)
            inputs[:, 8] = 0.5  # Xpostfault (default)
            inputs[:, 9] = 1.0  # tf (default)
            inputs[:, 10] = 1.2  # tc (default)

    return torch.tensor(inputs, dtype=torch.float32)


def validate_pinn_on_genrou(
    pinn_model_path: str,
    genrou_case_file: str,
    test_scenarios: List[Dict],
    device: str = "cpu",
) -> List[Dict]:
    """
    Validate PINN (trained on GENCLS) against GENROU simulations.

    This is the standard approach for publication.

    Parameters:
    -----------
    pinn_model_path : str
        Path to trained PINN model (trained on GENCLS)
    genrou_case_file : str
        Path to GENROU case file
    test_scenarios : list
        List of test scenario dictionaries with parameters
    device : str
        Device to use

    Returns:
    --------
    results : list
        List of validation results for each scenario
    """
    if not ANDES_AVAILABLE:
        raise ImportError("ANDES is required for GENROU validation")

    # Load PINN model
    print(f"Loading PINN model from: {pinn_model_path}")
    checkpoint = torch.load(pinn_model_path, map_location=device, weights_only=False)

    # Get config - try multiple locations
    exp_dir = Path(pinn_model_path).parent.parent
    config_path = exp_dir / "config.yaml"

    # Try parent directories (for nested experiment structures)
    if not config_path.exists():
        config_path = exp_dir.parent / "config.yaml"
    if not config_path.exists():
        config_path = exp_dir.parent.parent / "config.yaml"

    # Try to find config in experiment summary
    if not config_path.exists():
        summary_path = exp_dir / "experiment_summary.json"
        if summary_path.exists():
            import json

            with open(summary_path, "r") as f:
                summary = json.load(f)
                if "config_path" in summary:
                    config_path = Path(summary["config_path"])

    if config_path.exists():
        config = load_config(config_path)
        print(f"✓ Loaded config from: {config_path}")
    else:
        config = checkpoint.get("config", {})
        if not config:
            print("⚠️  Warning: Could not find config file, using defaults")

    model_config = dict(config.get("model", {}))
    ckpt_mc = checkpoint.get("model_config", {})
    model_config = {**model_config, **ckpt_mc}

    sd = checkpoint.get("model_state_dict", checkpoint)
    if isinstance(sd, dict):
        use_res_inf, inp_inf, hid_inf = infer_architecture_from_state_dict(sd)
        declared_res = model_config.get("use_residual")
        if declared_res is not None and bool(declared_res) != bool(use_res_inf):
            print(
                "⚠️  Checkpoint backbone does not match config/model_config use_residual; "
                f"using weight layout (use_residual={use_res_inf})."
            )
        model_config["use_residual"] = use_res_inf
        model_config["input_dim"] = inp_inf
        if hid_inf:
            model_config["hidden_dims"] = hid_inf

    input_method = model_config.get("input_method", "pe_direct")

    # Build model
    if input_method in ("pe_direct", "pe_direct_7") or model_config.get("use_pe_as_input", False):
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
            dropout=model_config.get("dropout", 0.0),
            use_standardization=model_config.get("use_standardization", True),
        )
    else:
        model = TrajectoryPredictionPINN(
            input_dim=model_config.get("input_dim", 11),
            hidden_dims=model_config.get("hidden_dims", [64, 64, 64, 64]),
            activation=model_config.get("activation", "tanh"),
            use_residual=model_config.get("use_residual", False),
            dropout=model_config.get("dropout", 0.0),
            use_standardization=model_config.get("use_standardization", True),
        )

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Get scalers from checkpoint if available
    scalers = checkpoint.get("scalers", {})
    if not scalers:
        # Try to load from experiment directory
        scaler_path = exp_dir / "scalers.pkl"
        if scaler_path.exists():
            import pickle

            with open(scaler_path, "rb") as f:
                scalers = pickle.load(f)
            print(f"✓ Loaded scalers from: {scaler_path}")
        else:
            print("⚠️  Warning: No scalers found. Using identity standardization.")
            scalers = {}

    # Set standardization to identity (must match loaded weights)
    input_dim_actual = int(getattr(model, "input_dim", 11))
    set_model_standardization_to_identity(model, input_dim_actual, 2, device)

    model.eval()
    model = model.to(device)
    print("✓ PINN model loaded")

    # Resolve GENROU case path (will reload for each scenario to avoid state issues)
    print(f"GENROU case file: {genrou_case_file}")
    try:
        case_path = andes.get_case(genrou_case_file)
    except Exception:
        case_path = genrou_case_file
    print(f"Resolved case path: {case_path}")

    # Validate on each scenario
    results = []
    for i, scenario in enumerate(test_scenarios):
        print(f"\nScenario {i + 1}/{len(test_scenarios)}")
        print(
            f"  Parameters: H={scenario.get('H', 'N/A')}, D={scenario.get('D', 'N/A')}, Pm={scenario.get('Pm', 'N/A')}"
        )

        try:
            traj = run_genrou_trajectory(scenario, str(case_path))
            if traj is None:
                continue

            genrou_time = traj["time"]
            genrou_delta = traj["delta"]
            genrou_omega = traj["omega"]
            pe_t = traj["Pe"]

            print(
                f"  Trajectory lengths: time={len(genrou_time)}, delta={len(genrou_delta)}, omega={len(genrou_omega)}"
            )
            if len(genrou_delta) > 0:
                print(
                    f"  Delta range: [{np.min(genrou_delta):.4f}, {np.max(genrou_delta):.4f}] rad"
                )
            if len(genrou_omega) > 0:
                print(f"  Omega range: [{np.min(genrou_omega):.4f}, {np.max(genrou_omega):.4f}] pu")

            # Get PINN prediction
            pinn_inputs = prepare_pinn_inputs(
                t=genrou_time,
                delta0=scenario.get("delta0", 0.5),
                omega0=scenario.get("omega0", 1.0),
                H=scenario.get("H", 6.0),
                D=scenario.get("D", 1.0),
                Pm=scenario.get("Pm", 0.8),
                Pe_t=pe_t,
                input_method=input_method,
                scalers=scalers if scalers else None,
            )

            with torch.no_grad():
                pinn_inputs = pinn_inputs.to(device)
                pinn_output = model(pinn_inputs)

                if scalers and "delta" in scalers and "omega" in scalers:
                    from utils.normalization import denormalize_array

                    pinn_delta = denormalize_array(
                        pinn_output[:, 0].cpu().numpy(), scalers["delta"]
                    )
                    pinn_omega = denormalize_array(
                        pinn_output[:, 1].cpu().numpy(), scalers["omega"]
                    )
                else:
                    pinn_delta = pinn_output[:, 0].cpu().numpy()
                    pinn_omega = pinn_output[:, 1].cpu().numpy()

            min_len = min(len(genrou_delta), len(pinn_delta))
            if min_len == 0:
                print(f"  ⚠️  Warning: Empty trajectories, skipping scenario")
                continue

            genrou_delta = genrou_delta[:min_len]
            genrou_omega = genrou_omega[:min_len]
            pinn_delta = pinn_delta[:min_len]
            pinn_omega = pinn_omega[:min_len]

            # Principal values: ANDES may integrate δ without folding; training labels use [-π, π].
            genrou_delta = np.arctan2(np.sin(genrou_delta), np.cos(genrou_delta))
            pinn_delta = np.arctan2(np.sin(pinn_delta), np.cos(pinn_delta))

            if np.any(np.isnan(genrou_delta)) or np.any(np.isnan(genrou_omega)):
                print(f"  ⚠️  Warning: GENROU trajectories contain NaN, skipping scenario")
                continue
            if np.any(np.isnan(pinn_delta)) or np.any(np.isnan(pinn_omega)):
                print(f"  ⚠️  Warning: PINN predictions contain NaN, skipping scenario")
                continue

            if np.any(np.isinf(genrou_delta)) or np.any(np.isinf(genrou_omega)):
                print(f"  ⚠️  Warning: GENROU trajectories contain Inf, skipping scenario")
                continue
            if np.any(np.isinf(pinn_delta)) or np.any(np.isinf(pinn_omega)):
                print(f"  ⚠️  Warning: PINN predictions contain Inf, skipping scenario")
                continue

            try:
                metrics = compute_trajectory_metrics(
                    delta_pred=pinn_delta,
                    omega_pred=pinn_omega,
                    delta_true=genrou_delta,
                    omega_true=genrou_omega,
                )
            except Exception as e:
                print(f"  ⚠️  Warning: Error computing metrics: {e}, skipping scenario")
                continue

            results.append(
                {
                    "scenario": scenario,
                    "delta_rmse": metrics.get("delta_rmse", 0.0),
                    "delta_rmse_wrapped": metrics.get("delta_rmse_wrapped", 0.0),
                    "omega_rmse": metrics.get("omega_rmse", 0.0),
                    "delta_r2": metrics.get("delta_r2", 0.0),
                    "omega_r2": metrics.get("omega_r2", 0.0),
                    "delta_mae": metrics.get("delta_mae", 0.0),
                    "delta_mae_wrapped": metrics.get("delta_mae_wrapped", 0.0),
                    "omega_mae": metrics.get("omega_mae", 0.0),
                }
            )

            print(
                f"  RMSE δ (wrapped): {metrics.get('delta_rmse_wrapped', 0):.4f} rad, "
                f"R² δ: {metrics.get('delta_r2', 0):.4f}, R² ω: {metrics.get('omega_r2', 0):.4f}"
            )

        except Exception as e:
            print(f"  ✗ Scenario failed: {e}")
            import traceback

            traceback.print_exc()
            continue

    return results
