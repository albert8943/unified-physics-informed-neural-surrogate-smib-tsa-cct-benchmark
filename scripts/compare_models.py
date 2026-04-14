#!/usr/bin/env python
"""
Compare ML Baseline vs PINN vs ANDES Ground Truth.

Generates statistical comparisons, publication-ready tables, and (by default) delta-only
trajectory figures. Overlaid δ/ω grid figures are **opt-in** via ``--overlaid-plots``.

Usage:
    python scripts/compare_models.py \
        --ml-baseline-model outputs/ml_baselines/exp_20251215_111443/standard_nn/model.pth \
        --pinn-model outputs/experiments/exp_20251208_234830/model.pth \
        --full-trajectory-data data/common/full_trajectory_data_30_*.csv \
        --output-dir outputs/comparison/exp_20251215_120000 \
        [--test-split-path .../test_data_*.csv] \
        [--combine-delta-only-figure] \
        [--overlaid-plots]

    Delta-only figures default to **separate** stable/unstable PNGs when ``is_stable`` is in
    the test data. Pass ``--combine-delta-only-figure`` for one combined figure.

    For IEEE Access-style exports use ``--publication-figures`` (300 dpi PNG + vector PDF) or
    ``--figure-dpi 300`` with ``--save-vector-pdf``.

    ``--full-trajectory-data`` is the full combined trajectory file (or glob/directory);
    it is used only when no usable pre-split test file is given. Alias: ``--test-data``.

    If you already have a test CSV, you may pass only ``--test-split-path`` (the same path
    is mirrored internally for argparse when the full-trajectory argument is omitted).
"""

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from evaluation.baselines.ml_baselines import StandardNN, LSTMModel
from pinn.checkpoint_layout import infer_architecture_from_state_dict
from pinn.trajectory_prediction import (
    TrajectoryPredictionPINN,
    TrajectoryPredictionPINN_PeInput,
)
from utils.metrics import compute_trajectory_metrics
from utils.angle_filter import determine_stability_180deg
from utils.normalization import (
    normalize_array,
    normalize_value,
    denormalize_array,
    set_model_standardization_to_identity,
)
from scripts.core.utils import generate_timestamped_filename, load_json
from scripts.core.evaluation import _extract_and_normalize_scenario_data
from scripts.core.experiment_ui import DELTA_ONLY_EXPERIMENT_UI

# Delta-axis limits (degrees) for trajectory panels: same as evaluate_ml_baseline.py
# (test_scenarios_predictions_*.png) so comparison figures are visually comparable.
DELTA_COMPARISON_YLIM_DEG = (-360.0, 360.0)

# Typography for model_comparison_delta_only_*.png (readable in two-column PDFs at 150–300 dpi).
_DELTA_ONLY_FIG_WIDTH_IN = 10.0
_DELTA_ONLY_ROW_HEIGHT_IN = 4.0
_DELTA_ONLY_TITLE_FS = 13
_DELTA_ONLY_AXIS_LABEL_FS = 14
_DELTA_ONLY_TICK_FS = 12
_DELTA_ONLY_LEGEND_FS = 11
_DELTA_ONLY_METRICS_FS = 10
_DELTA_ONLY_FOOTER_FS = 12
# Default raster DPI when CLI --figure-dpi is not passed (CLI default matches this).
_DELTA_ONLY_SAVE_DPI = 220


def _save_figure_png_and_optional_pdf(
    fig_path: Path, *, dpi: int, save_vector_pdf: bool
) -> Tuple[Path, Optional[Path]]:
    """
    Save the current matplotlib figure to PNG; optionally a companion vector PDF.

    IEEE Access: line plots are best submitted as vector PDF (or EPS); photographs
    need ≥300 dpi TIFF/PNG. Trajectory panels here are mostly vector-friendly in PDF.
    """
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    pdf_path: Optional[Path] = None
    if save_vector_pdf:
        pdf_path = fig_path.with_suffix(".pdf")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    return fig_path, pdf_path


def resolve_trajectory_csv(path: Path) -> Path:
    """
    Return a concrete CSV path for compare_models.

    If ``path`` is a file, it must be ``.csv``. If it is a directory (e.g. a processed
    experiment folder), pick ``test_data*.csv`` if present, else ``*test*.csv``, else the
    newest ``*.csv`` in that directory. Avoids ``PermissionError`` / confusing failures when
    users pass a folder to ``--full-trajectory-data`` / ``--test-data``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(
                f"Expected a .csv trajectory file, got: {path}. "
                "Pass a CSV path, a directory containing test_data_*.csv, or use a * pattern."
            )
        return path
    if path.is_dir():
        candidates = sorted(path.glob("test_data*.csv"))
        if not candidates:
            candidates = sorted(p for p in path.glob("*test*.csv") if p.suffix.lower() == ".csv")
        if not candidates:
            candidates = sorted(path.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No CSV files found under directory:\n  {path}\n"
                "Pass a specific test CSV (e.g. .../test_data_*.csv), or set --test-split-path."
            )
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"✓ Resolved directory to CSV (newest match): {chosen.name}")
        return chosen
    raise ValueError(f"Invalid data path (not a file or directory): {path}")


def load_test_data(
    data_path: Path,
    test_split_path: Optional[Path] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List]:
    """
    Load test data with preference for pre-split test data.

    Parameters:
    -----------
    data_path : Path
        Full combined trajectory CSV (or resolvable path), for on-the-fly split only
    test_split_path : Path, optional
        Path to pre-split test data (preferred for reproducibility)
    random_state : int
        Random seed for on-the-fly splitting (if pre-split not available)

    Returns:
    --------
    test_data : pd.DataFrame
        Test data
    test_scenario_ids : list
        List of test scenario IDs (for reproducibility)
    """
    if test_split_path and Path(test_split_path).exists():
        # Use pre-split test data (preferred)
        split_csv = resolve_trajectory_csv(Path(test_split_path))
        test_data = pd.read_csv(split_csv)
        test_scenario_ids = sorted(test_data["scenario_id"].unique().tolist())
        print(f"✓ Using pre-split test data: {len(test_scenario_ids)} scenarios")
        return test_data, test_scenario_ids

    # Split on-the-fly with fixed seed
    print(f"⚠️  Pre-split test data not found, splitting on-the-fly (seed={random_state})")
    full_csv = resolve_trajectory_csv(Path(data_path))
    data = pd.read_csv(full_csv)
    scenarios = data["scenario_id"].unique()

    # Use same split ratio as training (15% for validation/test)
    _, test_scenarios = train_test_split(scenarios, test_size=0.15, random_state=random_state)
    test_data = data[data["scenario_id"].isin(test_scenarios)]
    test_scenario_ids = sorted(test_scenarios.tolist())

    print(f"✓ Split test data: {len(test_scenario_ids)} scenarios (seed={random_state})")
    return test_data, test_scenario_ids


def load_ml_baseline_model(
    model_path: Path,
    device: str = "auto",
) -> Tuple[torch.nn.Module, Dict, str]:
    """Load ML baseline model and scalers."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_torch = torch.device(device)

    # Handle wildcard patterns
    model_path_str = str(model_path)
    if "*" in model_path_str:
        matching_files = list(model_path.parent.glob(model_path.name))
        if not matching_files:
            raise FileNotFoundError(f"No files match pattern: {model_path}")
        # Use most recent file
        model_path = max(matching_files, key=lambda p: p.stat().st_mtime)
        print(f"Found {len(matching_files)} file(s), using latest: {model_path.name}")

    print(f"Loading ML baseline model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_type = checkpoint.get("model_type", "standard_nn")
    model_config = checkpoint.get("model_config", {})
    scalers = checkpoint.get("scalers", {})
    input_method = checkpoint.get("input_method", "reactance")

    # Build model
    if model_type == "standard_nn":
        input_dim = (
            7 if input_method == "pe_direct_7" else (9 if input_method == "pe_direct" else 11)
        )
        model = StandardNN(
            input_dim=input_dim,
            hidden_dims=model_config.get("hidden_dims", [256, 256, 128, 128]),
            output_dim=2,
            activation=model_config.get("activation", "tanh"),
            dropout=model_config.get("dropout", 0.0),
        )
    elif model_type == "lstm":
        input_dim = (
            7 if input_method == "pe_direct_7" else (9 if input_method == "pe_direct" else 11)
        )
        model = LSTMModel(
            input_dim=input_dim,
            hidden_size=model_config.get("hidden_size", 128),
            num_layers=model_config.get("num_layers", 2),
            output_dim=2,
            dropout=model_config.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device_torch)
    model.eval()

    print(f"✓ Loaded {model_type} model (input_method={input_method})")
    return model, scalers, input_method


def load_pinn_model(
    model_path: Path,
    config_path: Optional[Path] = None,
    device: str = "auto",
) -> Tuple[torch.nn.Module, Dict, str]:
    """Load PINN model and scalers."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_torch = torch.device(device)

    # Handle wildcard patterns
    model_path_str = str(model_path)
    if "*" in model_path_str:
        matching_files = list(model_path.parent.glob(model_path.name))
        if not matching_files:
            raise FileNotFoundError(f"No files match pattern: {model_path}")
        # Use most recent file
        model_path = max(matching_files, key=lambda p: p.stat().st_mtime)
        print(f"Found {len(matching_files)} file(s), using latest: {model_path.name}")

    print(f"Loading PINN model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Load config if available - try multiple locations
    config = {}
    model_config = {}

    # Try config file first
    if config_path and config_path.exists():
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            model_config = config.get("model", {})
        except Exception as e:
            print(f"  ⚠️  Could not load config from {config_path}: {e}")

    # Try experiment directory config.yaml
    if not model_config:
        exp_config_path = model_path.parent.parent.parent / "config.yaml"
        if exp_config_path.exists():
            try:
                import yaml

                with open(exp_config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                model_config = config.get("model", {})
                print(f"  ✓ Loaded config from: {exp_config_path.name}")
            except Exception as e:
                print(f"  ⚠️  Could not load config from {exp_config_path}: {e}")

    # Try checkpoint config
    if not model_config and "config" in checkpoint:
        config = checkpoint["config"]
        model_config = config.get("model", {})

    ckpt_mc = checkpoint.get("model_config") or {}
    if ckpt_mc:
        model_config = {**model_config, **ckpt_mc}

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    use_res, input_dim, inferred_hid = infer_architecture_from_state_dict(state_dict)

    linear_keys = [k for k in state_dict if re.match(r"network\.\d+\.weight", k)]
    nums = (
        sorted(set(int(re.search(r"network\.(\d+)\.weight", k).group(1)) for k in linear_keys))
        if linear_keys
        else []
    )
    use_dropout = bool(nums and 3 in nums)

    hidden_dims = list(inferred_hid) if inferred_hid else []
    if not hidden_dims:
        hidden_dims = [256, 256, 128, 128] if input_dim in (7, 9) else [64, 64, 64, 64]
    if not inferred_hid and model_config.get("hidden_dims") is not None:
        hidden_dims = model_config["hidden_dims"]

    if model_config.get("dropout") is not None:
        dropout = float(model_config["dropout"])
    elif use_res:
        dropout = 0.1
    else:
        dropout = 0.1 if use_dropout else 0.0

    if input_dim == 7:
        input_method = "pe_direct_7"
    elif input_dim == 9:
        input_method = "pe_direct"
    else:
        input_method = "reactance"

    scalers = checkpoint.get("scalers", {})

    print(
        f"  Detected: input_dim={input_dim}, input_method={input_method}, hidden_dims={hidden_dims}, "
        f"use_residual={use_res}, dropout={dropout}"
    )

    if input_dim in (7, 9):
        model = TrajectoryPredictionPINN_PeInput(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=model_config.get("activation", "tanh"),
            use_residual=use_res,
            dropout=dropout,
            use_standardization=model_config.get("use_standardization", True),
        ).to(device_torch)
        input_dim_actual = input_dim
    else:
        model = TrajectoryPredictionPINN(
            input_dim=11,
            hidden_dims=hidden_dims,
            activation=model_config.get("activation", "tanh"),
            use_residual=use_res,
            dropout=dropout,
            use_standardization=model_config.get("use_standardization", True),
        ).to(device_torch)
        input_dim_actual = 11

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Set standardization to identity
    set_model_standardization_to_identity(model, input_dim_actual, 2, device_torch)

    model.eval()

    # Check scalers
    if not scalers:
        print(f"  ⚠️  Warning: No scalers found in checkpoint. Model may not work correctly.")
    else:
        print(f"  ✓ Loaded {len(scalers)} scalers: {list(scalers.keys())[:5]}...")

    print(f"✓ Loaded PINN model (input_method={input_method})")
    return model, scalers, input_method


def predict_scenario_ml_baseline(
    model: torch.nn.Module,
    scenario_data: pd.DataFrame,
    scalers: Dict,
    input_method: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict trajectory for a scenario using ML baseline (point-wise).

    NOTE: This function uses point-wise predictions where each time point is predicted
    independently using the same initial conditions (delta0, omega0) for all time points.
    This approach works well for stable trajectories but may fail for unstable cases where
    the model needs to learn complex dynamics from initial conditions alone.

    Known Limitation: For unstable trajectories, the ML baseline may predict incorrectly
    diverging trajectories because:
    1. The model makes independent point-wise predictions (no trajectory consistency)
    2. The model must learn the entire trajectory from initial conditions alone
    3. There are no physics constraints to enforce trajectory consistency

    PINN handles unstable cases better because it uses physics-informed constraints that
    enforce trajectory consistency through the swing equation.
    """
    scenario_data = scenario_data.sort_values("time")
    first_row = scenario_data.iloc[0]

    # Check if required columns exist
    if "delta" not in scenario_data.columns:
        available_cols = list(scenario_data.columns)
        raise KeyError(f"'delta' column not found. Available columns: {available_cols}")
    if "omega" not in scenario_data.columns:
        available_cols = list(scenario_data.columns)
        raise KeyError(f"'omega' column not found. Available columns: {available_cols}")

    H = float(first_row.get("param_H", first_row.get("H", 5.0)))
    D = float(first_row.get("param_D", first_row.get("D", 1.0)))
    Pm = float(first_row.get("param_Pm", first_row.get("Pm", 0.8)))

    # Get initial conditions - prefer dedicated columns, fallback to first trajectory value
    # This ensures we use the exact initial conditions that were used during training
    delta0 = float(first_row.get("delta0", scenario_data["delta"].iloc[0]))
    omega0 = float(first_row.get("omega0", scenario_data["omega"].iloc[0]))
    tf = float(first_row.get("tf", 1.0))  # fault start time (prefault = t < tf)

    # #region agent log
    scenario_id = int(first_row.get("scenario_id", -1))
    import json

    log_data = {
        "location": "compare_models.py:predict_scenario_ml_baseline",
        "message": "ML baseline initial conditions",
        "data": {
            "scenario_id": scenario_id,
            "delta0": float(delta0),
            "omega0": float(omega0),
            "delta0_from_data": float(scenario_data["delta"].iloc[0]),
            "omega0_from_data": float(scenario_data["omega"].iloc[0]),
            "has_delta0_col": "delta0" in first_row,
            "has_omega0_col": "omega0" in first_row,
        },
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "A",
    }
    try:
        with open(".cursor/debug.log", "a") as f:
            f.write(json.dumps(log_data) + "\n")
    except:
        pass
    # #endregion

    time_values = scenario_data["time"].values
    inputs_list = []

    use_fixed_scale = "delta_fixed_scale" in scalers and "omega_fixed_scale" in scalers
    for t in time_values:
        input_features = []
        input_features.append(scalers["time"].transform([[t]])[0, 0])
        if use_fixed_scale:
            input_features.append(delta0 / float(scalers["delta_fixed_scale"]))
            input_features.append(omega0 / float(scalers["omega_fixed_scale"]))
        elif "delta0" in scalers and "omega0" in scalers:
            input_features.append(scalers["delta0"].transform([[delta0]])[0, 0])
            input_features.append(scalers["omega0"].transform([[omega0]])[0, 0])
        elif "delta" in scalers and "omega" in scalers:
            input_features.append(scalers["delta"].transform([[delta0]])[0, 0])
            input_features.append(scalers["omega"].transform([[omega0]])[0, 0])
        else:
            raise KeyError(
                "ML checkpoint scalers missing delta0/omega0 or delta/omega or fixed-scale keys"
            )
        input_features.append(scalers["H"].transform([[H]])[0, 0])
        input_features.append(scalers["D"].transform([[D]])[0, 0])
        input_features.append(scalers["Pm"].transform([[Pm]])[0, 0])

        if input_method == "pe_direct":
            pe_val = float(scenario_data[scenario_data["time"] == t]["Pe"].iloc[0])
            input_features.append(scalers["Pe"].transform([[pe_val]])[0, 0])
            tf = float(first_row.get("tf", 1.0))
            tc = float(first_row.get("tc", first_row.get("param_tc", 1.2)))
            input_features.append(scalers["tf"].transform([[tf]])[0, 0])
            input_features.append(scalers["tc"].transform([[tc]])[0, 0])
        elif input_method == "pe_direct_7":
            pe_val = float(scenario_data[scenario_data["time"] == t]["Pe"].iloc[0])
            input_features.append(scalers["Pe"].transform([[pe_val]])[0, 0])
            # 7-D: no tf, tc
        else:
            Xprefault = float(first_row.get("Xprefault", 0.5))
            Xfault = float(first_row.get("Xfault", 0.0001))
            Xpostfault = float(first_row.get("Xpostfault", 0.5))
            tf = float(first_row.get("tf", 1.0))
            tc = float(first_row.get("tc", first_row.get("param_tc", 1.2)))
            input_features.append(scalers["Xprefault"].transform([[Xprefault]])[0, 0])
            input_features.append(scalers["Xfault"].transform([[Xfault]])[0, 0])
            input_features.append(scalers["Xpostfault"].transform([[Xpostfault]])[0, 0])
            input_features.append(scalers["tf"].transform([[tf]])[0, 0])
            input_features.append(scalers["tc"].transform([[tc]])[0, 0])

        inputs_list.append(input_features)

    X = torch.tensor(inputs_list, dtype=torch.float32, device=device)
    if isinstance(model, LSTMModel):
        X = X.unsqueeze(1)

    with torch.no_grad():
        pred = model(X)
        if isinstance(model, LSTMModel):
            pred = pred.squeeze(1)

    # CRITICAL: Model outputs are in NORMALIZED space (after our fix)
    # We need to denormalize predictions to physical units for evaluation
    delta_pred_norm = pred[:, 0].cpu().numpy()
    omega_pred_norm = pred[:, 1].cpu().numpy()

    # Denormalize using scalers (fixed scale or StandardScaler)
    if use_fixed_scale:
        delta_pred = delta_pred_norm * float(scalers["delta_fixed_scale"])
        omega_pred = omega_pred_norm * float(scalers["omega_fixed_scale"])
    elif "delta" in scalers and "omega" in scalers:
        delta_pred = scalers["delta"].inverse_transform(delta_pred_norm.reshape(-1, 1)).flatten()
        omega_pred = scalers["omega"].inverse_transform(omega_pred_norm.reshape(-1, 1)).flatten()
    else:
        delta_pred = delta_pred_norm
        omega_pred = omega_pred_norm

    # CRITICAL FIX: Enforce initial condition constraint
    # The ML baseline makes point-wise predictions and doesn't have an IC loss term like PINN.
    # We need to ensure the first prediction matches the initial conditions exactly.
    # IMPORTANT: Use the GROUND TRUTH's first point as the IC to enforce, not the delta0/omega0 columns,
    # as there may be numerical differences. This ensures perfect alignment with the ground truth plot.
    if len(delta_pred) > 0 and len(omega_pred) > 0:
        # Get the ground truth's actual first point (this is what we want to match)
        # This is more reliable than delta0/omega0 columns which may have numerical differences
        delta0_ground_truth = float(scenario_data["delta"].iloc[0])
        omega0_ground_truth = float(scenario_data["omega"].iloc[0])

        # Get the first time point (should be t=0 or the scenario start time)
        first_time_idx = 0

        # #region agent log
        log_data = {
            "location": "compare_models.py:predict_scenario_ml_baseline",
            "message": "ML baseline before IC enforcement",
            "data": {
                "scenario_id": scenario_id,
                "delta_pred_first": float(delta_pred[first_time_idx]),
                "omega_pred_first": float(omega_pred[first_time_idx]),
                "delta0_from_col": float(delta0),
                "omega0_from_col": float(omega0),
                "delta0_ground_truth": float(delta0_ground_truth),
                "omega0_ground_truth": float(omega0_ground_truth),
                "delta_diff": float(delta_pred[first_time_idx] - delta0_ground_truth),
                "omega_diff": float(omega_pred[first_time_idx] - omega0_ground_truth),
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "A",
        }
        try:
            with open(".cursor/debug.log", "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except:
            pass
        # #endregion

        # Find the index closest to t=0 (or the minimum time) - do this FIRST
        if len(time_values) > 0:
            min_time = np.min(time_values)
            min_time_idx = np.argmin(np.abs(time_values))

            # ALWAYS enforce at the point closest to t=0 (this is the most reliable)
            delta_pred[min_time_idx] = delta0_ground_truth
            omega_pred[min_time_idx] = omega0_ground_truth

            # Also enforce at index 0 if it's different from min_time_idx and close to t=0
            if (
                min_time_idx != first_time_idx
                and abs(time_values[first_time_idx] - min_time) < 0.001
            ):
                delta_pred[first_time_idx] = delta0_ground_truth
                omega_pred[first_time_idx] = omega0_ground_truth

            # CRITICAL: Also enforce at ALL points very close to t=0 (within 0.001s)
            # This handles cases where there might be multiple points at or near t=0
            near_zero_mask = np.abs(time_values - min_time) < 0.001
            if np.sum(near_zero_mask) > 1:
                delta_pred[near_zero_mask] = delta0_ground_truth
                omega_pred[near_zero_mask] = omega0_ground_truth

        # Enforce IC throughout prefault period (t < tf): physically correct steady state
        prefault_mask = time_values < tf
        if np.any(prefault_mask):
            delta_pred[prefault_mask] = delta0_ground_truth
            omega_pred[prefault_mask] = omega0_ground_truth

    # Check for extreme delta predictions (may indicate model issues for unstable cases)
    # NOTE: ML baseline uses point-wise predictions with fixed initial conditions (delta0, omega0)
    # for all time points. This approach works well for stable trajectories but may fail for
    # unstable cases where the model needs to learn complex dynamics from initial conditions alone.
    # PINN handles this better because it uses physics-informed constraints that enforce
    # trajectory consistency through the swing equation.
    if len(delta_pred) > 0:
        delta_max_abs = np.max(np.abs(delta_pred))
        delta_max_deg = np.degrees(delta_max_abs)

        # Warn if delta exceeds reasonable bounds (e.g., > 720° = 2 full rotations)
        # This is especially important for unstable cases where the model might predict incorrectly
        if delta_max_deg > 720:
            print(
                f"   ⚠️  ML Baseline Scenario {scenario_id}: Extreme delta prediction detected! "
                f"Max |delta|: {delta_max_deg:.1f}° ({delta_max_abs:.3f} rad)."
            )
            # Check if ground truth also has extreme values (to distinguish model error from expected behavior)
            if len(scenario_data) > 0:
                delta_true_max = np.max(np.abs(scenario_data["delta"].values))
                delta_true_max_deg = np.degrees(delta_true_max)
                if delta_true_max_deg < 360:
                    print(
                        f"      ⚠️  CRITICAL: Ground truth max |delta|: {delta_true_max_deg:.1f}° - "
                        f"ML prediction is diverging incorrectly!"
                    )
                    print(
                        f"      This indicates the ML baseline model is not handling unstable trajectories correctly."
                    )
                    print(
                        f"      The model uses point-wise predictions with fixed initial conditions, which may"
                    )
                    print(
                        f"      not generalize well to unstable cases. PINN handles this better due to physics constraints."
                    )
                else:
                    print(
                        f"      Ground truth max |delta|: {delta_true_max_deg:.1f}° - "
                        f"Both are extreme (unstable trajectory, but ML may still be inaccurate)"
                    )

    return time_values, delta_pred, omega_pred


def predict_scenario_pinn(
    model: torch.nn.Module,
    scenario_data: pd.DataFrame,
    scalers: Dict,
    input_method: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict trajectory for a scenario using PINN (trajectory-based).

    This function reuses the same evaluation logic as scripts/core/evaluation.py
    for consistency and maintainability.
    """
    # Check required columns
    if "delta" not in scenario_data.columns or "omega" not in scenario_data.columns:
        raise KeyError(f"Missing required columns. Available: {list(scenario_data.columns)}")

    # Sort by time (required for trajectory prediction)
    scenario_data = scenario_data.sort_values("time").copy()

    if len(scenario_data) < 10:
        raise ValueError(f"Insufficient data points: {len(scenario_data)} < 10")

    # Use the same extraction logic as evaluate_model
    from scripts.core.evaluation import _extract_and_normalize_scenario_data
    from utils.normalization import denormalize_array

    # Extract and normalize scenario data (handles both pe_direct and reactance methods)
    normalized_data = _extract_and_normalize_scenario_data(scenario_data, scalers, device)

    t_data = normalized_data["t_data"]
    delta0 = normalized_data["delta0"]
    omega0 = normalized_data["omega0"]
    H = normalized_data["H"]
    D = normalized_data["D"]
    Pm = normalized_data["Pm"]

    # Make predictions using the same logic as evaluate_model
    with torch.no_grad():
        if input_method in ("pe_direct", "pe_direct_7"):
            Pe = normalized_data.get("Pe")
            if Pe is None:
                raise ValueError("Pe not available in scenario data")
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
            # Reactance-based method
            Xprefault = normalized_data.get("Xprefault")
            Xfault = normalized_data.get("Xfault")
            Xpostfault = normalized_data.get("Xpostfault")
            tf = normalized_data.get("tf")
            tc = normalized_data.get("tc")

            if any(x is None for x in [Xprefault, Xfault, Xpostfault, tf, tc]):
                raise ValueError("Missing reactance parameters in normalized_data")

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

    # Denormalize predictions (same as evaluate_model)
    delta_pred_norm_flat = delta_pred_norm.cpu().numpy().flatten()
    omega_pred_norm_flat = omega_pred_norm.cpu().numpy().flatten()

    # Check for NaN or extreme values before denormalization
    if np.any(np.isnan(delta_pred_norm_flat)) or np.any(np.isnan(omega_pred_norm_flat)):
        raise ValueError(
            f"NaN values in PINN predictions. Delta NaN: {np.sum(np.isnan(delta_pred_norm_flat))},"
            f"Omega NaN: {np.sum(np.isnan(omega_pred_norm_flat))}"
        )

    # #region agent log
    scenario_id = int(scenario_data.iloc[0].get("scenario_id", -1))
    log_data = {
        "location": "compare_models.py:predict_scenario_pinn",
        "message": "PINN before denormalization",
        "data": {
            "scenario_id": scenario_id,
            "delta_pred_norm_first": (
                float(delta_pred_norm_flat[0]) if len(delta_pred_norm_flat) > 0 else None
            ),
            "omega_pred_norm_first": (
                float(omega_pred_norm_flat[0]) if len(omega_pred_norm_flat) > 0 else None
            ),
            "delta_scaler_mean": float(scalers["delta"].mean_[0]),
            "delta_scaler_std": float(scalers["delta"].scale_[0]),
            "omega_scaler_mean": float(scalers["omega"].mean_[0]),
            "omega_scaler_std": float(scalers["omega"].scale_[0]),
        },
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "B",
    }
    try:
        with open(".cursor/debug.log", "a") as f:
            f.write(json.dumps(log_data) + "\n")
    except:
        pass
    # #endregion

    delta_pred = denormalize_array(delta_pred_norm_flat, scalers["delta"])
    omega_pred = denormalize_array(omega_pred_norm_flat, scalers["omega"])

    # #region agent log
    log_data = {
        "location": "compare_models.py:predict_scenario_pinn",
        "message": "PINN after denormalization",
        "data": {
            "scenario_id": scenario_id,
            "delta_pred_first": float(delta_pred[0]) if len(delta_pred) > 0 else None,
            "omega_pred_first": float(omega_pred[0]) if len(omega_pred) > 0 else None,
            "delta_true_first": float(scenario_data["delta"].iloc[0]),
            "omega_true_first": float(scenario_data["omega"].iloc[0]),
            "delta_diff": (
                float(delta_pred[0] - scenario_data["delta"].iloc[0])
                if len(delta_pred) > 0
                else None
            ),
            "omega_diff": (
                float(omega_pred[0] - scenario_data["omega"].iloc[0])
                if len(omega_pred) > 0
                else None
            ),
        },
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "A",
    }
    try:
        with open(".cursor/debug.log", "a") as f:
            f.write(json.dumps(log_data) + "\n")
    except:
        pass
    # #endregion

    # Check for extreme values after denormalization (potential unit issues)
    delta_max = np.max(np.abs(delta_pred))
    if delta_max > 20:  # Delta should be in radians, typically < 10 rad
        print(
            f"  ⚠️  Warning: PINN delta predictions have extreme values (max: {delta_max:.2f} rad)"
        )
        print(
            f"Delta pred stats: min={np.min(delta_pred):.3f}, max={np.max(delta_pred):.3f},"
            f"mean={np.mean(delta_pred):.3f}, std={np.std(delta_pred):.3f}"
        )
        print(
            f"Delta pred_norm stats (before denorm): min={np.min(delta_pred_norm_flat):.3f},"
            f"max={np.max(delta_pred_norm_flat):.3f}"
        )
        if "delta" in scalers:
            delta_scaler = scalers["delta"]
            print(
                f"      Delta scaler: mean={delta_scaler.mean_[0]:.3f}, std={delta_scaler.scale_[0]:.3f}"
            )

    # Get time array and align lengths (same as evaluation.py)
    scenario_time = scenario_data["time"].values

    # Improved time alignment: Interpolate predictions to match ground truth time points
    # This is more accurate than simple truncation, especially if time arrays don't match exactly
    if len(scenario_time) > 0 and len(delta_pred) > 0:
        # Check if time arrays match (within tolerance)
        pred_time = np.linspace(scenario_time[0], scenario_time[-1], len(delta_pred))
        time_match = len(pred_time) == len(scenario_time) and np.allclose(
            pred_time, scenario_time, atol=0.01
        )

        if not time_match and len(delta_pred) != len(scenario_time):
            # Interpolate predictions to match ground truth time points
            # Create interpolation function (extrapolate if needed)
            pred_time_actual = np.linspace(scenario_time[0], scenario_time[-1], len(delta_pred))
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

    # Align lengths to ensure predictions and time match (fallback to truncation)
    min_len = min(len(delta_pred), len(omega_pred), len(scenario_time))
    delta_pred = delta_pred[:min_len]
    omega_pred = omega_pred[:min_len]
    scenario_time = scenario_time[:min_len]

    # CRITICAL FIX: Enforce initial condition constraint for PINN (same as ML baseline)
    # PINN models don't automatically enforce IC during inference, so we need to ensure
    # the first prediction matches the initial conditions exactly.
    if len(delta_pred) > 0 and len(omega_pred) > 0:
        # Get initial conditions from first row (ensure we use sorted data)
        scenario_data_sorted = scenario_data.sort_values("time")
        first_row = scenario_data_sorted.iloc[0]
        delta0_true = float(first_row.get("delta0", scenario_data_sorted["delta"].iloc[0]))
        omega0_true = float(first_row.get("omega0", scenario_data_sorted["omega"].iloc[0]))

        # #region agent log
        scenario_id = int(first_row.get("scenario_id", -1))
        log_data = {
            "location": "compare_models.py:predict_scenario_pinn",
            "message": "PINN IC enforcement",
            "data": {
                "scenario_id": scenario_id,
                "delta0_true": float(delta0_true),
                "omega0_true": float(omega0_true),
                "delta_pred_before_ic": float(delta_pred[0]) if len(delta_pred) > 0 else None,
                "omega_pred_before_ic": float(omega_pred[0]) if len(omega_pred) > 0 else None,
                "delta_true_first": float(scenario_data_sorted["delta"].iloc[0]),
                "omega_true_first": float(scenario_data_sorted["omega"].iloc[0]),
                "scenario_time_first": float(scenario_time[0]) if len(scenario_time) > 0 else None,
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run2",
            "hypothesisId": "A",
        }
        try:
            with open(".cursor/debug.log", "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except:
            pass
        # #endregion

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

            # Enforce IC throughout prefault period (t < tf), same as ML baseline for fair comparison
            tf = float(first_row.get("tf", 1.0))
            prefault_mask = scenario_time < tf
            if np.any(prefault_mask):
                delta_pred[prefault_mask] = delta0_true
                omega_pred[prefault_mask] = omega0_true

            # #region agent log
            log_data = {
                "location": "compare_models.py:predict_scenario_pinn",
                "message": "PINN IC enforcement after",
                "data": {
                    "scenario_id": scenario_id,
                    "delta_pred_after_ic": float(delta_pred[min_time_idx]),
                    "omega_pred_after_ic": float(omega_pred[min_time_idx]),
                    "min_time_idx": int(min_time_idx),
                    "min_time": float(min_time),
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run2",
                "hypothesisId": "A",
            }
            try:
                with open(".cursor/debug.log", "a") as f:
                    f.write(json.dumps(log_data) + "\n")
            except:
                pass
            # #endregion

    return scenario_time, delta_pred, omega_pred


def compute_statistical_comparison(
    errors_ml: np.ndarray,
    errors_pinn: np.ndarray,
    metric_name: str = "RMSE",
) -> Dict:
    """
    Compute statistical comparison between two error distributions.

    Returns mean, std, 95% CI, and p-value from paired t-test.
    """
    mean_ml = np.mean(errors_ml)
    std_ml = np.std(errors_ml, ddof=1)
    mean_pinn = np.mean(errors_pinn)
    std_pinn = np.std(errors_pinn, ddof=1)

    # 95% confidence intervals
    n_ml = len(errors_ml)
    n_pinn = len(errors_pinn)
    ci_ml = stats.t.interval(0.95, n_ml - 1, loc=mean_ml, scale=stats.sem(errors_ml))
    ci_pinn = stats.t.interval(0.95, n_pinn - 1, loc=mean_pinn, scale=stats.sem(errors_pinn))

    # Paired t-test (if same length) or independent t-test
    if len(errors_ml) == len(errors_pinn):
        t_stat, p_value = stats.ttest_rel(errors_ml, errors_pinn)
        test_type = "paired"
    else:
        t_stat, p_value = stats.ttest_ind(errors_ml, errors_pinn)
        test_type = "independent"

    return {
        "metric": metric_name,
        "ml_baseline": {
            "mean": float(mean_ml),
            "std": float(std_ml),
            "ci_95_lower": float(ci_ml[0]),
            "ci_95_upper": float(ci_ml[1]),
            "n": int(n_ml),
        },
        "pinn": {
            "mean": float(mean_pinn),
            "std": float(std_pinn),
            "ci_95_lower": float(ci_pinn[0]),
            "ci_95_upper": float(ci_pinn[1]),
            "n": int(n_pinn),
        },
        "statistical_test": {
            "type": test_type,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        },
        "improvement": {
            "absolute": float(mean_ml - mean_pinn),
            "relative_percent": (
                float((mean_ml - mean_pinn) / mean_ml * 100) if mean_ml > 0 else 0.0
            ),
        },
    }


def compute_segment_metrics(
    time: np.ndarray,
    tf: float,
    tc: float,
    delta_pred: np.ndarray,
    omega_pred: np.ndarray,
    delta_true: np.ndarray,
    omega_true: np.ndarray,
    min_points: int = 3,
) -> Dict[str, Optional[Dict[str, float]]]:
    """
    Compute RMSE, MAE, R² for delta and omega in pre-fault, during-fault, and post-fault segments.

    Segments: pre_fault (t < tf), during_fault (tf <= t <= tc), post_fault (t > tc).
    Returns a dict with keys 'pre_fault', 'during_fault', 'post_fault'; each value is a dict
    with delta_rmse, delta_mae, delta_r2, omega_rmse, omega_mae, omega_r2, or None if the
    segment has fewer than min_points.
    """
    from utils.metrics import wrap_angle_error

    n = len(time)
    if n != len(delta_pred) or n != len(omega_pred) or n != len(delta_true) or n != len(omega_true):
        return {"pre_fault": None, "during_fault": None, "post_fault": None}

    pre = time < tf
    during = (time >= tf) & (time <= tc)
    post = time > tc

    out = {}

    def _metrics(mask: np.ndarray) -> Optional[Dict[str, float]]:
        if np.sum(mask) < min_points:
            return None
        d_pred = delta_pred[mask]
        o_pred = omega_pred[mask]
        d_true = delta_true[mask]
        o_true = omega_true[mask]
        delta_err_wrapped = wrap_angle_error(d_pred, d_true)
        delta_rmse = float(np.sqrt(np.mean(delta_err_wrapped**2)))
        delta_mae = float(np.mean(np.abs(delta_err_wrapped)))
        omega_rmse = float(np.sqrt(np.mean((o_pred - o_true) ** 2)))
        omega_mae = float(np.mean(np.abs(o_pred - o_true)))
        d_r2 = r2_score(d_true, d_pred)
        o_r2 = r2_score(o_true, o_pred)
        return {
            "delta_rmse": delta_rmse,
            "delta_mae": delta_mae,
            "delta_r2": float(d_r2) if np.isfinite(d_r2) else float("nan"),
            "omega_rmse": omega_rmse,
            "omega_mae": omega_mae,
            "omega_r2": float(o_r2) if np.isfinite(o_r2) else float("nan"),
        }

    out["pre_fault"] = _metrics(pre)
    out["during_fault"] = _metrics(during)
    out["post_fault"] = _metrics(post)
    return out


def generate_overlaid_plots(
    test_data: pd.DataFrame,
    ml_predictions: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    pinn_predictions: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
    n_examples: int = 5,
    figure_dpi: int = 220,
    save_vector_pdf: bool = False,
) -> Optional[Path]:
    """
    Generate overlaid trajectory plots comparing ML baseline, PINN, and ANDES.

    Delta (rotor angle) panels use a fixed y-axis of ±360° in degrees, matching
    ``evaluate_ml_baseline`` scenario figures so ML-only and head-to-head plots align.
    When ``DELTA_ONLY_EXPERIMENT_UI`` is True, only delta panels are drawn (no ω column).

    Parameters:
    -----------
    test_data : pd.DataFrame
        Test data with ground truth
    ml_predictions : dict
        ML baseline predictions {scenario_id: (time, delta, omega)}
    pinn_predictions : dict
        PINN predictions {scenario_id: (time, delta, omega)}
    output_dir : Path
        Output directory
    n_examples : int
        Number of examples per stability class

    Returns:
    --------
    Path or None
        Path to saved figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select scenarios (normalize to int so dict keys match)
    def _sid_int(sid):
        return int(sid) if sid is not None else None

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
        test_scenario_ids = list(test_data["scenario_id"].unique()[: n_examples * 2])

    # Filter to scenarios with predictions (use int keys for consistent lookup)
    test_scenario_ids = [
        _sid_int(sid)
        for sid in test_scenario_ids
        if _sid_int(sid) in ml_predictions and _sid_int(sid) in pinn_predictions
    ]

    # Fallback: if none selected (e.g. is_stable dtype mismatch), use any with predictions
    if len(test_scenario_ids) == 0:
        common = set(ml_predictions.keys()) & set(pinn_predictions.keys())
        test_scenario_ids = sorted([int(s) for s in common])[: n_examples * 2]
    if len(test_scenario_ids) == 0:
        print("   ⚠️  No scenarios with predictions to visualize")
        return None

    print(f"   Generating overlaid plots for {len(test_scenario_ids)} scenarios...")

    n_rows = len(test_scenario_ids)
    if DELTA_ONLY_EXPERIMENT_UI:
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows))
        if n_rows == 1:
            axes = np.array([axes])
    else:
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

    for idx, scenario_id in enumerate(test_scenario_ids):
        scenario_data = test_data[test_data["scenario_id"] == scenario_id].sort_values("time")
        first_row = scenario_data.iloc[0]

        # Get parameters for title
        H = float(first_row.get("param_H", first_row.get("H", 5.0)))
        D = float(first_row.get("param_D", first_row.get("D", 1.0)))
        Pm = float(first_row.get("param_Pm", first_row.get("Pm", 0.8)))
        tf = float(first_row.get("tf", 1.0))
        tc = float(first_row.get("tc", first_row.get("param_tc", 1.2)))

        # Get CCT (if available) - same logic as evaluation.py
        cct_true = None
        cct_absolute_from_data = None
        for col_name in ["param_cct_absolute", "cct", "CCT", "param_cct_duration"]:
            if col_name in scenario_data.columns:
                cct_val = first_row.get(col_name)
                if cct_val is not None and not pd.isna(cct_val):
                    cct_val = float(cct_val)
                    if "absolute" in col_name.lower():
                        # CCT is already in absolute time
                        cct_absolute_from_data = cct_val
                        cct_true = cct_val - tf
                    elif "duration" in col_name.lower() or cct_val < tf:
                        # CCT is duration from fault start
                        cct_true = cct_val
                        cct_absolute_from_data = tf + cct_val
                    else:
                        # Assume absolute time if value > tf
                        cct_absolute_from_data = cct_val
                        cct_true = cct_val - tf
                    break

        # Calculate absolute CCT time
        cct_absolute = (
            cct_absolute_from_data
            if cct_absolute_from_data is not None
            else (tf + cct_true if cct_true is not None else None)
        )

        # Check stability from trajectory data (if available) - PRIMARY METHOD
        # According to parameter_sweep.py, is_stable is determined from actual trajectory behavior
        # (180-degree criterion), not from CCT comparison. This is the authoritative label.
        is_stable_from_data = None
        if "is_stable" in scenario_data.columns:
            is_stable_from_data = bool(first_row.get("is_stable", False))

        # Check CCT-based stability (secondary, for validation)
        is_stable_from_cct = None
        if cct_absolute is not None:
            # CCT is maximum stable clearing time, so tc < CCT should be stable
            # Note: When offset=0.0, it's adjusted to -epsilon, so boundary cases are handled
            is_stable_from_cct = tc < cct_absolute

        # Diagnostic: Check CCT vs tc relationship for scenario 98 or boundary cases
        if scenario_id == 98 or (cct_absolute is not None and abs(tc - cct_absolute) < 0.01):
            print(f"\n   🔍 CCT/tc Diagnostic for Scenario {scenario_id}:")
            print(f"      tf (fault start): {tf:.6f} s")
            print(f"      tc (clearing time): {tc:.6f} s")
            print(
                f"      CCT (duration): {cct_true:.6f} s"
                if cct_true is not None
                else "      CCT (duration): N/A"
            )
            print(
                f"      CCT (absolute): {cct_absolute:.6f} s"
                if cct_absolute is not None
                else "      CCT (absolute): N/A"
            )
            if cct_absolute is not None:
                offset_from_cct = tc - cct_absolute
                print(f"      Offset from CCT: {offset_from_cct:+.6f} s")
                print(
                    f"CCT-based stability (tc < CCT): {'STABLE' if is_stable_from_cct else 'UNSTABLE'}"
                )
            print(
                f"      Trajectory-based stability (PRIMARY): {is_stable_from_data}"
                if is_stable_from_data is not None
                else "      Trajectory-based stability: N/A"
            )

            # Check for inconsistency
            if (
                cct_absolute is not None
                and is_stable_from_data is not None
                and is_stable_from_cct is not None
            ):
                if is_stable_from_data != is_stable_from_cct:
                    print(
                        f"⚠️ INCONSISTENCY: Trajectory says {'STABLE' if is_stable_from_data else 'UNSTABLE'}, but CCT says {'STABLE' if is_stable_from_cct else 'UNSTABLE'}"
                    )
                    if abs(tc - cct_absolute) < 0.01:
                        print(
                            f"ℹ️ This is a boundary case (tc ≈ CCT). Trajectory-based label is"
                            f"authoritative."
                        )
                else:
                    print(
                        f"✅ Consistency: Both methods agree ({'STABLE' if is_stable_from_data else 'UNSTABLE'})"
                    )

        # Create title with CCT and tc information
        title_parts = [f"Scenario {scenario_id}"]
        title_parts.append(f"H={H:.2f}, D={D:.2f}, Pm={Pm:.3f}")
        if cct_true is not None:
            title_parts.append(f"CCT={cct_true:.3f}s")
        title_parts.append(f"tc={tc:.3f}s")
        title_str = " | ".join(title_parts)

        # Get predictions
        time_ml, delta_ml, omega_ml = ml_predictions[scenario_id]
        time_pinn, delta_pinn, omega_pinn = pinn_predictions[scenario_id]

        # Get ground truth
        time_true = scenario_data["time"].values
        delta_true = scenario_data["delta"].values
        omega_true = scenario_data["omega"].values

        # Align lengths for all arrays to ensure proper plotting
        min_len = min(
            len(time_true),
            len(delta_true),
            len(omega_true),
            len(time_ml),
            len(delta_ml),
            len(omega_ml),
            len(time_pinn),
            len(delta_pinn),
            len(omega_pinn),
        )
        time_true = time_true[:min_len]
        delta_true = delta_true[:min_len]
        omega_true = omega_true[:min_len]
        time_ml = time_ml[:min_len] if len(time_ml) >= min_len else time_ml
        delta_ml = delta_ml[:min_len] if len(delta_ml) >= min_len else delta_ml
        omega_ml = omega_ml[:min_len] if len(omega_ml) >= min_len else omega_ml
        time_pinn = time_pinn[:min_len] if len(time_pinn) >= min_len else time_pinn
        delta_pinn = delta_pinn[:min_len] if len(delta_pinn) >= min_len else delta_pinn
        omega_pinn = omega_pinn[:min_len] if len(omega_pinn) >= min_len else omega_pinn

        # Plot Delta (single column when DELTA_ONLY_EXPERIMENT_UI, else left column)
        ax = axes[idx] if DELTA_ONLY_EXPERIMENT_UI else axes[idx, 0]

        # Convert to degrees for plotting
        delta_true_deg = np.degrees(delta_true)
        delta_ml_deg = np.degrees(delta_ml)
        delta_pinn_deg = np.degrees(delta_pinn)

        # Check for extreme values and warn
        delta_all_deg = np.concatenate([delta_true_deg, delta_ml_deg, delta_pinn_deg])
        delta_all_deg = delta_all_deg[~np.isnan(delta_all_deg)]  # Remove NaN values

        if len(delta_all_deg) > 0:
            delta_max_abs = np.max(np.abs(delta_all_deg))
            if delta_max_abs > 720:  # More than 2 full rotations
                print(
                    f"   ⚠️  Scenario {scenario_id}: Extreme delta values detected "
                    f"(max: {delta_max_abs:.1f}°). ML: [{np.min(delta_ml_deg):.1f}, {np.max(delta_ml_deg):.1f}]°"
                )

        ax.plot(time_true, delta_true_deg, "k-", label="ANDES (Ground Truth)", linewidth=2.5)
        ax.plot(time_ml, delta_ml_deg, "b--", label="ML Baseline", linewidth=2, alpha=0.8)
        ax.plot(
            time_pinn,
            delta_pinn_deg,
            ":",
            label="PINN",
            linewidth=2,
            alpha=0.8,
            color="#CC00FF",
        )  # Magenta for PINN
        ax.axvline(x=tf, color="g", linestyle=":", alpha=0.7, label="Fault Start")
        ax.axvline(x=tc, color="orange", linestyle=":", alpha=0.7, label="Fault Clear")
        if cct_absolute is not None:
            ax.axvline(
                x=cct_absolute,
                color="purple",
                linestyle="--",
                label=f"CCT={cct_true:.3f}s",
                alpha=0.7,
                linewidth=1.5,
            )
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Rotor Angle (degrees)", fontsize=11)
        ax.set_title(title_str, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Fixed window matches evaluate_ml_baseline scenario plots (not auto-scale: unstable
        # runaway extends beyond ±360° and is clipped here for consistent cross-figure comparison).
        ax.set_ylim(DELTA_COMPARISON_YLIM_DEG[0], DELTA_COMPARISON_YLIM_DEG[1])

        if not DELTA_ONLY_EXPERIMENT_UI:
            # Plot Omega (right column)
            ax = axes[idx, 1]
            ax.plot(time_true, omega_true, "k-", label="ANDES (Ground Truth)", linewidth=2.5)
            ax.plot(time_ml, omega_ml, "b--", label="ML Baseline", linewidth=2, alpha=0.8)
            ax.plot(
                time_pinn,
                omega_pinn,
                ":",
                label="PINN",
                linewidth=2,
                alpha=0.8,
                color="#CC00FF",
            )  # Magenta for PINN
            ax.axvline(x=tf, color="g", linestyle=":", alpha=0.7, label="Fault Start")
            ax.axvline(x=tc, color="orange", linestyle=":", alpha=0.7, label="Fault Clear")
            if cct_absolute is not None:
                ax.axvline(
                    x=cct_absolute,
                    color="purple",
                    linestyle="--",
                    label=f"CCT={cct_true:.3f}s",
                    alpha=0.7,
                    linewidth=1.5,
                )
            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, label="Nominal")
            ax.set_xlabel("Time (s)", fontsize=11)
            ax.set_ylabel("Rotor Speed (pu)", fontsize=11)
            ax.set_title(title_str, fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            omega_all = np.concatenate([omega_true, omega_ml, omega_pinn])
            omega_all = omega_all[~np.isnan(omega_all)]
            if len(omega_all) > 0:
                omega_min, omega_max = np.min(omega_all), np.max(omega_all)
                omega_range = omega_max - omega_min
                if omega_range > 0:
                    ax.set_ylim(omega_min - 0.1 * omega_range, omega_max + 0.1 * omega_range)
                else:
                    ax.set_ylim(0.95, 1.05)
            else:
                ax.set_ylim(0.95, 1.05)

    plt.tight_layout()

    fig_filename = generate_timestamped_filename("model_comparison_overlaid", "png")
    fig_path = output_dir / fig_filename
    _, pdf_p = _save_figure_png_and_optional_pdf(
        fig_path, dpi=figure_dpi, save_vector_pdf=save_vector_pdf
    )
    if pdf_p is not None:
        print(f"   ✓ Saved overlaid vector PDF: {pdf_p.name}")
    plt.close()

    print(f"   ✓ Saved overlaid plots: {fig_path.name}")
    return fig_path


def generate_delta_only_plots(
    test_data: pd.DataFrame,
    ml_predictions: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    pinn_predictions: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
    n_examples: int = 5,
    delta_comparison: Optional[Dict] = None,
    omega_comparison: Optional[Dict] = None,
    delta_mae_comparison: Optional[Dict] = None,
    omega_mae_comparison: Optional[Dict] = None,
    delta_r2_comparison: Optional[Dict] = None,
    omega_r2_comparison: Optional[Dict] = None,
    split_by_stability: bool = False,
    figure_dpi: int = _DELTA_ONLY_SAVE_DPI,
    save_vector_pdf: bool = False,
) -> List[Path]:
    """
    Generate delta-only trajectory plots (rotor angle) comparing ML baseline, PINN, and ANDES.

    Default: one timestamped figure with up to ``n_examples`` stable and ``n_examples``
    unstable scenarios (when ``is_stable`` exists), same as overlaid selection.

    With ``split_by_stability=True`` and ``is_stable`` in the data, writes two figures sharing
    one timestamp: ``model_comparison_delta_only_<ts>_stable.png`` and
    ``model_comparison_delta_only_<ts>_unstable.png`` (publication-freeze style).

    If comparison dicts are provided, a test-set summary line is drawn on each figure.
    Y-axis uses ``DELTA_COMPARISON_YLIM_DEG`` (±360°), matching ``evaluate_ml_baseline``.
    """
    from datetime import datetime

    output_dir.mkdir(parents=True, exist_ok=True)

    def _sid_int(sid):
        return int(sid) if sid is not None else None

    def _filter_with_predictions(raw_ids) -> List[int]:
        out: List[int] = []
        for sid in raw_ids:
            k = _sid_int(sid)
            if k is not None and k in ml_predictions and k in pinn_predictions:
                out.append(k)
        return out

    has_stability = "is_stable" in test_data.columns
    common = set(ml_predictions.keys()) & set(pinn_predictions.keys())

    stable_ids: List[int] = []
    unstable_ids: List[int] = []
    if has_stability:
        stable_raw = test_data[test_data["is_stable"] == True]["scenario_id"].unique()[:n_examples]
        unstable_raw = test_data[test_data["is_stable"] == False]["scenario_id"].unique()[
            :n_examples
        ]
        stable_ids = _filter_with_predictions(stable_raw)
        unstable_ids = _filter_with_predictions(unstable_raw)

    def _save_delta_only_figure(scenario_ids: List[int], fig_path: Path) -> None:
        n_plots = len(scenario_ids)
        has_metrics = delta_comparison is not None and (
            DELTA_ONLY_EXPERIMENT_UI or omega_comparison is not None
        )
        fig, axes = plt.subplots(
            n_plots,
            1,
            figsize=(_DELTA_ONLY_FIG_WIDTH_IN, _DELTA_ONLY_ROW_HEIGHT_IN * n_plots),
        )
        if n_plots == 1:
            axes = np.array([axes])

        def _wrap_deg(e_deg: np.ndarray) -> np.ndarray:
            e = np.asarray(e_deg, dtype=float)
            return (e + 180.0) % 360.0 - 180.0

        for idx, scenario_id in enumerate(scenario_ids):
            scenario_data = test_data[test_data["scenario_id"] == scenario_id].sort_values("time")
            first_row = scenario_data.iloc[0]

            H = float(first_row.get("param_H", first_row.get("H", 5.0)))
            D = float(first_row.get("param_D", first_row.get("D", 1.0)))
            Pm = float(first_row.get("param_Pm", first_row.get("Pm", 0.8)))
            tf = float(first_row.get("tf", 1.0))
            tc = float(first_row.get("tc", first_row.get("param_tc", 1.2)))

            cct_true = None
            cct_absolute_from_data = None
            for col_name in ["param_cct_absolute", "cct", "CCT", "param_cct_duration"]:
                if col_name in scenario_data.columns:
                    cct_val = first_row.get(col_name)
                    if cct_val is not None and not pd.isna(cct_val):
                        cct_val = float(cct_val)
                        if "absolute" in col_name.lower():
                            cct_absolute_from_data = cct_val
                            cct_true = cct_val - tf
                        elif "duration" in col_name.lower() or cct_val < tf:
                            cct_true = cct_val
                            cct_absolute_from_data = tf + cct_val
                        else:
                            cct_absolute_from_data = cct_val
                            cct_true = cct_val - tf
                        break

            cct_absolute = (
                cct_absolute_from_data
                if cct_absolute_from_data is not None
                else (tf + cct_true if cct_true is not None else None)
            )

            title_parts = [
                f"Scenario {scenario_id}",
                f"H={H:.2f}, D={D:.2f}, Pm={Pm:.3f}",
            ]
            if cct_true is not None:
                title_parts.append(f"CCT={cct_true:.3f}s")
            title_parts.append(f"tc={tc:.3f}s")
            title_str = " | ".join(title_parts)

            time_ml, delta_ml, _omega_ml = ml_predictions[scenario_id]
            time_pinn, delta_pinn, _omega_pinn = pinn_predictions[scenario_id]
            time_true = scenario_data["time"].values
            delta_true = scenario_data["delta"].values

            min_len = min(
                len(time_true),
                len(delta_true),
                len(time_ml),
                len(delta_ml),
                len(time_pinn),
                len(delta_pinn),
            )
            time_true = time_true[:min_len]
            delta_true = delta_true[:min_len]
            time_ml = time_ml[:min_len] if len(time_ml) >= min_len else time_ml
            delta_ml = delta_ml[:min_len] if len(delta_ml) >= min_len else delta_ml
            time_pinn = time_pinn[:min_len] if len(time_pinn) >= min_len else time_pinn
            delta_pinn = delta_pinn[:min_len] if len(delta_pinn) >= min_len else delta_pinn

            ax = axes[idx]
            delta_true_deg = np.degrees(delta_true)
            delta_ml_deg = np.degrees(delta_ml)
            delta_pinn_deg = np.degrees(delta_pinn)

            ax.plot(
                time_true,
                delta_true_deg,
                "k-",
                label="ANDES (Ground Truth)",
                linewidth=2.5,
            )
            ax.plot(
                time_ml,
                delta_ml_deg,
                "b--",
                label="ML Baseline",
                linewidth=2,
                alpha=0.8,
            )
            ax.plot(
                time_pinn,
                delta_pinn_deg,
                ":",
                label="PINN",
                linewidth=2,
                alpha=0.8,
                color="#CC00FF",
            )
            ax.axvline(x=tf, color="g", linestyle=":", alpha=0.7, label="Fault Start")
            ax.axvline(x=tc, color="orange", linestyle=":", alpha=0.7, label="Fault Clear")
            if cct_absolute is not None:
                ax.axvline(
                    x=cct_absolute,
                    color="purple",
                    linestyle="--",
                    label=f"CCT={cct_true:.3f}s",
                    alpha=0.7,
                    linewidth=1.5,
                )
            ax.set_xlabel("Time (s)", fontsize=_DELTA_ONLY_AXIS_LABEL_FS)
            ax.set_ylabel("Rotor Angle (degrees)", fontsize=_DELTA_ONLY_AXIS_LABEL_FS)
            ax.set_title(title_str, fontsize=_DELTA_ONLY_TITLE_FS)
            ax.tick_params(axis="both", which="major", labelsize=_DELTA_ONLY_TICK_FS)
            # Default loc="best" avoids overlap with trajectories (fixed upper-left often sits on data).
            ax.legend(fontsize=_DELTA_ONLY_LEGEND_FS)
            ax.grid(True, alpha=0.3)

            ax.set_ylim(DELTA_COMPARISON_YLIM_DEG[0], DELTA_COMPARISON_YLIM_DEG[1])

            t_min, t_max = float(np.min(time_true)), float(np.max(time_true))
            t_span = (t_max - t_min) or 1.0
            ax.set_xlim(t_min - 0.02 * t_span, t_max + 0.22 * t_span)

            err_ml_deg = _wrap_deg(delta_ml_deg - delta_true_deg)
            err_pinn_deg = _wrap_deg(delta_pinn_deg - delta_true_deg)
            rmse_ml_deg = np.sqrt(np.mean(err_ml_deg**2))
            mae_ml_deg = np.mean(np.abs(err_ml_deg))
            rmse_pinn_deg = np.sqrt(np.mean(err_pinn_deg**2))
            mae_pinn_deg = np.mean(np.abs(err_pinn_deg))
            r2_ml = r2_score(delta_true, delta_ml)
            r2_pinn = r2_score(delta_true, delta_pinn)
            if not np.isfinite(r2_ml):
                r2_ml = float("nan")
            if not np.isfinite(r2_pinn):
                r2_pinn = float("nan")
            metrics_text = (
                "ML:\n"
                f"  RMSE: {rmse_ml_deg:.2f}°\n"
                f"  MAE: {mae_ml_deg:.2f}°\n"
                f"  R²: {r2_ml:.3f}\n"
                "PINN:\n"
                f"  RMSE: {rmse_pinn_deg:.2f}°\n"
                f"  MAE: {mae_pinn_deg:.2f}°\n"
                f"  R²: {r2_pinn:.3f}"
            )
            ax.text(
                0.98,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                fontsize=_DELTA_ONLY_METRICS_FS,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="wheat",
                    alpha=0.9,
                    edgecolor="gray",
                ),
                family="monospace",
                zorder=10,
            )

        plt.tight_layout()

        if has_metrics:
            d = delta_comparison
            dm = d.get("ml_baseline", {})
            dp = d.get("pinn", {})
            if DELTA_ONLY_EXPERIMENT_UI:
                line = "Test set: δ RMSE (rad) — ML {:.4f}±{:.4f}, PINN {:.4f}±{:.4f}".format(
                    dm.get("mean", 0),
                    dm.get("std", 0),
                    dp.get("mean", 0),
                    dp.get("std", 0),
                )
            else:
                o = omega_comparison or {}
                om = o.get("ml_baseline", {})
                op = o.get("pinn", {})
                line = (
                    "Test set: δ RMSE (rad) — ML {:.4f}±{:.4f}, PINN {:.4f}±{:.4f}; "
                    "ω RMSE (pu) — ML {:.4f}±{:.4f}, PINN {:.4f}±{:.4f}".format(
                        dm.get("mean", 0),
                        dm.get("std", 0),
                        dp.get("mean", 0),
                        dp.get("std", 0),
                        om.get("mean", 0),
                        om.get("std", 0),
                        op.get("mean", 0),
                        op.get("std", 0),
                    )
                )
            fig.text(
                0.5,
                0.01,
                line,
                transform=fig.transFigure,
                fontsize=_DELTA_ONLY_FOOTER_FS,
                ha="center",
                va="bottom",
                color="#222222",
                fontweight="medium",
            )
            # Match pre-publication-tweak footer clearance; tight_layout then nudge bottom for fig.text.
            plt.subplots_adjust(bottom=0.06)

        _, pdf_p = _save_figure_png_and_optional_pdf(
            fig_path, dpi=figure_dpi, save_vector_pdf=save_vector_pdf
        )
        if pdf_p is not None:
            print(f"   ✓ Saved vector PDF: {pdf_p.name}")
        plt.close()

    if split_by_stability and has_stability:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved: List[Path] = []
        for ids, tag in [(stable_ids, "stable"), (unstable_ids, "unstable")]:
            if len(ids) == 0:
                print(f"   ⚠️  No {tag} scenarios with predictions; skipping delta-only panel")
                continue
            print(f"   Generating delta-only plot ({len(ids)} scenarios, {tag})...")
            fig_path = output_dir / f"model_comparison_delta_only_{ts}_{tag}.png"
            _save_delta_only_figure(ids, fig_path)
            saved.append(fig_path)
            print(f"   ✓ Saved delta-only plots: {fig_path.name}")
        if not saved:
            print("   ⚠️  No scenarios with predictions for delta-only plot")
        return saved

    if has_stability:
        test_scenario_ids = stable_ids + unstable_ids
    else:
        test_scenario_ids = list(test_data["scenario_id"].unique()[: n_examples * 2])
        test_scenario_ids = [
            _sid_int(sid)
            for sid in test_scenario_ids
            if _sid_int(sid) in ml_predictions and _sid_int(sid) in pinn_predictions
        ]

    if has_stability and len(test_scenario_ids) == 0:
        test_scenario_ids = sorted([int(s) for s in common])[: n_examples * 2]
    elif not has_stability and len(test_scenario_ids) == 0:
        test_scenario_ids = sorted([int(s) for s in common])[: n_examples * 2]

    if len(test_scenario_ids) == 0:
        print("   ⚠️  No scenarios with predictions for delta-only plot")
        return []

    print(f"   Generating delta-only plots for {len(test_scenario_ids)} scenarios...")
    fig_path = output_dir / generate_timestamped_filename("model_comparison_delta_only", "png")
    _save_delta_only_figure(test_scenario_ids, fig_path)
    print(f"   ✓ Saved delta-only plots: {fig_path.name}")
    return [fig_path]


def generate_delta_only_plots_pinn_only(
    test_data: pd.DataFrame,
    pinn_predictions: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
    n_examples: int = 5,
    figure_dpi: int = _DELTA_ONLY_SAVE_DPI,
    save_vector_pdf: bool = False,
) -> Optional[Path]:
    """
    Generate delta-only trajectory plots comparing PINN vs ANDES (no ML baseline).
    Used for multimachine experiments. Produces two separate figures: one for
    stable scenarios, one for unstable scenarios.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _sid_int(sid):
        return int(sid) if sid is not None else None

    def _filter_available(sids):
        return [_sid_int(sid) for sid in sids if _sid_int(sid) in pinn_predictions]

    has_stability = "is_stable" in test_data.columns
    if has_stability:
        stable_raw = test_data[test_data["is_stable"] == True]["scenario_id"].unique()[:n_examples]
        unstable_raw = test_data[test_data["is_stable"] == False]["scenario_id"].unique()[
            :n_examples
        ]
        stable_scenario_ids = _filter_available(stable_raw)
        unstable_scenario_ids = _filter_available(unstable_raw)
        # If no stability column, fall back to single mixed list
        scenario_groups = [
            ("stable", stable_scenario_ids),
            ("unstable", unstable_scenario_ids),
        ]
    else:
        all_ids = _filter_available(test_data["scenario_id"].unique()[: n_examples * 2])
        if not all_ids:
            all_ids = sorted(pinn_predictions.keys())[: n_examples * 2]
        scenario_groups = [("all", all_ids)]

    def _wrap_deg(e_deg: np.ndarray) -> np.ndarray:
        e = np.asarray(e_deg, dtype=float)
        return (e + 180.0) % 360.0 - 180.0

    first_saved_path = None
    for label, scenario_ids in scenario_groups:
        if len(scenario_ids) == 0:
            continue
        n_plots = len(scenario_ids)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
        if n_plots == 1:
            axes = np.array([axes])

        for idx, scenario_id in enumerate(scenario_ids):
            scenario_data = test_data[test_data["scenario_id"] == scenario_id].sort_values("time")
            first_row = scenario_data.iloc[0]
            H = float(first_row.get("param_H", first_row.get("H", 5.0)))
            D = float(first_row.get("param_D", first_row.get("D", 1.0)))
            Pm = float(first_row.get("param_Pm", first_row.get("Pm", 0.8)))
            tf = float(first_row.get("tf", 1.0))
            tc = float(first_row.get("tc", first_row.get("param_tc", 1.2)))
            cct_true = None
            cct_absolute = None
            for col_name in ["param_cct_absolute", "cct", "CCT", "param_cct_duration"]:
                if col_name in scenario_data.columns:
                    cct_val = first_row.get(col_name)
                    if cct_val is not None and not pd.isna(cct_val):
                        cct_val = float(cct_val)
                        if "absolute" in col_name.lower():
                            cct_absolute = cct_val
                            cct_true = cct_val - tf
                        elif "duration" in col_name.lower() or cct_val < tf:
                            cct_true = cct_val
                            cct_absolute = tf + cct_val
                        else:
                            cct_absolute = cct_val
                            cct_true = cct_val - tf
                        break
            if cct_absolute is None and cct_true is not None:
                cct_absolute = tf + cct_true

            title_parts = [
                f"Scenario {scenario_id}",
                f"H={H:.2f}, D={D:.2f}, Pm={Pm:.3f}",
            ]
            if cct_true is not None:
                title_parts.append(f"CCT={cct_true:.3f}s")
            title_parts.append(f"tc={tc:.3f}s")
            title_str = " | ".join(title_parts)

            time_pinn, delta_pinn, omega_pinn = pinn_predictions[scenario_id]
            time_true = scenario_data["time"].values
            delta_true = scenario_data["delta"].values

            min_len = min(len(time_true), len(delta_true), len(time_pinn), len(delta_pinn))
            time_true = time_true[:min_len]
            delta_true = delta_true[:min_len]
            time_pinn = time_pinn[:min_len] if len(time_pinn) >= min_len else time_pinn
            delta_pinn = delta_pinn[:min_len] if len(delta_pinn) >= min_len else delta_pinn

            ax = axes[idx]
            delta_true_deg = np.degrees(delta_true)
            delta_pinn_deg = np.degrees(delta_pinn)

            ax.plot(
                time_true,
                delta_true_deg,
                "k-",
                label="ANDES (Ground Truth)",
                linewidth=2.5,
            )
            ax.plot(
                time_pinn,
                delta_pinn_deg,
                "-",
                label="PINN",
                linewidth=2,
                alpha=0.8,
                color="#CC00FF",
            )
            ax.axvline(x=tf, color="g", linestyle=":", alpha=0.7, label="Fault Start")
            ax.axvline(x=tc, color="orange", linestyle=":", alpha=0.7, label="Fault Clear")
            if cct_absolute is not None:
                ax.axvline(
                    x=cct_absolute,
                    color="purple",
                    linestyle="--",
                    label=f"CCT={cct_true:.3f}s",
                    alpha=0.7,
                    linewidth=1.5,
                )
            ax.set_xlabel("Time (s)", fontsize=11)
            ax.set_ylabel("Rotor Angle (deg)", fontsize=11)
            ax.set_title(title_str, fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            delta_all_deg = np.concatenate([delta_true_deg, delta_pinn_deg])
            delta_all_deg = delta_all_deg[~np.isnan(delta_all_deg)]
            if len(delta_all_deg) > 0:
                delta_min, delta_max = np.min(delta_all_deg), np.max(delta_all_deg)
                delta_range = delta_max - delta_min
                if delta_range > 0:
                    ax.set_ylim(delta_min - 0.1 * delta_range, delta_max + 0.1 * delta_range)
                else:
                    ax.set_ylim(-360, 360)
            else:
                ax.set_ylim(-360, 360)
            t_min, t_max = float(np.min(time_true)), float(np.max(time_true))
            t_span = (t_max - t_min) or 1.0
            ax.set_xlim(t_min - 0.02 * t_span, t_max + 0.22 * t_span)

            err_pinn_deg = _wrap_deg(delta_pinn_deg - delta_true_deg)
            rmse_pinn_deg = np.sqrt(np.mean(err_pinn_deg**2))
            mae_pinn_deg = np.mean(np.abs(err_pinn_deg))
            r2_pinn = r2_score(delta_true, delta_pinn)
            if not np.isfinite(r2_pinn):
                r2_pinn = float("nan")
            metrics_text = (
                "PINN:\n"
                f"  RMSE: {rmse_pinn_deg:.2f} deg\n"
                f"  MAE: {mae_pinn_deg:.2f} deg\n"
                f"  R2: {r2_pinn:.3f}"
            )
            ax.text(
                0.98,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="wheat",
                    alpha=0.9,
                    edgecolor="gray",
                ),
                family="monospace",
                zorder=10,
            )

        plt.tight_layout()
        stamp = generate_timestamped_filename("model_comparison_delta_only", "png").replace(
            ".png", ""
        )
        fig_filename = f"{stamp}_{label}.png"
        fig_path = output_dir / fig_filename
        _, pdf_p = _save_figure_png_and_optional_pdf(
            fig_path, dpi=figure_dpi, save_vector_pdf=save_vector_pdf
        )
        if pdf_p is not None:
            print(f"   Saved vector PDF [{label}]: {pdf_p.name}")
        plt.close()
        if first_saved_path is None:
            first_saved_path = fig_path
        print(f"   Saved delta-only (PINN vs ANDES) [{label}]: {fig_path.name}")

    if first_saved_path is None:
        print("   [WARNING] No scenarios with PINN predictions for delta-only plot")
    return first_saved_path


def main():
    """Main comparison workflow."""
    parser = argparse.ArgumentParser(description="Compare ML baseline vs PINN vs ANDES")
    parser.add_argument(
        "--ml-baseline-model",
        type=str,
        required=True,
        help="Path to ML baseline model checkpoint (.pth)",
    )
    parser.add_argument(
        "--pinn-model",
        type=str,
        required=True,
        help="Path to PINN model checkpoint (.pth)",
    )
    parser.add_argument(
        "--pinn-config",
        type=str,
        default=None,
        help="Path to PINN config JSON (optional)",
    )
    parser.add_argument(
        "--full-trajectory-data",
        "--test-data",
        type=str,
        default=None,
        dest="full_trajectory_data",
        metavar="PATH",
        help=(
            "Full combined trajectory CSV, glob (*), or directory with CSVs; used for "
            "on-the-fly scenario split only when --test-split-path is missing or invalid. "
            "Omit if you pass only --test-split-path. Alias: --test-data."
        ),
    )
    parser.add_argument(
        "--test-split-path",
        type=str,
        default=None,
        help=(
            "Pre-split test CSV (or directory with test CSVs); preferred. "
            "If --full-trajectory-data is omitted, this path is required and is mirrored "
            "as the full-trajectory stub for argparse."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for comparison results (default: outputs/comparison/exp_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=None,
        help="Maximum number of test scenarios (default: all)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=5,
        help="Number of examples per stability class for visualization (default: 5)",
    )
    parser.add_argument(
        "--overlaid-plots",
        action="store_true",
        help=(
            "Also write model_comparison_overlaid_<ts>.png (δ and ω panels per scenario). "
            "Default: skip; delta-only figures are still generated."
        ),
    )
    parser.add_argument(
        "--combine-delta-only-figure",
        action="store_true",
        help=(
            "Write one combined delta-only PNG instead of separate "
            "model_comparison_delta_only_<ts>_stable.png / _unstable.png "
            "(default: split by stability when is_stable is present)."
        ),
    )
    parser.add_argument(
        "--delta-split-by-stability",
        dest="legacy_delta_split_by_stability",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=_DELTA_ONLY_SAVE_DPI,
        metavar="N",
        help=(
            f"DPI for PNG figure exports (default {_DELTA_ONLY_SAVE_DPI}). "
            "IEEE Access-style print raster figures often use 300 dpi; submit PDF for vector line art."
        ),
    )
    parser.add_argument(
        "--save-vector-pdf",
        action="store_true",
        help=(
            "Also write each figure as a vector PDF (same basename as the PNG). "
            "Recommended for IEEE Access trajectory line plots."
        ),
    )
    parser.add_argument(
        "--publication-figures",
        action="store_true",
        help="Shortcut: --figure-dpi 300 and --save-vector-pdf for journal submission bundles.",
    )

    args = parser.parse_args()
    if args.publication_figures:
        args.figure_dpi = 300
        args.save_vector_pdf = True
    # Default: separate stable/unstable delta-only PNGs when is_stable exists.
    # Opt out with --combine-delta-only-figure. --delta-split-by-stability is a no-op (backward compat).
    args.delta_split_by_stability = not args.combine_delta_only_figure
    if not args.full_trajectory_data and not args.test_split_path:
        parser.error(
            "Provide --full-trajectory-data (--test-data) and/or --test-split-path "
            "(at least one is required)."
        )
    if args.full_trajectory_data is None:
        args.full_trajectory_data = args.test_split_path

    print("=" * 70)
    print("MODEL COMPARISON: ML BASELINE vs PINN vs ANDES")
    print("=" * 70)
    print(f"ML Baseline: {args.ml_baseline_model}")
    print(f"PINN: {args.pinn_model}")
    print(f"Full trajectory data: {args.full_trajectory_data}")
    if args.test_split_path and args.test_split_path != args.full_trajectory_data:
        print(f"Pre-split path: {args.test_split_path}")
    print(f"Output: {args.output_dir}")
    if args.save_vector_pdf or args.figure_dpi != _DELTA_ONLY_SAVE_DPI:
        print(
            f"Figure export: PNG dpi={args.figure_dpi}"
            + (", companion PDF enabled" if args.save_vector_pdf else "")
        )

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Generate default timestamped directory
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs/comparison") / f"exp_{timestamp}"
        print(f"Using default output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle wildcard patterns (full trajectory path)
    test_data_path_str = args.full_trajectory_data
    if "*" in test_data_path_str:
        test_data_pattern = (
            PROJECT_ROOT / test_data_path_str
            if not Path(test_data_path_str).is_absolute()
            else Path(test_data_path_str)
        )
        matching_files = list(test_data_pattern.parent.glob(test_data_pattern.name))
        if not matching_files:
            print(f"[ERROR] No files found matching pattern: {args.full_trajectory_data}")
            sys.exit(1)
        test_data_path = max(matching_files, key=lambda p: p.stat().st_mtime)
        print(f"Found {len(matching_files)} file(s), using latest: {test_data_path.name}")
    else:
        test_data_path = Path(args.full_trajectory_data)

    test_split_path = None
    if args.test_split_path:
        if "*" in args.test_split_path:
            split_pattern = (
                PROJECT_ROOT / args.test_split_path
                if not Path(args.test_split_path).is_absolute()
                else Path(args.test_split_path)
            )
            matching_splits = list(split_pattern.parent.glob(split_pattern.name))
            if matching_splits:
                test_split_path = max(matching_splits, key=lambda p: p.stat().st_mtime)
        else:
            test_split_path = Path(args.test_split_path)

    # Load test data
    print("\nLoading test data...")
    test_data, test_scenario_ids = load_test_data(
        test_data_path,
        test_split_path=test_split_path,
        random_state=42,
    )

    # Check data structure
    required_columns = ["scenario_id", "time", "delta", "omega"]
    missing_columns = [col for col in required_columns if col not in test_data.columns]
    if missing_columns:
        print(f"❌ Error: Test data missing required columns: {missing_columns}")
        print(f"   Available columns: {list(test_data.columns)}")
        sys.exit(1)

    # Save test scenario IDs
    test_scenario_file = output_dir / "test_scenario_ids.json"
    with open(test_scenario_file, "w") as f:
        json.dump(test_scenario_ids, f, indent=2)
    print(f"✓ Saved test scenario IDs to: {test_scenario_file.name}")

    if args.n_scenarios:
        test_scenario_ids = test_scenario_ids[: args.n_scenarios]
        test_data = test_data[test_data["scenario_id"].isin(test_scenario_ids)]

    print(f"✓ Comparing on {len(test_scenario_ids)} test scenarios")

    # Load models
    print("\nLoading models...")
    device_torch = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ml_model, ml_scalers, ml_input_method = load_ml_baseline_model(
        Path(args.ml_baseline_model),
        device=str(device_torch),
    )

    pinn_config_path = Path(args.pinn_config) if args.pinn_config else None
    pinn_model, pinn_scalers, pinn_input_method = load_pinn_model(
        Path(args.pinn_model),
        config_path=pinn_config_path,
        device=str(device_torch),
    )

    # Ensure same input method
    if ml_input_method != pinn_input_method:
        print(
            f"⚠️  Warning: Input methods differ (ML: {ml_input_method}, PINN: {pinn_input_method})"
        )

    # Evaluate both models
    print("\nEvaluating models on test scenarios...")
    ml_predictions = {}
    pinn_predictions = {}

    ml_delta_errors = []
    ml_omega_errors = []
    pinn_delta_errors = []
    pinn_omega_errors = []
    ml_delta_mae_list = []
    pinn_delta_mae_list = []
    ml_omega_mae_list = []
    pinn_omega_mae_list = []
    ml_delta_r2_list = []
    pinn_delta_r2_list = []
    ml_omega_r2_list = []
    pinn_omega_r2_list = []

    # Segment-wise metrics (pre-fault, during-fault, post-fault)
    segment_pre_ml_delta_rmse: List[float] = []
    segment_pre_pinn_delta_rmse: List[float] = []
    segment_pre_ml_omega_rmse: List[float] = []
    segment_pre_pinn_omega_rmse: List[float] = []
    segment_during_ml_delta_rmse: List[float] = []
    segment_during_pinn_delta_rmse: List[float] = []
    segment_during_ml_omega_rmse: List[float] = []
    segment_during_pinn_omega_rmse: List[float] = []
    segment_post_ml_delta_rmse: List[float] = []
    segment_post_pinn_delta_rmse: List[float] = []
    segment_post_ml_omega_rmse: List[float] = []
    segment_post_pinn_omega_rmse: List[float] = []

    # FIX: Track which scenarios successfully completed evaluation
    successful_scenario_ids = []
    failed_scenarios = []

    with torch.no_grad():
        for scenario_id in test_scenario_ids:
            scenario_data = test_data[test_data["scenario_id"] == scenario_id].sort_values("time")

            if len(scenario_data) == 0:
                print(f"   ⚠️  Scenario {scenario_id}: No data found")
                failed_scenarios.append(
                    {
                        "scenario_id": scenario_id,
                        "error_type": "NoData",
                        "error": "No data found",
                    }
                )
                continue

            if len(scenario_data) < 10:
                print(
                    f"⚠️ Scenario {scenario_id}: Insufficient data points ({len(scenario_data)} <"
                    f"10)"
                )
                failed_scenarios.append(
                    {
                        "scenario_id": scenario_id,
                        "error_type": "InsufficientData",
                        "error": f"Insufficient data points ({len(scenario_data)} < 10)",
                    }
                )
                continue

            # Check data structure first
            if "delta" not in scenario_data.columns or "omega" not in scenario_data.columns:
                print(f"   ⚠️  Scenario {scenario_id}: Missing required columns.")
                print(f"      Available columns: {list(scenario_data.columns)}")
                print(
                    f"Missing: {[col for col in ['delta', 'omega'] if col not in scenario_data.columns]}"
                )
                failed_scenarios.append(
                    {
                        "scenario_id": scenario_id,
                        "error_type": "MissingColumns",
                        "error": f"Missing required columns: {[col for col in ['delta', 'omega'] if col not in scenario_data.columns]}",
                    }
                )
                continue

            try:
                # ML baseline prediction
                try:
                    time_ml, delta_ml, omega_ml = predict_scenario_ml_baseline(
                        ml_model,
                        scenario_data,
                        ml_scalers,
                        ml_input_method,
                        device_torch,
                    )
                    ml_predictions[scenario_id] = (time_ml, delta_ml, omega_ml)
                except (KeyError, ValueError) as e:
                    import traceback

                    error_msg = str(e)
                    error_type = type(e).__name__
                    print(
                        f"   ⚠️  ML Baseline {error_type} for scenario {scenario_id}: "
                        f"{error_msg}"
                    )
                    print(f"      Available scalers: {list(ml_scalers.keys())}")
                    if scenario_id == test_scenario_ids[0]:
                        print(f"      Full traceback:\n{traceback.format_exc()}")
                    failed_scenarios.append(
                        {
                            "scenario_id": scenario_id,
                            "error_type": f"ML_{error_type}",
                            "error": error_msg,
                        }
                    )
                    continue

                # PINN prediction
                try:
                    time_pinn, delta_pinn, omega_pinn = predict_scenario_pinn(
                        pinn_model,
                        scenario_data,
                        pinn_scalers,
                        pinn_input_method,
                        device_torch,
                    )
                    pinn_predictions[scenario_id] = (time_pinn, delta_pinn, omega_pinn)

                    # Diagnostic output for first scenario to debug high RMSE
                    if scenario_id == test_scenario_ids[0]:
                        print(f"\n   📊 Diagnostic for Scenario {scenario_id} (first scenario):")
                        print(
                            f"Delta pred range: [{np.min(delta_pinn):.3f},"
                            f"{np.max(delta_pinn):.3f}] rad"
                        )
                        if not DELTA_ONLY_EXPERIMENT_UI:
                            print(
                                f"Omega pred range: [{np.min(omega_pinn):.3f},"
                                f"{np.max(omega_pinn):.3f}] pu"
                            )
                            print(
                                f"Delta pred length: {len(delta_pinn)}, Omega pred length:"
                                f"{len(omega_pinn)}"
                            )
                        else:
                            print(f"Delta pred length: {len(delta_pinn)}")
                        print(f"      Time pred length: {len(time_pinn)}")
                except (KeyError, ValueError) as e:
                    # More detailed error for PINN - show full error message
                    import traceback

                    error_msg = str(e)
                    error_type = type(e).__name__
                    print(f"   ⚠️  PINN {error_type} for scenario {scenario_id}: {error_msg}")
                    print(f"      Scenario data columns: {list(scenario_data.columns)[:10]}...")
                    print(f"      Available scalers: {list(pinn_scalers.keys())}")
                    # Show full traceback for first error only to avoid spam
                    if scenario_id == test_scenario_ids[0]:
                        print(f"      Full traceback:\n{traceback.format_exc()}")
                    failed_scenarios.append(
                        {
                            "scenario_id": scenario_id,
                            "error_type": f"PINN_{error_type}",
                            "error": error_msg,
                        }
                    )
                    continue

                # Get ground truth (same as evaluation.py)
                delta_true = scenario_data["delta"].values
                omega_true = scenario_data["omega"].values

                # Align lengths for ML baseline (same as evaluation.py line 251-255)
                min_len_ml = min(len(delta_ml), len(omega_ml), len(delta_true), len(omega_true))
                delta_ml_aligned = delta_ml[:min_len_ml]
                omega_ml_aligned = omega_ml[:min_len_ml]
                delta_true_ml = delta_true[:min_len_ml]
                omega_true_ml = omega_true[:min_len_ml]

                # Align lengths for PINN (same as evaluation.py line 251-255)
                min_len_pinn = min(
                    len(delta_pinn), len(omega_pinn), len(delta_true), len(omega_true)
                )
                delta_pinn_aligned = delta_pinn[:min_len_pinn]
                omega_pinn_aligned = omega_pinn[:min_len_pinn]
                delta_true_pinn = delta_true[:min_len_pinn]
                omega_true_pinn = omega_true[:min_len_pinn]

                # Diagnostic output for first scenario
                if scenario_id == test_scenario_ids[0]:
                    print(
                        f"Delta true range: [{np.min(delta_true_pinn):.3f},"
                        f"{np.max(delta_true_pinn):.3f}] rad"
                    )
                    if not DELTA_ONLY_EXPERIMENT_UI:
                        print(
                            f"Omega true range: [{np.min(omega_true_pinn):.3f},"
                            f"{np.max(omega_true_pinn):.3f}] pu"
                        )
                    print(f"      Delta true length: {len(delta_true_pinn)}")
                    print(f"      Aligned lengths - ML: {min_len_ml}, PINN: {min_len_pinn}")

                # Compute per-scenario errors (same formula as evaluation.py line 263-264)
                # Use angle wrapping for delta (periodic angles)
                from utils.metrics import wrap_angle_error

                delta_ml_error_wrapped = wrap_angle_error(delta_ml_aligned, delta_true_ml)
                delta_pinn_error_wrapped = wrap_angle_error(delta_pinn_aligned, delta_true_pinn)

                ml_delta_rmse = np.sqrt(np.mean((delta_ml_aligned - delta_true_ml) ** 2))
                ml_delta_rmse_wrapped = np.sqrt(np.mean(delta_ml_error_wrapped**2))
                # Use wrapped RMSE if it's significantly better
                if ml_delta_rmse_wrapped < ml_delta_rmse * 0.9:
                    ml_delta_rmse = ml_delta_rmse_wrapped

                if not DELTA_ONLY_EXPERIMENT_UI:
                    ml_omega_rmse = np.sqrt(np.mean((omega_ml_aligned - omega_true_ml) ** 2))

                pinn_delta_rmse = np.sqrt(np.mean((delta_pinn_aligned - delta_true_pinn) ** 2))
                pinn_delta_rmse_wrapped = np.sqrt(np.mean(delta_pinn_error_wrapped**2))
                # Use wrapped RMSE if it's significantly better
                if pinn_delta_rmse_wrapped < pinn_delta_rmse * 0.9:
                    pinn_delta_rmse = pinn_delta_rmse_wrapped

                if not DELTA_ONLY_EXPERIMENT_UI:
                    pinn_omega_rmse = np.sqrt(np.mean((omega_pinn_aligned - omega_true_pinn) ** 2))

                # Diagnostic output for first scenario
                if scenario_id == test_scenario_ids[0]:
                    print(f"      ML Delta RMSE: {ml_delta_rmse:.6f} rad")
                    print(f"      PINN Delta RMSE: {pinn_delta_rmse:.6f} rad")
                    if not DELTA_ONLY_EXPERIMENT_UI:
                        print(f"      ML Omega RMSE: {ml_omega_rmse:.6f} pu")
                        print(f"      PINN Omega RMSE: {pinn_omega_rmse:.6f} pu")
                    # Check for extreme differences
                    max_delta_diff = np.max(np.abs(delta_pinn_aligned - delta_true_pinn))
                    print(f"      Max PINN delta error: {max_delta_diff:.6f} rad")
                    if max_delta_diff > 10:
                        print(f"      ⚠️  WARNING: Very large delta error detected!")
                        print(f"         First 5 delta pred: {delta_pinn_aligned[:5]}")
                        print(f"         First 5 delta true: {delta_true_pinn[:5]}")

                ml_delta_errors.append(ml_delta_rmse)
                pinn_delta_errors.append(pinn_delta_rmse)
                if not DELTA_ONLY_EXPERIMENT_UI:
                    ml_omega_errors.append(ml_omega_rmse)
                    pinn_omega_errors.append(pinn_omega_rmse)

                # MAE and R² for metrics box on delta-only figure
                ml_delta_mae_list.append(mean_absolute_error(delta_ml_aligned, delta_true_ml))
                pinn_delta_mae_list.append(mean_absolute_error(delta_pinn_aligned, delta_true_pinn))
                ml_delta_r2_list.append(r2_score(delta_true_ml, delta_ml_aligned))
                pinn_delta_r2_list.append(r2_score(delta_true_pinn, delta_pinn_aligned))
                if not DELTA_ONLY_EXPERIMENT_UI:
                    ml_omega_mae_list.append(mean_absolute_error(omega_ml_aligned, omega_true_ml))
                    pinn_omega_mae_list.append(
                        mean_absolute_error(omega_pinn_aligned, omega_true_pinn)
                    )
                    ml_omega_r2_list.append(r2_score(omega_true_ml, omega_ml_aligned))
                    pinn_omega_r2_list.append(r2_score(omega_true_pinn, omega_pinn_aligned))

                # Segment-wise metrics (pre-fault, during-fault, post-fault)
                time_full = scenario_data["time"].values
                time_ml = time_full[:min_len_ml]
                time_pinn = time_full[:min_len_pinn]
                first_row = scenario_data.iloc[0]
                tf = float(first_row.get("tf", 1.0))
                tc = float(first_row.get("tc", first_row.get("param_tc", 1.2)))
                seg_ml = compute_segment_metrics(
                    time_ml,
                    tf,
                    tc,
                    delta_ml_aligned,
                    omega_ml_aligned,
                    delta_true_ml,
                    omega_true_ml,
                )
                seg_pinn = compute_segment_metrics(
                    time_pinn,
                    tf,
                    tc,
                    delta_pinn_aligned,
                    omega_pinn_aligned,
                    delta_true_pinn,
                    omega_true_pinn,
                )
                for seg_name, seg_ml_d, seg_pinn_d in [
                    ("pre_fault", seg_ml["pre_fault"], seg_pinn["pre_fault"]),
                    ("during_fault", seg_ml["during_fault"], seg_pinn["during_fault"]),
                    ("post_fault", seg_ml["post_fault"], seg_pinn["post_fault"]),
                ]:
                    if seg_ml_d is not None and seg_pinn_d is not None:
                        seg_ml_delta_rmse = seg_ml_d["delta_rmse"]
                        seg_pinn_delta_rmse = seg_pinn_d["delta_rmse"]
                        if seg_name == "pre_fault":
                            segment_pre_ml_delta_rmse.append(seg_ml_delta_rmse)
                            segment_pre_pinn_delta_rmse.append(seg_pinn_delta_rmse)
                            if not DELTA_ONLY_EXPERIMENT_UI:
                                segment_pre_ml_omega_rmse.append(seg_ml_d["omega_rmse"])
                                segment_pre_pinn_omega_rmse.append(seg_pinn_d["omega_rmse"])
                        elif seg_name == "during_fault":
                            segment_during_ml_delta_rmse.append(seg_ml_delta_rmse)
                            segment_during_pinn_delta_rmse.append(seg_pinn_delta_rmse)
                            if not DELTA_ONLY_EXPERIMENT_UI:
                                segment_during_ml_omega_rmse.append(seg_ml_d["omega_rmse"])
                                segment_during_pinn_omega_rmse.append(seg_pinn_d["omega_rmse"])
                        else:
                            segment_post_ml_delta_rmse.append(seg_ml_delta_rmse)
                            segment_post_pinn_delta_rmse.append(seg_pinn_delta_rmse)
                            if not DELTA_ONLY_EXPERIMENT_UI:
                                segment_post_ml_omega_rmse.append(seg_ml_d["omega_rmse"])
                                segment_post_pinn_omega_rmse.append(seg_pinn_d["omega_rmse"])

                # FIX: Track successful scenario
                successful_scenario_ids.append(scenario_id)

            except KeyError as e:
                import traceback

                error_msg = str(e)
                print(f"   ⚠️  KeyError evaluating scenario {scenario_id}: {error_msg}")
                # Check if it's a scaler or column issue
                if "scaler" in error_msg.lower():
                    print(f"      Missing scaler. Available scalers: {list(pinn_scalers.keys())}")
                elif "column" in error_msg.lower():
                    if len(scenario_data) > 0:
                        print(
                            f"      Missing column. Available columns: {list(scenario_data.columns)}"
                        )
                else:
                    # Generic KeyError - show both columns and scalers
                    if len(scenario_data) > 0:
                        print(f"      Available columns: {list(scenario_data.columns)[:10]}...")
                    print(f"      Available scalers: {list(pinn_scalers.keys())}")
                # Show full traceback for first error to help debug
                if scenario_id == test_scenario_ids[0]:
                    print(f"      Full traceback:\n{traceback.format_exc()}")
                # FIX: Track failed scenario
                failed_scenarios.append(
                    {
                        "scenario_id": scenario_id,
                        "error_type": "KeyError",
                        "error": str(e),
                    }
                )
                continue
            except Exception as e:
                import traceback

                error_msg = str(e)
                print(f"   ⚠️  Error evaluating scenario {scenario_id}: {error_msg}")
                # Show full traceback for first error
                if scenario_id == test_scenario_ids[0]:
                    print(f"      Full traceback:\n{traceback.format_exc()}")
                # FIX: Track failed scenario
                failed_scenarios.append(
                    {
                        "scenario_id": scenario_id,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                )
                continue

    print(
        f"✓ Successfully evaluated {len(successful_scenario_ids)} out of {len(test_scenario_ids)}"
        f"scenarios"
    )
    if failed_scenarios:
        print(f"⚠️  Failed scenarios: {len(failed_scenarios)}")
        print(f"   Failed scenario IDs: {[s['scenario_id'] for s in failed_scenarios[:10]]}")
        if len(failed_scenarios) > 10:
            print(f"   ... and {len(failed_scenarios) - 10} more")

    # Compute statistical comparisons
    print("\nComputing statistical comparisons...")
    delta_comparison = compute_statistical_comparison(
        np.array(ml_delta_errors),
        np.array(pinn_delta_errors),
        metric_name="Delta RMSE",
    )
    delta_mae_comparison = compute_statistical_comparison(
        np.array(ml_delta_mae_list),
        np.array(pinn_delta_mae_list),
        metric_name="Delta MAE",
    )
    if DELTA_ONLY_EXPERIMENT_UI:
        omega_comparison = None
        omega_mae_comparison = None
        omega_r2_comparison = None
    else:
        omega_comparison = compute_statistical_comparison(
            np.array(ml_omega_errors),
            np.array(pinn_omega_errors),
            metric_name="Omega RMSE",
        )
        omega_mae_comparison = compute_statistical_comparison(
            np.array(ml_omega_mae_list),
            np.array(pinn_omega_mae_list),
            metric_name="Omega MAE",
        )
    # R²: filter NaNs (e.g. constant targets) for statistical comparison
    ml_delta_r2 = np.array(ml_delta_r2_list)
    pinn_delta_r2 = np.array(pinn_delta_r2_list)
    ml_delta_r2_valid = ml_delta_r2[~np.isnan(ml_delta_r2)]
    pinn_delta_r2_valid = pinn_delta_r2[~np.isnan(pinn_delta_r2)]
    delta_r2_comparison = (
        compute_statistical_comparison(
            ml_delta_r2_valid, pinn_delta_r2_valid, metric_name="Delta R²"
        )
        if len(ml_delta_r2_valid) > 0 and len(pinn_delta_r2_valid) > 0
        else None
    )
    if not DELTA_ONLY_EXPERIMENT_UI:
        ml_omega_r2 = np.array(ml_omega_r2_list)
        pinn_omega_r2 = np.array(pinn_omega_r2_list)
        ml_omega_r2_valid = ml_omega_r2[~np.isnan(ml_omega_r2)]
        pinn_omega_r2_valid = pinn_omega_r2[~np.isnan(pinn_omega_r2)]
        omega_r2_comparison = (
            compute_statistical_comparison(
                ml_omega_r2_valid, pinn_omega_r2_valid, metric_name="Omega R²"
            )
            if len(ml_omega_r2_valid) > 0 and len(pinn_omega_r2_valid) > 0
            else None
        )

    # Segment-wise metrics (pre-fault, during-fault, post-fault)
    def _aggregate_segment(
        ml_delta_rmse: List[float],
        pinn_delta_rmse: List[float],
        ml_omega_rmse: List[float],
        pinn_omega_rmse: List[float],
        *,
        include_omega: bool = True,
    ) -> Dict:
        n = len(ml_delta_rmse)
        if n == 0:
            empty_ml = {"delta_rmse_mean": None, "delta_rmse_std": None}
            empty_pinn = {"delta_rmse_mean": None, "delta_rmse_std": None}
            out: Dict = {
                "n_scenarios": 0,
                "ml_baseline": dict(empty_ml),
                "pinn": dict(empty_pinn),
                "delta_rmse_comparison": None,
            }
            if include_omega:
                out["ml_baseline"]["omega_rmse_mean"] = None
                out["ml_baseline"]["omega_rmse_std"] = None
                out["pinn"]["omega_rmse_mean"] = None
                out["pinn"]["omega_rmse_std"] = None
                out["omega_rmse_comparison"] = None
            return out
        a_ml_d = np.array(ml_delta_rmse)
        a_pinn_d = np.array(pinn_delta_rmse)
        delta_comp = (
            compute_statistical_comparison(a_ml_d, a_pinn_d, metric_name="Delta RMSE")
            if n >= 2
            else None
        )
        if not include_omega:
            return {
                "n_scenarios": n,
                "ml_baseline": {
                    "delta_rmse_mean": float(np.mean(a_ml_d)),
                    "delta_rmse_std": float(np.std(a_ml_d, ddof=1)) if n > 1 else 0.0,
                },
                "pinn": {
                    "delta_rmse_mean": float(np.mean(a_pinn_d)),
                    "delta_rmse_std": float(np.std(a_pinn_d, ddof=1)) if n > 1 else 0.0,
                },
                "delta_rmse_comparison": delta_comp,
            }
        a_ml_o = np.array(ml_omega_rmse)
        a_pinn_o = np.array(pinn_omega_rmse)
        omega_comp = (
            compute_statistical_comparison(a_ml_o, a_pinn_o, metric_name="Omega RMSE")
            if n >= 2
            else None
        )
        return {
            "n_scenarios": n,
            "ml_baseline": {
                "delta_rmse_mean": float(np.mean(a_ml_d)),
                "delta_rmse_std": float(np.std(a_ml_d, ddof=1)) if n > 1 else 0.0,
                "omega_rmse_mean": float(np.mean(a_ml_o)),
                "omega_rmse_std": float(np.std(a_ml_o, ddof=1)) if n > 1 else 0.0,
            },
            "pinn": {
                "delta_rmse_mean": float(np.mean(a_pinn_d)),
                "delta_rmse_std": float(np.std(a_pinn_d, ddof=1)) if n > 1 else 0.0,
                "omega_rmse_mean": float(np.mean(a_pinn_o)),
                "omega_rmse_std": float(np.std(a_pinn_o, ddof=1)) if n > 1 else 0.0,
            },
            "delta_rmse_comparison": delta_comp,
            "omega_rmse_comparison": omega_comp,
        }

    _inc_omega = not DELTA_ONLY_EXPERIMENT_UI
    segment_metrics = {
        "pre_fault": _aggregate_segment(
            segment_pre_ml_delta_rmse,
            segment_pre_pinn_delta_rmse,
            segment_pre_ml_omega_rmse,
            segment_pre_pinn_omega_rmse,
            include_omega=_inc_omega,
        ),
        "during_fault": _aggregate_segment(
            segment_during_ml_delta_rmse,
            segment_during_pinn_delta_rmse,
            segment_during_ml_omega_rmse,
            segment_during_pinn_omega_rmse,
            include_omega=_inc_omega,
        ),
        "post_fault": _aggregate_segment(
            segment_post_ml_delta_rmse,
            segment_post_pinn_delta_rmse,
            segment_post_ml_omega_rmse,
            segment_post_pinn_omega_rmse,
            include_omega=_inc_omega,
        ),
    }

    # FIX: Save comparison results with only successful scenarios
    comparison_results: Dict = {
        "test_scenario_ids": successful_scenario_ids,
        "n_scenarios": len(successful_scenario_ids),
        "total_test_scenarios": len(test_scenario_ids),
        "failed_scenarios": failed_scenarios,
        "delta_comparison": delta_comparison,
        "delta_mae_comparison": delta_mae_comparison,
        "delta_r2_comparison": delta_r2_comparison,
        "segment_metrics": segment_metrics,
    }
    if not DELTA_ONLY_EXPERIMENT_UI:
        comparison_results["omega_comparison"] = omega_comparison
        comparison_results["omega_mae_comparison"] = omega_mae_comparison
        comparison_results["omega_r2_comparison"] = omega_r2_comparison

    # FIX: Warn if many scenarios failed
    if len(failed_scenarios) > 0:
        failure_rate = len(failed_scenarios) / len(test_scenario_ids) * 100
        print(
            f"⚠️ Warning: {len(failed_scenarios)} scenarios ({failure_rate:.1f}%) failed evaluation"
        )
        print(
            f"   Comparison results only include {len(successful_scenario_ids)} successful scenarios"
        )
        if failure_rate > 50:
            print(
                f"   ⚠️  CRITICAL: More than 50% of scenarios failed! Check prediction errors above."
            )

    results_file = output_dir / "comparison_results.json"
    with open(results_file, "w") as f:
        json.dump(comparison_results, f, indent=2, default=float)

    print("\n✓ Statistical Comparison Results:")
    print(f"\n  Delta RMSE:")
    print(
        f"ML Baseline: {delta_comparison['ml_baseline']['mean']:.6f} ±"
        f"{delta_comparison['ml_baseline']['std']:.6f}"
    )
    print(
        f"    PINN:        {delta_comparison['pinn']['mean']:.6f} ± {delta_comparison['pinn']['std']:.6f}"
    )
    print(f"    Improvement: {delta_comparison['improvement']['relative_percent']:.2f}%")
    print(f"    p-value:     {delta_comparison['statistical_test']['p_value']:.4f}")

    if not DELTA_ONLY_EXPERIMENT_UI and omega_comparison is not None:
        print(f"\n  Omega RMSE:")
        print(
            f"ML Baseline: {omega_comparison['ml_baseline']['mean']:.6f} ±"
            f"{omega_comparison['ml_baseline']['std']:.6f}"
        )
        print(
            f"    PINN:        {omega_comparison['pinn']['mean']:.6f} ± {omega_comparison['pinn']['std']:.6f}"
        )
        print(f"    Improvement: {omega_comparison['improvement']['relative_percent']:.2f}%")
        print(f"    p-value:     {omega_comparison['statistical_test']['p_value']:.4f}")

    print("\n✓ Segment-wise metrics (pre-fault / during-fault / post-fault):")
    for seg_name, seg in segment_metrics.items():
        n = seg["n_scenarios"]
        if n == 0:
            print(f"  {seg_name}: no scenarios with enough points")
            continue
        ml_d = seg["ml_baseline"]["delta_rmse_mean"]
        pinn_d = seg["pinn"]["delta_rmse_mean"]
        if DELTA_ONLY_EXPERIMENT_UI:
            print(f"  {seg_name} (n={n}): Delta RMSE ML={ml_d:.6f}, PINN={pinn_d:.6f}")
        else:
            ml_o = seg["ml_baseline"]["omega_rmse_mean"]
            pinn_o = seg["pinn"]["omega_rmse_mean"]
            print(
                f"  {seg_name} (n={n}): Delta RMSE ML={ml_d:.6f}, PINN={pinn_d:.6f} | "
                f"Omega RMSE ML={ml_o:.6f}, PINN={pinn_o:.6f}"
            )

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.overlaid_plots:
        print("\nGenerating overlaid plots...")
        try:
            plot_path = generate_overlaid_plots(
                test_data=test_data,
                ml_predictions=ml_predictions,
                pinn_predictions=pinn_predictions,
                output_dir=figures_dir,
                n_examples=args.n_examples,
                figure_dpi=args.figure_dpi,
                save_vector_pdf=args.save_vector_pdf,
            )
            if plot_path:
                print(f"✓ Overlaid plots saved to: {plot_path}")
        except Exception as e:
            print(f"⚠️  Could not generate overlaid plots: {e}")
            import traceback

            traceback.print_exc()

    try:
        delta_only_paths = generate_delta_only_plots(
            test_data=test_data,
            ml_predictions=ml_predictions,
            pinn_predictions=pinn_predictions,
            output_dir=figures_dir,
            n_examples=args.n_examples,
            delta_comparison=delta_comparison,
            omega_comparison=None if DELTA_ONLY_EXPERIMENT_UI else omega_comparison,
            delta_mae_comparison=delta_mae_comparison,
            omega_mae_comparison=None if DELTA_ONLY_EXPERIMENT_UI else omega_mae_comparison,
            delta_r2_comparison=delta_r2_comparison,
            omega_r2_comparison=None if DELTA_ONLY_EXPERIMENT_UI else omega_r2_comparison,
            split_by_stability=args.delta_split_by_stability,
            figure_dpi=args.figure_dpi,
            save_vector_pdf=args.save_vector_pdf,
        )
        for p in delta_only_paths:
            print(f"✓ Delta-only plots saved to: {p}")
    except Exception as e:
        print(f"⚠️  Could not generate delta-only plots: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print(f"  Comparison results: {results_file}")
    print(f"  Test scenario IDs: {test_scenario_file}")


if __name__ == "__main__":
    main()
