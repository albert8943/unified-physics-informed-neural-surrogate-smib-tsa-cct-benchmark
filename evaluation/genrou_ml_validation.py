"""
Evaluate a trained ML baseline (StandardNN / LSTM) against GENROU reference trajectories.

Uses the same ANDES GENROU simulation path as ``validate_pinn_on_genrou`` so PINN
and ML numbers are directly comparable for a given scenario list.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False

from evaluation.genrou_simulation import run_genrou_trajectory
from utils.metrics import compute_trajectory_metrics


def validate_ml_on_genrou(
    ml_model_path: str,
    genrou_case_file: str,
    test_scenarios: List[Dict],
    device: str = "auto",
) -> List[Dict]:
    """
    Run GENROU TDS per scenario, build a synthetic trajectory CSV row set, and
    score ``predict_scenario_ml_baseline`` against GENROU δ/ω.

    The ML checkpoint must match training (``pe_direct_7`` or ``pe_direct`` etc.);
    scalers in the checkpoint are applied the same way as in
    ``scripts/evaluate_ml_baseline.py``.
    """
    if not ANDES_AVAILABLE:
        raise ImportError("ANDES is required for GENROU validation")

    from scripts.evaluate_ml_baseline import (
        load_ml_baseline_model,
        predict_scenario_ml_baseline,
    )

    print(f"Loading ML baseline from: {ml_model_path}")
    model, scalers, input_method = load_ml_baseline_model(Path(ml_model_path), device=device)
    device_torch = next(model.parameters()).device

    print(f"GENROU case file: {genrou_case_file}")
    try:
        case_path = andes.get_case(genrou_case_file)
    except Exception:
        case_path = genrou_case_file
    print(f"Resolved case path: {case_path}")

    results: List[Dict] = []
    for i, scenario in enumerate(test_scenarios):
        print(f"\nScenario {i + 1}/{len(test_scenarios)} (ML)")
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

            n = min(len(genrou_time), len(genrou_delta), len(genrou_omega), len(pe_t))
            if n == 0:
                continue
            genrou_time = genrou_time[:n]
            genrou_delta = genrou_delta[:n]
            genrou_omega = genrou_omega[:n]
            pe_t = pe_t[:n]

            delta0 = float(scenario.get("delta0", genrou_delta[0]))
            omega0 = float(scenario.get("omega0", genrou_omega[0]))
            H = float(scenario.get("H", 6.0))
            D = float(scenario.get("D", 1.0))
            Pm = float(scenario.get("Pm", 0.8))
            tf = float(scenario.get("tf", 1.0))
            tc = float(scenario.get("tc", 1.2))

            scenario_df = pd.DataFrame(
                {
                    "time": genrou_time,
                    "Pe": pe_t,
                    "delta": genrou_delta,
                    "omega": genrou_omega,
                    "delta0": delta0,
                    "omega0": omega0,
                    "H": H,
                    "D": D,
                    "Pm": Pm,
                    "tf": tf,
                    "tc": tc,
                }
            )

            _, ml_delta, ml_omega = predict_scenario_ml_baseline(
                model, scenario_df, scalers, input_method, device_torch
            )

            min_len = min(len(genrou_delta), len(ml_delta))
            if min_len == 0:
                continue
            genrou_delta = genrou_delta[:min_len]
            genrou_omega = genrou_omega[:min_len]
            ml_delta = ml_delta[:min_len]
            ml_omega = ml_omega[:min_len]

            genrou_delta = np.arctan2(np.sin(genrou_delta), np.cos(genrou_delta))
            ml_delta = np.arctan2(np.sin(ml_delta), np.cos(ml_delta))

            if np.any(np.isnan(genrou_delta)) or np.any(np.isnan(ml_delta)):
                print("  ⚠️  Skipping scenario (NaN in trajectories)")
                continue

            metrics = compute_trajectory_metrics(
                delta_pred=ml_delta,
                omega_pred=ml_omega,
                delta_true=genrou_delta,
                omega_true=genrou_omega,
            )
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
