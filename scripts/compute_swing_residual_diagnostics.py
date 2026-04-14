#!/usr/bin/env python
"""
Compute swing-equation residual magnitudes |r_f| for PINN vs ML on a fixed trajectory split.

Definition (SMIB, algebraically equivalent to the acceleration form used in training when
``dδ/dt = 2π f_n (ω-1)`` holds; see ``evaluation.swing_residual_diagnostics``):

    r_f(t) = dω/dt - (1/(M·ω_syn))·(P_m - P_e(t) - D·ω_syn·(ω-1)),
    M = 2H,  ω_syn = 2π f_n.

This ω-ODE form avoids double differencing ``δ`` on a fixed grid; it matches the structure
implied by differentiating ``M δ̈ + D δ̇ = P_m - P_e`` under the training angle–speed map.
``P_e`` is the ANDES ``Pe`` column (``pe_direct_7`` input). ``dω/dt`` uses ``numpy.gradient``.

Example (independent val-gate Phase 4 test split; adjust checkpoint paths):

    python scripts/compute_swing_residual_diagnostics.py ^
      --pinn-model outputs/campaign_indep_pinn/pinn_lp_05/pinn/model/best_model_*.pth ^
      --ml-baseline-model outputs/campaign_indep_ml/ml_uw2_ow50/ml_baseline/standard_nn/model/model.pth ^
      --test-split-path data/processed/exp_20260211_190612/test_data_20260211_190612.csv ^
      --output-json docs/publication/pre_specs/swing_residual_diag_valgate_test.json

Optional: ``--time-mask postfault`` evaluates only t ≥ tf (excludes prefault steady segment).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Windows UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from evaluation.swing_residual_diagnostics import (
    SwingResidualStats,
    aggregate_mean_of_scenario_means,
    aggregate_pooled,
    apply_time_mask,
    compute_swing_residual_rf,
    stats_abs,
)
from scripts.compare_models import (
    load_ml_baseline_model,
    load_pinn_model,
    load_test_data,
    predict_scenario_ml_baseline,
    predict_scenario_pinn,
    resolve_trajectory_csv,
)


def _prepare_scenario_df(raw: pd.DataFrame) -> pd.DataFrame:
    s = raw.sort_values("time").copy()
    if "time" in s.columns:
        s = s.drop_duplicates(subset=["time"], keep="last")
    return s


def _scenario_params(first: pd.Series) -> Tuple[float, float, float, float]:
    H = float(first.get("param_H", first.get("H", 5.0)))
    D = float(first.get("param_D", first.get("D", 1.0)))
    Pm = float(first.get("param_Pm", first.get("Pm", 0.8)))
    tf = float(first.get("tf", 1.0))
    return H, D, Pm, tf


def _run_model_on_scenario(
    name: str,
    model: torch.nn.Module,
    scalers: Dict,
    input_method: str,
    device: torch.device,
    scenario: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name == "pinn":
        t, d, w = predict_scenario_pinn(model, scenario, scalers, input_method, device)
    elif name == "ml":
        t, d, w = predict_scenario_ml_baseline(model, scenario, scalers, input_method, device)
    else:
        raise ValueError(name)
    return t, d, w


def evaluate_split(
    test_data: pd.DataFrame,
    device: torch.device,
    pinn_bundle: Optional[Tuple] = None,
    ml_bundle: Optional[Tuple] = None,
    fn_hz: float = 60.0,
    time_mask: str = "all",
    include_andes: bool = False,
    min_points: int = 4,
) -> Dict:
    scenario_ids = sorted(test_data["scenario_id"].unique().tolist())

    pinn_abs_chunks: List[np.ndarray] = []
    ml_abs_chunks: List[np.ndarray] = []
    andes_abs_chunks: List[np.ndarray] = []

    pinn_per_scen: List[SwingResidualStats] = []
    ml_per_scen: List[SwingResidualStats] = []
    andes_per_scen: List[SwingResidualStats] = []

    skipped: List[Dict] = []

    for sid in scenario_ids:
        raw = test_data[test_data["scenario_id"] == sid]
        scenario = _prepare_scenario_df(raw)
        if "Pe" not in scenario.columns:
            raise KeyError("Column 'Pe' is required for pe_direct-style residuals.")

        first = scenario.iloc[0]
        H, D, Pm, tf = _scenario_params(first)
        t_full = scenario["time"].values.astype(np.float64)
        Pe_full = scenario["Pe"].values.astype(np.float64)
        mask = apply_time_mask(t_full, tf, time_mask)  # type: ignore[arg-type]
        if int(np.sum(mask)) < min_points:
            skipped.append({"scenario_id": int(sid), "reason": "too_few_points_after_mask"})
            continue

        t = t_full[mask]
        Pe = Pe_full[mask]

        if include_andes:
            w_true = scenario["omega"].values.astype(np.float64)[mask]
            rf_andes = compute_swing_residual_rf(t, w_true, H, D, Pm, Pe, fn_hz=fn_hz)
            st = stats_abs(rf_andes)
            andes_per_scen.append(st)
            andes_abs_chunks.append(np.abs(rf_andes))

        if pinn_bundle is not None:
            model, scalers, input_method = pinn_bundle
            _, _d_pred, w_pred = _run_model_on_scenario(
                "pinn", model, scalers, input_method, device, scenario
            )
            if len(w_pred) != len(t_full):
                skipped.append({"scenario_id": int(sid), "reason": "pinn_length_mismatch"})
            else:
                w_m = w_pred[mask]
                rf_p = compute_swing_residual_rf(t, w_m, H, D, Pm, Pe, fn_hz=fn_hz)
                stp = stats_abs(rf_p)
                pinn_per_scen.append(stp)
                pinn_abs_chunks.append(np.abs(rf_p))

        if ml_bundle is not None:
            model, scalers, input_method = ml_bundle
            _, _d_pred, w_pred = _run_model_on_scenario(
                "ml", model, scalers, input_method, device, scenario
            )
            if len(w_pred) != len(t_full):
                skipped.append({"scenario_id": int(sid), "reason": "ml_length_mismatch"})
            else:
                w_m = w_pred[mask]
                rf_m = compute_swing_residual_rf(t, w_m, H, D, Pm, Pe, fn_hz=fn_hz)
                stm = stats_abs(rf_m)
                ml_per_scen.append(stm)
                ml_abs_chunks.append(np.abs(rf_m))

    out: Dict = {
        "n_scenarios_requested": len(scenario_ids),
        "n_scenarios_in_aggregate": {
            "pinn": len(pinn_per_scen),
            "ml_baseline": len(ml_per_scen),
            "andes_truth": len(andes_per_scen),
        },
        "skipped": skipped,
        "time_mask": time_mask,
        "fn_hz": fn_hz,
        "equation": "r_f = domega_dt - (Pm - Pe - D*omega_syn*(omega-1))/(M*omega_syn); M=2H; omega_syn=2*pi*fn",
    }

    def pack_pooled(per_scen: List[SwingResidualStats], chunks: List[np.ndarray]) -> Dict:
        pooled = aggregate_pooled(chunks)
        m_mean, m_std = aggregate_mean_of_scenario_means(per_scen)
        return {
            "pooled_mean_abs": pooled.mean_abs,
            "pooled_std_abs": pooled.std_abs,
            "pooled_max_abs": pooled.max_abs,
            "pooled_n_points": pooled.n_points,
            "mean_of_scenario_means_abs": m_mean,
            "std_across_scenario_means_abs": m_std,
            "n_scenarios_in_aggregate": len(per_scen),
        }

    if pinn_bundle is not None and pinn_abs_chunks:
        out["pinn"] = pack_pooled(pinn_per_scen, pinn_abs_chunks)
    if ml_bundle is not None and ml_abs_chunks:
        out["ml_baseline"] = pack_pooled(ml_per_scen, ml_abs_chunks)
    if include_andes and andes_abs_chunks:
        out["andes_truth"] = pack_pooled(andes_per_scen, andes_abs_chunks)

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Swing residual |r_f| diagnostics for PINN vs ML.")
    p.add_argument("--test-split-path", type=Path, required=True, help="Pre-split test CSV.")
    p.add_argument("--pinn-model", type=Path, default=None, help="PINN checkpoint (.pth).")
    p.add_argument("--ml-baseline-model", type=Path, default=None, help="ML baseline .pth.")
    p.add_argument(
        "--pinn-config",
        type=Path,
        default=None,
        help="Optional YAML config for PINN metadata (else inferred from checkpoint).",
    )
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--fn-hz", type=float, default=60.0, help="Nominal frequency for dδ/dt.")
    p.add_argument(
        "--time-mask",
        choices=["all", "postfault"],
        default="all",
        help="all: every time sample; postfault: only t >= tf.",
    )
    p.add_argument(
        "--include-andes-truth",
        action="store_true",
        help="Also compute residuals using observed ω from the CSV (sanity / discretization floor).",
    )
    p.add_argument("--output-json", type=Path, default=None, help="Write full JSON summary.")
    p.add_argument(
        "--min-points", type=int, default=4, help="Skip scenario if fewer samples after mask."
    )
    args = p.parse_args()

    if args.pinn_model is None and args.ml_baseline_model is None and not args.include_andes_truth:
        p.error(
            "Provide --pinn-model and/or --ml-baseline-model, or pass --include-andes-truth alone."
        )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_torch = torch.device(device)

    split_path = resolve_trajectory_csv(Path(args.test_split_path))
    test_data, scenario_ids = load_test_data(
        data_path=split_path,
        test_split_path=split_path,
    )
    _ = scenario_ids

    pinn_b = None
    if args.pinn_model is not None:
        model, scalers, im = load_pinn_model(
            Path(args.pinn_model), config_path=args.pinn_config, device=device
        )
        pinn_b = (model, scalers, im)

    ml_b = None
    if args.ml_baseline_model is not None:
        model, scalers, im = load_ml_baseline_model(Path(args.ml_baseline_model), device=device)
        ml_b = (model, scalers, im)

    result = evaluate_split(
        test_data=test_data,
        device=device_torch,
        pinn_bundle=pinn_b,
        ml_bundle=ml_b,
        fn_hz=args.fn_hz,
        time_mask=args.time_mask,
        include_andes=args.include_andes_truth,
        min_points=args.min_points,
    )

    result["meta"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "test_split_path": str(split_path.resolve()),
        "pinn_model": str(Path(args.pinn_model).resolve()) if args.pinn_model else None,
        "ml_baseline_model": str(Path(args.ml_baseline_model).resolve())
        if args.ml_baseline_model
        else None,
    }

    # Compact table-friendly copy (matches manuscript table: mean / std / max of |r_f|)
    table = {}
    for key in ("pinn", "ml_baseline", "andes_truth"):
        block = result.get(key)
        if not block:
            continue
        table[key] = {
            "mean_abs_rf": block["pooled_mean_abs"],
            "std_abs_rf": block["pooled_std_abs"],
            "max_abs_rf": block["pooled_max_abs"],
        }

    result["table"] = table

    print(json.dumps(table, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
