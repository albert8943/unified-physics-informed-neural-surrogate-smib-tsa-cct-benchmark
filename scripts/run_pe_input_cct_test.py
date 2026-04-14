#!/usr/bin/env python
"""
Pe-input (pe_direct_7) evaluation on a fixed test split for:

1. **CCT** — binary search (estimate_cct_binary_search) with Pe(t) warped per
   candidate clearing time.
2. **Stability at nominal clearing** — one forward pass per scenario using the
   simulated clearing time ``tc`` from the CSV. The time grid is **each scenario’s
   CSV ``time`` column** (same span as trajectory / ``compare_models`` figures), not
   a fixed ``0 … simulation_time`` grid. ``check_stability`` uses ``--stability-mode``
   (default ``persistence_fraction``: ≥ ``--persistence-violation-fraction`` of samples
   with |δ| ≥ π in some ``--persistence-window-seconds`` interval; optional ``terminal``,
   ``global_max``, or ``final_window``). Labels: ``is_stable`` or, if absent,
   ``is_stable_from_cct``.

   The CCT binary-search inner loop uses the **same per-scenario CSV ``time`` axis**
   by default (``--cct-inner-time-grid scenario_csv_time``), matching nominal screening.
   Use ``--cct-inner-time-grid linspace`` with ``--simulation-time`` and
   ``--t-eval-steps`` to reproduce the older uniform ``[0,T]`` inner grid.

Typical invocation from repo root (paths are examples; PINN* should match the headline
trajectory checkpoint, e.g. plain MLP retrain):

    python scripts/run_pe_input_cct_test.py ^
      --test-csv data/processed/exp_20260211_190612/test_data_20260211_190612.csv ^
      --pinn-model outputs/expt_residual_backbone_retrain_20260407/pinn_nores_lp05/pinn/best_model_20260407_172450.pth ^
      --pinn-config outputs/expt_residual_backbone_retrain_20260407/pinn_nores_lp05/config.yaml ^
      --ml-model outputs/campaign_indep_ml/ml_uw2_ow50/ml_baseline/standard_nn/model/model.pth ^
      --output-dir outputs/campaign_indep_cct_pe_input_plain_pinn_20260408

After results exist, paste rows into the paper without re-running models::

    python scripts/run_pe_input_cct_test.py --from-json path/to/pe_input_cct_test_results.json

Or append ``--print-latex-tables`` to a full run to print the same snippets.

Figures (confusion matrices; CCT parity); write PNGs next to access.tex::

    python scripts/plot_pe_input_cct_paper_figures.py --json path/to/pe_input_cct_test_results.json ^
      --output-dir "paper_writing/IEEE Access Template/figures/pe_input_cct"

Use ``\\includegraphics{figures/pe_input_cct/<stem>.png}`` from ``text/5_Results...``
(paths relative to ``access.tex``).

JSON extras: ``n_prediction_errors`` (nominal stability), ``n_cct_prediction_errors`` (CCT block),
``mean_abs_rel_error`` (mean of ``abs_error_s / cct_true`` on successful scenarios),
and ``estimate_cct_binary_search`` info now uses bracket-width ``converged`` plus ``bracket_width``.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.compare_models import load_pinn_model  # noqa: E402
from scripts.core.evaluation import _extract_and_normalize_scenario_data  # noqa: E402
from scripts.evaluate_ml_baseline import load_ml_baseline_model  # noqa: E402
from evaluation.baselines.ml_baselines import LSTMModel  # noqa: E402
from utils.cct_binary_search import estimate_cct_binary_search  # noqa: E402
from utils.normalization import denormalize_array, normalize_array  # noqa: E402
from utils.pe_hypothetical_profile import pe_profile_for_hypothetical_clearing  # noqa: E402
from utils.stability_checker import check_stability  # noqa: E402


def _parse_persistence_fraction(value: str) -> float:
    x = float(value)
    if x <= 0 or x > 1.0:
        raise argparse.ArgumentTypeError(
            f"--persistence-violation-fraction must be in (0, 1], got {value!r}"
        )
    return x


def _parse_positive_float(value: str) -> float:
    x = float(value)
    if x <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive float, got {value!r}")
    return x


def _parse_stability_mode(value: str) -> str:
    """CLI normalizer: global_max | terminal | final_window | persistence_fraction."""
    v = value.strip().lower().replace("-", "_")
    allowed = ("global_max", "terminal", "final_window", "persistence_fraction")
    if v not in allowed:
        raise argparse.ArgumentTypeError(f"stability mode must be one of {allowed}, got {value!r}")
    return v


def _parse_cct_inner_time_grid(value: str) -> str:
    """CLI normalizer: scenario_csv_time | linspace."""
    v = value.strip().lower().replace("-", "_")
    allowed = ("scenario_csv_time", "linspace")
    if v not in allowed:
        raise argparse.ArgumentTypeError(
            f"--cct-inner-time-grid must be one of {allowed}, got {value!r}"
        )
    return v


def _ml_row_features(
    scalers: Dict[str, Any],
    t_i: float,
    pe_i: float,
    delta0: float,
    omega0: float,
    H: float,
    D: float,
    Pm: float,
) -> List[float]:
    use_fixed = "delta_fixed_scale" in scalers and "omega_fixed_scale" in scalers
    feats: List[float] = [float(scalers["time"].transform([[float(t_i)]])[0, 0])]
    if use_fixed:
        feats.append(delta0 / float(scalers["delta_fixed_scale"]))
        feats.append(omega0 / float(scalers["omega_fixed_scale"]))
    elif "delta0" in scalers and "omega0" in scalers:
        feats.append(float(scalers["delta0"].transform([[delta0]])[0, 0]))
        feats.append(float(scalers["omega0"].transform([[omega0]])[0, 0]))
    elif "delta" in scalers and "omega" in scalers:
        feats.append(float(scalers["delta"].transform([[delta0]])[0, 0]))
        feats.append(float(scalers["omega"].transform([[omega0]])[0, 0]))
    else:
        raise KeyError("ML scalers need fixed-scale or delta0/omega0 or delta/omega")
    feats.append(float(scalers["H"].transform([[H]])[0, 0]))
    feats.append(float(scalers["D"].transform([[D]])[0, 0]))
    feats.append(float(scalers["Pm"].transform([[Pm]])[0, 0]))
    feats.append(float(scalers["Pe"].transform([[float(pe_i)]])[0, 0]))
    return feats


def _try_git_rev() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except OSError:
        pass
    return None


class PeDirect7CCTWrapper:
    """
    Adapts pe_direct_7 PINN or StandardNN to the reactance-style ``predict`` API
    expected by ``estimate_cct_binary_search`` (Xprefault/Xfault/Xpostfault ignored).
    """

    def __init__(
        self,
        backend: str,
        model: torch.nn.Module,
        scalers: Dict[str, Any],
        scenario_df: pd.DataFrame,
        input_method: str,
        device: str,
    ):
        if backend not in ("pinn", "ml"):
            raise ValueError("backend must be 'pinn' or 'ml'")
        self.backend = backend
        self.model = model
        self.scalers = scalers
        self.scenario_df = scenario_df.sort_values("time").reset_index(drop=True)
        self.input_method = input_method
        self.device = device
        self.t_ref = self.scenario_df["time"].values.astype(np.float64)
        self.pe_ref = self.scenario_df["Pe"].values.astype(np.float64)
        row = self.scenario_df.iloc[0]
        self.tc_ref = float(row.get("tc", row.get("param_tc", 1.2)))
        self.tf_ref = float(row.get("tf", 1.0))

    def predict(
        self,
        t,
        delta0: float,
        omega0: float,
        H: float,
        D: float,
        Pm: float,
        Xprefault: float,
        Xfault: float,
        Xpostfault: float,
        tf: float,
        tc: float,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        t_eval = np.asarray(t, dtype=np.float64).reshape(-1)
        pe_hyp = pe_profile_for_hypothetical_clearing(
            t_eval, self.t_ref, self.pe_ref, float(tf), self.tc_ref, float(tc)
        )
        dev = torch.device(device)
        if self.backend == "pinn":
            return self._predict_pinn(t_eval, pe_hyp, dev)
        return self._predict_ml(
            t_eval, pe_hyp, float(delta0), float(omega0), float(H), float(D), float(Pm), dev
        )

    def _predict_pinn(
        self, t_eval: np.ndarray, pe_hyp: np.ndarray, dev: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        norm = _extract_and_normalize_scenario_data(self.scenario_df, self.scalers, dev)
        t_tensor = torch.tensor(
            normalize_array(t_eval.astype(np.float32), self.scalers["time"]),
            dtype=torch.float32,
            device=dev,
        )
        pe_tensor = torch.tensor(
            normalize_array(pe_hyp.astype(np.float32), self.scalers["Pe"]),
            dtype=torch.float32,
            device=dev,
        )
        alpha = norm.get("Pload", norm["Pm"])
        with torch.no_grad():
            d_norm, w_norm = self.model.predict_trajectory(
                t=t_tensor,
                delta0=norm["delta0"],
                omega0=norm["omega0"],
                H=norm["H"],
                D=norm["D"],
                alpha=alpha,
                Pe=pe_tensor,
                tf=norm.get("tf"),
                tc=norm.get("tc"),
            )
        delta = denormalize_array(d_norm.detach().cpu().numpy().flatten(), self.scalers["delta"])
        omega = denormalize_array(w_norm.detach().cpu().numpy().flatten(), self.scalers["omega"])
        return delta, omega

    def _predict_ml(
        self,
        t_eval: np.ndarray,
        pe_hyp: np.ndarray,
        delta0: float,
        omega0: float,
        H: float,
        D: float,
        Pm: float,
        dev: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.input_method != "pe_direct_7":
            raise ValueError("This script only supports pe_direct_7 for ML CCT")
        scalers = self.scalers
        rows: List[List[float]] = [
            _ml_row_features(scalers, float(t_i), float(pe_i), delta0, omega0, H, D, Pm)
            for t_i, pe_i in zip(t_eval, pe_hyp)
        ]

        X = torch.tensor(rows, dtype=torch.float32, device=dev)
        if isinstance(self.model, LSTMModel):
            X = X.unsqueeze(1)
        with torch.no_grad():
            pred = self.model(X)
            if isinstance(self.model, LSTMModel):
                pred = pred.squeeze(1)
        dn = pred[:, 0].cpu().numpy()
        wn = pred[:, 1].cpu().numpy()
        if "delta_fixed_scale" in scalers and "omega_fixed_scale" in scalers:
            delta_pred = dn * float(scalers["delta_fixed_scale"])
            omega_pred = wn * float(scalers["omega_fixed_scale"])
        elif "delta" in scalers and "omega" in scalers:
            delta_pred = scalers["delta"].inverse_transform(dn.reshape(-1, 1)).flatten()
            omega_pred = scalers["omega"].inverse_transform(wn.reshape(-1, 1)).flatten()
        else:
            delta_pred, omega_pred = dn, wn

        # Match compare_models / evaluate_ml_baseline IC enforcement at t_min
        if len(delta_pred) > 0:
            min_idx = int(np.argmin(np.abs(t_eval - np.min(t_eval))))
            delta_pred = np.array(delta_pred, copy=True)
            omega_pred = np.array(omega_pred, copy=True)
            delta_pred[min_idx] = float(self.scenario_df["delta"].iloc[0])
            omega_pred[min_idx] = float(self.scenario_df["omega"].iloc[0])
        return delta_pred, omega_pred


def _scenario_cct_ground_truth(first_row: pd.Series) -> Optional[float]:
    for col in ("param_cct_absolute", "cct", "CCT", "cct_absolute"):
        if col in first_row.index:
            v = first_row.get(col)
            if v is not None and not pd.isna(v):
                return float(v)
    return None


def _coerce_bool_label(v: Any) -> Optional[bool]:
    """Parse CSV / pandas scalar into bool; avoid ``bool('False') is True``."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (ValueError, TypeError):
        pass
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        iv = int(v)
        if iv == 0:
            return False
        if iv == 1:
            return True
        return None
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        if fv == 0.0:
            return False
        if fv == 1.0:
            return True
        return None
    return None


def _nominal_stability_time_grid(sdf: pd.DataFrame) -> np.ndarray:
    """
    Time axis for nominal-clearing evaluation: ANDES/CSV stamps only.

    Matches trajectory figures that align predictions to ``scenario_data['time']``
    (not a uniform 0…T_sim grid).
    """
    return sdf["time"].values.astype(np.float64)


def _ground_truth_stable_label(first_row: pd.Series) -> Optional[bool]:
    """ANDES / dataset label for stability at the simulated clearing in this row."""
    for col in ("is_stable", "is_stable_from_cct"):
        if col not in first_row.index:
            continue
        parsed = _coerce_bool_label(first_row[col])
        if parsed is not None:
            return parsed
    return None


def _binary_classification_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Any]:
    """Positive class = stable (True)."""
    n = len(y_true)
    if n == 0:
        return {
            "n": 0,
            "accuracy": None,
            "precision_stable": None,
            "recall_stable": None,
            "f1_stable": None,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }
    yt = np.array(y_true, dtype=bool)
    yp = np.array(y_pred, dtype=bool)
    acc = float(np.mean(yt == yp))
    tp = int(np.sum(yt & yp))
    tn = int(np.sum(~yt & ~yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "n": n,
        "accuracy": acc,
        "precision_stable": prec,
        "recall_stable": rec,
        "f1_stable": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_stability_at_nominal_clearing(
    label: str,
    backend: str,
    model: torch.nn.Module,
    scalers: Dict[str, Any],
    input_method: str,
    test_df: pd.DataFrame,
    scenario_ids: List[int],
    device: str,
    delta_threshold: float,
    stability_mode: str = "persistence_fraction",
    final_window_seconds: float = 0.25,
    persistence_window_seconds: float = 0.25,
    persistence_violation_fraction: float = 0.9,
) -> Dict[str, Any]:
    """
    Classify stable vs unstable at the **nominal** clearing time in each scenario row.

    Uses ``check_stability`` with ``stability_mode`` and related kwargs. Time samples
    are **per-scenario CSV ``time``** (same as trajectory plots),
    not ``np.linspace(0, simulation_time, n)``.
    """
    y_true: List[bool] = []
    y_pred: List[bool] = []
    per_scenario: List[Dict[str, Any]] = []

    for sid in scenario_ids:
        sdf = test_df[test_df["scenario_id"] == sid].copy()
        if len(sdf) < 2:
            continue
        sdf = sdf.sort_values("time")
        t_eval = _nominal_stability_time_grid(sdf)
        row0 = sdf.iloc[0]
        gt = _ground_truth_stable_label(row0)
        if gt is None:
            continue
        tf = float(row0.get("tf", 1.0))
        tc_nom = float(row0.get("tc", row0.get("param_tc", 1.2)))
        delta0 = float(row0.get("delta0", sdf["delta"].iloc[0]))
        omega0 = float(row0.get("omega0", sdf["omega"].iloc[0]))
        H = float(row0.get("param_H", row0.get("H", 5.0)))
        D = float(row0.get("param_D", row0.get("D", 1.0)))
        Pm = float(row0.get("param_Pm", row0.get("Pm", 0.8)))

        wrap = PeDirect7CCTWrapper(
            backend=backend,
            model=model,
            scalers=scalers,
            scenario_df=sdf,
            input_method=input_method,
            device=device,
        )
        try:
            delta_pred, omega_pred = wrap.predict(
                t_eval,
                delta0,
                omega0,
                H,
                D,
                Pm,
                0.5,
                1e-4,
                0.5,
                tf,
                tc_nom,
                device=device,
            )
            pred_stable = bool(
                check_stability(
                    delta_pred,
                    omega_pred,
                    delta_threshold=delta_threshold,
                    omega_threshold=10.0,
                    stability_mode=stability_mode,
                    time=t_eval,
                    final_window_seconds=final_window_seconds,
                    persistence_window_seconds=persistence_window_seconds,
                    persistence_violation_fraction=persistence_violation_fraction,
                )
            )
            y_true.append(gt)
            y_pred.append(pred_stable)
            per_scenario.append(
                {
                    "scenario_id": int(sid),
                    "tc_nominal_s": float(tc_nom),
                    "t_end_s": float(t_eval[-1]) if len(t_eval) else None,
                    "n_time_samples": int(len(t_eval)),
                    "stable_label": gt,
                    "stable_pred": pred_stable,
                    "correct": gt == pred_stable,
                }
            )
        except Exception as e:
            per_scenario.append(
                {
                    "scenario_id": int(sid),
                    "tc_nominal_s": float(tc_nom),
                    "stable_label": gt,
                    "error": str(e),
                }
            )

    metrics = _binary_classification_metrics(y_true, y_pred)
    n_err = sum(1 for r in per_scenario if "error" in r)
    return {
        "label": label,
        "backend": backend,
        "task": "stability_at_nominal_clearing_time",
        "positive_class_is_stable": True,
        "nominal_time_grid": "scenario_csv_time",
        "delta_threshold_rad": float(delta_threshold),
        "stability_mode": stability_mode,
        "final_window_seconds": float(final_window_seconds),
        "persistence_window_seconds": float(persistence_window_seconds),
        "persistence_violation_fraction": float(persistence_violation_fraction),
        "n_prediction_errors": int(n_err),
        "metrics": metrics,
        "per_scenario": per_scenario,
    }


def run_one_checkpoint(
    label: str,
    backend: str,
    model: torch.nn.Module,
    scalers: Dict[str, Any],
    input_method: str,
    test_df: pd.DataFrame,
    scenario_ids: List[int],
    device: str,
    simulation_time: float,
    t_eval_steps: int,
    cct_inner_time_grid: str,
    search_low_pad: float,
    search_high_pad: float,
    tolerance: float,
    max_iterations: int,
    delta_threshold: float,
    stability_mode: str = "persistence_fraction",
    final_window_seconds: float = 0.25,
    persistence_window_seconds: float = 0.25,
    persistence_violation_fraction: float = 0.9,
) -> Dict[str, Any]:
    per_scenario: List[Dict[str, Any]] = []
    errors_abs: List[float] = []

    for sid in scenario_ids:
        sdf = test_df[test_df["scenario_id"] == sid].copy()
        if len(sdf) < 2:
            continue
        sdf = sdf.sort_values("time")
        row0 = sdf.iloc[0]
        cct_true = _scenario_cct_ground_truth(row0)
        if cct_true is None:
            continue
        if cct_inner_time_grid == "scenario_csv_time":
            t_eval = _nominal_stability_time_grid(sdf)
        else:
            t_eval = np.linspace(0.0, simulation_time, int(t_eval_steps))
        tf = float(row0.get("tf", 1.0))
        delta0 = float(row0.get("delta0", sdf["delta"].iloc[0]))
        omega0 = float(row0.get("omega0", sdf["omega"].iloc[0]))
        H = float(row0.get("param_H", row0.get("H", 5.0)))
        D = float(row0.get("param_D", row0.get("D", 1.0)))
        Pm = float(row0.get("param_Pm", row0.get("Pm", 0.8)))

        wrap = PeDirect7CCTWrapper(
            backend=backend,
            model=model,
            scalers=scalers,
            scenario_df=sdf,
            input_method=input_method,
            device=device,
        )

        try:
            cct_est, info = estimate_cct_binary_search(
                trajectory_model=wrap,
                delta0=delta0,
                omega0=omega0,
                H=H,
                D=D,
                Pm=Pm,
                Xprefault=0.5,
                Xfault=1e-4,
                Xpostfault=0.5,
                tf=tf,
                t_eval=t_eval,
                low=tf + search_low_pad,
                high=tf + search_high_pad,
                tolerance=tolerance,
                max_iterations=max_iterations,
                delta_threshold=delta_threshold,
                omega_threshold=10.0,
                stability_mode=stability_mode,
                final_window_seconds=final_window_seconds,
                persistence_window_seconds=persistence_window_seconds,
                persistence_violation_fraction=persistence_violation_fraction,
                device=device,
                verbose=False,
            )
            err = abs(float(cct_est) - float(cct_true))
            errors_abs.append(err)
            bracket_low = float(tf + search_low_pad)
            bracket_high = float(tf + search_high_pad)
            per_scenario.append(
                {
                    "scenario_id": int(sid),
                    "cct_true": float(cct_true),
                    "cct_est": float(cct_est),
                    "abs_error_s": float(err),
                    "converged": bool(info.get("converged")),
                    "n_time_inner": int(len(t_eval)),
                    "tf_s": float(tf),
                    "bracket_low_s": bracket_low,
                    "bracket_high_s": bracket_high,
                }
            )
        except Exception as e:
            per_scenario.append(
                {
                    "scenario_id": int(sid),
                    "cct_true": float(cct_true),
                    "error": str(e),
                }
            )

    n_ok = len(errors_abs)
    n_fail = sum(1 for r in per_scenario if "error" in r)
    thr_ms = 0.005
    pct_5ms = float(100.0 * np.mean(np.array(errors_abs) < thr_ms)) if errors_abs else None
    rel_ratios: List[float] = []
    for r in per_scenario:
        if "abs_error_s" not in r or "cct_true" not in r:
            continue
        den = float(r["cct_true"])
        if den <= 0.0:
            continue
        rel_ratios.append(float(r["abs_error_s"]) / den)
    mean_abs_rel = float(np.mean(rel_ratios)) if rel_ratios else None
    summary = {
        "label": label,
        "backend": backend,
        "cct_inner_time_grid": cct_inner_time_grid,
        "stability_mode": stability_mode,
        "final_window_seconds": float(final_window_seconds),
        "persistence_window_seconds": float(persistence_window_seconds),
        "persistence_violation_fraction": float(persistence_violation_fraction),
        "n_scenarios_attempted": len(per_scenario),
        "n_scenarios_with_error_metric": n_ok,
        "n_cct_prediction_errors": int(n_fail),
        "mae_s": float(np.mean(errors_abs)) if errors_abs else None,
        "rmse_s": float(np.sqrt(np.mean(np.square(errors_abs)))) if errors_abs else None,
        "mean_abs_rel_error": mean_abs_rel,
        "pct_under_5ms": pct_5ms,
        "per_scenario": per_scenario,
    }
    return summary


def _latex_cell_metric(val: Any, *, kind: str = "float3") -> str:
    """Format one table cell; ``---`` if missing or NaN."""
    if val is None:
        return "---"
    try:
        x = float(val)
    except (TypeError, ValueError):
        return "---"
    if np.isnan(x):
        return "---"
    if kind == "float3":
        return f"{x:.3f}"
    if kind == "float2":
        return f"{x:.2f}"
    if kind == "float1":
        return f"{x:.1f}"
    if kind == "int":
        return str(int(round(x)))
    return str(x)


def format_latex_table_rows(results: Dict[str, Any]) -> str:
    """
    LaTeX ``tabular`` rows for Table~\\ref{tab:stability_nominal} and
    Table~\\ref{tab:cct_primary} (PINN* / Std NN* only; paste into
    ``5_Results_and_Discussion.tex``).
    """
    lines: List[str] = []
    lines.append("% --- Table~\\ref{tab:stability_nominal} (replace PINN* / Std NN* rows) ---")
    stab = results.get("stability_at_nominal_tc") or {}
    for row_label, key in (("PINN*", "pinn"), ("Std NN*", "ml")):
        block = stab.get(key) or {}
        m = block.get("metrics") or {}
        acc = _latex_cell_metric(m.get("accuracy"), kind="float3")
        prec = _latex_cell_metric(m.get("precision_stable"), kind="float3")
        rec = _latex_cell_metric(m.get("recall_stable"), kind="float3")
        f1 = _latex_cell_metric(m.get("f1_stable"), kind="float3")
        n = m.get("n")
        n_s = f"{int(n)}" if n is not None else "?"
        lines.append(
            f"{row_label} & {acc} & {prec} & {rec} & {f1} \\\\ " f"% n={n_s} scenarios with labels"
        )
    lines.append("")
    lines.append(
        "% --- Table~\\ref{tab:cct_primary} PINN* / Std NN* rows "
        "(ms; mean $|\\Delta t|/t_{\\mathrm{CCT}}^{\\mathrm{ref}}|$ as \\%; EAC row unchanged) ---"
    )
    for row_label, key in (
        ("PINN* (Alg.~\\ref{alg:cct})", "pinn"),
        ("Std NN*", "ml"),
    ):
        block = results.get(key) or {}
        mae_ms = block.get("mae_s")
        rmse_ms = block.get("rmse_s")
        mae_s = None if mae_ms is None else float(mae_ms) * 1000.0
        rmse_s = None if rmse_ms is None else float(rmse_ms) * 1000.0
        mae_str = _latex_cell_metric(mae_s, kind="float2")
        rmse_str = _latex_cell_metric(rmse_s, kind="float2")
        mar = block.get("mean_abs_rel_error")
        mar_pct = None if mar is None else float(mar) * 100.0
        mar_str = _latex_cell_metric(mar_pct, kind="float2")
        pct = _latex_cell_metric(block.get("pct_under_5ms"), kind="float1")
        lines.append(f"{row_label} & {mae_str} & {rmse_str} & {mar_str} & {pct} \\\\")
    lines.append("")
    lines.append(
        "% Confusion (supplement): see stability_at_nominal_tc.{pinn,ml}.metrics "
        "{tp,tn,fp,fn} in the JSON."
    )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pe-input CCT on test split (PINN* and ML*, pe_direct_7)"
    )
    p.add_argument(
        "--from-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Load pe_input_cct_test_results.json, print LaTeX rows for paper tables, and exit.",
    )
    p.add_argument(
        "--print-latex-tables",
        action="store_true",
        help="After a full run, print LaTeX rows for tab:stability_nominal and tab:cct_primary.",
    )
    p.add_argument("--test-csv", type=Path, default=None, help="Pre-split test trajectory CSV")
    p.add_argument("--pinn-model", type=Path, default=None, help="PINN checkpoint (.pth)")
    p.add_argument("--ml-model", type=Path, default=None, help="ML baseline model.pth")
    p.add_argument(
        "--pinn-config",
        type=Path,
        default=None,
        help="Optional config YAML beside experiment (passed to load_pinn_model)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for pe_input_cct_test_results.json (required unless --from-json).",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--cct-inner-time-grid",
        type=_parse_cct_inner_time_grid,
        default="scenario_csv_time",
        help=(
            "CCT binary search: scenario_csv_time = each scenario's CSV time column "
            "(default, same as nominal clearing); linspace = uniform [0, T] with "
            "--simulation-time and --t-eval-steps."
        ),
    )
    p.add_argument(
        "--simulation-time",
        type=float,
        default=5.0,
        help="CCT inner loop when --cct-inner-time-grid linspace: end time T for np.linspace(0, T, steps).",
    )
    p.add_argument(
        "--t-eval-steps",
        type=int,
        default=500,
        help="CCT inner loop when --cct-inner-time-grid linspace: number of uniform time samples.",
    )
    p.add_argument(
        "--search-low-pad",
        type=float,
        default=1e-3,
        help="Binary search lower bound = tf + this",
    )
    p.add_argument(
        "--search-high-pad",
        type=float,
        default=1.0,
        help="Binary search upper bound = tf + this",
    )
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--max-iterations", type=int, default=20)
    p.add_argument("--delta-threshold", type=float, default=float(np.pi))
    p.add_argument(
        "--stability-mode",
        type=_parse_stability_mode,
        default="persistence_fraction",
        help=(
            "Angle-based stability: persistence_fraction (default: ≥ "
            "--persistence-violation-fraction of samples with |δ|≥ threshold in some "
            "--persistence-window-seconds interval), or terminal, global_max, final_window. "
            "Nominal clearing and default CCT inner loop both use each scenario's CSV time; "
            "use --cct-inner-time-grid linspace for a uniform [0,T] grid."
        ),
    )
    p.add_argument(
        "--final-window-seconds",
        type=float,
        default=0.25,
        help="Trailing time window for final_window mode (seconds)",
    )
    p.add_argument(
        "--persistence-window-seconds",
        type=_parse_positive_float,
        default=0.25,
        help="Sliding window width for persistence_fraction mode (seconds)",
    )
    p.add_argument(
        "--persistence-violation-fraction",
        type=_parse_persistence_fraction,
        default=0.9,
        help=(
            "For persistence_fraction: min. fraction of samples in a window with "
            "|δ|≥ --delta-threshold (default 0.9; must be in (0, 1])"
        ),
    )
    args = p.parse_args()

    if args.from_json is not None:
        path = args.from_json
        if not path.is_file():
            raise SystemExit(f"JSON not found: {path}")
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        print(format_latex_table_rows(loaded))
        return

    missing = [
        n
        for n, v in (
            ("--test-csv", args.test_csv),
            ("--pinn-model", args.pinn_model),
            ("--ml-model", args.ml_model),
            ("--output-dir", args.output_dir),
        )
        if v is None
    ]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)} (or use --from-json)")

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    test_df = pd.read_csv(args.test_csv)
    scenario_ids = sorted(test_df["scenario_id"].unique().tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pinn_model, pinn_scalers, pinn_input_method = load_pinn_model(
        args.pinn_model, config_path=args.pinn_config, device=device
    )
    p_dim = getattr(pinn_model, "input_dim", None)
    if pinn_input_method != "pe_direct_7" and p_dim != 7:
        print(
            f"⚠️  PINN input_method={pinn_input_method}, input_dim={p_dim}; "
            "expected pe_direct_7 / 7"
        )

    ml_model, ml_scalers, ml_input_method = load_ml_baseline_model(args.ml_model, device=device)
    if ml_input_method != "pe_direct_7":
        raise SystemExit(f"ML checkpoint input_method={ml_input_method}; need pe_direct_7")

    bs_note = "run_eac_comparison PINN CCT: low=tf+1e-3, high=tf+1, tol=1e-3, max_iter=20"
    results = {
        "protocol": {
            "driver": "scripts/run_pe_input_cct_test.py",
            "test_csv": str(args.test_csv.resolve()),
            "pe_mapping": ("utils.pe_hypothetical_profile.pe_profile_for_hypothetical_clearing"),
            "stability_rule": (
                "utils.stability_checker.check_stability (angle-only; "
                "stability_mode + delta_threshold; default persistence_fraction)"
            ),
            "stability_mode": args.stability_mode,
            "final_window_seconds": float(args.final_window_seconds),
            "persistence_window_seconds": float(args.persistence_window_seconds),
            "persistence_violation_fraction": float(args.persistence_violation_fraction),
            "delta_threshold_rad": float(args.delta_threshold),
            "stability_labels_csv": "is_stable, else is_stable_from_cct",
            "nominal_clearing_time_grid": (
                "per-scenario CSV time column (same span as trajectory figures); "
                "not --simulation-time"
            ),
            "cct_inner_time_grid": (
                "per-scenario CSV time column (same as nominal clearing)"
                if args.cct_inner_time_grid == "scenario_csv_time"
                else f"np.linspace(0, {args.simulation_time}, {args.t_eval_steps})"
            ),
            "cct_inner_time_grid_mode": args.cct_inner_time_grid,
            "binary_search_defaults": bs_note,
            "search_low_pad": float(args.search_low_pad),
            "search_high_pad": float(args.search_high_pad),
            "tolerance_s": float(args.tolerance),
            "max_iterations": int(args.max_iterations),
            "git_rev": _try_git_rev(),
        },
        "stability_at_nominal_tc": {
            "pinn": run_stability_at_nominal_clearing(
                "PINN_star",
                "pinn",
                pinn_model,
                pinn_scalers,
                pinn_input_method,
                test_df,
                scenario_ids,
                device,
                args.delta_threshold,
                stability_mode=args.stability_mode,
                final_window_seconds=args.final_window_seconds,
                persistence_window_seconds=args.persistence_window_seconds,
                persistence_violation_fraction=args.persistence_violation_fraction,
            ),
            "ml": run_stability_at_nominal_clearing(
                "StdNN_star",
                "ml",
                ml_model,
                ml_scalers,
                ml_input_method,
                test_df,
                scenario_ids,
                device,
                args.delta_threshold,
                stability_mode=args.stability_mode,
                final_window_seconds=args.final_window_seconds,
                persistence_window_seconds=args.persistence_window_seconds,
                persistence_violation_fraction=args.persistence_violation_fraction,
            ),
        },
        "pinn": run_one_checkpoint(
            "PINN_star",
            "pinn",
            pinn_model,
            pinn_scalers,
            pinn_input_method,
            test_df,
            scenario_ids,
            device,
            args.simulation_time,
            args.t_eval_steps,
            args.cct_inner_time_grid,
            args.search_low_pad,
            args.search_high_pad,
            args.tolerance,
            args.max_iterations,
            args.delta_threshold,
            stability_mode=args.stability_mode,
            final_window_seconds=args.final_window_seconds,
            persistence_window_seconds=args.persistence_window_seconds,
            persistence_violation_fraction=args.persistence_violation_fraction,
        ),
        "ml": run_one_checkpoint(
            "StdNN_star",
            "ml",
            ml_model,
            ml_scalers,
            ml_input_method,
            test_df,
            scenario_ids,
            device,
            args.simulation_time,
            args.t_eval_steps,
            args.cct_inner_time_grid,
            args.search_low_pad,
            args.search_high_pad,
            args.tolerance,
            args.max_iterations,
            args.delta_threshold,
            stability_mode=args.stability_mode,
            final_window_seconds=args.final_window_seconds,
            persistence_window_seconds=args.persistence_window_seconds,
            persistence_violation_fraction=args.persistence_violation_fraction,
        ),
    }

    out_json = args.output_dir / "pe_input_cct_test_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["stability_at_nominal_tc"], indent=2)[:2500])
    print("---")
    print(json.dumps(results["pinn"], indent=2)[:2000])
    print("---")
    print(json.dumps(results["ml"], indent=2)[:2000])
    print(f"\nWrote {out_json}")
    if args.print_latex_tables:
        print("\n" + format_latex_table_rows(results))


if __name__ == "__main__":
    main()
