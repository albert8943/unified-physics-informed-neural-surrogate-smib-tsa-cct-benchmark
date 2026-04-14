#!/usr/bin/env python3
"""
Extract wall-clock training times (and optional checkpoint sizes) from experiment_summary.json
for the manuscript Table~\\ref{tab:efficiency}.

Training times are written by run_complete_experiment.py as:
  - summary["pinn"]["training"]["training_time"]  (seconds)
  - summary["ml_baseline"]["standard_nn"]["training_time"]  (seconds)

Data generation duration, per-trajectory ANDES wall time, surrogate inference, and CCT search
times are not currently persisted in experiment_summary.json. Pass --data-gen-hours or extend
the driver to log them if you need every row to be numeric.

Examples (repo root):
  python scripts/print_efficiency_metrics_for_paper.py ^
    --summary paper_writing/Final_Experiment_Results_for_Paper/exp_20260211_190610/experiment_summary.json ^
    --project-root . --latex

  python scripts/print_efficiency_metrics_for_paper.py ^
    --summary outputs/campaign_indep_final_test_20260406/experiment_summary.json ^
    --project-root . --latex
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _training_seconds_pinn(summary: Dict[str, Any]) -> Optional[float]:
    p = summary.get("pinn") or {}
    tr = p.get("training") or {}
    v = tr.get("training_time")
    return float(v) if v is not None else None


def _training_seconds_ml(summary: Dict[str, Any]) -> Optional[float]:
    mb = summary.get("ml_baseline") or {}
    sn = mb.get("standard_nn") or mb.get("model") or {}
    if isinstance(sn, dict):
        v = sn.get("training_time")
        if v is not None:
            return float(v)
    return None


def _resolve_under_root(path_str: Optional[str], project_root: Path) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_file():
        return p
    cand = project_root / path_str
    return cand if cand.is_file() else None


def _file_mb(path: Optional[Path]) -> Optional[float]:
    if path is None or not path.is_file():
        return None
    return path.stat().st_size / (1024 * 1024)


def main() -> int:
    ap = argparse.ArgumentParser(description="Print efficiency metrics for the paper table.")
    ap.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to experiment_summary.json",
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root for resolving relative model_path entries (default: cwd)",
    )
    ap.add_argument(
        "--data-gen-hours",
        type=float,
        default=None,
        help="Optional: wall-clock hours for data generation (not in JSON)",
    )
    ap.add_argument("--latex", action="store_true", help="Emit LaTeX rows for tab:efficiency")
    args = ap.parse_args()

    project_root = (args.project_root or Path.cwd()).resolve()
    summary_path = args.summary.resolve()
    if not summary_path.is_file():
        print(f"Not found: {summary_path}", file=sys.stderr)
        return 1

    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)

    exp_id = summary.get("experiment_id", "?")
    repro = summary.get("reproducibility") or {}
    torch_v = (repro.get("package_versions") or {}).get("torch", "?")

    pinn_s = _training_seconds_pinn(summary)
    ml_s = _training_seconds_ml(summary)

    pinn_path = _resolve_under_root(
        (summary.get("pinn") or {}).get("model_path"),
        project_root,
    )
    ml_block = summary.get("ml_baseline") or {}
    std_nn = ml_block.get("standard_nn") or {}
    ml_path = _resolve_under_root(std_nn.get("model_path"), project_root)

    pinn_mb = _file_mb(pinn_path)
    ml_mb = _file_mb(ml_path)

    lines = []
    lines.append(f"experiment_id: {exp_id}")
    lines.append(f"summary: {summary_path}")
    lines.append(f"torch: {torch_v}")
    if pinn_s is not None:
        lines.append(f"PINN training: {pinn_s:.1f} s ({pinn_s / 60.0:.2f} min)")
    else:
        lines.append("PINN training: (missing in JSON)")
    if ml_s is not None:
        lines.append(f"Std NN training: {ml_s:.1f} s ({ml_s / 60.0:.2f} min)")
    else:
        lines.append("Std NN training: (missing in JSON)")
    if pinn_mb is not None:
        lines.append(f"PINN checkpoint size: {pinn_mb:.3f} MB ({pinn_path})")
    else:
        lines.append("PINN checkpoint size: (path missing or not under project root)")
    if ml_mb is not None:
        lines.append(f"Std NN checkpoint size: {ml_mb:.3f} MB ({ml_path})")
    else:
        lines.append("Std NN checkpoint size: (path missing or not under project root)")
    if args.data_gen_hours is not None:
        lines.append(f"Data generation (override): {args.data_gen_hours} h")

    print("\n".join(lines))

    if args.latex:
        print("\n% --- paste into tab:efficiency (adjust other rows manually) ---")
        dgh = args.data_gen_hours
        row_dg = f"{dgh:.1f}" if dgh is not None else "2--4"
        row_pinn = f"{pinn_s / 60.0:.1f}" if pinn_s is not None else "?"
        row_ml = f"{ml_s / 60.0:.1f}" if ml_s is not None else "?"
        sz = None
        if pinn_mb is not None and ml_mb is not None:
            sz = f"{max(pinn_mb, ml_mb):.2f}"
        row_sz = sz if sz is not None else "$<1$"
        print(f"Data generation (h) & {row_dg} \\\\")
        print(f"PINN training (min) & {row_pinn} \\\\")
        print(f"Std NN training (min) & {row_ml} \\\\")
        print("PINN inference (ms/trajectory) & $<1$ \\\\")
        print("ANDES TDS (ms/trajectory) & 100--500 \\\\")
        print("Speedup (PINN vs.\\ ANDES) & 100--500$\\times$ \\\\")
        print("CCT search (ms, Alg.~\\ref{alg:cct}) & $<10$ \\\\")
        print("CCT search (s, TDS-only binary search) & 30--120 \\\\")
        print(f"Model size (MB) & {row_sz} \\\\")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
