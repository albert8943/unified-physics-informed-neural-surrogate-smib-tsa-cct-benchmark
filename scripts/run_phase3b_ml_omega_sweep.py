#!/usr/bin/env python
"""
Phase 3b: ML baseline only — refine ``ml_baseline.scale_to_norm[1]`` (ω MSE weight) below 50
at fixed ``unstable_weight=2`` (same family as Phase 3 winner ``ml_uw2_ow50``).

For each ω weight, runs ``run_complete_experiment.py`` (ML train only), then
``evaluate_ml_baseline.py`` on the **validation** CSV for ``mean_scenario_delta_rmse`` (rad).

Writes ``docs/publication/pre_specs/phase3b_ml_omega_low_results_<YYYYMMDD>.json`` (or ``--out-json``).

Example (repo root, PowerShell)::

    python scripts/run_phase3b_ml_omega_sweep.py \\
      --data-dir data/processed/exp_20260211_190612 \\
      --output-dir outputs/campaign_ml_ext_omega_1_10_20_40 \\
      --pinn-model-path outputs/campaign_indep_pinn/pinn_lp_05/pinn/model/best_model_20260406_114201.pth

Aggregate existing runs only (no training)::

    python scripts/run_phase3b_ml_omega_sweep.py --aggregate-only \\
      --data-dir data/processed/exp_20260211_190612 \\
      --output-dir outputs/campaign_ml_ext_omega_1_10_20_40 \\
      --out-json docs/publication/pre_specs/phase3b_ml_omega_low_results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/smib/smib_pinn_ml_matched_pe_direct_7_parity_dropout_wd.yaml"
)
PHASE3_SUMMARY = (
    PROJECT_ROOT / "docs/publication/pre_specs/phase3_ml_hyperparam_results_20260406.json"
)


def _find_one_csv(data_dir: Path, glob_pat: str) -> Path:
    matches = sorted(data_dir.glob(glob_pat))
    if not matches:
        raise FileNotFoundError(f"No CSV matching {glob_pat!r} under {data_dir}")
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple CSVs for {glob_pat!r} under {data_dir}: {matches}. Pick one path explicitly."
        )
    return matches[0]


def _exp_id(unstable_w: float, omega_w: float) -> str:
    ow = int(omega_w) if float(omega_w).is_integer() else omega_w
    uwi = int(unstable_w) if float(unstable_w).is_integer() else unstable_w
    return f"ml_uw{uwi}_ow{ow}"


def _run(cmd: List[str], dry_run: bool) -> None:
    print("\n" + "=" * 70)
    print(" ".join(cmd))
    print("=" * 70 + "\n")
    if dry_run:
        return
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if r.returncode != 0:
        raise SystemExit(f"Command failed (exit {r.returncode})")


def _read_val_metrics(metrics_path: Path) -> Tuple[float, float]:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    j_delta = float(data.get("mean_scenario_delta_rmse", float("nan")))
    j_omega = float(data.get("mean_scenario_omega_rmse", float("nan")))
    return j_delta, j_omega


def _phase3_reference_row() -> Optional[Dict[str, Any]]:
    if not PHASE3_SUMMARY.is_file():
        return None
    doc = json.loads(PHASE3_SUMMARY.read_text(encoding="utf-8"))
    for c in doc.get("candidates") or []:
        if c.get("experiment_id") == "ml_uw2_ow50":
            return {
                "experiment_id": c.get("experiment_id"),
                "unstable_weight": c.get("unstable_weight"),
                "omega_mse_weight_scale_to_norm_1": c.get("omega_mse_weight_scale_to_norm_1"),
                "J_val_mean_scenario_delta_rmse_rad": c.get("J_val_mean_scenario_delta_rmse_rad"),
                "mean_scenario_omega_rmse_pu": c.get("mean_scenario_omega_rmse_pu"),
                "source": "archived_phase3_json",
                "phase3_json_relative": str(PHASE3_SUMMARY.relative_to(PROJECT_ROOT)).replace(
                    "\\", "/"
                ),
            }
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3b: ML ω-weight sweep at unstable_weight=2")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--pinn-model-path",
        type=Path,
        default=None,
        help="Frozen PINN checkpoint (ML-only runs still require this path in the driver)",
    )
    p.add_argument("--unstable-weight", type=float, default=2.0)
    p.add_argument(
        "--omega-weights",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 20.0, 40.0],
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training; read val_metrics/metrics.json under each experiment folder",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Default: docs/publication/pre_specs/phase3b_ml_omega_low_results_<date>.json",
    )
    args = p.parse_args()

    args.config = args.config.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.pinn_model_path is not None:
        args.pinn_model_path = args.pinn_model_path.resolve()
    if args.out_json is not None:
        args.out_json = args.out_json.resolve()

    train_csv: Optional[Path] = None
    val_csv: Optional[Path] = None
    if not args.aggregate_only:
        train_csv = _find_one_csv(args.data_dir, "train_data_*.csv")
        val_csv = _find_one_csv(args.data_dir, "val_data_*.csv")

    if not args.aggregate_only:
        if args.pinn_model_path is None:
            raise SystemExit("--pinn-model-path is required unless --aggregate-only")
        if not args.pinn_model_path.is_file():
            raise SystemExit(f"PINN checkpoint not found: {args.pinn_model_path}")

    out_json: Path
    if args.out_json is None:
        out_json = (
            PROJECT_ROOT
            / "docs/publication/pre_specs"
            / f"phase3b_ml_omega_low_results_{date.today().strftime('%Y%m%d')}.json"
        )
    else:
        out_json = args.out_json

    candidates: List[Dict[str, Any]] = []

    for ow in args.omega_weights:
        exp_id = _exp_id(args.unstable_weight, ow)
        exp_root = args.output_dir / exp_id
        model_path = exp_root / "ml_baseline" / "standard_nn" / "model" / "model.pth"
        val_metrics_path = exp_root / "val_metrics" / "metrics.json"

        if not args.aggregate_only:
            assert train_csv is not None and val_csv is not None
            if not args.dry_run:
                args.output_dir.mkdir(parents=True, exist_ok=True)
            cmd_train = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/run_complete_experiment.py"),
                "--config",
                str(args.config),
                "--skip-data-generation",
                "--skip-data-analysis",
                "--data-dir",
                str(args.data_dir),
                "--output-dir",
                str(args.output_dir),
                "--experiment-id",
                exp_id,
                "--skip-pinn-training",
                "--pinn-model-path",
                str(args.pinn_model_path),
                "--skip-comparison",
                "--skip-pinn-evaluation",
                "--skip-ml-baseline-evaluation",
                "--ml-baseline-unstable-weight",
                str(args.unstable_weight),
                "--ml-baseline-omega-weight",
                str(ow),
            ]
            _run(cmd_train, args.dry_run)

            cmd_eval = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/evaluate_ml_baseline.py"),
                "--model-path",
                str(model_path),
                "--test-data",
                str(train_csv),
                "--test-split-path",
                str(val_csv),
                "--output-dir",
                str(exp_root / "val_metrics"),
            ]
            if not args.dry_run and not model_path.is_file():
                raise SystemExit(f"Expected ML model after training: {model_path}")
            _run(cmd_eval, args.dry_run)

        if args.dry_run:
            candidates.append(
                {
                    "experiment_id": exp_id,
                    "unstable_weight": args.unstable_weight,
                    "omega_mse_weight_scale_to_norm_1": ow,
                    "dry_run": True,
                }
            )
            continue

        if not val_metrics_path.is_file():
            raise SystemExit(f"Missing val metrics (run training or fix paths): {val_metrics_path}")

        j_delta, j_omega = _read_val_metrics(val_metrics_path)
        candidates.append(
            {
                "experiment_id": exp_id,
                "unstable_weight": args.unstable_weight,
                "omega_mse_weight_scale_to_norm_1": ow,
                "J_val_mean_scenario_delta_rmse_rad": j_delta,
                "mean_scenario_omega_rmse_pu": j_omega,
                "model_relative": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "val_metrics_json_relative": str(
                    val_metrics_path.relative_to(PROJECT_ROOT)
                ).replace("\\", "/"),
            }
        )

    ref = _phase3_reference_row()
    pool: List[Dict[str, Any]] = list(candidates)
    if ref is not None:
        pool.append(ref)

    winner: Optional[Dict[str, Any]] = None
    if not args.dry_run and pool:
        numeric = [c for c in pool if "J_val_mean_scenario_delta_rmse_rad" in c]
        if numeric:
            winner = min(numeric, key=lambda c: float(c["J_val_mean_scenario_delta_rmse_rad"]))

    rec_lines = [
        "Selection: minimize validation mean per-scenario delta RMSE (same as Phase 3).",
        "PINN* unchanged; only ML omega MSE weight was swept at unstable_weight=2.",
    ]
    if winner:
        rec_lines.append(
            f"Recommended ML checkpoint for a new test comparison: experiment_id={winner.get('experiment_id')}, "
            f"omega_mse_weight={winner.get('omega_mse_weight_scale_to_norm_1')}, "
            f"J_val_mean_scenario_delta_rmse_rad={winner.get('J_val_mean_scenario_delta_rmse_rad'):.6f}."
        )
        if winner.get("experiment_id") == "ml_uw2_ow50":
            rec_lines.append(
                "Phase 3 winner ml_uw2_ow50 remains best; keep paper ML* row and figures as-is."
            )
        else:
            rec_lines.append(
                "Run compare_models once on test_data_*.csv vs frozen PINN* before updating the paper table."
            )
    else:
        rec_lines.append("No winner computed (dry-run or empty pool).")

    doc: Dict[str, Any] = {
        "campaign": "Phase 3b: ML ω MSE weight sweep below 50 at unstable_weight=2",
        "parent_phase3_json": str(PHASE3_SUMMARY.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "driver": "scripts/run_phase3b_ml_omega_sweep.py",
        "base_config": str(args.config.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "dataset_relative": str(args.data_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "train_split_glob": "train_data_*.csv",
        "val_split_glob": "val_data_*.csv",
        "train_csv_resolved": (
            str(train_csv.relative_to(PROJECT_ROOT)).replace("\\", "/") if train_csv else None
        ),
        "val_csv_resolved": (
            str(val_csv.relative_to(PROJECT_ROOT)).replace("\\", "/") if val_csv else None
        ),
        "unstable_weight": args.unstable_weight,
        "omega_weights_requested": list(args.omega_weights),
        "output_dir_relative": str(args.output_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "candidates_phase3b": candidates,
        "phase3_reference_ml_uw2_ow50": ref,
        "winner": winner,
        "recommendation_for_paper": rec_lines,
    }

    if not args.dry_run:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out_json}")

    print("\n--- recommendation (preview) ---")
    for line in rec_lines:
        print(line)


if __name__ == "__main__":
    main()
