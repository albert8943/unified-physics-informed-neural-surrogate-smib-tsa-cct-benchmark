#!/usr/bin/env python3
"""
Generate SMIB trajectory CSV from ANDES **GENROU** ground truth (for training or analysis).

Replays each distinct ``scenario_id`` from a **reference** trajectory CSV that was built with
the classical case (e.g. ``smib/SMIB.json``), using the same ``H``, ``D``, ``Pm``, ``tf``,
``tc`` metadata. For each scenario, runs :func:`evaluation.genrou_simulation.run_genrou_trajectory`
on ``test_cases/SMIB_genrou.json`` (or another GENROU case) and appends rows in the same
wide-table layout expected by ``data_generation.preprocessing`` / ``run_complete_experiment``
(raw ``trajectory_data_*.csv`` style).

**Stability** is labeled with :func:`utils.angle_filter.determine_stability_180deg` on the
**raw** (unfolded) rotor angle from GENROU before principal-value folding of ``delta`` for
storage (matches common SMIB preprocessing).

Usage (repo root)::

    python scripts/data_generation/generate_genrou_trajectory_dataset.py ^
      --reference-csv data/processed/exp_20260211_190612/train_data_20260211_190612.csv ^
      --output-csv data/raw/trajectory_data_genrou_20260408.csv ^
      --genrou-case test_cases/SMIB_genrou.json

Then run your usual preprocessing on ``trajectory_data_genrou_*.csv`` (or concatenate train+val+test
references into one reference file first).

To regenerate GENROU trajectories **with the same train/val/test scenario_id sets** as an existing
classical processed folder (same filters / feature engineering as ``scripts/preprocess_data.py``,
but **no random re-split**), use
``scripts/data_generation/build_genrou_mirrored_classical_splits.py``.

Notes
-----
- ``omega`` is stored as **pu deviation** (consistent with ``run_genrou_trajectory``).
- Requires ANDES importable in the active environment.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.genrou_simulation import run_genrou_trajectory  # noqa: E402
from utils.angle_filter import determine_stability_180deg  # noqa: E402


def _resolve_case_path(genrou_case: str) -> str:
    try:
        import andes

        return str(andes.get_case(genrou_case))
    except Exception:
        return str(Path(genrou_case).resolve())


def _scenario_row_to_dict(row: pd.Series) -> Dict[str, Any]:
    """Map a reference CSV first-row (per scenario) to keys expected by run_genrou_trajectory."""
    H = float(row.get("H", row.get("param_H", np.nan)))
    D = float(row.get("D", row.get("param_D", np.nan)))
    Pm = float(row.get("Pm", row.get("param_Pm", np.nan)))
    tf = float(row.get("tf", row.get("param_tf", np.nan)))
    tc = float(row.get("tc", row.get("param_tc", np.nan)))
    if any(np.isnan([H, D, Pm, tf, tc])):
        raise ValueError(f"Missing H/D/Pm/tf/tc in reference row: {row.to_dict()}")
    return {"H": H, "D": D, "Pm": Pm, "tf": tf, "tc": tc}


def load_scenario_table(reference_csv: Path, max_scenarios: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(reference_csv)
    if "scenario_id" not in df.columns:
        raise ValueError("reference CSV must contain column 'scenario_id'")
    first = df.sort_values("time").groupby("scenario_id", sort=False).first().reset_index()
    if max_scenarios is not None:
        first = first.iloc[: int(max_scenarios)]
    return first


def build_rows_for_scenario(
    scenario_id: Any,
    phys: Dict[str, Any],
    case_path: str,
    fold_delta: bool,
) -> Optional[pd.DataFrame]:
    traj = run_genrou_trajectory(phys, case_path)
    if traj is None:
        return None
    t = traj["time"]
    delta_raw = np.asarray(traj["delta"], dtype=float)
    omega = np.asarray(traj["omega"], dtype=float)
    pe = np.asarray(traj["Pe"], dtype=float)
    n = min(len(t), len(delta_raw), len(omega), len(pe))
    if n == 0:
        return None
    t = t[:n]
    delta_raw = delta_raw[:n]
    omega = omega[:n]
    pe = pe[:n]

    is_stable, _ = determine_stability_180deg(delta_raw, threshold_deg=180.0)
    delta_out = np.arctan2(np.sin(delta_raw), np.cos(delta_raw)) if fold_delta else delta_raw.copy()

    delta0 = float(delta_out[0])
    omega0 = float(omega[0])

    return pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "time": t,
            "delta": delta_out,
            "omega": omega,
            "Pe": pe,
            "Pm": phys["Pm"],
            "H": phys["H"],
            "D": phys["D"],
            "tf": phys["tf"],
            "tc": phys["tc"],
            "delta0": delta0,
            "omega0": omega0,
            "is_stable": is_stable,
            "machine_model": "GENROU",
        }
    )


def generate_genrou_from_scenario_table(
    scenarios: pd.DataFrame,
    genrou_case: str,
    *,
    fold_delta: bool = True,
    max_scenarios: Optional[int] = None,
) -> tuple[pd.DataFrame, List[Any]]:
    """
    Run GENROU for each row of a scenario-first table (columns include scenario_id, H, D, Pm, tf, tc).

    Returns (combined trajectory DataFrame, list of failed scenario_id values).
    """
    if max_scenarios is not None:
        scenarios = scenarios.iloc[: int(max_scenarios)].copy()
    case_path = _resolve_case_path(genrou_case)
    frames: List[pd.DataFrame] = []
    failed: List[Any] = []

    for _, row in scenarios.iterrows():
        sid = row["scenario_id"]
        try:
            phys = _scenario_row_to_dict(row)
        except ValueError as e:
            print(f"  skip scenario_id={sid}: {e}")
            failed.append(sid)
            continue
        df_s = build_rows_for_scenario(sid, phys, case_path, fold_delta=fold_delta)
        if df_s is None:
            print(f"  skip scenario_id={sid}: simulation or extraction failed")
            failed.append(sid)
            continue
        frames.append(df_s)
        print(f"  ok scenario_id={sid} rows={len(df_s)} stable={df_s['is_stable'].iloc[0]}")

    if not frames:
        return pd.DataFrame(), failed
    combined = pd.concat(frames, ignore_index=True)
    return combined, failed


def main() -> None:
    p = argparse.ArgumentParser(description="Build GENROU trajectory CSV from reference scenarios")
    p.add_argument(
        "--reference-csv",
        type=Path,
        required=True,
        help=(
            "Classical trajectory CSV with scenario_id, time, H, D, Pm, tf, tc (or param_* aliases)."
        ),
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output path. Default: data/raw/trajectory_data_genrou_<timestamp>.csv",
    )
    p.add_argument(
        "--genrou-case",
        type=str,
        default="test_cases/SMIB_genrou.json",
        help="ANDES GENROU case (path or andes.get_case name)",
    )
    p.add_argument("--max-scenarios", type=int, default=None, help="Limit scenarios (smoke test)")
    p.add_argument(
        "--no-fold-delta",
        action="store_true",
        help="Store raw integrated delta (not recommended for training; preprocessing may fold).",
    )
    args = p.parse_args()

    ref = args.reference_csv.resolve()
    if not ref.is_file():
        print(f"Error: reference CSV not found: {ref}", file=sys.stderr)
        sys.exit(1)

    out = args.output_csv
    if out is None:
        raw_dir = PROJECT_ROOT / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = raw_dir / f"trajectory_data_genrou_{ts}.csv"
    else:
        out = Path(out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

    fold_delta = not args.no_fold_delta

    scenarios = load_scenario_table(ref, max_scenarios=args.max_scenarios)
    print(f"Reference: {ref} ({len(scenarios)} scenarios)")
    print(f"GENROU case: {_resolve_case_path(args.genrou_case)}")
    print(f"Output: {out}")

    combined, failed = generate_genrou_from_scenario_table(
        scenarios,
        args.genrou_case,
        fold_delta=fold_delta,
        max_scenarios=None,
    )

    if combined.empty:
        print("Error: no scenarios written.", file=sys.stderr)
        sys.exit(2)
    combined.to_csv(out, index=False)
    print(f"Wrote {len(combined)} rows, {combined['scenario_id'].nunique()} scenarios -> {out}")
    if failed:
        tail = "..." if len(failed) > 20 else ""
        print(f"Failed scenario_id list ({len(failed)}): {failed[:20]}{tail}")


if __name__ == "__main__":
    main()
