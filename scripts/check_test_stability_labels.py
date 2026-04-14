#!/usr/bin/env python
"""
Sanity-check stability label columns on a trajectory CSV (e.g. test split).

For each scenario_id: reads the first row (sorted by time if present) and compares
``is_stable`` vs ``is_stable_from_cct``. Optionally flags scenarios where a column
is non-constant along the trajectory.

Usage (repo root):
    python scripts/check_test_stability_labels.py \\
        --csv data/processed/exp_20260211_190612/test_data_20260211_190612.csv \\
        --out-json docs/publication/pre_specs/test_stability_label_consistency.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _as_bool_or_none(x: Any) -> Optional[bool]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    return bool(x)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare is_stable vs is_stable_from_cct per scenario")
    p.add_argument("--csv", type=Path, required=True, help="Trajectory CSV (e.g. test_data_*.csv)")
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write machine-readable summary (optional)",
    )
    args = p.parse_args()
    path = args.csv
    if not path.is_file():
        raise SystemExit(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if "scenario_id" not in df.columns:
        raise SystemExit("Column scenario_id missing")

    has_a = "is_stable" in df.columns
    has_b = "is_stable_from_cct" in df.columns
    if not has_a and not has_b:
        raise SystemExit("Neither is_stable nor is_stable_from_cct present")

    rows_out: List[Dict[str, Any]] = []
    agree = 0
    disagree = 0
    mismatch_ids: List[int] = []
    varying_a: List[int] = []
    varying_b: List[int] = []

    for sid, g in df.groupby("scenario_id", sort=True):
        g = g.sort_values("time") if "time" in g.columns else g
        r0 = g.iloc[0]
        va = _as_bool_or_none(r0["is_stable"]) if has_a else None
        vb = _as_bool_or_none(r0["is_stable_from_cct"]) if has_b else None

        if has_a:
            ua = g["is_stable"].dropna().unique()
            if len(ua) > 1:
                varying_a.append(int(sid))
        if has_b:
            ub = g["is_stable_from_cct"].dropna().unique()
            if len(ub) > 1:
                varying_b.append(int(sid))

        if has_a and has_b and va is not None and vb is not None:
            if va == vb:
                agree += 1
            else:
                disagree += 1
                mismatch_ids.append(int(sid))

        rows_out.append(
            {
                "scenario_id": int(sid),
                "is_stable": va,
                "is_stable_from_cct": vb,
            }
        )

    n_sc = len(rows_out)
    summary: Dict[str, Any] = {
        "csv": str(path.resolve()),
        "n_scenarios": n_sc,
        "columns": {"is_stable": has_a, "is_stable_from_cct": has_b},
        "pairwise_on_first_row": {
            "both_present": agree + disagree,
            "agree": agree,
            "disagree": disagree,
            "disagreement_scenario_ids": mismatch_ids,
        },
        "constant_along_trajectory": {
            "is_stable_varying_scenario_ids": varying_a,
            "is_stable_from_cct_varying_scenario_ids": varying_b,
        },
        "methods_note": (
            "Pe-input CCT driver and nominal-stability table use is_stable when present, "
            "else is_stable_from_cct (run_pe_input_cct_test._ground_truth_stable_label)."
        ),
    }

    print(json.dumps(summary, indent=2))
    if disagree == 0 and has_a and has_b and agree > 0:
        print(
            f"\nOK: is_stable and is_stable_from_cct agree on all {agree} scenarios "
            "where both are present (first row per scenario)."
        )
    elif has_a and has_b:
        print(f"\nWARNING: {disagree} scenario(s) disagree: {mismatch_ids}")
    if varying_a or varying_b:
        print(
            f"\nNote: label varies with time in some scenarios "
            f"(is_stable: {len(varying_a)}, is_stable_from_cct: {len(varying_b)})."
        )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        out = {**summary, "per_scenario_first_row": rows_out}
        args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
