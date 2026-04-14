#!/usr/bin/env python3
"""
Summarize segment-wise ML vs PINN metrics from comparison_results.json.

Example:
  python scripts/summarize_comparison_segments.py \\
    outputs/complete_experiments/exp_20260401_175318/comparison/comparison_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "comparison_json",
        type=Path,
        help="Path to comparison_results.json",
    )
    args = parser.parse_args()
    path: Path = args.comparison_json
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    seg = data.get("segment_metrics") or {}
    if not seg:
        print("No segment_metrics in JSON.")
        return 0

    print(f"File: {path}")
    print(f"Test scenarios: {data.get('n_scenarios', '?')}")
    for name, block in seg.items():
        print(f"\n=== {name} ===")
        ml_d = block.get("ml_baseline", {}).get("delta_rmse_mean")
        ml_o = block.get("ml_baseline", {}).get("omega_rmse_mean")
        p_d = block.get("pinn", {}).get("delta_rmse_mean")
        p_o = block.get("pinn", {}).get("omega_rmse_mean")
        print(f"  ML  delta_rmse_mean: {ml_d}")
        print(f"  PINN delta_rmse_mean: {p_d}")
        if ml_o is not None or p_o is not None:
            print(f"  ML  omega_rmse_mean: {ml_o}")
            print(f"  PINN omega_rmse_mean: {p_o}")
        dc = block.get("delta_rmse_comparison")
        oc = block.get("omega_rmse_comparison")
        if dc and "improvement" in dc:
            imp = dc["improvement"]
            print(
                f"  Delta paired improvement (PINN vs ML, mean RMSE): "
                f"absolute={imp.get('absolute')}, relative%={imp.get('relative_percent')}"
            )
        if oc and "improvement" in oc:
            imp = oc["improvement"]
            print(
                f"  Omega paired improvement: "
                f"absolute={imp.get('absolute')}, relative%={imp.get('relative_percent')}"
            )

    overall_d = data.get("delta_comparison", {})
    overall_o = data.get("omega_comparison", {})
    if overall_d:
        print("\n=== overall (scenario-mean delta RMSE) ===")
        print(f"  ML:   {overall_d.get('ml_baseline', {}).get('mean')}")
        print(f"  PINN: {overall_d.get('pinn', {}).get('mean')}")
    if overall_o:
        print("\n=== overall (scenario-mean omega RMSE) ===")
        print(f"  ML:   {overall_o.get('ml_baseline', {}).get('mean')}")
        print(f"  PINN: {overall_o.get('pinn', {}).get('mean')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
