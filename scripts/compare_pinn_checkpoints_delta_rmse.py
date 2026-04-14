#!/usr/bin/env python
"""
Pairwise per-scenario δ RMSE for two PINN checkpoints on the same trajectory CSV.

Use after retraining residual vs plain-MLP models to verify metrics differ (or quantify
agreement). Reuses ``evaluate_model.load_model_and_scalers`` and ``evaluate_scenario``.

Example::

    python scripts/compare_pinn_checkpoints_delta_rmse.py ^
      --checkpoint-a .../pinn_lp_05/pinn/model/best_model_*.pth ^
      --checkpoint-b .../pinn_nores_lp05/pinn/model/best_model_*.pth ^
      --data-path .../test_data_20260211_190612.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_model import (  # noqa: E402
    evaluate_scenario,
    load_model_and_scalers,
)


def _resolve_glob(path: Path) -> Path:
    s = str(path)
    if "*" not in s:
        return path
    matches = list(path.parent.glob(path.name))
    if not matches:
        raise FileNotFoundError(f"No match for glob: {path}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-a", type=Path, required=True)
    p.add_argument("--checkpoint-b", type=Path, required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-scenarios", type=int, default=None)
    args = p.parse_args()

    ck_a = _resolve_glob(args.checkpoint_a)
    ck_b = _resolve_glob(args.checkpoint_b)
    data_path = _resolve_glob(args.data_path)

    model_a, scalers_a, im_a = load_model_and_scalers(ck_a)
    model_b, scalers_b, im_b = load_model_and_scalers(ck_b)
    model_a = model_a.to(args.device)
    model_b = model_b.to(args.device)

    if im_a != im_b:
        print(f"Warning: input_method A={im_a} B={im_b} (should match for a fair comparison)")

    df = pd.read_csv(data_path)
    scenarios = list(df["scenario_id"].unique())
    if args.n_scenarios is not None:
        scenarios = scenarios[: args.n_scenarios]

    rmse_a, rmse_b = [], []
    for sid in scenarios:
        block = df[df["scenario_id"] == sid]
        ra = evaluate_scenario(model_a, block, scalers_a, args.device, input_method=im_a)[
            "delta_rmse"
        ]
        rb = evaluate_scenario(model_b, block, scalers_b, args.device, input_method=im_b)[
            "delta_rmse"
        ]
        rmse_a.append(float(ra))
        rmse_b.append(float(rb))

    a = np.array(rmse_a)
    b = np.array(rmse_b)
    diff = a - b
    print(f"Scenarios: {len(scenarios)}  CSV: {data_path}")
    print(f"Mean RMSE δ (rad)  A: {a.mean():.6f}  B: {b.mean():.6f}")
    print(f"Std (rad)          A: {a.std(ddof=1):.6f}  B: {b.std(ddof=1):.6f}")
    absd = np.abs(diff)
    print(f"Per-scenario |Δ| max: {np.max(absd):.6e}  mean |Δ|: {np.mean(absd):.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
