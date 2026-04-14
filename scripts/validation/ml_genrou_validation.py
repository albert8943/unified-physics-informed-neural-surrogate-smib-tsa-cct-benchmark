#!/usr/bin/env python
"""
Evaluate StandardNN / LSTM baseline on GENROU reference trajectories (same protocol as PINN).

Usage:
    python scripts/validation/ml_genrou_validation.py \\
        --ml-model outputs/campaign_indep_ml/ml_uw2_ow50/ml_baseline/standard_nn/model/model.pth \\
        --test-scenarios data/processed/exp_20260211_190612/test_data_20260211_190612.csv \\
        --output-dir outputs/publication/genrou_validation_ml \\
        [--genrou-case test_cases/SMIB_genrou.json] [--max-scenarios 5]
"""

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from evaluation.genrou_ml_validation import validate_ml_on_genrou
from scripts.core.utils import create_experiment_directory, generate_experiment_id


def load_test_scenarios(csv_path: Path, max_scenarios: Optional[int] = None) -> List[Dict]:
    """Same scenario list convention as scripts/validation/genrou_validation.py."""
    data = pd.read_csv(csv_path)
    scenarios = []
    scenario_ids = data["scenario_id"].unique()
    if max_scenarios is not None:
        scenario_ids = scenario_ids[:max_scenarios]

    for scenario_id in scenario_ids:
        row = data[data["scenario_id"] == scenario_id].iloc[0]
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "H": row.get("H", 6.0),
                "D": row.get("D", 1.0),
                "Pm": row.get("Pm", 0.8),
                "delta0": row.get("delta0", 0.5),
                "omega0": row.get("omega0", 1.0),
                "tf": row.get("tf", 1.0),
                "tc": row.get("tc", 1.2),
            }
        )
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="ML baseline vs GENROU validation")
    parser.add_argument(
        "--ml-model",
        type=str,
        required=True,
        help="Path to ML baseline checkpoint (model.pth)",
    )
    parser.add_argument(
        "--test-scenarios",
        type=str,
        required=True,
        help="Trajectory CSV (e.g. test_data_*.csv) with scenario_id, H, D, Pm, tf, tc, delta0, omega0",
    )
    parser.add_argument(
        "--genrou-case",
        type=str,
        default="test_cases/SMIB_genrou.json",
        help="GENROU ANDES case JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/publication/genrou_validation_ml",
        help="Base output directory",
    )
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Torch device for ML forward passes",
    )
    args = parser.parse_args()

    ml_path = Path(args.ml_model)
    if not ml_path.exists():
        print(f"Error: ML model not found: {ml_path}")
        sys.exit(1)

    scenarios = load_test_scenarios(Path(args.test_scenarios), args.max_scenarios)
    experiment_id = generate_experiment_id()
    dirs = create_experiment_directory(Path(args.output_dir), experiment_id)
    results_dir = dirs["results"]

    print("=" * 70)
    print("ML BASELINE — GENROU VALIDATION")
    print("=" * 70)
    print(f"Experiment: {experiment_id}")
    print(f"Scenarios: {len(scenarios)}")

    results = validate_ml_on_genrou(
        ml_model_path=str(ml_path),
        genrou_case_file=args.genrou_case,
        test_scenarios=scenarios,
        device=args.device,
    )

    out_json = results_dir / "ml_genrou_validation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    if results:
        delta_r2 = [r["delta_r2"] for r in results if not np.isnan(r.get("delta_r2", np.nan))]
        dw = [
            r["delta_rmse_wrapped"]
            for r in results
            if not np.isnan(r.get("delta_rmse_wrapped", np.nan))
        ]
        summary = {
            "n_scenarios": len(results),
            "delta_r2_mean": float(np.mean(delta_r2)) if delta_r2 else None,
            "delta_rmse_wrapped_mean": float(np.mean(dw)) if dw else None,
            "omega_r2_mean": float(
                np.mean([r["omega_r2"] for r in results if not np.isnan(r.get("omega_r2", np.nan))])
            )
            if results
            else None,
        }
        with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nMean δ RMSE wrapped (GENROU): {summary['delta_rmse_wrapped_mean']}")
        print(f"Mean δ R² (GENROU): {summary['delta_r2_mean']}")
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
