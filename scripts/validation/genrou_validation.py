#!/usr/bin/env python
"""
GENROU Validation Script.

Validates PINN trained on GENCLS against GENROU (detailed generator model).

Usage:
    python scripts/validation/genrou_validation.py \
        --pinn-model outputs/experiments/exp_XXX/model/best_model.pth \
        --genrou-case smib/SMIB_genrou.xlsx \
        --test-scenarios outputs/experiments/exp_XXX/data/preprocessed/test_data.csv \
        --output-dir outputs/genrou_validation
"""

import argparse
import sys
import io
from pathlib import Path
from typing import List, Dict, Optional

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
from scripts.core.utils import generate_experiment_id, create_experiment_directory
from evaluation.genrou_validation import validate_pinn_on_genrou


def load_test_scenarios(csv_path: Path, max_scenarios: Optional[int] = None) -> List[Dict]:
    """
    Load test scenarios from CSV file.

    Parameters:
    -----------
    csv_path : Path
        Path to test data CSV
    max_scenarios : int, optional
        Maximum number of scenarios to load (default: all scenarios)

    Returns:
    --------
    scenarios : list
        List of scenario dictionaries
    """
    data = pd.read_csv(csv_path)

    # Group by scenario
    scenarios = []
    scenario_ids = data["scenario_id"].unique()
    if max_scenarios is not None:
        scenario_ids = scenario_ids[:max_scenarios]

    for scenario_id in scenario_ids:
        scenario_data = data[data["scenario_id"] == scenario_id].iloc[0]
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "H": scenario_data.get("H", 6.0),
                "D": scenario_data.get("D", 1.0),
                "Pm": scenario_data.get("Pm", 0.8),
                "delta0": scenario_data.get("delta0", 0.5),
                "omega0": scenario_data.get("omega0", 1.0),
                "tf": scenario_data.get("tf", 1.0),
                "tc": scenario_data.get("tc", 1.2),
            }
        )

    return scenarios


def main():
    """Main GENROU validation workflow."""
    parser = argparse.ArgumentParser(description="Validate PINN on GENROU model")
    parser.add_argument(
        "--pinn-model",
        type=str,
        required=True,
        help="Path to trained PINN model (trained on GENCLS)",
    )
    parser.add_argument(
        "--genrou-case",
        type=str,
        required=True,
        help="Path to GENROU case file (e.g., smib/SMIB_genrou.xlsx)",
    )
    parser.add_argument(
        "--test-scenarios",
        type=str,
        help="Path to test scenarios CSV (or use default scenarios)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/genrou_validation",
        help="Output directory for validation results",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Maximum number of scenarios to validate (default: all scenarios)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GENROU VALIDATION")
    print("=" * 70)
    print(f"PINN Model: {args.pinn_model}")
    print(f"GENROU Case: {args.genrou_case}")

    # Load test scenarios
    if args.test_scenarios:
        test_scenarios = load_test_scenarios(Path(args.test_scenarios), args.max_scenarios)
        if args.max_scenarios is None:
            print(f"Loading all scenarios from: {args.test_scenarios}")
        else:
            print(f"Loading up to {args.max_scenarios} scenarios from: {args.test_scenarios}")
    else:
        # Default test scenarios
        test_scenarios = [
            {"H": 6.0, "D": 1.0, "Pm": 0.8, "delta0": 0.5, "omega0": 1.0, "tf": 1.0, "tc": 1.2},
            {"H": 5.0, "D": 1.5, "Pm": 0.7, "delta0": 0.6, "omega0": 1.0, "tf": 1.0, "tc": 1.3},
            {"H": 7.0, "D": 0.5, "Pm": 0.9, "delta0": 0.4, "omega0": 1.0, "tf": 1.0, "tc": 1.1},
        ]

    print(f"\nValidating on {len(test_scenarios)} scenarios")

    # Generate experiment ID and create experiment directory structure
    experiment_id = generate_experiment_id()
    base_output_dir = Path(args.output_dir)

    # Create experiment directory structure following project conventions
    dirs = create_experiment_directory(base_output_dir, experiment_id)

    # Use experiment root directory for results
    output_dir = dirs["root"]
    results_dir = dirs["results"]

    print(f"\nExperiment ID: {experiment_id}")
    print(f"Results will be saved to: {output_dir}")

    # Run validation
    results = validate_pinn_on_genrou(
        pinn_model_path=args.pinn_model,
        genrou_case_file=args.genrou_case,
        test_scenarios=test_scenarios,
    )

    # Save results in experiment directory
    results_file = results_dir / "genrou_validation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Compute summary statistics
    if len(results) > 0:
        import numpy as np

        delta_r2_values = [
            r.get("delta_r2", np.nan) for r in results if not np.isnan(r.get("delta_r2", np.nan))
        ]
        omega_r2_values = [
            r.get("omega_r2", np.nan) for r in results if not np.isnan(r.get("omega_r2", np.nan))
        ]
        delta_rmse_values = [
            r.get("delta_rmse", np.nan)
            for r in results
            if not np.isnan(r.get("delta_rmse", np.nan))
        ]
        delta_rmse_wrapped_values = [
            r.get("delta_rmse_wrapped", np.nan)
            for r in results
            if not np.isnan(r.get("delta_rmse_wrapped", np.nan))
        ]
        omega_rmse_values = [
            r.get("omega_rmse", np.nan)
            for r in results
            if not np.isnan(r.get("omega_rmse", np.nan))
        ]

        delta_r2_mean = np.nanmean(delta_r2_values) if len(delta_r2_values) > 0 else np.nan
        omega_r2_mean = np.nanmean(omega_r2_values) if len(omega_r2_values) > 0 else np.nan
        delta_rmse_mean = np.nanmean(delta_rmse_values) if len(delta_rmse_values) > 0 else np.nan
        delta_rmse_wrapped_mean = (
            np.nanmean(delta_rmse_wrapped_values) if len(delta_rmse_wrapped_values) > 0 else np.nan
        )
        omega_rmse_mean = np.nanmean(omega_rmse_values) if len(omega_rmse_values) > 0 else np.nan

        summary = {
            "n_scenarios": len(results),
            "delta_r2_mean": float(delta_r2_mean),
            "omega_r2_mean": float(omega_r2_mean),
            "delta_rmse_mean": float(delta_rmse_mean),
            "delta_rmse_wrapped_mean": float(delta_rmse_wrapped_mean),
            "omega_rmse_mean": float(omega_rmse_mean),
        }

        summary_file = results_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("GENROU VALIDATION SUMMARY")
        print("=" * 70)
        print(f"\nExperiment ID: {experiment_id}")
        print(f"Number of scenarios: {len(results)}")
        print(f"Mean R² Delta: {delta_r2_mean:.4f}")
        print(f"Mean R² Omega: {omega_r2_mean:.4f}")
        print(f"Mean RMSE Delta (wrapped): {delta_rmse_wrapped_mean:.4f} rad")
        print(f"Mean RMSE Delta (raw): {delta_rmse_mean:.4f} rad")
        print(f"Mean RMSE Omega: {omega_rmse_mean:.4f}")

        print(f"\n✓ Results saved to: {output_dir}")
        print(f"  Results: {results_file}")
        print(f"  Summary: {summary_file}")
        print(f"  Experiment folder: {experiment_id}")


if __name__ == "__main__":
    main()
