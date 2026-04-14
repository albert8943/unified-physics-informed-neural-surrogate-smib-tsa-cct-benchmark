#!/usr/bin/env python
"""
Regenerate statistical plots from existing results.

Usage:
    python scripts/regenerate_statistical_plots.py --results-dir outputs/statistical_validation
"""

import argparse
import sys
import json
import ast
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualization.statistical_plots import generate_statistical_plots


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate statistical plots from existing results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/statistical_validation",
        help="Directory containing raw_results.csv and statistical_summary.json",
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        sys.exit(1)

    # Load raw results
    raw_results_file = results_dir / "raw_results.csv"
    if not raw_results_file.exists():
        print(f"[ERROR] raw_results.csv not found: {raw_results_file}")
        sys.exit(1)

    print("Loading results...")
    df = pd.read_csv(raw_results_file)

    # Convert DataFrame to list of dicts (format expected by plotting functions)
    results = []
    for _, row in df.iterrows():
        result_dict = {
            "experiment_id": str(row.get("experiment_id", "")),
            "seed": int(row.get("seed", 0)),
            "metrics": {},
        }

        # Extract metrics - handle string representation with numpy types
        metrics = row.get("metrics", {})
        if isinstance(metrics, str):
            try:
                # Create safe namespace for eval (only allow numpy and basic types)
                safe_dict = {
                    "np": np,
                    "__builtins__": {},
                    "float64": np.float64,
                    "float32": np.float32,
                    "int64": np.int64,
                    "int32": np.int32,
                }
                # Use eval with safe namespace to parse the dict string
                metrics = eval(metrics, safe_dict)
            except (ValueError, SyntaxError, TypeError, NameError) as e:
                print(
                    f"Warning: Could not parse metrics for seed {row.get('seed', 'unknown')}: {e}"
                )
                metrics = {}
        elif not isinstance(metrics, dict):
            metrics = {}

        # Convert numpy types to Python native types
        cleaned_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                cleaned_metrics[key] = float(value)
            elif isinstance(value, (int, float)):
                cleaned_metrics[key] = float(value)
            else:
                cleaned_metrics[key] = value

        result_dict["metrics"] = cleaned_metrics
        results.append(result_dict)

    print(f"Loaded {len(results)} experiment results")

    # Regenerate plots
    plots_dir = results_dir / "figures"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\nRegenerating plots...")
    generate_statistical_plots(results, plots_dir)

    print(f"\n[OK] Plots regenerated in: {plots_dir}")


if __name__ == "__main__":
    main()
