#!/usr/bin/env python
"""
Re-evaluate statistical validation experiments on correct test data.

This script re-evaluates already-trained models from statistical validation
on the proper test data (instead of training data).
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.evaluation import evaluate_model
from scripts.core.utils import load_config, load_json, save_json
from scripts.analysis.statistical_summary import compute_statistical_summary


def reevaluate_experiments(
    experiments_dir: Path,
    test_data_path: Path,
    config_path: Path,
) -> List[Dict]:
    """Re-evaluate all experiments on correct test data."""
    config = load_config(config_path)
    all_results = []

    # Find all experiment directories
    exp_dirs = sorted(experiments_dir.glob("exp_*"))
    print(f"Found {len(exp_dirs)} experiments to re-evaluate")

    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue

        exp_id = exp_dir.name
        print(f"\n{'='*70}")
        print(f"Re-evaluating: {exp_id}")
        print(f"{'='*70}")

        # Find best model
        model_files = list(exp_dir.glob("**/best_model_*.pth"))
        if not model_files:
            print(f"  ⚠️  No model found, skipping")
            continue

        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        print(f"  Model: {model_path.name}")

        # Re-evaluate on test data
        try:
            evaluation_results = evaluate_model(
                config=config,
                model_path=model_path,
                test_data_path=test_data_path,
                output_dir=exp_dir,
            )

            # Extract metrics
            metrics = evaluation_results.get("metrics", {}) if evaluation_results else {}

            # Load original results to preserve other info
            original_summary_path = exp_dir / "experiment_summary.json"
            original_results = {}
            if original_summary_path.exists():
                original_results = load_json(original_summary_path)

            # Update with new metrics
            if "pinn" not in original_results:
                original_results["pinn"] = {}
            if "evaluation" not in original_results["pinn"]:
                original_results["pinn"]["evaluation"] = {}
            original_results["pinn"]["evaluation"]["metrics"] = metrics

            # Save updated summary
            save_json(original_results, original_summary_path)

            # Extract seed from config if available
            exp_config_path = exp_dir / "config.yaml"
            seed = 0
            if exp_config_path.exists():
                exp_config = load_config(exp_config_path)
                seed = exp_config.get("seed", 0)

            results = {
                "experiment_id": exp_id,
                "seed": seed,
                "metrics": metrics,
                "model_path": str(model_path),
                "experiment_dir": str(exp_dir),
            }
            all_results.append(results)

            print(f"  ✅ Re-evaluation complete")
            if metrics:
                print(f"     R² Delta: {metrics.get('r2_delta', 'N/A'):.4f}")
                print(f"     RMSE Delta: {metrics.get('rmse_delta', 'N/A'):.4f} rad")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback

            traceback.print_exc()
            continue

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate statistical validation experiments on correct test data"
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        required=True,
        help="Directory containing experiment directories",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        required=True,
        help="Path to test data CSV file",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for updated statistical summary (default: experiments_dir parent)",
    )

    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    test_data_path = Path(args.test_data_path)
    config_path = Path(args.config)

    if not experiments_dir.exists():
        print(f"❌ Experiments directory not found: {experiments_dir}")
        sys.exit(1)

    if not test_data_path.exists():
        print(f"❌ Test data not found: {test_data_path}")
        sys.exit(1)

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    print("=" * 70)
    print("RE-EVALUATING STATISTICAL VALIDATION EXPERIMENTS")
    print("=" * 70)
    print(f"Experiments: {experiments_dir}")
    print(f"Test data: {test_data_path}")
    print(f"Config: {config_path}")
    print("=" * 70)

    # Re-evaluate all experiments
    all_results = reevaluate_experiments(
        experiments_dir=experiments_dir,
        test_data_path=test_data_path,
        config_path=config_path,
    )

    if not all_results:
        print("❌ No experiments were successfully re-evaluated")
        sys.exit(1)

    # Save updated raw results
    output_dir = Path(args.output_dir) if args.output_dir else experiments_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    results_df = pd.DataFrame(all_results)
    raw_results_file = output_dir / "raw_results.csv"
    results_df.to_csv(raw_results_file, index=False)
    print(f"\n✅ Updated raw results saved to: {raw_results_file}")

    # Compute updated statistical summary
    print("\n" + "=" * 70)
    print("COMPUTING UPDATED STATISTICAL SUMMARY")
    print("=" * 70)

    from scripts.analysis.statistical_summary import compute_statistical_summary

    summary = compute_statistical_summary(all_results)
    summary_file = output_dir / "statistical_summary.json"
    save_json(summary, summary_file)
    print(f"✅ Updated statistical summary saved to: {summary_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("UPDATED STATISTICAL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nNumber of successful runs: {len(all_results)}")
    print(f"\nMetrics Summary:")
    for metric_name, stats in summary.items():
        if isinstance(stats, dict) and "mean" in stats:
            mean = stats["mean"]
            std = stats.get("std", 0.0)
            ci_lower = stats.get("ci_lower", mean - 1.96 * std)
            ci_upper = stats.get("ci_upper", mean + 1.96 * std)
            print(f"  {metric_name}:")
            print(f"    Mean: {mean:.6f}")
            print(f"    Std:  {std:.6f}")
            print(f"    95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")

    print(f"\n✅ Re-evaluation complete!")
    print(f"  Results: {output_dir}")


if __name__ == "__main__":
    main()
