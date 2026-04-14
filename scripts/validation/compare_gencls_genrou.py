#!/usr/bin/env python
"""
Compare GENCLS vs GENROU Performance.

Compares PINN performance on GENCLS (training/test) vs GENROU (validation)
to analyze generalization capability.

Usage:
    python scripts/validation/compare_gencls_genrou.py \
        --gencls-results outputs/publication/statistical_validation/experiments/exp_20260119_095052/pinn/results/metrics.json \
        --genrou-results outputs/publication/genrou_validation/genrou_validation_results.json \
        --output-dir outputs/publication/genrou_validation/comparison
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Fix encoding for Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats


def load_gencls_results(gencls_results_path: Path) -> Dict:
    """Load GENCLS evaluation results."""
    with open(gencls_results_path, "r") as f:
        return json.load(f)


def load_genrou_results(genrou_results_path: Path) -> List[Dict]:
    """Load GENROU validation results."""
    with open(genrou_results_path, "r") as f:
        return json.load(f)


def compute_statistical_comparison(gencls_metrics: Dict, genrou_results: List[Dict]) -> Dict:
    """
    Compute statistical comparison between GENCLS and GENROU performance.

    Parameters:
    -----------
    gencls_metrics : dict
        GENCLS evaluation metrics (single experiment)
    genrou_results : list
        List of GENROU validation results (one per scenario)

    Returns:
    --------
    comparison : dict
        Statistical comparison results
    """
    # Extract GENROU metrics per scenario
    genrou_delta_r2 = [
        r.get("delta_r2", np.nan) for r in genrou_results if not np.isnan(r.get("delta_r2", np.nan))
    ]
    genrou_omega_r2 = [
        r.get("omega_r2", np.nan) for r in genrou_results if not np.isnan(r.get("omega_r2", np.nan))
    ]
    genrou_delta_rmse = [
        r.get("delta_rmse", np.nan)
        for r in genrou_results
        if not np.isnan(r.get("delta_rmse", np.nan))
    ]
    genrou_omega_rmse = [
        r.get("omega_rmse", np.nan)
        for r in genrou_results
        if not np.isnan(r.get("omega_rmse", np.nan))
    ]

    # GENCLS metrics (single values, treat as population mean)
    gencls_delta_r2 = gencls_metrics.get("r2_delta", np.nan)
    gencls_omega_r2 = gencls_metrics.get("r2_omega", np.nan)
    gencls_delta_rmse = gencls_metrics.get("rmse_delta", np.nan)
    gencls_omega_rmse = gencls_metrics.get("rmse_omega", np.nan)

    comparison = {
        "gencls": {
            "delta_r2": gencls_delta_r2,
            "omega_r2": gencls_omega_r2,
            "delta_rmse": gencls_delta_rmse,
            "omega_rmse": gencls_omega_rmse,
        },
        "genrou": {
            "delta_r2": {
                "mean": float(np.nanmean(genrou_delta_r2)) if len(genrou_delta_r2) > 0 else np.nan,
                "std": float(np.nanstd(genrou_delta_r2)) if len(genrou_delta_r2) > 0 else np.nan,
                "n": len(genrou_delta_r2),
            },
            "omega_r2": {
                "mean": float(np.nanmean(genrou_omega_r2)) if len(genrou_omega_r2) > 0 else np.nan,
                "std": float(np.nanstd(genrou_omega_r2)) if len(genrou_omega_r2) > 0 else np.nan,
                "n": len(genrou_omega_r2),
            },
            "delta_rmse": {
                "mean": float(np.nanmean(genrou_delta_rmse))
                if len(genrou_delta_rmse) > 0
                else np.nan,
                "std": float(np.nanstd(genrou_delta_rmse))
                if len(genrou_delta_rmse) > 0
                else np.nan,
                "n": len(genrou_delta_rmse),
            },
            "omega_rmse": {
                "mean": float(np.nanmean(genrou_omega_rmse))
                if len(genrou_omega_rmse) > 0
                else np.nan,
                "std": float(np.nanstd(genrou_omega_rmse))
                if len(genrou_omega_rmse) > 0
                else np.nan,
                "n": len(genrou_omega_rmse),
            },
        },
    }

    # Compute confidence intervals (95% CI)
    if len(genrou_delta_r2) > 1:
        ci_delta_r2 = stats.t.interval(
            0.95,
            len(genrou_delta_r2) - 1,
            loc=np.nanmean(genrou_delta_r2),
            scale=stats.sem(genrou_delta_r2, nan_policy="omit"),
        )
        comparison["genrou"]["delta_r2"]["ci_95"] = [float(ci_delta_r2[0]), float(ci_delta_r2[1])]

    if len(genrou_omega_r2) > 1:
        ci_omega_r2 = stats.t.interval(
            0.95,
            len(genrou_omega_r2) - 1,
            loc=np.nanmean(genrou_omega_r2),
            scale=stats.sem(genrou_omega_r2, nan_policy="omit"),
        )
        comparison["genrou"]["omega_r2"]["ci_95"] = [float(ci_omega_r2[0]), float(ci_omega_r2[1])]

    if len(genrou_delta_rmse) > 1:
        ci_delta_rmse = stats.t.interval(
            0.95,
            len(genrou_delta_rmse) - 1,
            loc=np.nanmean(genrou_delta_rmse),
            scale=stats.sem(genrou_delta_rmse, nan_policy="omit"),
        )
        comparison["genrou"]["delta_rmse"]["ci_95"] = [
            float(ci_delta_rmse[0]),
            float(ci_delta_rmse[1]),
        ]

    if len(genrou_omega_rmse) > 1:
        ci_omega_rmse = stats.t.interval(
            0.95,
            len(genrou_omega_rmse) - 1,
            loc=np.nanmean(genrou_omega_rmse),
            scale=stats.sem(genrou_omega_rmse, nan_policy="omit"),
        )
        comparison["genrou"]["omega_rmse"]["ci_95"] = [
            float(ci_omega_rmse[0]),
            float(ci_omega_rmse[1]),
        ]

    # Compute performance degradation
    if not np.isnan(gencls_delta_r2) and not np.isnan(comparison["genrou"]["delta_r2"]["mean"]):
        degradation_delta_r2 = (
            (comparison["genrou"]["delta_r2"]["mean"] - gencls_delta_r2) / gencls_delta_r2
        ) * 100
        comparison["degradation"] = {
            "delta_r2_pct": float(degradation_delta_r2),
            "delta_rmse_pct": float(
                (
                    (comparison["genrou"]["delta_rmse"]["mean"] - gencls_delta_rmse)
                    / gencls_delta_rmse
                )
                * 100
            )
            if not np.isnan(gencls_delta_rmse)
            else np.nan,
        }

    return comparison


def create_comparison_table(comparison: Dict) -> pd.DataFrame:
    """Create comparison table for publication."""
    data = {
        "Metric": ["R² Delta", "R² Omega", "RMSE Delta (rad)", "RMSE Omega (pu)"],
        "GENCLS": [
            f"{comparison['gencls']['delta_r2']:.4f}",
            f"{comparison['gencls']['omega_r2']:.4f}",
            f"{comparison['gencls']['delta_rmse']:.4f}",
            f"{comparison['gencls']['omega_rmse']:.4f}",
        ],
        "GENROU": [
            f"{comparison['genrou']['delta_r2']['mean']:.4f} ± {comparison['genrou']['delta_r2']['std']:.4f}",
            f"{comparison['genrou']['omega_r2']['mean']:.4f} ± {comparison['genrou']['omega_r2']['std']:.4f}",
            f"{comparison['genrou']['delta_rmse']['mean']:.4f} ± {comparison['genrou']['delta_rmse']['std']:.4f}",
            f"{comparison['genrou']['omega_rmse']['mean']:.4f} ± {comparison['genrou']['omega_rmse']['std']:.4f}",
        ],
    }

    if "degradation" in comparison:
        data["Degradation (%)"] = [
            f"{comparison['degradation']['delta_r2_pct']:.1f}%",
            "N/A",
            f"{comparison['degradation']['delta_rmse_pct']:.1f}%",
            "N/A",
        ]

    return pd.DataFrame(data)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare GENCLS vs GENROU performance")
    parser.add_argument(
        "--gencls-results",
        type=str,
        help="Path to GENCLS evaluation results (metrics.json)",
    )
    parser.add_argument(
        "--genrou-results",
        type=str,
        default="outputs/publication/genrou_validation/genrou_validation_results.json",
        help="Path to GENROU validation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/publication/genrou_validation/comparison",
        help="Output directory for comparison results",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GENCLS vs GENROU COMPARISON")
    print("=" * 70)

    # Load results
    genrou_results_path = Path(args.genrou_results)
    if not genrou_results_path.exists():
        raise FileNotFoundError(f"GENROU results not found: {genrou_results_path}")

    genrou_results = load_genrou_results(genrou_results_path)
    print(f"✓ Loaded {len(genrou_results)} GENROU validation results")

    # Load GENCLS results if provided
    gencls_metrics = {}
    if args.gencls_results:
        gencls_results_path = Path(args.gencls_results)
        if gencls_results_path.exists():
            gencls_metrics = load_gencls_results(gencls_results_path)
            print(f"✓ Loaded GENCLS results from: {gencls_results_path}")
        else:
            print(f"⚠️  Warning: GENCLS results not found: {gencls_results_path}")
            print("   Will only report GENROU statistics")
    else:
        print("⚠️  No GENCLS results provided. Will only report GENROU statistics.")

    # Compute comparison
    if gencls_metrics:
        comparison = compute_statistical_comparison(gencls_metrics, genrou_results)
    else:
        # Only GENROU statistics
        comparison = {
            "genrou": {
                "delta_r2": {
                    "mean": float(np.nanmean([r.get("delta_r2", np.nan) for r in genrou_results])),
                    "std": float(np.nanstd([r.get("delta_r2", np.nan) for r in genrou_results])),
                    "n": len(genrou_results),
                },
                "omega_r2": {
                    "mean": float(np.nanmean([r.get("omega_r2", np.nan) for r in genrou_results])),
                    "std": float(np.nanstd([r.get("omega_r2", np.nan) for r in genrou_results])),
                    "n": len(genrou_results),
                },
                "delta_rmse": {
                    "mean": float(
                        np.nanmean([r.get("delta_rmse", np.nan) for r in genrou_results])
                    ),
                    "std": float(np.nanstd([r.get("delta_rmse", np.nan) for r in genrou_results])),
                    "n": len(genrou_results),
                },
                "omega_rmse": {
                    "mean": float(
                        np.nanmean([r.get("omega_rmse", np.nan) for r in genrou_results])
                    ),
                    "std": float(np.nanstd([r.get("omega_rmse", np.nan) for r in genrou_results])),
                    "n": len(genrou_results),
                },
            },
        }

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_file = output_dir / "comparison_results.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"✓ Comparison saved to: {comparison_file}")

    # Create comparison table
    if gencls_metrics:
        table = create_comparison_table(comparison)
        table_file = output_dir / "comparison_table.csv"
        table.to_csv(table_file, index=False)
        print(f"✓ Comparison table saved to: {table_file}")

        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(table.to_string(index=False))

    # Print summary
    print("\n" + "=" * 70)
    print("GENROU VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Number of scenarios: {len(genrou_results)}")
    print(f"\nDelta (Rotor Angle):")
    print(
        f"  R²: {comparison['genrou']['delta_r2']['mean']:.4f} ± {comparison['genrou']['delta_r2']['std']:.4f}"
    )
    if "ci_95" in comparison["genrou"]["delta_r2"]:
        print(
            f"  95% CI: [{comparison['genrou']['delta_r2']['ci_95'][0]:.4f}, {comparison['genrou']['delta_r2']['ci_95'][1]:.4f}]"
        )
    print(
        f"  RMSE: {comparison['genrou']['delta_rmse']['mean']:.4f} ± {comparison['genrou']['delta_rmse']['std']:.4f} rad"
    )

    print(f"\nOmega (Rotor Speed):")
    print(
        f"  R²: {comparison['genrou']['omega_r2']['mean']:.4f} ± {comparison['genrou']['omega_r2']['std']:.4f}"
    )
    if "ci_95" in comparison["genrou"]["omega_r2"]:
        print(
            f"  95% CI: [{comparison['genrou']['omega_r2']['ci_95'][0]:.4f}, {comparison['genrou']['omega_r2']['ci_95'][1]:.4f}]"
        )
    print(
        f"  RMSE: {comparison['genrou']['omega_rmse']['mean']:.4f} ± {comparison['genrou']['omega_rmse']['std']:.4f} pu"
    )

    if "degradation" in comparison:
        print(f"\nPerformance Degradation (GENCLS → GENROU):")
        print(f"  Delta R²: {comparison['degradation']['delta_r2_pct']:.1f}%")
        print(f"  Delta RMSE: {comparison['degradation']['delta_rmse_pct']:.1f}%")

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
