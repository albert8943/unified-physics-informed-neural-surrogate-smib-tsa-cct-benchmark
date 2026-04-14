#!/usr/bin/env python3
"""
Verify Previous Data: Check if Pm was Actually Varying

This script checks your previous data files to see if Pm was actually varying
or if the issue was always present but undetected.

Author: Albert
Date: 2025-12-19
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from glob import glob


def check_pm_variation(data_path: Path, verbose: bool = True):
    """
    Check if Pm was actually varying in previous data.

    Returns:
        dict with analysis results
    """
    if not data_path.exists():
        return {"error": f"File not found: {data_path}"}

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return {"error": f"Could not read file: {e}"}

    results = {
        "file": str(data_path),
        "total_scenarios": 0,
        "param_pm_variation": False,
        "pm_variation": False,
        "pe_t0_variation": False,
        "pm_constant_value": None,
        "pe_t0_constant_value": None,
        "mismatches": [],
    }

    # Check required columns
    required_cols = ["scenario_id", "param_Pm"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        results["error"] = f"Missing columns: {missing_cols}"
        return results

    # Get unique scenarios
    unique_scenarios = df["scenario_id"].unique()
    results["total_scenarios"] = len(unique_scenarios)

    # Check param_Pm variation
    param_pm_values = df.groupby("scenario_id")["param_Pm"].first().values
    unique_param_pm = np.unique(param_pm_values)
    results["param_pm_variation"] = len(unique_param_pm) > 1
    results["param_pm_unique_count"] = len(unique_param_pm)
    results["param_pm_range"] = [float(unique_param_pm.min()), float(unique_param_pm.max())]

    # Check Pm variation (if column exists)
    if "Pm" in df.columns:
        pm_values = df.groupby("scenario_id")["Pm"].first().values
        unique_pm = np.unique(pm_values)
        results["pm_variation"] = len(unique_pm) > 1
        results["pm_unique_count"] = len(unique_pm)
        results["pm_range"] = [float(unique_pm.min()), float(unique_pm.max())]

        if len(unique_pm) == 1:
            results["pm_constant_value"] = float(unique_pm[0])

    # Check Pe(t=0) variation
    if "Pe" in df.columns and "time" in df.columns:
        # Get Pe at minimum time for each scenario
        pe_t0_values = []
        for scenario_id in unique_scenarios:
            scenario_data = df[df["scenario_id"] == scenario_id]
            if len(scenario_data) > 0:
                min_time_idx = scenario_data["time"].idxmin()
                pe_t0 = scenario_data.loc[min_time_idx, "Pe"]
                pe_t0_values.append(pe_t0)

        if pe_t0_values:
            unique_pe_t0 = np.unique(pe_t0_values)
            results["pe_t0_variation"] = len(unique_pe_t0) > 1
            results["pe_t0_unique_count"] = len(unique_pe_t0)
            results["pe_t0_range"] = [float(unique_pe_t0.min()), float(unique_pe_t0.max())]

            if len(unique_pe_t0) == 1:
                results["pe_t0_constant_value"] = float(unique_pe_t0[0])

            # Check mismatches with param_Pm
            for i, scenario_id in enumerate(unique_scenarios[:10]):  # Check first 10
                param_pm = param_pm_values[i]
                pe_t0 = pe_t0_values[i]
                mismatch_pct = 100.0 * abs(pe_t0 - param_pm) / (abs(param_pm) + 1e-12)
                if mismatch_pct > 2.0:  # More than 2% mismatch
                    results["mismatches"].append(
                        {
                            "scenario_id": scenario_id,
                            "param_Pm": float(param_pm),
                            "Pe_t0": float(pe_t0),
                            "mismatch_pct": float(mismatch_pct),
                        }
                    )

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analysis: {data_path.name}")
        print(f"{'='*60}")
        print(f"Total scenarios: {results['total_scenarios']}")
        print(f"\nparam_Pm variation:")
        print(f"  Varies: {results['param_pm_variation']}")
        print(f"  Unique values: {results['param_pm_unique_count']}")
        if results["param_pm_variation"]:
            print(
                f"  Range: {results['param_pm_range'][0]:.6f} - {results['param_pm_range'][1]:.6f} pu"
            )

        if "Pm" in df.columns:
            print(f"\nPm (model truth) variation:")
            print(f"  Varies: {results['pm_variation']}")
            print(f"  Unique values: {results['pm_unique_count']}")
            if results["pm_constant_value"] is not None:
                print(
                    f"  ⚠️  CONSTANT: All scenarios have Pm = {results['pm_constant_value']:.6f} pu"
                )
            elif results["pm_variation"]:
                print(f"  Range: {results['pm_range'][0]:.6f} - {results['pm_range'][1]:.6f} pu")

        if "Pe" in df.columns:
            print(f"\nPe(t=0) variation:")
            print(f"  Varies: {results['pe_t0_variation']}")
            print(f"  Unique values: {results['pe_t0_unique_count']}")
            if results["pe_t0_constant_value"] is not None:
                print(
                    f"⚠️ CONSTANT: All scenarios have Pe(t=0) ="
                    f"{results['pe_t0_constant_value']:.6f} pu"
                )
            elif results["pe_t0_variation"]:
                print(
                    f"  Range: {results['pe_t0_range'][0]:.6f} - {results['pe_t0_range'][1]:.6f} pu"
                )

            if results["mismatches"]:
                print(
                    f"\n⚠️  Found {len(results['mismatches'])} scenarios with "
                    f"Pe(t=0) ≠ param_Pm:"
                )
                for mismatch in results["mismatches"][:5]:  # Show first 5
                    print(
                        f"  Scenario {mismatch['scenario_id']}: "
                        f"param_Pm={mismatch['param_Pm']:.6f}, "
                        f"Pe(t=0)={mismatch['Pe_t0']:.6f}, "
                        f"diff={mismatch['mismatch_pct']:.1f}%"
                    )

    return results


def main():
    """Main function to check previous data files."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify if Pm was actually varying in previous data files"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to data file or directory (supports wildcards)",
        default=None,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory to search for data files",
        default="data",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="File pattern to match (e.g., '*trajectory*.csv')",
        default="*.csv",
    )
    args = parser.parse_args()

    # Find data files
    data_files = []
    if args.data_path:
        # Single file or pattern
        if "*" in args.data_path:
            data_files = glob(args.data_path)
        else:
            data_files = [Path(args.data_path)]
    else:
        # Search in data directory
        data_dir = Path(args.data_dir)
        if data_dir.exists():
            data_files = list(data_dir.rglob(args.pattern))
        else:
            print(f"Error: Data directory not found: {data_dir}")
            return 1

    if not data_files:
        print("No data files found!")
        return 1

    print(f"Found {len(data_files)} data file(s)")
    print("=" * 60)

    all_results = []
    for data_file in sorted(data_files):
        if isinstance(data_file, str):
            data_file = Path(data_file)

        if not data_file.exists():
            continue

        result = check_pm_variation(data_file, verbose=True)
        result["file_path"] = str(data_file)
        all_results.append(result)

    # Summary across all files
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL FILES")
    print(f"{'='*60}")

    files_with_issue = []
    files_without_issue = []

    for result in all_results:
        if "error" in result:
            continue

        has_issue = False
        issues = []

        # Check if Pm is constant
        if result.get("pm_constant_value") is not None:
            has_issue = True
            issues.append(f"Pm constant at {result['pm_constant_value']:.6f} pu")

        # Check if Pe(t=0) is constant
        if result.get("pe_t0_constant_value") is not None:
            has_issue = True
            issues.append(f"Pe(t=0) constant at {result['pe_t0_constant_value']:.6f} pu")

        # Check if param_Pm varies but Pm/Pe doesn't
        if result.get("param_pm_variation") and not result.get("pm_variation"):
            has_issue = True
            issues.append("param_Pm varies but Pm doesn't")

        if result.get("param_pm_variation") and not result.get("pe_t0_variation"):
            has_issue = True
            issues.append("param_Pm varies but Pe(t=0) doesn't")

        if has_issue:
            files_with_issue.append((Path(result["file_path"]).name, issues))
        else:
            files_without_issue.append(Path(result["file_path"]).name)

    if files_with_issue:
        print(f"\n⚠️  FILES WITH ISSUES ({len(files_with_issue)}):")
        for filename, issues in files_with_issue:
            print(f"  {filename}")
            for issue in issues:
                print(f"    - {issue}")

    if files_without_issue:
        print(f"\n✅ FILES WITHOUT ISSUES ({len(files_without_issue)}):")
        for filename in files_without_issue[:5]:  # Show first 5
            print(f"  {filename}")
        if len(files_without_issue) > 5:
            print(f"  ... and {len(files_without_issue) - 5} more")

    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")

    if files_with_issue:
        print("⚠️  The issue was likely present in previous experiments!")
        print("\nWhy you might have missed it:")
        print("  1. You were looking at 'param_Pm' column (which varies) ✓")
        print("  2. But 'Pm' (model truth) or 'Pe(t=0)' was constant ✗")
        print("  3. No verification script was checking this before")
        print("\nRecommendation:")
        print("  - Regenerate data with case file modification fix")
        print("  - Use the new workflow going forward")
    else:
        print("✅ No issues found - Pm was varying correctly!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
