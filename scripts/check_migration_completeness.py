#!/usr/bin/env python
"""
Check that all migrated files contain full generated data (not summary statistics).

This script identifies files that are summary statistics (one row per scenario)
rather than full trajectory data (one row per time point).
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use direct path instead of importing
COMMON_DATA_DIR = PROJECT_ROOT / "data" / "common"


def check_file_type(file_path: Path) -> dict:
    """
    Check if a file is full trajectory data or summary statistics.

    Returns:
    --------
    dict with keys:
        - 'type': 'full', 'summary', or 'unknown'
        - 'rows': total number of rows
        - 'scenarios': number of unique scenarios (if applicable)
        - 'has_time': whether file has 'time' column
        - 'has_param_cols': whether file has param_H, param_D, param_Pm columns
        - 'has_summary_cols': whether file has H, D, Pm, CCT columns (summary format)
    """
    try:
        # Read first 100 rows to check structure
        df_sample = pd.read_csv(file_path, nrows=100)

        # Check for full trajectory data indicators
        has_time = "time" in df_sample.columns
        has_param_cols = all(c in df_sample.columns for c in ["param_H", "param_D", "param_Pm"])

        # Check for summary statistics indicators
        has_summary_cols = all(c in df_sample.columns for c in ["H", "D", "Pm", "CCT"])
        has_no_time = not has_time

        # Determine file type
        if has_time and has_param_cols:
            file_type = "full"
        elif has_summary_cols and has_no_time:
            file_type = "summary"
        else:
            file_type = "unknown"

        # Get full statistics
        df_full = pd.read_csv(file_path)
        total_rows = len(df_full)

        scenarios = None
        if "scenario_id" in df_full.columns:
            scenarios = df_full["scenario_id"].nunique()

        return {
            "type": file_type,
            "rows": total_rows,
            "scenarios": scenarios,
            "has_time": has_time,
            "has_param_cols": has_param_cols,
            "has_summary_cols": has_summary_cols,
        }
    except Exception as e:
        return {
            "type": "error",
            "error": str(e),
            "rows": 0,
        }


def main():
    """Main function to check all migrated files."""
    print("=" * 70)
    print("CHECKING MIGRATION COMPLETENESS")
    print("=" * 70)
    print()

    if not COMMON_DATA_DIR.exists():
        print(f"❌ Common data directory does not exist: {COMMON_DATA_DIR}")
        return

    # Find all trajectory data CSV files
    csv_files = list(COMMON_DATA_DIR.glob("trajectory_data_*.csv"))
    print(f"Found {len(csv_files)} trajectory data CSV files")
    print()

    if not csv_files:
        print("No trajectory data files found.")
        return

    # Check each file
    full_files = []
    summary_files = []
    unknown_files = []
    error_files = []

    for csv_file in sorted(csv_files):
        result = check_file_type(csv_file)

        if result["type"] == "full":
            full_files.append((csv_file, result))
        elif result["type"] == "summary":
            summary_files.append((csv_file, result))
        elif result["type"] == "error":
            error_files.append((csv_file, result))
        else:
            unknown_files.append((csv_file, result))

    # Print summary
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"✅ Full trajectory data files: {len(full_files)}")
    print(f"⚠️  Summary statistics files (incomplete): {len(summary_files)}")
    print(f"❓ Unknown format files: {len(unknown_files)}")
    print(f"❌ Error reading files: {len(error_files)}")
    print()

    # Report summary files (these are incomplete)
    if summary_files:
        print("=" * 70)
        print("⚠️  INCOMPLETE FILES (Summary Statistics Only)")
        print("=" * 70)
        print("These files contain summary statistics (one row per scenario)")
        print("instead of full trajectory data (one row per time point).")
        print()
        for csv_file, result in summary_files:
            print(f"  📊 {csv_file.name}")
            print(f"     Rows: {result['rows']}")
            if result["scenarios"]:
                print(f"     Scenarios: {result['scenarios']}")
            print()

    # Report unknown files
    if unknown_files:
        print("=" * 70)
        print("❓ UNKNOWN FORMAT FILES")
        print("=" * 70)
        for csv_file, result in unknown_files:
            print(f"  ❓ {csv_file.name}")
            print(f"     Rows: {result['rows']}")
            print(f"     Has time column: {result.get('has_time', False)}")
            print(f"     Has param columns: {result.get('has_param_cols', False)}")
            print(f"     Has summary columns: {result.get('has_summary_cols', False)}")
            print()

    # Report error files
    if error_files:
        print("=" * 70)
        print("❌ FILES WITH ERRORS")
        print("=" * 70)
        for csv_file, result in error_files:
            print(f"  ❌ {csv_file.name}")
            print(f"     Error: {result.get('error', 'Unknown error')}")
            print()

    # Show sample of full files
    if full_files:
        print("=" * 70)
        print("✅ SAMPLE OF FULL TRAJECTORY DATA FILES")
        print("=" * 70)
        for csv_file, result in full_files[:5]:
            print(f"  ✅ {csv_file.name}")
            print(f"     Rows: {result['rows']:,}")
            if result["scenarios"]:
                print(f"     Scenarios: {result['scenarios']}")
                avg_rows_per_scenario = (
                    result["rows"] / result["scenarios"] if result["scenarios"] > 0 else 0
                )
                print(f"     Avg rows/scenario: {avg_rows_per_scenario:.1f}")
            print()
        if len(full_files) > 5:
            print(f"  ... and {len(full_files) - 5} more full data files")
        print()

    # Final summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_files = len(csv_files)
    complete_files = len(full_files)
    incomplete_files = len(summary_files)

    if incomplete_files > 0:
        print(
            f"⚠️ WARNING: {incomplete_files} out of {total_files} files are incomplete (summary"
            f"statistics only)"
        )
        print(f"   These files should be replaced with full trajectory data if available.")
    else:
        print(f"✅ All {total_files} files contain full trajectory data!")

    print()


if __name__ == "__main__":
    main()
