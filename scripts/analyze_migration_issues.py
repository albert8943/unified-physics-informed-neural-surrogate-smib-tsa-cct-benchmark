#!/usr/bin/env python
"""
Analyze migration issues: duplicates and incomplete files.

This script identifies:
1. Duplicate files (same fingerprint, different timestamps)
2. Incomplete files (summary statistics instead of full trajectory data)
"""

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

COMMON_DATA_DIR = Path("data/common")
REGISTRY_PATH = COMMON_DATA_DIR / "registry.json"


def check_file_type(file_path: Path) -> dict:
    """Check if file is full trajectory data or summary statistics."""
    try:
        df_sample = pd.read_csv(file_path, nrows=10)
        has_time = "time" in df_sample.columns
        has_param = "param_H" in df_sample.columns
        has_summary = "H" in df_sample.columns and "CCT" in df_sample.columns and not has_time

        df_full = pd.read_csv(file_path)
        total_rows = len(df_full)
        scenarios = df_full["scenario_id"].nunique() if "scenario_id" in df_full.columns else None

        if has_time and has_param:
            file_type = "full"
        elif has_summary:
            file_type = "summary"
        else:
            file_type = "unknown"

        return {
            "type": file_type,
            "rows": total_rows,
            "scenarios": scenarios,
        }
    except Exception as e:
        return {"type": "error", "error": str(e), "rows": 0}


def analyze_duplicates():
    """Find duplicate files (same fingerprint, different timestamps)."""
    files = sorted(COMMON_DATA_DIR.glob("trajectory_data_*.csv"))

    # Group by base name (everything before timestamp)
    groups = defaultdict(list)
    for f in files:
        # Extract base name (everything before _YYYYMMDD)
        parts = f.stem.split("_2025")
        if len(parts) > 1:
            base_name = parts[0]
            groups[base_name].append(f)

    duplicates = {k: v for k, v in groups.items() if len(v) > 1}
    return duplicates, files


def analyze_completeness(files):
    """Check which files are complete (full data) vs incomplete (summary)."""
    summary_files = []
    full_files = []
    unknown_files = []

    for f in files:
        result = check_file_type(f)
        if result["type"] == "summary":
            summary_files.append((f, result))
        elif result["type"] == "full":
            full_files.append((f, result))
        else:
            unknown_files.append((f, result))

    return summary_files, full_files, unknown_files


def main():
    """Main analysis function."""
    print("=" * 70)
    print("MIGRATION ANALYSIS REPORT")
    print("=" * 70)
    print()

    # Analyze duplicates
    duplicates, all_files = analyze_duplicates()

    print("1. DUPLICATE FILES ANALYSIS")
    print("-" * 70)
    print(f"Total CSV files: {len(all_files)}")
    print(
        f"Files with duplicate base names (same fingerprint, different timestamps):"
        f"{len(duplicates)}"
    )
    print()

    if duplicates:
        total_duplicate_files = sum(len(v) for v in duplicates.values())
        unique_groups = len(duplicates)
        extra_files = total_duplicate_files - unique_groups

        print(f"  - {unique_groups} unique data configurations")
        print(f"  - {total_duplicate_files} total files (includes duplicates)")
        print(f"  - {extra_files} duplicate files that could be removed")
        print()
        print("Duplicate groups:")
        for base_name, files in sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {base_name}:")
            for f in sorted(files):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    - {f.name} ({size_mb:.2f} MB)")
            print()
    else:
        print("  No duplicates found!")
    print()

    # Analyze completeness
    summary_files, full_files, unknown_files = analyze_completeness(all_files)

    print("2. DATA COMPLETENESS ANALYSIS")
    print("-" * 70)
    print(f"Full trajectory data files: {len(full_files)}")
    print(f"Summary statistics files (INCOMPLETE): {len(summary_files)}")
    print(f"Unknown format files: {len(unknown_files)}")
    print()

    if summary_files:
        print("INCOMPLETE FILES (Summary Statistics Only):")
        print("These files contain one row per scenario instead of full time-series data.")
        print()
        for f, result in summary_files:
            print(f"  - {f.name}")
            print(f"    Rows: {result['rows']}")
            if result.get("scenarios"):
                print(f"    Scenarios: {result['scenarios']}")
            # Check metadata for original path
            metadata_path = f.with_suffix(".json").with_name(f.stem + "_metadata.json")
            if metadata_path.exists():
                try:
                    metadata = json.load(open(metadata_path))
                    original = metadata.get("original_path", "N/A")
                    if "trajectory_statistics" in original or "analysis" in original:
                        print(f"    Original: {Path(original).name}")
                except:
                    pass
            print()
    else:
        print("  All files contain full trajectory data!")
    print()

    # Summary statistics
    print("3. SUMMARY")
    print("-" * 70)
    total_files = len(all_files)
    complete_files = len(full_files)
    incomplete_files = len(summary_files)

    if duplicates:
        total_duplicate_files = sum(len(v) for v in duplicates.values())
        unique_groups = len(duplicates)
        extra_files = total_duplicate_files - unique_groups
        print(f"Duplicate files: {extra_files} files can be removed")
        print(f"  (Keep the most recent file from each duplicate group)")

    if incomplete_files > 0:
        print(f"Incomplete files: {incomplete_files} files are summary statistics only")
        print(f"  These need to be replaced with full trajectory data if available")

    if not duplicates and incomplete_files == 0:
        print("All files are complete and unique!")

    print()
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if duplicates:
        print("1. Remove duplicate files:")
        print("   - Keep the most recent file from each duplicate group")
        print("   - Remove older duplicates to save disk space")
        print()

    if incomplete_files:
        print("2. Replace incomplete files:")
        print("   - Find original full trajectory data files")
        print("   - Re-migrate with full data instead of summary statistics")
        print("   - Or regenerate data if originals are not available")
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()
