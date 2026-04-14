#!/usr/bin/env python
"""
Verify data files and optionally rename them to include data type prefix.

This script:
1. Checks all files in data/common/ to verify they are full trajectory data
2. Optionally renames files to include 'full_' prefix for clarity
3. Reports any issues found
"""

import shutil
from pathlib import Path

import pandas as pd

COMMON_DATA_DIR = Path("data/common")


def check_file_type(file_path: Path) -> dict:
    """Check file type and return detailed information."""
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
            "has_time": has_time,
            "has_param": has_param,
            "has_summary": has_summary,
        }
    except Exception as e:
        return {"type": "error", "error": str(e), "rows": 0}


def needs_rename(filename: str) -> bool:
    """Check if filename needs 'full_' prefix."""
    return not filename.startswith("full_") and not filename.startswith("summary_")


def rename_file_with_prefix(file_path: Path, dry_run: bool = False) -> bool:
    """Rename file to include 'full_' prefix if it's full trajectory data."""
    result = check_file_type(file_path)

    if result["type"] != "full":
        return False  # Don't rename non-full files

    if not needs_rename(file_path.name):
        return False  # Already has prefix

    new_name = f"full_{file_path.name}"
    new_path = file_path.parent / new_name

    # Also rename metadata file
    metadata_file = file_path.with_suffix(".json").with_name(file_path.stem + "_metadata.json")
    new_metadata_name = f"full_{metadata_file.name}"
    new_metadata_path = metadata_file.parent / new_metadata_name

    if dry_run:
        print(f"  [DRY RUN] Would rename: {file_path.name} -> {new_name}")
        if metadata_file.exists():
            print(f"  [DRY RUN] Would rename: {metadata_file.name} -> {new_metadata_name}")
        return True

    try:
        # Rename CSV file
        file_path.rename(new_path)
        print(f"  Renamed: {file_path.name} -> {new_name}")

        # Rename metadata file
        if metadata_file.exists():
            metadata_file.rename(new_metadata_path)
            print(f"  Renamed metadata: {metadata_file.name} -> {new_metadata_name}")

        return True
    except Exception as e:
        print(f"  ERROR renaming {file_path.name}: {e}")
        return False


def main():
    """Main verification and renaming function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify data files and optionally rename with data type prefix"
    )
    parser.add_argument(
        "--rename", action="store_true", help="Rename files to include 'full_' prefix"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without actually doing it"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("VERIFY AND RENAME DATA FILES")
    print("=" * 70)
    print()

    if args.dry_run:
        print("[DRY RUN MODE - No files will be modified]")
        print()

    if not COMMON_DATA_DIR.exists():
        print(f"Directory does not exist: {COMMON_DATA_DIR}")
        return

    # Find all CSV files
    csv_files = sorted(COMMON_DATA_DIR.glob("trajectory_data_*.csv"))
    print(f"Found {len(csv_files)} trajectory data files")
    print()

    if not csv_files:
        print("No files found.")
        return

    # Check each file
    full_files = []
    summary_files = []
    unknown_files = []
    error_files = []
    files_needing_rename = []

    for csv_file in csv_files:
        result = check_file_type(csv_file)

        if result["type"] == "full":
            full_files.append((csv_file, result))
            if needs_rename(csv_file.name):
                files_needing_rename.append(csv_file)
        elif result["type"] == "summary":
            summary_files.append((csv_file, result))
        elif result["type"] == "error":
            error_files.append((csv_file, result))
        else:
            unknown_files.append((csv_file, result))

    # Report
    print("=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    print(f"Full trajectory data files: {len(full_files)}")
    print(f"Summary statistics files: {len(summary_files)}")
    print(f"Unknown format files: {len(unknown_files)}")
    print(f"Error reading files: {len(error_files)}")
    print()

    if summary_files:
        print("WARNING: Summary statistics files found (should not be in common repository):")
        for csv_file, result in summary_files:
            print(f"  - {csv_file.name} ({result['rows']} rows)")
        print()

    if files_needing_rename:
        print(f"Files needing rename (missing 'full_' prefix): {len(files_needing_rename)}")
        if args.rename:
            print()
            print("Renaming files...")
            print("-" * 70)
            renamed_count = 0
            for csv_file in files_needing_rename:
                if rename_file_with_prefix(csv_file, dry_run=args.dry_run):
                    renamed_count += 1
            print()
            print(f"Renamed {renamed_count} files")
        else:
            print("  (Use --rename to add 'full_' prefix)")
            for csv_file in files_needing_rename[:10]:
                print(f"  - {csv_file.name}")
            if len(files_needing_rename) > 10:
                print(f"  ... and {len(files_needing_rename) - 10} more")
    else:
        print("All full data files already have proper naming!")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if summary_files:
        print(f"WARNING: {len(summary_files)} summary statistics files found")
        print("  These should be removed or stored separately")

    if not summary_files and not error_files:
        print("All files are valid full trajectory data!")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
