#!/usr/bin/env python
"""
Find and fix incomplete files (summary statistics) by locating full trajectory data.

This script:
1. Searches thoroughly in outputs/experiments for full trajectory data
2. Checks if incomplete files are duplicates of existing full files
3. Re-migrates or removes incomplete files as appropriate
"""

import json
import shutil
from pathlib import Path

import pandas as pd

COMMON_DATA_DIR = Path("data/common")
PROJECT_ROOT = Path(__file__).parent.parent


def check_file_type(file_path: Path) -> str:
    """Check if file is full trajectory data or summary statistics."""
    try:
        df_sample = pd.read_csv(file_path, nrows=10)
        has_time = "time" in df_sample.columns
        has_param = "param_H" in df_sample.columns
        has_summary = "H" in df_sample.columns and "CCT" in df_sample.columns and not has_time

        if has_time and has_param:
            return "full"
        elif has_summary:
            return "summary"
        else:
            return "unknown"
    except Exception:
        return "error"


def find_incomplete_files():
    """Find all incomplete files (summary statistics)."""
    incomplete_files = []
    for csv_file in sorted(COMMON_DATA_DIR.glob("trajectory_data_*.csv")):
        file_type = check_file_type(csv_file)
        if file_type == "summary":
            incomplete_files.append(csv_file)
    return incomplete_files


def search_experiment_directories_for_full_data(incomplete_file: Path):
    """Search all experiment directories for full trajectory data matching the incomplete file."""
    # Get metadata to understand what we're looking for
    metadata_path = incomplete_file.with_suffix(".json").with_name(
        incomplete_file.stem + "_metadata.json"
    )

    if not metadata_path.exists():
        return None

    try:
        metadata = json.load(open(metadata_path))
        gen_config = metadata.get("generation_config", {})
        param_ranges = gen_config.get("parameter_ranges", {})

        # Search in outputs/experiments
        exp_dir = PROJECT_ROOT / "outputs" / "experiments"
        if not exp_dir.exists():
            return None

        # Look for trajectory_data_*.csv files (not trajectory_statistics)
        for pattern in ["trajectory_data_*.csv", "parameter_sweep_data_*.csv"]:
            for full_file in exp_dir.rglob(pattern):
                if "statistics" not in full_file.name.lower() and "analysis" not in str(full_file):
                    # Check if it's full data
                    full_type = check_file_type(full_file)
                    if full_type == "full":
                        # Try to match by checking parameter ranges in the file
                        try:
                            df_sample = pd.read_csv(full_file, nrows=1000)
                            if "param_H" in df_sample.columns:
                                h_range = (df_sample["param_H"].min(), df_sample["param_H"].max())
                                d_range = (df_sample["param_D"].min(), df_sample["param_D"].max())
                                pm_range = (
                                    df_sample["param_Pm"].min(),
                                    df_sample["param_Pm"].max(),
                                )

                                # Check if ranges match (with tolerance)
                                target_h = param_ranges.get("H", [])
                                target_d = param_ranges.get("D", [])
                                target_pm = param_ranges.get("Pm", [])

                                if (
                                    target_h
                                    and abs(h_range[0] - target_h[0]) < 0.1
                                    and abs(h_range[1] - target_h[1]) < 0.1
                                ):
                                    if (
                                        target_d
                                        and abs(d_range[0] - target_d[0]) < 0.1
                                        and abs(d_range[1] - target_d[1]) < 0.1
                                    ):
                                        if (
                                            target_pm
                                            and abs(pm_range[0] - target_pm[0]) < 0.1
                                            and abs(pm_range[1] - target_pm[1]) < 0.1
                                        ):
                                            return full_file
                        except Exception:
                            continue
    except Exception:
        pass

    return None


def check_if_duplicate_of_existing_full_file(incomplete_file: Path):
    """Check if this incomplete file is a duplicate of an existing full file."""
    # Get metadata
    metadata_path = incomplete_file.with_suffix(".json").with_name(
        incomplete_file.stem + "_metadata.json"
    )

    if not metadata_path.exists():
        return None

    try:
        metadata = json.load(open(metadata_path))
        gen_config = metadata.get("generation_config", {})
        param_ranges = gen_config.get("parameter_ranges", {})

        # Check all full files in common directory
        for full_file in COMMON_DATA_DIR.glob("trajectory_data_*.csv"):
            if full_file == incomplete_file:
                continue

            full_type = check_file_type(full_file)
            if full_type == "full":
                # Check metadata
                full_metadata_path = full_file.with_suffix(".json").with_name(
                    full_file.stem + "_metadata.json"
                )
                if full_metadata_path.exists():
                    try:
                        full_metadata = json.load(open(full_metadata_path))
                        full_gen_config = full_metadata.get("generation_config", {})
                        full_param_ranges = full_gen_config.get("parameter_ranges", {})

                        # Compare parameter ranges
                        if (
                            param_ranges.get("H") == full_param_ranges.get("H")
                            and param_ranges.get("D") == full_param_ranges.get("D")
                            and param_ranges.get("Pm") == full_param_ranges.get("Pm")
                        ):
                            # Same parameters - check if same fingerprint
                            if metadata.get("data_fingerprint") == full_metadata.get(
                                "data_fingerprint"
                            ):
                                return full_file
                    except Exception:
                        continue
    except Exception:
        pass

    return None


def main():
    """Main function to find and fix incomplete files."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Find and fix incomplete files (summary statistics)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without actually doing it"
    )
    parser.add_argument(
        "--remove-if-no-full-data",
        action="store_true",
        help="Remove incomplete files if full data cannot be found",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("FIND AND FIX INCOMPLETE FILES")
    print("=" * 70)
    print()

    if args.dry_run:
        print("[DRY RUN MODE - No files will be modified]")
        print()

    # Find incomplete files
    incomplete_files = find_incomplete_files()
    print(f"Found {len(incomplete_files)} incomplete files")
    print()

    files_to_remigrate = []
    files_to_remove = []
    files_needing_regeneration = []

    for incomplete_file in incomplete_files:
        print(f"Checking: {incomplete_file.name}")

        # First, check if it's a duplicate of existing full file
        duplicate_full = check_if_duplicate_of_existing_full_file(incomplete_file)
        if duplicate_full:
            print(f"  [DUPLICATE] Found existing full file: {duplicate_full.name}")
            files_to_remove.append((incomplete_file, "duplicate of existing full file"))
            continue

        # Search experiment directories
        full_data = search_experiment_directories_for_full_data(incomplete_file)
        if full_data:
            print(f"  [FOUND] Full data at: {full_data}")
            files_to_remigrate.append((incomplete_file, full_data))
        else:
            print(f"  [NOT FOUND] Full data not found")
            files_needing_regeneration.append(incomplete_file)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files to re-migrate: {len(files_to_remigrate)}")
    print(f"Files to remove (duplicates): {len(files_to_remove)}")
    print(f"Files needing regeneration: {len(files_needing_regeneration)}")
    print()

    # Re-migrate files
    if files_to_remigrate:
        print("=" * 70)
        print("RE-MIGRATING FILES")
        print("=" * 70)
        for incomplete, full_data in files_to_remigrate:
            if args.dry_run:
                print(f"[DRY RUN] Would re-migrate: {incomplete.name}")
                print(f"  Source: {full_data}")
            else:
                try:
                    # Remove incomplete file
                    incomplete.unlink()
                    metadata_file = incomplete.with_suffix(".json").with_name(
                        incomplete.stem + "_metadata.json"
                    )
                    if metadata_file.exists():
                        metadata_file.unlink()

                    # Copy full data
                    shutil.copy2(full_data, incomplete)
                    print(f"[OK] Re-migrated: {incomplete.name}")

                    # Verify
                    file_type = check_file_type(incomplete)
                    if file_type == "full":
                        df = pd.read_csv(incomplete)
                        scenarios = (
                            df["scenario_id"].nunique() if "scenario_id" in df.columns else None
                        )
                        print(f"  Verified: {len(df):,} rows, {scenarios} scenarios")
                except Exception as e:
                    print(f"  ERROR: {e}")
        print()

    # Remove duplicate files
    if files_to_remove:
        print("=" * 70)
        print("REMOVING DUPLICATE FILES")
        print("=" * 70)
        for incomplete, reason in files_to_remove:
            if args.dry_run:
                print(f"[DRY RUN] Would remove: {incomplete.name} ({reason})")
            else:
                try:
                    incomplete.unlink()
                    metadata_file = incomplete.with_suffix(".json").with_name(
                        incomplete.stem + "_metadata.json"
                    )
                    if metadata_file.exists():
                        metadata_file.unlink()
                    print(f"[OK] Removed: {incomplete.name} ({reason})")
                except Exception as e:
                    print(f"  ERROR: {e}")
        print()

    # Files needing regeneration
    if files_needing_regeneration:
        print("=" * 70)
        print("FILES NEEDING REGENERATION")
        print("=" * 70)
        for f in files_needing_regeneration:
            print(f"  - {f.name}")

        if args.remove_if_no_full_data:
            print()
            print("Removing files (--remove-if-no-full-data specified)...")
            for f in files_needing_regeneration:
                if args.dry_run:
                    print(f"[DRY RUN] Would remove: {f.name}")
                else:
                    try:
                        f.unlink()
                        metadata_file = f.with_suffix(".json").with_name(f.stem + "_metadata.json")
                        if metadata_file.exists():
                            metadata_file.unlink()
                        print(f"[OK] Removed: {f.name}")
                    except Exception as e:
                        print(f"  ERROR: {e}")
        else:
            print()
            print("These files need to be regenerated or removed manually.")
            print("Use --remove-if-no-full-data to remove them automatically.")
        print()

    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to actually modify files.")


if __name__ == "__main__":
    main()
