#!/usr/bin/env python
"""
Re-migrate incomplete files (summary statistics) with full trajectory data.

This script:
1. Removes incomplete summary statistics files
2. Re-migrates the full trajectory data files
3. Updates the registry
"""

import json
import shutil
from pathlib import Path

import pandas as pd

COMMON_DATA_DIR = Path("data/common")
REGISTRY_PATH = COMMON_DATA_DIR / "registry.json"


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


def find_full_data_for_incomplete(incomplete_file: Path):
    """Try to find full trajectory data for an incomplete file."""
    # Get metadata to find original path
    metadata_path = incomplete_file.with_suffix(".json").with_name(
        incomplete_file.stem + "_metadata.json"
    )

    if not metadata_path.exists():
        return None

    try:
        metadata = json.load(open(metadata_path))
        original_path = Path(metadata.get("original_path", ""))

        # Check if original path exists and is full data
        if original_path.exists():
            original_type = check_file_type(original_path)
            if original_type == "full":
                return original_path

        # Try to find full data in experiment directory
        if "experiments" in str(original_path):
            exp_dir = original_path.parent.parent
            # Look for trajectory_data_*.csv (not trajectory_statistics)
            for pattern in ["trajectory_data_*.csv", "parameter_sweep_data_*.csv"]:
                for full_file in exp_dir.rglob(pattern):
                    if "statistics" not in full_file.name.lower():
                        full_type = check_file_type(full_file)
                        if full_type == "full":
                            return full_file
    except Exception:
        pass

    return None


def remigrate_file(incomplete_file: Path, full_data_path: Path, dry_run: bool = False):
    """Re-migrate an incomplete file with full data."""
    print(f"\nRe-migrating: {incomplete_file.name}")
    print(f"  Full data source: {full_data_path}")

    if dry_run:
        print(f"  [DRY RUN] Would remove: {incomplete_file.name}")
        print(f"  [DRY RUN] Would copy full data to: {COMMON_DATA_DIR}")
        return True

    try:
        # Remove incomplete file and its metadata
        if incomplete_file.exists():
            incomplete_file.unlink()
            print(f"  Removed incomplete file: {incomplete_file.name}")

        metadata_file = incomplete_file.with_suffix(".json").with_name(
            incomplete_file.stem + "_metadata.json"
        )
        if metadata_file.exists():
            metadata_file.unlink()
            print(f"  Removed metadata: {metadata_file.name}")

        # Copy full data file
        new_filename = incomplete_file.name  # Keep same filename structure
        new_path = COMMON_DATA_DIR / new_filename

        shutil.copy2(full_data_path, new_path)
        print(f"  Copied full data: {new_path.name}")

        # Verify it's full data
        file_type = check_file_type(new_path)
        if file_type == "full":
            df = pd.read_csv(new_path)
            scenarios = df["scenario_id"].nunique() if "scenario_id" in df.columns else None
            print(f"  Verified: Full trajectory data ({len(df):,} rows, {scenarios} scenarios)")
        else:
            print(f"  WARNING: File type is {file_type}, not 'full'!")

        # Create/update metadata
        # Try to load original metadata to preserve config
        original_metadata_path = full_data_path.parent / f"{full_data_path.stem}_metadata.json"
        if original_metadata_path.exists():
            try:
                original_metadata = json.load(open(original_metadata_path))
                # Update metadata
                metadata = original_metadata.copy()
                metadata["original_path"] = str(full_data_path)
                metadata["remigration_timestamp"] = pd.Timestamp.now().isoformat()
                metadata[
                    "note"
                ] = "Re-migrated from full trajectory data (replaced summary statistics)"
            except:
                metadata = {
                    "original_path": str(full_data_path),
                    "remigration_timestamp": pd.Timestamp.now().isoformat(),
                    "note": "Re-migrated from full trajectory data (replaced summary statistics)",
                }
        else:
            metadata = {
                "original_path": str(full_data_path),
                "remigration_timestamp": pd.Timestamp.now().isoformat(),
                "note": "Re-migrated from full trajectory data (replaced summary statistics)",
            }

        # Save metadata
        new_metadata_path = new_path.with_suffix(".json").with_name(
            new_path.stem + "_metadata.json"
        )
        with open(new_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Created metadata: {new_metadata_path.name}")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    """Main re-migration function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-migrate incomplete files with full trajectory data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be re-migrated without actually doing it",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("RE-MIGRATE INCOMPLETE FILES")
    print("=" * 70)
    print()

    if args.dry_run:
        print("[DRY RUN MODE - No files will be modified]")
        print()

    # Find incomplete files
    incomplete_files = []
    for csv_file in sorted(COMMON_DATA_DIR.glob("trajectory_data_*.csv")):
        file_type = check_file_type(csv_file)
        if file_type == "summary":
            incomplete_files.append(csv_file)

    print(f"Found {len(incomplete_files)} incomplete files")
    print()

    # Find files with available full data
    files_to_remigrate = []
    files_needing_regeneration = []

    for incomplete_file in incomplete_files:
        full_data_path = find_full_data_for_incomplete(incomplete_file)
        if full_data_path:
            files_to_remigrate.append((incomplete_file, full_data_path))
        else:
            files_needing_regeneration.append(incomplete_file)

    if files_to_remigrate:
        print(f"Files with available full data ({len(files_to_remigrate)}):")
        for incomplete, full_data in files_to_remigrate:
            print(f"  - {incomplete.name}")
            print(f"    Full data: {full_data}")
        print()

        if not args.dry_run:
            print("Re-migrating files...")
            print("-" * 70)

        success_count = 0
        for incomplete, full_data in files_to_remigrate:
            if remigrate_file(incomplete, full_data, dry_run=args.dry_run):
                success_count += 1

        print()
        print(f"Successfully re-migrated: {success_count}/{len(files_to_remigrate)} files")
    else:
        print("No files found with available full data to re-migrate.")

    if files_needing_regeneration:
        print()
        print(f"Files needing regeneration ({len(files_needing_regeneration)}):")
        for f in files_needing_regeneration:
            print(f"  - {f.name}")
        print("  (Full trajectory data not found - these need to be regenerated)")

    print()
    print("=" * 70)
    print("RE-MIGRATION COMPLETE")
    print("=" * 70)

    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to actually re-migrate files.")


if __name__ == "__main__":
    main()
