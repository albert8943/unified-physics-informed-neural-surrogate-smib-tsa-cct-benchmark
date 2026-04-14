#!/usr/bin/env python
"""
Cleanup duplicate files in common repository.

This script:
1. Removes duplicate files (same fingerprint, different timestamps) - keeps most recent
2. Identifies incomplete files (summary statistics) that need full trajectory data
3. Checks if full trajectory data exists for incomplete files
4. Updates registry after cleanup
"""

import json
import shutil
from collections import defaultdict
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


def find_duplicates():
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

    # Only return groups with duplicates
    duplicates = {
        k: sorted(v, key=lambda x: x.stat().st_mtime, reverse=True)
        for k, v in groups.items()
        if len(v) > 1
    }
    return duplicates


def find_incomplete_files():
    """Find files that are summary statistics instead of full trajectory data."""
    files = sorted(COMMON_DATA_DIR.glob("trajectory_data_*.csv"))
    incomplete = []

    for f in files:
        file_type = check_file_type(f)
        if file_type == "summary":
            incomplete.append(f)

    return incomplete


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


def load_registry():
    """Load registry if it exists."""
    if REGISTRY_PATH.exists():
        return json.load(open(REGISTRY_PATH))
    return {"fingerprints": {}, "files": []}


def update_registry_after_cleanup(removed_files):
    """Update registry to remove entries for deleted files."""
    if not REGISTRY_PATH.exists():
        return

    registry = load_registry()
    removed_filenames = {f.name for f in removed_files}

    # Remove from files list
    registry["files"] = [
        f for f in registry.get("files", []) if f.get("filename") not in removed_filenames
    ]

    # Update fingerprints to point to remaining files
    for fp_key, fp_data in registry.get("fingerprints", {}).items():
        if fp_data.get("filename") in removed_filenames:
            # Find replacement file with same fingerprint
            base_name = fp_data.get("filename", "").split("_2025")[0]
            remaining = list(COMMON_DATA_DIR.glob(f"{base_name}_*.csv"))
            if remaining:
                # Use most recent
                replacement = max(remaining, key=lambda x: x.stat().st_mtime)
                fp_data["filename"] = replacement.name
                # Update timestamp
                timestamp_part = (
                    replacement.stem.split("_2025")[1] if "_2025" in replacement.stem else ""
                )
                if timestamp_part:
                    fp_data["timestamp"] = f"2025{timestamp_part}"

    # Save updated registry
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def main():
    """Main cleanup function."""
    import argparse

    parser = argparse.ArgumentParser(description="Cleanup duplicate files in common repository")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing files",
    )
    parser.add_argument(
        "--keep-all-duplicates",
        action="store_true",
        help="Don't remove duplicates, only report incomplete files",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CLEANUP MIGRATION DUPLICATES")
    print("=" * 70)
    print()

    if args.dry_run:
        print("[DRY RUN MODE - No files will be deleted]")
        print()

    # Find duplicates
    duplicates = find_duplicates()

    if duplicates and not args.keep_all_duplicates:
        print("1. DUPLICATE FILES")
        print("-" * 70)
        total_duplicates = sum(len(v) - 1 for v in duplicates.values())
        total_size_mb = 0

        files_to_remove = []
        files_to_keep = []

        for base_name, files in sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True):
            # Keep most recent (first in sorted list)
            keep_file = files[0]
            remove_files = files[1:]

            files_to_keep.append(keep_file)
            files_to_remove.extend(remove_files)

            group_size_mb = sum(f.stat().st_size / (1024 * 1024) for f in remove_files)
            total_size_mb += group_size_mb

            print(f"  {base_name}:")
            print(f"    Keeping: {keep_file.name} (most recent)")
            for f in remove_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    Removing: {f.name} ({size_mb:.2f} MB)")
            print()

        print(f"Total duplicate files to remove: {total_duplicates}")
        print(f"Total space to free: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
        print()

        if not args.dry_run:
            print("Removing duplicate files...")
            removed_count = 0
            for f in files_to_remove:
                try:
                    # Remove CSV file
                    f.unlink()
                    # Remove metadata file if exists
                    metadata_file = f.with_suffix(".json").with_name(f.stem + "_metadata.json")
                    if metadata_file.exists():
                        metadata_file.unlink()
                    removed_count += 1
                    print(f"  Removed: {f.name}")
                except Exception as e:
                    print(f"  Error removing {f.name}: {e}")

            print(f"\nRemoved {removed_count} duplicate files")

            # Update registry
            print("Updating registry...")
            update_registry_after_cleanup(files_to_remove)
            print("Registry updated")
            print()
    elif duplicates:
        print("1. DUPLICATE FILES (keeping all - use without --keep-all-duplicates to remove)")
        print("-" * 70)
        for base_name, files in sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {base_name}: {len(files)} duplicate files")
        print()
    else:
        print("1. DUPLICATE FILES")
        print("-" * 70)
        print("  No duplicates found!")
        print()

    # Find incomplete files
    incomplete_files = find_incomplete_files()

    print("2. INCOMPLETE FILES (Summary Statistics)")
    print("-" * 70)

    if incomplete_files:
        print(f"Found {len(incomplete_files)} incomplete files")
        print()

        files_with_full_data = []
        files_needing_regeneration = []

        for incomplete_file in incomplete_files:
            print(f"  {incomplete_file.name}")

            # Check file size
            df = pd.read_csv(incomplete_file)
            print(f"    Rows: {len(df)} (should be ~5,000+ per scenario)")

            # Try to find full data
            full_data_path = find_full_data_for_incomplete(incomplete_file)
            if full_data_path:
                print(f"    [FOUND] Full data available at: {full_data_path}")
                files_with_full_data.append((incomplete_file, full_data_path))
            else:
                print(f"    [NOT FOUND] Full data not found - needs regeneration")
                files_needing_regeneration.append(incomplete_file)
            print()

        if files_with_full_data:
            print("Files with available full data:")
            for incomplete, full_data in files_with_full_data:
                print(f"  - {incomplete.name}")
                print(f"    Full data: {full_data}")
                print(f"    Action: Re-migrate with full data")
            print()

        if files_needing_regeneration:
            print("Files needing regeneration:")
            for f in files_needing_regeneration:
                print(f"  - {f.name}")
            print()
    else:
        print("  No incomplete files found!")
        print()

    print("=" * 70)
    print("CLEANUP COMPLETE")
    print("=" * 70)

    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to actually remove files.")


if __name__ == "__main__":
    main()
