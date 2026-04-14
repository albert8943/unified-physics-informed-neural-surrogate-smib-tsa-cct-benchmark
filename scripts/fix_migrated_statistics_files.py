#!/usr/bin/env python
"""
Fix Incorrectly Migrated Statistics Files.

This script fixes the issue where trajectory_statistics files were migrated
instead of the full trajectory_data files. It:
1. Finds incorrectly migrated statistics files in data/common/
2. Locates the corresponding full trajectory_data files in experiment directories
3. Replaces the statistics files with the full data files
4. Updates metadata appropriately
"""

import argparse
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Simple JSON load/save functions to avoid heavy imports
def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict, path: Path):
    """Save JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def find_corresponding_trajectory_data(statistics_file: Path) -> Optional[Path]:
    """
    Find the corresponding full trajectory_data file for a statistics file.

    Parameters:
    -----------
    statistics_file : Path
        Path to trajectory_statistics CSV file

    Returns:
    --------
    trajectory_data_path : Path or None
        Path to corresponding trajectory_data file, or None if not found
    """
    # Try to find the experiment directory from the statistics file path
    # Statistics files are usually in: outputs/experiments/exp_*/analysis/
    # Full data files are usually in: outputs/experiments/exp_*/data/

    # Method 1: Use metadata original_path to find experiment directory
    metadata_file = statistics_file.with_name(statistics_file.stem + "_metadata.json")
    if metadata_file.exists():
        try:
            metadata = load_json(metadata_file)
            original_path = metadata.get("original_path", "")
            if original_path:
                # Extract experiment directory from original_path
                # e.g., "outputs/experiments/exp_20251209_174627/analysis/..."
                original_path_obj = Path(original_path)
                if "experiments" in original_path_obj.parts:
                    exp_idx = None
                    for i, part in enumerate(original_path_obj.parts):
                        if part.startswith("exp_"):
                            exp_idx = i
                            break

                    if exp_idx is not None:
                        # Reconstruct path to experiment directory
                        exp_dir = PROJECT_ROOT / Path(*original_path_obj.parts[: exp_idx + 1])
                        data_dir = exp_dir / "data"

                        if data_dir.exists():
                            # Look for trajectory_data files in data directory
                            trajectory_files = list(data_dir.glob("trajectory_data_*.csv"))
                            if trajectory_files:
                                # Return the most recent one (by modification time)
                                return max(trajectory_files, key=lambda p: p.stat().st_mtime)
        except Exception as e:
            print(f"    Warning: Could not parse metadata: {e}")

    # Method 1b: Check if we can infer from the statistics file location (if in common repo)
    if "analysis" in statistics_file.parts or "common" in statistics_file.parts:
        # Try to find experiment directory from metadata
        pass  # Already handled above

    # Method 2: Search all experiment directories for matching timestamp
    # Extract timestamp from statistics filename (e.g., trajectory_statistics_20251209_174637.csv)
    stats_name = statistics_file.name
    if "_" in stats_name:
        parts = stats_name.replace("trajectory_statistics_", "").replace(".csv", "").split("_")
        if len(parts) >= 2:
            date_part = parts[0]  # e.g., "20251209"
            time_part = parts[1]  # e.g., "174637"

            # Search for trajectory_data files with similar timestamp
            experiments_dir = PROJECT_ROOT / "outputs" / "experiments"
            if experiments_dir.exists():
                for exp_dir in experiments_dir.glob("exp_*"):
                    data_dir = exp_dir / "data"
                    if data_dir.exists():
                        # Look for files with same date
                        matching_files = list(data_dir.glob(f"trajectory_data_{date_part}_*.csv"))
                        if matching_files:
                            # Return the one closest in time
                            return min(
                                matching_files,
                                key=lambda p: (
                                    abs(int(p.stem.split("_")[-1]) - int(time_part))
                                    if len(p.stem.split("_")) > 2
                                    else float("inf")
                                ),
                            )

    # Method 3: Search by reading metadata if available
    metadata_file = statistics_file.parent.parent / "config.yaml"
    if metadata_file.exists():
        # Try to find data file in same experiment
        exp_dir = metadata_file.parent
        data_dir = exp_dir / "data"
        if data_dir.exists():
            trajectory_files = list(data_dir.glob("trajectory_data_*.csv"))
            if trajectory_files:
                return max(trajectory_files, key=lambda p: p.stat().st_mtime)

    return None


def is_statistics_file(file_path: Path) -> bool:
    """Check if a file is a statistics file (not full data)."""
    name_lower = file_path.name.lower()
    return "trajectory_statistics" in name_lower or "_statistics" in name_lower


def find_incorrectly_migrated_files(common_dir: Path) -> list[Tuple[Path, Dict]]:
    """
    Find statistics files that were incorrectly migrated to data/common/.

    Parameters:
    -----------
    common_dir : Path
        Path to data/common/ directory

    Returns:
    --------
    incorrect_files : list
        List of tuples (file_path, metadata_dict) for incorrectly migrated files
    """
    incorrect_files = []

    if not common_dir.exists():
        return incorrect_files

    # Find all CSV files in common directory
    for csv_file in common_dir.glob("*.csv"):
        # Check metadata first (most reliable)
        metadata_file = csv_file.with_name(csv_file.stem + "_metadata.json")
        if metadata_file.exists():
            try:
                metadata = load_json(metadata_file)
                # Check if original_path points to a statistics file
                original_path = metadata.get("original_path", "")
                if (
                    "trajectory_statistics" in original_path.lower()
                    or "analysis" in original_path.lower()
                ):
                    incorrect_files.append((csv_file, metadata))
                    continue
            except Exception as e:
                print(f"Warning: Could not read metadata for {csv_file.name}: {e}")

        # Also check file contents - statistics files don't have time/delta/omega columns
        # They have summary columns like delta_max, delta_mean, etc.
        try:
            df_sample = pd.read_csv(csv_file, nrows=5)
            # Check if it has time-series columns (full data) or summary columns (statistics)
            has_time_series = (
                "time" in df_sample.columns
                and "delta" in df_sample.columns
                and "omega" in df_sample.columns
            )
            has_summary_stats = (
                "delta_max" in df_sample.columns or "delta_mean" in df_sample.columns
            )

            # If it has summary stats but not time-series, it's a statistics file
            if has_summary_stats and not has_time_series:
                # Check metadata again or create empty metadata
                if not metadata_file.exists():
                    incorrect_files.append((csv_file, {}))
                elif (csv_file, {}) not in incorrect_files:  # Avoid duplicates
                    try:
                        metadata = load_json(metadata_file)
                        incorrect_files.append((csv_file, metadata))
                    except:
                        incorrect_files.append((csv_file, {}))
        except Exception as e:
            # If we can't read the file, skip it
            print(f"Warning: Could not read {csv_file.name}: {e}")

    return incorrect_files


def replace_with_full_data(
    statistics_file: Path,
    trajectory_data_file: Path,
    metadata: Dict,
    dry_run: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Replace statistics file with full trajectory data file.

    Parameters:
    -----------
    statistics_file : Path
        Path to incorrectly migrated statistics file
    trajectory_data_file : Path
        Path to full trajectory data file
    metadata : dict
        Existing metadata
    dry_run : bool
        If True, don't actually replace files

    Returns:
    --------
    success : bool
        True if replacement succeeded
    error_msg : str or None
        Error message if failed
    """
    try:
        # Verify trajectory_data_file exists and has correct format
        df = pd.read_csv(trajectory_data_file, nrows=10)
        required_cols = ["time", "delta", "omega", "scenario_id"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Trajectory data file missing required columns: {missing_cols}"

        if dry_run:
            print(f"📋 [DRY RUN] Would replace: {statistics_file.name}")
            print(f"   With: {trajectory_data_file.name}")
            print(
                f"Size: {statistics_file.stat().st_size / 1024:.1f} KB →"
                f"{trajectory_data_file.stat().st_size / 1024 / 1024:.1f} MB"
            )
            return True, None

        # Backup original file
        backup_file = statistics_file.with_suffix(".csv.backup")
        shutil.copy2(statistics_file, backup_file)

        # Copy full data file
        shutil.copy2(trajectory_data_file, statistics_file)

        # Update metadata
        metadata_file = statistics_file.with_suffix(".csv").with_name(
            statistics_file.stem + "_metadata.json"
        )

        # Update original_path to point to the full data file
        metadata["original_path"] = str(trajectory_data_file)
        metadata["migration_timestamp"] = pd.Timestamp.now().isoformat()
        metadata["note"] = "Fixed: Replaced statistics file with full trajectory data"

        # Update statistics in metadata
        df_full = pd.read_csv(trajectory_data_file)
        if "statistics" not in metadata:
            metadata["statistics"] = {}
        metadata["statistics"]["total_rows"] = len(df_full)
        if "scenario_id" in df_full.columns:
            metadata["statistics"]["unique_scenarios"] = df_full["scenario_id"].nunique()

        save_json(metadata, metadata_file)

        print(f"[OK] Replaced: {statistics_file.name}")
        print(f"   Backup saved to: {backup_file.name}")
        print(f"   New size: {statistics_file.stat().st_size / 1024 / 1024:.1f} MB")

        return True, None

    except Exception as e:
        return False, f"Error replacing file: {str(e)}"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Fix incorrectly migrated statistics files in data/common/"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without actually replacing files",
    )
    parser.add_argument(
        "--common-dir",
        type=str,
        default="data/common",
        help="Path to common repository directory (default: data/common)",
    )
    parser.add_argument(
        "--delete-backups",
        action="store_true",
        help="Delete backup files after successful replacement",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FIX INCORRECTLY MIGRATED STATISTICS FILES")
    print("=" * 70)
    print(f"Common directory: {args.common_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    common_dir = PROJECT_ROOT / args.common_dir

    # Find incorrectly migrated files
    print("Searching for incorrectly migrated statistics files...")
    incorrect_files = find_incorrectly_migrated_files(common_dir)

    if not incorrect_files:
        print("[OK] No incorrectly migrated statistics files found.")
        return

    print(f"Found {len(incorrect_files)} incorrectly migrated file(s):")
    for file_path, metadata in incorrect_files:
        print(f"  • {file_path.name}")
    print()

    # Find corresponding full data files and replace
    fixed_count = 0
    not_found_count = 0
    error_count = 0

    for statistics_file, metadata in incorrect_files:
        print(f"\nProcessing: {statistics_file.name}")

        # Find corresponding trajectory_data file
        trajectory_data_file = find_corresponding_trajectory_data(statistics_file)

        if trajectory_data_file is None:
            print(f"  ❌ Could not find corresponding trajectory_data file")
            print(f"     Please manually locate the full data file for: {statistics_file.name}")
            not_found_count += 1
            continue

        if not trajectory_data_file.exists():
            print(f"  ❌ Trajectory data file not found: {trajectory_data_file}")
            not_found_count += 1
            continue

        print(f"  Found: {trajectory_data_file.name}")

        # Replace file
        success, error = replace_with_full_data(
            statistics_file, trajectory_data_file, metadata, args.dry_run
        )

        if success:
            fixed_count += 1
        else:
            print(f"  ❌ {error}")
            error_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("FIX SUMMARY")
    print("=" * 70)
    print(f"Total incorrectly migrated files: {len(incorrect_files)}")
    print(f"Successfully fixed: {fixed_count}")
    print(f"Could not find corresponding data: {not_found_count}")
    print(f"Errors: {error_count}")

    if args.dry_run:
        print("\n⚠️  This was a dry run. Use without --dry-run to actually fix files.")
    else:
        print("\n[OK] Fix complete!")
        if args.delete_backups:
            print("Deleting backup files...")
            for backup in common_dir.glob("*.backup"):
                backup.unlink()
                print(f"  Deleted: {backup.name}")


if __name__ == "__main__":
    main()
