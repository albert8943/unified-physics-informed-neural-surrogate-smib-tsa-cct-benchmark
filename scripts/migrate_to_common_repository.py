#!/usr/bin/env python
"""
Migration Script: Organize Existing Data to Common Repository.

This script migrates existing data files from old locations (data/generated/,
outputs/experiments/*/data/) to the new common repository structure (data/common/)
with proper fingerprinting, metadata, and registry updates.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.common_repository import (
    COMMON_DATA_DIR,
    _ensure_common_directories,
    _update_data_registry,
    compute_file_checksum,
    generate_data_filename,
    generate_timestamp,
    get_git_branch,
    get_git_commit,
    get_package_versions,
    load_json,
    save_json,
)
from scripts.core.fingerprinting import compute_data_fingerprint


def infer_task_from_filename(filename: str) -> str:
    """
    Infer task type from filename.

    Parameters:
    -----------
    filename : str
        Filename to analyze

    Returns:
    --------
    task : str
        Inferred task type: 'trajectory' or 'parameter_estimation'
    """
    filename_lower = filename.lower()
    if "parameter_estimation" in filename_lower or "parameter_est" in filename_lower:
        return "parameter_estimation"
    elif "cct" in filename_lower:
        # CCT uses trajectory data
        return "trajectory"
    else:
        # Default to trajectory
        return "trajectory"


def find_experiment_config_from_path(data_path: Path) -> Optional[Dict]:
    """
    Try to find experiment config from data path.

    Parameters:
    -----------
    data_path : Path
        Path to data file

    Returns:
    --------
    config : dict or None
        Experiment config if found
    """
    try:
        # Check if data is in outputs/experiments directory
        path_parts = data_path.parts
        for i, part in enumerate(path_parts):
            if part.startswith("exp_") and len(part) > 4:
                exp_dir = PROJECT_ROOT / Path(*path_parts[: i + 1])
                # Try experiment_summary.json first
                summary_path = exp_dir / "experiment_summary.json"
                if summary_path.exists():
                    summary = load_json(summary_path)
                    return summary.get("config", summary)
                # Try reproducibility.json as fallback
                repro_path = exp_dir / "reproducibility.json"
                if repro_path.exists():
                    repro = load_json(repro_path)
                    return repro.get("config", repro)
    except Exception:
        pass

    return None


def extract_generation_config_from_exp(exp_config: Dict) -> Optional[Dict]:
    """Extract generation config from experiment config."""
    if "data" in exp_config and "generation" in exp_config["data"]:
        return exp_config["data"]["generation"]
    elif "generation" in exp_config:
        return exp_config["generation"]
    return None


def infer_config_from_data(data_path: Path, task: str) -> Dict:
    """
    Infer configuration from data file by analyzing its contents.
    First tries to find experiment config, then falls back to inferring from data.

    Parameters:
    -----------
    data_path : Path
        Path to data file
    task : str
        Task type

    Returns:
    --------
    config : dict
        Inferred configuration dictionary
    """
    # First, try to find experiment config
    exp_config = find_experiment_config_from_path(data_path)
    if exp_config:
        gen_config_exp = extract_generation_config_from_exp(exp_config)
        if gen_config_exp:
            # Use experiment config as base
            config = {
                "data": {
                    "task": task,
                    "generation": gen_config_exp.copy(),  # Use actual config values
                },
                "reproducibility": exp_config.get("reproducibility", {"random_seed": None}),
            }
            # Still need to infer actual parameter ranges from data (for filename)
            # But use config values for other fields
            try:
                df = pd.read_csv(data_path, nrows=1000)
                param_ranges = {}
                if "H" in df.columns:
                    h_min, h_max = df["H"].min(), df["H"].max()
                    param_ranges["H"] = [round(float(h_min), 2), round(float(h_max), 2), 5]
                if "D" in df.columns:
                    d_min, d_max = df["D"].min(), df["D"].max()
                    param_ranges["D"] = [round(float(d_min), 2), round(float(d_max), 2), 5]
                if "Pm" in df.columns:
                    pm_min, pm_max = df["Pm"].min(), df["Pm"].max()
                    param_ranges["Pm"] = [round(float(pm_min), 2), round(float(pm_max), 2), 5]
                # Update with actual sampled ranges (for filename)
                if param_ranges:
                    config["data"]["generation"]["parameter_ranges"] = param_ranges
            except Exception:
                pass  # Keep config parameter ranges if can't read data
            return config

    # Fallback: infer from data only
    try:
        # Read a sample of the data to infer parameters
        df = pd.read_csv(data_path, nrows=1000)

        # Infer parameter ranges from data
        # Round to reasonable values to avoid overly precise filenames
        param_ranges = {}
        if "H" in df.columns:
            h_min, h_max = df["H"].min(), df["H"].max()
            # Round to 2 decimal places for cleaner filenames
            param_ranges["H"] = [round(float(h_min), 2), round(float(h_max), 2), 5]

        if "D" in df.columns:
            d_min, d_max = df["D"].min(), df["D"].max()
            param_ranges["D"] = [round(float(d_min), 2), round(float(d_max), 2), 5]

        if "Pm" in df.columns:
            pm_min, pm_max = df["Pm"].min(), df["Pm"].max()
            param_ranges["Pm"] = [round(float(pm_min), 2), round(float(pm_max), 2), 5]

        # Build config structure
        config = {
            "data": {
                "task": task,
                "generation": {
                    "case_file": "smib/SMIB.json",  # Default
                    "parameter_ranges": param_ranges,
                    "sampling_strategy": "full_factorial",  # Default (will be corrected later)
                    "simulation_time": 5.0,  # Default
                    "time_step": 0.001,  # Default
                },
            },
            "reproducibility": {
                "random_seed": None,  # Unknown for existing data
            },
        }

        return config
    except Exception as e:
        print(f"⚠️  Warning: Could not infer config from {data_path.name}: {e}")
        # Return default config
        return {
            "data": {
                "task": task,
                "generation": {
                    "case_file": "smib/SMIB.json",
                    "parameter_ranges": {
                        "H": [2.0, 10.0, 5],
                        "D": [0.5, 3.0, 5],
                    },
                    "sampling_strategy": "full_factorial",
                    "simulation_time": 5.0,
                    "time_step": 0.001,
                },
            },
            "reproducibility": {"random_seed": None},
        }


def is_summary_statistics_file(file_path: Path) -> bool:
    """Check if a file is summary statistics (should be excluded from migration)."""
    try:
        df_sample = pd.read_csv(file_path, nrows=10)
        # Summary statistics have H, D, Pm, CCT columns but no 'time' column
        has_summary_cols = all(c in df_sample.columns for c in ["H", "D", "Pm", "CCT"])
        has_no_time = "time" not in df_sample.columns
        has_no_param_cols = "param_H" not in df_sample.columns
        return has_summary_cols and has_no_time and has_no_param_cols
    except Exception:
        return False


def find_existing_data_files(
    search_dirs: List[Path], include_preprocessed: bool = False
) -> List[Path]:
    """
    Find all existing data files in old locations.

    Parameters:
    -----------
    search_dirs : list
        List of directories to search
    include_preprocessed : bool
        If True, also search for preprocessed data files (train/val/test splits)

    Returns:
    --------
    data_files : list
        List of found data file paths
    """
    data_files = []
    patterns = [
        "**/*parameter_sweep*.csv",
        "**/*trajectory*.csv",
        "**/*parameter_estimation*.csv",
        "**/*parameter_est*.csv",
        "**/*cct*.csv",
    ]

    # Add preprocessed data patterns if requested
    if include_preprocessed:
        patterns.extend(
            [
                "**/train_data_*.csv",
                "**/val_data_*.csv",
                "**/test_data_*.csv",
            ]
        )

    # Files to exclude (these are derived/analysis files, not raw training data)
    exclude_patterns = [
        "*trajectory_statistics*.csv",  # Exclude summary statistics
        "*analysis*.csv",  # Exclude analysis files
        "*summary*.csv",  # Exclude summary files
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for pattern in patterns:
            found = list(search_dir.glob(pattern))
            # Filter out excluded files (derived/analysis files, not raw training data)
            for file_path in found:
                filename = file_path.name.lower()
                should_exclude = any(
                    exclude_term in filename
                    for exclude_term in ["trajectory_statistics", "analysis", "_summary_"]
                )
                # Also check if it's summary statistics by content
                if not should_exclude and is_summary_statistics_file(file_path):
                    print(f"  Excluding summary statistics file: {file_path.name}")
                    should_exclude = True
                if not should_exclude:
                    data_files.append(file_path)

    # Remove duplicates and sort
    data_files = sorted(set(data_files), key=lambda p: p.stat().st_mtime, reverse=True)
    return data_files


def find_preprocessing_metadata(preprocessed_file: Path) -> Optional[Dict]:
    """
    Find preprocessing metadata JSON file for a preprocessed data file.

    Parameters:
    -----------
    preprocessed_file : Path
        Path to preprocessed data file (train_data_*.csv, val_data_*.csv, test_data_*.csv)

    Returns:
    --------
    metadata : dict or None
        Preprocessing metadata if found
    """
    # Look for preprocessing_metadata_*.json in the same directory
    preprocessed_dir = preprocessed_file.parent
    metadata_files = list(preprocessed_dir.glob("preprocessing_metadata_*.json"))

    if metadata_files:
        # Use the most recent metadata file
        metadata_file = max(metadata_files, key=lambda p: p.stat().st_mtime)
        try:
            return load_json(metadata_file)
        except Exception:
            pass

    return None


def is_preprocessed_file(filename: str) -> bool:
    """Check if a filename indicates preprocessed data."""
    filename_lower = filename.lower()
    return any(x in filename_lower for x in ["train_data_", "val_data_", "test_data_"])


def migrate_preprocessed_data_file(
    data_path: Path, dry_run: bool = False, overwrite: bool = False
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Migrate a preprocessed data file (train/val/test split) to common repository.

    Preprocessed data is stored with a special naming scheme that includes:
    - Source data fingerprint
    - Split type (train/val/test)
    - Preprocessing parameters

    Parameters:
    -----------
    data_path : Path
        Path to preprocessed data file
    dry_run : bool
        If True, don't actually move files
    overwrite : bool
        If True, overwrite existing files

    Returns:
    --------
    success : bool
        True if migration succeeded
    new_path : Path or None
        New path in common repository
    error_msg : str or None
        Error message if failed
    """
    try:
        # Find preprocessing metadata
        prep_metadata = find_preprocessing_metadata(data_path)

        if not prep_metadata:
            return False, None, "No preprocessing metadata found"

        # Determine split type
        filename_lower = data_path.name.lower()
        if "train_data_" in filename_lower:
            split_type = "train"
        elif "val_data_" in filename_lower:
            split_type = "val"
        elif "test_data_" in filename_lower:
            split_type = "test"
        else:
            return False, None, f"Unknown split type in filename: {data_path.name}"

        # Get source data path from metadata
        source_file = prep_metadata.get("input_file")
        if not source_file:
            return False, None, "No source data file found in preprocessing metadata"

        source_path = Path(source_file)
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path

        # Try to find source data in common repository
        # First, try to infer config from source file
        if source_path.exists():
            task = infer_task_from_filename(source_path.name)
            source_config = infer_config_from_data(source_path, task)
            source_fingerprint = compute_data_fingerprint(source_config)
        else:
            # Source file doesn't exist, can't determine fingerprint
            # Use a hash of the preprocessing parameters instead
            prep_params = prep_metadata.get("splitting", {})
            param_str = json.dumps(prep_params, sort_keys=True)
            source_fingerprint = hashlib.md5(param_str.encode()).hexdigest()[:8]
            task = "trajectory"  # Default

        # Create filename for preprocessed data
        # Format: {split_type}_data_{source_fingerprint}_{prep_params_hash}_{timestamp}.csv
        prep_params = prep_metadata.get("splitting", {})
        prep_hash = hashlib.md5(json.dumps(prep_params, sort_keys=True).encode()).hexdigest()[:8]
        timestamp = generate_timestamp()

        new_filename = (
            f"{split_type}_data_"
            f"{source_fingerprint[:8] if isinstance(source_fingerprint, str) else str(source_fingerprint)[:8]}_"
            f"{prep_hash}_{timestamp}.csv"
        )
        new_path = COMMON_DATA_DIR / new_filename

        if dry_run:
            print(f"📋 [DRY RUN] Would migrate preprocessed: {data_path.name} → {new_filename}")
            return True, new_path, None

        # Copy file
        import shutil

        shutil.copy2(data_path, new_path)
        print(f"✓ Migrated preprocessed: {data_path.name} → {new_filename}")

        # Create metadata
        file_checksum = compute_file_checksum(new_path)
        df = pd.read_csv(data_path)

        metadata = {
            "data_type": "preprocessed",
            "split_type": split_type,
            "source_data_file": str(source_file),
            "source_fingerprint": (
                source_fingerprint[:8]
                if isinstance(source_fingerprint, str)
                else str(source_fingerprint)[:8]
            ),
            "preprocessing_metadata": prep_metadata,
            "file_checksum": file_checksum,
            "statistics": {
                "total_rows": len(df),
                "unique_scenarios": (
                    df["scenario_id"].nunique() if "scenario_id" in df.columns else None
                ),
            },
            "reproducibility": {
                "git_commit": get_git_commit(),
                "git_branch": get_git_branch(),
                "package_versions": get_package_versions(),
                "timestamp": datetime.now().isoformat(),
            },
        }

        # Save metadata
        metadata_path = new_path.with_suffix(".json").with_name(new_path.stem + "_metadata.json")
        save_json(metadata, metadata_path)

        return True, new_path, None

    except Exception as e:
        return False, None, f"Error migrating preprocessed data: {str(e)}"


def migrate_data_file(
    data_path: Path, dry_run: bool = False, overwrite: bool = False
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Migrate a single data file to common repository.

    Parameters:
    -----------
    data_path : Path
        Path to data file to migrate
    dry_run : bool
        If True, don't actually move files (default: False)
    overwrite : bool
        If True, overwrite existing files in common repository (default: False)

    Returns:
    --------
    success : bool
        True if migration succeeded
    new_path : Path or None
        New path in common repository (None if failed)
    error_msg : str or None
        Error message if failed
    """
    try:
        # Check if this is preprocessed data
        is_preprocessed = is_preprocessed_file(data_path.name)

        if is_preprocessed:
            # Handle preprocessed data differently
            return migrate_preprocessed_data_file(data_path, dry_run, overwrite)

        # Infer task type
        task = infer_task_from_filename(data_path.name)

        # Infer config from data
        config = infer_config_from_data(data_path, task)

        # Compute fingerprint
        fingerprint = compute_data_fingerprint(config)

        # Check if already exists in common repository
        from scripts.core.common_repository import find_data_by_fingerprint

        existing = find_data_by_fingerprint(fingerprint, task)
        if existing and existing.exists() and not overwrite:
            print(
                f"⏭️ Skipping {data_path.name} (duplicate fingerprint, already exists:"
                f"{existing.name})"
            )
            return True, None, None  # Return None for new_path to indicate it was skipped

        # Read data to get sample count
        # n_samples should be the number of unique parameter combinations (H, D, Pm)
        # NOT the number of scenarios/trajectories
        # Handle both formats: full trajectory data (param_H, param_D, param_Pm) and summary stats (H, D, Pm)
        df = pd.read_csv(data_path)
        if all(c in df.columns for c in ["param_H", "param_D", "param_Pm"]):
            # Full trajectory data format: count unique parameter combinations
            param_combos = df[["param_H", "param_D", "param_Pm"]].drop_duplicates()
            n_samples = len(param_combos)
        elif all(c in df.columns for c in ["H", "D", "Pm"]):
            # Summary statistics format: count unique parameter combinations
            param_combos = df[["H", "D", "Pm"]].drop_duplicates()
            n_samples = len(param_combos)
        elif "scenario_id" in df.columns:
            # Fallback: try to infer from scenarios (less accurate)
            n_scenarios = df["scenario_id"].nunique()
            # Assume typical n_samples_per_combination = 5
            n_samples = round(n_scenarios / 5)
            if n_samples == 0:
                n_samples = 1
        else:
            n_samples = len(df)

        # Generate new filename
        timestamp = generate_timestamp()
        new_filename = generate_data_filename(task, config, fingerprint, n_samples, timestamp)
        new_path = COMMON_DATA_DIR / new_filename

        if dry_run:
            print(f"📋 [DRY RUN] Would migrate: {data_path.name} → {new_filename}")
            return True, new_path, None

        # Copy file (don't move, keep original)
        import shutil

        shutil.copy2(data_path, new_path)
        print(f"✓ Migrated: {data_path.name} → {new_filename}")

        # Compute checksum
        file_checksum = compute_file_checksum(new_path)

        # Generate metadata
        metadata = {
            "data_fingerprint": fingerprint,
            "file_checksum": file_checksum,
            "task": task,
            "n_samples": n_samples,
            "filename": new_filename,
            "original_path": str(data_path),
            "migration_timestamp": datetime.now().isoformat(),
            "generation_config": config.get("data", {}).get("generation", {}),
            "statistics": {
                "total_rows": len(df),
                "unique_scenarios": (
                    df["scenario_id"].nunique() if "scenario_id" in df.columns else None
                ),
            },
            "reproducibility": {
                "git_commit": get_git_commit(),
                "git_branch": get_git_branch(),
                "package_versions": get_package_versions(),
                "python_version": sys.version,
                "random_seed": config.get("reproducibility", {}).get("random_seed"),
                "timestamp": datetime.now().isoformat(),
                "note": "Migrated from legacy location - some reproducibility info may be incomplete",
            },
            "models_trained": [],
        }

        # Save metadata
        metadata_path = new_path.with_suffix(".json").with_name(new_path.stem + "_metadata.json")
        save_json(metadata, metadata_path)

        # Update registry
        _update_data_registry(fingerprint, new_filename, task, n_samples, config, timestamp)

        return True, new_path, None

    except Exception as e:
        error_msg = f"Error migrating {data_path.name}: {str(e)}"
        print(f"❌ {error_msg}")
        return False, None, error_msg


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate existing data files to common repository structure"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually moving files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in common repository",
    )
    parser.add_argument(
        "--search-dirs",
        nargs="+",
        default=[
            "data/generated",
            "outputs/experiments",
            "data",
        ],
        help="Directories to search for existing data files",
    )
    parser.add_argument(
        "--include-preprocessed",
        action="store_true",
        help="Also migrate processed data (train/val/test splits) from data/processed/",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DATA MIGRATION TO COMMON REPOSITORY")
    print("=" * 70)
    print(f"Search directories: {args.search_dirs}")
    print(f"Dry run: {args.dry_run}")
    print(f"Overwrite: {args.overwrite}")
    print()

    # Ensure common directories exist
    _ensure_common_directories()

    # Convert search dirs to Path objects
    search_dirs = [PROJECT_ROOT / d for d in args.search_dirs]

    # Add data/processed if including processed data
    if args.include_preprocessed:
        prep_dir = PROJECT_ROOT / "data" / "preprocessed"
        if prep_dir.exists() and prep_dir not in search_dirs:
            search_dirs.append(prep_dir)

    # Find existing data files
    print("Searching for existing data files...")
    if args.include_preprocessed:
        print("  Including preprocessed data (train/val/test splits)")
    data_files = find_existing_data_files(
        search_dirs, include_preprocessed=args.include_preprocessed
    )
    print(f"Found {len(data_files)} data file(s)")
    print()

    if not data_files:
        print("No data files found to migrate.")
        return

    # Migrate each file
    new_files_count = 0
    duplicate_count = 0
    error_count = 0

    for data_path in data_files:
        success, new_path, error = migrate_data_file(data_path, args.dry_run, args.overwrite)
        if success:
            if new_path and new_path.exists() and new_path != data_path:
                # New file was created
                new_files_count += 1
            else:
                # File was skipped (duplicate fingerprint)
                duplicate_count += 1
        else:
            error_count += 1

    # Count unique files in common repository
    unique_count = 0
    if not args.dry_run and COMMON_DATA_DIR.exists():
        csv_files = list(COMMON_DATA_DIR.glob("*.csv"))
        unique_count = len(csv_files)

    print()
    print("=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {len(data_files)}")
    print(f"New unique files migrated: {new_files_count}")
    print(f"Skipped (duplicate fingerprints): {duplicate_count}")
    print(f"Errors: {error_count}")
    if unique_count > 0:
        print(f"Total unique files in data/common/: {unique_count}")
    print()
    print("Note: Files with identical generation parameters have the same fingerprint")
    print("      and are automatically deduplicated (this prevents duplicate data!)")

    if args.dry_run:
        print()
        print("⚠️  This was a dry run. Use without --dry-run to actually migrate files.")


if __name__ == "__main__":
    from datetime import datetime

    main()
