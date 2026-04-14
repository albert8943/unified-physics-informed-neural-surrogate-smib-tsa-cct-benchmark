#!/usr/bin/env python
"""
Fix sample counts in migrated filenames to use unique scenario count instead of total rows.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import only what we need to avoid dependency issues
import json

COMMON_DATA_DIR = PROJECT_ROOT / "data" / "common"


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    """Save JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _load_registry(registry_path: Path) -> dict:
    """Load registry JSON file."""
    if not registry_path.exists():
        return {}
    try:
        return load_json(registry_path)
    except Exception:
        return {}


def _save_registry(registry: dict, registry_path: Path):
    """Save registry JSON file."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(registry, registry_path)


def generate_timestamp() -> str:
    """Generate timestamp string."""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _extract_key_params_for_filename(config: dict, task: str) -> str:
    """Extract key parameters for filename (simplified version)."""
    gen_config = config.get("data", {}).get("generation", {})
    param_ranges = gen_config.get("parameter_ranges", {})
    params = []

    def format_param(name: str, min_val: float, max_val: float) -> str:
        min_str = f"{min_val:.2f}".rstrip("0").rstrip(".")
        max_str = f"{max_val:.2f}".rstrip("0").rstrip(".")
        if not min_str or min_str == ".":
            min_str = "0"
        if not max_str or max_str == ".":
            max_str = "0"
        min_str = min_str.lstrip(".").rstrip(".")
        max_str = max_str.lstrip(".").rstrip(".")
        return f"{name}{min_str}-{max_str}"

    if task == "trajectory":
        H_val = param_ranges.get("H", (2.0, 10.0, 5))
        if isinstance(H_val, (list, tuple)) and len(H_val) >= 2:
            params.append(format_param("H", float(H_val[0]), float(H_val[1])))
        D_val = param_ranges.get("D", (0.5, 3.0, 5))
        if isinstance(D_val, (list, tuple)) and len(D_val) >= 2:
            params.append(format_param("D", float(D_val[0]), float(D_val[1])))
        # Check for load range (preferred) or Pm range (backward compatibility)
        load_val = param_ranges.get("load", None)
        if load_val is not None and isinstance(load_val, (list, tuple)) and len(load_val) >= 2:
            params.append(format_param("Pload", float(load_val[0]), float(load_val[1])))
        else:
            # Fallback to Pm for backward compatibility
            Pm_val = param_ranges.get("Pm", None)
            if Pm_val is not None and isinstance(Pm_val, (list, tuple)) and len(Pm_val) >= 2:
                params.append(format_param("Pm", float(Pm_val[0]), float(Pm_val[1])))

    return "_".join(params) if params else "default"


def generate_data_filename(
    task: str, config: dict, fingerprint: str, n_samples: int, timestamp: str
) -> str:
    """Generate filename with correct sample count."""
    key_params = _extract_key_params_for_filename(config, task)
    fp_short = fingerprint[:8]
    parts = [task, "data", str(n_samples)]
    if key_params:
        parts.append(key_params)
    parts.extend([fp_short, timestamp])
    return "_".join(parts) + ".csv"


def fix_file_sample_count(data_path: Path, dry_run: bool = False) -> Tuple[bool, Optional[Path]]:
    """
    Fix sample count in filename to use unique scenarios.

    Parameters:
    -----------
    data_path : Path
        Path to data file
    dry_run : bool
        If True, don't actually rename (default: False)

    Returns:
    --------
    success : bool
        True if fix succeeded
    new_path : Path or None
        New path if renamed
    """
    try:
        # Load data to get correct count
        df = pd.read_csv(data_path)

        # Get correct sample count
        # n_samples should be the number of unique parameter combinations (H, D, Pm)
        # NOT the number of scenarios/trajectories
        # Handle both formats: full trajectory data (param_H, param_D, param_Pm) and summary stats (H, D, Pm)
        if all(c in df.columns for c in ["param_H", "param_D", "param_Pm"]):
            # Full trajectory data format: count unique parameter combinations
            param_combos = df[["param_H", "param_D", "param_Pm"]].drop_duplicates()
            correct_n_samples = len(param_combos)
        elif all(c in df.columns for c in ["H", "D", "Pm"]):
            # Summary statistics format: count unique parameter combinations
            param_combos = df[["H", "D", "Pm"]].drop_duplicates()
            correct_n_samples = len(param_combos)
        elif "scenario_id" in df.columns:
            # Fallback: try to infer from scenarios (less accurate)
            n_scenarios = df["scenario_id"].nunique()
            # Assume typical n_samples_per_combination = 5
            correct_n_samples = round(n_scenarios / 5)
            if correct_n_samples == 0:
                correct_n_samples = 1
        else:
            correct_n_samples = len(df)

        # Parse current filename
        stem = data_path.stem
        parts = stem.split("_")

        if len(parts) < 6:
            print(f"[WARNING] Could not parse filename: {data_path.name}")
            return False, None

        # Find fingerprint (8 hex chars)
        fingerprint_idx = None
        for i, part in enumerate(parts):
            if len(part) == 8 and all(c in "0123456789abcdef" for c in part.lower()):
                fingerprint_idx = i
                break

        if fingerprint_idx is None:
            print(f"[WARNING] Could not find fingerprint in: {data_path.name}")
            return False, None

        task = parts[0]
        current_n_samples = parts[2]

        # Check if count needs fixing
        if str(correct_n_samples) == current_n_samples:
            # Already correct
            return True, data_path

        # Load metadata to get config
        metadata_path = data_path.with_suffix(".json").with_name(data_path.stem + "_metadata.json")
        if not metadata_path.exists():
            print(f"[WARNING] Metadata not found for: {data_path.name}")
            return False, None

        metadata = load_json(metadata_path)
        config = {
            "data": {
                "task": metadata.get("task", task),
                "generation": metadata.get("generation_config", {}),
            },
            "reproducibility": metadata.get("reproducibility", {}),
        }
        fingerprint = metadata.get("data_fingerprint")
        if not fingerprint:
            print(f"[WARNING] Fingerprint not found in metadata: {data_path.name}")
            return False, None

        # Generate new filename with correct count
        # Extract full timestamp - it should be in format YYYYMMDD_HHMMSS
        # Check if last part looks like a timestamp (8 digits_date_6 digits_time or 14 digits)
        if len(parts) > fingerprint_idx + 1:
            last_part = parts[-1]
            # Check if it's a full timestamp (14 digits: YYYYMMDDHHMMSS) or partial
            if len(last_part) == 14 and last_part.isdigit():
                # Full timestamp without underscore: YYYYMMDDHHMMSS
                timestamp = f"{last_part[:8]}_{last_part[8:]}"
            elif "_" in last_part and len(last_part.split("_")) == 2:
                # Already has underscore: YYYYMMDD_HHMMSS
                timestamp = last_part
            elif len(parts) > fingerprint_idx + 2:
                # Date and time might be separate parts
                if (
                    parts[-2].isdigit()
                    and len(parts[-2]) == 8
                    and parts[-1].isdigit()
                    and len(parts[-1]) == 6
                ):
                    timestamp = f"{parts[-2]}_{parts[-1]}"
                else:
                    timestamp = last_part
            else:
                timestamp = last_part
        else:
            timestamp = generate_timestamp()
        new_filename = generate_data_filename(
            task, config, fingerprint, correct_n_samples, timestamp
        )

        if new_filename == data_path.name:
            # No change needed
            return True, data_path

        new_path = data_path.parent / new_filename

        if dry_run:
            print(f"[DRY RUN] Would rename: {data_path.name}")
            print(f"   Current count: {current_n_samples} -> Correct count: {correct_n_samples}")
            print(f"   -> {new_filename}")
            return True, new_path

        # Rename file
        data_path.rename(new_path)
        print(f"[OK] Renamed: {data_path.name} -> {new_filename}")
        print(f"   Fixed count: {current_n_samples} -> {correct_n_samples}")

        # Update metadata file
        new_metadata_path = new_path.with_suffix(".json").with_name(
            new_path.stem + "_metadata.json"
        )
        metadata_path.rename(new_metadata_path)
        metadata["filename"] = new_filename
        metadata["n_samples"] = correct_n_samples
        save_json(metadata, new_metadata_path)

        # Update registry
        registry = _load_registry(COMMON_DATA_DIR / "registry.json")
        if "fingerprints" in registry and fingerprint in registry["fingerprints"]:
            registry["fingerprints"][fingerprint]["filename"] = new_filename
            registry["fingerprints"][fingerprint]["n_samples"] = correct_n_samples
            _save_registry(registry, COMMON_DATA_DIR / "registry.json")

        return True, new_path

    except Exception as e:
        print(f"[ERROR] Error fixing {data_path.name}: {str(e)}")
        return False, None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix sample counts in migrated filenames")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be fixed without renaming"
    )
    parser.add_argument(
        "--pattern", type=str, default="*.csv", help="File pattern to match (default: *.csv)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FIX SAMPLE COUNTS IN FILENAMES")
    print("=" * 70)
    print(f"Directory: {COMMON_DATA_DIR}")
    print(f"Pattern: {args.pattern}")
    print(f"Dry run: {args.dry_run}")
    print()

    if not COMMON_DATA_DIR.exists():
        print(f"[ERROR] Common data directory not found: {COMMON_DATA_DIR}")
        return

    # Find all CSV files
    data_files = list(COMMON_DATA_DIR.glob(args.pattern))
    data_files = [f for f in data_files if f.is_file() and f.suffix == ".csv"]

    print(f"Found {len(data_files)} data file(s)")
    print()

    if not data_files:
        print("No data files found.")
        return

    # Fix each file
    fixed_count = 0
    skip_count = 0
    error_count = 0

    for data_path in sorted(data_files):
        success, new_path = fix_file_sample_count(data_path, args.dry_run)
        if success:
            if new_path and new_path != data_path:
                fixed_count += 1
            else:
                skip_count += 1
        else:
            error_count += 1

    print()
    print("=" * 70)
    print("FIX SUMMARY")
    print("=" * 70)
    print(f"Successfully fixed: {fixed_count}")
    print(f"Skipped (already correct): {skip_count}")
    print(f"Errors: {error_count}")
    print(f"Total files processed: {len(data_files)}")

    if args.dry_run:
        print()
        print("[NOTE] This was a dry run. Use without --dry-run to actually fix filenames.")


if __name__ == "__main__":
    main()
