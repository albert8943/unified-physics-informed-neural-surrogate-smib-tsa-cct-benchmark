#!/usr/bin/env python
"""
Fix Migrated Filenames: Clean up parameter formatting in migrated files.

This script renames files in data/common/ to have cleaner parameter formatting
(2 decimal places instead of long decimals, fix double dots).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.common_repository import (
    COMMON_DATA_DIR,
    _load_registry,
    _save_registry,
    generate_timestamp,
    load_json,
    save_json,
)


def parse_filename(filename: str) -> Optional[Dict]:
    """
    Parse a migrated filename to extract components.

    Format: {task}_data_{n_samples}_{key_params}_{fingerprint}_{timestamp}.csv

    Returns:
    --------
    components : dict or None
        Parsed components: task, n_samples, key_params, fingerprint, timestamp
    """
    if not filename.endswith(".csv"):
        return None

    stem = filename[:-4]  # Remove .csv
    parts = stem.split("_")

    if len(parts) < 6:
        return None

    # Find where fingerprint starts (8 hex chars)
    fingerprint_idx = None
    for i, part in enumerate(parts):
        if len(part) == 8 and all(c in "0123456789abcdef" for c in part.lower()):
            fingerprint_idx = i
            break

    if fingerprint_idx is None:
        return None

    task = parts[0]
    if parts[1] != "data":
        return None

    n_samples = parts[2]

    # Key params are between "data" and fingerprint
    key_params_parts = parts[3:fingerprint_idx]
    key_params = "_".join(key_params_parts)

    fingerprint = parts[fingerprint_idx]
    timestamp = "_".join(parts[fingerprint_idx + 1 :])

    return {
        "task": task,
        "n_samples": n_samples,
        "key_params": key_params,
        "fingerprint": fingerprint,
        "timestamp": timestamp,
    }


def fix_filename_parameters(key_params_str: str) -> str:
    """
    Fix parameter formatting in filename (round to 2 decimals, fix double dots).

    Parameters:
    -----------
    key_params_str : str
        Original key params string (e.g., "H1.5725421644747255-4.448227632790804")

    Returns:
    --------
    fixed_params : str
        Fixed key params string (e.g., "H1.57-4.45")
    """
    if not key_params_str:
        return "default"

    # Split by parameter (H, D, Pm)
    params = []
    current_param = None
    current_value = ""

    i = 0
    while i < len(key_params_str):
        char = key_params_str[i]

        # Check if this starts a new parameter
        if char in ["H", "D", "P"] and i + 1 < len(key_params_str) and key_params_str[i + 1] == "m":
            # Save previous param if exists
            if current_param:
                params.append((current_param, current_value))
            current_param = "Pm"
            current_value = ""
            i += 2
        elif char in ["H", "D"] and (i == 0 or key_params_str[i - 1] == "_"):
            # Save previous param if exists
            if current_param:
                params.append((current_param, current_value))
            current_param = char
            current_value = ""
            i += 1
        else:
            current_value += char
            i += 1

    # Add last param
    if current_param:
        params.append((current_param, current_value))

    # Fix each parameter
    fixed_params = []
    for param_name, param_value in params:
        # Parse range (min-max)
        if "-" in param_value:
            try:
                min_str, max_str = param_value.split("-", 1)
                # Remove any double dots
                min_str = min_str.replace("..", ".")
                max_str = max_str.replace("..", ".")

                min_val = float(min_str)
                max_val = float(max_str)

                # Round to 2 decimal places
                min_val = round(min_val, 2)
                max_val = round(max_val, 2)

                # Format cleanly
                min_fmt = f"{min_val:.2f}".rstrip("0").rstrip(".")
                max_fmt = f"{max_val:.2f}".rstrip("0").rstrip(".")

                if not min_fmt:
                    min_fmt = "0"
                if not max_fmt:
                    max_fmt = "0"

                fixed_params.append(f"{param_name}{min_fmt}-{max_fmt}")
            except (ValueError, AttributeError):
                # If parsing fails, keep original
                fixed_params.append(f"{param_name}{param_value}")
        else:
            fixed_params.append(f"{param_name}{param_value}")

    return "_".join(fixed_params) if fixed_params else "default"


def fix_migrated_file(data_path: Path, dry_run: bool = False) -> Tuple[bool, Optional[Path]]:
    """
    Fix a single migrated file's filename.

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
        # Try to use metadata first (more reliable)
        metadata_path = data_path.with_suffix(".json").with_name(data_path.stem + "_metadata.json")
        if metadata_path.exists():
            try:
                metadata = load_json(metadata_path)
                config = {
                    "data": {
                        "task": metadata.get("task", "trajectory"),
                        "generation": metadata.get("generation_config", {}),
                    },
                    "reproducibility": metadata.get("reproducibility", {}),
                }
                fingerprint = metadata.get("data_fingerprint")
                n_samples = metadata.get("n_samples")
                task = metadata.get("task", "trajectory")

                if fingerprint and n_samples:
                    # Regenerate filename with fixed formatting
                    from scripts.core.common_repository import generate_data_filename

                    # Extract timestamp from current filename
                    parts = data_path.stem.split("_")
                    timestamp = "_".join(parts[-2:]) if len(parts) >= 2 else generate_timestamp()

                    new_filename = generate_data_filename(
                        task, config, fingerprint, n_samples, timestamp
                    )

                    if new_filename == data_path.name:
                        # No change needed
                        return True, data_path

                    new_path = data_path.parent / new_filename

                    if dry_run:
                        print(f"📋 [DRY RUN] Would rename: {data_path.name}")
                        print(f"   → {new_filename}")
                        return True, new_path

                    # Rename file
                    data_path.rename(new_path)
                    print(f"✓ Renamed: {data_path.name} → {new_filename}")

                    # Update metadata file
                    new_metadata_path = new_path.with_suffix(".json").with_name(
                        new_path.stem + "_metadata.json"
                    )
                    metadata_path.rename(new_metadata_path)
                    metadata["filename"] = new_filename
                    save_json(metadata, new_metadata_path)

                    # Update registry
                    registry = _load_registry(COMMON_DATA_DIR / "registry.json")
                    if "fingerprints" in registry and fingerprint in registry["fingerprints"]:
                        registry["fingerprints"][fingerprint]["filename"] = new_filename
                        _save_registry(registry, COMMON_DATA_DIR / "registry.json")

                    return True, new_path
            except Exception as e:
                print(f"⚠️  Could not use metadata for {data_path.name}: {e}")
                # Fall through to filename parsing method

        # Fallback: Parse filename
        components = parse_filename(data_path.name)
        if not components:
            print(f"⚠️  Could not parse filename: {data_path.name}")
            return False, None

        # Fix key params
        fixed_key_params = fix_filename_parameters(components["key_params"])

        # Generate new filename
        new_filename = (
            f"{components['task']}_data_{components['n_samples']}_"
            f"{fixed_key_params}_{components['fingerprint']}_{components['timestamp']}.csv"
        )

        if new_filename == data_path.name:
            # No change needed
            return True, data_path

        new_path = data_path.parent / new_filename

        if dry_run:
            print(f"📋 [DRY RUN] Would rename: {data_path.name}")
            print(f"   → {new_filename}")
            return True, new_path

        # Rename file
        data_path.rename(new_path)
        print(f"✓ Renamed: {data_path.name} → {new_filename}")

        # Update metadata file if exists
        old_metadata_path = data_path.with_suffix(".json").with_name(
            data_path.stem + "_metadata.json"
        )
        if old_metadata_path.exists():
            new_metadata_path = new_path.with_suffix(".json").with_name(
                new_path.stem + "_metadata.json"
            )
            old_metadata_path.rename(new_metadata_path)

            # Update filename in metadata
            metadata = load_json(new_metadata_path)
            metadata["filename"] = new_filename
            save_json(metadata, new_metadata_path)

        # Update registry
        registry = _load_registry(COMMON_DATA_DIR / "registry.json")
        fingerprint = components["fingerprint"]
        if "fingerprints" in registry and fingerprint in registry["fingerprints"]:
            registry["fingerprints"][fingerprint]["filename"] = new_filename
            _save_registry(registry, COMMON_DATA_DIR / "registry.json")

        return True, new_path

    except Exception as e:
        print(f"❌ Error fixing {data_path.name}: {str(e)}")
        return False, None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix parameter formatting in migrated filenames")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without actually renaming",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="File pattern to match (default: *.csv)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FIX MIGRATED FILENAMES")
    print("=" * 70)
    print(f"Directory: {COMMON_DATA_DIR}")
    print(f"Pattern: {args.pattern}")
    print(f"Dry run: {args.dry_run}")
    print()

    if not COMMON_DATA_DIR.exists():
        print(f"❌ Common data directory not found: {COMMON_DATA_DIR}")
        return

    # Find all CSV files
    data_files = list(COMMON_DATA_DIR.glob(args.pattern))
    data_files = [f for f in data_files if f.is_file() and f.suffix == ".csv"]

    print(f"Found {len(data_files)} data file(s)")
    print()

    if not data_files:
        print("No data files found to fix.")
        return

    # Fix each file
    success_count = 0
    skip_count = 0
    error_count = 0

    for data_path in sorted(data_files):
        success, new_path = fix_migrated_file(data_path, args.dry_run)
        if success:
            if new_path and new_path != data_path:
                success_count += 1
            else:
                skip_count += 1
        else:
            error_count += 1

    print()
    print("=" * 70)
    print("FIX SUMMARY")
    print("=" * 70)
    print(f"Successfully renamed: {success_count}")
    print(f"Skipped (no change needed): {skip_count}")
    print(f"Errors: {error_count}")
    print(f"Total files processed: {len(data_files)}")

    if args.dry_run:
        print()
        print("⚠️  This was a dry run. Use without --dry-run to actually rename files.")


if __name__ == "__main__":
    main()
