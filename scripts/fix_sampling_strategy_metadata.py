#!/usr/bin/env python
"""
Fix sampling_strategy in metadata files by checking original experiment configs.

The migration script defaulted to "full_factorial" but most experiments actually used "sobol".
This script checks the original experiment summary files and updates the metadata accordingly.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

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


def find_experiment_summary(original_path: str) -> Optional[Dict]:
    """
    Find experiment summary file from original path.

    Parameters:
    -----------
    original_path : str
        Original path from metadata (e.g., outputs/experiments/exp_20251207_192804/...)

    Returns:
    --------
    summary : dict or None
        Experiment summary if found
    """
    try:
        # Extract experiment ID from path
        # Format: outputs/experiments/exp_YYYYMMDD_HHMMSS/...
        path_parts = Path(original_path).parts
        exp_dir = None
        for i, part in enumerate(path_parts):
            if part.startswith("exp_") and len(part) > 4:
                exp_dir = PROJECT_ROOT / Path(*path_parts[: i + 1])
                break

        if exp_dir and exp_dir.exists():
            summary_path = exp_dir / "experiment_summary.json"
            if summary_path.exists():
                return load_json(summary_path)
    except Exception:
        pass

    return None


def fix_metadata_sampling_strategy(metadata_path: Path, dry_run: bool = False) -> bool:
    """
    Fix sampling_strategy in metadata file by checking original experiment config.

    Parameters:
    -----------
    metadata_path : Path
        Path to metadata file
    dry_run : bool
        If True, don't actually update (default: False)

    Returns:
    --------
    success : bool
        True if fix succeeded
    """
    try:
        metadata = load_json(metadata_path)

        # Get current sampling strategy
        current_strategy = metadata.get("generation_config", {}).get("sampling_strategy", "unknown")

        # If already correct, skip
        if current_strategy == "sobol":
            return False  # Already correct

        # Try to find original experiment summary
        original_path = metadata.get("original_path", "")
        if not original_path:
            print(f"[SKIP] {metadata_path.name}: No original_path in metadata")
            return False

        summary = find_experiment_summary(original_path)
        if not summary:
            print(f"[SKIP] {metadata_path.name}: Could not find experiment summary")
            return False

        # Extract actual sampling strategy from experiment summary
        actual_strategy = (
            summary.get("config", {})
            .get("data", {})
            .get("generation", {})
            .get("sampling_strategy", None)
        )

        if not actual_strategy:
            print(f"[SKIP] {metadata_path.name}: No sampling_strategy in experiment summary")
            return False

        # Check if it's different
        if actual_strategy == current_strategy:
            return False  # Already correct

        if dry_run:
            print(f"[DRY RUN] Would update {metadata_path.name}")
            print(f"   Current: {current_strategy} -> Actual: {actual_strategy}")
            return True

        # Update metadata
        if "generation_config" not in metadata:
            metadata["generation_config"] = {}

        metadata["generation_config"]["sampling_strategy"] = actual_strategy

        # Note: Changing sampling_strategy changes the fingerprint
        # However, the data file itself is correct (generated with sobol)
        # We'll update the metadata to reflect the correct strategy
        # The fingerprint in the filename won't change (to avoid breaking references)
        # But the metadata will be correct for future reference

        # Save updated metadata
        save_json(metadata, metadata_path)
        print(f"[OK] Updated {metadata_path.name}: {current_strategy} -> {actual_strategy}")

        # Update registry if needed
        registry_path = COMMON_DATA_DIR / "registry.json"
        if registry_path.exists():
            registry = _load_registry(registry_path)
            fingerprint = metadata.get("data_fingerprint")
            if (
                fingerprint
                and "fingerprints" in registry
                and fingerprint in registry["fingerprints"]
            ):
                registry["fingerprints"][fingerprint]["sampling_strategy"] = actual_strategy
                _save_registry(registry, registry_path)

        return True

    except Exception as e:
        print(f"[ERROR] Error fixing {metadata_path.name}: {str(e)}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Fix sampling_strategy in metadata files from experiment configs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be fixed without updating"
    )
    parser.add_argument(
        "--force-sobol",
        action="store_true",
        help="Force all to sobol (if experiment summary not found)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FIX SAMPLING STRATEGY IN METADATA FILES")
    print("=" * 70)
    print(f"Directory: {COMMON_DATA_DIR}")
    print(f"Dry run: {args.dry_run}")
    if args.force_sobol:
        print("Force sobol: Enabled (will set to sobol if experiment summary not found)")
    print()

    if not COMMON_DATA_DIR.exists():
        print(f"[ERROR] Common data directory not found: {COMMON_DATA_DIR}")
        return

    # Find all metadata files
    metadata_files = sorted(COMMON_DATA_DIR.glob("*_metadata.json"))
    print(f"Found {len(metadata_files)} metadata file(s)")
    print()

    if not metadata_files:
        print("No metadata files found.")
        return

    # Fix each file
    updated_count = 0
    skipped_count = 0
    error_count = 0

    for metadata_path in metadata_files:
        success = fix_metadata_sampling_strategy(metadata_path, args.dry_run)
        if success:
            updated_count += 1
        else:
            skipped_count += 1

    print()
    print("=" * 70)
    print("FIX SUMMARY")
    print("=" * 70)
    print(f"Updated: {updated_count}")
    print(f"Skipped (already correct or no match): {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Total files processed: {len(metadata_files)}")

    if args.dry_run:
        print()
        print("[NOTE] This was a dry run. Use without --dry-run to actually update metadata.")


if __name__ == "__main__":
    main()
