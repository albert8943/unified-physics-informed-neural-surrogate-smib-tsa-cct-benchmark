#!/usr/bin/env python
"""Fix timestamps in filenames to use full format from metadata."""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
COMMON_DATA_DIR = PROJECT_ROOT / "data" / "common"


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    """Save JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def fix_timestamp(data_path: Path, dry_run: bool = False) -> bool:
    """Fix timestamp in filename using metadata."""
    try:
        # Load metadata
        metadata_path = data_path.with_suffix(".json").with_name(data_path.stem + "_metadata.json")
        if not metadata_path.exists():
            return False

        metadata = load_json(metadata_path)
        original_filename = metadata.get("filename", "")
        migration_timestamp = metadata.get("migration_timestamp", "")

        # Extract full timestamp from migration_timestamp or original filename
        if migration_timestamp:
            from datetime import datetime

            dt = datetime.fromisoformat(migration_timestamp.replace("Z", "+00:00"))
            full_timestamp = dt.strftime("%Y%m%d_%H%M%S")
        elif original_filename:
            # Try to extract from original filename
            parts = original_filename.replace(".csv", "").split("_")
            # Look for timestamp pattern YYYYMMDD_HHMMSS
            for i in range(len(parts) - 1):
                if (
                    len(parts[i]) == 8
                    and parts[i].isdigit()
                    and len(parts[i + 1]) == 6
                    and parts[i + 1].isdigit()
                ):
                    full_timestamp = f"{parts[i]}_{parts[i+1]}"
                    break
            else:
                return False  # Couldn't find timestamp
        else:
            return False

        # Parse current filename
        stem = data_path.stem
        parts = stem.split("_")

        # Find fingerprint
        fingerprint_idx = None
        for i, part in enumerate(parts):
            if len(part) == 8 and all(c in "0123456789abcdef" for c in part.lower()):
                fingerprint_idx = i
                break

        if fingerprint_idx is None:
            return False

        # Check if timestamp needs fixing
        current_timestamp = (
            "_".join(parts[fingerprint_idx + 1 :]) if len(parts) > fingerprint_idx + 1 else ""
        )
        if current_timestamp == full_timestamp or len(current_timestamp) >= 13:
            # Already correct or has full timestamp
            return False

        # Reconstruct filename with full timestamp
        new_parts = parts[: fingerprint_idx + 1] + [full_timestamp]
        new_filename = "_".join(new_parts) + ".csv"
        new_path = data_path.parent / new_filename

        if new_filename == data_path.name:
            return False

        if dry_run:
            print(f"[DRY RUN] Would fix timestamp: {data_path.name}")
            print(f"   Current: {current_timestamp} -> Full: {full_timestamp}")
            print(f"   -> {new_filename}")
            return True

        # Rename file
        data_path.rename(new_path)
        print(f"[OK] Fixed timestamp: {data_path.name} -> {new_filename}")

        # Update metadata
        new_metadata_path = new_path.with_suffix(".json").with_name(
            new_path.stem + "_metadata.json"
        )
        metadata_path.rename(new_metadata_path)
        metadata["filename"] = new_filename
        save_json(metadata, new_metadata_path)

        # Update registry
        registry_path = COMMON_DATA_DIR / "registry.json"
        if registry_path.exists():
            registry = load_json(registry_path)
            fingerprint = metadata.get("data_fingerprint")
            if (
                fingerprint
                and "fingerprints" in registry
                and fingerprint in registry["fingerprints"]
            ):
                registry["fingerprints"][fingerprint]["filename"] = new_filename
                save_json(registry, registry_path)

        return True

    except Exception as e:
        print(f"[ERROR] Error fixing {data_path.name}: {str(e)}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix timestamps in filenames")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    args = parser.parse_args()

    print("=" * 70)
    print("FIX TIMESTAMPS IN FILENAMES")
    print("=" * 70)
    print()

    if not COMMON_DATA_DIR.exists():
        print(f"[ERROR] Common data directory not found: {COMMON_DATA_DIR}")
        return

    csv_files = list(COMMON_DATA_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    fixed = 0
    skipped = 0

    for f in sorted(csv_files):
        if fix_timestamp(f, args.dry_run):
            fixed += 1
        else:
            skipped += 1

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Fixed: {fixed}")
    print(f"Skipped (already correct): {skipped}")


if __name__ == "__main__":
    main()
