#!/usr/bin/env python
"""Check common repository status and explain deduplication."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
COMMON_DATA_DIR = PROJECT_ROOT / "data" / "common"
REGISTRY_PATH = COMMON_DATA_DIR / "registry.json"

print("=" * 70)
print("COMMON REPOSITORY STATUS")
print("=" * 70)
print()

# Count CSV files
csv_files = list(COMMON_DATA_DIR.glob("*.csv")) if COMMON_DATA_DIR.exists() else []
print(f"CSV files in data/common/: {len(csv_files)}")

# Count metadata files
metadata_files = list(COMMON_DATA_DIR.glob("*_metadata.json")) if COMMON_DATA_DIR.exists() else []
print(f"Metadata files: {len(metadata_files)}")

# Check registry
if REGISTRY_PATH.exists():
    registry = json.load(open(REGISTRY_PATH))
    fingerprints = registry.get("fingerprints", {})
    print(f"Unique fingerprints in registry: {len(fingerprints)}")
    print()
    print("This means you have", len(fingerprints), "unique data generation configurations.")
    print()
    print("Why fewer files than processed?")
    print("  - Files with identical generation parameters have the same fingerprint")
    print("  - Duplicate fingerprints are automatically deduplicated")
    print("  - This prevents storing the same data multiple times")
    print()
    print("Example unique configurations:")
    for i, (fp, entry) in enumerate(list(fingerprints.items())[:5]):
        filename = entry.get("filename", "N/A")
        task = entry.get("task", "N/A")
        n_samples = entry.get("n_samples", "N/A")
        print(f"  {i+1}. {task} - {n_samples} samples - {filename[:60]}...")
else:
    print("Registry not found. Run migration first.")

print()
print("=" * 70)
