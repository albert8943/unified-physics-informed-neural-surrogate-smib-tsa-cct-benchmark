#!/usr/bin/env python
"""Verify that n_samples in filenames matches actual parameter combinations."""

import pandas as pd
from pathlib import Path

COMMON_DATA_DIR = Path("data/common")

files = sorted(COMMON_DATA_DIR.glob("trajectory_data_*.csv"))

print("=" * 70)
print("VERIFYING N_SAMPLES IN FILENAMES")
print("=" * 70)
print()

all_correct = True

for f in files:
    # Parse filename to get n_samples
    parts = f.stem.split("_")
    if len(parts) < 3:
        continue
    filename_n_samples = int(parts[2])  # e.g., "20" from "trajectory_data_20_..."

    # Load data and count parameter combinations
    df = pd.read_csv(f)
    if all(c in df.columns for c in ["H", "D", "Pm"]):
        param_combos = df[["H", "D", "Pm"]].drop_duplicates()
        actual_n_samples = len(param_combos)
    elif all(c in df.columns for c in ["param_H", "param_D", "param_Pm"]):
        param_combos = df[["param_H", "param_D", "param_Pm"]].drop_duplicates()
        actual_n_samples = len(param_combos)
    else:
        print(f"⚠️  {f.name}: Cannot determine parameter combinations (missing H/D/Pm columns)")
        continue

    # Check if correct
    is_correct = filename_n_samples == actual_n_samples
    status = "[OK]" if is_correct else "[ERROR]"

    print(f"{status} {f.name}")
    print(f"   Filename shows: {filename_n_samples} parameter combinations")
    print(f"   Actual: {actual_n_samples} parameter combinations")
    if not is_correct:
        print(f"   ⚠️  MISMATCH!")
        all_correct = False
    print()

print("=" * 70)
if all_correct:
    print("[OK] All filenames are correct!")
else:
    print("[ERROR] Some filenames need fixing!")
print("=" * 70)
