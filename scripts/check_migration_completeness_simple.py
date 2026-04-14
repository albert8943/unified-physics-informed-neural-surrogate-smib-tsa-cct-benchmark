#!/usr/bin/env python
"""Simple script to check migration completeness."""

import pandas as pd
from pathlib import Path

COMMON_DATA_DIR = Path("data/common")


def check_file(file_path):
    """Check if file is full or summary."""
    try:
        df = pd.read_csv(file_path, nrows=10)
        has_time = "time" in df.columns
        has_param = "param_H" in df.columns
        has_summary = "H" in df.columns and "CCT" in df.columns and not has_time

        if has_time and has_param:
            return "full"
        elif has_summary:
            return "summary"
        else:
            return "unknown"
    except Exception as e:
        return f"error: {e}"


if __name__ == "__main__":
    files = sorted(COMMON_DATA_DIR.glob("trajectory_data_*.csv"))
    print(f"Found {len(files)} files\n")

    summary = []
    full = []
    unknown = []

    for f in files:
        result = check_file(f)
        if result == "summary":
            summary.append(f)
        elif result == "full":
            full.append(f)
        else:
            unknown.append((f, result))

    print(f"[OK] Full trajectory data: {len(full)}")
    print(f"[WARN] Summary statistics (incomplete): {len(summary)}")
    print(f"[?] Unknown/Error: {len(unknown)}\n")

    if summary:
        print("INCOMPLETE FILES (Summary Statistics):")
        for f in summary:
            df = pd.read_csv(f)
            print(f"  - {f.name}: {len(df)} rows")
        print()

    if full:
        print("FULL DATA FILES (sample):")
        for f in full[:5]:
            df = pd.read_csv(f)
            scenarios = df["scenario_id"].nunique() if "scenario_id" in df.columns else "N/A"
            print(f"  - {f.name}: {len(df):,} rows, {scenarios} scenarios")
        if len(full) > 5:
            print(f"  ... and {len(full) - 5} more")
