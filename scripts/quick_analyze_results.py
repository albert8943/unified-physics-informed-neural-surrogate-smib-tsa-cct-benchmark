#!/usr/bin/env python
"""Quick analysis of experiment results."""
import pandas as pd
import sys
from pathlib import Path

data_path = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("data/generated/smib_cct_load_variation_test/trajectory_data_20260105_113248.csv")
)
df = pd.read_csv(data_path)
scenarios = df.groupby("scenario_id").first()

print("=== KEY FINDINGS ===\n")
print(f"Total scenarios: {len(scenarios)}")
print(f"Expected: 5 load levels × 5 trajectories = 25 scenarios")
print(f"Actual: {len(scenarios)} scenarios")
print()

if "param_cct_absolute" in scenarios.columns and "clearing_time" in scenarios.columns:
    scenarios["offset"] = scenarios["clearing_time"] - scenarios["param_cct_absolute"]
    print("Clearing time offsets from CCT:")
    print(f"  Min: {scenarios['offset'].min():.6f}s")
    print(f"  Max: {scenarios['offset'].max():.6f}s")
    print(f"  Mean: {scenarios['offset'].mean():.6f}s")
    print(f"  Negative (stable): {(scenarios['offset'] < 0).sum()}")
    print(f"  Zero/Positive (unstable): {(scenarios['offset'] >= 0).sum()}")
    print()
    print("Sample scenarios:")
    print(
        scenarios[["scenario_id", "clearing_time", "param_cct_absolute", "offset", "is_stable"]]
        .head(10)
        .to_string()
    )
