#!/usr/bin/env python
"""Check file counts to understand the discrepancy."""

import pandas as pd
from pathlib import Path

file_path = Path(
    "data/common/trajectory_data_99_H1.63-4.39_D0.57-2.45_Pm0.42-0.9_ad9660ec_20251212_115822.csv"
)

df = pd.read_csv(file_path)
print(f"File: {file_path.name}")
print(f"Total rows (including header): {len(df) + 1}")
print(f"Data rows: {len(df)}")
print(
    f"Unique scenario_id: {df['scenario_id'].nunique() if 'scenario_id' in df.columns else 'N/A'}"
)
print(f"scenario_id range: {df['scenario_id'].min()} to {df['scenario_id'].max()}")
print(f"scenario_id values: {sorted(df['scenario_id'].unique())[:10]}... (showing first 10)")
