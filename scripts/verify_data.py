#!/usr/bin/env python
"""Quick script to verify generated data with detailed analysis"""
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.data_utils import find_data_file

if len(sys.argv) > 1:
    csv_file = find_data_file(data_path=sys.argv[1])
else:
    # Find latest file (searches quick/moderate/comprehensive)
    try:
        csv_file = find_data_file()  # Defaults to quick_test
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n💡 Tips:")
        print("   - Specify file: python scripts/verify_data.py <path_to_file.csv>")
        print(
            "   - Or specify directory: python scripts/verify_data.py --data-dir data/generated/moderate_test"
        )
        sys.exit(1)

csv_file = str(csv_file)

csv_file_name = Path(csv_file).name
print(f"📊 Verifying: {csv_file_name}")
print("=" * 70)

df = pd.read_csv(csv_file)

print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

if "scenario_id" in df.columns:
    n_scenarios = df["scenario_id"].nunique()
    print(f"Unique scenarios (trajectories): {n_scenarios}")

    # Check parameter combinations
    if all(c in df.columns for c in ["param_H", "param_D", "param_Pm"]):
        param_combos = df[["param_H", "param_D", "param_Pm"]].drop_duplicates()
        n_combos = len(param_combos)
        print(f"Unique parameter combinations: {n_combos}")

        # Count scenarios per combination
        scenarios_per_combo = df.groupby(["param_H", "param_D", "param_Pm"])[
            "scenario_id"
        ].nunique()
        expected_per_combo = scenarios_per_combo.mode()[0] if len(scenarios_per_combo) > 0 else 5

        print("\nScenarios per combination:")
        scenario_counts = scenarios_per_combo.value_counts().sort_index()
        for count, freq in scenario_counts.items():
            print(f"  {count} scenarios: {freq} combination(s)")

        # Check for missing scenarios
        missing = scenarios_per_combo[scenarios_per_combo != expected_per_combo]
        if len(missing) > 0:
            print("\n⚠️  Combinations with unexpected scenario count:")
            for (h, d, pm), count in missing.items():
                msg = (
                    f"  H={h:.3f}, D={d:.3f}, Pm={pm:.3f}: "
                    f"{count} scenarios (expected {expected_per_combo})"
                )
                print(msg)

                # Show expected vs actual clearing times for problematic combinations
                subset = df[(df["param_H"] == h) & (df["param_D"] == d) & (df["param_Pm"] == pm)]
                cct = subset["param_cct_absolute"].dropna().unique()
                if len(cct) > 0:
                    cct_val = cct[0]
                    expected_offsets = [-0.004, -0.002, 0.0, 0.002, 0.004]
                    expected_times = [cct_val + offset for offset in expected_offsets]
                    actual_times = sorted(subset["param_tc"].unique())
                    missing_times = set(expected_times) - set(actual_times)
                    if missing_times:
                        print(f"    CCT: {cct_val:.6f}s")
                        missing_str = [f"{t:.6f}" for t in sorted(missing_times)]
                        print(f"    Missing clearing times: {missing_str}")

        # Show parameter combinations
        print("\nParameter combinations:")
        for idx, row in param_combos.iterrows():
            subset = df[
                (df["param_H"] == row["param_H"])
                & (df["param_D"] == row["param_D"])
                & (df["param_Pm"] == row["param_Pm"])
            ]
            n_scenarios_combo = subset["scenario_id"].nunique()
            status = "✅" if n_scenarios_combo == expected_per_combo else "⚠️"
            msg = (
                f"{status} H={row['param_H']:.3f}, D={row['param_D']:.3f}, "
                f"Pm={row['param_Pm']:.3f} ({n_scenarios_combo} scenarios)"
            )
            print(msg)

if "is_stable" in df.columns:
    print("\nStability distribution:")
    stability_counts = df["is_stable"].value_counts()
    for val, count in stability_counts.items():
        label = "Stable" if val else "Unstable"
        print(f"  {label}: {count:,} rows")

    # Count unique stable/unstable scenarios
    if "scenario_id" in df.columns:
        scenario_stability = df.groupby("scenario_id")["is_stable"].first()
        stable_scenarios = scenario_stability.sum()
        unstable_scenarios = len(scenario_stability) - stable_scenarios
        print("\nTrajectory stability:")
        print(f"  Stable trajectories: {stable_scenarios}")
        print(f"  Unstable trajectories: {unstable_scenarios}")

if "param_cct_absolute" in df.columns:
    cct_values = df["param_cct_absolute"].dropna().unique()
    if len(cct_values) > 0:
        print("\nCCT Analysis:")
        print(f"  Unique CCT values: {len(cct_values)}")

        # Check for duplicate CCT values (different combinations with same CCT)
        required_cols = ["param_H", "param_D", "param_Pm", "param_cct_absolute"]
        if all(c in df.columns for c in required_cols):
            cct_df = df[df["param_cct_absolute"].notna()]
            cct_combos = cct_df[required_cols].drop_duplicates()
            cct_counts = cct_combos["param_cct_absolute"].value_counts()
            duplicates = cct_counts[cct_counts > 1]
            if len(duplicates) > 0:
                print("  ⚠️  Duplicate CCT values (different combinations):")
                for cct_val, count in duplicates.items():
                    print(f"    CCT={cct_val:.6f}s: {count} combinations")
                    combos_with_cct = cct_combos[cct_combos["param_cct_absolute"] == cct_val]
                    for idx, row in combos_with_cct.iterrows():
                        msg = (
                            f"      - H={row['param_H']:.3f}, "
                            f"D={row['param_D']:.3f}, Pm={row['param_Pm']:.3f}"
                        )
                        print(msg)

        # Show all CCT values
        print("\n  All CCT values:")
        for cct in sorted(cct_values):
            print(f"    CCT: {cct:.6f}s")

        # Check combinations without CCT
        if all(c in df.columns for c in ["param_H", "param_D", "param_Pm"]):
            cct_df = df[df["param_cct_absolute"].notna()]
            combos_with_cct = cct_df[["param_H", "param_D", "param_Pm"]].drop_duplicates()
            merged = param_combos.merge(
                combos_with_cct,
                on=["param_H", "param_D", "param_Pm"],
                how="left",
                indicator=True,
            )
            combos_without_cct = merged[merged["_merge"] == "left_only"][
                ["param_H", "param_D", "param_Pm"]
            ]
            if len(combos_without_cct) > 0:
                n_no_cct = len(combos_without_cct)
                print(f"\n  ⚠️  Combinations with NO CCT (CCT finding failed): {n_no_cct}")
                for idx, row in combos_without_cct.iterrows():
                    msg = (
                        f"    H={row['param_H']:.3f}, "
                        f"D={row['param_D']:.3f}, Pm={row['param_Pm']:.3f}"
                    )
                    print(msg)

# Summary
if "scenario_id" in df.columns and all(c in df.columns for c in ["param_H", "param_D", "param_Pm"]):
    param_combos = df[["param_H", "param_D", "param_Pm"]].drop_duplicates()
    n_combos = len(param_combos)
    scenarios_per_combo = df.groupby(["param_H", "param_D", "param_Pm"])["scenario_id"].nunique()
    expected_per_combo = scenarios_per_combo.mode()[0] if len(scenarios_per_combo) > 0 else 5
    expected_scenarios = n_combos * expected_per_combo
    actual_scenarios = df["scenario_id"].nunique()

    print("\n📈 Summary:")
    msg = (
        f"  Expected scenarios: {expected_scenarios} " f"({n_combos} combos × {expected_per_combo})"
    )
    print(msg)
    print(f"  Actual scenarios: {actual_scenarios}")
    if expected_scenarios != actual_scenarios:
        missing = expected_scenarios - actual_scenarios
        print(f"  ⚠️  Missing: {missing} trajectory(ies)")
        print("\n💡 Explanation:")
        print(
            "  - Missing trajectories usually occur when simulations fail "
            "at the stability boundary"
        )
        print(
            "  - This is common for trajectories at exactly CCT (offset 0.0) "
            "due to numerical sensitivity"
        )
        print(
            "  - The remaining trajectories still provide good coverage " "around the CCT boundary"
        )
    else:
        print("  ✅ All expected trajectories generated successfully!")

print("\n✅ Data verification complete!")
print(f"📁 File: {csv_file}")
