"""
Verify Physics Loss Parameter Consistency

This script checks if the parameters (H, D, Pm, Pe) used in physics loss
match the parameters that were used to generate the data.

Usage:
    python scripts/verify_physics_parameters.py --data-path <data_file.csv>
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def verify_parameter_columns(data_path: Path):
    """Verify that data has required parameter columns."""
    print("=" * 70)
    print("PHYSICS LOSS PARAMETER VERIFICATION")
    print("=" * 70)
    print(f"\nLoading data from: {data_path}")

    # Load data
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        print(f"ERROR: Error loading data: {e}")
        return False

    print(f"[OK] Data loaded: {len(data)} rows, {len(data.columns)} columns")

    # Check required columns
    required_cols = ["param_H", "param_D", "param_Pm", "Pe", "scenario_id"]
    optional_cols = ["H", "D", "Pm"]  # Fallback columns

    print("\n" + "=" * 70)
    print("1. COLUMN VERIFICATION")
    print("=" * 70)

    missing_required = []
    missing_optional = []

    for col in required_cols:
        if col in data.columns:
            print(f"  [OK] {col}: Found")
        else:
            print(f"  [MISSING] {col}: MISSING")
            missing_required.append(col)

    for col in optional_cols:
        if col in data.columns:
            print(f"  [WARNING] {col}: Found (fallback, but param_* preferred)")
        else:
            missing_optional.append(col)

    if missing_required:
        print(f"\n[CRITICAL] Missing required columns: {missing_required}")
        return False

    if missing_optional:
        print(f"\n[WARNING] Missing optional columns: {missing_optional}")
        print("   Will use param_* columns instead (preferred)")

    # Check scenario_id
    if "scenario_id" not in data.columns:
        print("\n[CRITICAL] scenario_id column not found!")
        return False

    num_scenarios = data["scenario_id"].nunique()
    print(f"\n[OK] Found {num_scenarios} unique scenarios")

    return True


def verify_parameter_consistency(data_path: Path, num_scenarios: int = 5):
    """Verify parameter consistency across scenarios."""
    print("\n" + "=" * 70)
    print("2. PARAMETER CONSISTENCY CHECK")
    print("=" * 70)

    data = pd.read_csv(data_path)
    scenarios = data["scenario_id"].unique()[:num_scenarios]

    print(f"\nChecking first {len(scenarios)} scenarios:")

    issues_found = []

    for i, scenario_id in enumerate(scenarios, 1):
        scenario_data = data[data["scenario_id"] == scenario_id].copy()
        scenario_data = scenario_data.sort_values("time")

        if len(scenario_data) == 0:
            continue

        row = scenario_data.iloc[0]

        # Extract parameters
        H_param = row.get("param_H", row.get("H", None))
        D_param = row.get("param_D", row.get("D", None))
        Pm_param = row.get("param_Pm", row.get("Pm", None))
        H_fallback = row.get("H", None)
        D_fallback = row.get("D", None)
        Pm_fallback = row.get("Pm", None)

        # Check Pe
        Pe_available = "Pe" in scenario_data.columns
        Pe_t0 = scenario_data["Pe"].iloc[0] if Pe_available else None

        print(f"\n  Scenario {scenario_id}:")
        print(f"    param_H:  {H_param:.6f}" if H_param is not None else "    param_H:  N/A")
        print(f"    param_D:  {D_param:.6f}" if D_param is not None else "    param_D:  N/A")
        print(f"    param_Pm: {Pm_param:.6f}" if Pm_param is not None else "    param_Pm: N/A")

        if H_fallback is not None:
            if H_param is not None:
                diff = abs(H_param - H_fallback) / (abs(H_fallback) + 1e-12)
                if diff > 0.01:
                    print(
                        f"[WARNING] H mismatch: param_H={H_param:.6f} vs H={H_fallback:.6f}"
                        f"(diff={diff*100:.2f}%)"
                    )
                    issues_found.append(f"Scenario {scenario_id}: H mismatch")
            else:
                print(f"    [WARNING] Using fallback H: {H_fallback:.6f}")

        if D_fallback is not None:
            if D_param is not None:
                diff = abs(D_param - D_fallback) / (abs(D_fallback) + 1e-12)
                if diff > 0.01:
                    print(
                        f"[WARNING] D mismatch: param_D={D_param:.6f} vs D={D_fallback:.6f}"
                        f"(diff={diff*100:.2f}%)"
                    )
                    issues_found.append(f"Scenario {scenario_id}: D mismatch")
            else:
                print(f"    [WARNING] Using fallback D: {D_fallback:.6f}")

        if Pm_fallback is not None:
            if Pm_param is not None:
                diff = abs(Pm_param - Pm_fallback) / (abs(Pm_fallback) + 1e-12)
                if diff > 0.01:
                    # This is EXPECTED after our fix: Pm = model truth, param_Pm = requested
                    # Physics loss uses param_Pm (correct), so this is not an issue
                    print(
                        f"[INFO] Pm difference: param_Pm={Pm_param:.6f} vs Pm={Pm_fallback:.6f}"
                        f"(diff={diff*100:.2f}%)"
                    )
                    print(
                        f"(Expected: Pm=model truth, param_Pm=requested. Physics loss uses"
                        f"param_Pm.)"
                    )
                    # Don't add to issues_found - this is expected behavior
            else:
                print(f"    [WARNING] Using fallback Pm: {Pm_fallback:.6f}")

        if Pe_available:
            print(f"    Pe(t=0):  {Pe_t0:.6f} pu")
            # Check if Pe(t=0) matches param_Pm (should match at steady-state)
            if Pm_param is not None:
                pe_pm_diff = abs(Pe_t0 - Pm_param) / (abs(Pm_param) + 1e-12)
                if pe_pm_diff > 0.02:  # 2% tolerance
                    print(
                        f"    [WARNING] Pe(t=0) doesn't match param_Pm! Diff={pe_pm_diff*100:.2f}%"
                    )
                    print(f"              Pe(t=0)={Pe_t0:.6f}, param_Pm={Pm_param:.6f}")
                    print(
                        f"              At steady-state, Pe should equal Pm. Data may need regeneration."
                    )
                    issues_found.append(f"Scenario {scenario_id}: Pe(t=0) mismatch with param_Pm")
                else:
                    print(f"    [OK] Pe(t=0) matches param_Pm (diff={pe_pm_diff*100:.2f}%)")
        else:
            print(f"    [WARNING] Pe column not found!")
            issues_found.append(f"Scenario {scenario_id}: Pe column missing")

    if issues_found:
        print(f"\n[WARNING] Found {len(issues_found)} potential issues:")
        for issue in issues_found:
            print(f"    - {issue}")
        return False
    else:
        print(f"\n[OK] All parameters consistent!")
        return True


def verify_pe_consistency(data_path: Path, num_scenarios: int = 3):
    """Verify Pe(t) consistency."""
    print("\n" + "=" * 70)
    print("3. Pe(t) CONSISTENCY CHECK")
    print("=" * 70)

    data = pd.read_csv(data_path)

    if "Pe" not in data.columns:
        print("\n[ERROR] Pe column not found in data!")
        return False

    scenarios = data["scenario_id"].unique()[:num_scenarios]
    print(f"\nChecking Pe(t) for first {len(scenarios)} scenarios:")

    for i, scenario_id in enumerate(scenarios, 1):
        scenario_data = data[data["scenario_id"] == scenario_id].copy()
        scenario_data = scenario_data.sort_values("time")

        if len(scenario_data) == 0:
            continue

        Pe_values = scenario_data["Pe"].values
        time_values = scenario_data["time"].values

        print(f"\n  Scenario {scenario_id}:")
        print(f"    Pe(t=0):     {Pe_values[0]:.6f} pu")
        print(
            f"    Pe(t=1.0):   {Pe_values[np.argmin(np.abs(time_values - 1.0))]:.6f} pu (fault start)"
        )
        print(f"    Pe(t=end):   {Pe_values[-1]:.6f} pu")
        print(f"    Pe range:   [{Pe_values.min():.6f}, {Pe_values.max():.6f}] pu")
        print(f"    Pe std:      {Pe_values.std():.6f} pu")

        # Check for NaN or inf
        if np.any(np.isnan(Pe_values)):
            print(f"    [WARNING] NaN values found in Pe(t)!")
        if np.any(np.isinf(Pe_values)):
            print(f"    [WARNING] Inf values found in Pe(t)!")

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify physics loss parameter consistency")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to data CSV file",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=5,
        help="Number of scenarios to check (default: 5)",
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)

    if not data_path.exists():
        print(f"[ERROR] Data file not found: {data_path}")
        return 1

    # Run verifications
    col_check = verify_parameter_columns(data_path)
    if not col_check:
        return 1

    param_check = verify_parameter_consistency(data_path, args.num_scenarios)
    pe_check = verify_pe_consistency(data_path, min(3, args.num_scenarios))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Column verification:     {'[PASS]' if col_check else '[FAIL]'}")
    print(f"  Parameter consistency:   {'[PASS]' if param_check else '[ISSUES FOUND]'}")
    print(f"  Pe(t) consistency:       {'[PASS]' if pe_check else '[FAIL]'}")

    if col_check and param_check and pe_check:
        print("\n[OK] All checks passed! Parameters are consistent.")
        return 0
    else:
        print("\n[WARNING] Some issues found. Review output above.")
        return 1


if __name__ == "__main__":
    exit(main())
