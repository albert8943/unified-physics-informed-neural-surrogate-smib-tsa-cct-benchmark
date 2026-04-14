"""
Diagnostic script for power balance validation issues.

This script helps identify why power balance validation is failing by:
1. Checking P_e values at different time points
2. Comparing P_e from data vs calculated
3. Verifying voltages, reactance, and angles
4. Suggesting fixes
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly from physics_validation to avoid torch dependency
import importlib.util

physics_validation_path = project_root / "utils" / "physics_validation.py"
spec = importlib.util.spec_from_file_location("physics_validation", physics_validation_path)
physics_validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(physics_validation)

extract_voltages_from_andes = physics_validation.extract_voltages_from_andes
find_smib_case = physics_validation.find_smib_case


def diagnose_power_balance(data_path: str, case_file: str = None):
    """
    Diagnose power balance validation issues.

    Parameters:
    -----------
    data_path : str
        Path to data CSV file
    case_file : str, optional
        Path to ANDES case file. If None, will try to find SMIB case.
    """
    print("=" * 70)
    print("POWER BALANCE DIAGNOSTIC")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"[OK] Loaded {len(data):,} rows")

    # Get first scenario
    if "scenario_id" in data.columns:
        first_scenario_id = data["scenario_id"].iloc[0]
        scenario_data = data[data["scenario_id"] == first_scenario_id].copy()
        print(f"[OK] Analyzing scenario_id: {first_scenario_id}")
    else:
        scenario_data = data.copy()
        print("[WARNING] No scenario_id column found, analyzing all data")

    # 1. Check P_e at different time points
    print("\n" + "=" * 70)
    print("1. P_e VALUES AT DIFFERENT TIME POINTS")
    print("=" * 70)

    # Steady-state (t < 0.5)
    steady_state = scenario_data[scenario_data["time"] < 0.5].copy()
    if len(steady_state) > 0:
        t0_data = steady_state.loc[steady_state["time"].idxmin()]
        print(f"\nSteady-state (t = {t0_data['time']:.6f} s):")
        print(f"  P_m: {t0_data['Pm']:.6f} pu")
        print(f"  P_e: {t0_data['Pe']:.6f} pu")
        print(f"  delta0:  {t0_data['delta']:.6f} rad ({np.degrees(t0_data['delta']):.2f} deg)")
        print(f"  omega0:  {t0_data['omega']:.6f} pu")
        if "Xprefault" in t0_data:
            print(f"  X_prefault: {t0_data['Xprefault']:.6f} pu")

        error = abs(t0_data["Pm"] - t0_data["Pe"])
        error_pct = 100 * error / t0_data["Pm"] if t0_data["Pm"] != 0 else 0
        print(f"\n  Power balance error:")
        print(f"    Absolute: {error:.6f} pu")
        print(f"    Relative: {error_pct:.2f}%")

        if error_pct > 5:
            print(f"    [FAIL] FAILED (error > 5%)")
        else:
            print(f"    [OK] PASSED")
    else:
        print("[WARNING] No steady-state data found (t < 0.5)")
        t0_data = scenario_data.iloc[0]

    # During fault (t ≈ 1.0)
    during_fault = scenario_data[
        (scenario_data["time"] >= 0.99) & (scenario_data["time"] <= 1.01)
    ].copy()
    if len(during_fault) > 0:
        fault_data = during_fault.iloc[0]
        print(f"\nDuring fault (t = {fault_data['time']:.6f} s):")
        print(f"  P_m: {fault_data['Pm']:.6f} pu")
        print(f"  P_e: {fault_data['Pe']:.6f} pu")
        print(f"  delta:   {fault_data['delta']:.6f} rad")
        print(f"  Note: P_e should be near zero during fault")

    # Post-fault (t > 1.1)
    post_fault = scenario_data[scenario_data["time"] > 1.1].copy()
    if len(post_fault) > 0:
        post_data = post_fault.iloc[0]
        print(f"\nPost-fault (t = {post_data['time']:.6f} s):")
        print(f"  P_m: {post_data['Pm']:.6f} pu")
        print(f"  P_e: {post_data['Pe']:.6f} pu")
        print(f"  delta:   {post_data['delta']:.6f} rad")

    # 2. Calculate P_e from formula
    print("\n" + "=" * 70)
    print("2. CALCULATED P_e FROM FORMULA")
    print("=" * 70)

    # Extract voltages
    if case_file is None:
        case_file = str(find_smib_case())

    print(f"\nExtracting voltages from: {case_file}")
    V1, V2 = extract_voltages_from_andes(case_file, gen_idx=0)

    if V1 is None or V2 is None:
        print("[WARNING] Could not extract voltages, using defaults")
        V1 = 1.05
        V2 = 1.0
    else:
        print(f"[OK] Extracted V1 = {V1:.6f} pu")
        print(f"[OK] Extracted V2 = {V2:.6f} pu")

    # Check if voltages are in data
    if "V1" in t0_data:
        V1_data = t0_data["V1"]
        print(f"  V1 from data: {V1_data:.6f} pu")
        if abs(V1_data - V1) > 0.01:
            print(f"  [WARNING] V1 mismatch: extracted ({V1:.6f}) vs data ({V1_data:.6f})")

    if "V2" in t0_data:
        V2_data = t0_data["V2"]
        print(f"  V2 from data: {V2_data:.6f} pu")
        if abs(V2_data - V2) > 0.01:
            print(f"  [WARNING] V2 mismatch: extracted ({V2:.6f}) vs data ({V2_data:.6f})")

    # Calculate P_e
    if "Xprefault" in t0_data:
        Xprefault = t0_data["Xprefault"]
        delta0 = t0_data["delta"]
        Pm = t0_data["Pm"]

        Pe_calc = (V1 * V2 / Xprefault) * np.sin(delta0)

        print(f"\nCalculated P_e:")
        print(f"  Formula: P_e = (V1 * V2 / X_prefault) * sin(delta0)")
        print(f"  P_e = ({V1:.6f} × {V2:.6f} / {Xprefault:.6f}) × sin({delta0:.6f})")
        print(f"  P_e = {Pe_calc:.6f} pu")
        print(f"\n  Comparison:")
        print(f"    P_m:           {Pm:.6f} pu")
        print(f"    P_e (data):    {t0_data['Pe']:.6f} pu")
        print(f"    P_e (calc):    {Pe_calc:.6f} pu")

        error_data = abs(t0_data["Pe"] - Pm)
        error_calc = abs(Pe_calc - Pm)

        print(f"\n  Errors:")
        print(f"    P_e (data) vs P_m: {error_data:.6f} pu ({100*error_data/Pm:.2f}%)")
        print(f"    P_e (calc) vs P_m: {error_calc:.6f} pu ({100*error_calc/Pm:.2f}%)")

        # Determine which is more accurate
        if error_data < error_calc:
            print(f"\n  -> P_e from data is closer to P_m")
            print(f"    But still has {100*error_data/Pm:.2f}% error")
        else:
            print(f"\n  -> P_e calculated is closer to P_m")
            print(f"    But still has {100*error_calc/Pm:.2f}% error")
            print(f"    This suggests voltages or reactance might be wrong")

    # 3. Check what V1×V2 should be
    print("\n" + "=" * 70)
    print("3. REQUIRED V1 * V2 FOR POWER BALANCE")
    print("=" * 70)

    if "Xprefault" in t0_data and abs(np.sin(delta0)) > 1e-6:
        V1V2_required = Pm * Xprefault / np.sin(delta0)
        V1V2_current = V1 * V2

        print(f"\nFor P_e = P_m:")
        print(f"  Required V1 * V2 = P_m * X_prefault / sin(delta0)")
        print(f"  Required V1 * V2 = {Pm:.6f} * {Xprefault:.6f} / sin({delta0:.6f})")
        print(f"  Required V1 * V2 = {V1V2_required:.6f} pu")
        print(f"  Current V1 * V2  = {V1V2_current:.6f} pu")
        print(f"  Difference       = {abs(V1V2_required - V1V2_current):.6f} pu")

        if 0.9 <= V1V2_required <= 1.2:
            print(f"\n  [OK] Required V1*V2 is reasonable (0.9-1.2 pu)")
            print(f"    Suggested fix: Adjust voltages")
            V1_suggested = 1.05
            V2_suggested = V1V2_required / V1_suggested
            print(f"    Suggested: V1 = {V1_suggested:.4f} pu, V2 = {V2_suggested:.4f} pu")
        else:
            print(f"\n  [WARNING] Required V1*V2 ({V1V2_required:.4f}) is unusual")
            print(f"    This suggests:")
            print(f"    - X_prefault might be wrong")
            print(f"    - delta0 might not be at steady-state")
            print(f"    - P_m value might be incorrect")

    # 4. Recommendations
    print("\n" + "=" * 70)
    print("4. RECOMMENDATIONS")
    print("=" * 70)

    recommendations = []

    # Check P_e from data
    if abs(t0_data["Pe"] - Pm) > 0.5 * Pm:
        recommendations.append(
            "P_e from data differs significantly from P_m (>50%)"
            "\n  -> Likely cause: P_e extraction from ANDES is incorrect"
            "\n  -> Fix: Verify P_e extraction method in data_generation/andes_extractor.py"
            "\n  -> Check: Is P_e from wrong time point or sign convention issue?"
        )

    # Check calculated P_e
    if "Xprefault" in t0_data:
        if abs(Pe_calc - Pm) > 0.5 * Pm:
            recommendations.append(
                "P_e calculated from formula differs significantly from P_m (>50%)"
                "\n  -> Likely cause: Voltages (V1, V2) or reactance (X_prefault) are incorrect"
                "\n  -> Fix: Extract voltages from TDS results, not power flow"
                "\n  -> Fix: Verify reactance calculation includes all components"
            )

    # Check voltages
    if abs(V1V2_required - V1V2_current) > 0.1:
        recommendations.append(
            f"V1*V2 mismatch: required {V1V2_required:.4f} vs current {V1V2_current:.4f}"
            "\n  -> Likely cause: Voltages extracted from power flow don't match simulation"
            "\n  -> Fix: Extract voltages from TDS results at t=0"
        )

    if len(recommendations) == 0:
        print("\n[OK] No obvious issues found. Power balance should be satisfied.")
    else:
        print("\n[WARNING] Issues found:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\nFor detailed fixes, see: docs/guides/POWER_BALANCE_VALIDATION_FIXES.md")


def main():
    parser = argparse.ArgumentParser(description="Diagnose power balance validation issues")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to data CSV file",
    )
    parser.add_argument(
        "--case-file",
        type=str,
        default=None,
        help="Path to ANDES case file (default: auto-detect SMIB case)",
    )

    args = parser.parse_args()

    diagnose_power_balance(args.data_path, args.case_file)


if __name__ == "__main__":
    main()
