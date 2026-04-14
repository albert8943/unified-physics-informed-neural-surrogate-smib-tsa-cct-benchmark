"""
Physics Validation Utilities for PINN Training.

Critical pre-flight checks to ensure physics assumptions are valid
before training the PINN model.
"""

import andes
import numpy as np
import pandas as pd
from pathlib import Path


def validate_omega_units(data: pd.DataFrame) -> bool:
    """
    Verify omega is in per-unit (required for physics equation).

    The swing equation dδ/dt = 2πf₀(ω - 1) assumes ω is in per-unit
    where ω = 1.0 pu corresponds to synchronous speed.

    Parameters:
    -----------
    data : pd.DataFrame
        Trajectory data with 'time' and 'omega' columns

    Returns:
    --------
    bool : True if omega is in pu, False otherwise
    """
    print("\n" + "=" * 70)
    print("OMEGA UNIT VALIDATION")
    print("=" * 70)

    # Check steady-state omega (before fault at t < 0.5s)
    steady_state_data = data[data["time"] < 0.5]

    if len(steady_state_data) == 0:
        print("⚠️ No steady-state data found (t < 0.5s)")
        return None

    steady_omega = steady_state_data["omega"].mean()
    omega_std = steady_state_data["omega"].std()

    print(f"Steady-state omega statistics:")
    print(f"  Mean: {steady_omega:.6f}")
    print(f"  Std:  {omega_std:.6f}")

    # Check if omega is in per-unit (should be ~1.0 pu at steady state)
    if 0.95 <= steady_omega <= 1.05:
        print(f"\n✓ Omega appears to be in PER-UNIT")
        print(f"  Steady-state value ({steady_omega:.4f}) is close to 1.0 pu")
        print(f"  This is CORRECT for the physics equation")
        return True

    # Check if omega is in Hz (would be ~60 Hz for 60 Hz system)
    elif 59 <= steady_omega <= 61:
        print(f"\n✗ Omega appears to be in HERTZ")
        print(f"  Steady-state value ({steady_omega:.2f}) is close to 60 Hz")
        print(f"  ⚠️ CRITICAL ERROR: Physics equation requires per-unit!")
        print(f"\n  FIX: Convert to pu: omega_pu = omega_hz / 60.0")
        return False

    # Check if omega is in rad/s (would be ~377 rad/s for 60 Hz system)
    elif 370 <= steady_omega <= 385:
        print(f"\n✗ Omega appears to be in RAD/S")
        print(f"  Steady-state value ({steady_omega:.2f}) is close to 377 rad/s")
        print(f"  ⚠️ CRITICAL ERROR: Physics equation requires per-unit!")
        print(f"\n  FIX: Convert to pu: omega_pu = omega_rad_s / (2*π*60)")
        return False

    else:
        print(f"\n⚠️ Omega units are UNCLEAR")
        print(f"  Steady-state value: {steady_omega:.4f}")
        print(f"  Expected ranges:")
        print(f"    - Per-unit: 0.95 - 1.05")
        print(f"    - Hertz: 59 - 61")
        print(f"    - Rad/s: 370 - 385")
        print(f"\n  Please verify the omega units in your data!")
        return None


def validate_power_balance(data: pd.DataFrame, V1: float = None, V2: float = None) -> bool:
    """
    Check pre-fault power balance: P_e = P_m at steady-state.

    At steady-state (before fault), the electrical power should equal
    the mechanical power: P_e = (V₁V₂/X)*sin(δ) = P_m

    Parameters:
    -----------
    data : pd.DataFrame
        Scenario data with steady-state values
    V1 : float, optional
        Generator voltage (pu). If None, will try to extract from data or calculate.
    V2 : float, optional
        Infinite bus voltage (pu). If None, will try to extract from data or calculate.

    Returns:
    --------
    bool : True if power balance is satisfied (error < 0.01 pu)
    """
    print("\n" + "=" * 70)
    print("PRE-FAULT POWER BALANCE VALIDATION")
    print("=" * 70)

    # Get first data point (steady-state before fault)
    if len(data) == 0:
        print("✗ No data provided")
        return False

    # Filter for steady-state data (before fault, typically t < 0.5s)
    # Use minimum time point (t ≈ 0) for best steady-state representation
    if "scenario_id" in data.columns and "time" in data.columns:
        first_scenario_id = data["scenario_id"].iloc[0]
        first_scenario = data[data["scenario_id"] == first_scenario_id]
        steady_state_data = first_scenario[first_scenario["time"] < 0.5]
        if len(steady_state_data) > 0:
            # Use minimum time point (closest to t=0) for steady-state
            t0_data = steady_state_data.loc[steady_state_data["time"].idxmin()]
        else:
            # Fallback to first row if no pre-fault data found
            t0_data = first_scenario.iloc[0]
    elif "time" in data.columns:
        # Filter for steady-state (t < 0.5s)
        steady_state_data = data[data["time"] < 0.5]
        if len(steady_state_data) > 0:
            # Use minimum time point (closest to t=0) for steady-state
            t0_data = steady_state_data.loc[steady_state_data["time"].idxmin()]
        else:
            t0_data = data.iloc[0]
    else:
        # No time column, use first row
        t0_data = data.iloc[0]

    # Extract values
    delta0 = t0_data["delta"]
    # CRITICAL: Use Pm (model truth from ANDES) for physics consistency, not param_Pm (requested)
    # param_Pm is metadata and may not match what ANDES actually simulated
    Pm = t0_data.get("Pm", t0_data.get("param_Pm"))
    # Xprefault may not exist in data when using pe_direct input method
    Xprefault = t0_data.get("Xprefault", None)
    t0_time = t0_data.get("time", 0.0)

    # If Xprefault is missing, we can't compute SMIB formula but can still validate using Pe from data
    if Xprefault is None:
        print("⚠️  Xprefault not found in data (likely using pe_direct input method)")
        print(
            "  Will validate power balance using P_e from data only (cannot compute SMIB formula)"
        )

    # Check if data has actual P_e column (preferred over calculated)
    Pe_actual = None
    if "Pe" in t0_data:
        Pe_actual = t0_data["Pe"]
        # Handle sign convention (some systems use negative for generation)
        if Pe_actual < 0:
            Pe_actual = abs(Pe_actual)  # Use absolute value for comparison

    # Try to extract voltages from data if not provided
    if V1 is None:
        V1 = t0_data.get("V1", t0_data.get("V_gen", t0_data.get("voltage", 1.05)))
        if V1 == 1.05 and "V1" not in t0_data and "V_gen" not in t0_data:
            print("⚠️ V1 not found in data, using default 1.05 pu")

    if V2 is None:
        V2 = t0_data.get("V2", t0_data.get("V_bus", t0_data.get("V_inf", 1.0)))
        if V2 == 1.0 and "V2" not in t0_data and "V_bus" not in t0_data:
            print("⚠️ V2 not found in data, using default 1.0 pu")

    # Compute electrical power at steady-state using SMIB formula (for reference only)
    # NOTE: This formula is NOT authoritative - ANDES uses network reduction, Thevenin equivalents,
    # etc. that may not match this simplified SMIB model. Use only as a reference/comparison.
    Pe_calc = None
    if Xprefault is not None and abs(Xprefault) >= 1e-9:
        Pe_calc = (V1 * V2 / Xprefault) * np.sin(delta0)
    elif Xprefault is None:
        print(
            "⚠️  Xprefault not available. Cannot compute SMIB formula (using pe_direct input method)."
        )
    else:
        print("⚠️  Xprefault is invalid (too small). Cannot compute SMIB formula.")

    # PRIMARY CHECK: Compare P_e from data (ANDES-native) with P_m
    # This is the authoritative check since P_e from data comes directly from ANDES simulation
    if Pe_actual is not None:
        Pe_to_compare = Pe_actual
        Pe_source = "from data (ANDES-native)"
    elif Pe_calc is not None:
        # Fallback: if no P_e in data, use calculated (but warn)
        Pe_to_compare = Pe_calc
        Pe_source = "calculated from formula (WARNING: may not match ANDES model)"
        print(
            f"⚠️ No P_e column in data. Using SMIB formula as fallback, "
            f"but this may not match ANDES's internal model."
        )
    else:
        # Cannot validate without either Pe from data or Xprefault for formula
        print(
            "⚠️  Cannot validate power balance: Missing both P_e from data and Xprefault for formula."
        )
        print("  This is expected when using pe_direct input method without P_e column.")
        return True  # Return True to not block execution, but validation is incomplete

    # Compute error (compare P_e with P_m)
    error = abs(Pe_to_compare - Pm)
    error_percent = 100 * error / Pm if Pm != 0 else float("inf")

    print(f"Pre-fault power balance (t = {t0_time:.6f} s):")
    print(f"  P_m (mechanical): {Pm:.6f} pu")
    if Pe_actual is not None:
        print(f"  P_e (from data, ANDES-native): {Pe_actual:.6f} pu")
        if Pe_calc is not None:
            print(f"  P_e (SMIB formula, reference): {Pe_calc:.6f} pu")
            print(f"    Note: SMIB formula may not match ANDES's internal model")
    elif Pe_calc is not None:
        print(f"  P_e (calculated from formula): {Pe_calc:.6f} pu")
        print(f"    ⚠️ WARNING: No P_e in data. Formula may not match ANDES model.")
    print(f"  δ₀ (rotor angle): {delta0:.6f} rad ({np.degrees(delta0):.2f}°)")
    if Xprefault is not None:
        print(f"  X_prefault:       {Xprefault:.6f} pu")
    print(f"  V₁ (used):        {V1:.6f} pu")
    print(f"  V₂ (used):        {V2:.6f} pu")
    print(f"  V₁ × V₂:          {V1 * V2:.6f} pu")
    print(f"\nPower balance error (P_e from data vs P_m):")
    print(f"  Absolute: {error:.6f} pu")
    print(f"  Relative: {error_percent:.2f}%")

    # Check if balance is satisfied
    if error < 0.01:
        print(f"\n✓ Power balance OK (error < 1% of Pm)")
        print(f"  P_e ≈ P_m at steady-state")
        return True
    elif error < 0.05:
        print(f"\n⚠️ Power balance acceptable (error < 5% of Pm)")
        print(f"  Small mismatch may affect physics loss accuracy")
        return True
    else:
        print(f"\n✗ Power balance FAILED (error > 5% of Pm)")

        # PRIMARY DIAGNOSTIC: Compare P_e from data with P_m
        print(f"\n  DIAGNOSTIC: Power balance check")
        if Pe_actual is not None:
            print(f"    P_e (from data, ANDES-native): {Pe_actual:.6f} pu")
            print(f"    P_m (mechanical):              {Pm:.6f} pu")
            print(f"    Error:                         {error:.6f} pu ({error_percent:.2f}%)")
            print(f"\n    This suggests:")
            print(f"    - P_m may not have been correctly applied to ANDES before simulation")
            print(f"    - OR P_e extraction may be incorrect")
            print(f"    - OR data is not at true steady-state (t≈0)")
        else:
            print(f"    P_e (calculated from formula): {Pe_calc:.6f} pu")
            print(f"    P_m (mechanical):               {Pm:.6f} pu")
            print(f"    Error:                         {error:.6f} pu ({error_percent:.2f}%)")
            print(f"    ⚠️ WARNING: No P_e in data. Formula may not match ANDES model.")

        # Reference comparison: Show SMIB formula result (for information only)
        if Pe_actual is not None:
            calc_error = abs(Pe_calc - Pe_actual)
            print(f"\n  Reference comparison (SMIB formula vs ANDES-native P_e):")
            print(f"    P_e (SMIB formula): {Pe_calc:.6f} pu")
            print(f"    P_e (from data):   {Pe_actual:.6f} pu")
            error_pct = 100 * calc_error / (Pe_actual + 1e-8)
            print(f"    Difference:        {calc_error:.6f} pu ({error_pct:.2f}%)")
            if calc_error > 0.1:
                print(f"    Note: Large difference is expected - SMIB formula is simplified")
                print(f"          and may not match ANDES's network reduction/Thevenin model.")

        print(f"\n  Possible causes:")
        print(f"    1. P_m not correctly applied to ANDES before power flow initialization")
        print(f"    2. P_e extraction error (check extraction method)")
        print(f"    3. Data not at true steady-state (check time point)")
        print(f"    4. Parameter mismatch between metadata and actual simulation")
        print(f"\n  FIX: Verify P_m is set in ANDES (ss.GENCLS.tm0.v) before PFlow.run()")
        print(f"       Check that P_e extraction uses correct variable index")
        return False


def verify_system_frequency(case_file: str) -> float:
    """
    Confirm system frequency (affects M = 2H calculation).

    The inertia constant conversion M = 2H is valid for 60 Hz systems.
    For other frequencies, use: M = 2H * (f_nominal / 60)

    Parameters:
    -----------
    case_file : str
        Path to ANDES case file

    Returns:
    --------
    float : System frequency in Hz
    """
    print("\n" + "=" * 70)
    print("SYSTEM FREQUENCY VALIDATION")
    print("=" * 70)

    try:
        # Load ANDES case
        ss = andes.load(case_file, no_output=True, default_config=True)

        # Get system frequency
        f_nominal = ss.config.freq

        print(f"System frequency: {f_nominal} Hz")

        if f_nominal == 60.0:
            print(f"\n✓ System is 60 Hz")
            print(f"  Standard conversion: M = 2H")
            print(f"  ω_base = 2π × 60 = 377 rad/s")
        elif f_nominal == 50.0:
            print(f"\n⚠️ System is 50 Hz (not 60 Hz)")
            print(f"  Modified conversion: M = 2H × (50/60) = 2H × 0.833")
            print(f"  ω_base = 2π × 50 = 314 rad/s")
            print(f"\n  FIX: Update M calculation in code:")
            print(f"       M = 2 * H * (50 / 60)")
        else:
            print(f"\n⚠️ Non-standard frequency: {f_nominal} Hz")
            print(f"  Modified conversion: M = 2H × ({f_nominal}/60)")
            print(f"\n  FIX: Update M calculation in code:")
            print(f"       M = 2 * H * ({f_nominal} / 60)")

        return f_nominal

    except Exception as e:
        print(f"\n✗ Error loading case file: {e}")
        print(f"  Could not verify system frequency")
        return None


def find_smib_case() -> Path:
    """
    Find SMIB case file path.

    Tries to get the SMIB case from ANDES built-in cases.
    Falls back to default path if not found.

    Returns:
    --------
    Path
        Path to SMIB case file
    """
    try:
        case_path = andes.get_case("smib/SMIB.json")
        return Path(case_path)
    except Exception:
        # Fallback to default path
        return Path("smib/SMIB.json")


def extract_voltages_from_andes(case_file: str, gen_idx: int = 0) -> tuple[float, float]:
    """
    Extract V1 (generator voltage) and V2 (infinite bus voltage) from ANDES case.

    Parameters:
    -----------
    case_file : str
        Path to ANDES case file
    gen_idx : int
        Generator index (default: 0)

    Returns:
    --------
    tuple[float, float]
        (V1, V2) voltages in pu, or (None, None) if extraction fails
    """
    try:
        # Load ANDES case and run power flow
        ss = andes.load(case_file, no_output=True, default_config=True)

        # Run power flow to get voltages
        ss.PFlow.run()

        if not hasattr(ss, "PFlow") or not (hasattr(ss.PFlow, "converged") and ss.PFlow.converged):
            return None, None

        # Extract V1 (generator bus voltage)
        V1 = None
        if hasattr(ss, "GENCLS") and hasattr(ss.GENCLS, "bus") and hasattr(ss.GENCLS.bus, "v"):
            gen_bus = (
                ss.GENCLS.bus.v[gen_idx]
                if hasattr(ss.GENCLS.bus.v, "__getitem__")
                else ss.GENCLS.bus.v
            )

            if hasattr(ss, "Bus") and hasattr(ss.Bus, "v") and hasattr(ss.Bus.v, "v"):
                bus_indices = ss.Bus.idx.v if hasattr(ss.Bus.idx, "v") else []
                if hasattr(bus_indices, "__iter__"):
                    try:
                        bus_idx = list(bus_indices).index(gen_bus)
                        V1 = float(ss.Bus.v.v[bus_idx])
                    except (ValueError, IndexError):
                        pass

        # Extract V2 (infinite bus voltage - usually the other bus in SMIB)
        V2 = None
        if hasattr(ss, "Bus") and hasattr(ss.Bus, "v") and hasattr(ss.Bus.v, "v"):
            # For SMIB, V2 is typically the infinite bus (slack bus or bus 2)
            # Try to find the bus that's not the generator bus
            bus_indices = ss.Bus.idx.v if hasattr(ss.Bus.idx, "v") else []
            if hasattr(bus_indices, "__iter__") and len(bus_indices) > 1:
                # Get all bus voltages
                all_voltages = ss.Bus.v.v
                if hasattr(all_voltages, "__iter__"):
                    # Find the bus that's not the generator bus
                    for i, bus_id in enumerate(bus_indices):
                        if bus_id != gen_bus:
                            V2 = float(all_voltages[i])
                            break
                    # If only one bus found, use it as V2 (infinite bus)
                    if V2 is None and len(bus_indices) == 1:
                        V2 = float(all_voltages[0])

        return V1, V2

    except Exception as e:
        print(f"⚠️  Could not extract voltages from ANDES: {e}")
        return None, None


def run_all_validations(
    data: pd.DataFrame, case_file: str, V1: float = None, V2: float = None
) -> dict:
    """
    Run all physics validations and return results.

    Parameters:
    -----------
    data : pd.DataFrame
        Trajectory data
    case_file : str
        Path to ANDES case file
    V1 : float, optional
        Generator voltage. If None, will try to extract from data or ANDES case.
    V2 : float, optional
        Infinite bus voltage. If None, will try to extract from data or ANDES case.

    Returns:
    --------
    dict : Validation results
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE PHYSICS VALIDATION")
    print("=" * 70)
    print("\nRunning pre-flight checks before training...")

    results = {}

    # 1. Omega units
    results["omega_valid"] = validate_omega_units(data)

    # 2. Extract voltages if not provided
    if V1 is None or V2 is None:
        # First, try to extract from data
        if len(data) > 0:
            t0_data = data.iloc[0]
            if V1 is None:
                V1 = t0_data.get("V1", t0_data.get("V_gen", None))
            if V2 is None:
                V2 = t0_data.get("V2", t0_data.get("V_bus", t0_data.get("V_inf", None)))

        # If still not found, try to extract from ANDES case file
        if (V1 is None or V2 is None) and case_file:
            print("\nAttempting to extract voltages from ANDES case file...")
            V1_extracted, V2_extracted = extract_voltages_from_andes(case_file)
            if V1 is None and V1_extracted is not None:
                V1 = V1_extracted
                print(f"✓ Extracted V1 = {V1:.6f} pu from ANDES")
            if V2 is None and V2_extracted is not None:
                V2 = V2_extracted
                print(f"✓ Extracted V2 = {V2:.6f} pu from ANDES")

        # If still None, use defaults (will trigger warning in validate_power_balance)
        if V1 is None:
            V1 = 1.05
        if V2 is None:
            V2 = 1.0

    # 3. Power balance (on scenario-level data)
    if "scenario_id" in data.columns:
        # Get first time point of first scenario
        first_scenario = data[data["scenario_id"] == data["scenario_id"].iloc[0]]
        scenario_data = first_scenario.groupby("scenario_id").first().reset_index()
        results["power_balance_valid"] = validate_power_balance(scenario_data, V1, V2)
    else:
        results["power_balance_valid"] = validate_power_balance(data, V1, V2)

    # 3. System frequency
    results["system_frequency"] = verify_system_frequency(case_file)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = all(
        [
            results["omega_valid"] == True,
            results["power_balance_valid"] == True,
            results["system_frequency"] in [50.0, 60.0],
        ]
    )

    if results["omega_valid"] == True:
        omega_status = "✓ PASS"
    elif results["omega_valid"] == False:
        omega_status = "✗ FAIL"
    else:
        omega_status = "⚠️ UNCLEAR"
    print(f"\nOmega units:     {omega_status}")
    print(f"Power balance:   {'✓ PASS' if results['power_balance_valid'] else '✗ FAIL'}")
    print(f"System frequency: {results['system_frequency']} Hz")

    if all_passed:
        print(f"\n{'='*70}")
        print("✓ ALL VALIDATIONS PASSED")
        print("=" * 70)
        print("\nYou can proceed with training!")
    else:
        print(f"\n{'='*70}")
        print("✗ SOME VALIDATIONS FAILED")
        print("=" * 70)
        print("\n⚠️ STOP! Fix the issues before training:")
        if results["omega_valid"] == False:
            print("  - Convert omega to per-unit")
        if not results["power_balance_valid"]:
            print("  - Check V1, V2, or pre-fault data")
        if results["system_frequency"] not in [50.0, 60.0]:
            print("  - Update M calculation for non-standard frequency")

    return results


if __name__ == "__main__":
    print("Physics Validation Utilities")
    print(
        "Import and use validate_omega_units(), validate_power_balance(), verify_system_frequency()"
    )
