#!/usr/bin/env python3
"""
Diagnostic script to investigate why CCT finding fails 100% of the time.

This script tests CCT finding with verbose output to identify the root cause.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import andes
from data_generation.andes_utils.cct_finder import find_cct, test_clearing_time

# Configure verbose logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def diagnose_cct_failure():
    """Diagnose why CCT finding is failing."""

    # Use the same parameters from the experiment
    # Get case file from ANDES built-in cases
    try:
        case_path = andes.get_case("smib/SMIB.json")
        print(f"Using ANDES case: {case_path}")
    except Exception as e:
        print(f"ERROR: Could not get SMIB case from ANDES: {e}")
        print("Make sure ANDES is installed: pip install andes")
        return
    fault_start_time = 1.0
    simulation_time = 5.0
    fault_bus = 3
    fault_reactance = 0.0001
    time_step = 0.002

    # Parameters from the experiment (H=2.79, D=2.13, Pm=0.538)
    M = 2.793088423088193  # H converted to M (M = 2*H for GENCLS)
    D = 2.1287359446287155
    Pm = 0.5382420598752796

    # Search range from parameter_sweep.py
    min_tc = fault_start_time + 0.01  # 1.01s
    max_tc = simulation_time - 0.5  # 4.5s

    print("=" * 70)
    print("CCT FINDING DIAGNOSTIC")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  M (inertia): {M:.3f} s")
    print(f"  D (damping): {D:.3f} pu")
    print(f"  Pm (mechanical power): {Pm:.3f} pu")
    print(f"  Search range: [{min_tc:.3f}, {max_tc:.3f}] s")
    print(f"  Fault start time: {fault_start_time:.3f} s")
    print(f"  Simulation time: {simulation_time:.3f} s")
    print("=" * 70)

    # Test boundaries first
    print("\n[STEP 1] Testing minimum clearing time (min_tc)...")
    print(f"  Testing tc = {min_tc:.6f} s")
    is_stable_min, max_angle_min, traj_min, metrics_min, ss_min = test_clearing_time(
        case_path=case_path,
        Pm=Pm,
        M=M,
        D=D,
        clearing_time=min_tc,
        fault_start_time=fault_start_time,
        fault_bus=fault_bus,
        fault_reactance=fault_reactance,
        simulation_time=simulation_time,
        time_step=time_step,
        logger=logger,
        ss=None,
        reload_system=True,
    )
    print(f"  Result: {'STABLE' if is_stable_min else 'UNSTABLE'}")
    print(f"  Max angle: {max_angle_min:.2f}°")
    if "error" in metrics_min:
        print(f"  Error: {metrics_min['error']}")

    print("\n[STEP 2] Testing maximum clearing time (max_tc)...")
    print(f"  Testing tc = {max_tc:.6f} s")
    is_stable_max, max_angle_max, traj_max, metrics_max, ss_max = test_clearing_time(
        case_path=case_path,
        Pm=Pm,
        M=M,
        D=D,
        clearing_time=max_tc,
        fault_start_time=fault_start_time,
        fault_bus=fault_bus,
        fault_reactance=fault_reactance,
        simulation_time=simulation_time,
        time_step=time_step,
        logger=logger,
        ss=None,
        reload_system=True,
    )
    print(f"  Result: {'STABLE' if is_stable_max else 'UNSTABLE'}")
    print(f"  Max angle: {max_angle_max:.2f}°")
    if "error" in metrics_max:
        print(f"  Error: {metrics_max['error']}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS:")
    print("=" * 70)

    if is_stable_min and is_stable_max:
        print("[X] PROBLEM IDENTIFIED: System is ALWAYS STABLE")
        print(f"   - Stable at min_tc = {min_tc:.3f} s")
        print(f"   - Stable at max_tc = {max_tc:.3f} s")
        print(f"   - CCT is BEYOND the search range (> {max_tc:.3f} s)")
        print("\n   SOLUTIONS:")
        print("   1. Increase max_tc (e.g., max_tc = simulation_time - 0.1)")
        print("   2. Reduce system stability (lower H, higher Pm, or longer fault)")
        print("   3. Use fixed clearing times instead of CCT-based sampling")
    elif not is_stable_min and not is_stable_max:
        print("[X] PROBLEM IDENTIFIED: System is ALWAYS UNSTABLE")
        print(f"   - Unstable at min_tc = {min_tc:.3f} s")
        print(f"   - Unstable at max_tc = {max_tc:.3f} s")
        print(f"   - CCT is BELOW the search range (< {min_tc:.3f} s)")
        print("\n   SOLUTIONS:")
        print("   1. Decrease min_tc (e.g., min_tc = fault_start_time + 0.001)")
        print("   2. Increase system stability (higher H, lower Pm, or shorter fault)")
        print("   3. Use fixed clearing times instead of CCT-based sampling")
    else:
        print("✓ System has stable/unstable boundary in search range")
        print("   CCT finding should work. Testing full CCT finding...")

        # Try full CCT finding
        print("\n[STEP 3] Running full CCT finding...")
        cct_duration, uncertainty, stable_result, unstable_result = find_cct(
            case_path=case_path,
            Pm=Pm,
            M=M,
            D=D,
            fault_start_time=fault_start_time,
            fault_bus=fault_bus,
            fault_reactance=fault_reactance,
            min_tc=min_tc,
            max_tc=max_tc,
            simulation_time=simulation_time,
            time_step=time_step,
            tolerance_initial=0.01,
            tolerance_final=0.001,
            max_iterations=50,
            logger=logger,
            ss=None,
            reload_system=True,
        )

        if cct_duration is None:
            print("[X] CCT finding failed even though boundary exists")
            print("   This may indicate:")
            print("   - Binary search convergence issues")
            print("   - Power flow failures during search")
            print("   - Numerical instability")
        else:
            print(f"[OK] CCT found: {cct_duration:.6f} s")
            print(f"  Uncertainty: +/-{uncertainty:.6f} s")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    print("1. If system is always stable:")
    print("   - Increase max_tc in parameter_sweep.py (line 1845)")
    print("   - Or disable CCT-based sampling (use fixed clearing times)")
    print("2. If system is always unstable:")
    print("   - Decrease min_tc in parameter_sweep.py (line 1844)")
    print("   - Or check if parameters are physically reasonable")
    print("3. Add verbose logging to find_cct() to see binary search progress")
    print("=" * 70)


if __name__ == "__main__":
    diagnose_cct_failure()
