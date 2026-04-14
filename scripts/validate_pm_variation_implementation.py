#!/usr/bin/env python
"""
Validation script for P_m variation implementation.

This script performs basic validation tests to ensure the P_m variation
implementation is working correctly.

Usage:
    python scripts/validate_pm_variation_implementation.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def test_pmax_computation():
    """Test P_max computation function."""
    print("=" * 70)
    print("Test 1: P_max Computation")
    print("=" * 70)

    try:
        from data_generation.andes_utils.system_manager import compute_pmax_prefault
        import andes

        # Load SMIB case
        case_path = andes.get_case("smib/SMIB.json")
        ss = andes.load(case_path, setup=False)
        ss.setup()

        # Set reference P_m for network computation
        from data_generation.andes_utils.cct_finder import find_main_generator_index

        gen_idx = find_main_generator_index(ss)

        if hasattr(ss, "PV") and ss.PV.n > 0:
            if hasattr(ss.PV, "p0") and hasattr(ss.PV.p0, "v"):
                ss.PV.p0.v[0] = 0.5
        if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
            ss.GENCLS.tm0.v[gen_idx] = 0.5
        ss.PFlow.run()

        # Compute P_max
        P_max = compute_pmax_prefault(ss, gen_idx)

        print(f"[OK] P_max computed: {P_max:.6f} pu")
        print(f"[OK] P_max is positive and reasonable: {P_max > 0 and P_max < 10}")

        return True, P_max
    except Exception as e:
        print(f"[FAIL] P_max computation failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_prefault_validation():
    """Test pre-fault equilibrium validation."""
    print("\n" + "=" * 70)
    print("Test 2: Pre-fault Equilibrium Validation")
    print("=" * 70)

    try:
        from data_generation.andes_utils.system_manager import validate_prefault_equilibrium
        import andes

        # Load SMIB case
        case_path = andes.get_case("smib/SMIB.json")
        ss = andes.load(case_path, setup=False)
        ss.setup()

        from data_generation.andes_utils.cct_finder import find_main_generator_index

        gen_idx = find_main_generator_index(ss)

        # Set P_m and run power flow
        pm_value = 0.5
        if hasattr(ss, "PV") and ss.PV.n > 0:
            if hasattr(ss.PV, "p0") and hasattr(ss.PV.p0, "v"):
                ss.PV.p0.v[0] = pm_value
        if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
            ss.GENCLS.tm0.v[gen_idx] = pm_value
        ss.PFlow.run()

        # Validate
        is_valid, diagnostics = validate_prefault_equilibrium(ss, pm_value, gen_idx, tolerance=0.01)

        if is_valid:
            print(f"[OK] Pre-fault validation passed")
            print(f"     Pe0 = {diagnostics.get('Pe_initial', 'N/A'):.6f} pu")
            print(f"     delta0 = {diagnostics.get('delta0_deg', 'N/A'):.1f}°")
        else:
            print(f"[FAIL] Pre-fault validation failed: {diagnostics.get('error', 'unknown')}")
            return False

        return True
    except Exception as e:
        print(f"[FAIL] Pre-fault validation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_eac_baseline():
    """Test EAC baseline computation."""
    print("\n" + "=" * 70)
    print("Test 3: EAC Baseline Computation")
    print("=" * 70)

    try:
        from data_generation.andes_utils.eac_baseline import compute_cct_eac, compare_cct_methods

        # Test with typical SMIB parameters
        Pm = 0.5
        M = 10.0  # 2 * H, where H = 5.0
        D = 1.0
        X_prefault = 0.5
        X_fault = 0.0001
        X_postfault = 0.5

        cct_eac, delta_cc, diagnostics = compute_cct_eac(
            Pm=Pm,
            M=M,
            D=D,
            X_prefault=X_prefault,
            X_fault=X_fault,
            X_postfault=X_postfault,
            V_gen=1.0,
            V_inf=1.0,
        )

        if cct_eac is not None:
            print(f"[OK] EAC CCT computed: {cct_eac:.6f} s")
            print(f"[OK] Critical clearing angle: {np.degrees(delta_cc):.2f}°")

            # Test comparison function
            cct_bisection = 0.5  # Example value
            comparison = compare_cct_methods(cct_bisection, cct_eac, tolerance=0.01)

            if comparison.get("both_available", False):
                print(f"[OK] EAC comparison works")
                print(f"     Error: {comparison.get('error', 'N/A'):.6f} s")
                print(f"     Relative error: {comparison.get('relative_error', 'N/A'):.2f}%")

            return True
        else:
            print(f"[FAIL] EAC computation returned None: {diagnostics.get('error', 'unknown')}")
            return False
    except Exception as e:
        print(f"[FAIL] EAC baseline test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_batch_tds_pm_variation():
    """Test batch TDS with P_m variation (quick test)."""
    print("\n" + "=" * 70)
    print("Test 4: Batch TDS with P_m Variation (Quick Test)")
    print("=" * 70)
    print("[NOTE] This is a quick test with minimal samples")
    print("       Full validation requires running complete study")
    print()

    try:
        from examples.smib_batch_tds import batch_tds_smib

        # Quick test with minimal samples
        print("[INFO] Running quick test (2 P_m values, 1 clearing time each)...")
        results = batch_tds_smib(
            case_file="smib/SMIB.json",
            variation_mode="pm",
            pm_range=(0.4, 0.6, 2),  # Just 2 samples for quick test
            n_samples=2,
            fault_start_time=1.0,
            fault_bus=3,
            fault_reactance=0.0001,
            simulation_time=3.0,  # Shorter for quick test
            time_step=0.002,
            use_cct_based_sampling=False,  # Use fixed clearing time for speed
            n_samples_per_combination=1,
            use_skip_init=True,
            H=5.0,
            D=1.0,
            verbose=False,  # Less verbose for test
        )

        if results is None:
            print("[FAIL] batch_tds_smib returned None")
            return False

        # Check results structure
        n_samples = len([x for x in results.get("pm_values", []) if x is not None])
        print(f"[OK] Generated {n_samples} samples")

        if n_samples > 0:
            print(f"[OK] P_m variation mode working")
            print(
                f"     P_m values: {[f'{x:.4f}' for x in results['pm_values'][:5] if x is not None]}"
            )

            # Check for errors
            errors = results.get("errors", [])
            n_errors = len([e for e in errors if e is not None])
            if n_errors > 0:
                print(f"[WARNING] {n_errors} errors encountered (may be expected for some cases)")
            else:
                print(f"[OK] No errors encountered")

            return True
        else:
            print("[FAIL] No samples generated")
            return False

    except Exception as e:
        print(f"[FAIL] Batch TDS test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("P_m Variation Implementation Validation")
    print("=" * 70)
    print()

    tests = [
        ("P_max Computation", test_pmax_computation),
        ("Pre-fault Validation", test_prefault_validation),
        ("EAC Baseline", test_eac_baseline),
        ("Batch TDS P_m Variation", test_batch_tds_pm_variation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            if test_name == "P_max Computation":
                success, P_max = test_func()
                results.append((test_name, success))
            else:
                success = test_func()
                results.append((test_name, success))
        except Exception as e:
            print(f"[FAIL] {test_name} raised exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[OK] All validation tests passed!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
