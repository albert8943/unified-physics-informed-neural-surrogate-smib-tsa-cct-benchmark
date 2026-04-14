"""
Quick verification script to test the new workflow setup.

This script verifies that:
1. All required modules can be imported
2. Case file modifier functions are available
3. Verification helpers are available
4. Parameter sweep can use the new workflow
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 70)
    print("VERIFYING NEW WORKFLOW SETUP")
    print("=" * 70)

    errors = []

    # Test case file modifier
    print("\n1. Testing case_file_modifier imports...")
    try:
        from data_generation.andes_utils.case_file_modifier import (
            modify_case_file_generator_setpoint,
            get_default_pm_from_case_file,
            modify_case_file_multiple_generators,
        )

        print("   [OK] case_file_modifier imports successful")
    except ImportError as e:
        print(f"   [ERROR] case_file_modifier import failed: {e}")
        errors.append(f"case_file_modifier: {e}")

    # Test verification helpers
    print("\n2. Testing verification_helpers imports...")
    try:
        from data_generation.andes_utils.verification_helpers import (
            verify_power_flow_converged,
            verify_generator_setpoints,
            verify_power_balance,
        )

        print("   [OK] verification_helpers imports successful")
    except ImportError as e:
        print(f"   [ERROR] verification_helpers import failed: {e}")
        errors.append(f"verification_helpers: {e}")

    # Test parameter_sweep can import them
    print("\n3. Testing parameter_sweep module...")
    try:
        from data_generation.parameter_sweep import generate_parameter_sweep

        print("   [OK] parameter_sweep import successful")

        # Check if CASE_FILE_MODIFIER_AVAILABLE is set
        import data_generation.parameter_sweep as ps_module

        if hasattr(ps_module, "CASE_FILE_MODIFIER_AVAILABLE"):
            if ps_module.CASE_FILE_MODIFIER_AVAILABLE:
                print("   [OK] CASE_FILE_MODIFIER_AVAILABLE = True")
            else:
                print("   [WARNING] CASE_FILE_MODIFIER_AVAILABLE = False (will use fallback)")
                errors.append("CASE_FILE_MODIFIER_AVAILABLE is False")
        else:
            print("   [WARNING] CASE_FILE_MODIFIER_AVAILABLE attribute not found")
    except ImportError as e:
        print(f"   [ERROR] parameter_sweep import failed: {e}")
        errors.append(f"parameter_sweep: {e}")

    # Test data_generation wrapper (optional - requires torch)
    print("\n4. Testing data_generation wrapper...")
    try:
        from scripts.core.data_generation import generate_training_data

        print("   [OK] data_generation wrapper import successful")
    except ImportError as e:
        if "torch" in str(e).lower():
            print(f"   [SKIP] data_generation wrapper requires torch (not critical for workflow)")
            print("         This is expected if torch is not installed in current environment.")
        else:
            print(f"   [ERROR] data_generation wrapper import failed: {e}")
            errors.append(f"data_generation wrapper: {e}")

    # Summary
    print("\n" + "=" * 70)
    if errors:
        print("[FAILED] VERIFICATION FAILED")
        print("\nErrors found:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease check the error messages above and fix the issues.")
        return False
    else:
        print("[PASSED] VERIFICATION PASSED")
        print("\nAll modules are properly set up!")
        print("You can use your existing scripts with the new workflow.")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
