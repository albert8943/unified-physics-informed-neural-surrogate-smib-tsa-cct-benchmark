#!/usr/bin/env python
"""
Quick test script for SMIB CCT Load Variation experiment.

Tests the configuration with a small sample (3-5 load levels) to ensure:
- CCT finding works correctly
- Power balance validation works
- Data generation completes successfully

Usage:
    python scripts/test_smib_cct_load_variation.py
"""

import sys
import yaml
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.data_generation import generate_training_data
from scripts.core.utils import load_config


def main():
    """Test SMIB CCT load variation experiment with small sample."""
    print("=" * 70)
    print("SMIB CCT Load Variation - Quick Test")
    print("=" * 70)
    print()

    # Load config
    config_path = PROJECT_ROOT / "configs" / "experiments" / "smib_cct_load_variation.yaml"
    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    # Modify for quick test: reduce to 5 load levels
    print("\nModifying config for quick test (5 load levels instead of 18)...")
    original_alpha = config["data"]["generation"]["parameter_ranges"]["alpha"]
    config["data"]["generation"]["parameter_ranges"]["alpha"] = [0.4, 1.2, 5]
    print(f"  Original alpha range: {original_alpha}")
    print(f"  Test alpha range: {config['data']['generation']['parameter_ranges']['alpha']}")
    print(f"  Expected scenarios: 5 load levels × 5 trajectories = 25 scenarios")

    # Create test output directory
    output_dir = PROJECT_ROOT / "data" / "generated" / "smib_cct_load_variation_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Verify configuration
    print("\nVerifying configuration...")
    gen_config = config["data"]["generation"]
    param_ranges = gen_config["parameter_ranges"]

    print(f"  Case file: {gen_config.get('case_file')}")
    print(f"  H: {param_ranges.get('H')} (fixed)")
    print(f"  D: {param_ranges.get('D')} (fixed)")
    print(f"  Alpha: {param_ranges.get('alpha')} (varying)")
    print(f"  Sampling strategy: {gen_config.get('sampling_strategy')}")
    print(f"  CCT-based sampling: {gen_config.get('use_cct_based_sampling')}")
    print(f"  Trajectories per combination: {gen_config.get('n_samples_per_combination')}")

    # Run data generation
    print("\n" + "=" * 70)
    print("Starting data generation...")
    print("=" * 70)
    print("This may take 10-20 minutes for 5 load levels...")
    print()

    try:
        data_path, validation_results = generate_training_data(
            config=config,
            output_dir=output_dir,
            validate_physics=True,
            skip_if_exists=False,
            use_common_repository=False,  # Use local directory for test
            force_regenerate=False,
        )

        print("\n" + "=" * 70)
        print("✓ Data generation completed successfully!")
        print("=" * 70)
        print(f"Data saved to: {data_path}")

        if validation_results:
            print("\nValidation results:")
            print(
                f"  Physics validation: {'PASSED' if validation_results.get('passed', False) else 'FAILED'}"
            )
            if validation_results.get("issues"):
                print(f"  Issues found: {len(validation_results['issues'])}")
                for issue in validation_results["issues"][:5]:  # Show first 5
                    print(f"    - {issue}")

        print("\n" + "=" * 70)
        print("Quick test completed successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Review the generated data file")
        print("  2. Check metadata for CCT values at each load level")
        print("  3. Verify power balance and stability distribution")
        print("  4. If test passes, run full experiment with 18 load levels")
        print()

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ Data generation failed!")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
