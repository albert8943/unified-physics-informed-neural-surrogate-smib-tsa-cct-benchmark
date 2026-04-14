#!/usr/bin/env python3
"""
Test script to verify Pe matches Pm after TDS.init() with the updated case file modifier.

This script generates a single trajectory with verbose output to verify that:
1. Case file modification works (PV/Slack components are modified)
2. TDS.init() reads correct Pe from modified case file
3. Pe(t=0) matches Pm within acceptable tolerance
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_generation.parameter_sweep import generate_parameter_sweep


def test_single_trajectory():
    """Test single trajectory generation with verbose output."""

    print("=" * 80)
    print("SINGLE TRAJECTORY TEST - Pe INITIALIZATION VERIFICATION")
    print("=" * 80)
    print()
    print("This test generates a single trajectory with:")
    print("  - alpha = 1.0 (base load)")
    print("  - H = 4.0 (typical inertia)")
    print("  - D = 1.0 (typical damping)")
    print("  - Verbose output to track Pe initialization")
    print()
    print("Expected results:")
    print("  1. Case file modification: PV.p0 or Slack.p0 modified")
    print("  2. TDS.init(): Pe matches expected Pm")
    print("  3. Pe(t=0) matches Pm within 1%")
    print()
    print("=" * 80)
    print()

    # Generate single trajectory with verbose output
    # Use our custom SMIB case with StaticGen component
    custom_case_path = project_root / "test_cases" / "smib_with_staticgen" / "SMIB_StaticGen.json"
    result = generate_parameter_sweep(
        case_file=str(custom_case_path),
        output_dir="data",
        # Use load variation with alpha=1.0 (base load)
        alpha_range=(1.0, 1.0, 1),  # Single alpha
        H_range=(4.0, 4.0, 1),  # Single H
        D_range=(1.0, 1.0, 1),  # Single D
        # Configuration
        fault_clearing_times=[1.15],  # Single clearing time (must be >= fault_start_time=1.0s)
        fault_locations=[3],  # Single fault location
        use_cct_based_sampling=False,  # Disable CCT finding for simplicity
        fault_start_time=1.0,  # Fault starts at 1.0s
        verbose=True,  # Enable verbose output
    )

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()

    if result is not None and isinstance(result, dict):
        if "trajectories" in result:
            num_trajectories = len(result["trajectories"])
        elif "data" in result:
            num_trajectories = len(result["data"])
        else:
            num_trajectories = 0
        print(f"[SUCCESS] Generated {num_trajectories} trajectory(ies)")
        if "output_path" in result:
            print(f"[SUCCESS] Data saved to: {result['output_path']}")
        print()
        print("Review the output above to verify:")
        print("  1. '[PHASE 1] [LOAD VARIATION] Modified case file' message")
        print("  2. '[SUCCESS] Case file modification worked!' message")
        print("  3. '[SUCCESS] Pe(t=0) matches Pm within 1%' message")
        print()
        return 0
    else:
        print("[ERROR] Trajectory generation failed")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(test_single_trajectory())
