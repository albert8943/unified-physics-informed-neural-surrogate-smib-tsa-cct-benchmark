#!/usr/bin/env python
"""
Compare data generation configs to ensure consistency.

Usage:
    python scripts/utils/compare_configs.py
"""

import sys
from pathlib import Path
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compare_configs():
    """Compare quick, moderate, and comprehensive configs."""

    config_dir = PROJECT_ROOT / "configs" / "data_generation"

    # Load configs
    quick = yaml.safe_load(open(config_dir / "quick.yaml"))
    moderate = yaml.safe_load(open(config_dir / "moderate.yaml"))
    comprehensive = yaml.safe_load(open(config_dir / "comprehensive.yaml"))

    print("=" * 70)
    print("CONFIG COMPARISON: quick vs moderate vs comprehensive")
    print("=" * 70)
    print()

    # Core parameters that should match
    core_params = {
        "sampling.strategy": ("sampling", "strategy"),
        "sampling.n_samples_per_combination": ("sampling", "n_samples_per_combination"),
        "sampling.additional_clearing_time_offsets": (
            "sampling",
            "additional_clearing_time_offsets",
        ),
        "sampling.use_cct_based_sampling": ("sampling", "use_cct_based_sampling"),
        "simulation.time_step": ("simulation", "time_step"),
        "parameter_ranges.Pm": ("parameter_ranges", "Pm"),
        "parameter_ranges.M": ("parameter_ranges", "M"),
        "parameter_ranges.D": ("parameter_ranges", "D"),
    }

    print("✅ CORE PARAMETERS (Should Match):")
    print("-" * 70)
    all_match = True
    for param_name, (section, key) in core_params.items():
        quick_val = quick.get(section, {}).get(key)
        mod_val = moderate.get(section, {}).get(key)
        comp_val = comprehensive.get(section, {}).get(key)

        match = quick_val == mod_val == comp_val
        status = "✅" if match else "❌"

        if not match:
            all_match = False

        print(f"{status} {param_name}:")
        print(f"   Quick:         {quick_val}")
        print(f"   Moderate:      {mod_val}")
        print(f"   Comprehensive: {comp_val}")
        print()

    # Expected differences
    print("⚠️  INTENTIONAL DIFFERENCES (Expected):")
    print("-" * 70)

    # n_samples
    print(f"n_samples:")
    print(
        f"Quick: {quick['sampling']['n_samples']} → {quick['sampling']['n_samples'] * quick['sampling']['n_samples_per_combination']} trajectories"
    )
    print(
        f"Moderate: {moderate['sampling']['n_samples']} → {moderate['sampling']['n_samples'] * moderate['sampling']['n_samples_per_combination']} trajectories"
    )
    print(
        f"Comprehensive: {comprehensive['sampling']['n_samples']} → {comprehensive['sampling']['n_samples'] * comprehensive['sampling']['n_samples_per_combination']} trajectories"
    )
    print()

    # CCT finding
    print("CCT Finding (tolerance_final):")
    print(f"   Quick:         {quick['cct_finding']['tolerance_final']} (relaxed for speed)")
    print(f"   Moderate:      {moderate['cct_finding']['tolerance_final']} (strict)")
    print(f"   Comprehensive: {comprehensive['cct_finding']['tolerance_final']} (strict)")
    print()

    # Data quality
    print("Data Quality (min_completeness):")
    print(f"   Quick:         {quick['data_quality']['min_completeness']} (relaxed)")
    print(f"   Moderate:      {moderate['data_quality']['min_completeness']} (strict)")
    print(f"   Comprehensive: {comprehensive['data_quality']['min_completeness']} (strict)")
    print()

    # Summary
    print("=" * 70)
    if all_match:
        print("✅ ALL CORE PARAMETERS MATCH!")
        print()
        print("Summary:")
        print("  - All configs use same sampling strategy (Sobol)")
        print("  - All configs use same parameter ranges")
        print("  - All configs use same time_step (0.002s)")
        print("  - All configs use same clearing time offsets")
        print("  - Differences are intentional (speed vs. accuracy)")
        print()
        print("✅ No errors found - configs are consistent!")
    else:
        print("❌ SOME CORE PARAMETERS DO NOT MATCH!")
        print("   Please review the differences above.")

    print("=" * 70)


if __name__ == "__main__":
    compare_configs()
