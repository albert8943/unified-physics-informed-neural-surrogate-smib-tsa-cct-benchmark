#!/usr/bin/env python
"""
Complete GENROU Validation Workflow Script.

Runs the entire GENROU validation workflow:
1. Setup (create/validate case file)
2. Run validation experiments
3. Compare GENCLS vs GENROU
4. Generate summary report

Usage:
    python scripts/validation/run_complete_genrou_validation.py \
        --pinn-model validation/delta_weight_sweep/exp_delta20.0_omega40.0/exp_20260116_014607/pinn/model/best_model_20260116_020404.pth \
        --test-scenarios data/processed/exp_20260116_014611/test_data_20260116_014614.csv \
        --output-dir outputs/publication/genrou_validation \
        [--quick-test] [--skip-setup] [--skip-comparison]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_command(cmd: list, description: str, check: bool = True) -> bool:
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=check, capture_output=False)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {cmd[0]}")
        return False


def check_prerequisites(
    genrou_case: Path,
    test_scenarios: Path,
    pinn_model: Path,
) -> bool:
    """Check if all prerequisites are met."""
    print("\n" + "=" * 70)
    print("CHECKING PREREQUISITES")
    print("=" * 70)

    all_ok = True

    # Check GENROU case file
    if genrou_case.exists():
        print(f"✓ GENROU case file exists: {genrou_case}")
    else:
        print(f"✗ GENROU case file missing: {genrou_case}")
        all_ok = False

    # Check test data
    if test_scenarios.exists():
        print(f"✓ Test data exists: {test_scenarios}")
    else:
        print(f"✗ Test data missing: {test_scenarios}")
        all_ok = False

    # Check model (handle wildcards)
    model_files = list(pinn_model.parent.glob(pinn_model.name))
    if model_files:
        print(f"✓ PINN model found: {model_files[0]}")
    else:
        print(f"✗ PINN model missing: {pinn_model}")
        all_ok = False

    return all_ok


def create_genrou_case(
    input_case: str = "smib/SMIB.json",
    output_case: str = "test_cases/SMIB_genrou.json",
) -> bool:
    """Create GENROU case file."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "validation" / "create_genrou_case.py"),
        "--input-case",
        input_case,
        "--output-case",
        output_case,
    ]
    return run_command(cmd, "Create GENROU Case File")


def validate_genrou_case(
    genrou_case: str = "test_cases/SMIB_genrou.json",
) -> bool:
    """Validate GENROU case file."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "validation" / "validate_genrou_case.py"),
        "--genrou-case",
        genrou_case,
    ]
    return run_command(cmd, "Validate GENROU Case File", check=False)


def run_genrou_validation(
    pinn_model: str,
    genrou_case: str,
    test_scenarios: str,
    output_dir: str,
    max_scenarios: Optional[int] = None,
) -> bool:
    """Run GENROU validation."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "validation" / "genrou_validation.py"),
        "--pinn-model",
        pinn_model,
        "--genrou-case",
        genrou_case,
        "--test-scenarios",
        test_scenarios,
        "--output-dir",
        output_dir,
    ]

    if max_scenarios:
        cmd.extend(["--max-scenarios", str(max_scenarios)])

    return run_command(cmd, "Run GENROU Validation")


def compare_gencls_genrou(
    genrou_results: str,
    output_dir: str,
    gencls_results: Optional[str] = None,
) -> bool:
    """Compare GENCLS vs GENROU."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "validation" / "compare_gencls_genrou.py"),
        "--genrou-results",
        genrou_results,
        "--output-dir",
        output_dir,
    ]

    if gencls_results:
        cmd.extend(["--gencls-results", gencls_results])

    return run_command(cmd, "Compare GENCLS vs GENROU", check=False)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Complete GENROU Validation Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (3 scenarios)
  python scripts/validation/run_complete_genrou_validation.py \\
      --pinn-model validation/delta_weight_sweep/exp_delta20.0_omega40.0/exp_20260116_014607/pinn/model/best_model_20260116_020404.pth \\
      --test-scenarios data/processed/exp_20260116_014611/test_data_20260116_014614.csv \\
      --output-dir outputs/publication/genrou_validation \\
      --quick-test

  # Full validation (all scenarios)
  python scripts/validation/run_complete_genrou_validation.py \\
      --pinn-model validation/delta_weight_sweep/exp_delta20.0_omega40.0/exp_20260116_014607/pinn/model/best_model_20260116_020404.pth \\
      --test-scenarios data/processed/exp_20260116_014611/test_data_20260116_014614.csv \\
      --output-dir outputs/publication/genrou_validation

  # With GENCLS comparison
  python scripts/validation/run_complete_genrou_validation.py \\
      --pinn-model validation/delta_weight_sweep/exp_delta20.0_omega40.0/exp_20260116_014607/pinn/model/best_model_20260116_020404.pth \\
      --test-scenarios data/processed/exp_20260116_014611/test_data_20260116_014614.csv \\
      --output-dir outputs/publication/genrou_validation \\
      --gencls-results outputs/publication/statistical_validation/experiments/exp_20260119_095052/pinn/results/metrics.json
        """,
    )

    parser.add_argument(
        "--pinn-model",
        type=str,
        required=True,
        help="Path to trained PINN model (.pth file, wildcards supported)",
    )
    parser.add_argument(
        "--test-scenarios",
        type=str,
        required=True,
        help="Path to test data CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/publication/genrou_validation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--genrou-case",
        type=str,
        default="test_cases/SMIB_genrou.json",
        help="Path to GENROU case file (will be created if missing)",
    )
    parser.add_argument(
        "--gencls-results",
        type=str,
        default=None,
        help="Path to GENCLS evaluation results (optional, for comparison)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with 3 scenarios only",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip case file creation/validation (assume already done)",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip GENCLS vs GENROU comparison",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Maximum number of scenarios to validate (overrides --quick-test)",
    )

    args = parser.parse_args()

    # Resolve paths
    genrou_case_path = Path(args.genrou_case)
    test_scenarios_path = Path(args.test_scenarios)
    pinn_model_path = Path(args.pinn_model)
    base_output_dir = Path(args.output_dir)

    # Note: genrou_validation.py will generate its own experiment_id

    # Handle wildcards in model path
    if "*" in str(pinn_model_path):
        model_files = list(pinn_model_path.parent.glob(pinn_model_path.name))
        if model_files:
            pinn_model_path = model_files[0]
            print(f"Found model: {pinn_model_path}")
        else:
            print(f"Error: No model found matching {args.pinn_model}")
            sys.exit(1)

    # Determine max scenarios
    max_scenarios = args.max_scenarios
    if args.quick_test and max_scenarios is None:
        max_scenarios = 3

    print("=" * 70)
    print("GENROU VALIDATION - COMPLETE WORKFLOW")
    print("=" * 70)
    print(f"PINN Model: {pinn_model_path}")
    print(f"Test Scenarios: {test_scenarios_path}")
    print(f"GENROU Case: {genrou_case_path}")
    print(f"Base Output Directory: {base_output_dir}")
    print(f"Max Scenarios: {max_scenarios if max_scenarios else 'All'}")
    print(f"Quick Test: {args.quick_test}")
    print("Note: Timestamped experiment folder will be created automatically")
    print("=" * 70)

    # Phase 1: Setup
    if not args.skip_setup:
        print("\n" + "=" * 70)
        print("PHASE 1: SETUP & PREPARATION")
        print("=" * 70)

        # Check prerequisites
        if not check_prerequisites(genrou_case_path, test_scenarios_path, pinn_model_path):
            # Try to create GENROU case file if missing
            if not genrou_case_path.exists():
                print("\nGENROU case file missing. Creating...")
                if not create_genrou_case(
                    input_case="smib/SMIB.json",
                    output_case=str(genrou_case_path),
                ):
                    print("Error: Failed to create GENROU case file")
                    sys.exit(1)

            # Re-check prerequisites
            if not check_prerequisites(genrou_case_path, test_scenarios_path, pinn_model_path):
                print("Error: Prerequisites not met. Please check file paths.")
                sys.exit(1)

        # Validate case file
        validate_genrou_case(str(genrou_case_path))
    else:
        print("\nSkipping setup phase (--skip-setup)")

    # Phase 2: Run Validation
    print("\n" + "=" * 70)
    print("PHASE 2: RUN VALIDATION EXPERIMENTS")
    print("=" * 70)

    if not run_genrou_validation(
        pinn_model=str(pinn_model_path),
        genrou_case=str(genrou_case_path),
        test_scenarios=str(test_scenarios_path),
        output_dir=str(base_output_dir),
        max_scenarios=max_scenarios,
    ):
        print("Error: Validation failed")
        sys.exit(1)

    # Phase 3: Comparison
    # Find the latest experiment directory created by genrou_validation.py
    if not args.skip_comparison:
        print("\n" + "=" * 70)
        print("PHASE 3: COMPARISON & ANALYSIS")
        print("=" * 70)

        # Find latest experiment directory
        exp_dirs = sorted(
            [d for d in base_output_dir.glob("exp_*") if d.is_dir()],
            key=lambda x: x.name,
            reverse=True,
        )
        genrou_results_path = None
        if exp_dirs:
            latest_exp_dir = exp_dirs[0]
            results_dir = latest_exp_dir / "results"
            genrou_results_path = results_dir / "genrou_validation_results.json"
            comparison_output_dir = results_dir / "comparison"

        if genrou_results_path is not None and genrou_results_path.exists():
            compare_gencls_genrou(
                genrou_results=str(genrou_results_path),
                output_dir=str(comparison_output_dir),
                gencls_results=args.gencls_results,
            )
        else:
            print(f"Warning: Results file not found: {genrou_results_path}")
            print("Skipping comparison")
    else:
        print("\nSkipping comparison phase (--skip-comparison)")

    # Summary
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    # Find latest experiment directory for summary
    exp_dirs = sorted(
        [d for d in base_output_dir.glob("exp_*") if d.is_dir()], key=lambda x: x.name, reverse=True
    )
    if exp_dirs:
        latest_exp_dir = exp_dirs[0]
        results_dir = latest_exp_dir / "results"
        actual_experiment_id = latest_exp_dir.name
    else:
        latest_exp_dir = base_output_dir
        results_dir = base_output_dir
        actual_experiment_id = "unknown"

    print(f"\nExperiment ID: {actual_experiment_id}")
    print(f"Results saved to: {latest_exp_dir}")
    print(f"\nOutput files:")
    print(f"  - Validation results: {results_dir / 'genrou_validation_results.json'}")
    print(f"  - Summary: {results_dir / 'summary.json'}")
    if (
        not args.skip_comparison
        and (results_dir / "comparison" / "comparison_results.json").exists()
    ):
        print(f"  - Comparison: {results_dir / 'comparison' / 'comparison_results.json'}")

    print("\nNext steps:")
    print("  1. Review results in output directory")
    print("  2. Create visualizations (see GENROU_VALIDATION_PUBLICATION_WORKFLOW.md)")
    print("  3. Generate publication materials")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
