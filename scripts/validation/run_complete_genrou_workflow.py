#!/usr/bin/env python
"""
Complete GENROU Validation Workflow: One-Command Solution.

Runs the entire GENROU validation and analysis pipeline:
1. Full validation (26 scenarios)
2. Statistical analysis
3. Parameter sensitivity analysis
4. Visualizations
5. GENCLS comparison (optional)
6. LaTeX table generation
7. Trajectory figure (optional)

Usage:
    python scripts/validation/run_complete_genrou_workflow.py \
        --pinn-model validation/delta_weight_sweep/exp_delta20.0_omega40.0/exp_20260116_014607/pinn/model/best_model_20260116_020404.pth \
        --test-scenarios data/processed/exp_20260116_014611/test_data_20260116_014614.csv \
        --output-dir outputs/publication/genrou_validation \
        [--gencls-results PATH] \
        [--skip-analysis] \
        [--skip-trajectory] \
        [--dpi 300]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Fix encoding for Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def find_latest_experiment(output_dir: Path) -> Optional[Path]:
    """
    Find the latest experiment directory in the output directory.

    Parameters:
    -----------
    output_dir : Path
        Base output directory

    Returns:
    --------
    latest_exp : Path or None
        Path to latest experiment directory
    """
    if not output_dir.exists():
        return None

    # Find all experiment directories
    exp_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")]

    if not exp_dirs:
        return None

    # Sort by modification time (newest first)
    exp_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return exp_dirs[0]


def wait_for_validation_complete(exp_dir: Path, timeout: int = 7200) -> bool:
    """
    Wait for validation to complete by checking for results file.

    Parameters:
    -----------
    exp_dir : Path
        Experiment directory
    timeout : int
        Maximum wait time in seconds (default: 2 hours)

    Returns:
    --------
    success : bool
        True if validation completed, False if timeout
    """
    results_file = exp_dir / "results" / "genrou_validation_results.json"

    print(f"\n{'='*70}")
    print("WAITING FOR VALIDATION TO COMPLETE")
    print(f"{'='*70}")
    print(f"Monitoring: {results_file}")
    print(f"Timeout: {timeout} seconds ({timeout/60:.1f} minutes)")
    print()

    start_time = time.time()
    check_interval = 30  # Check every 30 seconds

    while time.time() - start_time < timeout:
        if results_file.exists():
            # Check if file is still being written (size changes)
            size1 = results_file.stat().st_size
            time.sleep(5)
            size2 = results_file.stat().st_size

            if size1 == size2 and size1 > 0:
                print(f"✓ Validation complete! Results file found: {results_file}")
                return True

        elapsed = int(time.time() - start_time)
        print(f"  Waiting... ({elapsed}s elapsed, checking every {check_interval}s)", end="\r")
        time.sleep(check_interval)

    print(f"\n⚠️  Timeout reached. Validation may still be running.")
    print(f"   Check manually: {results_file}")
    return False


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


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Complete GENROU Validation Workflow (Validation + Analysis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete workflow (validation + analysis)
  python scripts/validation/run_complete_genrou_workflow.py \\
      --pinn-model validation/delta_weight_sweep/exp_delta20.0_omega40.0/exp_20260116_014607/pinn/model/best_model_20260116_020404.pth \\
      --test-scenarios data/processed/exp_20260116_014611/test_data_20260116_014614.csv \\
      --output-dir outputs/publication/genrou_validation \\
      --gencls-results outputs/publication/statistical_validation/experiments/exp_20260119_095052/pinn/results/metrics.json

  # Validation only (skip analysis)
  python scripts/validation/run_complete_genrou_workflow.py \\
      --pinn-model validation/delta_weight_sweep/exp_delta20.0_omega40.0/exp_20260116_014607/pinn/model/best_model_20260116_020404.pth \\
      --test-scenarios data/processed/exp_20260116_014611/test_data_20260116_014614.csv \\
      --output-dir outputs/publication/genrou_validation \\
      --skip-analysis
        """,
    )

    parser.add_argument(
        "--pinn-model",
        type=str,
        required=True,
        help="Path to trained PINN model",
    )
    parser.add_argument(
        "--test-scenarios",
        type=str,
        required=True,
        help="Path to test scenarios CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory for experiments",
    )
    parser.add_argument(
        "--gencls-results",
        type=str,
        help="Path to GENCLS evaluation results JSON (optional, for comparison)",
    )
    parser.add_argument(
        "--pinn-model-for-trajectory",
        type=str,
        help="Path to PINN model for trajectory figure (default: same as --pinn-model)",
    )
    parser.add_argument(
        "--genrou-case",
        type=str,
        default="test_cases/SMIB_genrou.json",
        help="Path to GENROU case file (default: test_cases/SMIB_genrou.json)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution (default: 300)",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis steps (validation only)",
    )
    parser.add_argument(
        "--skip-trajectory",
        action="store_true",
        help="Skip trajectory figure generation",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        help="Maximum number of scenarios to validate (default: all)",
    )
    parser.add_argument(
        "--validation-timeout",
        type=int,
        default=7200,
        help="Timeout for validation in seconds (default: 7200 = 2 hours)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GENROU VALIDATION - COMPLETE WORKFLOW")
    print("=" * 70)
    print(f"PINN Model: {args.pinn_model}")
    print(f"Test Scenarios: {args.test_scenarios}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Skip Analysis: {args.skip_analysis}")
    print(f"DPI: {args.dpi}")
    print("=" * 70)

    # Resolve paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run Validation
    print("\n" + "=" * 70)
    print("PHASE 1: VALIDATION")
    print("=" * 70)

    validation_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "validation" / "run_complete_genrou_validation.py"),
        "--pinn-model",
        str(args.pinn_model),
        "--test-scenarios",
        str(args.test_scenarios),
        "--output-dir",
        str(output_dir),
    ]

    if args.max_scenarios:
        validation_cmd.extend(["--max-scenarios", str(args.max_scenarios)])

    if not run_command(validation_cmd, "GENROU Validation", check=False):
        print("\n⚠️  Validation failed or was interrupted.")
        print("   Analysis will be skipped. Check validation output above.")
        return 1

    # Find latest experiment directory
    latest_exp = find_latest_experiment(output_dir)
    if not latest_exp:
        print("\n⚠️  Could not find experiment directory.")
        print("   Validation may have failed. Check output above.")
        return 1

    print(f"\n✓ Found experiment directory: {latest_exp}")

    # Wait for validation to complete
    results_file = latest_exp / "results" / "genrou_validation_results.json"
    if not results_file.exists():
        print("\n⚠️  Results file not found immediately. Waiting for validation to complete...")
        if not wait_for_validation_complete(latest_exp, timeout=args.validation_timeout):
            print("\n⚠️  Validation timeout. Results may be incomplete.")
            print("   You can run analysis manually later with:")
            print(f"   python scripts/validation/run_complete_genrou_analysis.py \\")
            print(f"       --results {results_file} \\")
            print(f"       --output-dir {latest_exp}")
            return 1

    # Step 2: Run Analysis (if not skipped)
    if args.skip_analysis:
        print("\n" + "=" * 70)
        print("ANALYSIS SKIPPED (--skip-analysis)")
        print("=" * 70)
        print(f"\n✓ Validation complete: {latest_exp}")
        print(f"\nTo run analysis later:")
        print(f"  python scripts/validation/run_complete_genrou_analysis.py \\")
        print(f"      --results {results_file} \\")
        print(f"      --output-dir {latest_exp}")
        if args.gencls_results:
            print(f"      --gencls-results {args.gencls_results}")
        return 0

    print("\n" + "=" * 70)
    print("PHASE 2: ANALYSIS")
    print("=" * 70)

    analysis_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "validation" / "run_complete_genrou_analysis.py"),
        "--results",
        str(results_file),
        "--output-dir",
        str(latest_exp),
        "--dpi",
        str(args.dpi),
    ]

    if args.gencls_results:
        analysis_cmd.extend(["--gencls-results", str(args.gencls_results)])

    if not args.skip_trajectory:
        pinn_model_for_traj = args.pinn_model_for_trajectory or args.pinn_model
        genrou_case_path = Path(args.genrou_case)
        if genrou_case_path.exists():
            analysis_cmd.extend(
                [
                    "--pinn-model",
                    str(pinn_model_for_traj),
                    "--genrou-case",
                    str(genrou_case_path),
                ]
            )
        else:
            print(f"⚠️  GENROU case file not found: {genrou_case_path}")
            print("   Trajectory figure will be skipped.")
    else:
        analysis_cmd.append("--skip-trajectory")

    if not run_command(analysis_cmd, "Complete Analysis", check=False):
        print("\n⚠️  Some analysis steps may have failed. Check output above.")
        return 1

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE WORKFLOW FINISHED")
    print("=" * 70)
    print(f"\n✅ Validation: {latest_exp}")
    print(f"✅ Analysis: {latest_exp}")
    print(f"\n📁 Output directories:")
    print(f"  - Results: {latest_exp / 'results'}")
    print(f"  - Analysis: {latest_exp / 'analysis'}")
    print(f"  - Figures: {latest_exp / 'figures'}")
    print(f"  - Tables: {latest_exp / 'tables'}")

    print(f"\n📊 Generated files:")
    print(f"  - Validation results: {results_file}")
    print(f"  - Statistical analysis: {latest_exp / 'analysis' / 'statistical_analysis.json'}")
    print(f"  - Parameter sensitivity: {latest_exp / 'analysis' / 'parameter_sensitivity.json'}")
    print(f"  - Error analysis figure: {latest_exp / 'figures' / 'error_analysis.png'}")
    print(
        f"  - Parameter sensitivity figure: {latest_exp / 'figures' / 'parameter_sensitivity.png'}"
    )
    print(f"  - Metric distributions: {latest_exp / 'figures' / 'metric_distributions.png'}")
    print(f"  - Comparison table: {latest_exp / 'tables' / 'table_comparison.tex'}")
    print(
        f"  - Parameter sensitivity table: {latest_exp / 'tables' / 'table_parameter_sensitivity.tex'}"
    )

    print("\n🎉 All done! Publication materials are ready.")

    # Note: Experiment summary is automatically generated by run_complete_genrou_analysis.py
    # If analysis was skipped, generate summary manually
    if args.skip_analysis:
        print("\n" + "=" * 70)
        print("GENERATING EXPERIMENT SUMMARY")
        print("=" * 70)

        try:
            summary_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "validation" / "generate_experiment_summary.py"),
                "--experiment-dir",
                str(latest_exp),
            ]
            if args.gencls_results:
                summary_cmd.extend(["--gencls-results", str(args.gencls_results)])

            if run_command(
                summary_cmd,
                "Generate Experiment Summary",
                check=False,
            ):
                print(f"✓ Experiment summary saved to: {latest_exp / 'EXPERIMENT_SUMMARY.md'}")
        except Exception as e:
            print(f"⚠️  Warning: Could not generate experiment summary: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
