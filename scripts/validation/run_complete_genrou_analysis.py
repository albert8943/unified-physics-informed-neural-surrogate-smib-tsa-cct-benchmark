#!/usr/bin/env python
"""
Complete GENROU Validation Analysis Workflow.

Runs all analysis steps after validation:
1. Statistical analysis
2. Parameter sensitivity analysis
3. Visualizations (error analysis, parameter sensitivity, distributions)
4. GENCLS comparison
5. LaTeX table generation
6. Trajectory figure (if trajectory data available)

Usage:
    python scripts/validation/run_complete_genrou_analysis.py \
        --results outputs/publication/genrou_validation/exp_YYYYMMDD_HHMMSS/results/genrou_validation_results.json \
        --output-dir outputs/publication/genrou_validation/exp_YYYYMMDD_HHMMSS \
        [--gencls-results PATH] \
        [--pinn-model PATH] \
        [--genrou-case PATH] \
        [--dpi 300]
"""

import argparse
import subprocess
import sys
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
        description="Complete GENROU Validation Analysis Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete analysis (all steps)
  python scripts/validation/run_complete_genrou_analysis.py \\
      --results outputs/publication/genrou_validation/exp_20260120_104644/results/genrou_validation_results.json \\
      --output-dir outputs/publication/genrou_validation/exp_20260120_104644 \\
      --gencls-results outputs/publication/statistical_validation/experiments/exp_20260119_095052/pinn/results/metrics.json

  # Analysis without GENCLS comparison
  python scripts/validation/run_complete_genrou_analysis.py \\
      --results outputs/publication/genrou_validation/exp_20260120_104644/results/genrou_validation_results.json \\
      --output-dir outputs/publication/genrou_validation/exp_20260120_104644
        """,
    )

    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to GENROU validation results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory (experiment folder)",
    )
    parser.add_argument(
        "--gencls-results",
        type=str,
        help="Path to GENCLS evaluation results JSON (optional, for comparison)",
    )
    parser.add_argument(
        "--pinn-model",
        type=str,
        help="Path to PINN model (optional, for trajectory figure)",
    )
    parser.add_argument(
        "--genrou-case",
        type=str,
        help="Path to GENROU case file (optional, for trajectory figure)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution (default: 300)",
    )
    parser.add_argument(
        "--skip-trajectory",
        action="store_true",
        help="Skip trajectory figure generation",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GENROU VALIDATION - COMPLETE ANALYSIS WORKFLOW")
    print("=" * 70)
    print(f"Results: {args.results}")
    print(f"Output Directory: {args.output_dir}")
    print(f"DPI: {args.dpi}")
    print("=" * 70)

    # Resolve paths
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    output_dir = Path(args.output_dir)
    analysis_dir = output_dir / "analysis"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    comparison_dir = output_dir / "results" / "comparison"

    # Create directories
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    total_steps = 5

    # Step 1: Statistical Analysis
    print("\n" + "=" * 70)
    print("PHASE 1: STATISTICAL ANALYSIS")
    print("=" * 70)

    if run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validation" / "analyze_genrou_results.py"),
            "--results",
            str(results_path),
            "--output-dir",
            str(analysis_dir),
        ],
        "Statistical Analysis",
    ):
        success_count += 1

    # Step 2: Parameter Sensitivity Analysis
    print("\n" + "=" * 70)
    print("PHASE 2: PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)

    if run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validation" / "analyze_parameter_sensitivity.py"),
            "--results",
            str(results_path),
            "--output-dir",
            str(analysis_dir),
        ],
        "Parameter Sensitivity Analysis",
    ):
        success_count += 1

    # Step 3: Visualizations
    print("\n" + "=" * 70)
    print("PHASE 3: VISUALIZATIONS")
    print("=" * 70)

    if run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validation" / "visualize_genrou_results.py"),
            "--results",
            str(results_path),
            "--output-dir",
            str(figures_dir),
            "--dpi",
            str(args.dpi),
        ],
        "Generate Visualizations",
    ):
        success_count += 1

    # Step 4: GENCLS Comparison
    print("\n" + "=" * 70)
    print("PHASE 4: GENCLS COMPARISON")
    print("=" * 70)

    if args.gencls_results:
        if run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "validation" / "compare_gencls_genrou.py"),
                "--genrou-results",
                str(results_path),
                "--gencls-results",
                str(args.gencls_results),
                "--output-dir",
                str(comparison_dir),
            ],
            "GENCLS vs GENROU Comparison",
            check=False,  # Don't fail if comparison fails
        ):
            success_count += 1
    else:
        print("⚠️  Skipping GENCLS comparison (--gencls-results not provided)")
        total_steps -= 1

    # Step 5: LaTeX Tables
    print("\n" + "=" * 70)
    print("PHASE 5: PUBLICATION TABLES")
    print("=" * 70)

    table_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "validation" / "create_publication_tables.py"),
        "--genrou-results",
        str(results_path),
        "--output-dir",
        str(tables_dir),
    ]
    if args.gencls_results:
        table_cmd.extend(["--gencls-results", str(args.gencls_results)])

    if run_command(
        table_cmd,
        "Generate LaTeX Tables",
        check=False,  # Don't fail if tables fail
    ):
        success_count += 1

    # Step 6: Trajectory Figure (Optional)
    if not args.skip_trajectory and args.pinn_model and args.genrou_case:
        print("\n" + "=" * 70)
        print("PHASE 6: TRAJECTORY FIGURE (OPTIONAL)")
        print("=" * 70)

        if run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "validation" / "create_trajectory_figure.py"),
                "--results",
                str(results_path),
                "--output-dir",
                str(figures_dir),
                "--pinn-model",
                str(args.pinn_model),
                "--genrou-case",
                str(args.genrou_case),
                "--dpi",
                str(args.dpi),
            ],
            "Create Trajectory Figure",
            check=False,  # Don't fail if trajectory figure fails
        ):
            success_count += 1
    else:
        if not args.skip_trajectory:
            print("\n⚠️  Skipping trajectory figure (requires --pinn-model and --genrou-case)")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\nCompleted: {success_count}/{total_steps} steps")
    print(f"\nOutput directories:")
    print(f"  - Analysis: {analysis_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Tables: {tables_dir}")
    if args.gencls_results:
        print(f"  - Comparison: {comparison_dir}")

    print(f"\nGenerated files:")
    print(f"  - Statistical analysis: {analysis_dir / 'statistical_analysis.json'}")
    print(f"  - Parameter sensitivity: {analysis_dir / 'parameter_sensitivity.json'}")
    print(f"  - Error analysis figure: {figures_dir / 'error_analysis.png'}")
    print(f"  - Parameter sensitivity figure: {figures_dir / 'parameter_sensitivity.png'}")
    print(f"  - Metric distributions: {figures_dir / 'metric_distributions.png'}")
    print(f"  - Comparison table: {tables_dir / 'table_comparison.tex'}")
    print(f"  - Parameter sensitivity table: {tables_dir / 'table_parameter_sensitivity.tex'}")

    if success_count == total_steps:
        print("\n✅ All analysis steps completed successfully!")
    else:
        print(f"\n⚠️  Some steps failed or were skipped. Check output above.")

    # Generate experiment summary
    print("\n" + "=" * 70)
    print("GENERATING EXPERIMENT SUMMARY")
    print("=" * 70)

    try:
        summary_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "validation" / "generate_experiment_summary.py"),
            "--experiment-dir",
            str(output_dir),
        ]
        if args.gencls_results:
            summary_cmd.extend(["--gencls-results", str(args.gencls_results)])

        if run_command(
            summary_cmd,
            "Generate Experiment Summary",
            check=False,  # Don't fail if summary generation fails
        ):
            print(f"✓ Experiment summary saved to: {output_dir / 'EXPERIMENT_SUMMARY.md'}")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate experiment summary: {e}")
        print("   You can generate it manually later with:")
        print(
            f"   python scripts/validation/generate_experiment_summary.py --experiment-dir {output_dir}"
        )


if __name__ == "__main__":
    main()
