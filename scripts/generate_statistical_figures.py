#!/usr/bin/env python
"""
Generate Statistical Figures from Validation Results.

Regenerates or generates statistical plots (box plots, confidence intervals, distributions)
from statistical validation results.

Usage:
    python scripts/generate_statistical_figures.py \
        --results-dir outputs/publication/statistical_validation \
        --output-dir outputs/publication/paper_figures
"""

import argparse
import json
import sys
from pathlib import Path

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualization.statistical_plots import generate_statistical_plots
from scripts.core.utils import load_json


def load_results_from_directory(results_dir: Path) -> list:
    """
    Load results from statistical validation directory.

    Tries to load from raw_results.json first, then falls back to
    loading from individual experiment directories.
    """
    results = []

    # Try loading from raw_results.json first
    raw_results_json = results_dir / "raw_results.json"
    if raw_results_json.exists():
        print(f"Loading results from: {raw_results_json}")
        results = load_json(raw_results_json)
        print(f"✓ Loaded {len(results)} experiments from raw_results.json")
        return results

    # Fall back to loading from individual experiment directories
    experiments_dir = results_dir / "experiments"
    if not experiments_dir.exists():
        print(f"❌ Error: Experiments directory not found: {experiments_dir}")
        sys.exit(1)

    print(f"Loading results from individual experiment directories...")
    exp_dirs = sorted(experiments_dir.glob("exp_*"))

    if not exp_dirs:
        print(f"❌ Error: No experiment directories found in {experiments_dir}")
        sys.exit(1)

    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue

        result = None

        # Try to load experiment_summary.json first (most complete)
        summary_file = exp_dir / "experiment_summary.json"
        if summary_file.exists():
            summary = load_json(summary_file)
            # Convert to format expected by statistical plots
            result = {
                "experiment_id": exp_dir.name,
                "seed": summary.get(
                    "seed", summary.get("reproducibility", {}).get("random_seed", None)
                ),
                "metrics": summary.get("pinn", {}).get("evaluation", {}).get("metrics", {})
                or summary.get("metrics", {}),
                "pinn": {
                    "metrics": summary.get("pinn", {}).get("evaluation", {}).get("metrics", {})
                    or summary.get("metrics", {}),
                    "training_history": summary.get("pinn", {})
                    .get("training", {})
                    .get("history", {}),
                    "model_path": summary.get("pinn", {}).get("model_path", ""),
                },
                "experiment_dir": str(exp_dir),
            }

            # Add ML baseline if available
            if "ml_baseline" in summary:
                result["ml_baseline"] = summary["ml_baseline"]
        else:
            # Fall back to loading from metrics JSON files and config
            metrics_files = sorted(exp_dir.glob("metrics_*.json"))
            config_file = exp_dir / "config.yaml"

            if metrics_files:
                # Use the latest metrics file
                metrics_file = metrics_files[-1]
                metrics = load_json(metrics_file)

                # Load seed from config
                seed = None
                if config_file.exists():
                    import yaml

                    with open(config_file, "r") as f:
                        config = yaml.safe_load(f)
                        seed = config.get("seed")

                result = {
                    "experiment_id": exp_dir.name,
                    "seed": seed,
                    "metrics": metrics,
                    "pinn": {
                        "metrics": metrics,
                        "training_history": {},
                        "model_path": "",
                    },
                    "experiment_dir": str(exp_dir),
                }

        if result:
            results.append(result)
        else:
            print(f"⚠️  Warning: Could not load results from {exp_dir.name}, skipping")

    print(f"✓ Loaded {len(results)} experiments from individual directories")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate statistical figures from validation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/publication/statistical_validation",
        help="Directory containing statistical validation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: results-dir/figures or outputs/publication/paper_figures)",
    )
    parser.add_argument(
        "--copy-to-paper-figures",
        action="store_true",
        help="Also copy figures to outputs/publication/paper_figures/",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir

    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        sys.exit(1)

    print("=" * 70)
    print("GENERATING STATISTICAL FIGURES")
    print("=" * 70)
    print(f"Results directory: {results_dir}")

    # Load results
    results = load_results_from_directory(results_dir)

    if not results:
        print("❌ Error: No results found to generate figures from")
        sys.exit(1)

    print(f"✓ Loaded {len(results)} experiments")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        # Default: use figures directory in results_dir, or paper_figures
        output_dir = results_dir / "figures"
        if not output_dir.exists():
            output_dir = PROJECT_ROOT / "outputs" / "publication" / "paper_figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate figures
    print("\nGenerating statistical plots...")
    generate_statistical_plots(results, output_dir)

    print(f"\n✅ Statistical figures generated successfully!")
    print(f"   Output directory: {output_dir}")
    print(f"   Files generated:")
    print(f"     - box_plots.png")
    print(f"     - confidence_intervals.png")
    print(f"     - distributions.png")

    # Copy to paper_figures if requested (and output_dir is different)
    paper_figures_dir = PROJECT_ROOT / "outputs" / "publication" / "paper_figures"
    if args.copy_to_paper_figures and output_dir != paper_figures_dir:
        paper_figures_dir.mkdir(parents=True, exist_ok=True)

        import shutil

        for fig_file in ["box_plots.png", "confidence_intervals.png", "distributions.png"]:
            src = output_dir / fig_file
            dst = paper_figures_dir / fig_file
            if src.exists() and src != dst:
                shutil.copy2(src, dst)
                print(f"   ✓ Copied {fig_file} to paper_figures/")

    print("\n" + "=" * 70)
    print("STATISTICAL FIGURES GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
