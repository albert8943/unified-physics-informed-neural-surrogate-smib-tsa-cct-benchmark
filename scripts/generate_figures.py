#!/usr/bin/env python
"""Quick script to generate analysis figures for experiment data."""

import sys
from pathlib import Path

# Set matplotlib backend before importing (important for headless environments)
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.analyze_data import (
    load_data,
    generate_parameter_space_figures,
    generate_trajectory_figures,
    generate_cct_figures,
)


def main():
    # Paths
    data_path = Path(
        "outputs/experiments/exp_20251207_195343/data/parameter_sweep_data_20251207_200743.csv"
    )
    output_dir = Path("outputs/experiments/exp_20251207_195343/analysis")
    figures_dir = output_dir / "figures"

    print("=" * 70)
    print("GENERATING ANALYSIS FIGURES")
    print("=" * 70)
    print(f"Data file: {data_path}")
    print(f"Output directory: {figures_dir}")
    print()

    # Check if data file exists
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return

    # Create figures directory
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created figures directory: {figures_dir}")
    print()

    # Load data
    print("Loading data...")
    try:
        df = load_data(data_path)
        print(f"✓ Loaded {len(df):,} rows")
        print()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback

        traceback.print_exc()
        return

    # Generate figures
    print("Generating figures...")
    print("-" * 70)

    try:
        print("1. Parameter space figures...")
        param_figures = generate_parameter_space_figures(df, figures_dir, ["png"])
        print(f"   ✓ Generated {len(param_figures)} figures")

        print("2. Trajectory figures...")
        traj_figures = generate_trajectory_figures(df, figures_dir, ["png"])
        print(f"   ✓ Generated {len(traj_figures)} figures")

        print("3. CCT analysis figures...")
        cct_figures = generate_cct_figures(df, figures_dir, ["png"])
        print(f"   ✓ Generated {len(cct_figures)} figures")

        print()
        print("=" * 70)
        print("✅ FIGURES GENERATED SUCCESSFULLY")
        print("=" * 70)
        print(f"Total figures: {len(param_figures) + len(traj_figures) + len(cct_figures)}")
        print(f"Location: {figures_dir}")
        print()
        print("Generated files:")
        all_figures = list(figures_dir.glob("*.png"))
        for fig in sorted(all_figures):
            size_mb = fig.stat().st_size / (1024 * 1024)
            print(f"  - {fig.name} ({size_mb:.2f} MB)")

    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
