#!/usr/bin/env python
"""Debug script to test figure saving."""

import sys
from pathlib import Path

# Set matplotlib backend FIRST
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_figure_saving():
    figures_dir = Path("outputs/experiments/exp_20251207_195343/analysis/figures")

    print("=" * 70)
    print("DEBUG: FIGURE SAVING TEST")
    print("=" * 70)
    print(f"Target directory: {figures_dir}")
    print(f"Absolute path: {figures_dir.resolve()}")
    print()

    # Create directory
    try:
        figures_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory created/exists: {figures_dir.exists()}")
    except Exception as e:
        print(f"❌ Error creating directory: {e}")
        return

    # Create a simple test figure
    print("\nCreating test figure...")
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, "b-", linewidth=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Test Figure")
    plt.tight_layout()

    # Try to save
    test_file = figures_dir / "test_figure.png"
    print(f"\nAttempting to save to: {test_file}")
    print(f"Absolute path: {test_file.resolve()}")

    try:
        plt.savefig(test_file, dpi=300, bbox_inches="tight", format="png")
        print("✓ plt.savefig() completed without error")
        plt.close()

        # Check if file exists
        if test_file.exists():
            size = test_file.stat().st_size
            print(f"✅ SUCCESS! File exists: {test_file.name}")
            print(f"   Size: {size:,} bytes ({size/1024:.1f} KB)")
        else:
            print(f"❌ FAILED! File does not exist after savefig()")
            print(f"   Directory contents: {list(figures_dir.iterdir())}")
    except Exception as e:
        print(f"❌ ERROR during savefig(): {e}")
        import traceback

        traceback.print_exc()
        plt.close()


if __name__ == "__main__":
    test_figure_saving()
