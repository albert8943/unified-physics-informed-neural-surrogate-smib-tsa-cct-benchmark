#!/usr/bin/env python
"""
Generate PINN loss curves from existing training history.

Usage:
    python scripts/generate_pinn_loss_curves.py \
        --history-file outputs/complete_experiments/exp_20251216_200840/pinn/training_history_20251216_203301.json \
        --output-dir outputs/complete_experiments/exp_20251216_200840/pinn/results/figures \
        --config configs/experiments/hyperparameter_tuning.yaml
"""

import argparse
import sys
import io
from pathlib import Path

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Generate PINN loss curves from training history")
    parser.add_argument(
        "--history-file",
        type=str,
        required=True,
        help="Path to training_history_*.json file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for figures (e.g., pinn/results/figures)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file path (optional, for metadata)",
    )

    args = parser.parse_args()

    history_path = Path(args.history_file)
    output_dir = Path(args.output_dir)

    if not history_path.exists():
        print(f"❌ Training history file not found: {history_path}")
        sys.exit(1)

    print("=" * 70)
    print("GENERATING PINN LOSS CURVES")
    print("=" * 70)
    print(f"History file: {history_path}")
    print(f"Output directory: {output_dir}")

    # Load training history
    print("\nLoading training history...")
    with open(history_path, "r") as f:
        training_history = json.load(f)

    train_losses = training_history.get("train_losses", [])
    val_losses = training_history.get("val_losses", [])

    print("Loaded training history:")
    print(f"  - Training losses: {len(train_losses)} epochs")
    print(f"  - Validation losses: {len(val_losses)} epochs")

    if not train_losses:
        print("❌ No training losses found in history file")
        sys.exit(1)

    # Generate loss curves figure
    print("\nGenerating loss curves...")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        epochs = list(range(1, len(train_losses) + 1))

        # Plot training loss
        ax.plot(epochs, train_losses, label="Training Loss", color="#1f77b4", linewidth=2)

        # Plot validation loss if available
        if val_losses and len(val_losses) == len(train_losses):
            ax.plot(
                epochs,
                val_losses,
                label="Validation Loss",
                color="#ff7f0e",
                linewidth=2,
                linestyle="--",
            )
        elif val_losses:
            print(
                f"⚠️ Validation losses length ({len(val_losses)}) doesn't match training losses"
                f"({len(train_losses)})"
            )
            # Try to plot what we have
            val_epochs = list(range(1, len(val_losses) + 1))
            ax.plot(
                val_epochs,
                val_losses,
                label="Validation Loss",
                color="#ff7f0e",
                linewidth=2,
                linestyle="--",
            )

        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Loss", fontweight="bold")
        ax.set_title("Training and Validation Loss", fontweight="bold")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"loss_curves_{timestamp}.png"
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\n[OK] Loss curves saved to: {output_path}")
        print(f"   File size: {output_path.stat().st_size:,} bytes")

    except Exception as e:
        print(f"\n❌ Error generating figures: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
