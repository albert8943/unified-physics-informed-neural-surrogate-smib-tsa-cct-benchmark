#!/usr/bin/env python
"""
Environment Check Utility

Check current environment and suggest workflow.

This script helps you determine:
- Whether you're in Colab or local environment
- GPU availability
- Recommended workflow for current setup

Usage:
    python scripts/utils/check_environment.py
"""

import sys
from pathlib import Path

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import google.colab

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def check_environment():
    """Check and report environment status."""
    print("🔍 Environment Check")
    print("=" * 60)
    print()

    # Check location
    if IN_COLAB:
        print("📍 Location: Google Colab ☁️")
        print("   Path: /content/PINN_project_using_andes")
    else:
        project_root = Path(__file__).parent.parent.parent
        print(f"📍 Location: Local Machine 💻")
        print(f"   Path: {project_root}")
    print()

    # Check PyTorch
    if TORCH_AVAILABLE:
        print("✅ PyTorch: Available")
        print(f"   Version: {torch.__version__}")

        # Check CUDA
        if torch.cuda.is_available():
            print(f"   ✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            device = "cuda"
        else:
            print("   ⚠️  GPU Not Available (using CPU)")
            device = "cpu"
    else:
        print("❌ PyTorch: Not Available")
        device = "unknown"
    print()

    # Recommendations
    print("💡 Recommendations:")
    print("-" * 60)

    if IN_COLAB:
        if device == "cuda":
            print("✅ Perfect setup for GPU training!")
            print("   → Use for: Large data generation, full training")
            print("   → Device: cuda")
        else:
            print("⚠️  Colab but no GPU - request GPU runtime")
            print("   → Runtime → Change runtime type → GPU")
            print("   → Device: cpu (until GPU enabled)")
    else:
        if device == "cpu":
            print("💻 Local CPU setup")
            print("   → Use for: Development, testing, small experiments")
            print("   → Device: cpu")
            print()
            print("   💡 For GPU training:")
            print("      → Use Google Colab")
            print("      → Sync code: git push, then git pull in Colab")
        else:
            print("❓ Unknown setup - check PyTorch installation")
    print()

    # Workflow suggestion
    print("🔄 Suggested Workflow:")
    print("-" * 60)
    if IN_COLAB and device == "cuda":
        print("1. Pull latest code: !git pull origin main")
        print("2. Run training: !python training/train_trajectory.py --device cuda")
        print("3. Results saved to Google Drive automatically")
    elif not IN_COLAB:
        print("1. Develop/test code locally")
        print("2. Quick test: python training/train_trajectory.py --device cpu --epochs 10")
        print("3. Sync to Colab: python scripts/utils/sync_for_colab.py")
        print("4. Train on Colab with GPU")
    print()

    return {
        "in_colab": IN_COLAB,
        "device": device,
        "torch_available": TORCH_AVAILABLE,
    }


if __name__ == "__main__":
    check_environment()
