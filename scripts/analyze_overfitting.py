"""
Analyze overfitting in PINN and ML Baseline models from experiment checkpoints.

This script loads training history from checkpoint files and provides
detailed overfitting analysis with recommendations.
"""

import json
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse


def load_training_history(checkpoint_path: Path) -> Optional[Dict]:
    """Load training history from checkpoint file."""
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        history = checkpoint.get("training_history", {})
        if history:
            return history

        # Try loading from separate JSON file
        json_path = (
            checkpoint_path.parent / f"training_history_{checkpoint_path.stem.split('_')[-1]}.json"
        )
        if json_path.exists():
            with open(json_path, "r") as f:
                return json.load(f)

        return None
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None


def analyze_overfitting(train_losses: list, val_losses: list, best_epoch: int) -> Dict:
    """Analyze overfitting from training and validation losses."""
    if not train_losses or not val_losses:
        return {}

    # Calculate metrics
    initial_train = train_losses[0]
    initial_val = val_losses[0]
    final_train = train_losses[-1]
    final_val = val_losses[-1]

    # Loss at best epoch
    if best_epoch < len(train_losses) and best_epoch < len(val_losses):
        train_at_best = train_losses[best_epoch]
        val_at_best = val_losses[best_epoch]
    else:
        train_at_best = final_train
        val_at_best = final_val

    # Overfitting gap
    gap = val_at_best - train_at_best
    gap_percent = (gap / train_at_best * 100) if train_at_best > 0 else 0

    # Loss reduction
    train_reduction = (
        ((initial_train - final_train) / initial_train * 100) if initial_train > 0 else 0
    )
    val_reduction = ((initial_val - val_at_best) / initial_val * 100) if initial_val > 0 else 0

    # Determine severity
    if gap_percent > 50:
        severity = "SEVERE"
    elif gap_percent > 30:
        severity = "MODERATE"
    elif gap_percent > 10:
        severity = "MILD"
    else:
        severity = "NONE"

    # Check if validation loss increases after best epoch
    val_increases = False
    if best_epoch < len(val_losses) - 1:
        val_after_best = val_losses[best_epoch + 1 :]
        if any(v > val_at_best for v in val_after_best):
            val_increases = True
            max_val_after = max(val_after_best)
            increase_percent = (
                ((max_val_after - val_at_best) / val_at_best * 100) if val_at_best > 0 else 0
            )
        else:
            increase_percent = 0
    else:
        increase_percent = 0

    return {
        "initial_train": initial_train,
        "initial_val": initial_val,
        "final_train": final_train,
        "final_val": final_val,
        "train_at_best": train_at_best,
        "val_at_best": val_at_best,
        "best_epoch": best_epoch,
        "gap": gap,
        "gap_percent": gap_percent,
        "severity": severity,
        "train_reduction": train_reduction,
        "val_reduction": val_reduction,
        "val_increases_after_best": val_increases,
        "val_increase_percent": increase_percent,
        "total_epochs": len(train_losses),
    }


def print_analysis(model_name: str, analysis: Dict, config: Optional[Dict] = None):
    """Print formatted overfitting analysis."""
    print("\n" + "=" * 70)
    print(f"OVERFITTING ANALYSIS: {model_name}")
    print("=" * 70)

    if not analysis:
        print("[WARNING] No training history available")
        return

    print(f"\n[Training Metrics]")
    print(f"  Total Epochs: {analysis['total_epochs']}")
    print(f"  Best Epoch: {analysis['best_epoch']}")
    print(f"  Best Validation Loss: {analysis['val_at_best']:.4f}")
    print(f"  Training Loss at Best: {analysis['train_at_best']:.4f}")

    print(f"\n[Loss Progression]")
    print(f"  Initial Training Loss: {analysis['initial_train']:.4f}")
    print(f"  Initial Validation Loss: {analysis['initial_val']:.4f}")
    print(
        f"  Final Training Loss: {analysis['final_train']:.4f} ({analysis['train_reduction']:.1f}% reduction)"
    )
    print(f"  Final Validation Loss: {analysis['final_val']:.4f}")

    print(f"\n[Overfitting Analysis]")
    print(f"  Gap (Val - Train): {analysis['gap']:.4f} ({analysis['gap_percent']:.1f}% higher)")
    print(f"  Severity: {analysis['severity']}")

    if analysis["val_increases_after_best"]:
        print(
            f"  WARNING: Validation loss increased by {analysis['val_increase_percent']:.1f}% after best epoch"
        )

    print(f"\n[Current Configuration]")
    if config:
        dropout = float(config.get("dropout", 0.0))
        weight_decay = float(config.get("weight_decay", 1e-5))
        patience = config.get("early_stopping_patience", None)
        if patience is not None:
            patience = int(patience)

        print(f"  Dropout: {dropout}")
        print(f"  Weight Decay: {weight_decay}")
        print(f"  Early Stopping Patience: {patience}")

        # Check if settings are appropriate
        issues = []
        if dropout == 0.0 and analysis["severity"] in ["MODERATE", "SEVERE"]:
            issues.append("[X] Dropout is 0.0 (should be 0.1-0.3 for regularization)")
        if weight_decay < 1e-4 and analysis["severity"] in ["MODERATE", "SEVERE"]:
            issues.append(f"[X] Weight decay is too low ({weight_decay}, should be 1e-4 to 1e-3)")
        if patience and patience > 100 and "pinn" in model_name.lower():
            issues.append(f"[!] Early stopping patience is high ({patience}, recommended: 50-100)")
        if patience and patience > 20 and "ml" in model_name.lower():
            issues.append(
                f"[X] Early stopping patience is too high ({patience}, should be 10-20 for ML baselines)"
            )

        if issues:
            print(f"\n  Issues Found:")
            for issue in issues:
                print(f"    {issue}")
        else:
            print(f"  [OK] Configuration looks reasonable")

    print(f"\n[Recommendations]")
    if analysis["severity"] == "SEVERE":
        print("  [CRITICAL] Severe overfitting detected!")
        print("    1. Add dropout: 0.2-0.3")
        print("    2. Increase weight decay: 1e-4 to 1e-3")
        print("    3. Reduce early stopping patience: 10-20 epochs")
        print("    4. Consider reducing model capacity (smaller hidden dimensions)")
    elif analysis["severity"] == "MODERATE":
        print("  [WARNING] Moderate overfitting detected")
        print("    1. Add dropout: 0.1-0.2")
        print("    2. Increase weight decay: 1e-4")
        print("    3. Reduce early stopping patience")
    elif analysis["severity"] == "MILD":
        print("  [INFO] Mild overfitting (acceptable)")
        print("    1. Consider adding dropout: 0.1")
        print("    2. Monitor validation loss closely")
    else:
        print("  [OK] No significant overfitting detected")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze overfitting in experiment models")
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to experiment directory (e.g., outputs/complete_experiments/exp_YYYYMMDD_HHMMSS)",
    )
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return

    # Load config
    config_path = experiment_dir / "config.yaml"
    config = None
    if config_path.exists():
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    model_config = config.get("model", {}) if config else {}
    training_config = config.get("training", {}) if config else {}

    # Analyze PINN
    pinn_checkpoint = experiment_dir / "pinn" / "best_model_*.pth"
    pinn_checkpoints = list(experiment_dir.glob("pinn/best_model_*.pth"))
    if pinn_checkpoints:
        pinn_history = load_training_history(pinn_checkpoints[0])
        if pinn_history:
            train_losses = pinn_history.get("train_losses", [])
            val_losses = pinn_history.get("val_losses", [])
            if train_losses and val_losses:
                best_epoch = val_losses.index(min(val_losses)) if val_losses else 0
                analysis = analyze_overfitting(train_losses, val_losses, best_epoch)
                print_analysis("PINN Model", analysis, {**model_config, **training_config})

    # Analyze ML Baseline
    ml_checkpoints = list(experiment_dir.glob("ml_baseline/*/best_model_*.pth"))
    if not ml_checkpoints:
        ml_checkpoints = list(experiment_dir.glob("ml_baseline/*/model.pth"))

    for ml_checkpoint in ml_checkpoints:
        model_type = ml_checkpoint.parent.parent.name
        ml_history = load_training_history(ml_checkpoint)
        if ml_history:
            train_losses = ml_history.get("train_losses", [])
            val_losses = ml_history.get("val_losses", [])
            if train_losses and val_losses:
                best_epoch = val_losses.index(min(val_losses)) if val_losses else 0
                analysis = analyze_overfitting(train_losses, val_losses, best_epoch)
                print_analysis(
                    f"ML Baseline ({model_type})", analysis, {**model_config, **training_config}
                )


if __name__ == "__main__":
    main()
