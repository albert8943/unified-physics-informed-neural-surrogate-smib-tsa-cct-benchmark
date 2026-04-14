#!/usr/bin/env python
"""Extract all December 8 experiments and their metrics."""
import json
from pathlib import Path
from datetime import datetime


def extract_exp_data(exp_id):
    """Extract experiment data."""
    exp_dir = Path(f"outputs/experiments/{exp_id}")
    summary = exp_dir / "experiment_summary.json"
    if not summary.exists():
        return None

    try:
        with open(summary, "r") as f:
            data = json.load(f)

        config = data.get("config", {})
        metrics = data.get("metrics", {})
        if "metrics" in metrics and isinstance(metrics["metrics"], dict):
            metrics = metrics["metrics"]

        model = config.get("model", {})
        training = config.get("training", {})
        data_gen = config.get("data", {}).get("generation", {})
        loss = config.get("loss", {})

        hidden_dims = model.get("hidden_dims", [])
        arch = "x".join(map(str, hidden_dims)) if hidden_dims else "unknown"

        scale_to_norm = loss.get("scale_to_norm", [1.0, 100.0])
        omega_weight = scale_to_norm[1] if len(scale_to_norm) > 1 else 100.0

        exp_date = datetime.fromtimestamp(exp_dir.stat().st_mtime).date()

        return {
            "exp_id": exp_id,
            "date": exp_date.strftime("%Y-%m-%d"),
            "n_samples": data_gen.get("n_samples", 0),
            "epochs": training.get("epochs", 0),
            "arch": arch,
            "lambda_p": loss.get("lambda_physics", 0),
            "lambda_type": "Fixed" if loss.get("use_fixed_lambda", True) else "Adaptive",
            "omega_w": omega_weight,
            "rmse_d": metrics.get("rmse_delta"),
            "rmse_o": metrics.get("rmse_omega"),
            "r2_d": metrics.get("r2_delta"),
            "r2_o": metrics.get("r2_omega"),
        }
    except Exception as e:
        print(f"Error processing {exp_id}: {e}")
        return None


# Find all Dec 8 experiments
exps_dir = Path("outputs/experiments")
dec8_exps = [d for d in exps_dir.glob("exp_20251208_*") if (d / "experiment_summary.json").exists()]

print(f"Found {len(dec8_exps)} December 8 experiments\n")

rows = []
for exp_dir in sorted(dec8_exps, key=lambda x: x.stat().st_mtime):
    info = extract_exp_data(exp_dir.name)
    if info:
        rows.append(info)

# Sort by n_samples, epochs
rows.sort(key=lambda x: (x["n_samples"], x["epochs"]))

# Print table
print("=" * 140)
print("ALL DECEMBER 8 EXPERIMENTS")
print("=" * 140)
print(
    f"{'Exp ID':<20} {'Date':<12} {'n_samp':<8} {'Epochs':<8} {'Architecture':<20} {'λ_p':<8}"
    f"{'Type':<10} {'ω_w':<8} {'RMSE Δ':<10} {'RMSE Ω':<10} {'R² Δ':<10} {'R² Ω':<10}"
)
print("-" * 140)

for r in rows:
    print(
        f"{r['exp_id']:<20} {r['date']:<12} {r['n_samples']:<8} {r['epochs']:<8} {r['arch']:<20} "
        f"{r['lambda_p']:<8.2f} {r['lambda_type']:<10} {r['omega_w']:<8.0f} "
        f"{r['rmse_d']:<10.4f if r['rmse_d'] else 'N/A':<10} "
        f"{r['rmse_o']:<10.4f if r['rmse_o'] else 'N/A':<10} "
        f"{r['r2_d']:<10.4f if r['r2_d'] else 'N/A':<10} "
        f"{r['r2_o']:<10.4f if r['r2_o'] else 'N/A':<10}"
    )

print("=" * 140)

# Save to file for easy copy-paste
with open("dec8_experiments_list.txt", "w") as f:
    f.write("All December 8 Experiments:\n")
    f.write("=" * 140 + "\n")
    for r in rows:
        f.write(
            f"{r['exp_id']}: n_samples={r['n_samples']}, epochs={r['epochs']}, "
            f"arch={r['arch']}, lambda_p={r['lambda_p']:.2f}, "
            f"RMSE_d={r['rmse_d']:.4f if r['rmse_d'] else 'N/A'}, "
            f"R2_d={r['r2_d']:.4f if r['r2_d'] else 'N/A'}, "
            f"R2_o={r['r2_o']:.4f if r['r2_o'] else 'N/A'}\n"
        )

print(f"\n✅ Saved to dec8_experiments_list.txt")
