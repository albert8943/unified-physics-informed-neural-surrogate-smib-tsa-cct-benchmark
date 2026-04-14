#!/usr/bin/env python
"""Create comprehensive comparison table."""
import json
from pathlib import Path
from datetime import datetime


def extract_info(exp_dir):
    """Extract experiment information."""
    summary = exp_dir / "experiment_summary.json"
    if not summary.exists():
        return None

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
        "exp_id": exp_dir.name,
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


# Find experiments
exps_dir = Path("outputs/experiments")
all_exps = [d for d in exps_dir.glob("exp_*") if (d / "experiment_summary.json").exists()]

# Extract data
rows = []
for exp_dir in all_exps:
    info = extract_info(exp_dir)
    if info:
        rows.append(info)

# Filter for Dec 8-9
yesterday = datetime(2024, 12, 8).date()
today = datetime(2024, 12, 9).date()
filtered = [r for r in rows if r["date"] in ["2024-12-08", "2024-12-09"]]

# Sort by date, n_samples, epochs
filtered.sort(key=lambda x: (x["date"], x["n_samples"], x["epochs"]))

# Print table
print("\n" + "=" * 140)
print("COMPREHENSIVE EXPERIMENT COMPARISON TABLE")
print("=" * 140)
print(
    f"{'Exp ID':<20} {'Date':<12} {'n_samp':<8} {'Epochs':<8} {'Architecture':<20} {'λ_p':<8}"
    f"{'Type':<10} {'ω_w':<8} {'RMSE Δ':<10} {'RMSE Ω':<10} {'R² Δ':<10} {'R² Ω':<10}"
)
print("-" * 140)

for r in filtered:
    print(
        f"{r['exp_id']:<20} {r['date']:<12} {r['n_samples']:<8} {r['epochs']:<8} {r['arch']:<20} "
        f"{r['lambda_p']:<8.2f} {r['lambda_type']:<10} {r['omega_w']:<8.0f} "
        f"{r['rmse_d']:<10.4f if r['rmse_d'] else 'N/A':<10} "
        f"{r['rmse_o']:<10.4f if r['rmse_o'] else 'N/A':<10} "
        f"{r['r2_d']:<10.4f if r['r2_d'] else 'N/A':<10} "
        f"{r['r2_o']:<10.4f if r['r2_o'] else 'N/A':<10}"
    )

print("=" * 140)

# Summary by n_samples
print("\nSUMMARY BY n_samples:")
print("=" * 140)
for n in sorted(set(r["n_samples"] for r in filtered)):
    subset = [r for r in filtered if r["n_samples"] == n]
    print(f"\nn_samples = {n} ({len(subset)} experiments):")
    if subset:
        best_rmse = min(subset, key=lambda x: x["rmse_d"] if x["rmse_d"] else float("inf"))
        best_r2o = max(subset, key=lambda x: x["r2_o"] if x["r2_o"] else 0)
        print(
            f"Best RMSE Delta: {best_rmse['rmse_d']:.4f} (epochs={best_rmse['epochs']},"
            f"{best_rmse['exp_id']})"
        )
        print(
            f"Best R² Omega: {best_r2o['r2_o']:.4f} (epochs={best_r2o['epochs']},"
            f"{best_r2o['exp_id']})"
        )
