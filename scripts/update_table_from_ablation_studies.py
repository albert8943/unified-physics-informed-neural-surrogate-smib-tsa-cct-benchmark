#!/usr/bin/env python
"""Update comparison table using ablation study CSV files as source of truth."""
import csv
import json
from pathlib import Path

# Read lambda ablation CSV
lambda_csv = Path("outputs/ablation_studies/lambda_ablation_comparison_20251208_171717.csv")
omega_csv = Path("outputs/ablation_studies/omega_weight_ablation_comparison_20251208_204120.csv")

# Store data from CSVs
lambda_data = {}
omega_data = {}

# Read lambda ablation data
with open(lambda_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        exp_id = row["experiment_id"]
        if exp_id.startswith("exp_20251208"):
            lambda_data[exp_id] = {
                "lambda_type": row["lambda_type"],
                "lambda_physics": row["lambda_physics"],
                "n_samples": float(row["n_samples"]),
                "epochs": int(row["epochs"]),
                "rmse_delta": float(row["rmse_delta"]),
                "rmse_omega": float(row["rmse_omega"]),
                "r2_delta": float(row["r2_delta"]),
                "r2_omega": float(row["r2_omega"]),
            }

# Read omega weight ablation data
with open(omega_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        exp_id = row["experiment_id"]
        if exp_id.startswith("exp_20251208"):
            omega_data[exp_id] = {
                "omega_weight": float(row["omega_weight"]),
            }

# Get architecture and omega_weight from experiment summaries
for exp_id in list(lambda_data.keys()):
    exp_dir = Path(f"outputs/experiments/{exp_id}")
    summary = exp_dir / "experiment_summary.json"
    if summary.exists():
        with open(summary, "r") as f:
            data = json.load(f)
            config = data.get("config", {})
            model = config.get("model", {})
            loss = config.get("loss", {})

            hidden_dims = model.get("hidden_dims", [])
            arch = "x".join(map(str, hidden_dims)) if hidden_dims else "unknown"
            lambda_data[exp_id]["arch"] = arch

            scale_to_norm = loss.get("scale_to_norm", [1.0, 100.0])
            omega_weight = scale_to_norm[1] if len(scale_to_norm) > 1 else 100.0
            lambda_data[exp_id]["omega_weight"] = omega_weight

# Print findings
print("=" * 100)
print("LAMBDA ABLATION EXPERIMENTS FROM CSV")
print("=" * 100)
print(
    f"{'Exp ID':<20} {'Type':<15} {'λ_p':<8} {'n_samp':<8} {'Epochs':<8} {'RMSE Δ':<10} {'R² Δ':<10} {'R² Ω':<10}"
)
print("-" * 100)

adaptive_exps = []
fixed_exps = []

for exp_id, data in sorted(lambda_data.items()):
    lambda_type = data["lambda_type"]
    if "Adaptive" in lambda_type:
        adaptive_exps.append(exp_id)
    else:
        fixed_exps.append(exp_id)

    print(
        f"{exp_id:<20} {lambda_type:<15} {data['lambda_physics']:<8} "
        f"{int(data['n_samples']):<8} {data['epochs']:<8} "
        f"{data['rmse_delta']:<10.3f} {data['r2_delta']:<10.3f} {data['r2_omega']:<10.3f}"
    )

print("\n" + "=" * 100)
print(f"Total: {len(lambda_data)} experiments")
print(f"Adaptive: {len(adaptive_exps)}")
print(f"Fixed: {len(fixed_exps)}")
print("\nAdaptive experiments:")
for exp in adaptive_exps:
    print(f"  - {exp}")
print("\nFixed experiments:")
for exp in fixed_exps:
    print(f"  - {exp}")
