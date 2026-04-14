#!/usr/bin/env python
"""Update comparison table with all December 8 experiments."""
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


# Find all Dec 8 and Dec 9 experiments
exps_dir = Path("outputs/experiments")
dec8_exps = [d for d in exps_dir.glob("exp_20251208_*") if (d / "experiment_summary.json").exists()]
dec9_exps = [d for d in exps_dir.glob("exp_20251209_*") if (d / "experiment_summary.json").exists()]

print(f"Found {len(dec8_exps)} December 8 experiments")
print(f"Found {len(dec9_exps)} December 9 experiments\n")

all_rows = []
for exp_dir in sorted(dec8_exps + dec9_exps, key=lambda x: (x.stat().st_mtime, x.name)):
    info = extract_exp_data(exp_dir.name)
    if info:
        all_rows.append(info)

# Sort by date, n_samples, epochs
all_rows.sort(key=lambda x: (x["date"], x["n_samples"], x["epochs"]))

# Generate markdown table
print("Generating updated comparison table...\n")

table_lines = []
table_lines.append(
    "| Experiment ID       | Date       | n_samples | Epochs | Architecture    | λ_physics | Lambda Type | ω_weight | RMSE Δ    | RMSE Ω | R² Δ      | R² Ω      |"
)
table_lines.append(
    "| ------------------- | ---------- | --------- | ------ | --------------- | --------- | ----------- | -------- | --------- | ------ | --------- | --------- |"
)

for r in all_rows:
    # Format values
    rmse_d_str = f"{r['rmse_d']:.3f}" if r["rmse_d"] else "N/A"
    rmse_o_str = f"{r['rmse_o']:.3f}" if r["rmse_o"] else "N/A"
    r2_d_str = f"{r['r2_d']:.3f}" if r["r2_d"] else "N/A"
    r2_o_str = f"{r['r2_o']:.3f}" if r["r2_o"] else "N/A"

    # Highlight best values
    if r["exp_id"] == "exp_20251208_111254":
        rmse_d_str = f"**{rmse_d_str}**"
        r2_d_str = f"**{r2_d_str}**"
    if r["r2_o"] and r["r2_o"] >= 0.60:
        r2_o_str = f"**{r2_o_str}**"
    if r["lambda_p"] == 0.5 and r["date"] == "2024-12-08":
        lambda_p_str = f"**{r['lambda_p']:.2f}**"
    else:
        lambda_p_str = f"{r['lambda_p']:.2f}"

    table_lines.append(
        f"| {r['exp_id']} | {r['date']} | {r['n_samples']} | {r['epochs']} | {r['arch']} | "
        f"{lambda_p_str} | {r['lambda_type']} | {r['omega_w']:.0f} | "
        f"{rmse_d_str} | {rmse_o_str} | {r2_d_str} | {r2_o_str} |"
    )

# Print table
for line in table_lines:
    print(line)

# Save to file
with open("updated_comparison_table.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(table_lines))

print(f"\n✅ Updated table saved to updated_comparison_table.txt")
print(f"Total experiments in table: {len(all_rows)}")
