import json
from pathlib import Path
from datetime import datetime


def get_exp_data(exp_id):
    """Get experiment data."""
    exp_dir = Path(f"outputs/experiments/{exp_id}")
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


# Key experiments from yesterday and today
exps = [
    "exp_20251208_104847",  # n_samples=20, epochs=300
    "exp_20251208_110512",  # n_samples=20, epochs=200
    "exp_20251208_110020",  # n_samples=20, epochs=100
    "exp_20251208_111254",  # n_samples=30, epochs=300
    "exp_20251208_112817",  # n_samples=30, epochs=200
    "exp_20251208_112345",  # n_samples=30, epochs=100
    "exp_20251208_113622",  # n_samples=40, epochs=300
    "exp_20251208_104035",  # n_samples=40, epochs=200
    "exp_20251208_103623",  # n_samples=40, epochs=100
    "exp_20251209_002148",  # n_samples=50, epochs=500
    "exp_20251209_015130",  # n_samples=50, epochs=600
    "exp_20251209_032332",  # n_samples=50, epochs=700
    "exp_20251209_045842",  # n_samples=50, epochs=800
    "exp_20251209_063757",  # n_samples=50, epochs=900
    "exp_20251209_082231",  # n_samples=50, epochs=1000
]

rows = []
for exp_id in exps:
    data = get_exp_data(exp_id)
    if data:
        rows.append(data)

# Sort
rows.sort(key=lambda x: (x["date"], x["n_samples"], x["epochs"]))

# Write to file
with open("COMPREHENSIVE_COMPARISON_TABLE.md", "w", encoding="utf-8") as f:
    f.write("# Comprehensive Experiment Comparison Table\n\n")
    f.write("## Overview\n\n")
    f.write(
        "This table compares experiments from **December 8, 2024** (yesterday) and **December 9, 2024** (today).\n\n"
    )
    f.write("## Comparison Table\n\n")
    f.write(
        "| Experiment ID | Date | n_samples | Epochs | Architecture | λ_physics | Lambda Type | ω_weight | RMSE Δ | RMSE Ω | R² Δ | R² Ω |\n"
    )
    f.write(
        "|--------------|------|-----------|--------|--------------|-----------|-------------|----------|--------|--------|------|------|\n"
    )

    for r in rows:
        f.write(
            f"| {r['exp_id']} | {r['date']} | {r['n_samples']} | {r['epochs']} | {r['arch']} | "
            f"{r['lambda_p']:.2f} | {r['lambda_type']} | {r['omega_w']:.0f} | "
            f"{r['rmse_d']:.4f if r['rmse_d'] else 'N/A'} | "
            f"{r['rmse_o']:.4f if r['rmse_o'] else 'N/A'} | "
            f"{r['r2_d']:.4f if r['r2_d'] else 'N/A'} | "
            f"{r['r2_o']:.4f if r['r2_o'] else 'N/A'} |\n"
        )

    f.write("\n## Summary by n_samples\n\n")
    for n in sorted(set(r["n_samples"] for r in rows)):
        subset = [r for r in rows if r["n_samples"] == n]
        f.write(f"### n_samples = {n} ({len(subset)} experiments)\n\n")
        if subset:
            best_rmse = min(subset, key=lambda x: x["rmse_d"] if x["rmse_d"] else float("inf"))
            best_r2o = max(subset, key=lambda x: x["r2_o"] if x["r2_o"] else 0)
            f.write(
                f"- **Best RMSE Delta**: {best_rmse['rmse_d']:.4f} (epochs={best_rmse['epochs']}, {best_rmse['exp_id']})\n"
            )
            f.write(
                f"- **Best R² Omega**: {best_r2o['r2_o']:.4f} (epochs={best_r2o['epochs']}, {best_r2o['exp_id']})\n\n"
            )

print("Table generated in COMPREHENSIVE_COMPARISON_TABLE.md")
