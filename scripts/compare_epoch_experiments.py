#!/usr/bin/env python
"""Quick script to compare epoch experiments."""
import json
from pathlib import Path

exps = [
    ("exp_20251208_175106", 400),
    ("exp_20251208_180305", 800),
    ("exp_20251208_181500", 1200),
    ("exp_20251208_182657", 1600),
    ("exp_20251208_183850", 2000),
]

print("=" * 80)
print("EPOCH COMPARISON RESULTS (n_samples=30, fixed lambda=0.5)")
print("=" * 80)
print(f"{'Epochs':<8} {'RMSE Delta':<12} {'R² Delta':<10} {'RMSE Omega':<12} {'R² Omega':<10}")
print("-" * 80)

results = []
for exp_id, epochs in exps:
    summary_path = Path(f"outputs/experiments/{exp_id}/experiment_summary.json")
    if not summary_path.exists():
        print(f"⚠️  {exp_id} not found", flush=True)
        continue

    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
        # Metrics are under "metrics" key at top level
        metrics = data.get("metrics", {})
    except Exception as e:
        print(f"⚠️  Error loading {exp_id}: {e}", flush=True)
        continue

    rmse_delta = metrics.get("rmse_delta", 0)
    r2_delta = metrics.get("r2_delta", 0)
    rmse_omega = metrics.get("rmse_omega", 0)
    r2_omega = metrics.get("r2_omega", 0)

    results.append(
        {
            "epochs": epochs,
            "rmse_delta": rmse_delta,
            "r2_delta": r2_delta,
            "rmse_omega": rmse_omega,
            "r2_omega": r2_omega,
        }
    )

    print(
        f"{epochs:<8} {rmse_delta:<12.4f} {r2_delta:<10.4f} {rmse_omega:<12.4f} {r2_omega:<10.4f}"
    )

print("=" * 80)

# Find best
if results:
    best_rmse = min(results, key=lambda x: x["rmse_delta"])
    best_r2 = max(results, key=lambda x: x["r2_delta"])

    print(f"\n📊 Best RMSE Delta: {best_rmse['epochs']} epochs (RMSE={best_rmse['rmse_delta']:.4f})")
    print(f"📊 Best R² Delta: {best_r2['epochs']} epochs (R²={best_r2['r2_delta']:.4f})")

    # Improvement analysis
    baseline = results[0]  # 400 epochs
    print("\n📈 Improvement from 400 to 2000 epochs:")

    # Calculate RMSE Delta improvement with zero-division check
    if baseline["rmse_delta"] > 1e-10:  # Avoid division by zero
        rmse_diff = baseline["rmse_delta"] - results[-1]["rmse_delta"]
        rmse_improvement = (rmse_diff / baseline["rmse_delta"]) * 100
        rmse_improvement_str = f"{rmse_improvement:.1f}% improvement"
    else:
        rmse_improvement_str = "N/A (baseline RMSE too small)"

    baseline_rmse = baseline["rmse_delta"]
    final_rmse = results[-1]["rmse_delta"]
    print(f"   RMSE Delta: {baseline_rmse:.4f} → {final_rmse:.4f} " f"({rmse_improvement_str})")

    # Calculate R² Delta improvement with zero-division check
    # Use abs for R² which can be negative
    if abs(baseline["r2_delta"]) > 1e-10:
        r2_diff = results[-1]["r2_delta"] - baseline["r2_delta"]
        r2_improvement = (r2_diff / abs(baseline["r2_delta"])) * 100
        r2_improvement_str = f"{r2_improvement:.1f}% improvement"
    else:
        r2_improvement_str = "N/A (baseline R² too small)"

    baseline_r2 = baseline["r2_delta"]
    final_r2 = results[-1]["r2_delta"]
    print(f"   R² Delta: {baseline_r2:.4f} → {final_r2:.4f} " f"({r2_improvement_str})")
