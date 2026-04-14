#!/usr/bin/env python
"""
Three-bar chart for Std NN* vs primary plain-MLP PINN* vs optional residual PINN variant
(test mean per-scenario δ RMSE).

Reads headline means and (when present) per-scenario sample std devs from:
  - primary JSON: frozen ML* vs plain-MLP PINN* (both means + stds);
  - variant JSON: same ML* row optional; must include PINN residual headline pinn mean/std.

Example (repo root)::

    python scripts/plot_pinn_residual_ablation_bars.py ^
      --output "paper_writing/IEEE Access Template/figures/pinn_residual_ablation/pinn_residual_ablation_delta_rmse.png"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_W_IN = 3.46
FIG_DPI = 300


def _load_headline(
    primary_path: Path, variant_path: Path
) -> Tuple[float, float, float, float, float, float]:
    """Returns (ml_mean, pinn_plain_mean, pinn_res_mean, ml_std, pinn_plain_std, pinn_res_std)."""
    with open(primary_path, encoding="utf-8") as f:
        pri: Dict[str, Any] = json.load(f)
    with open(variant_path, encoding="utf-8") as f:
        var: Dict[str, Any] = json.load(f)
    hp = pri["headline_delta_rmse_mean_per_scenario_rad"]
    hv = var["headline_delta_rmse_mean_per_scenario_rad"]
    ml = float(hp["ml_baseline"])
    pinn_plain = float(hp["pinn"])
    pinn_res = float(hv["pinn"])
    sp = pri.get("headline_delta_rmse_std_per_scenario_rad") or {}
    sv = var.get("headline_delta_rmse_std_per_scenario_rad") or {}
    ml_std = float(sp.get("ml_baseline", sv.get("ml_baseline", 0.083)))
    pinn_plain_std = float(sp.get("pinn", 0.100))
    pinn_res_std = float(sv.get("pinn", 0.110))
    return ml, pinn_plain, pinn_res, ml_std, pinn_plain_std, pinn_res_std


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-json",
        type=Path,
        default=REPO_ROOT
        / "docs/publication/pre_specs/phase4_test_comparison_results_20260407.json",
        help="Pre-spec JSON: Std NN* vs primary plain-MLP PINN* (test compare).",
    )
    parser.add_argument(
        "--variant-json",
        type=Path,
        default=REPO_ROOT
        / "docs/publication/pre_specs/pinn_residual_backbone_variant_test_20260407.json",
        help="Pre-spec JSON: residual PINN variant headline (pinn mean/std).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "paper_writing/IEEE Access Template/figures/pinn_residual_ablation/pinn_residual_ablation_delta_rmse.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--ml-std",
        type=float,
        default=None,
        help="Override Std NN* sample std (default: from primary JSON if present).",
    )
    parser.add_argument(
        "--pinn-std",
        type=float,
        default=None,
        help="Override both PINN error bars (default: per-model std from JSONs).",
    )
    args = parser.parse_args()

    ml_mean, pinn_plain_mean, pinn_res_mean, ml_s, pp_s, pr_s = _load_headline(
        args.primary_json, args.variant_json
    )
    if args.ml_std is not None:
        ml_s = args.ml_std
    if args.pinn_std is not None:
        pp_s = args.pinn_std
        pr_s = args.pinn_std

    labels = [
        "Std NN*",
        "PINN*\n(plain MLP)",
        "PINN\n(residual)",
    ]
    means = np.array([ml_mean, pinn_plain_mean, pinn_res_mean], dtype=float)
    yerr = np.array([ml_s, pp_s, pr_s], dtype=float)
    x = np.arange(len(labels))
    colors = ["#4e79a7", "#59a14f", "#e15759"]

    fig_h = 2.4
    fig, ax = plt.subplots(figsize=(FIG_W_IN, fig_h), layout="constrained")
    bars = ax.bar(
        x,
        means,
        yerr=yerr,
        capsize=3,
        color=colors,
        edgecolor="0.2",
        linewidth=0.6,
        error_kw={"linewidth": 0.9, "ecolor": "0.25"},
    )
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel(r"Mean per-scenario RMSE$_\delta$ (rad)")
    ax.set_title(r"Test split ($n=26$); frozen Std NN*")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ymax = float(np.max(means + yerr))
    ax.set_ylim(0.0, ymax * 1.12)
    for rect, m in zip(bars, means):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.012,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
