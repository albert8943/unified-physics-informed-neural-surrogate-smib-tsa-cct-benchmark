#!/usr/bin/env python
"""
Generate publication figures from multimachine evaluation metrics CSV.

Reads the CSV produced by evaluate_multimachine_model.py (with delta_rmse_rad,
is_stable, is_stable_pred, param_tc, param_H, param_D, param_load/param_Pm,
delta_rmse_gen0_rad, ...) and produces:
  1. Error distribution (histogram/box of RMSE in deg, by stable vs unstable)
  2. Error vs parameter (RMSE vs tc, vs load)
  3. Per-generator and overall RMSE/MAE summary (bar chart + table CSV)
  4. Stability agreement (ANDES vs PINN: confusion-style or % correct)

Usage:
  python scripts/plot_multimachine_publication_figures.py --metrics-csv results/multimachine_eval/multimachine_eval_metrics_*.csv --output-dir results/multimachine_eval
  python scripts/plot_multimachine_publication_figures.py --metrics-dir results/multimachine_eval --output-dir results/multimachine_eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL = True
except ImportError:
    _MPL = False


def _rad2deg(x: float) -> float:
    return float(np.rad2deg(x))


def plot_error_distribution(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> None:
    """Histogram and/or box plot of scenario-level RMSE (deg), by stable vs unstable."""
    if "delta_rmse_rad" not in df.columns or df.empty:
        return
    rmse_deg = np.rad2deg(df["delta_rmse_rad"].values)
    has_stability = "is_stable" in df.columns
    fig, axes = plt.subplots(1, 2 if has_stability else 1, figsize=(6 if has_stability else 5, 4))
    if has_stability:
        ax_hist, ax_box = axes
        stable_deg = rmse_deg[df["is_stable"].astype(bool)]
        unstable_deg = rmse_deg[~df["is_stable"].astype(bool)]
        ax_hist.hist(
            stable_deg,
            bins=min(25, max(5, len(stable_deg))),
            alpha=0.7,
            label="Stable",
            color="C0",
            edgecolor="black",
        )
        ax_hist.hist(
            unstable_deg,
            bins=min(25, max(5, len(unstable_deg))),
            alpha=0.7,
            label="Unstable",
            color="C1",
            edgecolor="black",
        )
        ax_hist.set_xlabel(r"RMSE $\delta$ (deg)")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Error distribution")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        bp = ax_box.boxplot(
            [stable_deg[~np.isnan(stable_deg)], unstable_deg[~np.isnan(unstable_deg)]],
            tick_labels=["Stable", "Unstable"],
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("lightgray")
        ax_box.set_ylabel(r"RMSE $\delta$ (deg)")
        ax_box.set_title("Error by stability")
        ax_box.grid(True, alpha=0.3, axis="y")
    else:
        ax_hist = np.atleast_1d(axes).flat[0]
        ax_hist.hist(
            rmse_deg, bins=min(30, max(10, len(rmse_deg))), alpha=0.7, color="C0", edgecolor="black"
        )
        ax_hist.set_xlabel(r"RMSE $\delta$ (deg)")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Error distribution (all scenarios)")
        ax_hist.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_dir / "multimachine_publication_error_distribution.png", dpi=dpi, bbox_inches="tight"
    )
    plt.close()


def plot_error_vs_parameter(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> None:
    """Scatter or box: RMSE (deg) vs tc, vs load (or H, D)."""
    if "delta_rmse_rad" not in df.columns or df.empty:
        return
    rmse_deg = np.rad2deg(df["delta_rmse_rad"].values)
    figs = []
    # vs clearing time
    if "param_tc" in df.columns and df["param_tc"].notna().any():
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        tc = df["param_tc"].values
        valid = ~np.isnan(tc) & ~np.isnan(rmse_deg)
        if valid.sum() > 0:
            ax.scatter(tc[valid], rmse_deg[valid], alpha=0.6, s=25)
            ax.set_xlabel("Clearing time tc (s)")
            ax.set_ylabel(r"RMSE $\delta$ (deg)")
            ax.set_title("Trajectory error vs clearing time")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            output_dir / "multimachine_publication_error_vs_tc.png", dpi=dpi, bbox_inches="tight"
        )
        plt.close()
        figs.append("multimachine_publication_error_vs_tc.png")
    # vs load (param_load or param_Pm)
    load_col = (
        "param_load"
        if "param_load" in df.columns and df["param_load"].notna().any()
        else "param_Pm"
    )
    if load_col in df.columns and df[load_col].notna().any():
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        load = df[load_col].values
        valid = ~np.isnan(load) & ~np.isnan(rmse_deg)
        if valid.sum() > 0:
            ax.scatter(load[valid], rmse_deg[valid], alpha=0.6, s=25)
            ax.set_xlabel("Load (pu)" if load_col == "param_load" else "Pm (pu)")
            ax.set_ylabel(r"RMSE $\delta$ (deg)")
            ax.set_title("Trajectory error vs load")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            output_dir / "multimachine_publication_error_vs_load.png", dpi=dpi, bbox_inches="tight"
        )
        plt.close()
        figs.append("multimachine_publication_error_vs_load.png")
    # vs H
    if "param_H" in df.columns and df["param_H"].notna().any():
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        H = df["param_H"].values
        valid = ~np.isnan(H) & ~np.isnan(rmse_deg)
        if valid.sum() > 0:
            ax.scatter(H[valid], rmse_deg[valid], alpha=0.6, s=25)
            ax.set_xlabel("H (s)")
            ax.set_ylabel(r"RMSE $\delta$ (deg)")
            ax.set_title("Trajectory error vs inertia H")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            output_dir / "multimachine_publication_error_vs_H.png", dpi=dpi, bbox_inches="tight"
        )
        plt.close()
        figs.append("multimachine_publication_error_vs_H.png")


def plot_per_generator_summary(
    df: pd.DataFrame, num_machines: int, output_dir: Path, dpi: int = 300
) -> None:
    """Bar chart and CSV table: per-generator and overall RMSE/MAE (deg)."""
    gen_rmse_cols = [
        f"delta_rmse_gen{i}_rad"
        for i in range(num_machines)
        if f"delta_rmse_gen{i}_rad" in df.columns
    ]
    gen_mae_cols = [
        f"delta_mae_gen{i}_rad"
        for i in range(num_machines)
        if f"delta_mae_gen{i}_rad" in df.columns
    ]
    if not gen_rmse_cols:
        # Fallback: use overall delta_rmse_rad for a single "overall" bar
        if "delta_rmse_rad" not in df.columns or df.empty:
            return
        overall_rad = df["delta_rmse_rad"].mean()
        overall_mae = (
            np.mean(np.abs(df["delta_rmse_rad"].values)) if "delta_rmse_rad" in df.columns else None
        )  # approx
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.bar([0], [_rad2deg(overall_rad)], color="C0", edgecolor="black")
        ax.set_xticks([0])
        ax.set_xticklabels(["Overall"])
        ax.set_ylabel(r"Mean RMSE $\delta$ (deg)")
        ax.set_title("Trajectory error (test set)")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(
            output_dir / "multimachine_publication_per_generator_rmse.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()
        summary = pd.DataFrame(
            [
                {
                    "generator": "overall",
                    "RMSE_deg": _rad2deg(overall_rad),
                    "MAE_deg": _rad2deg(overall_mae) if overall_mae is not None else None,
                }
            ]
        )
        summary.to_csv(output_dir / "multimachine_publication_summary_metrics.csv", index=False)
        return
    means_rmse_rad = [df[c].mean() for c in gen_rmse_cols]
    means_rmse_deg = [_rad2deg(m) for m in means_rmse_rad]
    means_mae_deg = (
        [_rad2deg(df[c].mean()) for c in gen_mae_cols]
        if gen_mae_cols
        else [None] * len(gen_rmse_cols)
    )
    overall_rad = (
        df["delta_rmse_rad"].mean() if "delta_rmse_rad" in df.columns else np.mean(means_rmse_rad)
    )
    mae_cols = [
        f"delta_mae_gen{i}_rad"
        for i in range(num_machines)
        if f"delta_mae_gen{i}_rad" in df.columns
    ]
    overall_mae_rad = np.mean([df[c].mean() for c in mae_cols]) if mae_cols else None
    labels = [f"Gen {i+1}" for i in range(len(gen_rmse_cols))] + ["Overall"]
    values_deg = means_rmse_deg + [_rad2deg(overall_rad)]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x = np.arange(len(labels))
    ax.bar(x, values_deg, color=["C0", "C1", "C2", "C3", "gray"][: len(labels)], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Mean RMSE $\delta$ (deg)")
    ax.set_title("Per-generator and overall trajectory error (test set)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(
        output_dir / "multimachine_publication_per_generator_rmse.png", dpi=dpi, bbox_inches="tight"
    )
    plt.close()
    rows = [
        {
            "generator": f"gen{i+1}",
            "RMSE_deg": means_rmse_deg[i],
            "MAE_deg": means_mae_deg[i]
            if i < len(means_mae_deg) and means_mae_deg[i] is not None
            else None,
        }
        for i in range(len(gen_rmse_cols))
    ]
    rows.append(
        {
            "generator": "overall",
            "RMSE_deg": _rad2deg(overall_rad),
            "MAE_deg": _rad2deg(overall_mae_rad) if overall_mae_rad is not None else None,
        }
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "multimachine_publication_summary_metrics.csv", index=False)


def plot_stability_agreement(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> None:
    """Stability agreement: ANDES vs PINN (is_stable vs is_stable_pred). Bar or simple text summary."""
    if "is_stable" not in df.columns or "is_stable_pred" not in df.columns or df.empty:
        return
    true_stable = df["is_stable"].astype(bool)
    pred_stable = df["is_stable_pred"].astype(bool)
    correct = (true_stable == pred_stable).sum()
    total = len(df)
    pct = 100.0 * correct / total if total else 0
    # Confusion-style 2x2 counts
    tp = ((true_stable) & (pred_stable)).sum()
    tn = ((~true_stable) & (~pred_stable)).sum()
    fp = ((~true_stable) & (pred_stable)).sum()
    fn = ((true_stable) & (~pred_stable)).sum()
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.axis("off")
    text = (
        "Stability classification (ANDES vs PINN)\n"
        f"  Correct: {correct} / {total} ({pct:.1f}%)\n"
        f"  TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}"
    )
    ax.text(
        0.1,
        0.5,
        text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="center",
        family="monospace",
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "multimachine_publication_stability_agreement.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()
    # Also save a small CSV for paper table
    summary = pd.DataFrame(
        [
            {
                "metric": "stability_accuracy_pct",
                "value": pct,
                "correct": correct,
                "total": total,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
            }
        ]
    )
    summary.to_csv(output_dir / "multimachine_publication_stability_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot publication figures from multimachine evaluation CSV"
    )
    parser.add_argument(
        "--metrics-csv", type=str, default=None, help="Path to multimachine_eval_metrics_*.csv"
    )
    parser.add_argument(
        "--metrics-dir", type=str, default=None, help="Directory to search for latest metrics CSV"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory (default: same as metrics)"
    )
    parser.add_argument(
        "--num-machines", type=int, default=4, help="Number of generators (default 4)"
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for figures (default 300)")
    parser.add_argument(
        "--skip-error-dist", action="store_true", help="Skip error distribution figure"
    )
    parser.add_argument(
        "--skip-error-vs-param", action="store_true", help="Skip error vs parameter figures"
    )
    parser.add_argument("--skip-per-gen", action="store_true", help="Skip per-generator summary")
    parser.add_argument(
        "--skip-stability", action="store_true", help="Skip stability agreement figure"
    )
    args = parser.parse_args()

    if not _MPL:
        print("matplotlib not available; cannot generate figures.")
        sys.exit(1)

    metrics_path = None
    if args.metrics_csv:
        p = Path(args.metrics_csv)
        if "*" in p.name:
            matches = list(p.parent.glob(p.name))
            metrics_path = max(matches, key=lambda x: x.stat().st_mtime) if matches else None
        else:
            metrics_path = p if p.exists() else None
    if metrics_path is None and args.metrics_dir:
        d = Path(args.metrics_dir)
        if not d.is_absolute():
            d = PROJECT_ROOT / d
        matches = list(d.glob("multimachine_eval_metrics_*.csv"))
        if matches:
            metrics_path = max(matches, key=lambda x: x.stat().st_mtime)
    if metrics_path is None or not metrics_path.exists():
        print("No metrics CSV found. Use --metrics-csv or --metrics-dir.")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path)
    if df.empty:
        print("Metrics CSV is empty.")
        sys.exit(0)

    if not args.skip_error_dist:
        plot_error_distribution(df, output_dir, dpi=args.dpi)
        print("  Saved: multimachine_publication_error_distribution.png")
    if not args.skip_error_vs_param:
        plot_error_vs_parameter(df, output_dir, dpi=args.dpi)
        print("  Saved: multimachine_publication_error_vs_*.png")
    if not args.skip_per_gen:
        plot_per_generator_summary(df, args.num_machines, output_dir, dpi=args.dpi)
        print(
            "  Saved: multimachine_publication_per_generator_rmse.png, multimachine_publication_summary_metrics.csv"
        )
    if not args.skip_stability:
        plot_stability_agreement(df, output_dir, dpi=args.dpi)
        print(
            "  Saved: multimachine_publication_stability_agreement.png, multimachine_publication_stability_metrics.csv"
        )

    print("Publication figures done.")


if __name__ == "__main__":
    main()
