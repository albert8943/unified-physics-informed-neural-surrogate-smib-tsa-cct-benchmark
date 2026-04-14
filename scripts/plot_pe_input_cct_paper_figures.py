#!/usr/bin/env python
"""
Publication-style figures from ``pe_input_cct_test_results.json`` (output of
``scripts/run_pe_input_cct_test.py``).

Generates IEEE-friendly PNGs for the manuscript:
  1. Nominal-clearing stability: multi-row confusion matrices (PINN* vs Std NN* per
     ``pw_*`` width) when those JSONs exist under the default sensitivity output directory,
     otherwise a single row from ``--json``; ``--nominal-confusion-single-row`` forces one row.
  2. CCT: parity plot (reference vs surrogate).

Example (repo root)::

    python scripts/plot_pe_input_cct_paper_figures.py ^
      --json outputs/campaign_indep_cct_pe_input_test_20260406/pe_input_cct_test_results.json ^
      --output-dir "paper_writing/IEEE Access Template/figures/pe_input_cct"

By default, the nominal-clearing confusion figure is a **multi-row** grid (one row per
``--nominal-multiwindow-seconds``) when ``pw_*/pe_input_cct_test_results.json`` exists for
each width under ``outputs/paper_pe_input_cct_sensitivity_pw`` (repo root). Otherwise it
falls back to a single row from ``--json``. Use ``--nominal-confusion-single-row`` to force
the one-row figure. Override the search directory with ``--nominal-multiwindow-dir``::

    python scripts/plot_pe_input_cct_paper_figures.py ^
      --json outputs/paper_pe_input_cct_persistence_default/pe_input_cct_test_results.json ^
      --output-dir "paper_writing/IEEE Access Template/figures/pe_input_cct" ^
      --nominal-multiwindow-dir path/to/pw_parent ^
      --nominal-multiwindow-seconds 0.05 0.15 0.25 0.5

LaTeX: use ``\\includegraphics{figures/pe_input_cct/<name>.png}`` paths relative to
``access.tex`` (not ``../figures/...``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOMINAL_MULTIWINDOW_DIR = PROJECT_ROOT / "outputs/paper_pe_input_cct_sensitivity_pw"

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Approximate IEEE single-column width (in) for ieeeaccess / IEEEtran one-column
FIG_SINGLE_COL_W = 3.46
FIG_DPI = 300


def _confusion_from_per_scenario(block: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """Return (tp, tn, fp, fn); positive class = stable."""
    tp = tn = fp = fn = 0
    for row in block.get("per_scenario") or []:
        if "error" in row or "stable_label" not in row or "stable_pred" not in row:
            continue
        gt, pr = bool(row["stable_label"]), bool(row["stable_pred"])
        if gt and pr:
            tp += 1
        elif not gt and not pr:
            tn += 1
        elif not gt and pr:
            fp += 1
        else:
            fn += 1
    return tp, tn, fp, fn


def verify_results(results: Dict[str, Any]) -> List[str]:
    """Cross-check JSON blocks; return human-readable issues (empty if consistent)."""
    issues: List[str] = []
    stab = results.get("stability_at_nominal_tc") or {}
    for key, name in (("pinn", "PINN"), ("ml", "ML")):
        block = stab.get(key) or {}
        m = block.get("metrics") or {}
        tp, tn, fp, fn = _confusion_from_per_scenario(block)
        for lbl, a, b in (
            ("tp", m.get("tp"), tp),
            ("tn", m.get("tn"), tn),
            ("fp", m.get("fp"), fp),
            ("fn", m.get("fn"), fn),
        ):
            if a is not None and int(a) != b:
                issues.append(f"stability {name}: metrics.{lbl}={a} but per_scenario implies {b}")
    for key, name in (("pinn", "PINN"), ("ml", "ML")):
        block = results.get(key) or {}
        rows = [
            r
            for r in block.get("per_scenario") or []
            if "abs_error_s" in r and "cct_true" in r and "cct_est" in r
        ]
        if not rows:
            continue
        errs = np.array([float(r["abs_error_s"]) for r in rows], dtype=float)
        mae = float(np.mean(errs))
        rmse = float(np.sqrt(np.mean(errs**2)))
        j_mae, j_rmse = block.get("mae_s"), block.get("rmse_s")
        if j_mae is not None and abs(mae - float(j_mae)) > 1e-9:
            issues.append(f"CCT {name}: mae_s mismatch (json {j_mae} vs per_scenario {mae})")
        if j_rmse is not None and abs(rmse - float(j_rmse)) > 1e-9:
            issues.append(f"CCT {name}: rmse_s mismatch (json {j_rmse} vs per_scenario {rmse})")
        j_mar = block.get("mean_abs_rel_error")
        if j_mar is not None:
            rels = [
                float(r["abs_error_s"]) / float(r["cct_true"])
                for r in rows
                if float(r["cct_true"]) > 0.0
            ]
            if rels:
                mar = float(np.mean(rels))
                if abs(mar - float(j_mar)) > 1e-9:
                    issues.append(
                        f"CCT {name}: mean_abs_rel_error mismatch "
                        f"(json {j_mar} vs per_scenario {mar})"
                    )
    return issues


def _apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "font.family": "serif",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "axes.grid": False,
        }
    )


def _confusion_counts_from_metrics(m: Dict[str, Any]) -> np.ndarray:
    """2x2 matrix [ [TN, FP], [FN, TP] ] with rows=true, cols=pred; positive class=stable."""
    tp = int(m.get("tp", 0))
    tn = int(m.get("tn", 0))
    fp = int(m.get("fp", 0))
    fn = int(m.get("fn", 0))
    return np.array([[tn, fp], [fn, tp]], dtype=float)


def _persistence_window_dirname(seconds: float) -> str:
    """Match scripts/run_pe_input_cct_persistence_window_sensitivity.py folder names."""
    w = round(float(seconds), 6)
    s = f"{w:.10f}".rstrip("0").rstrip(".")
    return "pw_" + s.replace(".", "p")


def _nominal_multiwindow_jsons_ready(base_dir: Path, window_seconds: Sequence[float]) -> bool:
    """True if each width has ``base_dir/pw_*/pe_input_cct_test_results.json``."""
    if not base_dir.is_dir():
        return False
    for ws in window_seconds:
        p = base_dir / _persistence_window_dirname(ws) / "pe_input_cct_test_results.json"
        if not p.is_file():
            return False
    return True


def plot_nominal_stability_confusion_multiwindow(
    base_dir: Path,
    window_seconds: List[float],
    out_path: Path,
) -> None:
    """
    One row per persistence-window width, two columns (PINN*, Std NN*), shared color scale.
    Expects ``base_dir / pw_<tag> / pe_input_cct_test_results.json`` for each width.

    Layout: column model names once; axis semantics follow common ML convention
    (rows = true class, columns = predicted class; stable = positive class).
    """
    loaded: List[Tuple[float, Dict[str, Any]]] = []
    for ws in window_seconds:
        p = base_dir / _persistence_window_dirname(ws) / "pe_input_cct_test_results.json"
        if not p.is_file():
            raise FileNotFoundError(f"Missing multi-window JSON for Δt_w={ws}: {p}")
        with open(p, encoding="utf-8") as f:
            loaded.append((float(ws), json.load(f)))

    nrows = len(loaded)
    _apply_ieee_style()
    fig_w = FIG_SINGLE_COL_W
    fig_h = min(0.78 * nrows + 1.35, 6.6)
    fig, axes = plt.subplots(nrows, 2, figsize=(fig_w, fig_h))
    if nrows == 1:
        axes = np.reshape(axes, (1, 2))

    # Class names on axes; "True class" is figure-level (avoids overlap with $\Delta t_w$).
    class_ticks = ["Unstable", "Stable"]

    all_cms: List[np.ndarray] = []
    for _ws, res in loaded:
        stab = res.get("stability_at_nominal_tc") or {}
        for key in ("pinn", "ml"):
            m = (stab.get(key) or {}).get("metrics") or {}
            all_cms.append(_confusion_counts_from_metrics(m))
    vmax = float(max(c.max() for c in all_cms))

    ims: List[Any] = []
    for row_idx, (ws, res) in enumerate(loaded):
        stab = res.get("stability_at_nominal_tc") or {}
        for col_idx, key in enumerate(("pinn", "ml")):
            ax = axes[row_idx, col_idx]
            m = (stab.get(key) or {}).get("metrics") or {}
            cm = _confusion_counts_from_metrics(m)
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(vmax, 1.0))
            ims.append(im)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.tick_params(length=2, pad=1)
            if row_idx == nrows - 1:
                ax.set_xticklabels(class_ticks, fontsize=6.5, rotation=12, ha="right")
            else:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_yticklabels(class_ticks, fontsize=6.5)
            else:
                ax.set_yticklabels([])
            thr_txt = 0.55 * vmax
            for (j, i), val in np.ndenumerate(cm):
                color = "white" if val > thr_txt else "black"
                ax.text(
                    i,
                    j,
                    f"{int(val)}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

    # Extra left margin: strip for "True class" (rotated) + $\Delta t_w$ column (no ylabel overlap).
    fig.subplots_adjust(left=0.29, right=0.86, top=0.86, bottom=0.17, hspace=0.28, wspace=0.12)

    y_grid_mid = 0.5 * (axes[-1, 0].get_position().y0 + axes[0, 0].get_position().y1)
    fig.text(
        0.03,
        y_grid_mid,
        "True class",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7.5,
        transform=fig.transFigure,
    )

    # Fixed x so row labels align and stay right of "True class".
    x_dt_labels = 0.14
    for row_idx, (ws, _res) in enumerate(loaded):
        pos = axes[row_idx, 0].get_position()
        fig.text(
            x_dt_labels,
            0.5 * (pos.y0 + pos.y1),
            rf"$\Delta t_w={ws:g}$ s",
            ha="right",
            va="center",
            fontsize=7.5,
            transform=fig.transFigure,
        )

    # Column headers (model names) once above the grid.
    pos00 = axes[0, 0].get_position()
    pos01 = axes[0, 1].get_position()
    y_hdr = min(pos00.y1, pos01.y1) + 0.028
    fig.text(0.5 * (pos00.x0 + pos00.x1), y_hdr, "PINN*", ha="center", va="bottom", fontsize=8)
    fig.text(0.5 * (pos01.x0 + pos01.x1), y_hdr, "Std NN*", ha="center", va="bottom", fontsize=8)

    # Predicted axis once (standard confusion-matrix wording).
    pos_bot = axes[nrows - 1, 0].get_position()
    pos_bot1 = axes[nrows - 1, 1].get_position()
    fig.text(
        0.5 * (pos_bot.x0 + pos_bot1.x1),
        0.045,
        "Predicted class",
        ha="center",
        va="bottom",
        fontsize=7,
        transform=fig.transFigure,
    )

    cbar_ax = fig.add_axes([0.89, 0.22, 0.022, 0.56])
    fig.colorbar(ims[0], cax=cbar_ax, label="Count")
    fig.suptitle(
        r"Nominal $t_c$: confusion matrices",
        fontsize=8,
        y=0.98,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)


def plot_nominal_stability_confusion(
    results: Dict[str, Any], out_path: Path, title_prefix: str = ""
) -> None:
    stab = results.get("stability_at_nominal_tc") or {}
    fig, axes = plt.subplots(1, 2, figsize=(FIG_SINGLE_COL_W, 1.65), constrained_layout=True)
    labels_row = ["True unstable", "True stable"]
    labels_col = ["Pred. unstable", "Pred. stable"]

    cms: List[np.ndarray] = []
    for key in ("pinn", "ml"):
        block = stab.get(key) or {}
        m = block.get("metrics") or {}
        cms.append(_confusion_counts_from_metrics(m))
    vmax = float(max((c.max() for c in cms), default=1.0))

    ims = []
    for ax, cm, display in zip(
        axes,
        cms,
        ("PINN*", "Std NN*"),
    ):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(vmax, 1.0))
        ims.append(im)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels_col, rotation=25, ha="right")
        ax.set_yticklabels(labels_row)
        ax.set_title(f"{title_prefix}{display}")
        thr_txt = 0.55 * vmax
        for (j, i), val in np.ndenumerate(cm):
            color = "white" if val > thr_txt else "black"
            ax.text(i, j, f"{int(val)}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(ims[0], ax=axes.ravel().tolist(), shrink=0.72, label="Count")
    fig.suptitle(
        r"Nominal $t_c$: confusion matrices",
        fontsize=8,
        y=1.05,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _cct_pairs(block: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    true_s: List[float] = []
    est_s: List[float] = []
    for row in block.get("per_scenario") or []:
        if "error" in row:
            continue
        if "cct_true" not in row or "cct_est" not in row:
            continue
        true_s.append(float(row["cct_true"]))
        est_s.append(float(row["cct_est"]))
    return np.asarray(true_s, dtype=float), np.asarray(est_s, dtype=float)


def plot_cct_parity(results: Dict[str, Any], out_path: Path) -> None:
    """Reference vs surrogate CCT (parity plot); scalar errors are reported in the paper tables."""
    pinn_b = results.get("pinn") or {}
    ml_b = results.get("ml") or {}
    t_p, e_p = _cct_pairs(pinn_b)
    t_m, e_m = _cct_pairs(ml_b)

    with plt.rc_context(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    ):
        fig = plt.figure(figsize=(FIG_SINGLE_COL_W, 3.25))
        ax_parity = fig.add_axes((0.13, 0.28, 0.85, 0.62))

        # Span both reference and surrogate so the identity line covers the visible range.
        parts = [a for a in (t_p, e_p, t_m, e_m) if getattr(a, "size", 0) > 0]
        all_vals = np.concatenate(parts) if parts else np.array([])
        if all_vals.size:
            lo = float(np.min(all_vals))
            hi = float(np.max(all_vals))
            pad = 0.02 * (hi - lo + 1e-9)
            ax_parity.plot(
                [lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.9, label="Identity"
            )
        ax_parity.scatter(
            t_p, e_p, s=16, marker="o", edgecolors="k", linewidths=0.3, label="PINN*", zorder=3
        )
        ax_parity.scatter(
            t_m, e_m, s=16, marker="s", edgecolors="k", linewidths=0.3, label="Std NN*", zorder=3
        )
        ax_parity.set_aspect("equal", adjustable="box")
        ax_parity.set_xlabel(r"Reference $t_{\mathrm{CCT}}^{\mathrm{ref}}$ (s)", labelpad=5)
        ax_parity.set_ylabel(r"Surrogate $\hat{t}_{\mathrm{CCT}}$ (s)")
        ax_parity.set_title("Parity (CCT)", pad=8)
        ax_parity.tick_params(axis="both", which="major", pad=2)

        h_leg, lab_leg = ax_parity.get_legend_handles_labels()
        ax_parity.legend(
            h_leg,
            lab_leg,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=3,
            frameon=True,
            fancybox=False,
            edgecolor="0.6",
            columnspacing=0.85,
            handletextpad=0.35,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", facecolor="white", pad_inches=0.04)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Pe-input CCT / stability paper figures from JSON.")
    p.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to pe_input_cct_test_results.json",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for PNG outputs",
    )
    p.add_argument(
        "--basename-confusion",
        type=str,
        default="pe_input_nominal_stability_confusion",
        help="Filename stem for confusion figure (adds .png)",
    )
    p.add_argument(
        "--basename-cct",
        type=str,
        default="pe_input_cct_parity",
        help="Filename stem for CCT parity figure",
    )
    p.add_argument(
        "--strict-verify",
        action="store_true",
        help="Exit with code 1 if JSON metrics disagree with per_scenario recomputation.",
    )
    p.add_argument(
        "--nominal-multiwindow-dir",
        type=Path,
        default=None,
        help=(
            "Parent of pw_*/pe_input_cct_test_results.json for nominal multi-row confusion. "
            "If omitted, try "
            f"{DEFAULT_NOMINAL_MULTIWINDOW_DIR} "
            "(all --nominal-multiwindow-seconds widths must be present)."
        ),
    )
    p.add_argument(
        "--nominal-confusion-single-row",
        action="store_true",
        help="Force one-row nominal confusion from --json (skip multi-window JSONs).",
    )
    p.add_argument(
        "--nominal-multiwindow-seconds",
        type=float,
        nargs="+",
        default=[0.05, 0.15, 0.25, 0.5],
        help="Persistence window widths (s) for nominal multi-row confusion grid.",
    )
    args = p.parse_args()

    if not args.json.is_file():
        print(f"JSON not found: {args.json}", file=sys.stderr)
        sys.exit(1)

    with open(args.json, encoding="utf-8") as f:
        results = json.load(f)

    v_issues = verify_results(results)
    for msg in v_issues:
        print(f"WARNING: {msg}", file=sys.stderr)
    if args.strict_verify and v_issues:
        sys.exit(1)

    windows = list(args.nominal_multiwindow_seconds)
    explicit_pw_dir = args.nominal_multiwindow_dir is not None
    multi_dir = args.nominal_multiwindow_dir or DEFAULT_NOMINAL_MULTIWINDOW_DIR

    if args.nominal_confusion_single_row:
        use_nominal_multiwindow = False
    elif _nominal_multiwindow_jsons_ready(multi_dir, windows):
        use_nominal_multiwindow = True
    elif explicit_pw_dir:
        print(
            f"ERROR: --nominal-multiwindow-dir {multi_dir} is missing required JSONs.",
            file=sys.stderr,
        )
        for ws in windows:
            need = multi_dir / _persistence_window_dirname(ws) / "pe_input_cct_test_results.json"
            if not need.is_file():
                print(f"  missing: {need}", file=sys.stderr)
        sys.exit(1)
    else:
        print(
            "WARNING: Nominal confusion multi-row inputs not found under "
            f"{multi_dir}; writing single-row figure from --json. "
            "Generate pw_* JSONs with scripts/run_pe_input_cct_persistence_window_sensitivity.py "
            "(e.g. --windows 0.05 0.15 0.25 0.5) or pass --nominal-multiwindow-dir.",
            file=sys.stderr,
        )
        use_nominal_multiwindow = False

    _apply_ieee_style()
    if use_nominal_multiwindow:
        plot_nominal_stability_confusion_multiwindow(
            multi_dir,
            windows,
            args.output_dir / f"{args.basename_confusion}.png",
        )
    else:
        plot_nominal_stability_confusion(
            results,
            args.output_dir / f"{args.basename_confusion}.png",
        )
    plot_cct_parity(
        results,
        args.output_dir / f"{args.basename_cct}.png",
    )
    print(
        f"Wrote:\n  {args.output_dir / (args.basename_confusion + '.png')}\n"
        f"  {args.output_dir / (args.basename_cct + '.png')}"
    )


if __name__ == "__main__":
    main()
