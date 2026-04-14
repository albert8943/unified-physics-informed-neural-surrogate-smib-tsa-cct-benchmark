#!/usr/bin/env python
"""
Sweep ``--persistence-window-seconds`` for ``scripts/run_pe_input_cct_test.py``,
aggregate nominal-stability and CCT metrics, and plot sensitivity curves.

Default grid (seconds): 0.05, 0.15, 0.25, 0.35, 0.45, 0.5

Example (repo root)::

    python scripts/run_pe_input_cct_persistence_window_sensitivity.py

    python scripts/run_pe_input_cct_persistence_window_sensitivity.py ^
      --skip-existing --aggregate-only

IEEE Access manuscript (match nominal-clearing four widths)::

    python scripts/run_pe_input_cct_persistence_window_sensitivity.py --aggregate-only ^
      --windows 0.05 0.15 0.25 0.5 ^
      --paper-figures-dir "paper_writing/IEEE Access Template/figures/pe_input_cct"

Validation-based \(\Delta t_w\) selection (Option A): point ``--test-csv`` at **val** CSV,
then run ``scripts/select_delta_tw_from_val_cct_sweep.py``; see
``docs/publication/rerun_recipes/RERUN_delta_tw_validation_selection_option_A.md``.

Outputs under ``--base-output-dir`` (default ``outputs/paper_pe_input_cct_sensitivity_pw``)::

    pw_0p05/pe_input_cct_test_results.json
    ...
    persistence_window_sensitivity_summary.csv
    persistence_window_sensitivity.png
    persistence_window_sensitivity_table.tex   (optional \\input{} fragment)
    persistence_window_cct_mae_rmse.png        (CCT MAE/RMSE vs.\ persistence window)
    persistence_window_cct_mae_rmse_table.tex (supplementary \\input{} fragment)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WINDOWS_S = (0.05, 0.15, 0.25, 0.35, 0.45, 0.5)

# Pinned headline checkpoints (same as docs/publication/rerun_recipes/RERUN_pe_input_cct_nominal_stability_and_figures.md)
DEFAULT_TEST_CSV = PROJECT_ROOT / "data/processed/exp_20260211_190612/test_data_20260211_190612.csv"
DEFAULT_PINN_MODEL = (
    PROJECT_ROOT
    / "outputs/expt_residual_backbone_retrain_20260407/pinn_nores_lp05/pinn/best_model_20260407_172450.pth"
)
DEFAULT_PINN_CONFIG = (
    PROJECT_ROOT / "outputs/expt_residual_backbone_retrain_20260407/pinn_nores_lp05/config.yaml"
)
DEFAULT_ML_MODEL = (
    PROJECT_ROOT / "outputs/campaign_indep_ml/ml_uw2_ow50/ml_baseline/standard_nn/model/model.pth"
)


def persistence_window_dirname(seconds: float) -> str:
    w = round(float(seconds), 6)
    s = f"{w:.10f}".rstrip("0").rstrip(".")
    return "pw_" + s.replace(".", "p")


def _metrics_row(pw_s: float, data: Dict[str, Any]) -> Dict[str, Any]:
    stab = data.get("stability_at_nominal_tc") or {}
    pinn_s = (stab.get("pinn") or {}).get("metrics") or {}
    ml_s = (stab.get("ml") or {}).get("metrics") or {}
    pinn_c = data.get("pinn") or {}
    ml_c = data.get("ml") or {}
    return {
        "persistence_window_s": pw_s,
        "pinn_accuracy": pinn_s.get("accuracy"),
        "pinn_precision_stable": pinn_s.get("precision_stable"),
        "pinn_recall_stable": pinn_s.get("recall_stable"),
        "pinn_f1_stable": pinn_s.get("f1_stable"),
        "pinn_tp": pinn_s.get("tp"),
        "pinn_tn": pinn_s.get("tn"),
        "pinn_fp": pinn_s.get("fp"),
        "pinn_fn": pinn_s.get("fn"),
        "ml_accuracy": ml_s.get("accuracy"),
        "ml_precision_stable": ml_s.get("precision_stable"),
        "ml_recall_stable": ml_s.get("recall_stable"),
        "ml_f1_stable": ml_s.get("f1_stable"),
        "ml_tp": ml_s.get("tp"),
        "ml_tn": ml_s.get("tn"),
        "ml_fp": ml_s.get("fp"),
        "ml_fn": ml_s.get("fn"),
        "pinn_cct_mae_s": pinn_c.get("mae_s"),
        "pinn_cct_rmse_s": pinn_c.get("rmse_s"),
        "ml_cct_mae_s": ml_c.get("mae_s"),
        "ml_cct_rmse_s": ml_c.get("rmse_s"),
    }


def load_summary_rows(base_dir: Path, windows_s: Sequence[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for w in windows_s:
        sub = base_dir / persistence_window_dirname(w) / "pe_input_cct_test_results.json"
        if not sub.is_file():
            raise FileNotFoundError(f"Missing results for window {w}: {sub}")
        with open(sub, encoding="utf-8") as f:
            data = json.load(f)
        proto = data.get("protocol") or {}
        pw_file = proto.get("persistence_window_seconds")
        if pw_file is not None and not math.isclose(
            float(pw_file), float(w), rel_tol=0.0, abs_tol=1e-5
        ):
            warnings.warn(
                f"{sub}: protocol.persistence_window_seconds={pw_file!r} does not match "
                f"expected window {w!r} (folder {persistence_window_dirname(w)}). "
                "Aggregate row uses the requested window label, not the JSON value.",
                UserWarning,
                stacklevel=2,
            )
        rows.append(_metrics_row(w, data))
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_latex_table(rows: List[Dict[str, Any]], path: Path) -> None:
    """Booktabs-style fragment: persistence window vs nominal accuracy and CCT MAE (ms)."""

    def _ms(x: Optional[float]) -> str:
        if x is None:
            return "---"
        return f"{float(x) * 1000.0:.2f}"

    def _acc(x: Optional[float]) -> str:
        if x is None:
            return "---"
        return f"{float(x):.3f}"

    lines = [
        r"% Auto-generated by scripts/run_pe_input_cct_persistence_window_sensitivity.py",
        r"\begin{tabular}{@{}c*{4}{c}@{}}",
        r"\toprule",
        r"$\Delta t_w$\,(s) & \multicolumn{2}{c}{Nom.\ acc.} & \multicolumn{2}{c}{CCT MAE\,(ms)} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r" & PINN* & Std NN* & PINN* & Std NN* \\",
        r"\midrule",
    ]
    for r in rows:
        pw = r["persistence_window_s"]
        lines.append(
            f"{pw:g} & {_acc(r.get('pinn_accuracy'))} & {_acc(r.get('ml_accuracy'))} & "
            f"{_ms(r.get('pinn_cct_mae_s'))} & {_ms(r.get('ml_cct_mae_s'))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_cct_mae_rmse_table(rows: List[Dict[str, Any]], path: Path) -> None:
    """Supplementary fragment: $\Delta t_w$ vs CCT MAE and RMSE (ms) for both models."""

    def _ms(x: Optional[float]) -> str:
        if x is None:
            return "---"
        return f"{float(x) * 1000.0:.2f}"

    lines = [
        r"% Auto-generated by scripts/run_pe_input_cct_persistence_window_sensitivity.py",
        r"\begin{tabular}{@{}c*{4}{c}@{}}",
        r"\toprule",
        r"$\Delta t_w$\,(s) & \multicolumn{2}{c}{PINN*} & \multicolumn{2}{c}{Std NN*} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r" & MAE\,(ms) & RMSE\,(ms) & MAE\,(ms) & RMSE\,(ms) \\",
        r"\midrule",
    ]
    for r in rows:
        pw = r["persistence_window_s"]
        lines.append(
            f"{pw:g} & {_ms(r.get('pinn_cct_mae_s'))} & {_ms(r.get('pinn_cct_rmse_s'))} & "
            f"{_ms(r.get('ml_cct_mae_s'))} & {_ms(r.get('ml_cct_rmse_s'))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _as_float_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    return float(x)


def plot_sensitivity(rows: List[Dict[str, Any]], out_png: Path, dpi: int = 300) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        raise ValueError("plot_sensitivity: no rows to plot")

    x = np.array([float(r["persistence_window_s"]) for r in rows], dtype=float)
    pinn_acc = np.array([_as_float_or_nan(r.get("pinn_accuracy")) for r in rows], dtype=float)
    ml_acc = np.array([_as_float_or_nan(r.get("ml_accuracy")) for r in rows], dtype=float)
    pinn_mae_ms = np.array(
        [_as_float_or_nan(r.get("pinn_cct_mae_s")) * 1000.0 for r in rows], dtype=float
    )
    ml_mae_ms = np.array(
        [_as_float_or_nan(r.get("ml_cct_mae_s")) * 1000.0 for r in rows], dtype=float
    )

    if np.all(np.isnan(pinn_acc)) and np.all(np.isnan(ml_acc)):
        raise ValueError("plot_sensitivity: all nominal accuracy values missing (None)")
    if np.all(np.isnan(pinn_mae_ms)) and np.all(np.isnan(ml_mae_ms)):
        raise ValueError("plot_sensitivity: all CCT MAE values missing (None)")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(3.6, 5.0), sharex=True, constrained_layout=True)

    ax0.plot(x, pinn_acc, "o-", color="#1f77b4", label="PINN*", ms=5, lw=1.2)
    ax0.plot(x, ml_acc, "s--", color="#ff7f0e", label="Std NN*", ms=4, lw=1.2)
    ax0.set_ylabel("Nominal accuracy")
    acc_stack = np.concatenate([pinn_acc, ml_acc])
    acc_finite = acc_stack[np.isfinite(acc_stack)]
    if acc_finite.size:
        lo = float(np.min(acc_finite))
        ax0.set_ylim(max(0.0, lo - 0.05), 1.02)
    ax0.grid(True, alpha=0.35, linestyle=":")
    ax0.legend(loc="lower right", fontsize=8)

    ax1.plot(x, pinn_mae_ms, "o-", color="#1f77b4", label="PINN*", ms=5, lw=1.2)
    ax1.plot(x, ml_mae_ms, "s--", color="#ff7f0e", label="Std NN*", ms=4, lw=1.2)
    ax1.set_xlabel(r"Persistence window $\Delta t_w$ (s)")
    ax1.set_ylabel("CCT MAE (ms)")
    ax1.grid(True, alpha=0.35, linestyle=":")
    ax1.legend(loc="upper left", fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_cct_mae_rmse(rows: List[Dict[str, Any]], out_png: Path, dpi: int = 300) -> None:
    """Two-panel figure: CCT MAE and CCT RMSE (ms) vs persistence window."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        raise ValueError("plot_cct_mae_rmse: no rows to plot")

    x = np.array([float(r["persistence_window_s"]) for r in rows], dtype=float)
    pinn_mae_ms = np.array(
        [_as_float_or_nan(r.get("pinn_cct_mae_s")) * 1000.0 for r in rows], dtype=float
    )
    pinn_rmse_ms = np.array(
        [_as_float_or_nan(r.get("pinn_cct_rmse_s")) * 1000.0 for r in rows], dtype=float
    )
    ml_mae_ms = np.array(
        [_as_float_or_nan(r.get("ml_cct_mae_s")) * 1000.0 for r in rows], dtype=float
    )
    ml_rmse_ms = np.array(
        [_as_float_or_nan(r.get("ml_cct_rmse_s")) * 1000.0 for r in rows], dtype=float
    )

    if np.all(np.isnan(pinn_mae_ms)) and np.all(np.isnan(ml_mae_ms)):
        raise ValueError("plot_cct_mae_rmse: all CCT MAE values missing (None)")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(3.6, 5.0), sharex=True, constrained_layout=True)

    ax0.plot(x, pinn_mae_ms, "o-", color="#1f77b4", label="PINN*", ms=5, lw=1.2)
    ax0.plot(x, ml_mae_ms, "s--", color="#ff7f0e", label="Std NN*", ms=4, lw=1.2)
    ax0.set_ylabel("CCT MAE (ms)")
    ax0.grid(True, alpha=0.35, linestyle=":")
    ax0.legend(loc="upper left", fontsize=8)

    ax1.plot(x, pinn_rmse_ms, "o-", color="#1f77b4", label="PINN*", ms=5, lw=1.2)
    ax1.plot(x, ml_rmse_ms, "s--", color="#ff7f0e", label="Std NN*", ms=4, lw=1.2)
    ax1.set_xlabel(r"Persistence window $\Delta t_w$ (s)")
    ax1.set_ylabel("CCT RMSE (ms)")
    ax1.grid(True, alpha=0.35, linestyle=":")
    ax1.legend(loc="upper left", fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _run_one_window(
    pw: float,
    out_dir: Path,
    test_csv: Path,
    pinn_model: Path,
    pinn_config: Path,
    ml_model: Path,
    device: str,
    extra_args: Sequence[str],
) -> None:
    cmd: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pe_input_cct_test.py"),
        "--test-csv",
        str(test_csv),
        "--pinn-model",
        str(pinn_model),
        "--pinn-config",
        str(pinn_config),
        "--ml-model",
        str(ml_model),
        "--output-dir",
        str(out_dir),
        "--stability-mode",
        "persistence_fraction",
        "--persistence-window-seconds",
        str(pw),
        "--device",
        device,
    ]
    cmd.extend(extra_args)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--windows",
        type=float,
        nargs="+",
        default=list(DEFAULT_WINDOWS_S),
        help="Persistence window widths in seconds (default: six-point grid).",
    )
    p.add_argument(
        "--base-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/paper_pe_input_cct_sensitivity_pw",
        help="Parent directory for per-window subfolders and summary artifacts.",
    )
    p.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    p.add_argument("--pinn-model", type=Path, default=DEFAULT_PINN_MODEL)
    p.add_argument("--pinn-config", type=Path, default=DEFAULT_PINN_CONFIG)
    p.add_argument("--ml-model", type=Path, default=DEFAULT_ML_MODEL)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a window if pe_input_cct_test_results.json already exists in its folder.",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not run models; only load JSONs, write CSV/LaTeX, and plot.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib figure (CSV and optional LaTeX still written).",
    )
    p.add_argument(
        "--no-latex",
        action="store_true",
        help="Skip writing persistence_window_sensitivity_table.tex and CCT MAE/RMSE fragment.",
    )
    p.add_argument(
        "--paper-figures-dir",
        type=Path,
        default=None,
        help="If set, also copy/write persistence_window_sensitivity.png here (e.g. paper_writing/.../figures/pe_input_cct).",
    )
    p.add_argument(
        "--print-latex-tables",
        action="store_true",
        help="Forward to run_pe_input_cct_test.py for each window (verbose stdout).",
    )
    p.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        metavar="EXTRA",
        help="Extra args after -- forwarded to run_pe_input_cct_test.py.",
    )
    args = p.parse_args()
    base: Path = args.base_output_dir
    windows: List[float] = [float(w) for w in args.windows]

    extra_to_child: List[str] = []
    if args.print_latex_tables:
        extra_to_child.append("--print-latex-tables")
    if args.passthrough:
        pt = list(args.passthrough)
        if pt and pt[0] == "--":
            pt = pt[1:]
        extra_to_child.extend(pt)

    if not args.aggregate_only:
        for path, label in (
            (args.test_csv, "--test-csv"),
            (args.pinn_model, "--pinn-model"),
            (args.pinn_config, "--pinn-config"),
            (args.ml_model, "--ml-model"),
        ):
            if not path.is_file():
                raise SystemExit(f"{label} not found: {path}")
        for w in windows:
            sub = base / persistence_window_dirname(w)
            out_json = sub / "pe_input_cct_test_results.json"
            if args.skip_existing and out_json.is_file():
                print(f"Skip (exists): {out_json}")
                continue
            print(f"Running persistence window {w} s -> {sub}", flush=True)
            _run_one_window(
                w,
                sub,
                args.test_csv,
                args.pinn_model,
                args.pinn_config,
                args.ml_model,
                args.device,
                extra_to_child,
            )

    rows = load_summary_rows(base, windows)
    csv_path = base / "persistence_window_sensitivity_summary.csv"
    write_csv(rows, csv_path)
    print(f"Wrote {csv_path}")

    if not args.no_latex:
        tex_path = base / "persistence_window_sensitivity_table.tex"
        write_latex_table(rows, tex_path)
        print(f"Wrote {tex_path}")
        cct_tex = base / "persistence_window_cct_mae_rmse_table.tex"
        write_latex_cct_mae_rmse_table(rows, cct_tex)
        print(f"Wrote {cct_tex}")

    png_path = base / "persistence_window_sensitivity.png"
    cct_png = base / "persistence_window_cct_mae_rmse.png"
    if not args.no_plot:
        plot_sensitivity(rows, png_path)
        print(f"Wrote {png_path}")
        plot_cct_mae_rmse(rows, cct_png)
        print(f"Wrote {cct_png}")
        if args.paper_figures_dir is not None:
            args.paper_figures_dir.mkdir(parents=True, exist_ok=True)
            dest = args.paper_figures_dir / "persistence_window_sensitivity.png"
            dest.write_bytes(png_path.read_bytes())
            print(f"Wrote {dest}")
            dest_cct = args.paper_figures_dir / "persistence_window_cct_mae_rmse.png"
            dest_cct.write_bytes(cct_png.read_bytes())
            print(f"Wrote {dest_cct}")
            if not args.no_latex:
                dest_tbl = args.paper_figures_dir / "persistence_window_cct_mae_rmse_table.tex"
                dest_tbl.write_text(cct_tex.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"Wrote {dest_tbl}")


if __name__ == "__main__":
    main()
