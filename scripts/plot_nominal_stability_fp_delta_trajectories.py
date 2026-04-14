#!/usr/bin/env python
"""
Plot rotor-angle trajectories for selected test scenarios (ANDES vs PINN vs ML).

Uses the same forward maps and CSV alignment as ``compare_models.py`` (``pe_direct_7``).
Default scenario IDs are the nominal-clearing false positives from
``run_pe_input_cct_test.py`` with per-scenario CSV time + terminal |δ| < π
(archived JSON: ``outputs/paper_pe_input_cct_scenario_grid_20260411/``): 4, 34, 113.

Example (repo root)::

    python scripts/plot_nominal_stability_fp_delta_trajectories.py ^
      --output-dir outputs/nominal_fp_delta_plots_20260411
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from scripts.compare_models import (  # noqa: E402
    load_pinn_model,
    predict_scenario_ml_baseline,
    predict_scenario_pinn,
)
from scripts.evaluate_ml_baseline import load_ml_baseline_model  # noqa: E402

RAD2DEG = 180.0 / np.pi


def _device_arg(device: str) -> torch.device:
    if device == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        d = device
    return torch.device(d)


def plot_scenarios(
    test_csv: Path,
    scenario_ids: Sequence[int],
    pinn_model_path: Path,
    pinn_config: Path | None,
    ml_model_path: Path,
    output_dir: Path,
    device: str,
    dpi: int,
    save_pdf: bool,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(test_csv)
    dev = _device_arg(device)

    pinn_model, pinn_scalers, pinn_im = load_pinn_model(
        pinn_model_path, config_path=pinn_config, device=str(dev)
    )
    ml_model, ml_scalers, ml_im = load_ml_baseline_model(ml_model_path, device=str(dev))

    written: List[Path] = []
    n = len(scenario_ids)
    fig, axes = plt.subplots(n, 1, figsize=(6.5, 3.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, sid in zip(axes, scenario_ids):
        sdf = df[df["scenario_id"] == int(sid)].sort_values("time")
        if len(sdf) < 10:
            ax.set_title(f"Scenario {sid}: insufficient rows ({len(sdf)})")
            continue

        t = sdf["time"].values
        delta_true = sdf["delta"].values
        row0 = sdf.iloc[0]
        tf = float(row0.get("tf", 1.0))
        tc = float(row0.get("tc", row0.get("param_tc", 1.2)))
        is_stable = row0.get("is_stable", row0.get("is_stable_from_cct", None))

        time_ml, delta_ml, _ = predict_scenario_ml_baseline(ml_model, sdf, ml_scalers, ml_im, dev)
        time_pinn, delta_pinn, _ = predict_scenario_pinn(
            pinn_model, sdf, pinn_scalers, pinn_im, dev
        )

        ax.plot(t, RAD2DEG * delta_true, color="black", linewidth=1.6, label="ANDES (truth)")
        ax.plot(
            time_ml, RAD2DEG * delta_ml, color="C0", linewidth=1.2, linestyle="--", label="Std NN*"
        )
        ax.plot(
            time_pinn,
            RAD2DEG * delta_pinn,
            color="C3",
            linewidth=1.2,
            linestyle=":",
            label="PINN*",
        )
        ax.axhline(180.0, color="gray", linewidth=0.8, linestyle="-", alpha=0.7)
        ax.axhline(-180.0, color="gray", linewidth=0.8, linestyle="-", alpha=0.7)
        ax.axvline(
            tf,
            color="orange",
            linewidth=0.9,
            linestyle=":",
            alpha=0.85,
            label="Fault end $t_f$",
        )
        ax.axvline(
            tc,
            color="red",
            linewidth=0.9,
            linestyle=":",
            alpha=0.85,
            label="Clear $t_c$",
        )

        stab_s = "stable" if bool(is_stable) else "unstable" if is_stable is not None else "?"
        ax.set_title(f"Scenario {sid} (ref. {stab_s}); " f"$t_\\mathrm{{end}}={t[-1]:.3f}$\\,s")
        ax.set_ylabel(r"$\delta$ (deg)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        r"Nominal-clearing false positives: terminal $|\delta| < \pi$ "
        r"on surrogate vs ANDES truth",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    png = output_dir / "nominal_fp_scenarios_delta_deg.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    written.append(png)
    if save_pdf:
        pdf = output_dir / "nominal_fp_scenarios_delta_deg.pdf"
        fig.savefig(pdf, bbox_inches="tight")
        written.append(pdf)
    plt.close(fig)
    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot ANDES vs PINN vs ML δ(t) for nominal-stability FP scenarios."
    )
    p.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/processed/exp_20260211_190612/test_data_20260211_190612.csv"),
    )
    p.add_argument(
        "--scenarios",
        type=int,
        nargs="+",
        default=[4, 34, 113],
        help="Scenario IDs (default: 4 34 113)",
    )
    p.add_argument(
        "--pinn-model",
        type=Path,
        default=Path(
            "outputs/expt_residual_backbone_retrain_20260407/pinn_nores_lp05/pinn/"
            "best_model_20260407_172450.pth"
        ),
    )
    p.add_argument(
        "--pinn-config",
        type=Path,
        default=Path("outputs/expt_residual_backbone_retrain_20260407/pinn_nores_lp05/config.yaml"),
    )
    p.add_argument(
        "--ml-model",
        type=Path,
        default=Path(
            "outputs/campaign_indep_ml/ml_uw2_ow50/ml_baseline/standard_nn/model/model.pth"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/nominal_fp_delta_trajectories"),
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--save-pdf", action="store_true")
    args = p.parse_args()

    paths = plot_scenarios(
        test_csv=args.test_csv,
        scenario_ids=args.scenarios,
        pinn_model_path=args.pinn_model,
        pinn_config=args.pinn_config if args.pinn_config.is_file() else None,
        ml_model_path=args.ml_model,
        output_dir=args.output_dir,
        device=args.device,
        dpi=args.dpi,
        save_pdf=args.save_pdf,
    )
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
