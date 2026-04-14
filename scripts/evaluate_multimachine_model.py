#!/usr/bin/env python
"""
Evaluate a trained Multimachine PINN on test data (raw multimachine CSV with delta_0..n, Pe_0..n).

Similar to evaluate_model.py for SMIB: load model + scalers, run on test scenarios, report
RMSE/MAE/R² and save plots. Does not modify SMIB evaluation code.

Usage:
    python scripts/evaluate_multimachine_model.py --model-path outputs/models/multimachine/best_model.pth --data-path data/multimachine/kundur/exp_*/parameter_sweep_data_*.csv --num-machines 4
    python scripts/evaluate_multimachine_model.py --model-path .../best_model.pth --data-dir data/multimachine/kundur/exp_20260220_180602 --num-machines 4 --output-dir results/multimachine_eval
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pinn.multimachine import MultimachinePINN
from scripts.core.utils import generate_timestamped_filename
from utils.normalization import normalize_value, denormalize_array


def load_checkpoint(model_path: Path, num_machines: int, hidden_dims=None):
    """Load MultimachinePINN and scalers from trainer checkpoint."""
    model_path = Path(model_path)
    if hidden_dims is None:
        hidden_dims = [128, 128, 128, 64]
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    scalers = ckpt.get("scalers", {})
    model = MultimachinePINN(
        num_machines=num_machines,
        input_dim_per_machine=9,
        hidden_dims=hidden_dims,
        activation="tanh",
        dropout=0.0,
        use_coi=True,
        use_pe_as_input=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model, scalers


def _prepare_scenario_tensors(scenario_data, scalers, num_machines, device):
    """Build (T, num_machines, 9) input and (T, num_machines, 2) true output for one scenario."""
    scenario_data = scenario_data.sort_values("time")
    tc_col = "param_tc" if "param_tc" in scenario_data.columns else "tc"
    row0 = scenario_data.iloc[0]
    tf_raw = float(row0.get("tf", row0.get("param_tf", 1.0)))
    tc_raw = float(row0.get(tc_col, 1.2))
    tf_norm = normalize_value(tf_raw, scalers["tf"]) if scalers.get("tf") is not None else 0.0
    tc_norm = normalize_value(tc_raw, scalers["tc"]) if scalers.get("tc") is not None else 0.0
    has_per_machine = any(f"delta_{i}" in scenario_data.columns for i in range(num_machines))

    inputs, truths = [], []
    for _, row in scenario_data.iterrows():
        t_norm = (
            normalize_value(float(row["time"]), scalers["time"]) if scalers.get("time") else 0.0
        )
        machine_in, machine_out = [], []
        for i in range(num_machines):
            if has_per_machine:
                d0 = float(row0.get(f"delta0_{i}", row0.get("delta0", 0.0)))
                o0 = float(row0.get(f"omega0_{i}", row0.get("omega0", 1.0)))
                H = float(row0.get(f"H_{i}", row0.get("param_H", row0.get("H", 3.0))))
                D = float(row0.get(f"D_{i}", row0.get("param_D", row0.get("D", 1.0))))
                Pm = float(row0.get(f"Pm_{i}", row0.get("param_Pm", row0.get("Pm", 0.7))))
                Pe = float(row.get(f"Pe_{i}", row.get("Pe", 0.0)))
                d = float(row.get(f"delta_{i}", row.get("delta", 0.0)))
                o = float(row.get(f"omega_{i}", row.get("omega", 1.0)))
            else:
                d0 = float(row0.get("delta0", 0.0))
                o0 = float(row0.get("omega0", 1.0))
                H = float(row0.get("param_H", row0.get("H", 3.0)))
                D = float(row0.get("param_D", row0.get("D", 1.0)))
                Pm = float(row0.get("param_Pm", row0.get("Pm", 0.7)))
                Pe = float(row.get(f"Pe_{i}", row.get("Pe", 0.0)))
                d = float(row.get("delta", 0.0))
                o = float(row.get("omega", 1.0))
            d0_n = normalize_value(d0, scalers["delta0"]) if "delta0" in scalers else 0.0
            o0_n = normalize_value(o0, scalers["omega0"]) if "omega0" in scalers else 0.0
            H_n = (
                normalize_value(H, scalers.get("H", scalers.get("param_H")))
                if scalers.get("H")
                else 0.0
            )
            D_n = (
                normalize_value(D, scalers.get("D", scalers.get("param_D")))
                if scalers.get("D")
                else 0.0
            )
            Pm_n = (
                normalize_value(Pm, scalers.get("Pm", scalers.get("param_Pm")))
                if scalers.get("Pm")
                else 0.0
            )
            Pe_n = (
                normalize_value(Pe, scalers.get(f"Pe_{i}", scalers.get("Pe")))
                if scalers.get(f"Pe_{i}", scalers.get("Pe"))
                else 0.0
            )
            machine_in.append([t_norm, d0_n, o0_n, H_n, D_n, Pm_n, Pe_n, tf_norm, tc_norm])
            d_n = (
                normalize_value(d, scalers.get(f"delta_{i}", scalers.get("delta")))
                if scalers.get(f"delta_{i}", scalers.get("delta"))
                else 0.0
            )
            o_n = (
                normalize_value(o, scalers.get(f"omega_{i}", scalers.get("omega")))
                if scalers.get(f"omega_{i}", scalers.get("omega"))
                else 0.0
            )
            machine_out.append([d_n, o_n])
        inputs.append(machine_in)
        truths.append(machine_out)
    X = torch.tensor(inputs, dtype=torch.float32, device=device)
    y = torch.tensor(truths, dtype=torch.float32, device=device)
    return X, y, scenario_data["time"].values, has_per_machine


def _inertia_weights(first_row, num_machines: int) -> np.ndarray:
    """M (inertia) per machine for COI. GENCLS: M = 2*H. Prefer param_M (uniform) or per-machine."""
    M = np.ones(num_machines, dtype=np.float64)
    if "param_M" in first_row and pd.notna(first_row.get("param_M")):
        M[:] = float(first_row["param_M"])
        return M
    H = first_row.get("param_H", first_row.get("H", 6.0))
    if pd.isna(H):
        return M
    # M = 2*H for GENCLS (omega in pu, time in s)
    H_val = float(H)
    for i in range(num_machines):
        Hi = first_row.get(f"H_{i}", first_row.get("param_H", H_val))
        M[i] = 2.0 * (float(Hi) if pd.notna(Hi) else H_val)
    return M


def _delta_to_coi_relative_deg(delta_rad: np.ndarray, M: np.ndarray) -> np.ndarray:
    """(T, n_mach) in rad -> (T, n_mach) in deg, COI-relative: delta_i - delta_COI."""
    # delta_COI = sum(M_i * delta_i) / sum(M_i)
    M = np.asarray(M, dtype=np.float64)
    if M.size != delta_rad.shape[1]:
        M = np.broadcast_to(M.mean(), delta_rad.shape[1])
    coi = (delta_rad * M).sum(axis=1) / M.sum()
    delta_rel_rad = delta_rad - coi[:, np.newaxis]
    return np.rad2deg(delta_rel_rad)


def evaluate_scenario(model, scenario_data, scalers, num_machines, device):
    """Run model on one scenario; return pred and true in physical units, and time."""
    X, y_norm, time_vals, has_per_machine = _prepare_scenario_tensors(
        scenario_data, scalers, num_machines, device
    )
    with torch.no_grad():
        delta_p, omega_p = model(X)
    delta_pred_n = delta_p.cpu().numpy()
    omega_pred_n = omega_p.cpu().numpy()
    y_true_n = y_norm.cpu().numpy()
    # Denormalize for metrics (use first machine's scaler if per-machine not present)
    delta_true_phys = np.zeros_like(delta_pred_n)
    omega_true_phys = np.zeros_like(omega_pred_n)
    delta_pred_phys = np.zeros_like(delta_pred_n)
    omega_pred_phys = np.zeros_like(omega_pred_n)
    for i in range(num_machines):
        d_sc = scalers.get(f"delta_{i}", scalers.get("delta"))
        o_sc = scalers.get(f"omega_{i}", scalers.get("omega"))
        if d_sc:
            delta_true_phys[:, i] = denormalize_array(y_true_n[:, i, 0], d_sc)
            delta_pred_phys[:, i] = denormalize_array(delta_pred_n[:, i], d_sc)
        if o_sc:
            omega_true_phys[:, i] = denormalize_array(y_true_n[:, i, 1], o_sc)
            omega_pred_phys[:, i] = denormalize_array(omega_pred_n[:, i], o_sc)
    return {
        "time": time_vals,
        "delta_true": delta_true_phys,
        "omega_true": omega_true_phys,
        "delta_pred": delta_pred_phys,
        "omega_pred": omega_pred_phys,
    }


# Match analysis script: gen colors, ±180 ref, fault on/clear, title id | Stable/Unstable | CCT | H | D | tc | load
_GEN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _plot_eval_rotor_angle_stable_unstable(
    results: list,
    num_machines: int,
    out_dir: Path,
    n_per_type: int = 5,
    random_state: int = 42,
    dpi: int = 300,
) -> None:
    """Two figures: stable and unstable sample trajectories (rotor-angle only, delta - delta_COI deg), analysis-style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(random_state)
    stable = [r for r in results if r.get("is_stable", False)]
    unstable = [r for r in results if not r.get("is_stable", False)]
    n_s = min(n_per_type, len(stable))
    n_u = min(n_per_type, len(unstable))
    stable_ids = list(rng.choice(len(stable), size=n_s, replace=False) if n_s else [])
    unstable_ids = list(rng.choice(len(unstable), size=n_u, replace=False) if n_u else [])

    def _draw_one(subs: list, label_type: str, suptitle: str, filename: str) -> None:
        if not subs:
            return
        n_rows = len(subs)
        fig, axes = plt.subplots(n_rows, 1, figsize=(6, 2.2 * n_rows))
        if n_rows == 1:
            axes = [axes]
        for idx, r in enumerate(subs):
            d = r["data"]
            t = d["time"]
            meta = r.get("title_meta", {})
            ax = axes[idx]
            for m in range(num_machines):
                c = _GEN_COLORS[m % len(_GEN_COLORS)]
                ax.plot(
                    t, d["delta_rel_true_deg"][:, m], "-", color=c, lw=1.5, label=f"True gen{m+1}"
                )
                ax.plot(
                    t, d["delta_rel_pred_deg"][:, m], "--", color=c, lw=1.2, label=f"Pred gen{m+1}"
                )
            ax.axhline(180, color="gray", linestyle="-.", linewidth=1, alpha=0.8)
            ax.axhline(-180, color="gray", linestyle="-.", linewidth=1, alpha=0.8)
            fault_start = meta.get("tf")
            fault_clear = meta.get("tc")
            fault_bus = meta.get("param_fault_bus")
            loc_str = f" (bus {fault_bus})" if fault_bus is not None else ""
            if fault_start is not None:
                ax.axvline(
                    fault_start,
                    color="#c0392b",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                    label=f"fault on{loc_str}",
                )
            if fault_clear is not None:
                ax.axvline(
                    fault_clear,
                    color="#27ae60",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.9,
                    label=f"fault clear{loc_str}",
                )
            ax.set_ylabel(r"$\delta - \delta_{\mathrm{COI}}$ (deg)")
            ax.set_xlabel("Time (s)" if idx == n_rows - 1 else "")
            ax.set_ylim(-360, 360)
            ax.set_xlim(0, max(t) if len(t) else 5)
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize=7,
                framealpha=0.95,
                borderaxespad=0,
            )
            parts = [f"id={r['scenario_id']}", label_type]
            if meta.get("cct_s") is not None:
                parts.append(f"CCT={meta['cct_s']:.3f}s")
            if meta.get("param_H") is not None:
                parts.append(f"H={float(meta['param_H']):.2f}")
            if meta.get("param_D") is not None:
                parts.append(f"D={float(meta['param_D']):.2f}")
            if meta.get("tc") is not None:
                parts.append(f"tc={float(meta['tc']):.3f}s")
            if meta.get("param_load") is not None:
                parts.append(f"load={float(meta['param_load']):.2f}")
            elif meta.get("param_Pm") is not None:
                parts.append(f"Pm={float(meta['param_Pm']):.2f}")
            ax.set_title("  |  ".join(parts), fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--")
        fig.suptitle(suptitle, fontsize=10)
        fig.tight_layout(rect=[0, 0, 0.78, 0.97])
        fig.savefig(out_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close()

    subs_s = [stable[i] for i in stable_ids] if stable_ids else []
    subs_u = [unstable[i] for i in unstable_ids] if unstable_ids else []
    _draw_one(
        subs_s,
        "Stable",
        "Time-domain: relative rotor angle (eval) — stable sample trajectories",
        "multimachine_eval_stable.png",
    )
    _draw_one(
        subs_u,
        "Unstable",
        "Time-domain: relative rotor angle (eval) — unstable sample trajectories",
        "multimachine_eval_unstable.png",
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multimachine PINN on test data")
    parser.add_argument("--model-path", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument(
        "--data-path", type=str, default=None, help="Path to parameter_sweep_data_*.csv"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing parameter_sweep_data_*.csv (used if --data-path not set)",
    )
    parser.add_argument(
        "--num-machines", type=int, required=True, help="Number of machines (e.g. 4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/multimachine_eval",
        help="Output for metrics and plots",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of scenarios for test (default 0.15)",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=None,
        help="Max test scenarios to evaluate (default: all test)",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for trajectory overlay figures (default 300, publication)",
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    if args.data_path:
        data_path = Path(args.data_path)
        if "*" in str(data_path):
            data_path = max(data_path.parent.glob(data_path.name), key=lambda p: p.stat().st_mtime)
    else:
        data_dir = PROJECT_ROOT / (args.data_dir or "data/multimachine/kundur")
        candidates = list(data_dir.glob("parameter_sweep_data_*.csv"))
        for d in data_dir.iterdir():
            if d.is_dir() and d.name.startswith("exp_"):
                candidates.extend(d.glob("parameter_sweep_data_*.csv"))
        if not candidates:
            print("No parameter_sweep_data_*.csv found. Use --data-path.")
            sys.exit(1)
        data_path = max(candidates, key=lambda p: p.stat().st_mtime)
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        sys.exit(1)

    print("=" * 70)
    print("MULTIMACHINE MODEL EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Num machines: {args.num_machines}")

    model, scalers = load_checkpoint(model_path, args.num_machines)
    model = model.to(device)

    df = pd.read_csv(data_path)
    if "time" not in df.columns and "time_s" in df.columns:
        df["time"] = df["time_s"]
    if "scenario_id" not in df.columns:
        print("CSV must contain scenario_id")
        sys.exit(1)
    scenario_ids = df["scenario_id"].unique()
    n_test = max(1, int(len(scenario_ids) * args.test_ratio))
    test_ids = scenario_ids[-n_test:]
    if args.n_scenarios is not None:
        test_ids = test_ids[: args.n_scenarios]
    print(f"Test scenarios: {len(test_ids)}")

    results = []
    all_delta_rmse = []
    for sid in test_ids:
        sc = df[df["scenario_id"] == sid]
        out = evaluate_scenario(model, sc, scalers, args.num_machines, device)
        delta_rmse = np.sqrt(np.mean((out["delta_pred"] - out["delta_true"]) ** 2))
        all_delta_rmse.append(delta_rmse)
        # Per-generator RMSE and MAE (rad) for publication summary
        per_gen_rmse_rad = []
        per_gen_mae_rad = []
        for i in range(args.num_machines):
            err = out["delta_pred"][:, i] - out["delta_true"][:, i]
            per_gen_rmse_rad.append(np.sqrt(np.mean(err**2)))
            per_gen_mae_rad.append(np.mean(np.abs(err)))
        row0 = sc.iloc[0]
        M = _inertia_weights(row0, args.num_machines)
        delta_rel_true_deg = _delta_to_coi_relative_deg(out["delta_true"], M)
        delta_rel_pred_deg = _delta_to_coi_relative_deg(out["delta_pred"], M)
        out["delta_rel_true_deg"] = delta_rel_true_deg
        out["delta_rel_pred_deg"] = delta_rel_pred_deg
        # is_stable: from CSV if present, else from 180 deg criterion on true trajectory
        if "is_stable" in sc.columns and pd.notna(sc["is_stable"].iloc[0]):
            is_stable = bool(sc["is_stable"].iloc[0])
        else:
            max_deg = np.abs(delta_rel_true_deg).max()
            is_stable = max_deg < 180.0
        # PINN-predicted stability: 180 deg rule on predicted trajectory
        max_pred_deg = np.abs(delta_rel_pred_deg).max()
        is_stable_pred = max_pred_deg < 180.0
        cct_s = None
        if "param_cct_absolute" in row0 and pd.notna(row0.get("param_cct_absolute")):
            cct_s = float(row0["param_cct_absolute"])
        elif "param_cct_duration" in row0 and pd.notna(row0.get("param_cct_duration")):
            cct_s = float(row0["param_cct_duration"])
        tc_val = (
            float(row0.get("param_tc", row0.get("tc", 1.2)))
            if pd.notna(row0.get("param_tc", row0.get("tc")))
            else None
        )
        param_H = row0.get("param_H")
        param_D = row0.get("param_D")
        param_load = row0.get("param_load")
        param_Pm = row0.get("param_Pm")
        title_meta = {
            "tf": float(row0.get("tf", row0.get("param_tf", 1.0)))
            if pd.notna(row0.get("tf", row0.get("param_tf")))
            else None,
            "tc": tc_val,
            "param_H": param_H,
            "param_D": param_D,
            "param_load": param_load,
            "param_Pm": param_Pm,
            "param_fault_bus": int(float(row0["param_fault_bus"]))
            if "param_fault_bus" in row0 and pd.notna(row0.get("param_fault_bus"))
            else None,
            "cct_s": cct_s,
        }
        rec = {
            "scenario_id": sid,
            "delta_rmse_rad": delta_rmse,
            "omega_rmse_pu": np.sqrt(np.mean((out["omega_pred"] - out["omega_true"]) ** 2)),
            "n_points": len(out["time"]),
            "is_stable": is_stable,
            "is_stable_pred": is_stable_pred,
            "title_meta": title_meta,
        }
        for i in range(args.num_machines):
            rec[f"delta_rmse_gen{i}_rad"] = per_gen_rmse_rad[i]
            rec[f"delta_mae_gen{i}_rad"] = per_gen_mae_rad[i]
        rec["param_tc"] = tc_val
        rec["param_H"] = float(param_H) if param_H is not None and pd.notna(param_H) else None
        rec["param_D"] = float(param_D) if param_D is not None and pd.notna(param_D) else None
        rec["param_load"] = (
            float(param_load) if param_load is not None and pd.notna(param_load) else None
        )
        rec["param_Pm"] = float(param_Pm) if param_Pm is not None and pd.notna(param_Pm) else None
        results.append(rec)
        results[-1]["data"] = out

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ("data", "title_meta")} for r in results]
    )
    metrics_path = out_dir / generate_timestamped_filename("multimachine_eval_metrics", "csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 70)
    print("RESULTS (rotor angle focused)")
    print("=" * 70)
    print(f"Delta RMSE (rad): {np.mean(all_delta_rmse):.6f} +/- {np.std(all_delta_rmse):.6f}")
    print(f"Metrics saved: {metrics_path}")

    # Rotor-angle-only plots: separate stable and unstable (analysis-style)
    try:
        _plot_eval_rotor_angle_stable_unstable(
            results,
            args.num_machines,
            out_dir,
            n_per_type=5,
            random_state=42,
            dpi=args.dpi,
        )
        print("Plots saved: multimachine_eval_stable.png, multimachine_eval_unstable.png")
    except Exception as e:
        print(f"Plots skipped: {e}")

    # Publication figures from metrics CSV (error distribution, error vs param, per-gen summary, stability)
    try:
        from scripts.plot_multimachine_publication_figures import (
            plot_error_distribution,
            plot_error_vs_parameter,
            plot_per_generator_summary,
            plot_stability_agreement,
        )

        plot_error_distribution(metrics_df, out_dir, dpi=args.dpi)
        plot_error_vs_parameter(metrics_df, out_dir, dpi=args.dpi)
        plot_per_generator_summary(metrics_df, args.num_machines, out_dir, dpi=args.dpi)
        plot_stability_agreement(metrics_df, out_dir, dpi=args.dpi)
        print(
            "Publication figures saved: multimachine_publication_*.png, *_summary_metrics.csv, *_stability_metrics.csv"
        )
    except Exception as e:
        print(f"Publication figures skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
