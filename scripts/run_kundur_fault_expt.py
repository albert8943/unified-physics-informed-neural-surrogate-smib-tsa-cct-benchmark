"""
Three-phase fault experiment on Kundur system (publication-level).

Applies a balanced three-phase fault at a chosen bus for a specified duration,
then clears the fault and runs time-domain simulation. Saves power flow and TDS
results (including delta, omega, Pe) and generates the same publication figures
as the no-fault run, so you can compare fault vs no-fault trajectories.

Methodology (for journal):
  - Pre-fault: power flow at base load (Kundur raw + GENCLS dyr).
  - Fault: three-phase (balanced) bus fault at t = T_FAULT_START (s).
  - Clearing: fault removed at t = T_FAULT_CLEAR (s). Duration = T_FAULT_CLEAR - T_FAULT_START.
  - Post-fault: system evolves with no further switching (toggles disabled).
  - Fault impedance: xf (pu), rf = 0 for bolted fault.

Output: outputs/kundur_simulations/exp_YYYYMMDD_HHMMSS_fault/
  - power_flow_*.csv, tds_trajectories.csv, run_info.txt (includes fault params)
  - figures/: pf_voltage_profile_and_angle, pf_load_power_per_bus,
    tds_Pe_trajectories, tds_delta_trajectories, tds_omega_trajectories

Usage:
  python scripts/run_kundur_fault_expt.py
  python scripts/run_kundur_fault_expt.py --fault-bus 8 --clear 1.2 --duration 0.15
  python scripts/run_kundur_fault_expt.py --config configs/publication/kundur_2area.yaml --alpha 0.9
  python scripts/run_kundur_fault_expt.py --config configs/publication/kundur_2area.yaml --all-alphas  # one run per alpha in config alpha_range
  python scripts/run_kundur_fault_expt.py --config configs/publication/kundur_2area.yaml --all-alphas --find-cct-per-alpha  # find CCT per alpha, run stable/unstable at CCT±offset, generate delta-only figures
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import andes
import numpy as np

# ----- Config (override with CLI) -----
QUIET = True
ANDES_STREAM_LEVEL = 40 if QUIET else 20
GENERATE_FIGURES = True

# Fault parameters (publication-style: clearly documented)
T_FAULT_START = 1.0  # s, fault application time
T_FAULT_CLEAR = 1.15  # s, fault clearing time (duration = 0.15 s)
FAULT_BUS = 7  # ANDES bus index (Kundur: 7 = bus name 3, 230 kV; 8 = bus name 13)
FAULT_XF = 0.0001  # pu, fault reactance (≈ bolted three-phase)
FAULT_RF = 0.0  # pu, fault resistance
TDS_TF = 5.0  # s, simulation end time
TDS_TSTEP = 0.01  # s
ALPHA_DEFAULT = 1.0  # load scale factor: P' = alpha*P_base, Q' = alpha*Q_base (1 = base case)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "kundur_simulations"


def main():
    parser = argparse.ArgumentParser(
        description="Kundur three-phase fault experiment (publication-level)."
    )
    parser.add_argument(
        "--fault-bus",
        type=int,
        default=FAULT_BUS,
        help=f"Bus index for fault location (default: {FAULT_BUS})",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=T_FAULT_START,
        help=f"Fault application time in seconds (default: {T_FAULT_START})",
    )
    parser.add_argument(
        "--clear",
        type=float,
        default=None,
        help="Fault clearing time in seconds (default: start + 0.15)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.15,
        help="Fault duration in seconds (used if --clear not set)",
    )
    parser.add_argument(
        "--sim-time",
        type=float,
        default=TDS_TF,
        help=f"TDS end time in seconds (default: {TDS_TF})",
    )
    parser.add_argument("--no-figures", action="store_true", help="Do not generate PNG figures")
    parser.add_argument("--verbose", action="store_true", help="Show ANDES and script output")
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/kundur_simulations/exp_YYYYMMDD_HHMMSS_fault)",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Save only TDS and run_info (no power flow CSVs); use for stable/unstable runs to avoid duplicates",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA_DEFAULT,
        help=f"Load scale factor: all PQ loads scaled as P'=alpha*P0, Q'=alpha*Q0 (default: {ALPHA_DEFAULT})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config (e.g. configs/publication/kundur_2area.yaml) to read fault, simulation_time, and alpha_range; CLI overrides.",
    )
    parser.add_argument(
        "--all-alphas",
        action="store_true",
        help="When using --config: run one experiment per alpha in config's alpha_range [min, max, n]; each saved to exp_*_fault_alpha_<value>.",
    )
    parser.add_argument(
        "--find-cct-per-alpha",
        action="store_true",
        help="When using --all-alphas: find CCT at each alpha, run stable (CCT-offset) and unstable (CCT+offset), generate tds_delta_trajectories_stable/unstable.png.",
    )
    parser.add_argument(
        "--cct-offset",
        type=float,
        default=0.02,
        help="Seconds from CCT for stable (CCT-offset) and unstable (CCT+offset) when using --find-cct-per-alpha (default: 0.02).",
    )
    args = parser.parse_args()

    fault_start = args.start
    fault_bus = args.fault_bus
    tf_sec = args.sim_time
    alpha_list_from_config = None
    if args.config:
        config_path = (
            Path(args.config).resolve()
            if Path(args.config).is_absolute()
            else PROJECT_ROOT / args.config
        )
        if config_path.exists():
            try:
                sys.path.insert(0, str(PROJECT_ROOT))
                from scripts.core.utils import load_config

                cfg = load_config(config_path)
                gen = cfg.get("data", {}).get("generation", {})
                f = gen.get("fault", {})
                if f:
                    fault_start = float(f.get("start_time", fault_start))
                    fault_bus = int(f.get("bus", fault_bus))
                tf_sec = float(gen.get("simulation_time", tf_sec))
                # alpha_range: [min, max] or [min, max, n] -> list of alpha values
                ar = gen.get("alpha_range")
                if isinstance(ar, (list, tuple)) and len(ar) >= 2:
                    a_min, a_max = float(ar[0]), float(ar[1])
                    n = int(ar[2]) if len(ar) >= 3 else 1
                    if n <= 1:
                        alpha_list_from_config = [
                            a_min if a_min == a_max else (a_min + a_max) / 2.0
                        ]
                    else:
                        # Round to 3 decimals so e.g. 0.9 is exact and folder names are alpha_0_900
                        alpha_list_from_config = [
                            round(x, 3) for x in np.linspace(a_min, a_max, n).tolist()
                        ]
            except Exception:
                pass

    # --all-alphas: run this script once per alpha (subprocess), then exit
    if (
        args.all_alphas
        and args.config
        and alpha_list_from_config
        and len(alpha_list_from_config) > 1
    ):
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmd_base = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(
                Path(args.config).resolve()
                if Path(args.config).is_absolute()
                else PROJECT_ROOT / args.config
            ),
        ]
        if args.no_figures:
            cmd_base.append("--no-figures")
        if args.minimal:
            cmd_base.append("--minimal")
        if args.verbose:
            cmd_base.append("--verbose")
        if args.find_cct_per_alpha:
            sys.path.insert(0, str(PROJECT_ROOT))
            from scripts.find_kundur_cct import find_cct_bisection
            from scripts.plot_kundur_publication_figures import plot_tds_delta_trajectories

            offset_s = args.cct_offset
        for a in alpha_list_from_config:
            # Use rounded alpha so folder name is e.g. alpha_0_900 not alpha_0_899
            a_label = round(a, 3)
            exp_name = f"exp_{run_ts}_fault_alpha_{a_label:.3f}".replace(".", "_")
            exp_dir = Path(OUTPUT_BASE / exp_name).resolve()
            exp_dir.mkdir(parents=True, exist_ok=True)
            if args.find_cct_per_alpha:
                if not args.verbose:
                    print(f"Alpha={a:.3f}: finding CCT ...", end=" ", flush=True)
                cct_s, _unc, _n, reason = find_cct_bisection(
                    fault_start=fault_start,
                    fault_bus=fault_bus,
                    sim_time=tf_sec,
                    duration_min=0.05,
                    duration_max=1.0,
                    tol=0.002,
                    angle_deg=180.0,
                    alpha=a,
                    verbose=args.verbose,
                )
                if cct_s is None:
                    print(f"Alpha={a:.3f}: CCT not found ({reason}). Skipping.")
                    try:
                        (exp_dir / "cct_not_found.txt").write_text(
                            f"alpha={a_label}\nCCT not found: {reason}\n", encoding="utf-8"
                        )
                    except Exception:
                        pass
                    continue
                if not args.verbose:
                    print(f"CCT={cct_s:.3f} s", end=" ", flush=True)
                stable_dir = exp_dir / "stable"
                unstable_dir = exp_dir / "unstable"
                stable_dir.mkdir(parents=True, exist_ok=True)
                unstable_dir.mkdir(parents=True, exist_ok=True)
                fault_clear_stable = fault_start + (cct_s - offset_s)
                fault_clear_unstable = fault_start + (cct_s + offset_s)
                run_cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(
                        Path(args.config).resolve()
                        if Path(args.config).is_absolute()
                        else PROJECT_ROOT / args.config
                    ),
                    "--alpha",
                    str(a),
                    "--sim-time",
                    str(tf_sec),
                    "--no-figures",
                    "--minimal",
                ]
                ret_s = subprocess.run(
                    run_cmd
                    + [
                        "--start",
                        str(fault_start),
                        "--clear",
                        str(fault_clear_stable),
                        "--exp-dir",
                        str(stable_dir),
                    ],
                    cwd=str(PROJECT_ROOT),
                    capture_output=not args.verbose,
                )
                if ret_s.returncode != 0:
                    print(f"Alpha={a:.3f}: stable run failed.")
                    sys.exit(ret_s.returncode)
                ret_u = subprocess.run(
                    run_cmd
                    + [
                        "--start",
                        str(fault_start),
                        "--clear",
                        str(fault_clear_unstable),
                        "--exp-dir",
                        str(unstable_dir),
                    ],
                    cwd=str(PROJECT_ROOT),
                    capture_output=not args.verbose,
                )
                if ret_u.returncode != 0:
                    print(f"Alpha={a:.3f}: unstable run failed.")
                    sys.exit(ret_u.returncode)
                out_figures = exp_dir / "figures"
                out_figures.mkdir(parents=True, exist_ok=True)
                plot_tds_delta_trajectories(
                    stable_dir, out_figures, cct_s=cct_s, filename_suffix="stable"
                )
                plot_tds_delta_trajectories(
                    unstable_dir, out_figures, cct_s=cct_s, filename_suffix="unstable"
                )
                if not args.verbose:
                    print("stable/unstable figures saved.", flush=True)
            else:
                cmd = cmd_base + ["--alpha", str(a), "--exp-dir", str(exp_dir)]
                ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
                if ret.returncode != 0:
                    sys.exit(ret.returncode)
        print(f"All-alphas done: {len(alpha_list_from_config)} runs under {OUTPUT_BASE}")
        return 0
    fault_clear = args.clear if args.clear is not None else fault_start + args.duration
    if fault_clear <= fault_start:
        print("Error: clearing time must be > start time.")
        sys.exit(1)
    quiet = not args.verbose
    gen_figures = not args.no_figures

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.exp_dir is not None:
        exp_dir = Path(args.exp_dir).resolve()
    else:
        exp_dir = OUTPUT_BASE / f"exp_{run_ts}_fault"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Single run: use config's first alpha if --config was given and --alpha not overridden
    if alpha_list_from_config is not None and args.alpha == ALPHA_DEFAULT:
        alpha = alpha_list_from_config[0]
    else:
        alpha = args.alpha

    case_path = andes.get_case("kundur/kundur.raw")
    dyr_path = andes.get_case("kundur/kundur_gencls.dyr")

    if quiet:
        andes.config_logger(stream_level=ANDES_STREAM_LEVEL)

    # Load without setup so we can add Fault and scale loads before setup()
    ss = andes.load(
        case_path,
        addfile=dyr_path,
        setup=False,
        default_config=True,
        no_output=quiet,
    )

    # Scale all PQ loads by alpha (before setup) so trajectories reflect different load levels
    if alpha != 1.0 and getattr(ss, "PQ", None) is not None and ss.PQ.n > 0:
        try:
            for i in range(ss.PQ.n):
                base_p = (
                    float(ss.PQ.p0.v[i]) if hasattr(ss.PQ.p0, "v") and i < len(ss.PQ.p0.v) else 0.5
                )
                base_q = (
                    float(ss.PQ.q0.v[i]) if hasattr(ss.PQ.q0, "v") and i < len(ss.PQ.q0.v) else 0.2
                )
                new_p = alpha * base_p
                new_q = alpha * base_q
                lid = getattr(ss.PQ.idx, "v", None)
                load_id = (
                    lid[i]
                    if lid is not None and hasattr(lid, "__getitem__") and i < len(lid)
                    else i
                )
                if hasattr(ss.PQ, "alter"):
                    ss.PQ.alter("p0", load_id, new_p)
                    ss.PQ.alter("q0", load_id, new_q)
                else:
                    ss.PQ.p0.v[i] = new_p
                    if hasattr(ss.PQ.q0, "v"):
                        ss.PQ.q0.v[i] = new_q
            if not quiet:
                print(f"Scaled {ss.PQ.n} PQ load(s) by alpha={alpha:.4f}")
        except Exception as e:
            if not quiet:
                print(f"[Warning] Load scaling by alpha: {e}")

    # Add three-phase fault if case has no fault (raw+dyr typically has none)
    fault_added = False
    if getattr(ss, "Fault", None) is None or ss.Fault.n == 0:
        try:
            ss.add(
                "Fault",
                {
                    "u": 1,
                    "name": f"F_bus{fault_bus}",
                    "bus": fault_bus,
                    "tf": fault_start,
                    "tc": fault_clear,
                    "rf": FAULT_RF,
                    "xf": FAULT_XF,
                },
            )
            fault_added = True
        except Exception as e:
            print(f"[Warning] Could not add Fault: {e}")
    else:
        # Modify existing fault
        try:
            if hasattr(ss.Fault, "alter"):
                ss.Fault.alter("tf", 0, fault_start)
                ss.Fault.alter("tc", 0, fault_clear)
                ss.Fault.alter("bus", 0, fault_bus)
                ss.Fault.alter("xf", 0, FAULT_XF)
                ss.Fault.alter("rf", 0, FAULT_RF)
                ss.Fault.alter("u", 0, 1)
        except Exception as e:
            print(f"[Warning] Could not alter Fault: {e}")

    ss.setup()
    if not (getattr(ss.PFlow, "converged", False)):
        ss.PFlow.run()
    pf_converged = getattr(ss.PFlow, "converged", None)
    if quiet:
        n_bus = getattr(getattr(ss, "Bus", None), "n", 0) or 0
        n_gen = getattr(getattr(ss, "GENCLS", None), "n", 0) or 0
        print(f"PF: converged={pf_converged}, {n_bus} buses, {n_gen} gen")
    else:
        print(f"Power flow converged: {pf_converged}")

    # Only run TDS and save trajectories when power flow converged (accuracy of generated data)
    if not pf_converged:
        print("[ERROR] Power flow did not converge. Skipping TDS and trajectory output.")
        try:
            with open(exp_dir / "run_info.txt", "w", encoding="utf-8") as f:
                f.write(f"run_timestamp: {run_ts}\n")
                f.write("power_flow_converged: False\n")
                f.write("error: power_flow_did_not_converge\n")
                f.write(f"experiment: three_phase_fault\n")
                f.write(f"output_dir: {exp_dir}\n")
        except Exception:
            pass
        return 1

    # Disable toggles so only the fault event occurs (no line trip)
    if hasattr(ss, "Toggle") and ss.Toggle.n > 0:
        try:
            if hasattr(ss.Toggle, "u") and hasattr(ss.Toggle.u, "v"):
                for i in range(ss.Toggle.n):
                    try:
                        ss.Toggle.u.v[i] = 0
                    except Exception:
                        ss.Toggle.alter("u", i + 1, 0)
            if quiet:
                print("Toggles disabled (fault-only event).")
        except Exception as e:
            print(f"[Warning] Could not disable Toggles: {e}")

    # Save power flow (same structure as run_kundur_expt). Skip when --minimal to avoid duplicate CSVs.
    if not args.minimal:
        try:
            with open(exp_dir / "power_flow_summary.txt", "w", encoding="utf-8") as f:
                f.write("Kundur power flow (three-phase fault exp)\n")
                f.write(f"converged: {pf_converged}\n")
            if hasattr(ss, "Bus") and hasattr(ss.Bus, "as_df"):
                df_bus = ss.Bus.as_df()
                if df_bus is not None and not df_bus.empty:
                    if hasattr(ss.Bus, "P") and hasattr(ss.Bus.P, "v") and ss.Bus.P.v is not None:
                        df_bus["P_pu"] = np.asarray(ss.Bus.P.v)
                    if hasattr(ss.Bus, "Q") and hasattr(ss.Bus.Q, "v") and ss.Bus.Q.v is not None:
                        df_bus["Q_pu"] = np.asarray(ss.Bus.Q.v)
                    df_bus.to_csv(exp_dir / "power_flow_bus.csv", index=False)
            if hasattr(ss, "GENCLS") and hasattr(ss.GENCLS, "as_df"):
                df_g = ss.GENCLS.as_df()
                if df_g is not None and not df_g.empty:
                    df_g.to_csv(exp_dir / "power_flow_gencls.csv", index=False)
            if hasattr(ss, "PQ") and hasattr(ss.PQ, "as_df"):
                df_pq = ss.PQ.as_df()
                if df_pq is not None and not df_pq.empty:
                    df_pq.to_csv(exp_dir / "power_flow_pq.csv", index=False)
        except Exception as e:
            print(f"[Warning] Power flow save: {e}")

    # TDS
    if hasattr(ss.TDS.config, "tf"):
        ss.TDS.config.tf = tf_sec
    if hasattr(ss.TDS.config, "tstep"):
        ss.TDS.config.tstep = TDS_TSTEP
    # Disable early stop on stability criteria so unstable runs complete full duration (0 to tf_sec)
    # and we get the full rotor angle trajectory for plotting (e.g. beyond 180 deg)
    if hasattr(ss.TDS.config, "criteria"):
        ss.TDS.config.criteria = 0
    if quiet:
        print(
            f"TDS: 0-{tf_sec} s (fault at t={fault_start}, clear at t={fault_clear}) ...",
            end=" ",
            flush=True,
        )
    ss.TDS.run()
    if quiet:
        print("done.")

    t = None
    y = None
    if (
        hasattr(ss, "dae")
        and ss.dae.ts is not None
        and hasattr(ss.dae.ts, "t")
        and len(ss.dae.ts.t) > 0
    ):
        t = np.asarray(ss.dae.ts.t)
        y = getattr(ss.dae.ts, "y", None)
    exit_code = getattr(ss, "exit_code", None)
    if quiet and t is not None:
        print(f"TDS: exit_code={exit_code}, {len(t)} points, t=[{t.min():.1f}, {t.max():.1f}] s")

    if t is not None:
        try:
            with open(exp_dir / "tds_summary.txt", "w", encoding="utf-8") as f:
                f.write(f"Kundur TDS (three-phase fault)\n")
                f.write(f"exit_code: {exit_code}\n")
                f.write(f"time_points: {len(t)}\n")
                f.write(f"t_min: {t.min():.6f} s, t_max: {t.max():.6f} s\n")
            n_gen = getattr(ss.GENCLS, "n", 0)
            x_ts = getattr(ss.dae.ts, "x", None)
            pe_a = getattr(ss.GENCLS.Pe, "a", None) if hasattr(ss, "GENCLS") else None
            delta_a = (
                getattr(getattr(ss.GENCLS, "delta", None), "a", None)
                if hasattr(ss, "GENCLS")
                else None
            )
            omega_a = (
                getattr(getattr(ss.GENCLS, "omega", None), "a", None)
                if hasattr(ss, "GENCLS")
                else None
            )
            has_pe = y is not None and y.ndim == 2 and pe_a is not None and len(pe_a) >= n_gen
            has_delta = (
                x_ts is not None
                and x_ts.ndim == 2
                and delta_a is not None
                and len(delta_a) >= n_gen
            )
            has_omega = (
                x_ts is not None
                and x_ts.ndim == 2
                and omega_a is not None
                and len(omega_a) >= n_gen
            )
            M_vals = None
            if has_delta and hasattr(ss.GENCLS, "M") and hasattr(ss.GENCLS.M, "v"):
                try:
                    M_vals = np.asarray(ss.GENCLS.M.v)[:n_gen]
                    has_delta_rel = len(M_vals) >= n_gen
                except Exception:
                    M_vals = None
                    has_delta_rel = False
            else:
                has_delta_rel = False
            has_omega_rel = has_omega and M_vals is not None
            if has_pe or has_delta or has_omega or has_delta_rel or has_omega_rel:
                headers = ["time_s"]
                if has_pe:
                    headers += [f"Pe_gen{i + 1}_pu" for i in range(n_gen)]
                if has_delta:
                    headers += [f"delta_gen{i + 1}_rad" for i in range(n_gen)]
                if has_delta_rel:
                    headers += [f"delta_rel_gen{i + 1}_rad" for i in range(n_gen)]
                if has_omega:
                    headers += [f"omega_gen{i + 1}_pu" for i in range(n_gen)]
                if has_omega_rel:
                    headers += [f"omega_rel_gen{i + 1}_pu" for i in range(n_gen)]
                with open(exp_dir / "tds_trajectories.csv", "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(headers)
                    for k in range(len(t)):
                        row = [float(t[k])]
                        if has_pe:
                            for i in range(n_gen):
                                idx = int(pe_a[i]) if i < len(pe_a) else None
                                row.append(
                                    float(y[k, idx]) if idx is not None and idx < y.shape[1] else ""
                                )
                        if has_delta:
                            delta_k = []
                            for i in range(n_gen):
                                idx = int(delta_a[i]) if i < len(delta_a) else None
                                val = (
                                    float(x_ts[k, idx])
                                    if idx is not None and idx < x_ts.shape[1]
                                    else None
                                )
                                delta_k.append(val)
                                row.append(val if val is not None else "")
                            if (
                                has_delta_rel
                                and M_vals is not None
                                and all(d is not None for d in delta_k)
                            ):
                                delta_coi = sum(M_vals[i] * delta_k[i] for i in range(n_gen)) / sum(
                                    M_vals
                                )
                                for i in range(n_gen):
                                    row.append(delta_k[i] - delta_coi)
                        if has_omega:
                            omega_k = []
                            for i in range(n_gen):
                                idx = int(omega_a[i]) if i < len(omega_a) else None
                                val = (
                                    float(x_ts[k, idx])
                                    if idx is not None and idx < x_ts.shape[1]
                                    else None
                                )
                                omega_k.append(val)
                                row.append(val if val is not None else "")
                            if (
                                has_omega_rel
                                and M_vals is not None
                                and all(o is not None for o in omega_k)
                            ):
                                omega_coi = sum(M_vals[i] * omega_k[i] for i in range(n_gen)) / sum(
                                    M_vals
                                )
                                for i in range(n_gen):
                                    row.append(omega_k[i] - omega_coi)
                        w.writerow(row)
        except Exception as e:
            print(f"[Warning] TDS save: {e}")

    # Run info including fault parameters (reproducibility for publication)
    try:
        with open(exp_dir / "run_info.txt", "w", encoding="utf-8") as f:
            f.write(f"run_timestamp: {run_ts}\n")
            f.write(f"experiment: three_phase_fault\n")
            f.write(f"case_file: {case_path}\n")
            f.write(f"addfile: {dyr_path}\n")
            f.write(f"power_flow_converged: {pf_converged}\n")
            f.write(f"tds_exit_code: {exit_code}\n")
            f.write(f"fault_bus: {fault_bus}\n")
            f.write(f"fault_start_t_s: {fault_start}\n")
            f.write(f"fault_clear_t_s: {fault_clear}\n")
            f.write(f"fault_duration_s: {fault_clear - fault_start}\n")
            f.write(f"fault_xf_pu: {FAULT_XF}\n")
            f.write(f"fault_rf_pu: {FAULT_RF}\n")
            f.write(f"tds_tf_s: {tf_sec}\n")
            f.write(f"load_alpha: {alpha}\n")
            f.write(f"output_dir: {exp_dir}\n")
    except Exception as e:
        print(f"[Warning] run_info save: {e}")

    # Generate all publication figures (Pe, delta, omega, PF). For --find-cct minimal runs this
    # is disabled (--no-figures); only delta stable/unstable figures are generated by run_kundur_expt.
    # Uncomment the block below for publication-level runs with all figures.
    if gen_figures:
        try:
            plot_script = PROJECT_ROOT / "scripts" / "plot_kundur_publication_figures.py"
            if plot_script.exists():
                subprocess.run(
                    [sys.executable, str(plot_script), "--exp-dir", str(exp_dir)],
                    cwd=str(PROJECT_ROOT),
                    check=False,
                    capture_output=quiet,
                )
        except Exception as e:
            print(f"[Warning] Figures: {e}")

    if quiet:
        print(f"Saved: {exp_dir}")
        if gen_figures and (exp_dir / "figures").exists():
            print(f"Figures: {exp_dir / 'figures'}")
    else:
        print(f"\nResults: {exp_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
