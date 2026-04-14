"""
Load Kundur system (raw + GENCLS dyr), run power flow at base load, then time-domain simulation.
Prints PF and TDS results; saves results to outputs/kundur_simulations/exp_YYYYMMDD_HHMMSS/.
Optionally plots TDS trajectories (set PLOT=True).

Fault + CCT mode: use --fault to run a three-phase fault experiment, or --find-cct to find
Critical Clearing Time by bisection, run the fault at CCT, and generate publication figures.

Usage:
  python scripts/run_kundur_expt.py
  python scripts/run_kundur_expt.py --fault --fault-bus 7 --duration 0.15
  python scripts/run_kundur_expt.py --find-cct --fault-bus 7
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import andes
import numpy as np

# Minimal terminal output: only essential status (set True for brief one-line summaries)
QUIET = True
# ANDES logging: 20=INFO, 30=WARNING, 40=ERROR (only errors when QUIET)
ANDES_STREAM_LEVEL = 40 if QUIET else 20

# Set True to show plots of TDS trajectories (interactive)
PLOT = False
# Set True to generate PNG figures in exp_dir/figures/ at the end (for quick cross-check)
GENERATE_FIGURES = True
# Set True to print .raw and .dyr model info (Bus, Line, GENCLS, etc.) in terminal
SHOW_RAW_DYR_INFO = False
# Set True to print first N lines of .raw and .dyr file contents
SHOW_RAW_FILES = False
RAW_FILE_LINES = 80  # max lines to show from each file

# Project root and output base (same pattern as run_complete_experiment / run_kundur_stage1)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "kundur_simulations"

# Timestamp for this run (folder and filenames)
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_DIR = OUTPUT_BASE / f"exp_{RUN_TS}"

# ----- CLI: fault + CCT mode (exit before loading ANDES if selected) -----
_parser = argparse.ArgumentParser(
    description="Kundur: base PF+TDS or three-phase fault experiment (optionally find CCT)."
)
_parser.add_argument(
    "--fault",
    action="store_true",
    help="Run three-phase fault experiment instead of base TDS",
)
_parser.add_argument(
    "--find-cct",
    action="store_true",
    help="Find CCT by bisection, then run fault at CCT and generate figures (implies --fault)",
)
_parser.add_argument("--fault-bus", type=int, default=7, help="Fault bus index (default: 7)")
_parser.add_argument(
    "--start", type=float, default=1.0, help="Fault start time in seconds (default: 1.0)"
)
_parser.add_argument(
    "--duration",
    type=float,
    default=0.15,
    help="Fault duration in seconds for --fault without --find-cct (default: 0.15)",
)
_parser.add_argument(
    "--duration-min",
    type=float,
    default=0.05,
    help="Min fault duration for CCT search (default: 0.05)",
)
_parser.add_argument(
    "--duration-max",
    type=float,
    default=1.0,
    help="Max fault duration for CCT search (default: 1.0)",
)
_parser.add_argument(
    "--tol", type=float, default=0.002, help="CCT bisection tolerance in seconds (default: 0.002)"
)
_parser.add_argument(
    "--angle-deg",
    type=float,
    default=180.0,
    help="Stability angle threshold in degrees for CCT (default: 180)",
)
_parser.add_argument(
    "--no-figures", action="store_true", help="Do not generate PNG figures (fault mode)"
)
_parser.add_argument("--verbose", action="store_true", help="Verbose output (fault/CCT mode)")
_CLI = _parser.parse_args()

if _CLI.fault or _CLI.find_cct:
    _exp_dir = OUTPUT_BASE / f"exp_{RUN_TS}_fault"
    _exp_dir.mkdir(parents=True, exist_ok=True)
    _duration = _CLI.duration
    if _CLI.find_cct:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.find_kundur_cct import find_cct_bisection

        _cct, _unc, _n_iter, _reason = find_cct_bisection(
            fault_start=_CLI.start,
            fault_bus=_CLI.fault_bus,
            sim_time=5.0,
            duration_min=_CLI.duration_min,
            duration_max=_CLI.duration_max,
            tol=_CLI.tol,
            angle_deg=_CLI.angle_deg,
            verbose=_CLI.verbose,
        )
        if _cct is None:
            print("CCT could not be determined (invalid bracket).")
            if _reason == "always_stable":
                print(
                    "  Both duration_min and duration_max are stable. Try --duration-max 1.5 or larger."
                )
            elif _reason == "always_unstable":
                print("  Both are unstable. Try smaller --duration-min.")
            sys.exit(1)
        _duration = _cct
        print(
            f"CCT found: {_cct:.4f} s (clearing at t={_CLI.start + _cct:.4f} s). "
            "Running fault at CCT and generating figures."
        )
    _cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_kundur_fault_expt.py"),
        "--fault-bus",
        str(_CLI.fault_bus),
        "--start",
        str(_CLI.start),
        "--duration",
        str(_duration),
        "--exp-dir",
        str(_exp_dir),
    ]
    # Minimal experiment: only delta stable/unstable figures (for quick runs). For publication-level
    # runs with all figures (Pe, omega, PF, etc.), remove the next two lines or add --full-figures.
    if _CLI.find_cct:
        _cmd.append("--no-figures")  # CCT run: no Pe/omega/PF figures; only stable/unstable below
    elif _CLI.no_figures:
        _cmd.append("--no-figures")
    if _CLI.verbose:
        _cmd.append("--verbose")
    _result = subprocess.run(_cmd, cwd=str(PROJECT_ROOT), capture_output=not _CLI.verbose)
    if _result.returncode != 0 and _CLI.verbose and _result.stderr:
        sys.stderr.write(_result.stderr.decode("utf-8", errors="replace"))
    # When --find-cct and figures enabled: generate only delta stable/unstable figures (CCT in legend)
    if _CLI.find_cct and _result.returncode == 0 and not _CLI.no_figures:
        _gen_script = PROJECT_ROOT / "scripts" / "generate_stable_unstable_figures.py"
        if _gen_script.exists():
            _r2 = subprocess.run(
                [sys.executable, str(_gen_script), "--exp-dir", str(_exp_dir)],
                cwd=str(PROJECT_ROOT),
                capture_output=not _CLI.verbose,
            )
            if _r2.returncode != 0 and _CLI.verbose:
                if _r2.stderr:
                    sys.stderr.write(_r2.stderr.decode("utf-8", errors="replace"))
            elif _r2.returncode == 0:
                print(
                    "Stable/unstable delta figures: tds_delta_trajectories_stable.png, tds_delta_trajectories_unstable.png"
                )
    print(f"Saved: {_exp_dir}")
    if (_exp_dir / "figures").exists():
        print(f"Figures: {_exp_dir / 'figures'}")
    sys.exit(0 if _result.returncode == 0 else 1)

# ----- Base run (no fault): load case and run PF + TDS -----
# Kundur with GENCLS (4 machines) - base load from case
case_path = andes.get_case("kundur/kundur.raw")
dyr_path = andes.get_case("kundur/kundur_gencls.dyr")

if QUIET:
    andes.config_logger(stream_level=ANDES_STREAM_LEVEL)
# andes.run() loads the case and runs power flow by default
ss = andes.run(case_path, addfile=dyr_path, default_config=True)

# Create output directory for this run
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ----- Disable Toggle(s) for no-event TDS (no line trip at t=2 s) -----
if hasattr(ss, "Toggle") and ss.Toggle.n > 0:
    try:
        if hasattr(ss.Toggle, "u") and hasattr(ss.Toggle.u, "v"):
            uv = ss.Toggle.u.v
            for i in range(ss.Toggle.n):
                if hasattr(uv, "__setitem__"):
                    uv[i] = 0
                else:
                    ss.Toggle.alter("u", i + 1, 0)
            print(
                "Toggles disabled (no line trip)."
                if QUIET
                else "[Setup] Disabled all Toggles (no line trip during TDS)."
            )
        else:
            for i in range(ss.Toggle.n):
                ss.Toggle.alter("u", i + 1, 0)
            print(
                "Toggles disabled (no line trip)."
                if QUIET
                else "[Setup] Disabled all Toggles (no line trip during TDS)."
            )
    except Exception as e:
        print(f"[Warning] Could not disable Toggles: {e}")


def _model_table(ss, name):
    """Print as_df() for model if available and n>0."""
    if not hasattr(ss, name):
        return
    m = getattr(ss, name)
    if not hasattr(m, "n") or m.n == 0:
        return
    print(f"\n--- {name} (n={m.n}) ---")
    if hasattr(m, "as_df"):
        try:
            df = m.as_df()
            if df is not None and not df.empty:
                print(df.to_string())
            else:
                print("  (empty table)")
        except Exception as e:
            print("  ", e)
    else:
        print("  (no as_df)")


# ----- From .raw (network & power flow data) -----
if SHOW_RAW_DYR_INFO:
    print("=" * 60)
    print("FROM .raw (network & power flow)")
    print("=" * 60)
    print("File:", case_path)
    for name in [
        "Bus",
        "Line",
        "Transformer",
        "PQ",
        "PV",
        "Slack",
        "Shunt",
        "Area",
        "Zone",
    ]:
        _model_table(ss, name)
if SHOW_RAW_FILES:
    print("\n--- .raw file (first {} lines) ---".format(RAW_FILE_LINES))
    try:
        with open(case_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= RAW_FILE_LINES:
                    print("  ...")
                    break
                print(" ", line.rstrip())
    except Exception as e:
        print("  ", e)

# ----- From .dyr (dynamics data) -----
if SHOW_RAW_DYR_INFO:
    print("\n" + "=" * 60)
    print("FROM .dyr (dynamics)")
    print("=" * 60)
    print("File:", dyr_path)
    for name in ["GENCLS", "GENROU", "GENSAE", "Gov", "Exc", "PSS", "TG"]:
        _model_table(ss, name)
if SHOW_RAW_FILES:
    print("\n--- .dyr file (first {} lines) ---".format(RAW_FILE_LINES))
    try:
        with open(dyr_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= RAW_FILE_LINES:
                    print("  ...")
                    break
                print("  ", line.rstrip())
    except Exception as e:
        print("  ", e)

# ----- Power flow results -----
pf_converged = getattr(ss.PFlow, "converged", None)
if QUIET:
    n_bus = getattr(getattr(ss, "Bus", None), "n", 0) or 0
    n_gen = getattr(getattr(ss, "GENCLS", None), "n", 0) or 0
    print(f"PF: converged={pf_converged}, {n_bus} buses, {n_gen} gen")
else:
    print("=" * 60)
    print("POWER FLOW RESULTS")
    print("=" * 60)
    print("Converged:", pf_converged)
    if (
        hasattr(ss, "Bus")
        and hasattr(ss.Bus, "v")
        and hasattr(ss.Bus.v, "a")
        and hasattr(ss, "dae")
        and ss.dae.y is not None
    ):
        try:
            v_a = ss.Bus.v.a
            a_a = getattr(ss.Bus.a, "a", None)
            n_bus = ss.Bus.n
            print(f"\nBuses: {n_bus}")
            for i in range(min(n_bus, 12)):
                v_idx = int(v_a[i]) if hasattr(v_a, "__getitem__") else int(v_a)
                V = float(ss.dae.y[v_idx])
                theta_deg = (
                    float(np.rad2deg(ss.dae.y[a_a[i]]))
                    if a_a is not None and hasattr(a_a, "__getitem__") and i < len(a_a)
                    else None
                )
                th_str = f", theta={theta_deg:.2f} deg" if theta_deg is not None else ""
                print(f"  Bus {i + 1}: V={V:.4f} pu{th_str}")
            if n_bus > 12:
                print(f"  ... ({n_bus} buses total)")
        except Exception as e:
            print("  (Bus V/theta:", e, ")")
    if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
        print(f"\nGENCLS: {ss.GENCLS.n} machines")
        pe_a = getattr(getattr(ss.GENCLS, "Pe", None), "a", None)
        pe_a_len = len(pe_a) if pe_a is not None and hasattr(pe_a, "__len__") else 0
        if (
            hasattr(ss, "dae")
            and ss.dae.y is not None
            and pe_a is not None
            and pe_a_len >= ss.GENCLS.n
        ):
            for i in range(ss.GENCLS.n):
                idx = int(pe_a[i]) if hasattr(pe_a, "__getitem__") else int(pe_a)
                Pe = float(ss.dae.y[idx]) if idx < len(ss.dae.y) else None
                tm0v = getattr(getattr(ss.GENCLS.tm0, "v", None), "__getitem__", None)
                Pm = float(ss.GENCLS.tm0.v[i]) if tm0v and i < len(ss.GENCLS.tm0.v) else None
                print(
                    f"  Gen {i + 1}: Pe={Pe:.4f} pu"
                    if Pe is not None
                    else f"  Gen {i + 1}: Pe=N/A",
                    end="",
                )
                print(f", Pm={Pm:.4f} pu" if Pm is not None else "")
        else:
            if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                print("  tm0 (pu):", list(np.round(ss.GENCLS.tm0.v, 4)))
            if pe_a_len == 0:
                print("  (Pe from PF not available until TDS; see TDS results below.)")
    if hasattr(ss, "PQ") and ss.PQ.n > 0:
        print(f"\nPQ loads: {ss.PQ.n}")
        for i in range(min(ss.PQ.n, 8)):
            p = float(ss.PQ.p0.v[i]) if hasattr(ss.PQ.p0.v, "__getitem__") else float(ss.PQ.p0.v)
            q = (
                float(ss.PQ.q0.v[i])
                if hasattr(ss.PQ.q0, "v") and hasattr(ss.PQ.q0.v, "__getitem__")
                else 0
            )
            print(f"  Load {i + 1}: P={p:.4f}, Q={q:.4f} pu")
        if ss.PQ.n > 8:
            print(f"  ... ({ss.PQ.n} loads total)")

# ----- Save power flow results -----
try:
    with open(EXP_DIR / "power_flow_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Kundur power flow summary (exp_{RUN_TS})\n")
        f.write(f"converged: {pf_converged}\n")
        f.write(f"buses: {ss.Bus.n if hasattr(ss, 'Bus') and hasattr(ss.Bus, 'n') else 'N/A'}\n")
        f.write(
            f"GENCLS: {ss.GENCLS.n if hasattr(ss, 'GENCLS') and hasattr(ss.GENCLS, 'n') else 0}\n"
        )
        f.write(f"PQ loads: {ss.PQ.n if hasattr(ss, 'PQ') and hasattr(ss.PQ, 'n') else 0}\n")
    if hasattr(ss, "Bus") and hasattr(ss.Bus, "as_df"):
        df_bus = ss.Bus.as_df()
        if df_bus is not None and not df_bus.empty:
            n_bus = len(df_bus)
            if hasattr(ss.Bus, "P") and hasattr(ss.Bus.P, "v") and ss.Bus.P.v is not None:
                df_bus["P_pu"] = np.asarray(ss.Bus.P.v)
            if hasattr(ss.Bus, "Q") and hasattr(ss.Bus.Q, "v") and ss.Bus.Q.v is not None:
                df_bus["Q_pu"] = np.asarray(ss.Bus.Q.v)
            # Use ANDES values only if they look meaningful (non-zero somewhere); else use fallback
            use_fallback_p = "P_pu" not in df_bus.columns or (
                "P_pu" in df_bus.columns
                and np.allclose(df_bus["P_pu"].values, 0.0)
                and hasattr(ss, "GENCLS")
            )
            use_fallback_q = "Q_pu" not in df_bus.columns or (
                "Q_pu" in df_bus.columns and np.allclose(df_bus["Q_pu"].values, 0.0)
            )
            if use_fallback_p or use_fallback_q:
                P_pu = np.zeros(n_bus)
                Q_pu = np.zeros(n_bus)

                def _bus_row_index(b):
                    b = int(b) if hasattr(b, "__int__") else int(b)
                    # ANDES bus index is usually 1-based position in Bus table
                    if 1 <= b <= n_bus:
                        return b - 1
                    if "name" in df_bus.columns:
                        match = df_bus.index[df_bus["name"] == b].tolist()
                        if match:
                            return match[0]
                    if "idx" in df_bus.columns:
                        match = df_bus.index[df_bus["idx"] == b].tolist()
                        if match:
                            return match[0]
                    return None

                if (
                    hasattr(ss, "GENCLS")
                    and hasattr(ss.GENCLS, "bus")
                    and hasattr(ss.GENCLS.bus, "v")
                ):
                    gen_bus = np.asarray(ss.GENCLS.bus.v)
                    # After PF, Pe.v may be unpopulated; use tm0 (Pm setpoint = Pe at steady state)
                    Pe = (
                        np.asarray(ss.GENCLS.tm0.v)
                        if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v")
                        else None
                    )
                    if Pe is None and hasattr(ss.GENCLS, "Pe") and hasattr(ss.GENCLS.Pe, "v"):
                        Pe = np.asarray(ss.GENCLS.Pe.v)
                    if Pe is not None:
                        for g in range(len(gen_bus)):
                            ri = _bus_row_index(gen_bus[g])
                            if ri is not None and g < len(Pe):
                                P_pu[ri] += float(Pe[g])
                if hasattr(ss, "PQ") and hasattr(ss.PQ, "bus") and hasattr(ss.PQ.bus, "v"):
                    load_bus = np.asarray(ss.PQ.bus.v)
                    p0 = np.asarray(ss.PQ.p0.v)
                    q0 = (
                        np.asarray(ss.PQ.q0.v)
                        if hasattr(ss.PQ, "q0") and hasattr(ss.PQ.q0, "v")
                        else np.zeros_like(p0)
                    )
                    for L in range(len(load_bus)):
                        ri = _bus_row_index(load_bus[L])
                        if ri is not None:
                            if L < len(p0):
                                P_pu[ri] -= float(p0[L])
                            if L < len(q0):
                                Q_pu[ri] -= float(q0[L])
                if use_fallback_p:
                    df_bus["P_pu"] = P_pu
                if use_fallback_q:
                    df_bus["Q_pu"] = Q_pu
            df_bus.to_csv(EXP_DIR / "power_flow_bus.csv", index=False)
    if hasattr(ss, "GENCLS") and hasattr(ss.GENCLS, "as_df"):
        df_gen = ss.GENCLS.as_df()
        if df_gen is not None and not df_gen.empty:
            df_gen.to_csv(EXP_DIR / "power_flow_gencls.csv", index=False)
    if hasattr(ss, "PQ") and hasattr(ss.PQ, "as_df"):
        df_pq = ss.PQ.as_df()
        if df_pq is not None and not df_pq.empty:
            df_pq.to_csv(EXP_DIR / "power_flow_pq.csv", index=False)
    if not QUIET:
        print(f"\n[Saved] Power flow results -> {EXP_DIR}")
except Exception as e:
    print(f"[Warning] Could not save power flow results: {e}")

# Only run TDS when power flow converged (accuracy of generated data)
if not pf_converged:
    print("[ERROR] Power flow did not converge. Skipping TDS.")
    try:
        with open(EXP_DIR / "run_info.txt", "w", encoding="utf-8") as f:
            f.write("power_flow_converged: False\n")
            f.write("error: power_flow_did_not_converge\n")
    except Exception:
        pass
    sys.exit(1)

# ----- Time-domain simulation -----
tf_sec = 5.0
if hasattr(ss.TDS.config, "tf"):
    ss.TDS.config.tf = tf_sec
if hasattr(ss.TDS.config, "tstep"):
    ss.TDS.config.tstep = 0.01
if QUIET:
    print(f"TDS: 0–{tf_sec} s ...", end=" ", flush=True)
else:
    print(f"\nRunning TDS 0–{tf_sec} s...")
ss.TDS.run()
if QUIET:
    print("done.")

# ----- TDS results -----
t = y = None
if (
    hasattr(ss, "dae")
    and ss.dae.ts is not None
    and hasattr(ss.dae.ts, "t")
    and len(ss.dae.ts.t) > 0
):
    t = np.asarray(ss.dae.ts.t)
    y = ss.dae.ts.y if hasattr(ss.dae.ts, "y") and ss.dae.ts.y is not None else None

exit_code = getattr(ss, "exit_code", None)
if QUIET:
    if t is None:
        print(f"TDS: no time series (exit_code={exit_code}).")
    else:
        print(f"TDS: exit_code={exit_code}, {len(t)} points, t=[{t.min():.1f}, {t.max():.1f}] s")
else:
    print("\n" + "=" * 60)
    print("TIME-DOMAIN SIMULATION RESULTS")
    print("=" * 60)
    print("exit_code:", exit_code, "(0 = success)")
    if t is None:
        print("No TDS time series available.")
    else:
        print(f"Time: {len(t)} points, t_min={t.min():.3f} s, t_max={t.max():.3f} s")
        if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0 and y is not None and y.ndim == 2:
            pe_a = getattr(ss.GENCLS.Pe, "a", None)
            for i in range(ss.GENCLS.n):
                pe_idx = (
                    int(pe_a[i])
                    if pe_a is not None and hasattr(pe_a, "__getitem__") and i < len(pe_a)
                    else None
                )
                if pe_idx is not None and pe_idx < y.shape[1]:
                    pe_0 = float(y[0, pe_idx])
                    pe_end = float(y[-1, pe_idx])
                    print(f"  Gen {i + 1}: Pe(t=0)={pe_0:.4f}, Pe(t=tf)={pe_end:.4f} pu")
                else:
                    print(f"  Gen {i + 1}: (Pe index N/A)")

# Optional: plot Pe(t) (when t available)
if t is not None and PLOT and y is not None and hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
    try:
        import matplotlib.pyplot as plt

        pe_a = getattr(ss.GENCLS.Pe, "a", None)
        if pe_a is not None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            for i in range(min(ss.GENCLS.n, 4)):
                idx = int(pe_a[i]) if hasattr(pe_a, "__getitem__") and i < len(pe_a) else None
                if idx is not None and idx < y.shape[1]:
                    ax.plot(t, y[:, idx], label=f"Gen {i + 1} Pe")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Pe (pu)")
            ax.legend()
            ax.grid(True)
            ax.set_title("Kundur TDS: electrical power")
            plt.tight_layout()
            plt.show()
    except ImportError:
        print("(Set PLOT=True and install matplotlib to plot.)")

# ----- Save TDS results -----
if t is not None:
    try:
        import csv

        with open(EXP_DIR / "tds_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"Kundur TDS summary (exp_{RUN_TS})\n")
            f.write(f"exit_code: {getattr(ss, 'exit_code', None)}\n")
            f.write(f"time_points: {len(t)}\n")
            f.write(f"t_min: {t.min():.6f} s, t_max: {t.max():.6f} s\n")
        n_gen = getattr(ss.GENCLS, "n", 0)
        x_ts = getattr(ss.dae.ts, "x", None)
        pe_a = getattr(ss.GENCLS.Pe, "a", None) if hasattr(ss, "GENCLS") else None
        delta_a = (
            getattr(getattr(ss.GENCLS, "delta", None), "a", None) if hasattr(ss, "GENCLS") else None
        )
        omega_a = (
            getattr(getattr(ss.GENCLS, "omega", None), "a", None) if hasattr(ss, "GENCLS") else None
        )
        has_pe = (
            y is not None
            and y.ndim == 2
            and pe_a is not None
            and hasattr(pe_a, "__len__")
            and len(pe_a) >= n_gen
        )
        has_delta = (
            x_ts is not None
            and x_ts.ndim == 2
            and delta_a is not None
            and hasattr(delta_a, "__len__")
            and len(delta_a) >= n_gen
        )
        has_omega = (
            x_ts is not None
            and x_ts.ndim == 2
            and omega_a is not None
            and hasattr(omega_a, "__len__")
            and len(omega_a) >= n_gen
        )
        # Relative rotor angle (delta - delta_COI) for fault/transient studies
        M_vals = None
        if has_delta and hasattr(ss.GENCLS, "M") and hasattr(ss.GENCLS.M, "v"):
            try:
                M_vals = np.asarray(ss.GENCLS.M.v)[:n_gen]
                if len(M_vals) >= n_gen:
                    has_delta_rel = True
                else:
                    M_vals = None
                    has_delta_rel = False
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
            with open(EXP_DIR / "tds_trajectories.csv", "w", encoding="utf-8", newline="") as f:
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
        if not QUIET:
            print(f"[Saved] TDS results -> {EXP_DIR}")
    except Exception as e:
        print(f"[Warning] Could not save TDS results: {e}")

# ----- Run info (case paths, timestamp) -----
try:
    with open(EXP_DIR / "run_info.txt", "w", encoding="utf-8") as f:
        f.write(f"run_timestamp: {RUN_TS}\n")
        f.write(f"case_file: {case_path}\n")
        f.write(f"addfile: {dyr_path}\n")
        f.write(f"power_flow_converged: {pf_converged}\n")
        f.write(f"tds_exit_code: {getattr(ss, 'exit_code', None)}\n")
        f.write(f"output_dir: {EXP_DIR}\n")
    if not QUIET:
        print(f"[Saved] Run info -> {EXP_DIR / 'run_info.txt'}")
except Exception as e:
    print(f"[Warning] Could not save run_info: {e}")

# ----- Generate PNG figures (optional; for cross-checking) -----
if GENERATE_FIGURES:
    try:
        import subprocess
        import sys

        plot_script = PROJECT_ROOT / "scripts" / "plot_kundur_publication_figures.py"
        if plot_script.exists():
            subprocess.run(
                [sys.executable, str(plot_script), "--exp-dir", str(EXP_DIR)],
                cwd=str(PROJECT_ROOT),
                check=False,
                capture_output=QUIET,
            )
        elif not QUIET:
            print(
                f"[Warning] Figure script not found: {plot_script}. Run: python scripts/plot_kundur_publication_figures.py --exp-dir {EXP_DIR}"
            )
    except Exception as e:
        print(
            f"[Warning] Could not generate figures: {e}. Run manually: python scripts/plot_kundur_publication_figures.py --exp-dir {EXP_DIR}"
        )

if QUIET:
    print(f"Saved: {EXP_DIR}")
    if GENERATE_FIGURES and (EXP_DIR / "figures").exists():
        print(f"Figures: {EXP_DIR / 'figures'}")
else:
    print(f"\nAll results saved under: {EXP_DIR}")
