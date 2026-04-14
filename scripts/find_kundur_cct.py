"""
Find Critical Clearing Time (CCT) for Kundur system using bisection.

Runs multiple fault experiments with different fault durations and uses
bisection to converge to the maximum duration for which the system remains
stable. Stability is judged from TDS trajectories: relative rotor angle
spread (COI-referred) must stay below a threshold over the simulation
window. The conventional TSA criterion is 180 deg (past the unstable
equilibrium, loss of synchronism); default is 180 deg (use 360 for a
more conservative threshold).

Usage:
  python scripts/find_kundur_cct.py
  python scripts/find_kundur_cct.py --fault-bus 8 --duration-max 0.5 --tol 0.002
  python scripts/find_kundur_cct.py --save-final   # save CCT run with figures

If the bracket is invalid (e.g. "CCT > duration_max"), increase --duration-max
or decrease --duration-min so that one end is stable and the other unstable.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "kundur_simulations"
FAULT_SCRIPT = PROJECT_ROOT / "scripts" / "run_kundur_fault_expt.py"


def _run_fault_experiment(
    fault_start: float,
    duration_s: float,
    fault_bus: int,
    sim_time: float,
    exp_dir: Path,
    quiet: bool = True,
    gen_figures: bool = False,
    alpha: float = 1.0,
) -> int:
    """Run one fault experiment; return process return code (0 = success)."""
    fault_clear = fault_start + duration_s
    cmd = [
        sys.executable,
        str(FAULT_SCRIPT),
        "--fault-bus",
        str(fault_bus),
        "--start",
        str(fault_start),
        "--clear",
        str(fault_clear),
        "--sim-time",
        str(sim_time),
        "--exp-dir",
        str(exp_dir),
    ]
    if alpha != 1.0:
        cmd.extend(["--alpha", str(alpha)])
    if not gen_figures:
        cmd.append("--no-figures")
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=quiet,
        text=True,
    )
    return result.returncode


def _read_exit_code(exp_dir: Path) -> Optional[int]:
    """Read tds_exit_code from run_info.txt; None if missing or unreadable."""
    run_info = exp_dir / "run_info.txt"
    if not run_info.exists():
        return None
    try:
        for line in run_info.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("tds_exit_code:"):
                return int(line.split(":", 1)[1].strip())
    except (ValueError, OSError):
        pass
    return None


def _max_angle_spread_rad(df: pd.DataFrame) -> float:
    """
    Maximum over time of (max_i delta_i - min_i delta_i) in radians.
    Prefer delta_rel columns; fallback to absolute delta.
    """
    delta_rel = [c for c in df.columns if c.startswith("delta_rel_gen") and c.endswith("_rad")]
    delta_abs = [
        c for c in df.columns if c.startswith("delta_gen") and c.endswith("_rad") and "rel" not in c
    ]
    cols = delta_rel if delta_rel else delta_abs
    if not cols:
        return np.inf  # no angle data -> treat as unstable
    deltas = df[cols].values
    # per row: spread = max - min
    spread = np.nanmax(deltas, axis=1) - np.nanmin(deltas, axis=1)
    return float(np.nanmax(np.abs(spread)))


def _is_stable(exp_dir: Path, angle_threshold_rad: float) -> bool:
    """
    True if TDS completed successfully and max angle spread < angle_threshold_rad.
    """
    exit_code = _read_exit_code(exp_dir)
    if exit_code is not None and exit_code != 0:
        return False
    csv_path = exp_dir / "tds_trajectories.csv"
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    max_spread = _max_angle_spread_rad(df)
    return max_spread < angle_threshold_rad


def find_cct_bisection(
    fault_start: float = 1.0,
    fault_bus: int = 7,
    sim_time: float = 5.0,
    duration_min: float = 0.05,
    duration_max: float = 0.5,
    tol: float = 0.002,
    max_iter: int = 30,
    angle_deg: float = 180.0,
    temp_dir: Optional[Path] = None,
    verbose: bool = False,
    alpha: float = 1.0,
) -> tuple[Optional[float], Optional[float], int, str]:
    """
    Find CCT (fault duration in seconds) using bisection.

    Returns:
        (cct_duration, uncertainty, n_iter, reason).
        If bracket invalid: cct_duration is None and reason is "always_stable" or "always_unstable".
    """
    angle_rad = np.deg2rad(angle_deg)
    if temp_dir is None:
        temp_dir = OUTPUT_BASE / "cct_search_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Bracket: test min and max duration
    if verbose:
        print(
            f"Bisection: duration in [{duration_min}, {duration_max}] s, tol={tol} s, angle < {angle_deg} deg"
        )
        print(
            f"  Fault: bus={fault_bus}, start={fault_start} s, sim_time={sim_time} s, alpha={alpha}"
        )
    rc_lo = _run_fault_experiment(
        fault_start, duration_min, fault_bus, sim_time, temp_dir, quiet=not verbose, alpha=alpha
    )
    stable_lo = _is_stable(temp_dir, angle_rad) if rc_lo == 0 else False
    if verbose:
        print(f"  duration={duration_min:.4f} s -> stable={stable_lo} (returncode={rc_lo})")

    rc_hi = _run_fault_experiment(
        fault_start, duration_max, fault_bus, sim_time, temp_dir, quiet=not verbose, alpha=alpha
    )
    stable_hi = _is_stable(temp_dir, angle_rad) if rc_hi == 0 else False
    if verbose:
        print(f"  duration={duration_max:.4f} s -> stable={stable_hi} (returncode={rc_hi})")

    if not stable_lo and not stable_hi:
        if verbose:
            print("  CCT < duration_min (always unstable in bracket).")
        return None, None, 0, "always_unstable"
    if stable_lo and stable_hi:
        if verbose:
            print("  CCT > duration_max (always stable in bracket).")
        return None, None, 0, "always_stable"

    low, high = duration_min, duration_max
    # Bisection: maintain invariant that at end low = max stable duration, high = min unstable
    n_iter = 0
    while (high - low) > tol and n_iter < max_iter:
        n_iter += 1
        mid = (low + high) / 2.0
        rc = _run_fault_experiment(
            fault_start, mid, fault_bus, sim_time, temp_dir, quiet=not verbose, alpha=alpha
        )
        st = _is_stable(temp_dir, angle_rad) if rc == 0 else False
        if verbose:
            print(
                f"  iter {n_iter}: duration={mid:.4f} s [low={low:.4f}, high={high:.4f}] -> stable={st}"
            )
        if st:
            low = mid
        else:
            high = mid

    cct = low
    uncertainty = (high - low) / 2.0
    return cct, uncertainty, n_iter, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find Critical Clearing Time (CCT) for Kundur system using bisection."
    )
    parser.add_argument("--fault-bus", type=int, default=7, help="Fault bus index (default: 7)")
    parser.add_argument(
        "--start", type=float, default=1.0, help="Fault start time in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--sim-time", type=float, default=5.0, help="TDS end time in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--duration-min",
        type=float,
        default=0.05,
        help="Minimum fault duration for search in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--duration-max",
        type=float,
        default=1.0,
        help="Maximum fault duration for search in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.002,
        help="Bisection convergence tolerance in seconds (default: 0.002)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=30, help="Max bisection iterations (default: 30)"
    )
    parser.add_argument(
        "--angle-deg",
        type=float,
        default=180.0,
        help="Stability threshold: max angle spread in degrees; 180 = conventional TSA (default), 360 = conservative",
    )
    parser.add_argument(
        "--save-final",
        action="store_true",
        help="Run one final experiment at CCT and save to exp_*_fault with figures",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Load scale factor: all PQ loads scaled by alpha (default: 1.0)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config (e.g. configs/publication/kundur_2area.yaml) to read fault and sim_time; CLI overrides.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each bisection step")
    args = parser.parse_args()

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
                    args.start = float(f.get("start_time", args.start))
                    args.fault_bus = int(f.get("bus", args.fault_bus))
                args.sim_time = float(gen.get("simulation_time", args.sim_time))
            except Exception:
                pass

    if args.duration_min >= args.duration_max:
        print("Error: duration-min must be < duration-max.")
        return 1

    cct, uncertainty, n_iter, reason = find_cct_bisection(
        fault_start=args.start,
        fault_bus=args.fault_bus,
        sim_time=args.sim_time,
        duration_min=args.duration_min,
        duration_max=args.duration_max,
        tol=args.tol,
        max_iter=args.max_iter,
        angle_deg=args.angle_deg,
        verbose=args.verbose,
        alpha=args.alpha,
    )

    if cct is None:
        print("CCT could not be determined (invalid bracket).")
        if reason == "always_stable":
            print(
                f"  Both duration_min={args.duration_min:.3f} s and duration_max={args.duration_max:.3f} s are stable."
            )
            print(
                f"  CCT is above {args.duration_max:.3f} s. Try a larger --duration-max (e.g. --duration-max 1.0)."
            )
        elif reason == "always_unstable":
            print(
                f"  Both duration_min={args.duration_min:.3f} s and duration_max={args.duration_max:.3f} s are unstable."
            )
            print(
                "  CCT is below duration_min. Try a smaller --duration-min (e.g. --duration-min 0.02)."
            )
        return 1

    print(f"CCT (fault duration): {cct:.4f} s  (± {uncertainty:.4f} s)")
    print(f"  Fault clearing time: {args.start + cct:.4f} s")
    print(f"  Bisection iterations: {n_iter}")

    if args.save_final and cct is not None:
        from datetime import datetime

        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dir = OUTPUT_BASE / f"exp_{run_ts}_fault_cct"
        final_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running final experiment at CCT and saving to {final_dir} ...")
        rc = _run_fault_experiment(
            args.start,
            cct,
            args.fault_bus,
            args.sim_time,
            final_dir,
            quiet=False,
            gen_figures=True,
            alpha=args.alpha,
        )
        if rc != 0:
            print("Warning: final experiment returned non-zero exit code.")
        else:
            print(f"Results and figures saved under {final_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
