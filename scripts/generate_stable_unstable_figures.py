"""
Generate stable and unstable delta-trajectory figures with CCT in the legend.

Reads an existing fault experiment directory (e.g. from run_kundur_expt.py --find-cct)
whose run_info.txt contains fault_start_t_s, fault_clear_t_s, fault_bus. Uses
fault_duration_s as CCT, runs two simulations: one at CCT - offset (stable) and
one at CCT + offset (unstable). With 180 deg threshold, unstable means clearing
after CCT so relative angles can exceed 180 deg / diverge. Default offset 0.02 s
ensures the unstable run clearly diverges within the 5 s simulation; if the
unstable figure still looks stable, try --offset 0.06 or 0.08.

Load variation: use --alpha to scale all PQ loads by alpha (P'=alpha*P0, Q'=alpha*Q0).
When --alpha is set, CCT is found at that load level and stable/unstable runs use
that alpha, producing trajectories at the chosen load level.

Config file (e.g. configs/publication/kundur_2area.yaml): use --config to read
alpha_range, fault (bus, start_time), and optional publication_figures (cct_offset, sim_time).
Use --all-alphas to generate figures for every alpha in config's alpha_range.

Usage:
  python scripts/generate_stable_unstable_figures.py --exp-dir outputs/.../exp_*_fault
  python scripts/generate_stable_unstable_figures.py --config configs/publication/kundur_2area.yaml --alpha 0.9
  python scripts/generate_stable_unstable_figures.py --config configs/publication/kundur_2area.yaml --all-alphas
  python scripts/generate_stable_unstable_figures.py --alpha 0.9
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "kundur_simulations"
FAULT_SCRIPT = PROJECT_ROOT / "scripts" / "run_kundur_fault_expt.py"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "publication" / "kundur_2area.yaml"
# Offset (s) from CCT. Use a large enough unstable offset so divergence is visible within 5 s
# (with 0.02 s the "unstable" run can look stable in the first 2 s; 0.04–0.05 s gives clear divergence)


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config; resolve path relative to PROJECT_ROOT if needed."""
    p = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.core.utils import load_config

    return load_config(p)


def _config_to_fault_and_alpha(
    config: Dict[str, Any],
) -> Tuple[float, int, float, float, List[float]]:
    """
    Read fault and alpha settings from config.
    Returns: (fault_start, fault_bus, offset, sim_time, alpha_list).
    alpha_list is one or more alpha values from alpha_range; if alpha_range has n>1,
    alpha_list is np.linspace(min, max, n). If single point, alpha_list = [that value].
    """
    gen = config.get("data", {}).get("generation", {})
    fault = gen.get("fault", {})
    fault_start = float(fault.get("start_time", 1.0))
    fault_bus = int(fault.get("bus", 7))
    sim_time = float(gen.get("simulation_time", 5.0))

    pub = config.get("publication_figures", {})
    if pub:
        offset = float(pub.get("cct_offset", 0.02))
        sim_time = float(pub.get("sim_time", sim_time))
        f2 = pub.get("fault", {})
        if f2:
            fault_start = float(f2.get("start_time", fault_start))
            fault_bus = int(f2.get("bus", fault_bus))
    else:
        offset = 0.02

    alpha_range = gen.get("alpha_range")
    alpha_list: List[float] = [1.0]
    if isinstance(alpha_range, (list, tuple)) and len(alpha_range) >= 2:
        a_min, a_max = float(alpha_range[0]), float(alpha_range[1])
        n = int(alpha_range[2]) if len(alpha_range) >= 3 else 1
        if n <= 1:
            alpha_list = [a_min if a_min == a_max else (a_min + a_max) / 2.0]
        else:
            alpha_list = np.linspace(a_min, a_max, n).tolist()

    return fault_start, fault_bus, offset, sim_time, alpha_list


def _read_run_info(exp_dir: Path):
    """Return (fault_start, fault_clear, fault_duration, fault_bus) from run_info.txt."""
    run_info = exp_dir / "run_info.txt"
    if not run_info.exists():
        return None, None, None, None
    fault_start = fault_clear = fault_duration = None
    fault_bus = None
    try:
        for line in run_info.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("fault_start_t_s:"):
                fault_start = float(line.split(":", 1)[1].strip())
            elif line.startswith("fault_clear_t_s:"):
                fault_clear = float(line.split(":", 1)[1].strip())
            elif line.startswith("fault_duration_s:"):
                fault_duration = float(line.split(":", 1)[1].strip())
            elif line.startswith("fault_bus:"):
                fault_bus = int(line.split(":", 1)[1].strip())
    except (ValueError, OSError):
        pass
    if fault_duration is None and fault_start is not None and fault_clear is not None:
        fault_duration = fault_clear - fault_start
    return fault_start, fault_clear, fault_duration, fault_bus


def _run_fault(
    exp_dir: Path,
    fault_start: float,
    duration_s: float,
    fault_bus: int,
    sim_time: float = 5.0,
    alpha: Optional[float] = None,
) -> int:
    """Run fault experiment into exp_dir (no figures). Return exit code."""
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
        "--no-figures",
        "--minimal",  # only tds_trajectories.csv + run_info; no power_flow_*.csv (avoid duplicates)
    ]
    if alpha is not None and alpha != 1.0:
        cmd.extend(["--alpha", str(alpha)])
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True).returncode


def _run_one_alpha(
    exp_dir: Path,
    alpha: float,
    fault_start: float,
    fault_bus: int,
    offset: float,
    sim_time: float,
    verbose: bool,
) -> int:
    """Generate stable/unstable figures for one alpha (find CCT, run TDS, plot). Return 0 on success."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.find_kundur_cct import find_cct_bisection
    from scripts.plot_kundur_publication_figures import plot_tds_delta_trajectories

    exp_dir = Path(exp_dir).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    stable_dir = exp_dir / "stable"
    unstable_dir = exp_dir / "unstable"
    stable_dir.mkdir(parents=True, exist_ok=True)
    unstable_dir.mkdir(parents=True, exist_ok=True)

    print(f"Alpha={alpha:.3f}: finding CCT ...")
    cct_s, _unc, _n, reason = find_cct_bisection(
        fault_start=fault_start,
        fault_bus=fault_bus,
        sim_time=sim_time,
        duration_min=0.05,
        duration_max=1.0,
        tol=0.002,
        angle_deg=180.0,
        alpha=alpha,
    )
    if cct_s is None:
        print(f"CCT could not be determined at alpha={alpha} ({reason}).")
        return 1
    print(f"Alpha={alpha:.3f}: CCT={cct_s:.4f} s, running stable/unstable ...")
    rc_s = _run_fault(
        stable_dir, fault_start, cct_s - offset, fault_bus, sim_time=sim_time, alpha=alpha
    )
    if rc_s != 0:
        print(f"Stable run failed at alpha={alpha}.")
        return 1
    rc_u = _run_fault(
        unstable_dir, fault_start, cct_s + offset, fault_bus, sim_time=sim_time, alpha=alpha
    )
    if rc_u != 0:
        print(f"Unstable run failed at alpha={alpha}.")
        return 1
    out_dir = exp_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_tds_delta_trajectories(stable_dir, out_dir, cct_s=cct_s, filename_suffix="stable")
    plot_tds_delta_trajectories(unstable_dir, out_dir, cct_s=cct_s, filename_suffix="unstable")
    print(f"Alpha={alpha:.3f}: figures -> {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate stable and unstable delta figures with CCT in legend."
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=None,
        help="Fault experiment dir (default: latest exp_*_fault under outputs/kundur_simulations)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.02,
        help="Offset (s) from CCT: stable = CCT - offset, unstable = CCT + offset (default: 0.02)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Load scale factor: scale all PQ loads by alpha, find CCT at this load, then generate stable/unstable figures (default: use existing exp_dir CCT)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config (e.g. configs/publication/kundur_2area.yaml) for alpha_range, fault, and optional publication_figures",
    )
    parser.add_argument(
        "--all-alphas",
        action="store_true",
        help="When using --config: generate figures for every alpha in config's alpha_range (min, max, n)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print subprocess output")
    args = parser.parse_args()

    config = None
    if args.config is not None:
        try:
            config = _load_config(args.config)
        except FileNotFoundError as e:
            print(e)
            return 1
        (
            fault_start_cfg,
            fault_bus_cfg,
            offset_cfg,
            sim_time_cfg,
            alpha_list_cfg,
        ) = _config_to_fault_and_alpha(config)
        if args.offset == 0.02 and offset_cfg != 0.02:
            args.offset = offset_cfg
        if args.all_alphas and len(alpha_list_cfg) > 1:
            # Run for each alpha in config range
            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            for a in alpha_list_cfg:
                exp_dir_a = OUTPUT_BASE / f"exp_{run_ts}_fault_alpha_{a:.3f}".replace(".", "_")
                exp_dir_a.mkdir(parents=True, exist_ok=True)
                ret = _run_one_alpha(
                    exp_dir=exp_dir_a,
                    alpha=a,
                    fault_start=fault_start_cfg,
                    fault_bus=fault_bus_cfg,
                    offset=args.offset,
                    sim_time=sim_time_cfg,
                    verbose=args.verbose,
                )
                if ret != 0:
                    return ret
            print("Stable/unstable figures generated for all alphas from config.")
            return 0
        # Single run: use --alpha if set, else first alpha from config
        if args.alpha is None and alpha_list_cfg:
            args.alpha = alpha_list_cfg[0]
        args._config_fault_start = fault_start_cfg
        args._config_fault_bus = fault_bus_cfg
        args._config_sim_time = sim_time_cfg

    alpha = args.alpha
    exp_dir = args.exp_dir
    fault_start = fault_bus = None
    sim_time = 5.0
    cct_s = None
    if hasattr(args, "_config_fault_start"):
        fault_start = args._config_fault_start
        fault_bus = args._config_fault_bus
        sim_time = getattr(args, "_config_sim_time", 5.0)

    if alpha is not None and alpha != 1.0:
        # Load variation: find CCT at this alpha, then run stable/unstable at this alpha
        if exp_dir is None:
            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = OUTPUT_BASE / f"exp_{run_ts}_fault_alpha_{alpha:.2f}".replace(".", "_")
        exp_dir = Path(exp_dir).resolve()
        exp_dir.mkdir(parents=True, exist_ok=True)
        if fault_start is None:
            fault_start = 1.0
        if fault_bus is None:
            fault_bus = 7
        print(f"Finding CCT at load alpha={alpha} ...")
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.find_kundur_cct import find_cct_bisection

        cct_s, _unc, _n, reason = find_cct_bisection(
            fault_start=fault_start,
            fault_bus=fault_bus,
            sim_time=sim_time,
            duration_min=0.05,
            duration_max=1.0,
            tol=0.002,
            angle_deg=180.0,
            alpha=alpha,
        )
        if cct_s is None:
            print(f"CCT could not be determined at alpha={alpha} ({reason}).")
            return 1
        print(f"CCT at alpha={alpha}: {cct_s:.4f} s")
    else:
        # Use existing exp_dir and run_info CCT
        if exp_dir is None:
            if not OUTPUT_BASE.exists():
                print(f"Output base not found: {OUTPUT_BASE}")
                return 1
            dirs = sorted(
                [d for d in OUTPUT_BASE.iterdir() if d.is_dir() and d.name.endswith("_fault")],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            if not dirs:
                print("No exp_*_fault directory found. Run run_kundur_expt.py --find-cct first.")
                return 1
            exp_dir = dirs[0]
            print(f"Using latest fault experiment: {exp_dir}")

        exp_dir = Path(exp_dir).resolve()
        if not exp_dir.exists():
            print(f"Experiment directory not found: {exp_dir}")
            return 1

        fault_start, _fault_clear, fault_duration, fault_bus = _read_run_info(exp_dir)
        if fault_start is None or fault_duration is None or fault_bus is None:
            print(
                "Could not read fault_start_t_s, fault_duration_s, or fault_bus from run_info.txt"
            )
            return 1
        cct_s = fault_duration

    offset = args.offset
    stable_dir = exp_dir / "stable"
    unstable_dir = exp_dir / "unstable"
    stable_dir.mkdir(parents=True, exist_ok=True)
    unstable_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"CCT = {cct_s:.3f} s. "
        f"Generating stable (CCT - {offset} s) and unstable (CCT + {offset} s) runs"
        + (f" at alpha={alpha}" if alpha is not None and alpha != 1.0 else "")
        + "..."
    )

    rc_stable = _run_fault(
        stable_dir, fault_start, cct_s - offset, fault_bus, sim_time=sim_time, alpha=alpha
    )
    if rc_stable != 0:
        print("Stable run failed (non-zero exit code).")
        if args.verbose:
            print("Check run_kundur_fault_expt.py output above.")
        return 1

    rc_unstable = _run_fault(
        unstable_dir, fault_start, cct_s + offset, fault_bus, sim_time=sim_time, alpha=alpha
    )
    if rc_unstable != 0:
        print("Unstable run failed (non-zero exit code).")
        if args.verbose:
            print("Check run_kundur_fault_expt.py output above.")
        return 1

    out_dir = exp_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.plot_kundur_publication_figures import plot_tds_delta_trajectories

    plot_tds_delta_trajectories(stable_dir, out_dir, cct_s=cct_s, filename_suffix="stable")
    plot_tds_delta_trajectories(unstable_dir, out_dir, cct_s=cct_s, filename_suffix="unstable")

    print("Stable and unstable figures saved to", out_dir)
    print("  - tds_delta_trajectories_stable.png")
    print("  - tds_delta_trajectories_unstable.png")
    if alpha is not None and alpha != 1.0:
        print(f"  (load scale alpha={alpha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
