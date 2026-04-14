#!/usr/bin/env python
"""
Generate multimachine (Kundur 2-area) trajectory data for PINN verification.

Workflow (implemented in data_generation/parameter_sweep.py, triggered by this script):
  For each operating point the following is enforced; there are no fallbacks that
  would produce incorrect data.

  1. Power flow for the current operating point (case + load level).
     If power flow does not converge -> reject (no TDS, no CSV row).
  2. Bisection on fault duration to find CCT for this operating point
     (use_cct_based_sampling + find_cct in andes_utils/cct_finder.py).
     If CCT is not found -> use fault_clearing_times from config as fallback (if provided),
     so trajectory data is still generated; param_cct_absolute is then filled from the
     stable/unstable boundary (estimated CCT). Otherwise skip this operating point.
  3. Clearing times = CCT ± additional_clearing_time_offsets only -> stable (below CCT)
     and unstable (above CCT) trajectories around the boundary.
  4. Time-domain simulation with a 3-phase fault at the defined location
     (fault.start_time, fault.bus, fault.reactance) for each clearing time.
  5. Save trajectories and metadata (clearing time, stability, H, D, alpha, etc.)
     in parameter_sweep_data_*.csv for ML/PINN training.
  6. If alpha_range is set: for each (H, D, alpha) operating point, repeat
     steps 1-5 (change load via alpha, then PF -> CCT -> offsets -> TDS -> save).

  Focus variables for multimachine: H, D, alpha. Pm is from power flow (load), not swept.

This script reads publication-style YAML config (e.g. configs/publication/kundur_2area.yaml),
calls generate_parameter_sweep_multimachine() with the parsed config, and writes CSV and
analysis into a timestamped run folder (exp_YYYYMMDD_HHMMSS) under the given output base.
Scenarios where power flow does not converge are rejected inside parameter_sweep.

Usage:
    # Default: configs/publication/kundur_2area.yaml, output data/multimachine/kundur
    python scripts/generate_multimachine_data.py

    # Custom config and output
    python scripts/generate_multimachine_data.py --config configs/publication/kundur_2area.yaml --output data/multimachine/kundur

    # Same as SMIB flow: one command to produce data for later training/eval
    python scripts/generate_multimachine_data.py --config configs/publication/kundur_2area.yaml

    # Skip automatic analysis (analysis results go to <output>/analysis by default)
    python scripts/generate_multimachine_data.py --output data/multimachine/kundur --skip-analysis

    # Reuse existing data: run analysis only (no generation)
    python scripts/generate_multimachine_data.py --analysis-only
    python scripts/generate_multimachine_data.py --analysis-only data/multimachine/kundur/exp_20260219_103349
    python scripts/generate_multimachine_data.py --analysis-only path/to/parameter_sweep_data_20260219.csv
"""

import argparse
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.core.utils import generate_experiment_id

# Suppress ANDES warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="andes")
warnings.filterwarnings("ignore", message=".*vf range.*")
warnings.filterwarnings("ignore", message=".*GENCLS.*")
warnings.filterwarnings("ignore", message=".*typical.*limit.*")
warnings.filterwarnings("ignore", message=".*Time step reduced.*")
warnings.filterwarnings("ignore", message=".*Convergence.*")

import yaml


def _parse_param_range(val, n_samples):
    """Convert [min, max] or [min, max, n] to (min, max, n) for parameter_sweep."""
    if val is None or (isinstance(val, (list, tuple)) and len(val) < 2):
        return None
    if isinstance(val, (list, tuple)):
        if len(val) == 2:
            return (float(val[0]), float(val[1]), n_samples if n_samples else 5)
        if len(val) == 3:
            return (float(val[0]), float(val[1]), int(val[2]))
    return None


def _find_latest_sweep_csv(directory: Path) -> Optional[Path]:
    """Return path to latest parameter_sweep_data_*.csv under directory (including exp_* subdirs)."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates = list(directory.glob("parameter_sweep_data_*.csv"))
    for sub in directory.iterdir():
        if sub.is_dir() and sub.name.startswith("exp_"):
            candidates.extend(sub.glob("parameter_sweep_data_*.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_analysis_only(data_path_arg, output_base, config_path, project_root):
    """Run analysis on existing data (reuse generated CSV)."""
    project_root = Path(project_root)
    output_path = Path(output_base)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    num_machines = config.get("data", {}).get("generation", {}).get("num_machines", 4)

    # Resolve CSV path
    if not data_path_arg or not data_path_arg.strip():
        csv_path = _find_latest_sweep_csv(output_path)
        if csv_path is None:
            print(
                f"No parameter_sweep_data_*.csv found under {output_path} "
                "(or in any exp_* subfolder). Run generation first or pass --analysis-only <path>."
            )
            sys.exit(1)
    else:
        p = Path(data_path_arg)
        if not p.is_absolute():
            p = project_root / p
        if p.is_file():
            csv_path = p
        elif p.is_dir():
            csv_path = _find_latest_sweep_csv(p)
            if csv_path is None:
                print(
                    f"No parameter_sweep_data_*.csv found under {p}. "
                    "Pass a CSV file or a directory that contains one."
                )
                sys.exit(1)
        else:
            print(f"Data path not found: {p}")
            sys.exit(1)

    analysis_dir = csv_path.parent / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Multimachine data analysis (reuse existing data)")
    print("=" * 70)
    print(f"Data: {csv_path}")
    print(f"Output: {analysis_dir}")
    print(f"Num machines: {num_machines}")
    print("-" * 70)

    analyze_script = project_root / "scripts" / "analyze_multimachine_data.py"
    cmd = [
        sys.executable,
        str(analyze_script),
        str(csv_path),
        "--plot",
        "--output-dir",
        str(analysis_dir),
        "--num-machines",
        str(num_machines),
    ]
    rc = subprocess.run(cmd, cwd=str(project_root))
    if rc.returncode != 0:
        sys.exit(rc.returncode)
    print("Analysis results saved in:", analysis_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate multimachine (Kundur 2-area) trajectory data from publication config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/publication/kundur_2area.yaml",
        help="Path to YAML config (default: configs/publication/kundur_2area.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/multimachine/kundur",
        help="Base output directory; each run creates a timestamped subfolder exp_YYYYMMDD_HHMMSS (default: data/multimachine/kundur)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite / regenerate even if output dir has data",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip automatic data analysis after generation (default: run analysis)",
    )
    parser.add_argument(
        "--analysis-only",
        type=str,
        nargs="?",
        const="",
        default=None,
        metavar="DATA_PATH",
        help="Reuse existing data: run analysis only (no generation). DATA_PATH can be a CSV file, "
        "an exp_* folder, or a directory; if omitted, use latest in --output.",
    )
    args = parser.parse_args()

    # ---- Analysis-only mode: reuse generated data ----
    if args.analysis_only is not None:
        _run_analysis_only(
            data_path_arg=args.analysis_only if args.analysis_only else None,
            output_base=args.output,
            config_path=args.config,
            project_root=project_root,
        )
        return

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    # Timestamped experiment folder for this run (local time or PINN_TZ; same as run_multimachine_complete_experiment)
    experiment_id = generate_experiment_id()
    run_dir = output_path / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    gen = config.get("data", {}).get("generation", {})
    case_file = gen.get("case_file", "kundur/kundur.json")
    addfile = gen.get("addfile")  # Optional DYR for PSS/E raw (e.g. kundur/kundur_gencls.dyr)
    num_machines = gen.get("num_machines", 4)
    use_pe_as_input = gen.get("use_pe_as_input", True)
    n_samples = gen.get("n_samples", 30)
    n_samples_per_combination = gen.get("n_samples_per_combination", 6)
    use_cct_based_sampling = gen.get("use_cct_based_sampling", True)
    cct_offsets = gen.get("additional_clearing_time_offsets")
    sampling_strategy = gen.get("sampling_strategy", "sobol")
    simulation_time = gen.get("simulation_time", 5.0)
    time_step = gen.get("time_step", 0.002)
    seed = config.get("reproducibility", {}).get("random_seed")

    fault = gen.get("fault", {})
    fault_start_time = fault.get("start_time", 1.0)
    fault_bus = fault.get("bus", 7)
    fault_reactance = fault.get("reactance", 0.0001)
    skip_fault = gen.get("skip_fault", False)  # If True, run TDS without fault (validate base case)
    # Optional: fallback clearing times when CCT finding fails (absolute times, e.g. [1.02, 1.05, 1.10, 1.15])
    fault_clearing_times = gen.get("fault_clearing_times")

    param = gen.get("parameter_ranges", {})
    case_defaults = gen.get("case_defaults") or {}
    H_scale_range = gen.get("H_scale_range")
    D_scale_range = gen.get("D_scale_range")
    H_base = case_defaults.get("H") if isinstance(case_defaults, dict) else None
    D_base = case_defaults.get("D") if isinstance(case_defaults, dict) else None
    # H/D: use scale factors around case defaults (like alpha for load) when set; else absolute parameter_ranges
    # Scale ranges can be [min, max] or [min, max, n] (n = number of points, like alpha_range)
    if (
        H_scale_range is not None
        and D_scale_range is not None
        and H_base is not None
        and D_base is not None
    ):
        n_H = int(H_scale_range[2]) if len(H_scale_range) > 2 else 1
        n_D = int(D_scale_range[2]) if len(D_scale_range) > 2 else 1
        H_range = (
            float(H_base) * float(H_scale_range[0]),
            float(H_base) * float(H_scale_range[1]),
            n_H,
        )
        D_range = (
            float(D_base) * float(D_scale_range[0]),
            float(D_base) * float(D_scale_range[1]),
            n_D,
        )
        if n_H > 1 or n_D > 1:
            n_samples = n_H * n_D
        print(
            f"H/D from case_defaults × scale: H in [{H_range[0]:.2f}, {H_range[1]:.2f}] (n={n_H}), "
            f"D in [{D_range[0]:.2f}, {D_range[1]:.2f}] (n={n_D}), base H={H_base}, D={D_base}"
        )
    else:
        H_range = _parse_param_range(param.get("H"), n_samples)
        D_range = _parse_param_range(param.get("D"), n_samples)
    # Multimachine: focus variables are H, D, alpha. Pm is NOT swept; when Pload (alpha) changes, Pm follows from power flow.
    alpha_range = gen.get("alpha_range")  # list [min, max] or [min, max, n]
    if isinstance(alpha_range, (list, tuple)) and len(alpha_range) >= 2:
        alpha_range = (
            float(alpha_range[0]),
            float(alpha_range[1]),
            int(alpha_range[2]) if len(alpha_range) > 2 else (n_samples or 8),
        )
    else:
        alpha_range = (1.0, 1.0, 1)  # default: single load level; Pm from PF
    base_load = gen.get("base_load")  # optional {"Pload": 0.5, "Qload": 0.2}
    # Pm from case: when ANDES doesn't set GENCLS.tm0 after PF, use these (scaled by alpha).
    # Prefer case_defaults.Pm from config; else read from case file (.raw + .dyr) via ANDES.
    _pm = case_defaults.get("Pm") if isinstance(case_defaults, dict) else None
    if _pm is not None:
        case_default_pm = (
            list(_pm) if isinstance(_pm, (list, tuple)) else [float(_pm)] * num_machines
        )
        if len(case_default_pm) < num_machines:
            case_default_pm = case_default_pm + [case_default_pm[-1]] * (
                num_machines - len(case_default_pm)
            )
        case_default_pm = case_default_pm[:num_machines]
    else:
        # Read per-generator P from case (e.g. .raw + .dyr) using ANDES power flow
        case_default_pm = None
        try:
            import andes
            from data_generation.andes_utils.case_file_modifier import (
                get_generator_p_from_andes_case,
            )

            _case_path = case_file
            if not Path(case_file).is_absolute():
                _case_path = str(project_root / case_file)
                if hasattr(andes, "get_case"):
                    try:
                        _resolved = andes.get_case(case_file)
                        if _resolved and Path(_resolved).exists():
                            _case_path = _resolved
                    except Exception:
                        pass
            _addfile_path = None
            if addfile:
                _addfile_path = (
                    addfile if Path(addfile).is_absolute() else str(project_root / addfile)
                )
                if hasattr(andes, "get_case"):
                    try:
                        _resolved = andes.get_case(addfile)
                        if _resolved and Path(_resolved).exists():
                            _addfile_path = _resolved
                    except Exception:
                        pass
            _plist = get_generator_p_from_andes_case(_case_path, _addfile_path)
            if _plist and len(_plist) >= num_machines:
                case_default_pm = [float(_plist[i]) for i in range(num_machines)]
                print(f"Pm from case (.raw/.dyr): {[f'{p:.4f}' for p in case_default_pm]}")
            elif _plist and len(_plist) > 0:
                # Pad or trim to num_machines
                case_default_pm = [
                    float(_plist[i]) if i < len(_plist) else float(_plist[-1])
                    for i in range(num_machines)
                ]
                print(
                    f"Pm from case (.raw/.dyr): {[f'{p:.4f}' for p in case_default_pm]} (padded/trimmed to {num_machines} machines)"
                )
        except Exception as e:
            print(f"[INFO] Could not read Pm from case file; will use config or fallback: {e}")

    # Per-machine: H and D from config; Pm_ranges always None for multimachine (Pm from load/alpha via power flow).
    H_ranges = [H_range] * num_machines if H_range else None
    D_ranges = [D_range] * num_machines if D_range else None
    Pm_ranges = None  # multimachine: Pm determined by load (alpha), not a sweep variable

    print("=" * 70)
    print("Multimachine data generation (Kundur 2-area)")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Output: {run_dir} (timestamped run folder)")
    print(
        f"Case: {case_file}"
        + (f"  addfile: {addfile}" if addfile else "")
        + f"  |  Machines: {num_machines}"
    )
    print(
        f"Sampling: {sampling_strategy}  n_samples={n_samples}  CCT-based={use_cct_based_sampling}"
    )
    print("Focus variables: H, D, alpha (Pm from load / power flow)")
    print(f"  alpha={alpha_range[0]:.2f}-{alpha_range[1]:.2f}, n={alpha_range[2]}")
    print(
        f"Fault: bus={fault_bus}  t_start={fault_start_time}s  X={fault_reactance}"
        + ("  [skip_fault=True: no fault]" if skip_fault else "")
    )
    print("-" * 70)

    from data_generation.parameter_sweep import generate_parameter_sweep_multimachine

    df = generate_parameter_sweep_multimachine(
        case_file=case_file,
        output_dir=str(run_dir),
        num_machines=num_machines,
        addfile=addfile,
        H_ranges=H_ranges,
        D_ranges=D_ranges,
        Pm_ranges=Pm_ranges,
        fault_clearing_times=fault_clearing_times,
        fault_locations=[fault_bus],
        simulation_time=simulation_time,
        time_step=time_step,
        verbose=True,
        sampling_strategy=sampling_strategy,
        task="trajectory",
        n_samples=n_samples,
        seed=seed,
        validate_quality=True,
        use_cct_based_sampling=use_cct_based_sampling,
        n_samples_per_combination=n_samples_per_combination,
        cct_offsets=cct_offsets,
        fault_start_time=fault_start_time,
        fault_bus=fault_bus,
        fault_reactance=fault_reactance,
        use_pe_as_input=use_pe_as_input,
        alpha_range=alpha_range,
        base_load=base_load,
        case_default_pm=case_default_pm,
        skip_fault=skip_fault,
    )

    print("-" * 70)
    print("Multimachine data generation completed.")
    print(f"  Rows: {len(df)}")
    if "scenario_id" in df.columns:
        print(f"  Scenarios: {df['scenario_id'].nunique()}")
    out_files = list(run_dir.glob("parameter_sweep_data_*.csv"))
    latest_csv = max(out_files, key=lambda p: p.stat().st_mtime) if out_files else None
    if latest_csv:
        print(f"  File: {latest_csv.name}")
    print(f"  Run folder: {run_dir}")

    # Automatic data analysis (report + figures in run_dir/analysis)
    if not args.skip_analysis and len(df) > 0 and latest_csv is not None:
        analysis_dir = run_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        print()
        print("-" * 70)
        print("Running data analysis (report and figures in output/analysis)...")
        print("-" * 70)
        analyze_script = project_root / "scripts" / "analyze_multimachine_data.py"
        cmd = [
            sys.executable,
            str(analyze_script),
            str(latest_csv),
            "--plot",
            "--output-dir",
            str(analysis_dir),
            "--num-machines",
            str(num_machines),
        ]
        rc = subprocess.run(cmd, cwd=str(project_root))
        if rc.returncode != 0:
            print("[Warning] Data analysis exited with code", rc.returncode)
        else:
            print("Analysis results saved in:", analysis_dir)
        print("-" * 70)

    print()
    print("Next: preprocess and train:")
    print(
        f"  python scripts/preprocess_data.py --data-path {run_dir}/parameter_sweep_data_*.csv ..."
    )
    print(
        f"  python training/train_multimachine_pe_input.py --data-dir <processed_dir> --num-machines {num_machines}"
    )


if __name__ == "__main__":
    main()
