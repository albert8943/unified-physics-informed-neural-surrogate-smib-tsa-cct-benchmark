#!/usr/bin/env python
"""
Data Generation Script

Generate training datasets using YAML configuration files.
Works with quick_test, moderate_test, comprehensive_test, or custom configs.

Usage:
    # Using pre-configured levels
    python scripts/generate_data.py --level quick
    python scripts/generate_data.py --level moderate
    python scripts/generate_data.py --level comprehensive

    # Using custom config file
    python scripts/generate_data.py --config configs/data_generation/my_custom_config.yaml
"""

import argparse
import subprocess
import sys
import warnings
import logging
from datetime import datetime
from pathlib import Path

# Suppress ANDES warnings for cleaner output (do this early, before ANDES imports)
warnings.filterwarnings("ignore", category=UserWarning, module="andes")
warnings.filterwarnings("ignore", message=".*vf range.*")
warnings.filterwarnings("ignore", message=".*GENCLS.*")
warnings.filterwarnings("ignore", message=".*typical.*limit.*")
warnings.filterwarnings("ignore", message=".*Time step reduced.*")
warnings.filterwarnings("ignore", message=".*Convergence.*")

# Set ANDES logger to ERROR level to suppress INFO/WARNING messages
# This must be done before ANDES is imported
logging.getLogger("andes").setLevel(logging.ERROR)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Generate test dataset using pre-configured configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (default, ~1 hour, 100 trajectories)
  python scripts/generate_data.py
  
  # Moderate test (~2.5 hours, 250 trajectories)
  python scripts/generate_data.py --level moderate
  
  # Comprehensive test (~10 hours, 1000 trajectories)
  python scripts/generate_data.py --level comprehensive
  
  # Custom config file
  python scripts/generate_data.py --config configs/data_generation/my_config.yaml
  
  # Custom output directory
  python scripts/generate_data.py --level quick --output data/generated/my_test
        """,
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["quick", "moderate", "comprehensive"],
        default=None,
        help="Pre-configured test level: quick, moderate, or comprehensive (alternative to --config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config file (alternative to --level)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/generated/{level}_test)",
    )
    parser.add_argument(
        "--timestamp-run",
        action="store_true",
        help="Create a timestamped subfolder exp_YYYYMMDD_HHMMSS for this run (like multimachine)",
    )
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run analyze_data.py on the generated CSV after generation (output in analysis/ subdir)",
    )

    args = parser.parse_args()

    # Determine config file
    if args.config:
        # Use custom config
        config_file = Path(args.config)
        if not config_file.is_absolute():
            config_file = project_root / config_file
        level_info = {
            "default_output": "data/generated/custom",
            "samples": "Custom",
            "time": "Unknown",
        }
    elif args.level:
        # Use pre-configured level
        config_map = {
            "quick": {
                "config": "configs/data_generation/quick.yaml",
                "default_output": "data/generated/quick_test",
                "samples": "100 trajectories (20 unique combinations × 5)",
                "time": "~1 hour",
            },
            "moderate": {
                "config": "configs/data_generation/moderate.yaml",
                "default_output": "data/generated/moderate_test",
                "samples": "250 trajectories (50 unique combinations × 5)",
                "time": "~2.5 hours",
            },
            "comprehensive": {
                "config": "configs/data_generation/comprehensive.yaml",
                "default_output": "data/generated/comprehensive_test",
                "samples": "1000 trajectories (200 unique combinations × 5)",
                "time": "~10 hours",
            },
        }
        level_info = config_map[args.level]
        config_file = project_root / level_info["config"]
    else:
        # Default to quick
        args.level = "quick"
        config_map = {
            "quick": {
                "config": "configs/data_generation/quick.yaml",
                "default_output": "data/generated/quick_test",
                "samples": "100 trajectories (20 unique combinations × 5)",
                "time": "~1 hour",
            },
        }
        level_info = config_map[args.level]
        config_file = project_root / level_info["config"]

    output_base = Path(args.output or level_info["default_output"])
    if not output_base.is_absolute():
        output_base = project_root / output_base
    output_base.mkdir(parents=True, exist_ok=True)
    if args.timestamp_run:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_base / f"exp_{run_ts}"
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = output_base

    # Check if config exists
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        print("   Please ensure the config file exists.")
        sys.exit(1)

    print("=" * 70)
    level_name = args.level.capitalize() if args.level else "Custom"
    print(f"Data Generation - {level_name}")
    print("=" * 70)
    print(f"Config file: {config_file}")
    print(f"Output directory: {output_path}")
    print()
    print("Expected results:")
    print(f"  - Samples: {level_info['samples']}")
    print(f"  - Time: {level_info['time']}")
    print()
    print("Generating data...")
    print("-" * 70)

    try:
        # Load config file
        import yaml

        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Extract parameters from config
        system_config = config.get("system", {})
        case_file = system_config.get("case_file", "smib/SMIB.json")

        param_ranges = config.get("parameter_ranges", {})
        # Check for load variation first (new default), fallback to Pm for backward compatibility
        if param_ranges.get("use_load_variation", False) and "load" in param_ranges:
            load_values = param_ranges.get("load", [0.6, 0.9])
            Pm_values = None  # Not used in load variation mode
        else:
            load_values = None
            Pm_values = param_ranges.get("Pm", [0.6, 0.8])  # Fallback to Pm
        M_values = param_ranges.get("M", [5.0, 7.0])
        D_values = param_ranges.get("D", [1.0])

        sampling_config = config.get("sampling", {})
        sampling_strategy = sampling_config.get("strategy", "full_factorial")
        # None for full factorial, number for Sobol/LHS
        n_samples = sampling_config.get("n_samples", None)
        n_samples_per_combination = sampling_config.get("n_samples_per_combination", 1)
        use_cct_based_sampling = sampling_config.get("use_cct_based_sampling", True)
        cct_offsets = sampling_config.get("additional_clearing_time_offsets", [0.0])

        fault_config = config.get("fault", {})
        fault_start_time = fault_config.get("start_time", 1.0)
        fault_bus = fault_config.get("bus", 3)
        fault_reactance = fault_config.get("reactance", 0.0001)

        sim_config = config.get("simulation", {})
        simulation_time = sim_config.get("time", 5.0)
        time_step = sim_config.get("time_step", 0.001)

        # Convert parameter lists to ranges (min, max, num_points) format
        # For Sobol/LHS: use (min, max) ranges, but num_points > 1 is needed to enable 3D sampling
        # For full_factorial: num_points determines grid size
        use_load_variation = (
            param_ranges.get("use_load_variation", False) and load_values is not None
        )

        # Handle load variation (new default)
        if use_load_variation:
            if isinstance(load_values, list):
                if len(load_values) == 3:
                    load_range = tuple(load_values)
                elif sampling_strategy in ["sobol", "lhs"] and n_samples is not None:
                    load_range = (min(load_values), max(load_values), 2)  # > 1 enables 3D sampling
                else:
                    load_range = (min(load_values), max(load_values), len(load_values))
            elif isinstance(load_values, tuple) and len(load_values) == 3:
                load_range = load_values
            else:
                load_range = load_values if load_values is not None else (0.4, 0.9, 2)
            Pm_range = None  # Not used in load variation mode
        # Handle Pm variation (backward compatibility)
        else:
            load_range = None
            if isinstance(Pm_values, list):
                # Check if it's already in [min, max, num_points] format (3 elements)
                if len(Pm_values) == 3:
                    # Treat as (min, max, num_points) tuple
                    Pm_range = tuple(Pm_values)
                elif sampling_strategy in ["sobol", "lhs"] and n_samples is not None:
                    # For Sobol/LHS, set num_points > 1 to enable 3D sampling (Pm will be varied)
                    # The actual count is controlled by n_samples, but we need Pm_n > 1 to trigger 3D sampling
                    Pm_range = (min(Pm_values), max(Pm_values), 2)  # > 1 enables 3D sampling
                else:
                    # For full factorial, use list length
                    Pm_range = (min(Pm_values), max(Pm_values), len(Pm_values))
            elif isinstance(Pm_values, tuple) and len(Pm_values) == 3:
                # Already in tuple format
                Pm_range = Pm_values
            else:
                Pm_range = Pm_values

        if isinstance(M_values, list):
            # Check if it's already in [min, max, num_points] format (3 elements)
            if len(M_values) == 3:
                # Treat as (min, max, num_points) tuple, convert M to H
                H_range = (M_values[0] / 2.0, M_values[1] / 2.0, M_values[2])
            else:
                # M = 2H, so convert M to H range
                H_min = min(M_values) / 2.0
                H_max = max(M_values) / 2.0
                if sampling_strategy in ["sobol", "lhs"] and n_samples is not None:
                    # For Sobol/LHS, just use min/max, n_samples controls count
                    H_range = (H_min, H_max, 1)  # num_points ignored for Sobol
                else:
                    # For full factorial, use list length
                    H_range = (H_min, H_max, len(M_values))
        elif isinstance(M_values, tuple) and len(M_values) == 3:
            # Already in tuple format, convert M to H
            H_range = (M_values[0] / 2.0, M_values[1] / 2.0, M_values[2])
        else:
            H_range = (3.0, 5.0, 5)  # Default

        if isinstance(D_values, list):
            # Check if it's already in [min, max, num_points] format (3 elements)
            if len(D_values) == 3:
                # Treat as (min, max, num_points) tuple
                D_range = tuple(D_values)
            elif sampling_strategy in ["sobol", "lhs"] and n_samples is not None:
                # For Sobol/LHS, just use min/max, n_samples controls count
                D_range = (min(D_values), max(D_values), 1)  # num_points ignored for Sobol
            else:
                # For full factorial, use list length
                D_range = (min(D_values), max(D_values), len(D_values))
        elif isinstance(D_values, tuple) and len(D_values) == 3:
            # Already in tuple format
            D_range = D_values
        else:
            D_range = D_values

        # Import and call generate_parameter_sweep
        from data_generation.parameter_sweep import generate_parameter_sweep

        print("Calling generate_parameter_sweep...")
        print(f"  Case file: {case_file}")
        print(f"  Sampling strategy: {sampling_strategy}")
        if n_samples is not None:
            print(f"  n_samples: {n_samples}")
        print(f"  H_range: {H_range}")
        print(f"  D_range: {D_range}")
        if use_load_variation:
            print(f"  load_range: {load_range} (load variation mode)")
        else:
            print(f"  Pm_range: {Pm_range} (Pm variation mode)")
        print(f"  n_samples_per_combination: {n_samples_per_combination}")
        print(f"  Output dir: {output_path}")

        # Prepare arguments for generate_parameter_sweep
        sweep_kwargs = {
            "case_file": case_file,
            "output_dir": str(output_path),
            "H_range": H_range,  # H_range (M = 2H, so H = M/2)
            "D_range": D_range,
            "simulation_time": simulation_time,
            "time_step": time_step,
            "n_samples_per_combination": n_samples_per_combination,
            "use_cct_based_sampling": use_cct_based_sampling,
            "cct_offsets": cct_offsets if cct_offsets else None,  # None means auto-generate
            "fault_start_time": fault_start_time,
            "fault_bus": fault_bus,
            "fault_reactance": fault_reactance,
            "task": "trajectory",
            "sampling_strategy": sampling_strategy,
            "n_samples": n_samples,  # Pass n_samples for Sobol/LHS
            "verbose": True,
        }

        # Add load or Pm range based on mode
        if use_load_variation:
            sweep_kwargs["load_range"] = load_range
            sweep_kwargs["use_load_variation"] = True
            load_q = param_ranges.get("load_q", None)
            if load_q is not None:
                sweep_kwargs["load_q_range"] = (
                    load_q if isinstance(load_q, (list, tuple)) else (load_q, load_q, 1)
                )
        else:
            sweep_kwargs["Pm_range"] = Pm_range
            sweep_kwargs["use_load_variation"] = False

        df = generate_parameter_sweep(**sweep_kwargs)

        print("-" * 70)
        print("✅ Data generation completed!")
        print(f"   Config: {config_file.name}")
        print(f"   Output: {output_path}")
        print(f"   Generated {len(df)} data points")
        print()

        # Check what files were created
        output_files = list(output_path.glob("*.csv"))
        if output_files:
            print("   Files created:")
            for f in output_files:
                print(f"     - {f.name}")
        else:
            print(f"   ⚠️  No CSV files found in {output_path}")
            print("   Check for other file formats or errors above")

        # Optional: run analysis (like multimachine pipeline)
        data_files = list(output_path.glob("parameter_sweep_data_*.csv"))
        latest_csv = max(data_files, key=lambda p: p.stat().st_mtime) if data_files else None
        if args.run_analysis and len(df) > 0 and latest_csv is not None:
            analysis_dir = output_path / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            figures_dir = analysis_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            print()
            print("-" * 70)
            print("Running data analysis (scripts/analyze_data.py)...")
            print("-" * 70)
            analyze_script = project_root / "scripts" / "analyze_data.py"
            cmd = [
                sys.executable,
                str(analyze_script),
                str(latest_csv),
                "--output-dir",
                str(analysis_dir),
            ]
            rc = subprocess.run(cmd, cwd=str(project_root))
            if rc.returncode != 0:
                print("[Warning] Data analysis exited with code", rc.returncode)
            else:
                print("Analysis results saved in:", analysis_dir)
            print("-" * 70)

        print()
        print("Next steps:")
        if latest_csv is not None:
            file_size = latest_csv.stat().st_size / 1024  # Size in KB
            verify_cmd = (
                f'python -c "import pandas as pd; '
                f"df = pd.read_csv('{latest_csv}'); "
                f"print(f'{{len(df)}} samples')\""
            )
            print(f"   1. Verify data: {verify_cmd}")
            train_cmd = (
                f"python training/train_trajectory.py "
                f"--data_dir {output_path} --device cpu --epochs 5"
            )
            print(f"   2. Test training: {train_cmd}")
            print(f"   📁 Data file: {latest_csv} ({file_size:.1f} KB)")
        else:
            print(f"   ⚠️  No data files found in {output_path}")
            print(f"   1. Check output directory: {output_path}")
            print(f"   2. Look for files: ls {output_path}/parameter_sweep_data_*.csv")
            print(
                "   Note: Files are now timestamped "
                "(e.g., parameter_sweep_data_YYYYMMDD_HHMMSS.csv)"
            )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
