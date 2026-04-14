#!/usr/bin/env python
"""
Run SMIB P_m Variation Study from YAML Configuration.

This script loads a scenario matrix from YAML config and executes P_m variation
batch TDS simulations with CCT-based sampling.

Usage:
    python scripts/run_smib_pm_variation_study.py --config configs/experiments/smib_pm_variation_scenarios.yaml
    python scripts/run_smib_pm_variation_study.py --config configs/experiments/smib_pm_variation_scenarios.yaml --output-dir examples/data/pinn_training/pm_variation_test
"""

import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from examples.smib_batch_tds import batch_tds_smib, save_batch_tds_results


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_experiment_metadata(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """Create experiment metadata for reproducibility."""
    import platform
    import sys
    import hashlib
    import random

    # Compute config checksum
    config_str = str(config)
    config_checksum = hashlib.md5(config_str.encode()).hexdigest()

    # Generate random seed if not in config
    random_seed = config.get("reproducibility", {}).get("random_seed", random.randint(1, 10000))

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config_file": str(config_path),
        "config_checksum": config_checksum,
        "random_seed": random_seed,
        "system_info": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
            },
        },
        "config": config,
    }

    # Try to get ANDES version
    try:
        import andes

        metadata["andes_version"] = getattr(andes, "__version__", "unknown")
        metadata["andes_path"] = (
            str(Path(andes.__file__).parent) if hasattr(andes, "__file__") else "unknown"
        )
    except ImportError:
        metadata["andes_version"] = "not_available"
        metadata["andes_path"] = "not_available"

    # Try to get numpy version
    try:
        import numpy

        metadata["numpy_version"] = numpy.__version__
    except ImportError:
        metadata["numpy_version"] = "not_available"

    # Try to get pandas version
    try:
        import pandas

        metadata["pandas_version"] = pandas.__version__
    except ImportError:
        metadata["pandas_version"] = "not_available"

    return metadata


def save_metadata(metadata: Dict[str, Any], output_dir: Path):
    """Save experiment metadata to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "experiment_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[METADATA] Saved to: {metadata_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run SMIB P_m Variation Study from YAML Configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config if provided)",
    )
    parser.add_argument(
        "--case-file",
        type=str,
        default=None,
        help="ANDES case file path (overrides config if provided)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Configuration file not found: {config_path}")
        return 1

    print("=" * 70)
    print("SMIB P_m Variation Study - Scenario Runner")
    print("=" * 70)
    print(f"Configuration: {config_path}")
    print()

    config = load_config(config_path)

    # Extract configuration parameters
    pm_config = config.get("pm_variation", {})
    fault_config = config.get("fault_config", {})
    inertia_config = config.get("inertia_sensitivity", {})
    cct_config = config.get("cct_config", {})
    sampling_config = config.get("sampling", {})
    sim_config = config.get("simulation", {})
    export_config = config.get("export", {})

    # Prepare batch_tds_smib arguments
    case_file = args.case_file or "smib/SMIB.json"
    pm_range = tuple(pm_config.get("pm_range", [0.3, 0.85, 15]))

    # Fault parameters
    fault_start_time = fault_config.get("fault_start_time", 1.0)
    fault_bus = fault_config.get("fault_bus", 3)
    fault_reactance = fault_config.get("fault_reactance", 0.0001)

    # Inertia parameters
    H = inertia_config.get("H_value", 5.0)
    D = inertia_config.get("D_value", 1.0)

    # Simulation parameters
    simulation_time = sim_config.get("simulation_time", 5.0)
    time_step = sim_config.get("time_step", 0.002)
    use_skip_init = sim_config.get("use_skip_init", True)

    # Sampling parameters
    use_cct_based_sampling = sampling_config.get("strategy", "hybrid") != "uniform"
    n_samples_per_combination = 5  # Default, can be computed from sampling config

    # Compute n_samples_per_combination from sampling config
    if sampling_config.get("strategy") == "hybrid":
        n_total = 0
        if sampling_config.get("cct_boundary", {}).get("enabled", False):
            n_total += sampling_config["cct_boundary"].get("n_samples", 5)
        if sampling_config.get("stable_region", {}).get("enabled", False):
            n_total += sampling_config["stable_region"].get("n_samples", 3)
        if sampling_config.get("unstable_region", {}).get("enabled", False):
            n_total += sampling_config["unstable_region"].get("n_samples", 2)
        if n_total > 0:
            n_samples_per_combination = n_total

    # CCT offsets (if specified in config)
    cct_offsets = None
    if "cct_boundary" in sampling_config and "offsets" in sampling_config["cct_boundary"]:
        cct_offsets = sampling_config["cct_boundary"]["offsets"]

    # Output directory
    output_dir = args.output_dir or export_config.get("output_dir", "examples/data/pinn_training")
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"pm_variation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[CONFIG] P_m range: [{pm_range[0]:.2f}, {pm_range[1]:.2f}] × P_max, {pm_range[2]} samples"
    )
    print(f"[CONFIG] H = {H:.2f} s, D = {D:.2f} pu")
    print(
        f"[CONFIG] Fault: bus {fault_bus}, start={fault_start_time:.2f}s, Xf={fault_reactance:.6f} pu"
    )
    print(f"[CONFIG] Simulation: t={simulation_time:.2f}s, dt={time_step:.4f}s")
    print(f"[CONFIG] Output: {output_dir}")
    print()

    # Create experiment metadata
    metadata = create_experiment_metadata(config, config_path)
    save_metadata(metadata, output_dir)

    # Run batch TDS
    print("=" * 70)
    print("Running Batch TDS with P_m Variation")
    print("=" * 70)
    print()

    try:
        results = batch_tds_smib(
            case_file=case_file,
            variation_mode="pm",
            pm_range=pm_range,
            n_samples=pm_range[2],
            fault_start_time=fault_start_time,
            fault_bus=fault_bus,
            fault_reactance=fault_reactance,
            simulation_time=simulation_time,
            time_step=time_step,
            use_cct_based_sampling=use_cct_based_sampling,
            n_samples_per_combination=n_samples_per_combination,
            cct_offsets=cct_offsets,
            use_skip_init=use_skip_init,
            H=H,
            D=D,
            verbose=args.verbose,
        )

        if results is None:
            print("[ERROR] Batch TDS returned None. Exiting.")
            return 1

        # Check if results are empty
        n_samples_generated = len([x for x in results.get("pm_values", []) if x is not None])
        if n_samples_generated == 0:
            print("[ERROR] No results generated. Exiting.")
            return 1

        print()
        print("=" * 70)
        print("Saving Results")
        print("=" * 70)

        # Save results
        export_format = export_config.get("format", "both")
        file_paths = save_batch_tds_results(
            results,
            output_dir=str(output_dir),
            format=export_format,
            data_version="1.0.0",
            fault_start_time=fault_start_time,
            fault_bus=fault_bus,
            fault_reactance=fault_reactance,
        )

        print(f"[OK] Results saved to: {output_dir}")
        print(f"[OK] Files: {file_paths}")

        # Compute error and data quality statistics
        errors = results.get("errors", [])
        data_quality = results.get("data_quality", [])

        n_errors = len([e for e in errors if e is not None])
        error_summary = {}
        if n_errors > 0:
            error_types = {}
            for e in errors:
                if e is not None:
                    error_type = e.split(":")[0] if ":" in e else "unknown"
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            error_summary = {
                "total_errors": n_errors,
                "error_types": error_types,
            }

        quality_summary = {}
        if data_quality:
            quality_counts = {}
            for q in data_quality:
                quality_counts[q] = quality_counts.get(q, 0) + 1
            quality_summary = {
                "quality_distribution": quality_counts,
                "pass_count": quality_counts.get("pass", 0),
                "total_checked": len(data_quality),
            }

        # Update metadata with results summary
        metadata["results_summary"] = {
            "n_samples": n_samples_generated,
            "n_stable": sum(results.get("stability_status", [])),
            "n_unstable": len(results.get("stability_status", []))
            - sum(results.get("stability_status", [])),
            "output_dir": str(output_dir),
            "files": {k: str(v) for k, v in file_paths.items()},
            "errors": error_summary,
            "data_quality": quality_summary,
        }
        save_metadata(metadata, output_dir)

        # Print error and quality summary
        if n_errors > 0:
            print(f"\n[ERROR SUMMARY] {n_errors} errors encountered:")
            for error_type, count in error_summary.get("error_types", {}).items():
                print(f"  - {error_type}: {count}")

        if quality_summary:
            print(f"\n[DATA QUALITY] Validation results:")
            print(
                f"  - Passed: {quality_summary.get('pass_count', 0)}/{quality_summary.get('total_checked', 0)}"
            )
            for quality_type, count in quality_summary.get("quality_distribution", {}).items():
                if quality_type != "pass":
                    print(f"  - {quality_type}: {count}")

        print()
        print("=" * 70)
        print("Study Completed Successfully!")
        print("=" * 70)
        print(f"Output directory: {output_dir}")
        print(f"Generated {n_samples_generated} samples")
        print(f"Stable: {metadata['results_summary']['n_stable']}")
        print(f"Unstable: {metadata['results_summary']['n_unstable']}")

        return 0

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
