"""
Data Generation Core Module.

This module wraps data generation functions with checkpointing and validation.
Used by main scripts (generate_data.py, run_experiment.py).
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.parameter_sweep import (
    generate_parameter_estimation_data,
    generate_trajectory_data,
)
from utils.physics_validation import find_smib_case, run_all_validations

from .common_repository import (
    find_data_by_fingerprint,
    find_latest_data,
    save_data_to_common,
    validate_data_integrity,
)
from .fingerprinting import compute_data_fingerprint
from .utils import generate_timestamped_filename, save_json
from datetime import datetime


def generate_training_data(
    config: Dict,
    output_dir: Path,
    validate_physics: bool = True,
    skip_if_exists: bool = False,
    use_common_repository: bool = True,
    force_regenerate: bool = False,
) -> Tuple[Path, Optional[Dict]]:
    """
    Generate training data based on configuration.

    Parameters:
    -----------
    config : dict
        Configuration dictionary with data generation parameters
    output_dir : Path
        Directory to save generated data (used if use_common_repository=False)
    validate_physics : bool
        Whether to validate physics constraints (default: True)
    skip_if_exists : bool
        If True, skip generation if data file already exists (default: False)
    use_common_repository : bool
        If True, use common repository for data storage (default: True)
    force_regenerate : bool
        If True, force regeneration even if data exists in common repository (default: False)

    Returns:
    --------
    data_path : Path
        Path to generated data file
    validation_results : dict or None
        Physics validation results, or None if validation skipped
    """
    # Determine task type
    task = config.get("data", {}).get("task", "trajectory")

    # Note: CCT estimation uses trajectory models via bisection, no separate data generation needed
    if task == "cct":
        print("⚠️  Note: CCT estimation uses trajectory models with bisection method.")
        print("  No separate CCT data generation needed. Using trajectory task instead.")
        task = "trajectory"
        config["data"]["task"] = "trajectory"

    # Check common repository first if enabled
    if use_common_repository and not force_regenerate:
        fingerprint = compute_data_fingerprint(config)
        existing_path = find_data_by_fingerprint(fingerprint, task)

        if existing_path and existing_path.exists():
            # Validate integrity
            is_valid, error_msg = validate_data_integrity(existing_path, verify_checksum=True)
            if is_valid:
                print(f"✓ Found existing data in common repository: {existing_path.name}")
                print("  Reusing existing data (use --force-regenerate to override)")

                # Still validate physics if requested
                validation_results = None
                if validate_physics:
                    print("\nValidating existing data physics...")
                    data = pd.read_csv(existing_path)
                    case_file = (
                        config.get("data", {})
                        .get("generation", {})
                        .get("case_file", "smib/SMIB.json")
                    )
                    validation_results = _validate_data_physics(data, case_file)

                return existing_path, validation_results
            else:
                print(f"⚠️  Existing data found but validation failed: {error_msg}")
                print("  Regenerating data...")
        else:
            # No exact fingerprint match - try to find most recent data with matching n_samples
            # This is a fallback to reuse similar data instead of always regenerating
            gen_config = config.get("data", {}).get("generation", {})
            n_samples = gen_config.get("n_samples")
            if n_samples is not None:
                # Try to find latest data with matching n_samples and task
                latest_path = find_latest_data(task, n_samples=n_samples)
                if latest_path and latest_path.exists():
                    # Validate integrity
                    is_valid, error_msg = validate_data_integrity(latest_path, verify_checksum=True)
                    if is_valid:
                        print(
                            f"✓ Found similar data (n_samples={n_samples}) in common repository: {latest_path.name}"
                        )
                        print(f"  Using most recently generated data (exact fingerprint not found)")
                        print("  (Use --force-regenerate to generate new data with exact config)")

                        # Still validate physics if requested
                        validation_results = None
                        if validate_physics:
                            print("\nValidating existing data physics...")
                            data = pd.read_csv(latest_path)
                            case_file = (
                                config.get("data", {})
                                .get("generation", {})
                                .get("case_file", "smib/SMIB.json")
                            )
                            validation_results = _validate_data_physics(data, case_file)

                        return latest_path, validation_results
                    else:
                        print(f"⚠️  Similar data found but validation failed: {error_msg}")
                        print("  Regenerating data...")

    # Fallback to output_dir if common repository not used
    if not use_common_repository:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get data generation parameters
    data_config = config.get("data", {})
    gen_config = data_config.get("generation", {})

    # Extract base_load from config (optional)
    base_load_config = gen_config.get("base_load", None)
    base_load = None
    if base_load_config is not None:
        if isinstance(base_load_config, dict):
            base_load = {
                "Pload": base_load_config.get(
                    "Pload", base_load_config.get("P", None)
                ),  # Support both Pload and P for backward compatibility
                "Qload": base_load_config.get(
                    "Qload", base_load_config.get("Q", None)
                ),  # Support both Qload and Q for backward compatibility
            }
            # Only use if Pload is valid
            if base_load["Pload"] is None or base_load["Pload"] <= 0:
                base_load = None

    # Extract parameter ranges
    param_ranges = gen_config.get("parameter_ranges", {})

    # Get n_samples for Sobol/LHS sampling (needed for 2-element format conversion)
    n_samples_cfg = gen_config.get("n_samples", None)

    # Convert to format expected by data generation functions
    # Handle both list [min, max, num_points] and tuple (min, max, num_points) formats
    # Prefer M (inertia coefficient) over H (inertia constant) for consistency
    # M is what's stored in case file, H is derived (H = M/2 for 60 Hz)

    # Check for M_range first (preferred)
    M_val = param_ranges.get("M", None)
    H_range = None  # Will be set from M if M is provided

    if M_val is not None:
        # M_range provided - convert to H_range for internal processing (H = M/2)
        if isinstance(M_val, list) and len(M_val) == 3:
            M_min, M_max, M_n = M_val
            # Convert M range to H range (H = M/2 for 60 Hz systems)
            H_range = (M_min / 2.0, M_max / 2.0, M_n)
        elif isinstance(M_val, tuple) and len(M_val) == 3:
            M_min, M_max, M_n = M_val
            H_range = (M_min / 2.0, M_max / 2.0, M_n)
    else:
        # No M_range - check for H_range (backward compatibility)
        H_val = param_ranges.get("H", None)
        if H_val is None:
            H_range = None  # Use case file default
        elif isinstance(H_val, (list, tuple)):
            if len(H_val) == 3:
                # Grid sampling format: [min, max, n_values]
                H_range = tuple(H_val)
            elif len(H_val) == 2:
                # Sobol/LHS format: [min, max] - add n_samples as third element
                H_range = tuple(H_val) + (n_samples_cfg,)
            else:
                H_range = None  # Invalid format
        else:
            H_range = None  # Use case file default if invalid

    D_val = param_ranges.get("D", None)
    if D_val is None:
        D_range = None  # Use case file default
    elif isinstance(D_val, (list, tuple)):
        if len(D_val) == 3:
            # Grid sampling format: [min, max, n_values]
            D_range = tuple(D_val)
        elif len(D_val) == 2:
            # Sobol/LHS format: [min, max] - add n_samples as third element
            D_range = tuple(D_val) + (n_samples_cfg,)
        else:
            D_range = None  # Invalid format
    else:
        D_range = None  # Use case file default if invalid

    # Support both Pm variation (old) and load variation (new)
    Pm_val = param_ranges.get("Pm", None)
    if Pm_val is not None:
        if isinstance(Pm_val, (list, tuple)):
            if len(Pm_val) == 3:
                # Grid sampling format: [min, max, n_values]
                Pm_range = tuple(Pm_val)
            elif len(Pm_val) == 2:
                # Sobol/LHS format: [min, max] - add n_samples as third element
                Pm_range = tuple(Pm_val) + (n_samples_cfg,)
            else:
                Pm_range = None  # Invalid format
        else:
            Pm_range = None
    else:
        Pm_range = None

    # NEW: Support alpha variation (unified approach for SMIB and multimachine)
    alpha_val = param_ranges.get("alpha", None)
    if alpha_val is not None:
        if isinstance(alpha_val, (list, tuple)):
            if len(alpha_val) == 3:
                # Full format: (min, max, num_points)
                alpha_range = tuple(alpha_val)
            elif len(alpha_val) == 2:
                # Short format: [min, max] - default to 2 points for range
                alpha_min, alpha_max = alpha_val
                alpha_range = (alpha_min, alpha_max, 2)
            else:
                alpha_range = None
        else:
            alpha_range = None
    else:
        alpha_range = None

    # Backward compatibility: Support load variation (DEPRECATED - use alpha_range instead)
    load_val = param_ranges.get("load", None)
    if load_val is not None:
        if isinstance(load_val, list) and len(load_val) == 3:
            load_range = tuple(load_val)
        elif isinstance(load_val, tuple) and len(load_val) == 3:
            load_range = load_val
        else:
            load_range = load_val if isinstance(load_val, tuple) else None
    else:
        load_range = None

    load_q_val = param_ranges.get("load_q", None)
    if load_q_val is not None:
        if isinstance(load_q_val, list) and len(load_q_val) == 3:
            load_q_range = tuple(load_q_val)
        elif isinstance(load_q_val, tuple) and len(load_q_val) == 3:
            load_q_range = load_q_val
        else:
            load_q_range = load_q_val if isinstance(load_q_val, tuple) else None
    else:
        load_q_range = None

    # Determine if using load variation
    use_load_variation = gen_config.get("parameter_ranges", {}).get("use_load_variation", False)
    if use_load_variation and load_range is None:
        # If use_load_variation=True but no load_range, fall back to Pm_range if available
        if Pm_range is not None:
            print(
                "[WARNING] use_load_variation=True but no load_range provided. Using Pm_range as load_range."
            )
            load_range = Pm_range
            Pm_range = None
        else:
            raise ValueError(
                "use_load_variation=True requires load_range to be specified in parameter_ranges"
            )

    # Get case file
    case_file = gen_config.get("case_file", "smib/SMIB.json")

    # Get sampling strategy
    sampling_strategy = gen_config.get("sampling_strategy", "full_factorial")
    n_samples = gen_config.get("n_samples", None)

    # Get simulation parameters
    simulation_time = gen_config.get("simulation_time", 5.0)
    time_step = gen_config.get("time_step", 0.001)

    # Get fault parameters
    fault_config = gen_config.get("fault", {})
    fault_start_time = fault_config.get("start_time", 1.0)
    fault_bus = fault_config.get("bus", 3)
    fault_reactance = fault_config.get("reactance", 0.0001)
    fault_clearing_times = gen_config.get("clearing_times", None)

    # Get CCT-based sampling settings
    use_cct_based_sampling = gen_config.get("use_cct_based_sampling", False)
    n_samples_per_combination = gen_config.get("n_samples_per_combination", 5)
    cct_offsets = gen_config.get("additional_clearing_time_offsets", None)

    # Get input method setting (for Pe(t) extraction)
    # Check both model config and data generation config for compatibility
    use_pe_as_input = gen_config.get("use_pe_as_input", False)
    if not use_pe_as_input:
        # Fallback: check model config if not in data generation config
        model_config = config.get("model", {})
        input_method = model_config.get("input_method", "reactance")
        if input_method in ("pe_direct", "pe_direct_7"):
            use_pe_as_input = True
            print(
                "[INFO] Model config uses Pe-direct input method. "
                "Enabling Pe(t) extraction in data generation."
            )

    # Get random seed
    seed = config.get("reproducibility", {}).get("random_seed", None)

    # Check if data already exists in output_dir (legacy behavior)
    if not use_common_repository:
        data_path = output_dir / f"{task}_data.csv"
        if skip_if_exists and data_path.exists():
            print(f"✓ Data file already exists: {data_path}")
            print("  Skipping data generation (use skip_if_exists=False to regenerate)")

            # Still validate if requested
            validation_results = None
            if validate_physics:
                print("\nValidating existing data...")
                data = pd.read_csv(data_path)
                case_file = (
                    config.get("data", {}).get("generation", {}).get("case_file", "smib/SMIB.json")
                )
                validation_results = _validate_data_physics(data, case_file)

            return data_path, validation_results

    print("=" * 70)
    print("DATA GENERATION")
    print("=" * 70)
    print(f"Task: {task}")
    print(f"Output: {output_dir}")
    print(f"Sampling strategy: {sampling_strategy}")

    # Generate data based on task
    if task == "trajectory":
        print("\nGenerating trajectory prediction data...")
        data = generate_trajectory_data(
            case_file=case_file,
            output_dir=str(output_dir),
            H_range=H_range,
            D_range=D_range,
            Pm_range=Pm_range,
            alpha_range=alpha_range,  # NEW: Unified approach
            load_q_alpha_range=None,  # Optional: independent Q scaling (default: same as alpha)
            # Backward compatibility:
            load_range=load_range,
            load_q_range=load_q_range,
            use_load_variation=use_load_variation,
            fault_clearing_times=fault_clearing_times,
            simulation_time=simulation_time,
            time_step=time_step,
            sampling_strategy=sampling_strategy,
            n_samples=n_samples,
            seed=seed,
            verbose=True,
            use_cct_based_sampling=use_cct_based_sampling,
            n_samples_per_combination=n_samples_per_combination,
            cct_offsets=cct_offsets,
            fault_start_time=fault_start_time,
            fault_bus=fault_bus,
            fault_reactance=fault_reactance,
            use_pe_as_input=use_pe_as_input,
            base_load=base_load,  # Pass base_load from config
        )
    elif task == "parameter_estimation":
        print("\nGenerating parameter estimation data...")
        data = generate_parameter_estimation_data(
            case_file=case_file,
            output_dir=str(output_dir),
            H_range=H_range,
            D_range=D_range,
            fault_clearing_times=fault_clearing_times or [0.15, 0.18, 0.20, 0.22, 0.25],
            simulation_time=simulation_time,
            time_step=time_step,
            n_samples=n_samples,
            seed=seed,
            verbose=True,
        )
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'trajectory' or 'parameter_estimation'")

    # Save data to appropriate location
    if use_common_repository:
        # Save to common repository with fingerprinting and metadata
        data_path, metadata = save_data_to_common(
            data, task, config, force_regenerate=force_regenerate
        )
        print(f"  Total samples: {len(data)}")
        unique_scenarios = (
            len(data["scenario_id"].unique()) if "scenario_id" in data.columns else "N/A"
        )
        print(f"Unique scenarios: {unique_scenarios}")
        print(f"  Data fingerprint: {metadata.get('data_fingerprint', 'N/A')[:16]}...")
    else:
        # Legacy: Save to output_dir with timestamp
        data_filename = generate_timestamped_filename(f"{task}_data", "csv")
        data_path = output_dir / data_filename
        data.to_csv(data_path, index=False)
        print(f"\n✓ Data saved to: {data_path}")
        print(f"  Total samples: {len(data)}")
        unique_scenarios = (
            len(data["scenario_id"].unique()) if "scenario_id" in data.columns else "N/A"
        )
        print(f"Unique scenarios: {unique_scenarios}")

    # Validate physics if requested
    validation_results = None
    if validate_physics:
        print("\n" + "=" * 70)
        print("PHYSICS VALIDATION")
        print("=" * 70)
        validation_results = _validate_data_physics(data, case_file)

        # Save validation results
        if validation_results:
            if use_common_repository:
                # Save validation results next to data file
                validation_path = data_path.parent / f"{data_path.stem}_validation.json"
            else:
                validation_path = output_dir / "data_validation.json"
            save_json(validation_results, validation_path)
            print(f"\n✓ Validation results saved to: {validation_path}")

    return data_path, validation_results


def _validate_data_physics(data: pd.DataFrame, case_file: str) -> Optional[Dict]:
    """
    Validate data physics constraints.

    Parameters:
    -----------
    data : pd.DataFrame
        Generated data
    case_file : str
        Path to case file

    Returns:
    --------
    validation_results : dict or None
        Validation results, or None if validation fails
    """
    try:
        # Find SMIB case file
        case_path = find_smib_case()
        if not case_path or not case_path.exists():
            print("⚠️  Could not find SMIB case file, skipping physics validation")
            return None

        # Run validations
        results = run_all_validations(data, str(case_path))

        # Check if validation passed
        all_valid = all(
            [
                results.get("omega_valid", False) == True,
                results.get("power_balance_valid", False) == True,
                results.get("system_frequency") in [50.0, 60.0],
            ]
        )

        if all_valid:
            print("✓ Physics validation PASSED")
        else:
            print("⚠️  Physics validation FAILED")
            print("  Some checks did not pass, but continuing...")

        return results

    except Exception as e:
        print(f"⚠️  Error during physics validation: {e}")
        print("  Continuing without validation...")
        return None
