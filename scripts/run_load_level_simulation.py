#!/usr/bin/env python3
"""
Load Level Time Domain Simulation Script.

This script provides an easy-to-use interface for running ANDES time domain
simulations at different load levels for SMIB (and extensible to multimachine).
Supports both batch data generation and single analysis runs.

Usage:
    # Single simulation (analysis mode)
    python scripts/run_load_level_simulation.py \\
        --case smib/SMIB.json --load 0.7 --mode analysis --plot

    # Load level sweep (data generation mode) - saves to data/common/
    python scripts/run_load_level_simulation.py \\
        --case smib/SMIB.json --load-range 0.4 0.9 10 \\
        --mode batch --task trajectory
"""

import argparse
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress ANDES warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*vf range.*")
warnings.filterwarnings("ignore", message=".*typical.*limit.*")

# Disable tqdm progress bars to avoid ipywidgets dependency issues
os.environ["TQDM_DISABLE"] = "1"
os.environ["TQDM_NOTEBOOK"] = "0"

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False
    print("Warning: ANDES not available. Some functions may not work.")

# Import project modules
# Note: Imports must be after sys.path.insert to find project modules
from data_generation.andes_extractor import (  # noqa: E402
    extract_pe_trajectories,
    extract_trajectories,
)
from data_generation.andes_utils.case_file_modifier import (  # noqa: E402
    add_pq_load_to_smib_case,
    check_smib_has_load,
)
from data_generation.andes_utils.data_validator import (  # noqa: E402
    check_stability,
    validate_data_quality,
)
from data_generation.validation import generate_validation_report  # noqa: E402
from scripts.core.common_repository import save_data_to_common  # noqa: E402
from scripts.core.utils import generate_timestamped_filename  # noqa: E402

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable, **kwargs):
        """Dummy tqdm replacement when tqdm is not available."""
        return iterable


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def run_single_load_simulation(
    case_file: str,
    load_p: Optional[float] = None,  # DEPRECATED: Use alpha instead
    alpha: Optional[float] = None,  # NEW: Uniform load multiplier (recommended)
    load_q: Optional[float] = None,
    H: Optional[float] = None,
    D: Optional[float] = None,
    fault_start_time: float = 1.0,
    fault_clearing_time: Optional[float] = None,
    fault_bus: int = 3,
    fault_reactance: float = 0.0001,
    simulation_time: float = 5.0,
    time_step: Optional[float] = None,
    validate: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run one TDS simulation for a specific load level.

    Parameters:
    -----------
    case_file : str
        ANDES case file path
    load_p : float, optional
        DEPRECATED: Active power load (pu). Use alpha instead for unified approach.
    alpha : float, optional
        NEW: Uniform load multiplier (e.g., 0.7 = 70% of base load).
        If provided, base loads are scaled by alpha (maintains power factor).
        Recommended for unified SMIB and multimachine approach.
    load_q : float, optional
        Reactive power load (pu, default=0).
        Only used if load_p is provided (backward compatibility).
        If alpha is provided, Q is scaled by same alpha (uniform scaling).
    H : float, optional
        Inertia constant (seconds). If None, uses case file default.
    D : float, optional
        Damping coefficient (pu). If None, uses case file default.
    fault_start_time : float
        Fault start time (seconds)
    fault_clearing_time : float, optional
        Fault clearing time (seconds). If None, uses default (fault_start_time + 0.2)
    fault_bus : int
        Bus index where fault occurs
    fault_reactance : float
        Fault reactance (pu)
    simulation_time : float
        Total simulation time (seconds)
    time_step : float, optional
        Time step (seconds). If None, uses ANDES default.
    validate : bool
        Whether to perform validation checks
    verbose : bool
        Print progress information

    Returns:
    --------
    results : dict
        Dictionary with keys:
        - 'time': np.ndarray - Time array
        - 'delta': np.ndarray - Rotor angle trajectory
        - 'omega': np.ndarray - Speed trajectory
        - 'Pe': np.ndarray - Electrical power trajectory
        - 'V': np.ndarray - Voltage trajectory (if available)
        - 'metadata': dict - Comprehensive metadata
    """
    if not ANDES_AVAILABLE:
        raise RuntimeError("ANDES is not available. Please install ANDES toolbox.")

    # Validate parameters
    if alpha is None and load_p is None:
        raise ValueError("Either alpha or load_p must be provided.")
    if alpha is not None and load_p is not None:
        warnings.warn("Both alpha and load_p provided. Using alpha (load_p ignored).")
        load_p = None

    if load_q is None:
        load_q = 0.0

    if fault_clearing_time is None:
        fault_clearing_time = fault_start_time + 0.2

    # Validate load level (for backward compatibility)
    if load_p is not None and (load_p < 0.3 or load_p > 1.0):
        warnings.warn(f"Load level {load_p} pu is outside typical range (0.3-1.0 pu)")
    # Validate alpha range
    if alpha is not None and (alpha < 0.4 or alpha > 1.2):
        warnings.warn(f"Alpha {alpha} is outside typical range (0.4-1.2)")

    # Get case file path
    if not os.path.isabs(case_file):
        try:
            case_path = andes.get_case(case_file)
        except Exception:
            case_path = case_file
    else:
        case_path = case_file

    if not os.path.exists(case_path):
        raise FileNotFoundError(f"Case file not found: {case_path}")

    # Check for load device and add if missing
    has_load, load_model_type, load_idx = check_smib_has_load(case_path)
    if not has_load:
        if verbose:
            print("[LOAD CHECK] No load device found. Adding PQ load to bus 3...")
        case_path = add_pq_load_to_smib_case(case_path=case_path, bus_idx=3, p0=0.7, q0=0.0)
        if verbose:
            print(f"[LOAD CHECK] Added PQ load to case file: {case_path}")

    # Load system
    if verbose:
        print(f"[LOAD] Loading system from {Path(case_path).name}...")
    ss = andes.load(case_path, setup=False, no_output=True, default_config=True)

    # Set generator parameters if provided
    if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
        if H is not None:
            M = 2.0 * H  # Convert H to M
            ss.GENCLS.M.v[0] = M
            if verbose:
                print(f"[GEN] Set inertia M={M:.4f} s (H={H:.4f} s)")
        if D is not None:
            ss.GENCLS.D.v[0] = D
            if verbose:
                print(f"[GEN] Set damping D={D:.4f} pu")

    # Set load level using ANDES alter() method (before setup())
    if hasattr(ss, "PQ") and ss.PQ.n > 0:
        load_idx_int = 0
        load_identifier = load_idx_int

        # Try to get UID from idx attribute
        if hasattr(ss.PQ, "idx") and hasattr(ss.PQ.idx, "v"):
            try:
                idx_array = ss.PQ.idx.v
                if hasattr(idx_array, "__getitem__") and len(idx_array) > load_idx_int:
                    load_uid = idx_array[load_idx_int]
                    load_identifier = load_uid
            except (IndexError, AttributeError, TypeError):
                pass

        # NEW: If alpha is provided, get base loads and scale by alpha
        if alpha is not None:
            # Get base load values
            base_p = float(ss.PQ.p0.v[load_idx_int]) if hasattr(ss.PQ, "p0") else 0.0
            base_q = (
                float(ss.PQ.q0.v[load_idx_int])
                if hasattr(ss.PQ, "q0") and hasattr(ss.PQ.q0, "v")
                else 0.0
            )
            # Scale by alpha (uniform scaling - maintains power factor)
            load_p = alpha * base_p
            load_q = alpha * base_q
            if verbose:
                print(
                    f"[ALPHA SCALING] alpha={alpha:.6f}: "
                    f"P={load_p:.6f} pu (α×{base_p:.6f}), Q={load_q:.6f} pu (α×{base_q:.6f})"
                )

        # Set load using alter() method
        if hasattr(ss.PQ, "alter"):
            try:
                ss.PQ.alter("p0", load_identifier, load_p)
                ss.PQ.alter("q0", load_identifier, load_q)
                if verbose:
                    print(f"[LOAD] Set load p0={load_p:.6f} pu, q0={load_q:.6f} pu")
            except Exception:
                # Fallback to direct access
                ss.PQ.p0.v[load_idx_int] = load_p
                ss.PQ.q0.v[load_idx_int] = load_q
                if verbose:
                    print(f"[LOAD] Set load p0={load_p:.6f} pu, q0={load_q:.6f} pu (direct access)")
        else:
            # Direct access
            ss.PQ.p0.v[load_idx_int] = load_p
            ss.PQ.q0.v[load_idx_int] = load_q
            if verbose:
                print(f"[LOAD] Set load p0={load_p:.6f} pu, q0={load_q:.6f} pu (direct access)")

    # Setup system
    ss.setup()

    # Run power flow
    if verbose:
        print("[PFLOW] Running power flow...")
    ss.PFlow.run()

    # CRITICAL: Check power flow convergence
    if not (hasattr(ss.PFlow, "converged") and ss.PFlow.converged):
        raise RuntimeError(
            f"Power flow failed to converge for load={load_p} pu. "
            f"Check system parameters and load level."
        )

    # Extract generator parameters after power flow
    if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
        Pm_actual = float(ss.GENCLS.tm0.v[0])
        M_actual = float(ss.GENCLS.M.v[0])
        D_actual = float(ss.GENCLS.D.v[0])
        H_actual = M_actual / 2.0
    else:
        Pm_actual = None
        M_actual = None
        D_actual = None
        H_actual = None

    # Power Systems Validation
    if validate:
        # Voltage validation
        if hasattr(ss, "Bus") and hasattr(ss.Bus, "v"):
            bus_voltages = ss.Bus.v.v
            v_min = float(np.min(bus_voltages))
            v_max = float(np.max(bus_voltages))
            if v_min < 0.9 or v_max > 1.1:
                warnings.warn(
                    f"Voltage out of normal range for load={load_p} pu: "
                    f"V_min={v_min:.4f} pu, V_max={v_max:.4f} pu"
                )
        else:
            v_min = None
            v_max = None

        # Generator capability check
        if Pm_actual is not None:
            if Pm_actual > 1.2 or Pm_actual < 0.3:
                warnings.warn(
                    f"Generator operating at extreme: Pm={Pm_actual:.4f} pu "
                    f"(typical range: 0.3-1.2 pu)"
                )

        # Power balance check
        if Pm_actual is not None:
            estimated_losses = 0.03 * Pm_actual  # Typical 3% losses
            power_balance_error = abs(Pm_actual - load_p - estimated_losses)
            if power_balance_error > 0.01:
                warnings.warn(
                    f"Power balance error: {power_balance_error:.6f} pu "
                    f"(Pm={Pm_actual:.6f}, Load={load_p:.6f}, Losses≈{estimated_losses:.6f})"
                )
    else:
        v_min = None
        v_max = None
        power_balance_error = None

    # Configure fault
    if hasattr(ss, "Fault") and ss.Fault.n > 0:
        ss.Fault.tf.v[0] = fault_start_time
        ss.Fault.tc.v[0] = fault_clearing_time
        ss.Fault.bus.v[0] = fault_bus
        ss.Fault.xf.v[0] = fault_reactance
    else:
        raise RuntimeError("No fault model found in system")

    # Configure TDS
    ss.TDS.config.tf = simulation_time
    if time_step is not None:
        ss.TDS.config.h = time_step
    ss.TDS.config.criteria = 0  # Disable early stopping
    if hasattr(ss.TDS.config, "plot"):
        ss.TDS.config.plot = True
    if hasattr(ss.TDS.config, "save_plt"):
        ss.TDS.config.save_plt = True

    # Initialize TDS
    if verbose:
        print("[TDS] Initializing TDS...")
    try:
        ss.TDS.init()
    except Exception as e:
        if verbose:
            print(f"[TDS] Warning: TDS.init() raised exception: {e}")

    # Run TDS
    if verbose:
        print(f"[TDS] Running simulation (t=0 to {simulation_time} s)...")
    with suppress_output():
        ss.TDS.run()

    # CRITICAL: Check TDS completion
    exit_code = getattr(ss, "exit_code", 0)
    simulation_completed = exit_code == 0
    if not simulation_completed:
        warnings.warn(
            f"TDS terminated early (exit_code={exit_code}). "
            f"This may indicate instability or numerical issues."
        )

    # Extract trajectories
    if verbose:
        print("[EXTRACT] Extracting trajectories...")
    try:
        # gen_idx can be None (uses first generator) or string UID
        trajectories = extract_trajectories(ss, gen_idx=None, Pm_actual=Pm_actual)
    except Exception as e:
        if verbose:
            print(f"[EXTRACT] Warning: extract_trajectories failed: {e}")
        trajectories = {}

    # Extract Pe if not already in trajectories
    if "Pe" not in trajectories:
        try:
            Pe_dict = extract_pe_trajectories(ss, gen_idx=None, Pm_actual=Pm_actual)
            if Pe_dict and "Pe" in Pe_dict:
                Pe_traj = Pe_dict["Pe"]
                # Handle both single array and dict of arrays
                if isinstance(Pe_traj, np.ndarray) and len(Pe_traj) > 0:
                    trajectories["Pe"] = Pe_traj
                elif isinstance(Pe_traj, dict) and 0 in Pe_traj:
                    # Multi-machine: use first generator
                    pe_array = Pe_traj[0]
                    if isinstance(pe_array, np.ndarray):
                        trajectories["Pe"] = pe_array
        except Exception as e:
            if verbose:
                print(f"[EXTRACT] Warning: Pe extraction failed: {e}")

    # Extract voltage if available
    if hasattr(ss, "TDS") and hasattr(ss.TDS, "plt") and hasattr(ss.TDS.plt, "Bus"):
        try:
            V_traj = ss.TDS.plt.Bus.v[:, 0]  # Generator bus voltage
            trajectories["V"] = V_traj
        except Exception:
            pass

    # Data Quality Validation
    if validate:
        is_valid, issues = validate_data_quality(trajectories)
        if not is_valid and verbose:
            print(f"[VALIDATE] Data quality issues: {issues}")
    else:
        is_valid = True
        issues = []

    # Physics Validation
    if validate and "Pe" in trajectories and Pm_actual is not None:
        Pe_initial = trajectories["Pe"][0] if len(trajectories["Pe"]) > 0 else None
        if Pe_initial is not None:
            power_balance_error_initial = abs(Pm_actual - Pe_initial)
            if power_balance_error_initial > 0.01:
                warnings.warn(
                    f"Power balance error at t=0: {power_balance_error_initial:.6f} pu "
                    f"(Pm={Pm_actual:.6f}, Pe(t=0)={Pe_initial:.6f})"
                )
    else:
        power_balance_error_initial = None

    # Stability Detection
    is_stable: Optional[bool] = None
    stability_metrics: Dict[str, Any] = {}
    if validate and trajectories:
        try:
            is_stable, stability_metrics = check_stability(ss, trajectories)
        except Exception as e:
            if verbose:
                print(f"[STABILITY] Warning: Stability check failed: {e}")

    # Calculate trajectory metrics
    max_delta = None
    max_omega_deviation = None
    if "delta" in trajectories:
        max_delta = float(np.max(np.abs(trajectories["delta"])))
    if "omega" in trajectories:
        max_omega_deviation = float(np.max(np.abs(trajectories["omega"] - 1.0)))

    # Build comprehensive metadata
    metadata = {
        # Operating point
        "load_p": load_p,
        "load_q": load_q,
        "Pm": Pm_actual,
        "H": H_actual,
        "D": D_actual,
        "M": M_actual,
        # Fault parameters
        "fault_start": fault_start_time,
        "fault_clear": fault_clearing_time,
        "fault_bus": fault_bus,
        "fault_reactance": fault_reactance,
        # System state
        "voltage_range": (v_min, v_max),
        "power_balance_error": power_balance_error,
        "power_balance_error_initial": power_balance_error_initial,
        # Stability
        "is_stable": is_stable,
        "stability_method": stability_metrics.get("method", "unknown"),
        "stability_metrics": stability_metrics,
        # Trajectory metrics
        "max_delta": max_delta,
        "max_omega_deviation": max_omega_deviation,
        # Simulation status
        "simulation_completed": simulation_completed,
        "exit_code": exit_code,
        "power_flow_converged": True,  # We checked this above
        "data_quality_valid": is_valid,
        "data_quality_issues": issues,
    }

    results = {
        "time": trajectories.get("time", np.array([])),
        "delta": trajectories.get("delta", np.array([])),
        "omega": trajectories.get("omega", np.array([])),
        "Pe": trajectories.get("Pe", np.array([])),
        "V": trajectories.get("V", np.array([])),
        "metadata": metadata,
    }

    if verbose:
        print(f"[DONE] Simulation completed. Stable: {is_stable}, Max delta: {max_delta:.4f} rad")

    return results


def run_load_level_sweep(
    case_file: str,
    alpha_range: Optional[Tuple[float, float, int]] = None,  # NEW: Unified approach (recommended)
    load_range: Optional[Tuple[float, float, int]] = None,  # DEPRECATED: Use alpha_range instead
    load_q: Optional[float] = None,
    H: Optional[float] = None,
    D: Optional[float] = None,
    fault_start_time: float = 1.0,
    fault_clearing_times: Optional[List[float]] = None,
    fault_bus: int = 3,
    fault_reactance: float = 0.0001,
    simulation_time: float = 5.0,
    time_step: Optional[float] = None,
    task: str = "trajectory",
    validate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run batch simulation over load level range.

    Parameters:
    -----------
    case_file : str
        ANDES case file path
    alpha_range : tuple, optional
        NEW: (min, max, n_points) for alpha multiplier (recommended).
        Example: (0.4, 1.2, 10) for 40% to 120% of base load.
    load_range : tuple, optional
        DEPRECATED: (min, max, n_points) for load levels. Use alpha_range instead.
    load_q : float, optional
        DEPRECATED: Reactive power load (pu). Q is now scaled by same alpha (maintains power factor).
    H : float, optional
        Inertia constant (seconds)
    D : float, optional
        Damping coefficient (pu)
    fault_start_time : float
        Fault start time (seconds)
    fault_clearing_times : list, optional
        List of fault clearing times (seconds). If None, uses default.
    fault_bus : int
        Bus index where fault occurs
    fault_reactance : float
        Fault reactance (pu)
    simulation_time : float
        Total simulation time (seconds)
    time_step : float, optional
        Time step (seconds)
    task : str
        Task type: 'trajectory' or 'parameter_estimation'
    validate : bool
        Whether to perform validation checks
    verbose : bool
        Print progress information

    Returns:
    --------
    df : pd.DataFrame
        DataFrame with all results in standardized format
    """
    # Use alpha_range if provided, otherwise fall back to load_range (backward compatibility)
    if alpha_range is not None:
        alpha_min, alpha_max, n_points = alpha_range
        alpha_values = np.linspace(alpha_min, alpha_max, n_points)
        load_levels = None  # Will be computed from base loads and alpha
        if verbose:
            print(
                f"[ALPHA SWEEP] Using alpha multiplier range: "
                f"{alpha_min:.3f} to {alpha_max:.3f} ({n_points} points)"
            )
    elif load_range is not None:
        load_min, load_max, n_points = load_range
        load_levels = np.linspace(load_min, load_max, n_points)
        alpha_values = None
        if verbose:
            print(
                f"[LOAD SWEEP] Using load range: "
                f"{load_min:.3f} to {load_max:.3f} pu ({n_points} points) [DEPRECATED]"
            )
    else:
        raise ValueError("Either alpha_range or load_range must be provided")

    if fault_clearing_times is None:
        fault_clearing_times = [
            fault_start_time + 0.15,
            fault_start_time + 0.18,
            fault_start_time + 0.20,
            fault_start_time + 0.22,
            fault_start_time + 0.25,
        ]

    if verbose:
        n_levels = len(alpha_values) if alpha_values is not None else len(load_levels)
        total_sims = n_levels * len(fault_clearing_times)
        level_type = "alpha values" if alpha_values is not None else "load levels"
        print(
            f"[SWEEP] Running sweep: {n_levels} {level_type}, "
            f"{len(fault_clearing_times)} clearing times = {total_sims} simulations"
        )

    all_results = []
    scenario_id = 0

    # Progress tracking
    if alpha_values is not None:
        iterator = tqdm(
            [(alpha, tc) for alpha in alpha_values for tc in fault_clearing_times],
            desc="Running simulations",
            disable=not TQDM_AVAILABLE or not verbose,
        )
    else:
        iterator = tqdm(
            [(load_p, tc) for load_p in load_levels for tc in fault_clearing_times],
            desc="Running simulations",
            disable=not TQDM_AVAILABLE or not verbose,
        )

    for level_value, fault_clearing_time in iterator:
        try:
            # Run simulation
            if alpha_values is not None:
                # Use alpha (unified approach)
                results = run_single_load_simulation(
                    case_file=case_file,
                    alpha=level_value,  # alpha value
                    load_p=None,  # Not used when alpha is provided
                    load_q=None,  # Q scaled by same alpha
                    H=H,
                    D=D,
                    fault_start_time=fault_start_time,
                    fault_clearing_time=fault_clearing_time,
                    fault_bus=fault_bus,
                    fault_reactance=fault_reactance,
                    simulation_time=simulation_time,
                    time_step=time_step,
                    validate=validate,
                    verbose=False,  # Suppress individual simulation output
                )
            else:
                # Use load_p (backward compatibility)
                results = run_single_load_simulation(
                    case_file=case_file,
                    load_p=level_value,  # load_p value
                    alpha=None,  # Not used when load_p is provided
                    load_q=load_q,
                    H=H,
                    D=D,
                    fault_start_time=fault_start_time,
                    fault_clearing_time=fault_clearing_time,
                    fault_bus=fault_bus,
                    fault_reactance=fault_reactance,
                    simulation_time=simulation_time,
                    time_step=time_step,
                    validate=validate,
                    verbose=False,  # Suppress individual simulation output
                )

            metadata = results["metadata"]

            # Convert to DataFrame format matching existing pipeline
            n_points = len(results["time"])
            if n_points > 0:
                df_scenario = pd.DataFrame(
                    {
                        "time": results["time"],
                        "delta": results["delta"],
                        "omega": results["omega"],
                        "Pe": results.get("Pe", np.full(n_points, np.nan)),
                        "V": results.get("V", np.full(n_points, np.nan)),
                        "param_H": np.full(n_points, metadata.get("H")),
                        "param_D": np.full(n_points, metadata.get("D")),
                        "alpha": np.full(
                            n_points, level_value if alpha_values is not None else np.nan
                        ),
                        "param_load_p": np.full(
                            n_points, level_value if load_levels is not None else np.nan
                        ),
                        "param_load_q": np.full(n_points, metadata.get("load_q", 0.0)),
                        "param_Pm": np.full(n_points, metadata.get("Pm")),
                        "param_tc": np.full(n_points, fault_clearing_time),
                        "is_stable": np.full(n_points, metadata.get("is_stable", False)),
                        "scenario_id": np.full(n_points, scenario_id),
                    }
                )
                all_results.append(df_scenario)
                scenario_id += 1
        except Exception as e:
            if verbose:
                if alpha_values is not None:
                    level_str = f"alpha={level_value:.4f}"
                else:
                    level_str = f"load={level_value:.4f} pu"
                print(
                    f"[ERROR] Simulation failed for {level_str}, "
                    f"tc={fault_clearing_time:.4f} s: {e}"
                )
            continue

    if len(all_results) == 0:
        raise RuntimeError("No successful simulations. Check parameters and system configuration.")

    # Concatenate all results
    df = pd.concat(all_results, ignore_index=True)

    if verbose:
        print(
            f"[SWEEP] Completed: {len(all_results)} successful simulations, "
            f"{len(df)} total data points"
        )

    return df


def analyze_load_levels(
    case_file: str,
    load_levels: List[float],
    use_alpha: bool = False,  # NEW: If True, load_levels contains alpha values
    load_q: Optional[float] = None,
    H: Optional[float] = None,
    D: Optional[float] = None,
    fault_start_time: float = 1.0,
    fault_clearing_time: float = 1.2,
    fault_bus: int = 3,
    fault_reactance: float = 0.0001,
    simulation_time: float = 5.0,
    time_step: Optional[float] = None,
    output_dir: Optional[str] = None,
    plot: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run simulations and create analysis plots.

    Parameters:
    -----------
    case_file : str
        ANDES case file path
    load_levels : list
        List of load levels (or alpha values if use_alpha=True) to analyze
    use_alpha : bool
        If True, load_levels contains alpha multiplier values (unified approach)
        If False, load_levels contains absolute load values (backward compatibility)
    load_q : float, optional
        DEPRECATED: Reactive power load (pu). Q is now scaled by same alpha (maintains power factor).
    H : float, optional
        Inertia constant (seconds)
    D : float, optional
        Damping coefficient (pu)
    fault_start_time : float
        Fault start time (seconds)
    fault_clearing_time : float
        Fault clearing time (seconds)
    fault_bus : int
        Bus index where fault occurs
    fault_reactance : float
        Fault reactance (pu)
    simulation_time : float
        Total simulation time (seconds)
    time_step : float, optional
        Time step (seconds)
    output_dir : str, optional
        Directory to save plots
    plot : bool
        Whether to generate plots
    verbose : bool
        Print progress information

    Returns:
    --------
    analysis : dict
        Dictionary with analysis results and plots
    """
    if output_dir is None:
        output_dir = "outputs/analysis"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_list = []
    for level_value in load_levels:
        if verbose:
            if use_alpha:
                print(f"\n[ANALYSIS] Running simulation for alpha={level_value:.4f}...")
            else:
                print(f"\n[ANALYSIS] Running simulation for load={level_value:.4f} pu...")
        try:
            if use_alpha:
                # Use alpha (unified approach)
                results = run_single_load_simulation(
                    case_file=case_file,
                    alpha=level_value,
                    load_p=None,
                    load_q=None,  # Q scaled by same alpha
                    H=H,
                    D=D,
                    fault_start_time=fault_start_time,
                    fault_clearing_time=fault_clearing_time,
                    fault_bus=fault_bus,
                    fault_reactance=fault_reactance,
                    simulation_time=simulation_time,
                    time_step=time_step,
                    validate=True,
                    verbose=verbose,
                )
            else:
                # Use load_p (backward compatibility)
                results = run_single_load_simulation(
                    case_file=case_file,
                    load_p=level_value,
                    alpha=None,
                    load_q=load_q,
                    H=H,
                    D=D,
                    fault_start_time=fault_start_time,
                    fault_clearing_time=fault_clearing_time,
                    fault_bus=fault_bus,
                    fault_reactance=fault_reactance,
                    simulation_time=simulation_time,
                    time_step=time_step,
                    validate=True,
                    verbose=verbose,
                )
            results_list.append(results)
        except Exception as e:
            if verbose:
                if use_alpha:
                    print(f"[ERROR] Simulation failed for alpha={level_value:.4f}: {e}")
                else:
                    print(f"[ERROR] Simulation failed for load={level_value:.4f} pu: {e}")
            continue

    if len(results_list) == 0:
        raise RuntimeError("No successful simulations for analysis.")

    # Generate plots if requested
    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(10, 12))

            for i, results in enumerate(results_list):
                load_p = results["metadata"]["load_p"]
                time = results["time"]
                delta = results["delta"]
                omega = results["omega"]
                Pe = results.get("Pe", np.array([]))

                label = f"Load={load_p:.3f} pu"

                # Rotor angle
                axes[0].plot(time, np.degrees(delta), label=label)
                axes[0].set_xlabel("Time (s)")
                axes[0].set_ylabel("Rotor Angle (degrees)")
                axes[0].set_title("Rotor Angle vs Time")
                axes[0].grid(True)
                axes[0].legend()

                # Speed
                axes[1].plot(time, omega, label=label)
                axes[1].set_xlabel("Time (s)")
                axes[1].set_ylabel("Speed (pu)")
                axes[1].set_title("Speed vs Time")
                axes[1].grid(True)
                axes[1].legend()

                # Electrical power
                if len(Pe) > 0:
                    axes[2].plot(time, Pe, label=label)
                    axes[2].set_xlabel("Time (s)")
                    axes[2].set_ylabel("Electrical Power (pu)")
                    axes[2].set_title("Electrical Power vs Time")
                    axes[2].grid(True)
                    axes[2].legend()

            plt.tight_layout()
            plot_filename = generate_timestamped_filename("load_level_analysis", "png")
            plot_path = output_path / plot_filename
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            if verbose:
                print(f"[PLOT] Saved analysis plot to {plot_path}")
            plt.close()
        except ImportError:
            if verbose:
                print("[PLOT] matplotlib not available, skipping plots")

    # Generate summary statistics
    summary = {
        "n_simulations": len(results_list),
        "load_levels": [r["metadata"]["load_p"] for r in results_list],
        "stability": [r["metadata"].get("is_stable") for r in results_list],
        "max_delta": [r["metadata"].get("max_delta") for r in results_list],
        "max_omega_deviation": [r["metadata"].get("max_omega_deviation") for r in results_list],
    }

    return {
        "results": results_list,
        "summary": summary,
        "output_dir": str(output_path),
    }


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run ANDES time domain simulations at different load levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--case",
        type=str,
        required=True,
        help="ANDES case file path (e.g., 'smib/SMIB.json')",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "analysis"],
        default="batch",
        help="Mode: 'batch' for data generation, 'analysis' for single/few simulations",
    )

    # Load level specification
    parser.add_argument(
        "--alpha",  # NEW: Unified approach (recommended)
        type=float,
        help="Single alpha multiplier (e.g., 0.7 = 70%% of base load) for analysis mode",
    )
    parser.add_argument(
        "--load",  # DEPRECATED: Use --alpha instead
        type=float,
        help="DEPRECATED: Single load level (pu) for analysis mode. Use --alpha instead.",
    )
    parser.add_argument(
        "--alpha-range",  # NEW: Unified approach (recommended)
        type=float,
        nargs=3,
        metavar=("MIN", "MAX", "N"),
        help="Alpha multiplier range: min max n_points (e.g., 0.4 1.2 10 for 40%%-120%% of base load)",
    )
    parser.add_argument(
        "--load-range",  # DEPRECATED: Use --alpha-range instead
        type=float,
        nargs=3,
        metavar=("MIN", "MAX", "N"),
        help="DEPRECATED: Load level range: min max n_points (for batch mode). Use --alpha-range instead.",
    )

    # Generator parameters
    parser.add_argument("--H", type=float, help="Inertia constant H (seconds)")
    parser.add_argument("--D", type=float, help="Damping coefficient D (pu)")

    # Fault parameters
    parser.add_argument(
        "--fault-start",
        type=float,
        default=1.0,
        help="Fault start time (seconds, default: 1.0)",
    )
    parser.add_argument(
        "--fault-clear",
        type=float,
        help="Fault clearing time (seconds, for single simulation)",
    )
    parser.add_argument(
        "--fault-clearing-times",
        type=float,
        nargs="+",
        help="List of fault clearing times (seconds, for batch mode)",
    )
    parser.add_argument(
        "--fault-bus",
        type=int,
        default=3,
        help="Fault bus index (default: 3)",
    )
    parser.add_argument(
        "--fault-reactance",
        type=float,
        default=0.0001,
        help="Fault reactance (pu, default: 0.0001)",
    )

    # Simulation parameters
    parser.add_argument(
        "--simulation-time",
        type=float,
        default=5.0,
        help="Total simulation time (seconds, default: 5.0)",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        help="Time step (seconds, default: ANDES default)",
    )

    # Task and validation
    parser.add_argument(
        "--task",
        type=str,
        default="trajectory",
        choices=["trajectory", "parameter_estimation"],
        help="Task type (default: trajectory)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Enable validation checks (default: True)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Disable validation checks",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (for analysis mode plots, batch mode saves to data/common/)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (analysis mode only)",
    )

    # Other options
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress information (default: True)",
    )
    parser.add_argument(
        "--quiet",
        action="store_false",
        dest="verbose",
        help="Suppress progress information",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "analysis":
        if args.alpha is None and args.load is None:
            parser.error("--alpha (or --load) is required for analysis mode")
        if args.alpha is not None and args.load is not None:
            print("[WARNING] Both --alpha and --load provided. Using --alpha (--load ignored).")
    if args.mode == "batch":
        if args.alpha_range is None and args.load_range is None:
            parser.error("--alpha-range (or --load-range) is required for batch mode")
        if args.alpha_range is not None and args.load_range is not None:
            print(
                "[WARNING] Both --alpha-range and --load-range provided. "
                "Using --alpha-range (--load-range ignored)."
            )

    try:
        if args.mode == "batch":
            # Batch mode: generate data and save to data/common/
            if args.verbose:
                print("=" * 70)
                print("Load Level Time Domain Simulation - Batch Mode")
                print("=" * 70)

            # Use alpha_range if provided, otherwise fall back to load_range (backward compatibility)
            alpha_range = args.alpha_range if args.alpha_range is not None else args.load_range
            load_range = args.load_range if args.alpha_range is None else None

            df = run_load_level_sweep(
                case_file=args.case,
                alpha_range=tuple(alpha_range) if alpha_range is not None else None,
                load_range=tuple(load_range) if load_range is not None else None,
                load_q=None,  # Q scaled by same alpha (maintains power factor)
                H=args.H,
                D=args.D,
                fault_start_time=args.fault_start,
                fault_clearing_times=args.fault_clearing_times,
                fault_bus=args.fault_bus,
                fault_reactance=args.fault_reactance,
                simulation_time=args.simulation_time,
                time_step=args.time_step,
                task=args.task,
                validate=args.validate,
                verbose=args.verbose,
            )

            # Save to common repository
            if args.verbose:
                print("\n[SAVE] Saving data to common repository...")

            # Build config dict for fingerprinting
            parameter_ranges = {
                "load": args.load_range,
            }
            if args.H is not None:
                parameter_ranges["H"] = [args.H, args.H, 1]
            if args.D is not None:
                parameter_ranges["D"] = [args.D, args.D, 1]

            config = {
                "data": {
                    "generation": {
                        "case_file": args.case,
                        "parameter_ranges": parameter_ranges,
                        "fault": {
                            "start_time": args.fault_start,
                            "bus": args.fault_bus,
                            "reactance": args.fault_reactance,
                        },
                        "simulation_time": args.simulation_time,
                        "time_step": args.time_step,
                    },
                },
            }

            data_path, metadata = save_data_to_common(
                data=df,
                task=args.task,
                config=config,
                force_regenerate=False,
            )

            if args.verbose:
                metadata_path = data_path.with_suffix(".json").with_name(
                    data_path.stem + "_metadata.json"
                )
                print(f"✓ Data saved to: {data_path}")
                print(f"✓ Metadata saved to: {metadata_path}")

            # Generate validation report
            if args.validate:
                try:
                    report = generate_validation_report(df, task=args.task)
                    if args.verbose:
                        print("\n[VALIDATION] Data quality report:")
                        print(report)
                except Exception as e:
                    if args.verbose:
                        print(f"[VALIDATION] Warning: Could not generate validation report: {e}")

        else:
            # Analysis mode: single or few simulations
            if args.verbose:
                print("=" * 70)
                print("Load Level Time Domain Simulation - Analysis Mode")
                print("=" * 70)

            # Use alpha if provided, otherwise fall back to load (backward compatibility)
            if args.alpha is not None:
                # Use alpha (unified approach)
                load_levels = [args.alpha]
                use_alpha = True
            elif args.load is not None:
                # Use load_p (backward compatibility)
                load_levels = [args.load]
                use_alpha = False
            else:
                # Check for range (batch mode should have been caught earlier)
                if args.alpha_range:
                    load_levels = np.linspace(
                        args.alpha_range[0], args.alpha_range[1], int(args.alpha_range[2])
                    ).tolist()
                    use_alpha = True
                elif args.load_range:
                    load_levels = np.linspace(
                        args.load_range[0], args.load_range[1], int(args.load_range[2])
                    ).tolist()
                    use_alpha = False
                else:
                    parser.error("--alpha or --load is required for analysis mode")

            analysis = analyze_load_levels(
                case_file=args.case,
                load_levels=load_levels,
                use_alpha=use_alpha,  # Pass flag to indicate alpha vs load_p
                load_q=None,
                H=args.H,
                D=args.D,
                fault_start_time=args.fault_start,
                fault_clearing_time=args.fault_clear or (args.fault_start + 0.2),
                fault_bus=args.fault_bus,
                fault_reactance=args.fault_reactance,
                simulation_time=args.simulation_time,
                time_step=args.time_step,
                output_dir=args.output,
                plot=args.plot,
                verbose=args.verbose,
            )

            if args.verbose:
                print("\n[ANALYSIS] Summary:")
                print(f"  Simulations: {analysis['summary']['n_simulations']}")
                print(f"  Load levels: {analysis['summary']['load_levels']}")
                print(f"  Stability: {analysis['summary']['stability']}")

    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
