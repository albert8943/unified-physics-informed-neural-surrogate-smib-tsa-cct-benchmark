#!/usr/bin/env python3
"""
CCT finder module.

Functions for finding Critical Clearing Time (CCT) using binary search.
"""

import logging
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False

# Handle imports for both module and direct execution
try:
    from .data_extractor import extract_trajectories_with_derived
    from .data_validator import (
        check_power_flow_convergence,
        check_stability,
        validate_generator_model,
        validate_perunit_consistency,
        validate_prefault_equilibrium,
    )
    from .system_manager import safe_get_array_value, suppress_output
except ImportError:
    # Fallback for direct execution or alternative import paths
    from pathlib import Path

    # Add project root to path if not already there
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from data_generation.andes_utils.data_extractor import extract_trajectories_with_derived
    from data_generation.andes_utils.data_validator import (
        check_power_flow_convergence,
        check_stability,
        validate_generator_model,
        validate_perunit_consistency,
        validate_prefault_equilibrium,
    )
    from data_generation.andes_utils.system_manager import safe_get_array_value, suppress_output

# COI-based stability for multimachine (n_gen > 1)
try:
    from utils.stability_checker import check_stability_multimachine_coi
except ImportError:
    check_stability_multimachine_coi = None


# Prevent direct execution - this is a module, not a script
if __name__ == "__main__":
    print("This is a module, not a script. Import it instead:")
    print("  from data_generation.andes_utils.cct_finder import find_cct")
    sys.exit(1)


def find_main_generator_index(ss: Any) -> int:
    """
    Find the index of the main generator (not the infinite bus).

    For SMIB systems: Simple logic - find generator with M < 1e6 (normal generator),
    skip generator with M > 1e6 (infinite bus).

    Parameters:
    -----------
    ss : andes.System
        ANDES system object (must be set up)

    Returns:
    --------
    gen_idx : int
        Index of the main generator (0-based)
    """
    if not hasattr(ss, "GENCLS") or ss.GENCLS.n == 0:
        return 0  # Default fallback

    # Simple logic for SMIB: Find generator with normal M value (M < 1e6)
    # Infinite bus has M > 1e6, main generator has M ~ 2-10 s
    for i in range(ss.GENCLS.n):
        if hasattr(ss.GENCLS, "M") and hasattr(ss.GENCLS.M, "v"):
            try:
                M_val = safe_get_array_value(ss.GENCLS.M.v, i)
                if M_val is not None and isinstance(M_val, (int, float)) and M_val < 1e6:
                    # Found main generator (normal M value) - return it
                    return i
            except (IndexError, AttributeError, TypeError):
                continue

    # Fallback: If no generator with M < 1e6 found, return first generator
    # (shouldn't happen in normal SMIB cases)
    return 0 if ss.GENCLS.n > 0 else 0


def change_generator_parameters(
    ss: Any,
    gen_idx: int = 0,
    Pm: Optional[float] = None,
    M: Optional[float] = None,
    D: Optional[float] = None,
) -> bool:
    """
    Change generator parameters (Pm, M, D) in ANDES system.

    Following smib_albert_cct.py pattern for consistency.

    Uses ANDES built-in methods:
    - ss.GENCLS.tm0.v[gen_idx] = Pm  # Mechanical power
    - ss.GENCLS.M.v[gen_idx] = M     # Inertia
    - ss.GENCLS.D.v[gen_idx] = D     # Damping

    Parameters:
    -----------
    ss : andes.System
        ANDES system object
    gen_idx : int
        Generator index (0-based)
    Pm : float, optional
        Mechanical power (pu) - sets both tm0 and P0
    M : float, optional
        Inertia constant (seconds)
    D : float, optional
        Damping coefficient (pu)

    Returns:
    --------
    success : bool
        True if parameters were set successfully
    """
    if not (hasattr(ss, "GENCLS") and ss.GENCLS.n > gen_idx):
        return False

    success = False

    # Change mechanical power (Pm)
    if Pm is not None:
        # Set both tm0 (for transient) and P0 (for power flow)
        if hasattr(ss.GENCLS, "tm0"):
            try:
                ss.GENCLS.tm0.v[gen_idx] = Pm
                success = True
            except Exception:
                pass

        if hasattr(ss.GENCLS, "P0"):
            try:
                ss.GENCLS.P0.v[gen_idx] = Pm
                success = True
            except Exception:
                pass

    # Change inertia (M)
    if M is not None:
        if hasattr(ss.GENCLS, "M"):
            try:
                ss.GENCLS.M.v[gen_idx] = M
                success = True
            except Exception:
                pass

    # Change damping (D)
    if D is not None:
        if hasattr(ss.GENCLS, "D"):
            try:
                ss.GENCLS.D.v[gen_idx] = D
                success = True
            except Exception:
                pass

    return success


def safe_get_tm0_value(ss: Any, gen_idx: int) -> float:
    """
    Safely get tm0 value from ANDES system, handling both scalar and array cases.

    Parameters:
    -----------
    ss : andes.System
        ANDES system object
    gen_idx : int
        Generator index

    Returns:
    --------
    float
        tm0 value for the specified generator
    """
    if (
        not hasattr(ss, "GENCLS")
        or not hasattr(ss.GENCLS, "tm0")
        or not hasattr(ss.GENCLS.tm0, "v")
    ):
        raise ValueError("GENCLS.tm0.v not available")

    tm0v = ss.GENCLS.tm0.v

    # Check if tm0.v is array-like
    if hasattr(tm0v, "__getitem__") and hasattr(tm0v, "__len__"):
        try:
            # Try to get length (may fail for scalars)
            length = len(tm0v)
            if length > gen_idx:
                return float(tm0v[gen_idx])
            else:
                # Array exists but index out of range, return first element
                return float(tm0v[0]) if length > 0 else float(tm0v)
        except (TypeError, AttributeError):
            # tm0.v is scalar or doesn't support len()
            return float(tm0v)
    else:
        # tm0.v is scalar
        return float(tm0v)


def configure_fault(
    ss: Any,
    fault_start_time: float,
    fault_clearing_time: float,
    fault_bus: int,
    fault_reactance: float,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, bool]:
    """
    Configure fault parameters in ANDES system.

    Adds a fault if it doesn't exist, or modifies existing fault.

    Following smib_albert_cct.py pattern for consistency.

    Uses ANDES built-in methods (recommended approach):
    - ss.Fault.alter('tc', idx, value) - Alter fault clearing time
    - ss.Fault.alter('tf', idx, value) - Alter fault start time
    - ss.Fault.alter('bus', idx, value) - Alter fault bus
    - ss.Fault.alter('xf', idx, value) - Alter fault reactance
    - ss.add("Fault", {...}) - Add new fault device

    Parameters:
    -----------
    ss : andes.System
        ANDES system object
    fault_start_time : float
        Fault start time (seconds)
    fault_clearing_time : float
        Fault clearing time (seconds)
    fault_bus : int
        Bus where fault occurs
    fault_reactance : float
        Fault reactance (pu) - 0.0001 for bolted fault
    logger : logging.Logger, optional
        Logger for warning/error messages

    Returns:
    --------
    (success, fault_was_added) : tuple
        success: True if fault was configured successfully
        fault_was_added: True if a NEW fault was added (requires setup()), "
        "False if existing fault was modified
    """
    try:
        # Check if fault model exists
        if hasattr(ss, "Fault") and ss.Fault.n > 0:
            # Modify existing fault using ANDES alter() method (recommended approach)
            # Find the fault idx (typically 0 for SMIB, but check all faults)
            fault_idx = None
            try:
                # Try to find fault on the specified bus
                if hasattr(ss.Fault, "bus"):
                    for i in range(ss.Fault.n):
                        if hasattr(ss.Fault.bus, "v") and ss.Fault.bus.v[i] == fault_bus:
                            fault_idx = i
                            break
                # If not found, use first fault (idx=0) as default
                if fault_idx is None:
                    fault_idx = 0
            except Exception:
                fault_idx = 0  # Default to first fault

            # Use alter() method to modify fault parameters (ANDES recommended approach)
            try:
                if hasattr(ss.Fault, "alter"):
                    ss.Fault.alter("tf", fault_idx, fault_start_time)
                    ss.Fault.alter("tc", fault_idx, fault_clearing_time)
                    if hasattr(ss.Fault, "bus"):
                        ss.Fault.alter("bus", fault_idx, fault_bus)
                    if hasattr(ss.Fault, "xf"):
                        ss.Fault.alter("xf", fault_idx, fault_reactance)
                    if hasattr(ss.Fault, "rf"):
                        ss.Fault.alter("rf", fault_idx, 0.0)  # Resistance (0 for bolted fault)
                    if hasattr(ss.Fault, "u"):
                        ss.Fault.alter("u", fault_idx, 1)  # Enable fault
                else:
                    # Fallback to direct access if alter() is not available
                    ss.Fault.tf.v[fault_idx] = fault_start_time
                    ss.Fault.tc.v[fault_idx] = fault_clearing_time
                    if hasattr(ss.Fault, "bus"):
                        ss.Fault.bus.v[fault_idx] = fault_bus
                    if hasattr(ss.Fault, "xf"):
                        ss.Fault.xf.v[fault_idx] = fault_reactance
                    if hasattr(ss.Fault, "rf"):
                        ss.Fault.rf.v[fault_idx] = 0.0
                    if hasattr(ss.Fault, "u"):
                        ss.Fault.u.v[fault_idx] = 1
            except Exception:
                # If alter() fails, try direct access as fallback
                try:
                    ss.Fault.tf.v[fault_idx] = fault_start_time
                    ss.Fault.tc.v[fault_idx] = fault_clearing_time
                    if hasattr(ss.Fault, "bus"):
                        ss.Fault.bus.v[fault_idx] = fault_bus
                    if hasattr(ss.Fault, "xf"):
                        ss.Fault.xf.v[fault_idx] = fault_reactance
                    if hasattr(ss.Fault, "rf"):
                        ss.Fault.rf.v[fault_idx] = 0.0
                    if hasattr(ss.Fault, "u"):
                        ss.Fault.u.v[fault_idx] = 1
                except Exception as e2:
                    error_msg = f"Could not modify fault: {e2}"
                    if logger:
                        logger.warning(error_msg)
                    print(f"[CCT] {error_msg}")
                    return False, False

            return True, False  # Success, but fault was NOT added (just modified)
        else:
            # Add a new fault if it doesn't exist (requires setup())
            try:
                ss.add(
                    "Fault",
                    {
                        "u": 1,  # Enable fault
                        "name": "F_bus{fault_bus}",
                        "bus": fault_bus,
                        "tf": fault_start_time,
                        "tc": fault_clearing_time,
                        "rf": 0.0,  # Resistance (0 for bolted fault)
                        "xf": fault_reactance,  # Reactance
                    },
                )
                return True, True  # Success, and fault WAS added (needs setup())
            except Exception as e:
                error_msg = f"Could not add fault: {e}"
                if logger:
                    logger.warning(error_msg)
                print(f"[CCT] {error_msg}")
                return False, False
    except Exception as e:
        error_msg = f"Error configuring fault: {e}"
        if logger:
            logger.warning(error_msg)
        print(f"[CCT] {error_msg}")
        return False, False


def test_clearing_time(
    case_path: str,
    Pm: float,
    M: float,
    D: float,
    clearing_time: float,
    fault_start_time: float = 1.0,
    fault_bus: int = 3,
    fault_reactance: float = 0.0001,
    simulation_time: float = 5.0,
    time_step: Optional[float] = None,
    tolerance: float = 1e-4,
    logger: Optional[logging.Logger] = None,
    ss: Optional[Any] = None,
    reload_system: bool = False,
    _retry_count: int = 0,  # Internal parameter to track retries
    max_angle_deg: Optional[float] = 360.0,  # Early stopping at this angle (None to disable)
    alpha: Optional[
        float
    ] = None,  # NEW: Load multiplier for load variation (if None, uses Pm directly)
    base_load: Optional[
        Dict[str, float]
    ] = None,  # NEW: Base load values {"Pload": 0.5, "Qload": 0.2}
    addfile: Optional[str] = None,  # Optional DYR/addfile path (e.g. Kundur GENCLS .dyr)
) -> Tuple[bool, float, Dict[str, Any], Dict[str, Any], Optional[Any]]:
    """
    Test a single clearing time and return stability result.

    DEFAULT: Follows ANDES batch processing pattern (reuses system object if provided)
    OPTIONAL: Can reload system for clean state if reload_system=True.

    Parameters:
    -----------
    case_path : str
        Path to ANDES case file (used if ss is None or reload_system=True)
    Pm : float
        Mechanical power (pu). If alpha is provided, this should be the Pm extracted
        after load scaling (for reference/verification). The actual Pm will be computed
        by power flow after load scaling.
    M : float
        Inertia constant (seconds)
    D : float
        Damping coefficient (pu)
    clearing_time : float
        Fault clearing time to test (seconds)
    fault_start_time : float
        Fault start time (seconds)
    fault_bus : int
        Bus where fault occurs
    fault_reactance : float
        Fault reactance (pu)
    simulation_time : float
        Total simulation time (seconds)
    time_step : float, optional
        Time step (seconds). If None (default), uses ANDES automatic time step
        based on system frequency (e.g., 33.33 ms for 60 Hz systems)
        If specified, attempts to force this time step (may be overridden by ANDES
        for numerical stability)
    tolerance : float
        Convergence tolerance
    logger : logging.Logger, optional
        Logger for error messages
    ss : andes.System, optional
        System object to reuse (ANDES batch processing pattern)
        If None or reload_system=True, system will be reloaded.
    reload_system : bool
        If True, reload system for clean state (optional, not default)
        Default: False (follows ANDES pattern of reusing system)
    max_angle_deg : float, optional
        Maximum rotor angle in degrees for early stopping. If trajectory exceeds this,
        it will be truncated at the first point exceeding the limit.
        Default: 360.0 (2π radians). Set to None to disable truncation.
    alpha : float, optional
        NEW: Load multiplier for load variation (e.g., 0.7 = 70% of base load).
        If provided, loads will be scaled by alpha BEFORE setting generator parameters.
        If None, uses Pm directly (Pm variation mode).
    base_load : dict, optional
        NEW: Base load values {"Pload": 0.5, "Qload": 0.2} for alpha scaling.
        If None and alpha is provided, will read from case file or use defaults.

    Returns:
    --------
    is_stable : bool
        True if system is stable
    max_angle : float
        Maximum rotor angle (degrees)
    trajectories_dict : dict
        Dictionary with trajectory data
    stability_metrics : dict
        Dictionary with stability metrics
    ss : andes.System or None
        System object (for further analysis if needed)
    """
    try:
        # DEFAULT: Follow ANDES batch processing pattern (reuse system if provided)
        # OPTIONAL: Reload system for clean state if requested
        setup_was_called = False
        if ss is None or reload_system:
            # Load with setup=False, add Fault if needed, then setup() so ANDES initializes with the fault
            # (adding Fault after setup() can fail with "Failed to configure fault")
            print("[CCT DEBUG] Loading system (Fault added before setup)...")
            try:
                load_kw = dict(setup=False, no_output=True, default_config=True)
                if addfile:
                    load_kw["addfile"] = addfile
                ss = andes.load(case_path, **load_kw)
                if not hasattr(ss, "Fault") or ss.Fault.n == 0:
                    try:
                        ss.add(
                            "Fault",
                            {
                                "u": 1,
                                "name": f"F_bus{fault_bus}",
                                "bus": fault_bus,
                                "tf": fault_start_time,
                                "tc": clearing_time,
                                "rf": 0.0,
                                "xf": fault_reactance,
                            },
                        )
                        print(
                            f"[CCT DEBUG] Added Fault at bus {fault_bus} (tc={clearing_time}s) before setup"
                        )
                    except Exception as e:
                        print(f"[CCT] Could not add Fault before setup: {e}")
                        if logger:
                            logger.error(f"Could not add Fault before setup: {e}")
                        return False, 180.0, {}, {"error": str(e)}, None
                ss.setup()
                setup_was_called = True
                print("[CCT DEBUG] System loaded and initialized (slack bus configured)")

                # CRITICAL: Reset power flow state immediately after load
                # When loading with setup=True, ANDES automatically runs power flow with default values
                # We need to reset the power flow state so it will use our modified parameters
                if hasattr(ss, "PFlow"):
                    if hasattr(ss.PFlow, "converged"):
                        ss.PFlow.converged = False
                    if hasattr(ss.PFlow, "initialized"):
                        ss.PFlow.initialized = False
                    print("[CCT DEBUG] Reset power flow state (will use modified parameters)")

                # Also reset TDS state
                if hasattr(ss, "TDS") and hasattr(ss.TDS, "initialized"):
                    ss.TDS.initialized = False
            except Exception as e:
                error_msg = f"Failed to load system: {e}"
                print(f"[CCT ERROR] {error_msg}")
                if logger:
                    logger.error(error_msg)
                return False, 180.0, {}, {"error": error_msg}, None
        else:
            # Reuse provided system (ANDES batch processing pattern)
            # System is already set up, so we can modify parameters directly
            setup_was_called = False  # Already initialized

        # Validate generator model
        is_valid, warning = validate_generator_model(ss)
        if not is_valid:
            if logger:
                logger.error(f"Generator model validation failed: {warning}")
            return False, 180.0, {}, {"error": warning}, None

        # Validate that GENCLS exists and has devices (after setup())
        if not hasattr(ss, "GENCLS") or ss.GENCLS.n == 0:
            error_msg = "GENCLS generator model not found or has no devices"
            if logger:
                logger.error(error_msg)
            return False, 180.0, {}, {"error": error_msg}, None

        # NEW: Handle load scaling if alpha is provided (load variation mode)
        # This must be done BEFORE setting generator parameters
        if alpha is not None:
            if logger:
                logger.info(f"[LOAD SCALING] Applying alpha={alpha:.3f} to loads...")

            # Determine base load values
            # For multimachine (addfile set), use case-native load so CCT bisection matches trajectory run
            base_p = None
            base_q = None
            if addfile and hasattr(ss, "PQ") and ss.PQ.n > 0:
                try:
                    p0 = ss.PQ.p0.v
                    base_p = (
                        float(p0[0])
                        if hasattr(p0, "__getitem__") and len(p0) > 0
                        else float(np.sum(p0))
                    )
                    if hasattr(ss.PQ, "q0") and hasattr(ss.PQ.q0, "v"):
                        q0 = ss.PQ.q0.v
                        base_q = float(q0[0]) if hasattr(q0, "__getitem__") and len(q0) > 0 else 0.2
                    else:
                        base_q = 0.2
                    if base_p and base_p > 0 and logger:
                        logger.info(
                            f"[LOAD SCALING] Using case-native load (multimachine): P={base_p:.6f} pu"
                        )
                except Exception:
                    base_p = None
                    base_q = None
            if base_p is None or base_p <= 0:
                if base_load:
                    base_p = base_load.get("Pload")
                    base_q = base_load.get("Qload")
                else:
                    if hasattr(ss, "PQ") and ss.PQ.n > 0:
                        load_idx = 0
                        if hasattr(ss.PQ, "p0") and hasattr(ss.PQ.p0, "v"):
                            try:
                                base_p = float(ss.PQ.p0.v[load_idx])
                            except (IndexError, AttributeError, TypeError):
                                base_p = 0.0
                        if hasattr(ss.PQ, "q0") and hasattr(ss.PQ.q0, "v"):
                            try:
                                base_q = float(ss.PQ.q0.v[load_idx])
                            except (IndexError, AttributeError, TypeError):
                                base_q = 0.0

            # Use defaults if not found
            if base_p is None or base_p <= 0:
                base_p = 0.5  # Default base load
            if base_q is None:
                base_q = 0.2  # Default base reactive load

            # Scale loads: for multimachine (addfile), scale ALL PQ loads uniformly by alpha
            if hasattr(ss, "PQ") and ss.PQ.n > 0:
                try:
                    n_loads = ss.PQ.n
                    for load_idx in range(n_loads):
                        if addfile and n_loads > 0:
                            # Per-load base from case (match parameter_sweep uniform scaling)
                            p0 = ss.PQ.p0.v
                            q0 = (
                                ss.PQ.q0.v
                                if hasattr(ss.PQ, "q0") and hasattr(ss.PQ.q0, "v")
                                else None
                            )
                            base_p_i = (
                                float(p0[load_idx])
                                if hasattr(p0, "__getitem__") and load_idx < len(p0)
                                else base_p
                            )
                            base_q_i = (
                                float(q0[load_idx])
                                if q0 is not None
                                and hasattr(q0, "__getitem__")
                                and load_idx < len(q0)
                                else base_q
                            )
                        else:
                            base_p_i = base_p
                            base_q_i = base_q
                        scaled_p = alpha * base_p_i
                        scaled_q = alpha * (base_q_i if base_q_i is not None else 0.2)
                        load_identifier = load_idx
                        if hasattr(ss.PQ, "idx") and hasattr(ss.PQ.idx, "v"):
                            try:
                                idx_array = ss.PQ.idx.v
                                if hasattr(idx_array, "__getitem__") and len(idx_array) > load_idx:
                                    load_identifier = idx_array[load_idx]
                            except (IndexError, AttributeError, TypeError):
                                pass
                        if hasattr(ss.PQ, "alter"):
                            ss.PQ.alter("p0", load_identifier, scaled_p)
                            ss.PQ.alter("q0", load_identifier, scaled_q)
                        else:
                            ss.PQ.p0.v[load_idx] = scaled_p
                            if hasattr(ss.PQ, "q0") and hasattr(ss.PQ.q0, "v"):
                                ss.PQ.q0.v[load_idx] = scaled_q
                    if logger and n_loads > 0:
                        logger.info(
                            f"[LOAD SCALING] Scaled {n_loads} PQ load(s) by α={alpha:.3f} "
                            f"(P base={base_p:.6f} pu)"
                        )
                except Exception as e:
                    error_msg = f"Failed to scale load with alpha={alpha:.3f}: {e}"
                    if logger:
                        logger.error(error_msg)
                    return False, 180.0, {}, {"error": error_msg}, None
            else:
                # No load device found - warn but continue (system may not need explicit load)
                if logger:
                    logger.warning(
                        f"[LOAD SCALING] No PQ load found in system. "
                        f"Load scaling (alpha={alpha:.3f}) may not be applied. "
                        f"System may use generator power directly."
                    )
                # Note: For SMIB systems, the generator may supply power directly to the infinite bus
                # without an explicit load device. In this case, Pm variation is equivalent to load variation.

        # Identify which generator is the infinite bus (slack) - should NOT be modified
        # Find main generator index (not infinite bus) - use find_main_generator_index()
        # This ensures consistent generator identification across all code paths
        gen_idx_to_modify = find_main_generator_index(ss)

        # Set generator parameters using alter() method (ANDES recommended approach)
        # CRITICAL: Use alter() AFTER setup() to ensure proper type conversion
        # alter() properly handles array/list conversion and ANDES internal state
        # CRITICAL: alter() needs device UID (from idx.v) or keyword arguments, not array index
        print(
            f"[CCT DEBUG] Setting Pm={Pm:.6f}, M={M:.2f}, D={D:.2f} on gen_idx={gen_idx_to_modify}"
            f" (using alter() method)"
        )
        try:
            # Get device identifier (UID or index) for alter()
            gen_identifier = gen_idx_to_modify  # Default to array index
            if hasattr(ss.GENCLS, "idx") and hasattr(ss.GENCLS.idx, "v"):
                try:
                    idx_array = ss.GENCLS.idx.v
                    if hasattr(idx_array, "__getitem__") and len(idx_array) > gen_idx_to_modify:
                        gen_uid = idx_array[gen_idx_to_modify]
                        gen_identifier = gen_uid  # Use UID if available (preferred)
                        print(f"[CCT DEBUG] Using device UID: {gen_identifier}")
                except (IndexError, AttributeError, TypeError):
                    # Fallback to array index
                    print(f"[CCT DEBUG] Using array index: {gen_identifier}")

            # Use alter() method (ANDES recommended) - works after setup()
            # Set M and D for ALL generators (uniform H/D); Pm/P0 only for main generator
            n_gen = getattr(ss.GENCLS, "n", 1)
            if hasattr(ss.GENCLS, "alter"):
                # Try with UID/index first
                try:
                    ss.GENCLS.alter("tm0", gen_identifier, Pm)
                    for gidx in range(n_gen):
                        ss.GENCLS.alter("M", gidx, M)
                        ss.GENCLS.alter("D", gidx, D)
                    # Also set P0 if it exists (for power flow setpoint)
                    if hasattr(ss.GENCLS, "P0"):
                        ss.GENCLS.alter("P0", gen_identifier, Pm)
                    print(
                        f"[CCT DEBUG] Parameters set via alter() method with identifier: {gen_identifier}"
                    )
                except (KeyError, IndexError, AttributeError) as alter_error:
                    # If alter() fails with identifier, try keyword arguments
                    print(
                        f"[CCT DEBUG] alter() with identifier failed: {alter_error}, trying keyword args..."
                    )
                    try:
                        ss.GENCLS.alter("tm0", gen_idx_to_modify, Pm)
                        for gidx in range(n_gen):
                            ss.GENCLS.alter("M", gidx, M)
                            ss.GENCLS.alter("D", gidx, D)
                        if hasattr(ss.GENCLS, "P0"):
                            ss.GENCLS.alter("P0", gen_idx_to_modify, Pm)
                        print(f"[CCT DEBUG] Parameters set via alter() with keyword args")
                    except Exception as kw_error:
                        # If both fail, fallback to direct access
                        print(
                            f"[CCT WARNING] alter() failed: {kw_error}, using direct access (fallback)"
                        )
                        raise alter_error  # Re-raise to trigger fallback
            else:
                # Fallback to direct access if alter() not available
                print("[CCT WARNING] alter() not available, using direct access (fallback)")
                raise AttributeError("alter() method not available")

        except Exception as e:
            # Fallback to direct access if alter() fails
            print(f"[CCT DEBUG] Falling back to direct access due to: {e}")
            try:
                if hasattr(ss.GENCLS.tm0, "v"):
                    if hasattr(ss.GENCLS.tm0.v, "__setitem__"):
                        ss.GENCLS.tm0.v[gen_idx_to_modify] = Pm
                    else:
                        ss.GENCLS.tm0.v = Pm

                if hasattr(ss.GENCLS.M, "v"):
                    if hasattr(ss.GENCLS.M.v, "__setitem__"):
                        for gidx in range(n_gen):
                            if gidx < len(ss.GENCLS.M.v):
                                ss.GENCLS.M.v[gidx] = M
                    else:
                        ss.GENCLS.M.v = M

                if hasattr(ss.GENCLS.D, "v"):
                    if hasattr(ss.GENCLS.D.v, "__setitem__"):
                        for gidx in range(n_gen):
                            if gidx < len(ss.GENCLS.D.v):
                                ss.GENCLS.D.v[gidx] = D
                    else:
                        ss.GENCLS.D.v = D

                if hasattr(ss.GENCLS, "P0") and hasattr(ss.GENCLS.P0, "v"):
                    if hasattr(ss.GENCLS.P0.v, "__setitem__"):
                        ss.GENCLS.P0.v[gen_idx_to_modify] = Pm
                    else:
                        ss.GENCLS.P0.v = Pm
                print(f"[CCT DEBUG] Parameters set via direct access (fallback)")
            except Exception as direct_error:
                error_msg = f"Failed to set generator parameters (both alter() and direct access failed): {direct_error}"
                print(f"[CCT ERROR] {error_msg}")
                if logger:
                    logger.error(error_msg)
                return False, 180.0, {}, {"error": error_msg}, None

        # Verify Pm was set correctly (after setup and alter(), before power flow)
        if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
            try:
                # CRITICAL: Check if tm0.v is array-like before accessing by index
                if hasattr(ss.GENCLS.tm0.v, "__getitem__") and hasattr(ss.GENCLS.tm0.v, "__len__"):
                    try:
                        if len(ss.GENCLS.tm0.v) > gen_idx_to_modify:
                            tm0_after_alter = ss.GENCLS.tm0.v[gen_idx_to_modify]
                        else:
                            tm0_after_alter = ss.GENCLS.tm0.v
                    except (TypeError, AttributeError):
                        # tm0.v is scalar or doesn't support len()
                        tm0_after_alter = ss.GENCLS.tm0.v
                else:
                    # tm0.v is scalar
                    tm0_after_alter = ss.GENCLS.tm0.v
                mismatch_pct = 100.0 * abs(tm0_after_alter - Pm) / (abs(Pm) + 1e-12)
                print(
                    f"[CCT DEBUG] After alter(): tm0.v[{gen_idx_to_modify}]={tm0_after_alter:.6f},"
                    f"requested={Pm:.6f}, mismatch={mismatch_pct:.3f}%"
                )
                if mismatch_pct > 0.01:  # More than 0.01% difference
                    # alter() may have failed, try direct access as fallback
                    print("[CCT WARNING] alter() may have failed, trying direct access fallback...")
                    if hasattr(ss.GENCLS.tm0.v, "__setitem__"):
                        ss.GENCLS.tm0.v[gen_idx_to_modify] = Pm
                    else:
                        ss.GENCLS.tm0.v = Pm
                    if hasattr(ss.GENCLS.M.v, "__setitem__"):
                        for gidx in range(n_gen):
                            if gidx < len(ss.GENCLS.M.v):
                                ss.GENCLS.M.v[gidx] = M
                    else:
                        ss.GENCLS.M.v = M
                    if hasattr(ss.GENCLS.D.v, "__setitem__"):
                        for gidx in range(n_gen):
                            if gidx < len(ss.GENCLS.D.v):
                                ss.GENCLS.D.v[gidx] = D
                    else:
                        ss.GENCLS.D.v = D
                    # CRITICAL: Also re-set P0 (power flow setpoint)
                    if hasattr(ss.GENCLS, "P0") and hasattr(ss.GENCLS.P0, "v"):
                        if hasattr(ss.GENCLS.P0.v, "__setitem__"):
                            ss.GENCLS.P0.v[gen_idx_to_modify] = Pm
                        else:
                            ss.GENCLS.P0.v = Pm
                    print(
                        f"[CCT FIX] Re-set parameters via direct access: Pm={Pm:.6f}, M={M:.2f},"
                        f"D={D:.2f}, P0={Pm:.6f}"
                    )
            except (IndexError, AttributeError, TypeError) as e:
                print(f"[CCT DEBUG] Verification failed: {e}")
                pass  # Verification failed, but continue

        # CRITICAL: Set Pm for both GENCLS.tm0 AND PV.p0 (ANDES requirement)
        # Main generator has both GENCLS (tm0) and PV (p0); both must match before power flow.
        # For multimachine use PV index = gen_idx_to_modify when available.
        try:
            # 1. Set PV.p0 for main generator (power flow input - voltage control)
            if hasattr(ss, "PV") and ss.PV.n > 0:
                if hasattr(ss.PV, "p0") and hasattr(ss.PV.p0, "v"):
                    pv_idx = gen_idx_to_modify if len(ss.PV.p0.v) > gen_idx_to_modify else 0
                    if hasattr(ss.PV.p0.v, "__setitem__"):
                        ss.PV.p0.v[pv_idx] = Pm
                    else:
                        ss.PV.p0.v = Pm
                    pv_p0_actual = (
                        ss.PV.p0.v[pv_idx]
                        if hasattr(ss.PV.p0.v, "__getitem__") and len(ss.PV.p0.v) > pv_idx
                        else ss.PV.p0.v
                    )
                    print(
                        f"[CCT DEBUG] Set PV power setpoint: PV.p0[{pv_idx}] = {Pm:.6f} pu "
                        f"(verified: {pv_p0_actual:.6f} pu)"
                    )
                else:
                    print(f"[CCT WARNING] PV.p0 not available")
            else:
                print(f"[CCT WARNING] No PV generators found")

            # 2. Set GENCLS.tm0 for main generator (mechanical power)
            if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                if hasattr(ss.GENCLS.tm0.v, "__setitem__"):
                    ss.GENCLS.tm0.v[gen_idx_to_modify] = Pm
                else:
                    ss.GENCLS.tm0.v = Pm
                # Verify tm0 was set correctly
                if hasattr(ss.GENCLS.tm0.v, "__getitem__") and hasattr(ss.GENCLS.tm0.v, "__len__"):
                    try:
                        if len(ss.GENCLS.tm0.v) > gen_idx_to_modify:
                            tm0_actual = ss.GENCLS.tm0.v[gen_idx_to_modify]
                        else:
                            tm0_actual = ss.GENCLS.tm0.v
                    except (TypeError, AttributeError):
                        tm0_actual = ss.GENCLS.tm0.v
                else:
                    tm0_actual = ss.GENCLS.tm0.v
                print(
                    f"[CCT DEBUG] Set mechanical power: GENCLS.tm0[{gen_idx_to_modify}] = {Pm:.6f} pu "
                    f"(verified: {tm0_actual:.6f} pu)"
                )
        except Exception as e:
            print(f"[CCT WARNING] Could not set power flow setpoint: {e}")
            import traceback

            print(f"[CCT WARNING] Traceback: {traceback.format_exc()}")

        # CRITICAL: Run power flow FIRST to establish steady-state equilibrium
        # This must be done BEFORE configuring the fault (following smib_albert_cct.py pattern)
        if hasattr(ss, "PFlow"):
            # CRITICAL: Reset convergence AND initialized flags to force complete re-run
            # When reusing system or after parameter changes, we need fresh power flow solution
            if hasattr(ss.PFlow, "converged"):
                ss.PFlow.converged = False
            if hasattr(ss.PFlow, "initialized"):
                ss.PFlow.initialized = False  # Clear cached PF solution

            # Also mark TDS as not initialized to force it to re-read power flow solution
            if hasattr(ss, "TDS") and hasattr(ss.TDS, "initialized"):
                ss.TDS.initialized = False

            # Run power flow
            try:
                ss.PFlow.run()
            except Exception as e:
                # If power flow fails, try to diagnose the issue
                error_msg = f"Power flow failed: {e}"
                # Check if slack bus issue
                if "slack" in str(e).lower() or "singular" in str(e).lower():
                    error_msg += " (Possible slack bus configuration issue)"
                if logger:
                    logger.error(error_msg)
                return False, 180.0, {}, {"error": error_msg}, None

            # Verify power flow converged
            if not (hasattr(ss.PFlow, "converged") and ss.PFlow.converged):
                # Try one more time with reset
                if hasattr(ss.PFlow, "converged"):
                    ss.PFlow.converged = False
                try:
                    ss.PFlow.run()
                except Exception as e:
                    error_msg = f"Power flow failed on retry: {e}"
                    if logger:
                        logger.error(error_msg)
                    return False, 180.0, {}, {"error": error_msg}, None
                if not (hasattr(ss.PFlow, "converged") and ss.PFlow.converged):
                    error_msg = "Power flow did not converge after retry"
                    if logger:
                        logger.error(error_msg)
                    return False, 180.0, {}, {"error": error_msg}, None

            # CRITICAL: Verify parameters are still correct AFTER power flow
            # Power flow may have used cached solution or reset parameters
            # For SMIB with NO LOAD: Verify power balance by checking infinite bus Pm
            # It should be -Pm (negative of main generator Pm)
            if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                try:
                    # Verify main generator tm0
                    tm0_after_pf = safe_get_tm0_value(ss, gen_idx_to_modify)
                    mismatch_pct = 100.0 * abs(tm0_after_pf - Pm) / (abs(Pm) + 1e-12)
                    print(
                        f"[CCT DEBUG] After power flow:"
                        f"tm0.v[{gen_idx_to_modify}]={tm0_after_pf:.6f}, requested={Pm:.6f},"
                        f"mismatch={mismatch_pct:.1f}%"
                    )

                    # For SMIB only (exactly 2 equivalent machines: one main gen + one infinite bus):
                    # verify power balance via infinite bus Pm. Skip for multimachine (n > 2).
                    is_smib = ss.GENCLS.n <= 2
                    if ss.GENCLS.n >= 1 and is_smib:
                        # Get infinite bus power from PV generator
                        inf_bus_pm = 0.0
                        found_inf_bus = False

                        if hasattr(ss, "PV") and ss.PV.n > 0:
                            if hasattr(ss.PV, "p0") and hasattr(ss.PV.p0, "v"):
                                pv_p0 = ss.PV.p0.v
                                # For SMIB: Look for the OTHER PV generator (not the main one we just set)
                                if hasattr(pv_p0, "__len__") and len(pv_p0) > 1:
                                    # CRITICAL FIX: Read from index 1 (infinite bus), not index 0 (main gen)
                                    # We set PV.p0[0] = Pm for main generator
                                    # So infinite bus should be at PV.p0[1]
                                    inf_bus_pm = float(pv_p0[1])
                                    found_inf_bus = True
                                    print(
                                        f"[CCT DEBUG] Found PV generator (infinite bus): "
                                        f"PV.p0[1] = {inf_bus_pm:.6f} pu"
                                    )

                        if found_inf_bus:
                            # Verify power balance: main generator + infinite bus ≈ 0
                            expected_inf_bus_pm = -Pm
                            inf_bus_mismatch_pct = (
                                100.0
                                * abs(inf_bus_pm - expected_inf_bus_pm)
                                / (abs(expected_inf_bus_pm) + 1e-12)
                            )

                            # Removed debug print for cleaner output

                            if inf_bus_mismatch_pct > 1.0:
                                error_msg = (
                                    f"[CCT ERROR] SMIB power balance violated after power flow!\n"
                                    f"  Main generator Pm: {tm0_after_pf:.6f} pu\n"
                                    f"  Infinite bus (PV) Pm: {inf_bus_pm:.6f} pu\n"
                                    f"  Expected infinite bus Pm: {expected_inf_bus_pm:.6f} pu\n"
                                    f"  Mismatch: {inf_bus_mismatch_pct:.2f}%\n"
                                    f"  For SMIB with no load, these should sum to zero."
                                )
                                print(error_msg)
                                if logger:
                                    logger.error(error_msg)
                                return False, 180.0, {}, {"error": error_msg}, None
                            # Removed success message for cleaner output
                        else:
                            # Removed warning for cleaner output - silently skip if no PV found
                            pass
                    elif ss.GENCLS.n > 2:
                        # Multimachine: skip SMIB power balance check (loads and multiple gens)
                        pass

                    # Also check main generator tm0 mismatch
                    if mismatch_pct > 0.01:  # More than 0.01% difference
                        error_msg = (
                            f"[CCT ERROR] Power flow changed tm0 after setting Pm!\n"
                            f"  Requested Pm: {Pm:.6f} pu\n"
                            f"  tm0 after PF: {tm0_after_pf:.6f} pu\n"
                            f"  Mismatch: {mismatch_pct:.2f}%\n"
                            f"  This is a fundamental ANDES parameter setting issue."
                        )
                        print(error_msg)
                        if logger:
                            logger.error(error_msg)
                        return False, 180.0, {}, {"error": error_msg}, None
                except (IndexError, AttributeError, TypeError) as e:
                    print(f"[CCT DEBUG] Post-PF verification failed: {e}")
                    pass  # Verification failed, but continue

            # CRITICAL: Validate system state after power flow
            # Check if system is in a valid state (not disconnected, has slack bus, etc.)
            # This catches cases where power flow "converges" but system is invalid
            try:
                # Check if buses have valid voltages (should be > 0 and < 2.0 pu typically)
                if hasattr(ss, "Bus") and hasattr(ss.Bus, "v") and hasattr(ss.Bus.v, "v"):
                    bus_voltages = ss.Bus.v.v
                    if len(bus_voltages) > 0:
                        # Check for invalid voltages (NaN, Inf, or extreme values)
                        if np.any(np.isnan(bus_voltages)) or np.any(np.isinf(bus_voltages)):
                            error_msg = (
                                "Invalid bus voltages detected (NaN/Inf) - "
                                "system may be disconnected"
                            )
                            if logger:
                                logger.error(error_msg)
                            return False, 180.0, {}, {"error": error_msg}, None
                        if np.any(bus_voltages <= 0) or np.any(bus_voltages > 2.0):
                            error_msg = (
                                f"Invalid bus voltages detected "
                                f"(range: {np.min(bus_voltages):.4f} - "
                                f"{np.max(bus_voltages):.4f} pu) - "
                                f"system may be in invalid state"
                            )
                            if logger:
                                logger.error(error_msg)
                            return False, 180.0, {}, {"error": error_msg}, None

                # Check if generators are properly connected
                if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
                    # Check if generator electrical power is reasonable
                    if hasattr(ss.GENCLS, "Pe") and hasattr(ss.GENCLS.Pe, "v"):
                        gen_power = ss.GENCLS.Pe.v
                        if len(gen_power) > 0:
                            if np.any(np.isnan(gen_power)) or np.any(np.isinf(gen_power)):
                                error_msg = "Invalid generator power detected (NaN/Inf) - system may be disconnected"
                                if logger:
                                    logger.error(error_msg)
                                return False, 180.0, {}, {"error": error_msg}, None
            except Exception as e:
                # If validation fails, it might indicate system is in invalid state
                error_msg = f"System state validation failed: {e} - system may be in invalid state"
                if logger:
                    logger.error(error_msg)
                return False, 180.0, {}, {"error": error_msg}, None
        else:
            error_msg = "PFlow module not available"
            if logger:
                logger.error(error_msg)
            return False, 180.0, {}, {"error": error_msg}, None

        # Now configure fault (disturbance) AFTER steady-state is established
        # Following smib_albert_cct.py pattern - use configure_fault() function
        fault_success, fault_was_added = configure_fault(
            ss, fault_start_time, clearing_time, fault_bus, fault_reactance, logger=logger
        )

        if not fault_success:
            error_msg = "Failed to configure fault"
            if logger:
                logger.error(error_msg)
            return False, 180.0, {}, {"error": error_msg}, None

        # Setup system (only if fault was added, otherwise already set up)
        setup_was_called = False
        if fault_was_added:
            try:
                ss.setup()
                setup_was_called = True
                # CRITICAL: Re-set generator parameters after setup() (it may reset them)
                # M and D: set for all generators (uniform H/D); Pm/P0 only for main generator
                n_gen_setup = getattr(ss.GENCLS, "n", 1)
                try:
                    if hasattr(ss.GENCLS, "alter"):
                        ss.GENCLS.alter("tm0", gen_idx_to_modify, Pm)
                        for gidx in range(n_gen_setup):
                            ss.GENCLS.alter("M", gidx, M)
                            ss.GENCLS.alter("D", gidx, D)
                        if hasattr(ss.GENCLS, "P0"):
                            ss.GENCLS.alter("P0", gen_idx_to_modify, Pm)
                    else:
                        change_generator_parameters(ss, gen_idx=gen_idx_to_modify, Pm=Pm, M=M, D=D)
                        # Ensure M and D for all generators (change_generator_parameters only sets one)
                        for gidx in range(n_gen_setup):
                            if gidx != gen_idx_to_modify and gidx < len(ss.GENCLS.M.v):
                                ss.GENCLS.M.v[gidx] = M
                            if gidx != gen_idx_to_modify and gidx < len(ss.GENCLS.D.v):
                                ss.GENCLS.D.v[gidx] = D

                    # Verify Pm was set correctly after setup()
                    if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                        try:
                            tm0_after_setup = safe_get_tm0_value(ss, gen_idx_to_modify)
                            mismatch_pct = 100.0 * abs(tm0_after_setup - Pm) / (abs(Pm) + 1e-12)
                            if mismatch_pct > 1.0:  # More than 1% difference
                                if logger:
                                    logger.warning(
                                        f"Pm mismatch after setup(): requested={Pm:.6f} pu, "
                                        f"actual={tm0_after_setup:.6f} pu ({mismatch_pct:.1f}%"
                                        f"difference)."
                                        f"Attempting direct access fix."
                                    )
                                # Try direct access as fallback
                                if hasattr(ss.GENCLS.tm0.v, "__getitem__"):
                                    ss.GENCLS.tm0.v[gen_idx_to_modify] = Pm
                                else:
                                    ss.GENCLS.tm0.v = Pm
                                # Also re-set P0 (power flow setpoint)
                                if hasattr(ss.GENCLS, "P0") and hasattr(ss.GENCLS.P0, "v"):
                                    if hasattr(ss.GENCLS.P0.v, "__setitem__"):
                                        ss.GENCLS.P0.v[gen_idx_to_modify] = Pm
                                    else:
                                        ss.GENCLS.P0.v = Pm
                        except (IndexError, AttributeError, TypeError):
                            pass  # Verification failed, but continue
                except Exception as param_err:
                    if logger:
                        logger.warning(f"Could not re-set parameters after setup(): {param_err}")
            except Exception as e:
                if logger:
                    logger.warning(f"Setup() failed or was already called: {e}")

        # Re-run power flow if setup() was called (it may have reset the power flow state)
        # Following smib_albert_cct.py pattern
        if setup_was_called and hasattr(ss, "PFlow"):
            # CRITICAL: Re-set power flow setpoint before power flow (after setup may have reset it)
            try:
                if hasattr(ss.GENCLS, "P0") and hasattr(ss.GENCLS.P0, "v"):
                    if hasattr(ss.GENCLS.P0.v, "__setitem__"):
                        ss.GENCLS.P0.v[gen_idx_to_modify] = Pm
                    else:
                        ss.GENCLS.P0.v = Pm
                    print(
                        f"[CCT DEBUG] Re-set power flow setpoint (after setup):"
                        f"P0.v[{gen_idx_to_modify}] = {Pm:.6f}"
                    )

                if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                    if hasattr(ss.GENCLS.tm0.v, "__setitem__"):
                        ss.GENCLS.tm0.v[gen_idx_to_modify] = Pm
                    else:
                        ss.GENCLS.tm0.v = Pm
                    print(
                        f"[CCT DEBUG] Re-set mechanical power (after setup):"
                        f"tm0.v[{gen_idx_to_modify}] = {Pm:.6f}"
                    )
            except Exception as e:
                print(f"[CCT WARNING] Could not re-set power flow setpoint: {e}")

            # CRITICAL: Reset convergence AND initialized flags to force complete re-run
            # This ensures fresh power flow solution after setup() (which may have reset state)
            if hasattr(ss.PFlow, "converged"):
                ss.PFlow.converged = False
            if hasattr(ss.PFlow, "initialized"):
                ss.PFlow.initialized = False  # Clear cached PF solution

            # Also mark TDS as not initialized to force it to re-read power flow solution
            if hasattr(ss, "TDS") and hasattr(ss.TDS, "initialized"):
                ss.TDS.initialized = False

            try:
                ss.PFlow.run()
                if not (hasattr(ss.PFlow, "converged") and ss.PFlow.converged):
                    error_msg = "Power flow did not converge after setup()"
                    if logger:
                        logger.error(error_msg)
                    return False, 180.0, {}, {"error": error_msg}, None

                # CRITICAL: Verify parameters are still correct AFTER power flow (after setup)
                # Power flow may have used cached solution or reset parameters
                if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                    try:
                        tm0_after_pf_setup = safe_get_tm0_value(ss, gen_idx_to_modify)
                        mismatch_pct = 100.0 * abs(tm0_after_pf_setup - Pm) / (abs(Pm) + 1e-12)
                        if mismatch_pct > 0.01:  # More than 0.01% difference
                            # Power flow reset parameters! Re-set them
                            if hasattr(ss.GENCLS.tm0.v, "__setitem__"):
                                ss.GENCLS.tm0.v[gen_idx_to_modify] = Pm
                            else:
                                ss.GENCLS.tm0.v = Pm
                            if hasattr(ss.GENCLS.M.v, "__setitem__"):
                                ss.GENCLS.M.v[gen_idx_to_modify] = M
                            else:
                                ss.GENCLS.M.v = M
                            if hasattr(ss.GENCLS.D.v, "__setitem__"):
                                ss.GENCLS.D.v[gen_idx_to_modify] = D
                            else:
                                ss.GENCLS.D.v = D
                            print(
                                f"[CCT FIX] Re-set parameters after PF (post-setup): Pm={Pm:.6f}"
                                f"(was {tm0_after_pf_setup:.6f}, {mismatch_pct:.1f}% mismatch)"
                            )
                            # Re-run power flow with correct parameters
                            if hasattr(ss.PFlow, "converged"):
                                ss.PFlow.converged = False
                            if hasattr(ss.PFlow, "initialized"):
                                ss.PFlow.initialized = False
                            try:
                                ss.PFlow.run()
                            except Exception as pf_retry_err:
                                print(f"[CCT WARNING] Power flow retry failed: {pf_retry_err}")
                    except (IndexError, AttributeError, TypeError) as e:
                        print(f"[CCT DEBUG] Post-PF (setup) verification failed: {e}")
                        pass  # Verification failed, but continue

            except Exception as e:
                error_msg = f"Power flow failed after setup(): {e}"
                if logger:
                    logger.error(error_msg)
                return False, 180.0, {}, {"error": error_msg}, None

        # CRITICAL: Verify power flow converged before proceeding
        # This ensures steady-state stability before the disturbance
        converged, pf_error = check_power_flow_convergence(ss)
        if not converged:
            error_msg = f"Power flow did not converge: {pf_error}"
            if logger:
                logger.error(error_msg)
            return False, 180.0, {}, {"error": error_msg}, None

        # Validate pre-fault equilibrium (using correct generator index)
        is_equilibrium, eq_error = validate_prefault_equilibrium(ss, gen_idx=gen_idx_to_modify)
        if not is_equilibrium:
            if logger:
                logger.warning(f"Pre-fault equilibrium validation failed: {eq_error}")
            return False, 180.0, {}, {"error": eq_error}, None

        # Configure TDS parameters (following ANDES manual: fixt, tstep, shrinkt)
        # DEFAULT: Use ANDES automatic time step (time_step=None)
        # OPTIONAL: Force user-specified time step (if time_step is provided)
        # ANDES will calculate optimal time step based on system frequency if not specified
        # CRITICAL: Set time step BEFORE init() to maximize chance of it being accepted
        if time_step is not None:
            # User wants to force a specific time step
            # Use ANDES's proper configuration parameters: fixt, tstep, shrinkt
            if hasattr(ss.TDS, "config"):
                # CRITICAL: Set fixt=1 for fixed step mode (ANDES manual parameter)
                if hasattr(ss.TDS.config, "fixt"):
                    ss.TDS.config.fixt = 1  # Fixed step mode (1 = fixed, 0 = adaptive)
                # Also try alternative names for compatibility
                if hasattr(ss.TDS.config, "fixed_step"):
                    ss.TDS.config.fixed_step = True

                # CRITICAL: Set tstep (initial/constant step size) - this is the key parameter!
                if hasattr(ss.TDS.config, "tstep"):
                    ss.TDS.config.tstep = time_step
                # Also set h as fallback
                if hasattr(ss.TDS.config, "h"):
                    ss.TDS.config.h = time_step
                if hasattr(ss.TDS.config, "dt"):
                    ss.TDS.config.dt = time_step

                # Set shrinkt to prevent step shrinking when Newton iterations fail
                if hasattr(ss.TDS.config, "shrinkt"):
                    ss.TDS.config.shrinkt = 0  # Don't allow shrinking for strict fixed step

                # Disable adaptive time stepping
                if hasattr(ss.TDS.config, "adaptive"):
                    ss.TDS.config.adaptive = False
                if hasattr(ss.TDS.config, "auto_h"):
                    ss.TDS.config.auto_h = False
                if hasattr(ss.TDS.config, "calc_h"):
                    ss.TDS.config.calc_h = False
            # Also try setting in solver if available
            if hasattr(ss.TDS, "solver") and hasattr(ss.TDS.solver, "h"):
                ss.TDS.solver.h = time_step
        # If time_step is None, ANDES will use its automatic calculation

        if simulation_time is not None:
            ss.TDS.config.tf = simulation_time
        if tolerance is not None:
            ss.TDS.config.tol = tolerance

        # CRITICAL: Disable automatic stability criteria checking
        # This allows simulation to run to completion so we can check stability ourselves
        # Otherwise ANDES will stop simulation early when criteria are violated
        # Set to 0 to disable: "To turn off, set [TDS].criteria = 0"
        if hasattr(ss.TDS.config, "criteria"):
            ss.TDS.config.criteria = 0  # 0 = disabled, allows full simulation

        # Disable Toggles (e.g. line trip at t=2s in Kundur) so only the fault event occurs
        # Aligns with scripts/run_kundur_fault_expt.py
        if hasattr(ss, "Toggle") and ss.Toggle.n > 0:
            try:
                for _i in range(ss.Toggle.n):
                    try:
                        if hasattr(ss.Toggle, "u") and hasattr(ss.Toggle.u, "v"):
                            ss.Toggle.u.v[_i] = 0
                        else:
                            ss.Toggle.alter("u", _i + 1, 0)
                    except Exception:
                        ss.Toggle.alter("u", _i + 1, 0)
            except Exception:
                pass

        # Enable plotter to store time series data
        if hasattr(ss.TDS.config, "plot"):
            ss.TDS.config.plot = True
        if hasattr(ss.TDS.config, "save_plt"):
            ss.TDS.config.save_plt = True
        if hasattr(ss.TDS.config, "store"):
            ss.TDS.config.store = True
        # CRITICAL: Set save_interval=1 to save every time step (not downsampled to 30 Hz)
        # This ensures output matches the requested time_step (e.g., 1 ms instead of 33.33 ms)
        if hasattr(ss.TDS.config, "save_interval"):
            ss.TDS.config.save_interval = 1  # Save every time step

        # CRITICAL: Verify tm0 is correct BEFORE calling TDS.init()
        # TDS.init() snapshots initial conditions, so they must be correct FIRST!
        if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
            try:
                tm0_before_tds_init = safe_get_tm0_value(ss, gen_idx_to_modify)
                mismatch_pct = 100.0 * abs(tm0_before_tds_init - Pm) / (abs(Pm) + 1e-12)
                if mismatch_pct > 0.01:  # More than 0.01% difference
                    # CRITICAL ERROR: setup() or power flow changed tm0!
                    error_msg = (
                        f"[CCT ERROR] setup() changed tm0 before TDS.init()!\n"
                        f"  Requested Pm: {Pm:.6f} pu\n"
                        f"  tm0 after setup: {tm0_before_tds_init:.6f} pu\n"
                        f"  Mismatch: {mismatch_pct:.2f}%\n"
                        f"  This is a fundamental ANDES parameter setting issue."
                    )
                    print(error_msg)
                    if logger:
                        logger.error(error_msg)
                    return False, 180.0, {}, {"error": error_msg}, None
                else:
                    print(
                        f"[CCT DEBUG] tm0 verified before TDS.init():"
                        f"tm0.v[{gen_idx_to_modify}]={tm0_before_tds_init:.6f}, requested={Pm:.6f},"
                        f"mismatch={mismatch_pct:.1f}% [OK]"
                    )
            except (IndexError, AttributeError, TypeError) as e:
                print(f"[CCT DEBUG] Pre-TDS.init() verification failed: {e}")
                pass  # Verification failed, but continue

        # CRITICAL: Verify power flow solution is valid before TDS.init()
        # If power flow failed (singular Jacobian), TDS.init() will fail with all NaN
        power_flow_valid = False
        if hasattr(ss, "PFlow") and hasattr(ss.PFlow, "converged"):
            power_flow_valid = ss.PFlow.converged
        else:
            # If no converged flag, check if we have valid bus voltages (not NaN)
            if hasattr(ss, "Bus") and hasattr(ss.Bus, "v") and hasattr(ss.Bus.v, "v"):
                bus_voltages = ss.Bus.v.v
                if len(bus_voltages) > 0:
                    power_flow_valid = not (
                        np.any(np.isnan(bus_voltages)) or np.any(np.isinf(bus_voltages))
                    )

        if not power_flow_valid:
            error_msg = (
                "Power flow did not converge or has invalid solution - cannot initialize TDS"
            )
            print(f"[CCT ERROR] {error_msg}")
            if logger:
                logger.error(error_msg)
            return False, 180.0, {}, {"error": error_msg}, None

        # CRITICAL: Reset TDS.initialized RIGHT BEFORE TDS.init() to ensure it re-reads power flow solution
        if hasattr(ss, "TDS") and hasattr(ss.TDS, "initialized"):
            ss.TDS.initialized = False  # Force TDS to re-read NEW power flow solution

        # Initialize TDS (power flow is already converged)
        # Note: Time step is set AFTER init() because ANDES may recalculate them during init()
        if hasattr(ss.TDS, "init"):
            ss.TDS.init()

            # CRITICAL: TDS.init() may reset parameters - check and fix immediately after init()
            # This prevents parameters from being wrong when TDS.run() is called
            if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                try:
                    tm0_after_init = safe_get_tm0_value(ss, gen_idx_to_modify)
                    mismatch_pct = 100.0 * abs(tm0_after_init - Pm) / (abs(Pm) + 1e-12)
                    if mismatch_pct > 0.01:  # More than 0.01% difference
                        print(
                            f"[CCT CRITICAL] TDS.init() reset tm0: tm0.v[{gen_idx_to_modify}]={tm0_after_init:.6f}, "
                            f"requested={Pm:.6f}, mismatch={mismatch_pct:.1f}%. Re-setting..."
                        )
                        # Re-set parameters immediately after TDS.init()
                        if hasattr(ss.GENCLS.tm0.v, "__setitem__"):
                            ss.GENCLS.tm0.v[gen_idx_to_modify] = Pm
                        else:
                            ss.GENCLS.tm0.v = Pm
                        n_gen_init = getattr(ss.GENCLS, "n", 1)
                        if hasattr(ss.GENCLS.M.v, "__setitem__"):
                            for gidx in range(n_gen_init):
                                if gidx < len(ss.GENCLS.M.v):
                                    ss.GENCLS.M.v[gidx] = M
                        else:
                            ss.GENCLS.M.v = M
                        if hasattr(ss.GENCLS.D.v, "__setitem__"):
                            for gidx in range(n_gen_init):
                                if gidx < len(ss.GENCLS.D.v):
                                    ss.GENCLS.D.v[gidx] = D
                        else:
                            ss.GENCLS.D.v = D
                        # Also re-set P0 if available
                        if hasattr(ss.GENCLS, "P0") and hasattr(ss.GENCLS.P0, "v"):
                            if hasattr(ss.GENCLS.P0.v, "__setitem__"):
                                ss.GENCLS.P0.v[gen_idx_to_modify] = Pm
                            else:
                                ss.GENCLS.P0.v = Pm

                        # CRITICAL: Also re-set PV.p0 (ANDES requires BOTH PV.p0 and GENCLS.tm0 to be synchronized)
                        # For multimachine use PV index matching gen_idx_to_modify when available
                        if hasattr(ss, "PV") and hasattr(ss.PV, "p0") and hasattr(ss.PV.p0, "v"):
                            if len(ss.PV.p0.v) > gen_idx_to_modify:
                                ss.PV.p0.v[gen_idx_to_modify] = Pm
                            elif len(ss.PV.p0.v) > 0:
                                ss.PV.p0.v[0] = Pm

                        print(
                            f"[CCT FIX] Re-set parameters immediately after TDS.init(): "
                            f"Pm={Pm:.6f}, M={M:.2f}, D={D:.2f}"
                        )
                        # Do NOT re-run PFlow + TDS.init() here: for Kundur/multimachine that
                        # causes "Slack generator is not defined" and singular Jacobian, leaving
                        # tm0 and DAE state as NaN. Only re-set parameters; TDS.run() will use
                        # them (initial conditions remain from first TDS.init()).
                        print(
                            "[CCT DEBUG] Parameters fixed after TDS.init() - will be used by TDS.run()"
                        )
                    else:
                        print(
                            f"[CCT DEBUG] Parameters OK after TDS.init(): "
                            f"tm0.v[{gen_idx_to_modify}]={tm0_after_init:.6f}, requested={Pm:.6f} [OK]"
                        )
                except (IndexError, AttributeError, TypeError) as e:
                    print(f"[CCT DEBUG] Parameter check after TDS.init() failed: {e}")
                    pass  # Continue even if check fails

            # DEFAULT: Use ANDES automatic time step (if time_step is None)
            # OPTIONAL: Force user-specified time step (if time_step is provided)
            if time_step is not None:
                # User wants to force a specific time step
                # CRITICAL: Re-set the proper ANDES parameters AFTER init() (ANDES may recalculate them during init())
                # CRITICAL: Use fixt, tstep, shrinkt (ANDES manual parameters)
                if hasattr(ss.TDS.config, "fixt"):
                    ss.TDS.config.fixt = 1  # Fixed step mode
                if hasattr(ss.TDS.config, "tstep"):
                    ss.TDS.config.tstep = time_step  # CRITICAL: This is the key parameter
                if hasattr(ss.TDS.config, "shrinkt"):
                    ss.TDS.config.shrinkt = 0  # Don't allow shrinking

                # Also set h/dt as fallback
                if hasattr(ss.TDS.config, "h"):
                    ss.TDS.config.h = time_step
                if hasattr(ss.TDS.config, "dt"):
                    ss.TDS.config.dt = time_step

                # Disable adaptive time stepping if available
                if hasattr(ss.TDS.config, "fixed_step"):
                    ss.TDS.config.fixed_step = True
                if hasattr(ss.TDS.config, "adaptive"):
                    ss.TDS.config.adaptive = False

                # Try to set in solver config if available
                if hasattr(ss.TDS, "solver") and hasattr(ss.TDS.solver, "h"):
                    ss.TDS.solver.h = time_step
                # Also try setting in the internal time step storage
                if hasattr(ss.TDS, "h"):
                    ss.TDS.h = time_step

                # Check actual time step after forcing (check tstep first, then h)
                actual_h = None
                if hasattr(ss.TDS.config, "tstep"):
                    actual_h = ss.TDS.config.tstep
                elif hasattr(ss.TDS.config, "h"):
                    actual_h = ss.TDS.config.h
                elif hasattr(ss.TDS, "solver") and hasattr(ss.TDS.solver, "h"):
                    actual_h = ss.TDS.solver.h

                # Warn if time step was overridden (but this is acceptable - ANDES prioritizes stability)
                if actual_h is not None and abs(actual_h - time_step) > 1e-6:
                    if logger:
                        logger.info(
                            f"Time step was set to {actual_h*1000:.2f} ms instead of"
                            f"{time_step*1000:.2f} ms"
                        )
                        logger.info(
                            "  ANDES overrode for numerical stability "
                            "(system frequency-based calculation)"
                        )
                        logger.info(
                            f"This is acceptable - {actual_h*1000:.2f} ms is standard for 60 Hz"
                            f"systems"
                        )
            else:
                # DEFAULT: Use ANDES automatic time step
                # Check what time step ANDES calculated
                actual_h = None
                if hasattr(ss.TDS.config, "h"):
                    actual_h = ss.TDS.config.h
                elif hasattr(ss.TDS, "solver") and hasattr(ss.TDS.solver, "h"):
                    actual_h = ss.TDS.solver.h

                if actual_h is not None and logger:
                    logger.info("Using ANDES automatic time step: {actual_h*1000:.2f} ms")
                    logger.info(
                        "  (Based on system frequency - standard for transient stability analysis)"
                    )

        # CRITICAL: Verify tm0 is correct RIGHT BEFORE TDS.run()
        # TDS uses initial conditions from power flow, so we must ensure they're correct
        if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
            try:
                tm0_before_tds_run = safe_get_tm0_value(ss, gen_idx_to_modify)
                mismatch_pct = 100.0 * abs(tm0_before_tds_run - Pm) / (abs(Pm) + 1e-12)
                if mismatch_pct > 0.01:  # More than 0.01% difference
                    # CRITICAL ERROR: tm0 changed before TDS.run()!
                    error_msg = (
                        f"[CCT ERROR] tm0 changed before TDS.run()!\n"
                        f"  Requested Pm: {Pm:.6f} pu\n"
                        f"  tm0 before TDS.run: {tm0_before_tds_run:.6f} pu\n"
                        f"  Mismatch: {mismatch_pct:.2f}%\n"
                        f"  This is a fundamental ANDES parameter setting issue."
                    )
                    print(error_msg)
                    if logger:
                        logger.error(error_msg)
                    return False, 180.0, {}, {"error": error_msg}, None
                else:
                    print(
                        f"[CCT DEBUG] tm0 verified before TDS.run():"
                        f"tm0.v[{gen_idx_to_modify}]={tm0_before_tds_run:.6f}, requested={Pm:.6f},"
                        f"mismatch={mismatch_pct:.1f}% [OK]"
                    )
            except (IndexError, AttributeError, TypeError) as e:
                print(f"[CCT DEBUG] Pre-TDS.run() verification failed: {e}")

        # Run TDS simulation
        # Suppress ANDES warnings about slack generator (expected when system becomes disconnected/unstable)
        try:
            with suppress_output():
                ss.TDS.run()

            # CRITICAL: Verify tm0 is still correct AFTER TDS.run()
            # TDS.run() may reset parameters or use cached initial conditions
            if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                try:
                    tm0_after_tds = safe_get_tm0_value(ss, gen_idx_to_modify)
                    mismatch_pct = 100.0 * abs(tm0_after_tds - Pm) / (abs(Pm) + 1e-12)
                    if mismatch_pct > 0.01:  # More than 0.01% difference
                        print(
                            f"[CCT CRITICAL] tm0 mismatch AFTER TDS.run():"
                            f"tm0.v[{gen_idx_to_modify}]={tm0_after_tds:.6f}, requested={Pm:.6f},"
                            f"mismatch={mismatch_pct:.1f}%"
                        )
                        print(
                            f"[CCT WARNING] TDS.run() reset tm0! This will cause Pe(t=0) mismatch."
                            f"Re-setting..."
                        )
                        # Re-set tm0 (though it's too late for this simulation, but helps for extraction)
                        if hasattr(ss.GENCLS.tm0.v, "__setitem__"):
                            ss.GENCLS.tm0.v[gen_idx_to_modify] = Pm
                        else:
                            ss.GENCLS.tm0.v = Pm
                        print(
                            f"[CCT FIX] Re-set tm0 after TDS.run(): tm0.v[{gen_idx_to_modify}] ="
                            f"{Pm:.6f}"
                        )
                    else:
                        print(
                            f"[CCT DEBUG] tm0 verified after TDS.run():"
                            f"tm0.v[{gen_idx_to_modify}]={tm0_after_tds:.6f}, requested={Pm:.6f},"
                            f"mismatch={mismatch_pct:.1f}% [OK]"
                        )
                except (IndexError, AttributeError, TypeError) as e:
                    print(f"[CCT DEBUG] Post-TDS.run() verification failed: {e}")
        except ValueError as ve:
            # Handle specific error: "could not broadcast input array from shape (0,) into shape (N,)"
            # This occurs when ANDES timer's _vstore is empty (system state corruption)
            error_msg = str(ve)
            if "could not broadcast" in error_msg and "shape (0,)" in error_msg:
                if logger:
                    logger.warning(
                        "System state corruption detected (empty _vstore). "
                        "Retrying with fresh system..."
                    )

                # Always retry with a fresh system reload, regardless of reload_system setting
                # This ensures we get a completely clean state
                # Limit retries to avoid infinite recursion
                if _retry_count < 1:  # Allow one retry
                    try:
                        # Recursively call with fresh system (force reload)
                        result = test_clearing_time(
                            case_path=case_path,
                            Pm=Pm,
                            M=M,
                            D=D,
                            clearing_time=clearing_time,
                            fault_start_time=fault_start_time,
                            fault_bus=fault_bus,
                            fault_reactance=fault_reactance,
                            simulation_time=simulation_time,
                            time_step=time_step,
                            tolerance=tolerance,
                            logger=logger,
                            ss=None,  # Force reload
                            reload_system=True,  # Use clean state
                            _retry_count=_retry_count + 1,  # Increment retry counter
                            alpha=alpha,  # NEW: Pass alpha for load scaling
                            base_load=base_load,  # NEW: Pass base_load for load scaling
                            addfile=addfile,
                        )
                        return result
                    except Exception as retry_error:
                        # If retry also fails, log and raise
                        if logger:
                            logger.error("Retry also failed: {retry_error}")
                        raise
                else:
                    # Too many retries, give up
                    if logger:
                        logger.error(
                            "Failed after {_retry_count} retries. This may indicate a persistent ANDES issue."
                        )
                    raise

        # Extract trajectories using the correct generator index
        # Note: gen_idx_to_modify was determined earlier (main generator, not infinite bus)
        trajectories_dict = extract_trajectories_with_derived(ss, gen_idx=gen_idx_to_modify)

        # Apply early stopping at max_angle_deg if specified
        truncated_at_360 = False
        if max_angle_deg is not None and "delta" in trajectories_dict:
            max_angle_rad = np.deg2rad(max_angle_deg)  # Convert to radians
            delta = trajectories_dict["delta"]
            time = trajectories_dict.get("time", np.arange(len(delta)))

            # Find first index where |delta| exceeds max_angle_rad
            abs_delta = np.abs(delta)
            exceed_mask = abs_delta > max_angle_rad

            if np.any(exceed_mask):
                # Find first index exceeding limit
                first_exceed_idx = np.where(exceed_mask)[0]
                if len(first_exceed_idx) > 0:
                    truncate_idx = first_exceed_idx[0]
                    # Truncate all trajectories at this index (keep up to but not including exceed point)
                    for key in trajectories_dict:
                        if (
                            isinstance(trajectories_dict[key], np.ndarray)
                            and len(trajectories_dict[key]) > truncate_idx
                        ):
                            trajectories_dict[key] = trajectories_dict[key][:truncate_idx]
                    truncated_at_360 = True
                    if logger:
                        logger.info(
                            f"Trajectory truncated at t={time[truncate_idx-1]:.4f}s "
                            f"(angle exceeded {max_angle_deg}° limit)"
                        )

        # Check stability: use COI-based criterion when multimachine (n_gen > 1)
        n_gen = getattr(ss.GENCLS, "n", 1)
        if n_gen > 1 and check_stability_multimachine_coi is not None:
            time_arr = trajectories_dict.get("time", np.array([]))
            delta_per_gen = []
            omega_per_gen = []
            for i in range(n_gen):
                traj_i = extract_trajectories_with_derived(ss, gen_idx=i)
                d_i = np.asarray(traj_i.get("delta", np.zeros(0))).flatten()
                o_i = np.asarray(
                    traj_i.get("omega", np.ones_like(d_i) if d_i.size else np.zeros(0))
                ).flatten()
                if time_arr.size > 0 and d_i.size != time_arr.size:
                    t_i = np.asarray(traj_i.get("time", time_arr)).flatten()
                    if t_i.size == d_i.size and t_i.size > 0:
                        d_i = np.interp(time_arr, t_i, d_i)
                        o_i = (
                            np.interp(time_arr, t_i, o_i)
                            if o_i.size == t_i.size
                            else np.ones_like(time_arr)
                        )
                    else:
                        d_i = np.resize(d_i, time_arr.size) if time_arr.size else d_i
                        o_i = np.resize(o_i, time_arr.size) if time_arr.size else o_i
                delta_per_gen.append(d_i)
                omega_per_gen.append(o_i)
            # Align lengths so COI stack is valid (use min length if still inconsistent)
            if delta_per_gen:
                min_len = min(len(d) for d in delta_per_gen)
                if min_len > 0:
                    delta_per_gen = [np.asarray(d).flatten()[:min_len] for d in delta_per_gen]
                    o_flat = [np.asarray(o).flatten() for o in omega_per_gen]
                    omega_per_gen = [
                        o[:min_len] if len(o) >= min_len else np.ones(min_len, dtype=float)
                        for o in o_flat
                    ]
            M_vals = (
                np.asarray(ss.GENCLS.M.v).flatten()[:n_gen]
                if hasattr(ss.GENCLS, "M") and hasattr(ss.GENCLS.M, "v")
                else np.zeros(n_gen)
            )
            if (
                delta_per_gen
                and all(len(d) > 0 for d in delta_per_gen)
                and len(M_vals) >= n_gen
                and np.sum(M_vals) > 0
            ):
                is_stable = check_stability_multimachine_coi(
                    delta_per_gen=delta_per_gen,
                    omega_per_gen=omega_per_gen,
                    M_vals=M_vals,
                    delta_threshold=np.pi,
                    omega_threshold=1.5,
                )
                delta_stack = np.array([np.asarray(d).flatten() for d in delta_per_gen])
                n_time = delta_stack.shape[1]
                M_sum = M_vals.sum()
                delta_coi = np.sum(M_vals[:, np.newaxis] * delta_stack, axis=0) / M_sum
                delta_rel = delta_stack - delta_coi
                max_angle = float(np.degrees(np.abs(delta_rel).max())) if n_time > 0 else 180.0
                stability_metrics = {"max_angle_deg": max_angle}
            else:
                is_stable, stability_metrics = check_stability(ss, trajectories_dict)
        else:
            is_stable, stability_metrics = check_stability(ss, trajectories_dict)

        # Add truncation metadata
        if truncated_at_360:
            stability_metrics["truncated_at_360"] = True
            stability_metrics["truncation_reason"] = f"Angle exceeded {max_angle_deg}° limit"
        else:
            stability_metrics["truncated_at_360"] = False

        # Get max angle
        max_angle = stability_metrics.get("max_angle_deg", 180.0)

        return is_stable, max_angle, trajectories_dict, stability_metrics, ss

    except Exception as e:
        error_msg = str(e)
        if logger:
            logger.error(
                f"Error testing clearing time {clearing_time:.4f}s: {error_msg}", exc_info=True
            )
        return False, 180.0, {}, {"error": error_msg}, None


def find_cct(
    case_path: str,
    Pm: float,
    M: float,
    D: float,
    fault_start_time: float = 1.0,
    fault_bus: int = 3,
    fault_reactance: float = 0.0001,
    min_tc: float = 0.1,
    max_tc: float = 2.0,
    simulation_time: float = 5.0,
    time_step: Optional[float] = None,
    tolerance_initial: float = 0.01,
    tolerance_final: float = 0.001,
    max_iterations: int = 50,
    logger: Optional[logging.Logger] = None,
    ss: Optional[Any] = None,
    reload_system: bool = False,
    alpha: Optional[float] = None,  # NEW: Load multiplier for load variation
    base_load: Optional[Dict[str, float]] = None,  # NEW: Base load values for alpha scaling
    addfile: Optional[str] = None,  # Optional DYR/addfile path (e.g. Kundur GENCLS .dyr)
) -> Tuple[Optional[float], Optional[float], Optional[Dict], Optional[Dict]]:
    """
    Find Critical Clearing Time (CCT) using binary search.

    DEFAULT: Follows ANDES batch processing pattern (reuses system object if provided)
    OPTIONAL: Can reload system for each test if reload_system=True (clean state approach)

    Parameters:
    -----------
    case_path : str
        Path to ANDES case file
    Pm : float
        Mechanical power (pu)
    M : float
        Inertia constant (seconds)
    D : float
        Damping coefficient (pu)
    fault_start_time : float
        Fault start time (seconds)
    fault_bus : int
        Bus where fault occurs
    fault_reactance : float
        Fault reactance (pu)
    min_tc : float
        Minimum clearing time to search (seconds)
    max_tc : float
        Maximum clearing time to search (seconds)
    simulation_time : float
        Total simulation time (seconds)
    time_step : float, optional
        Time step (seconds). If None (default), uses ANDES automatic time step
        based on system frequency (e.g., 33.33 ms for 60 Hz systems)
        If specified, attempts to force this time step (may be overridden by ANDES
        for numerical stability)
    tolerance_initial : float
        Initial tolerance for binary search (seconds)
    tolerance_final : float
        Final tolerance for binary search (seconds)
    max_iterations : int
        Maximum number of iterations
    logger : logging.Logger, optional
        Logger for progress messages
    ss : andes.System, optional
        System object to reuse (ANDES batch processing pattern)
    reload_system : bool
        If True, reload system for each test (clean state approach)

    Returns:
    --------
    cct : float or None
        Critical Clearing Time (seconds), or None if not found
    uncertainty : float or None
        Uncertainty in CCT (seconds)
    stable_result : dict or None
        Result for stable case (CCT - small_delta)
    unstable_result : dict or None
        Result for unstable case (CCT + small_delta)
    """
    # DEFAULT: Follow ANDES batch processing pattern (load system once, reuse it).
    # OPTIONAL: Reload system for each test if reload_system=True
    if ss is None and not reload_system:
        # Load system once (ANDES batch processing pattern)
        load_kw = dict(setup=False, no_output=True, default_config=True)
        if addfile:
            load_kw["addfile"] = addfile
        ss = andes.load(case_path, **load_kw)
        # Add Fault before setup() so ANDES initializes with the fault (avoids "Failed to configure fault" when adding after setup)
        if not hasattr(ss, "Fault") or ss.Fault.n == 0:
            try:
                ss.add(
                    "Fault",
                    {
                        "u": 1,
                        "name": f"F_bus{fault_bus}",
                        "bus": fault_bus,
                        "tf": fault_start_time,
                        "tc": min_tc,
                        "rf": 0.0,
                        "xf": fault_reactance,
                    },
                )
                if logger:
                    logger.info(f"[CCT] Added Fault at bus {fault_bus} before setup (tc={min_tc}s)")
            except Exception as e:
                msg = f"[CCT] Could not add Fault before setup: {e}"
                print(msg)
                if logger:
                    logger.error(msg)
                return None, None, None, None
        ss.setup()

        # Validate that GENCLS exists and has devices
        if not hasattr(ss, "GENCLS") or ss.GENCLS.n == 0:
            if logger:
                logger.error("GENCLS generator model not found or has no devices")
            return None, None, None, None

        # Test boundaries first
        if logger:
            logger.info(f"Testing boundaries: min_tc={min_tc:.4f}s, max_tc={max_tc:.4f}s")

    # Test minimum
    is_stable_min, max_angle_min, _, metrics_min, ss = test_clearing_time(
        case_path,
        Pm,
        M,
        D,
        min_tc,
        fault_start_time,
        fault_bus,
        fault_reactance,
        simulation_time,
        time_step,
        logger=logger,
        ss=ss,
        reload_system=reload_system,
        alpha=alpha,  # NEW: Pass alpha for load scaling
        base_load=base_load,  # NEW: Pass base_load for load scaling
        addfile=addfile,
    )

    # Test maximum
    is_stable_max, max_angle_max, _, metrics_max, ss = test_clearing_time(
        case_path,
        Pm,
        M,
        D,
        max_tc,
        fault_start_time,
        fault_bus,
        fault_reactance,
        simulation_time,
        time_step,
        logger=logger,
        ss=ss,
        reload_system=reload_system,
        alpha=alpha,  # NEW: Pass alpha for load scaling
        base_load=base_load,  # NEW: Pass base_load for load scaling
        addfile=addfile,
    )

    # Edge cases
    if not is_stable_min and not is_stable_max:
        # Always unstable: both short and long clearing times give unstable
        err_min = (metrics_min or {}).get("error", "")
        err_max = (metrics_max or {}).get("error", "")
        err_extra = ""
        if err_min or err_max:
            err_extra = f" Errors: min_tc={err_min!r}; max_tc={err_max!r}."
        msg = (
            f"[CCT] System always unstable (min_tc={min_tc:.3f}s, max_tc={max_tc:.3f}s). "
            f"min_tc max_angle={max_angle_min:.1f}°, max_tc max_angle={max_angle_max:.1f}°."
            f"{err_extra} "
            f"Possible causes: (1) CCT < min_tc (try shorter min_tc), (2) power flow / TDS error above, "
            f"(3) wrong operating point (Pm/load mismatch vs trajectory run), (4) stability criterion too strict."
        )
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None, None, None, None

    if is_stable_min and is_stable_max:
        # Always stable: CCT is longer than max_tc
        msg = (
            f"[CCT] System always stable (tested up to max_tc={max_tc:.3f}s). "
            f"CCT > max_tc. Try increasing max_tc or check fault is applied."
        )
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None, None, None, None

    # Binary search
    low = min_tc if is_stable_min else min_tc
    high = max_tc if not is_stable_max else max_tc

    # Ensure we have stable at low and unstable at high
    if not is_stable_min:
        # Find stable point by searching backwards
        test_tc = min_tc
        step = 0.01
        for _ in range(20):
            test_tc -= step
            if test_tc < fault_start_time:
                if logger:
                    logger.warning("Could not find stable point below minimum")
                return None, None, None, None
            if logger:
                logger.info(f"  Searching for stable point: tc={test_tc:.4f}s...")
            is_stable, _, _, _, ss = test_clearing_time(
                case_path,
                Pm,
                M,
                D,
                test_tc,
                fault_start_time,
                fault_bus,
                fault_reactance,
                simulation_time,
                time_step,
                logger=logger,
                ss=ss,
                reload_system=reload_system,
                alpha=alpha,  # NEW: Pass alpha for load scaling
                base_load=base_load,  # NEW: Pass base_load for load scaling
                addfile=addfile,
            )
            if is_stable:
                low = test_tc
                break
        else:
            if logger:
                logger.warning("Could not find stable point")
            return None, None, None, None

    if is_stable_max:
        # Find unstable point by searching forwards
        test_tc = max_tc
        step = 0.01
        for _ in range(20):
            test_tc += step
            if test_tc > simulation_time - 0.5:
                if logger:
                    logger.warning("Could not find unstable point above maximum")
                return None, None, None, None
            if logger:
                logger.info(f"  Searching for unstable point: tc={test_tc:.4f}s...")
            is_stable, _, _, _, ss = test_clearing_time(
                case_path,
                Pm,
                M,
                D,
                test_tc,
                fault_start_time,
                fault_bus,
                fault_reactance,
                simulation_time,
                time_step,
                logger=logger,
                ss=ss,
                reload_system=reload_system,
                alpha=alpha,  # NEW: Pass alpha for load scaling
                base_load=base_load,  # NEW: Pass base_load for load scaling
                addfile=addfile,
            )
            if not is_stable:
                high = test_tc
                break
        else:
            if logger:
                logger.warning("Could not find unstable point")
            return None, None, None, None

    # Binary search with adaptive tolerance
    tolerance = tolerance_initial
    iteration = 0

    while (high - low) > tolerance and iteration < max_iterations:
        iteration += 1
        mid = (low + high) / 2.0

        if logger:
            logger.info(
                f"Iteration {iteration}: Testing tc={mid:.6f}s (range: [{low:.6f}, {high:.6f}])"
            )

        is_stable, max_angle, _, metrics, ss = test_clearing_time(
            case_path,
            Pm,
            M,
            D,
            mid,
            fault_start_time,
            fault_bus,
            fault_reactance,
            simulation_time,
            time_step,
            logger=logger,
            ss=ss,
            reload_system=reload_system,
            alpha=alpha,  # NEW: Pass alpha for load scaling
            base_load=base_load,  # NEW: Pass base_load for load scaling
            addfile=addfile,
        )

        if is_stable:
            low = mid  # CCT is >= mid
        else:
            high = mid  # CCT is < mid

        # Refine tolerance near boundary
        if (high - low) < 0.01:
            tolerance = tolerance_final

    # CCT is the maximum stable clearing time (absolute)
    cct_absolute = low

    # Calculate uncertainty
    uncertainty = (high - low) / 2.0

    # CCT as duration (fault duration = clearing_time - fault_start_time)
    cct_duration = cct_absolute - fault_start_time

    # Calculate small_delta (ε) for stable/unstable test cases
    # Minimum offset of 4ms (0.004s) to avoid numerical precision issues at boundary
    small_delta = max(0.004, uncertainty)

    # Test stable and unstable cases near CCT (use absolute clearing times)
    stable_tc = max(fault_start_time, cct_absolute - small_delta)
    unstable_tc = min(simulation_time - 0.5, cct_absolute + small_delta)

    if logger:
        logger.info(f"Testing stable case: tc={stable_tc:.6f}s")
    is_stable_stable, _, traj_stable, metrics_stable, ss_stable = test_clearing_time(
        case_path,
        Pm,
        M,
        D,
        stable_tc,
        fault_start_time,
        fault_bus,
        fault_reactance,
        simulation_time,
        time_step,
        logger=logger,
        ss=ss,
        reload_system=reload_system,
        alpha=alpha,  # NEW: Pass alpha for load scaling
        base_load=base_load,  # NEW: Pass base_load for load scaling
        addfile=addfile,
    )

    if logger:
        logger.info(f"Testing unstable case: tc={unstable_tc:.6f}s")
    is_stable_unstable, _, traj_unstable, metrics_unstable, ss_unstable = test_clearing_time(
        case_path,
        Pm,
        M,
        D,
        unstable_tc,
        fault_start_time,
        fault_bus,
        fault_reactance,
        simulation_time,
        time_step,
        logger=logger,
        ss=ss_stable if not reload_system else None,
        reload_system=reload_system,
        alpha=alpha,  # NEW: Pass alpha for load scaling
        base_load=base_load,  # NEW: Pass base_load for load scaling
        addfile=addfile,
    )

    stable_result = {
        "clearing_time": stable_tc,
        "is_stable": is_stable_stable,
        "trajectories": traj_stable,
        "metrics": metrics_stable,
        "system": ss_stable,
    }

    unstable_result = {
        "clearing_time": unstable_tc,
        "is_stable": is_stable_unstable,
        "trajectories": traj_unstable,
        "metrics": metrics_unstable,
        "system": ss_unstable,
    }

    # Return CCT as duration, and include small_delta in results for summary box
    # Store absolute clearing time in results for plotting
    stable_result["cct_absolute"] = cct_absolute
    unstable_result["cct_absolute"] = cct_absolute
    stable_result["small_delta"] = small_delta
    unstable_result["small_delta"] = small_delta

    if logger:
        logger.info(f"CCT found (duration): {cct_duration:.6f}s ± {uncertainty:.6f}s")
        logger.info(f"  CCT (absolute clearing time): {cct_absolute:.6f}s")
        logger.info(f"  ε (small_delta): {small_delta:.6f}s")

    return cct_duration, uncertainty, stable_result, unstable_result


def find_cct_with_uncertainty(
    case_path: str,
    Pm: float,
    M: float,
    D: float,
    fault_start_time: float = 1.0,
    fault_bus: int = 3,
    fault_reactance: float = 0.0001,
    min_tc: float = 0.1,
    max_tc: float = 2.0,
    simulation_time: float = 5.0,
    time_step: Optional[float] = None,
    n_samples: int = 10,
    logger: Optional[logging.Logger] = None,
    reload_system: bool = False,
    alpha: Optional[float] = None,
    base_load: Optional[Dict[str, float]] = None,
    addfile: Optional[str] = None,
) -> Tuple[Optional[float], Optional[Tuple[float, float]], Dict[str, Any]]:
    """
    Find CCT with statistical uncertainty quantification.

    Tests multiple points near boundary and fits logistic regression
    to estimate CCT distribution.

    Parameters:
    -----------
    case_path : str
        Path to ANDES case file
    Pm : float
        Mechanical power (pu)
    M : float
        Inertia constant (seconds)
    D : float
        Damping coefficient (pu)
    fault_start_time : float
        Fault start time (seconds)
    fault_bus : int
        Bus where fault occurs
    fault_reactance : float
        Fault reactance (pu)
    min_tc : float
        Minimum clearing time (seconds)
    max_tc : float
        Maximum clearing time (seconds)
    simulation_time : float
        Total simulation time (seconds)
    time_step : float, optional
        Time step (seconds). If None (default), uses ANDES automatic time step
        based on system frequency (e.g., 33.33 ms for 60 Hz systems)
        If specified, attempts to force this time step (may be overridden by ANDES
        for numerical stability)
    n_samples : int
        Number of points to test near boundary (default: 10)
    logger : logging.Logger, optional
        Logger for progress messages
    reload_system : bool
        If True, reload system for each test (clean state approach for reliability)
        If False, load once and reuse system (ANDES batch processing pattern - faster but may have state issues)
        Default: False (ANDES pattern)

    Returns:
    --------
    cct : float or None
        Estimated CCT (seconds)
    confidence_interval : tuple or None
        (lower, upper) 95% confidence interval
    uncertainty_stats : dict
        Statistical information about uncertainty
    """
    # First find approximate CCT using binary search.
    # Use reload_system parameter passed from caller
    cct_approx, _, _, _ = find_cct(
        case_path,
        Pm,
        M,
        D,
        fault_start_time,
        fault_bus,
        fault_reactance,
        min_tc,
        max_tc,
        simulation_time,
        time_step,
        logger=logger,
        reload_system=reload_system,
        alpha=alpha,
        base_load=base_load,
        addfile=addfile,
    )

    if cct_approx is None:
        return None, None, {}

    # Test points around CCT
    # If reload_system=True, reload for each test (clean state approach)
    # If reload_system=False, load once and reuse (ANDES batch processing pattern)
    ss = None
    if not reload_system:
        # Load system once and reuse it for all test points (ANDES batch processing pattern)
        load_kw = dict(setup=False, no_output=True, default_config=True)
        if addfile:
            load_kw["addfile"] = addfile
        ss = andes.load(case_path, **load_kw)
        ss.setup()

        # Validate that GENCLS exists and has devices
        if not hasattr(ss, "GENCLS") or ss.GENCLS.n == 0:
            if logger:
                logger.error("GENCLS generator model not found or has no devices")
            return None, None, {}

    test_range = 0.05  # Test ±50ms around CCT
    test_times = np.linspace(
        max(fault_start_time, cct_approx - test_range),
        min(simulation_time - 0.5, cct_approx + test_range),
        n_samples,
    )

    stability_results = []
    for tc in test_times:
        is_stable, _, _, metrics, ss = test_clearing_time(
            case_path,
            Pm,
            M,
            D,
            tc,
            fault_start_time,
            fault_bus,
            fault_reactance,
            simulation_time,
            time_step,
            logger=logger,
            ss=ss,
            reload_system=reload_system,
            alpha=alpha,  # NEW: Pass alpha for load scaling (if None, uses Pm directly)
            base_load=base_load,  # NEW: Pass base_load for load scaling
            addfile=addfile,
        )
        stability_results.append({"clearing_time": tc, "is_stable": is_stable, "metrics": metrics})

    # Fit logistic regression to estimate CCT distribution
    try:
        from scipy.optimize import curve_fit

        # Prepare data
        tcs = np.array([r["clearing_time"] for r in stability_results])
        stable_flags = np.array([1 if r["is_stable"] else 0 for r in stability_results])

        # Logistic function: P(stable) = 1 / (1 + exp(-k*(t - t0)))
        def logistic(t, k, t0):
            return 1.0 / (1.0 + np.exp(-k * (t - t0)))

        # Initial guess
        p0 = [10.0, cct_approx]

        # Fit
        popt, pcov = curve_fit(logistic, tcs, stable_flags, p0=p0, maxfev=1000)
        k, t0 = popt

        # CCT is where P(stable) = 0.5, which is at t = t0
        cct_estimated = t0

        # Calculate confidence interval (95%)
        # Use standard error from covariance matrix
        t0_std = np.sqrt(pcov[1, 1])
        confidence_interval = (t0 - 1.96 * t0_std, t0 + 1.96 * t0_std)

        uncertainty_stats = {
            "cct_estimated": cct_estimated,
            "cct_approximate": cct_approx,
            "k_parameter": k,
            "t0_std": t0_std,
            "n_samples": n_samples,
            "test_range": test_range,
        }

        return cct_estimated, confidence_interval, uncertainty_stats

    except Exception as e:
        if logger:
            logger.warning(
                f"Statistical uncertainty calculation failed: {e}. Using binary search result."
            )
        return cct_approx, None, {"method": "binary_search_only"}
