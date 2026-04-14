#!/usr/bin/env python3
"""
System manager module.

ANDES system operations and utilities.
"""

import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False


@contextmanager
def suppress_output():
    """
    Context manager to suppress both stdout and stderr (for ANDES warnings)

    Usage:
        with suppress_output():
            ss.TDS.run()
    """
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


def load_andes_system(
    case_path: str, setup: bool = False, no_output: bool = True, default_config: bool = True
) -> Optional[Any]:
    """
    Load ANDES system from case file.

    Parameters:
    -----------
    case_path : str
        Path to ANDES case file
    setup : bool
        Whether to call setup() immediately
    no_output : bool
        Suppress ANDES output
    default_config : bool
        Use default configuration

    Returns:
    --------
    ss : andes.System or None
        Loaded system object
    """
    if not ANDES_AVAILABLE:
        return None

    try:
        ss = andes.load(case_path, setup=setup, no_output=no_output, default_config=default_config)
        return ss
    except Exception as e:
        print("Error loading ANDES system: {e}")
        return None


def safe_get_array_value(arr: Any, idx: int, default: Any = None) -> Any:
    """
    Safely get a value from an array by index.

    Parameters:
    -----------
    arr : array-like or scalar
        Array or scalar value
    idx : int
        Index to access
    default : any
        Default value if access fails

    Returns:
    --------
    value : any
        Array value at index, scalar value, or default
    """
    if arr is None:
        return default

    # If it's not indexable, return as-is (scalar)
    if not hasattr(arr, "__getitem__"):
        return arr

    # If it's indexable, check if it has elements
    try:
        arr_len = len(arr)
        if arr_len == 0:
            return default
        if idx >= arr_len:
            return default
        return arr[idx]
    except (TypeError, IndexError):
        # If len() fails or indexing fails, try to return as scalar
        try:
            return arr
        except Exception:
            return default


def compute_pmax_prefault(ss: Any, gen_idx: int) -> float:
    """
    Compute maximum power transfer (P_max) for pre-fault network.

    For SMIB: P_max = (V_gen * V_inf) / X_equiv
    where X_equiv is the equivalent reactance between generator and infinite bus.

    Parameters:
    -----------
    ss : andes.System
        ANDES system object (after power flow)
    gen_idx : int
        Generator index (main generator, not infinite bus)

    Returns:
    --------
    P_max : float
        Maximum power transfer (pu)
    """
    if not ANDES_AVAILABLE:
        raise ImportError("ANDES is not available. Cannot compute P_max.")

    # Extract generator bus voltage
    V_gen = 1.0  # Default
    if hasattr(ss.GENCLS, "bus") and hasattr(ss.GENCLS.bus, "v") and hasattr(ss, "Bus"):
        gen_bus = safe_get_array_value(ss.GENCLS.bus.v, gen_idx)
        if gen_bus is not None:
            bus_indices = ss.Bus.idx.v if hasattr(ss.Bus.idx, "v") else []
            if hasattr(bus_indices, "__iter__"):
                try:
                    bus_idx = list(bus_indices).index(gen_bus)
                    if hasattr(ss.Bus, "v") and hasattr(ss.Bus.v, "v"):
                        V_gen = safe_get_array_value(ss.Bus.v.v, bus_idx, default=1.0)
                except (ValueError, IndexError):
                    pass

    # Infinite bus voltage (constant, typically 1.0 pu)
    V_inf = 1.0

    # Compute equivalent reactance (network reduction)
    # Reuse existing reactance extraction function
    try:
        from .data_extractor import extract_network_reactances

        reactances = extract_network_reactances(ss, fault_bus=3)
        X_equiv = reactances.get("Xprefault", 0.5)  # Default if not found
    except Exception:
        # Fallback: Compute from network components
        # Generator transient reactance (Xd')
        Xd_prime = 0.25  # Default
        if hasattr(ss.GENCLS, "xd1") and hasattr(ss.GENCLS.xd1, "v"):
            Xd_prime = float(safe_get_array_value(ss.GENCLS.xd1.v, gen_idx, default=0.25))
        elif hasattr(ss.GENCLS, "xd") and hasattr(ss.GENCLS.xd, "v"):
            Xd_prime = float(safe_get_array_value(ss.GENCLS.xd.v, gen_idx, default=0.25))

        # Transmission line reactances
        Xline = 0.0
        if hasattr(ss, "Line") and ss.Line.n > 0:
            line_data = ss.Line.as_df()
            if "x" in line_data.columns:
                line_reactances = line_data["x"].values
                if np.all(line_reactances > 0):
                    if len(line_reactances) > 1:
                        # Parallel lines: 1/Xeq = sum(1/Xi)
                        Xline = 1.0 / np.sum(1.0 / line_reactances)
                    else:
                        Xline = float(line_reactances[0])

        X_equiv = Xd_prime + Xline

    # Compute P_max
    if X_equiv <= 0:
        raise ValueError(f"Invalid equivalent reactance: X_equiv={X_equiv} pu")

    P_max = (V_gen * V_inf) / X_equiv
    return float(P_max)


def validate_prefault_equilibrium(
    ss: Any, pm_value: float, gen_idx: int, tolerance: float = 0.01
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate pre-fault equilibrium exists and is stable.

    Parameters:
    -----------
    ss : andes.System
        ANDES system object (after power flow)
    pm_value : float
        Mechanical power value (pu)
    gen_idx : int
        Generator index
    tolerance : float
        Tolerance for power balance check (default: 0.01 pu = 1%)

    Returns:
    --------
    is_valid : bool
        True if equilibrium is valid
    diagnostics : dict
        Dictionary with validation details
    """
    diagnostics = {
        "Pm": pm_value,
        "power_flow_converged": False,
        "power_balance_error": None,
        "Pe_initial": None,
        "delta0_deg": None,
        "error": None,
    }

    # Check power flow convergence
    if not (hasattr(ss.PFlow, "converged") and ss.PFlow.converged):
        diagnostics["error"] = "power_flow_not_converged"
        return False, diagnostics

    diagnostics["power_flow_converged"] = True

    # Extract pre-fault conditions
    try:
        from .data_extractor import extract_prefault_conditions

        conditions = extract_prefault_conditions(ss, gen_idx)
        Pe_initial = conditions.get("Pe0", pm_value)
        delta0 = conditions.get("delta0", 0.0)
    except Exception as e:
        # Fallback extraction
        Pe_initial = pm_value  # Assume equal if extraction fails
        delta0 = 0.0
        if hasattr(ss.GENCLS, "delta") and hasattr(ss.GENCLS.delta, "v"):
            delta0 = float(safe_get_array_value(ss.GENCLS.delta.v, gen_idx, default=0.0))
        if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
            Pe_initial = float(safe_get_array_value(ss.GENCLS.tm0.v, gen_idx, default=pm_value))

    diagnostics["Pe_initial"] = Pe_initial
    diagnostics["delta0_deg"] = np.degrees(delta0) if delta0 is not None else 0.0

    # Verify power balance: Pm ≈ Pe at t=0
    power_balance_error = abs(pm_value - Pe_initial)
    diagnostics["power_balance_error"] = power_balance_error

    # For P_m variation mode, we may need slightly higher tolerance
    # because power flow solution might have small numerical differences
    effective_tolerance = tolerance
    if power_balance_error > tolerance and power_balance_error < tolerance * 2:
        # If error is small but above tolerance, log it but be more lenient
        # This handles numerical precision issues in power flow
        pass

    if power_balance_error > tolerance:
        diagnostics["error"] = "power_balance_violated"
        return False, diagnostics

    # Check initial rotor angle (should be < 90° for stability, typically < 45°)
    if delta0 > np.radians(85):  # Near stability limit (90°)
        diagnostics["error"] = "high_initial_angle"
        return False, diagnostics

    # All checks passed
    return True, diagnostics
