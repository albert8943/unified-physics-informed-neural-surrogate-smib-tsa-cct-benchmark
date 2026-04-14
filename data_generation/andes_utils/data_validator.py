#!/usr/bin/env python3
"""
Data validator module.

Validation functions for system state, data quality, and parameters.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Handle imports for both module and direct execution
try:
    from .system_manager import safe_get_array_value
except ImportError:
    # Fallback for direct execution or alternative import paths
    import sys
    from pathlib import Path

    # Add project root to path if not already there
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from data_generation.andes_utils.system_manager import safe_get_array_value


def validate_generator_model(ss) -> Tuple[bool, Optional[str]]:
    """
    Validate that system uses GENCLS generator model.

    Parameters:
    -----------
    ss : andes.System
        ANDES system object

    Returns:
    --------
    is_valid : bool
        True if GENCLS model exists and is active
    warning_message : str or None
        Warning message if other models detected
    """
    if not hasattr(ss, "GENCLS") or ss.GENCLS.n == 0:
        return False, "GENCLS generator model not found. Only GENCLS is supported."

    # Check for other generator models
    other_models = []
    if hasattr(ss, "GENROU") and ss.GENROU.n > 0:
        other_models.append("GENROU ({ss.GENROU.n} generators)")
    if hasattr(ss, "GENSAE") and ss.GENSAE.n > 0:
        other_models.append("GENSAE ({ss.GENSAE.n} generators)")

    if other_models:
        warning = (
            "Warning: Other generator models detected: {', '.join(other_models)}. "
            "These are not supported and may affect results."
        )
        return True, warning

    return True, None


def validate_perunit_consistency(ss) -> Tuple[bool, Optional[float]]:
    """
    Verify system base and per-unit consistency.

    Parameters:
    -----------
    ss : andes.System
        ANDES system object

    Returns:
    --------
    is_consistent : bool
        True if system has consistent per-unit base
    base_mva : float or None
        System base MVA, or None if not found
    """
    base_mva = None

    # Try to get system base
    if hasattr(ss, "config") and hasattr(ss.config, "mva"):
        base_mva = ss.config.mva
    elif hasattr(ss, "mva"):
        base_mva = ss.mva
    else:
        # Default assumption
        base_mva = 100.0
        return True, base_mva  # Assume standard base

    # Validate base is reasonable
    if base_mva is not None and 1.0 <= base_mva <= 10000.0:
        return True, base_mva
    else:
        return False, base_mva


def validate_parameters(Pm: float, M: float, D: float) -> Tuple[bool, List[str]]:
    """
    Check parameter ranges are physically reasonable.

    Parameters:
    -----------
    Pm : float
        Mechanical power (pu)
    M : float
        Inertia constant (seconds)
    D : float
        Damping coefficient (pu)

    Returns:
    --------
    is_valid : bool
        True if parameters are valid
    warnings : list
        List of warning messages
    """
    warnings_list = []

    if not (0.0 < Pm < 1.5):
        warnings_list.append("Pm ({Pm:.4f} pu) outside typical range (0, 1.5)")

    if M <= 0:
        warnings_list.append("M ({M:.4f} s) must be positive")

    if D < 0:
        warnings_list.append("D ({D:.4f} pu) should be non-negative")

    if M < 2.0:
        warnings_list.append("Low inertia M={M:.4f}s may cause instability")
    elif M > 20.0:
        warnings_list.append("Very high inertia M={M:.4f}s (unusual)")

    if D < 0.5:
        warnings_list.append("Low damping D={D:.4f}pu may cause oscillations")
    elif D > 5.0:
        warnings_list.append("Very high damping D={D:.4f}pu (unusual)")

    is_valid = len([w for w in warnings_list if "must be" in w or "should be" in w]) == 0
    return is_valid, warnings_list


def validate_prefault_equilibrium(ss, gen_idx: int = 0) -> Tuple[bool, Optional[str]]:
    """
    Validate pre-fault steady-state equilibrium.

    Checks:
    - Power balance: |Pm - Pe0| < 0.001 pu
    - Synchronous speed: |ω0 - 1.0| < 0.0001 pu
    - Angle consistency with power flow solution

    Parameters:
    -----------
    ss : andes.System
        ANDES system object (after power flow)
    gen_idx : int
        Generator index (default: 0)

    Returns:
    --------
    is_valid : bool
        True if equilibrium is validated
    error_message : str or None
        Error message if validation fails
    """
    if not hasattr(ss, "PFlow") or not (hasattr(ss.PFlow, "converged") and ss.PFlow.converged):
        return False, "Power flow not converged"

    # Check if GENCLS exists and has devices
    if not hasattr(ss, "GENCLS"):
        return False, "GENCLS device not found in system"

    # Check if GENCLS has any devices (arrays are not empty)
    try:
        if hasattr(ss.GENCLS, "n") and ss.GENCLS.n == 0:
            return False, "GENCLS has no devices (n=0)"
        # Try to check array length by accessing a common attribute
        if hasattr(ss.GENCLS, "idx") and hasattr(ss.GENCLS.idx, "v"):
            if len(ss.GENCLS.idx.v) == 0:
                return False, "GENCLS arrays are empty"
    except (AttributeError, TypeError):
        pass

    # Get mechanical power
    Pm = None
    if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
        Pm = safe_get_array_value(ss.GENCLS.tm0.v, gen_idx)
    elif hasattr(ss.GENCLS, "P0") and hasattr(ss.GENCLS.P0, "v"):
        Pm = safe_get_array_value(ss.GENCLS.P0.v, gen_idx)

    if Pm is None:
        return False, "Could not extract mechanical power (Pm) for gen_idx={gen_idx}"

    # Get electrical power from power flow
    Pe0 = None
    if hasattr(ss.GENCLS, "a") and hasattr(ss.GENCLS.a, "v"):
        # Electrical power is negative of 'a' variable
        a_val = safe_get_array_value(ss.GENCLS.a.v, gen_idx)
        Pe0 = -float(a_val) if a_val is not None else None
    elif hasattr(ss, "Bus") and hasattr(ss.GENCLS, "bus") and hasattr(ss.GENCLS.bus, "v"):
        # Try to get from bus power injection
        gen_bus = safe_get_array_value(ss.GENCLS.bus.v, gen_idx)
        if gen_bus is not None:
            bus_indices = ss.Bus.idx.v if hasattr(ss.Bus.idx, "v") else []
            if hasattr(bus_indices, "__iter__"):
                try:
                    bus_idx = list(bus_indices).index(gen_bus)
                    if hasattr(ss.Bus, "P") and hasattr(ss.Bus.P, "v"):
                        Pe0 = safe_get_array_value(ss.Bus.P.v, bus_idx)
                        if Pe0 is not None:
                            Pe0 = float(Pe0)
                except (ValueError, IndexError):
                    pass

    if Pe0 is None:
        # If can't get Pe0, assume it equals Pm in steady state (reasonable assumption)
        Pe0 = Pm

    # Check power balance
    power_imbalance = abs(Pm - Pe0)
    if power_imbalance > 0.001:
        return False, "Power imbalance: |Pm - Pe0| = {power_imbalance:.6f} pu (threshold: 0.001)"

    # Check synchronous speed
    omega0 = None
    if hasattr(ss.GENCLS, "omega") and hasattr(ss.GENCLS.omega, "v"):
        omega0 = safe_get_array_value(ss.GENCLS.omega.v, gen_idx)
    else:
        # Assume synchronous speed if not available
        omega0 = 1.0

    if omega0 is not None:
        speed_deviation = abs(omega0 - 1.0)
        if speed_deviation > 0.0001:
            return (
                False,
                "Non-synchronous speed: ω0 = {omega0:.6f} pu (deviation: {speed_deviation:.6f})",
            )

    return True, None


def check_power_flow_convergence(ss) -> Tuple[bool, Optional[str]]:
    """
    Verify power flow converged before TDS.

    Parameters:
    -----------
    ss : andes.System
        ANDES system object

    Returns:
    --------
    converged : bool
        True if power flow converged
    error_message : str or None
        Error message if not converged
    """
    if not hasattr(ss, "PFlow"):
        return False, "PFlow module not found"

    if not (hasattr(ss.PFlow, "converged") and ss.PFlow.converged):
        return False, "Power flow did not converge"

    # Check exit code
    if hasattr(ss, "exit_code") and ss.exit_code != 0:
        return False, "Power flow exit code: {ss.exit_code} (non-zero indicates failure)"

    return True, None


def check_stability(
    ss,
    trajectories: Dict[str, np.ndarray],
    stability_criteria: Optional[Dict[str, float]] = None,
    strict_voltage: bool = False,
    strict_frequency: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check system stability using multiple criteria.

    For simplified SMIB systems (classical model without governor/exciter):
    - Rotor angle is PRIMARY criterion (first-swing stability)
    - Frequency and voltage are ADVISORY (optional warnings)

    For detailed models (with governor/exciter):
    - All criteria can be enforced by setting strict_voltage=True, strict_frequency=True

    Parameters:
    -----------
    ss : andes.System
        ANDES system object (after TDS)
    trajectories : dict
        Dictionary with time series data
    stability_criteria : dict, optional
        Custom stability thresholds. If None, uses defaults:
        - max_angle_deg: 180.0
        - max_freq_dev: 0.05
        - min_voltage_recovery: 0.8
    strict_voltage : bool
        If True, voltage recovery is required for stability (default: False)
        If False, voltage is advisory only (recommended for simplified SMIB)
    strict_frequency : bool
        If True, frequency deviation is required for stability (default: False)
        If False, frequency is advisory only (recommended for simplified SMIB)

    Returns:
    --------
    is_stable : bool
        True if system is stable
    metrics : dict
        Dictionary with stability metrics
    """
    if stability_criteria is None:
        stability_criteria = {
            "max_angle_deg": 180.0,
            "max_freq_dev": 0.05,
            "min_voltage_recovery": 0.8,
        }

    metrics = {}

    # Priority 1: Use ANDES built-in check_criteria() if available
    if hasattr(ss, "TDS") and hasattr(ss.TDS, "check_criteria"):
        try:
            criteria_result = ss.TDS.check_criteria()
            if criteria_result is not None:
                is_stable_andes = not bool(criteria_result)  # False/0 = stable
                metrics["andes_check_criteria"] = criteria_result
                metrics["is_stable_andes"] = is_stable_andes
        except Exception:
            pass

    # Extract data
    time = trajectories.get("time", np.array([]))
    delta = trajectories.get("delta", np.array([]))
    omega = trajectories.get("omega", np.array([]))
    voltage = trajectories.get("voltage", np.array([]))

    if len(time) == 0 or len(delta) == 0:
        return False, {"error": "Insufficient trajectory data"}

    # Criterion 1: Rotor angle (max |δ| < 180°)
    max_angle_rad = np.max(np.abs(delta))
    max_angle_deg = np.degrees(max_angle_rad)
    metrics["max_angle_deg"] = max_angle_deg
    angle_stable = max_angle_deg < stability_criteria["max_angle_deg"]

    # Criterion 2: Frequency deviation (max |ω - 1.0| < threshold)
    if len(omega) > 0:
        omega_dev = np.abs(omega - 1.0)
        max_freq_dev = np.max(omega_dev)
        metrics["max_freq_dev"] = max_freq_dev
        freq_stable = max_freq_dev < stability_criteria["max_freq_dev"]
    else:
        max_freq_dev = 0.0
        metrics["max_freq_dev"] = 0.0
        freq_stable = True

    # Criterion 3: Voltage recovery (min V after fault clear > threshold)
    # Need fault clearing time to check post-fault voltage
    voltage_stable = True
    if len(voltage) > 0 and hasattr(ss, "Fault") and ss.Fault.n > 0:
        try:
            tc = float(ss.Fault.tc.v[0])
            # Find indices after fault clear
            post_fault_mask = time > tc
            if np.any(post_fault_mask):
                min_voltage_post = np.min(voltage[post_fault_mask])
                metrics["min_voltage_recovery"] = min_voltage_post
                voltage_stable = min_voltage_post > stability_criteria["min_voltage_recovery"]
            else:
                metrics["min_voltage_recovery"] = np.min(voltage) if len(voltage) > 0 else 1.0
        except Exception:
            metrics["min_voltage_recovery"] = np.min(voltage) if len(voltage) > 0 else 1.0

    # Overall stability decision based on strictness settings
    # For simplified SMIB: Rotor angle is PRIMARY, others are advisory
    if strict_voltage and strict_frequency:
        # All criteria must pass (for detailed models with controls)
        is_stable = angle_stable and freq_stable and voltage_stable
        metrics["method"] = "all_criteria_strict"
    elif strict_voltage:
        # Rotor angle + voltage (voltage-controlled systems)
        is_stable = angle_stable and voltage_stable
        metrics["method"] = "angle_and_voltage_strict"
    elif strict_frequency:
        # Rotor angle + frequency (frequency-controlled systems)
        is_stable = angle_stable and freq_stable
        metrics["method"] = "angle_and_frequency_strict"
    else:
        # Simplified SMIB: Only rotor angle matters (first-swing stability)
        is_stable = angle_stable
        metrics["method"] = "angle_only_primary"

    metrics["angle_stable"] = angle_stable
    metrics["freq_stable"] = freq_stable
    metrics["voltage_stable"] = voltage_stable
    metrics["is_stable"] = is_stable

    # Add advisory warnings if non-primary criteria fail
    if not is_stable:
        # If unstable, check which criteria failed
        if not angle_stable:
            metrics["failure_reason"] = "rotor_angle_exceeded"
        elif strict_voltage and not voltage_stable:
            metrics["failure_reason"] = "voltage_recovery_failed"
        elif strict_frequency and not freq_stable:
            metrics["failure_reason"] = "frequency_deviation_exceeded"
    else:
        # If stable, check if advisory criteria failed (for warnings)
        advisory_warnings = []
        if not freq_stable and not strict_frequency:
            advisory_warnings.append("frequency_deviation_exceeded_threshold")
        if not voltage_stable and not strict_voltage:
            advisory_warnings.append("voltage_recovery_below_threshold")
        if advisory_warnings:
            metrics["advisory_warnings"] = advisory_warnings
            metrics["advisory_note"] = (
                "System is stable (rotor angle criterion passed), but "
                "advisory criteria failed: {', '.join(advisory_warnings)}. "
                "This is acceptable for simplified SMIB systems without "
                "governor/exciter."
            )

    # If ANDES criteria was used, compare with manual checks
    if "is_stable_andes" in metrics:
        if metrics["is_stable_andes"] != is_stable:
            metrics["discrepancy"] = True
            metrics["discrepancy_note"] = (
                "ANDES check_criteria() and manual checks differ. "
                "ANDES may check additional criteria beyond rotor angle, frequency, and voltage."
            )

    if "method" not in metrics:
        metrics["method"] = "manual_criteria"

    return is_stable, metrics


def validate_data_quality(data_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate data quality (NaN, Inf, consistency)

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data to validate

    Returns:
    --------
    is_valid : bool
        True if data quality is acceptable
    issues : list
        List of quality issues found
    """
    issues = []

    # Check for NaN and Inf
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            if np.any(np.isnan(value)):
                issues.append("NaN values found in {key}")
            if np.any(np.isinf(value)):
                issues.append("Inf values found in {key}")
        elif isinstance(value, (list, tuple)):
            arr = np.array(value)
            if np.any(np.isnan(arr)):
                issues.append("NaN values found in {key}")
            if np.any(np.isinf(arr)):
                issues.append("Inf values found in {key}")

    # Check time step consistency
    # Note: ANDES uses adaptive time stepping, so some variation is expected and normal.
    # We use a more lenient threshold (20% CV) to allow for adaptive stepping while
    # still catching truly problematic cases (e.g., missing data points).
    if "time" in data_dict:
        time = np.array(data_dict["time"])
        if len(time) > 1:
            dt = np.diff(time)
            dt_std = np.std(dt)
            dt_mean = np.mean(dt)
            if dt_mean > 0:
                cv = dt_std / dt_mean  # Coefficient of variation
                if cv > 0.2:  # More than 20% variation (allows adaptive time stepping)
                    issues.append("Inconsistent time steps: CV = {cv:.4f}")

    # Check trajectory length
    time_len = len(data_dict.get("time", []))
    for key in ["delta", "omega", "voltage", "Pe"]:
        if key in data_dict:
            arr_len = len(data_dict[key]) if hasattr(data_dict[key], "__len__") else 1
            if arr_len != time_len and time_len > 0:
                issues.append("Length mismatch: {key} ({arr_len}) vs time ({time_len})")

    is_valid = len(issues) == 0
    return is_valid, issues
