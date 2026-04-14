#!/usr/bin/env python3
"""
Data extractor module.

Functions for extracting trajectories and system parameters from ANDES simulations.
"""

import warnings
from typing import Dict

import numpy as np

from data_generation.andes_extractor import extract_system_reactances, extract_trajectories

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


def extract_prefault_conditions(ss, gen_idx: int = 0) -> Dict[str, float]:
    """
    Extract pre-fault steady-state conditions from power flow solution.

    Parameters:
    -----------
    ss : andes.System
        ANDES system object (after power flow)
    gen_idx : int
        Generator index (default: 0)

    Returns:
    --------
    dict : Dictionary containing:
        - delta0: Initial rotor angle (rad)
        - omega0: Initial speed (pu, should be 1.0)
        - V0: Initial voltage (pu)
        - Pe0: Initial electrical power (pu)
    """
    conditions = {}

    # Check if GENCLS exists
    if not hasattr(ss, "GENCLS"):
        # Return default values if GENCLS doesn't exist
        return {"delta0": 0.0, "omega0": 1.0, "V0": 1.0, "Pe0": 0.8}

    # Get rotor angle (delta0)
    if hasattr(ss.GENCLS, "delta") and hasattr(ss.GENCLS.delta, "v"):
        delta0 = safe_get_array_value(ss.GENCLS.delta.v, gen_idx, default=0.0)
        conditions["delta0"] = float(delta0) if delta0 is not None else 0.0
    else:
        conditions["delta0"] = 0.0

    # Get speed (omega0)
    if hasattr(ss.GENCLS, "omega") and hasattr(ss.GENCLS.omega, "v"):
        omega0 = safe_get_array_value(ss.GENCLS.omega.v, gen_idx, default=1.0)
        conditions["omega0"] = float(omega0) if omega0 is not None else 1.0
    else:
        conditions["omega0"] = 1.0  # Assume synchronous speed

    # Get voltage (V0) from generator bus
    V0 = None
    if hasattr(ss.GENCLS, "bus") and hasattr(ss.GENCLS.bus, "v") and hasattr(ss, "Bus"):
        gen_bus = safe_get_array_value(ss.GENCLS.bus.v, gen_idx)
        if gen_bus is not None:
            bus_indices = ss.Bus.idx.v if hasattr(ss.Bus.idx, "v") else []
            if hasattr(bus_indices, "__iter__"):
                try:
                    bus_idx = list(bus_indices).index(gen_bus)
                    if hasattr(ss.Bus, "v") and hasattr(ss.Bus.v, "v"):
                        V0 = safe_get_array_value(ss.Bus.v.v, bus_idx)
                except (ValueError, IndexError):
                    pass

    conditions["V0"] = float(V0) if V0 is not None else 1.0

    # Get electrical power (Pe0) - try multiple methods in order of reliability
    Pe0 = None

    # Method 1: From Bus.P injection (most reliable for power flow results)
    if (
        Pe0 is None
        and hasattr(ss.GENCLS, "bus")
        and hasattr(ss.GENCLS.bus, "v")
        and hasattr(ss, "Bus")
    ):
        gen_bus = safe_get_array_value(ss.GENCLS.bus.v, gen_idx)
        if gen_bus is not None:
            bus_indices = ss.Bus.idx.v if hasattr(ss.Bus.idx, "v") else []
            if hasattr(bus_indices, "__iter__"):
                try:
                    bus_idx = list(bus_indices).index(gen_bus)
                    if hasattr(ss.Bus, "P") and hasattr(ss.Bus.P, "v"):
                        P_val = safe_get_array_value(ss.Bus.P.v, bus_idx)
                        if P_val is not None and abs(P_val) > 1e-6:
                            Pe0 = float(P_val)
                except (ValueError, IndexError):
                    pass

    # Method 2: From GENCLS.a (electrical power variable, Pe = -a)
    if Pe0 is None and hasattr(ss.GENCLS, "a") and hasattr(ss.GENCLS.a, "v"):
        a_val = safe_get_array_value(ss.GENCLS.a.v, gen_idx)
        if a_val is not None:
            Pe0 = -float(a_val)

    # Method 3: From PV.p0 (setpoint, should match Pe in steady state)
    if Pe0 is None and hasattr(ss, "PV") and hasattr(ss.PV, "p0") and hasattr(ss.PV.p0, "v"):
        if len(ss.PV.p0.v) > 0:
            pv_val = safe_get_array_value(ss.PV.p0.v, 0)
            if pv_val is not None:
                Pe0 = float(pv_val)

    # Method 4: Fallback to tm0 (mechanical power, should equal Pe in steady state)
    if Pe0 is None and hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
        Pe0 = safe_get_array_value(ss.GENCLS.tm0.v, gen_idx)

    conditions["Pe0"] = float(Pe0) if Pe0 is not None else conditions.get("Pm", 0.8)

    return conditions


def extract_network_reactances(ss, fault_bus: int) -> Dict[str, float]:
    """
    Extract network reactances: Xprefault, Xfault, Xpostfault.

    Parameters:
    -----------
    ss : andes.System
        ANDES system object
    fault_bus : int
        Bus where fault occurs

    Returns:
    --------
    dict : Dictionary containing:
        - Xprefault: Pre-fault equivalent reactance (pu)
        - Xfault: Fault reactance (pu)
        - Xpostfault: Post-fault equivalent reactance (pu)
        - tf: Fault start time (s)
        - tc: Fault clearing time (s)
    """
    # Use existing function if available.
    try:
        return extract_system_reactances(ss)
    except Exception:
        pass

    # Fallback implementation
    reactances = {}

    # Get fault information
    if hasattr(ss, "Fault") and ss.Fault.n > 0:
        reactances["tf"] = float(ss.Fault.tf.v[0]) if hasattr(ss.Fault, "tf") else 1.0
        reactances["tc"] = float(ss.Fault.tc.v[0]) if hasattr(ss.Fault, "tc") else 1.1
        reactances["Xfault"] = float(ss.Fault.xf.v[0]) if hasattr(ss.Fault, "xf") else 0.0001
    else:
        reactances["tf"] = 1.0
        reactances["tc"] = 1.1
        reactances["Xfault"] = 0.0001

    # Calculate Xprefault for SMIB: Xd' + Xtransformer + Xline
    # 1. Generator transient reactance (Xd')
    Xd_prime = None
    if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
        if hasattr(ss.GENCLS, "xd1"):
            Xd_prime = float(ss.GENCLS.xd1.v[0]) if hasattr(ss.GENCLS.xd1, "v") else None
        elif hasattr(ss.GENCLS, "xd"):
            Xd_prime = float(ss.GENCLS.xd.v[0]) if hasattr(ss.GENCLS.xd, "v") else None

    if Xd_prime is None or Xd_prime <= 0:
        Xd_prime = 0.25  # Typical Xd' for classical generator model

    # 2. Transformer reactance (if any)
    Xtransformer = 0.0
    if hasattr(ss, "Transformer") and ss.Transformer.n > 0:
        trans_data = ss.Transformer.as_df()
        if "x" in trans_data.columns:
            trans_reactances = trans_data["x"].values
            Xtransformer = float(np.sum(trans_reactances[trans_reactances > 0]))

    # 3. Transmission line reactances
    Xline = 0.0
    if hasattr(ss, "Line") and ss.Line.n > 0:
        line_data = ss.Line.as_df()
        line_reactances = line_data["x"].values
        if np.all(line_reactances > 0):
            if len(line_reactances) > 1:
                # Parallel lines: 1/Xeq = sum(1/Xi)
                Xline = 1.0 / np.sum(1.0 / line_reactances)
            else:
                Xline = float(line_reactances[0])
        else:
            Xline = (
                float(np.min(line_reactances[line_reactances > 0]))
                if np.any(line_reactances > 0)
                else 0.0
            )
    else:
        Xline = 0.1  # Default

    # Total pre-fault reactance
    reactances["Xprefault"] = float(Xd_prime + Xtransformer + Xline)

    # Xpostfault: Check if lines are tripped
    reactances["Xpostfault"] = reactances["Xprefault"]  # Default: same as pre-fault

    if hasattr(ss, "Toggle") and ss.Toggle.n > 0:
        # If toggles configured, may indicate line tripping
        # For SMIB: If one of two parallel lines trips, Xpostfault = 2 × Xprefault
        # This is simplified - in practice, need to recalculate network topology
        toggle_data = ss.Toggle.as_df()
        # Check if any toggles affect lines
        if "model" in toggle_data.columns and "Line" in toggle_data["model"].values:
            # Assume one line tripped (simplified)
            # For two parallel lines, if one trips: Xpostfault = 2 × Xprefault
            if ss.Line.n == 2:
                reactances["Xpostfault"] = 2.0 * reactances["Xprefault"]

    # Validate: Xfault < Xprefault
    if reactances["Xfault"] >= reactances["Xprefault"]:
        warnings.warn(
            "Xfault ({reactances['Xfault']:.6f}) >= Xprefault ({reactances['Xprefault']:.6f}). "
            "This may indicate an issue."
        )

    return reactances


def extract_trajectories_with_derived(ss, gen_idx: int = 0) -> Dict[str, np.ndarray]:
    """
    Extract trajectories and add basic derived features.

    Parameters:
    -----------
    ss : andes.System
        ANDES system object (after TDS)
    gen_idx : int
        Generator index (default: 0)

    Returns:
    --------
    dict : Dictionary containing:
        - time: Time array
        - delta: Rotor angle (rad)
        - omega: Rotor speed (pu)
        - voltage: Voltage array
        - Pe: Electrical power array
        - delta_dev: Delta deviation from initial (delta - delta0)
        - omega_dev: Omega deviation from 1.0 (omega - 1.0)
        - normalized_time: Normalized time (for fault period)
    """
    # Use existing function if available.
    try:
        trajectories = extract_trajectories(ss, gen_idx=str(gen_idx))
    except Exception:
        # Fallback extraction
        trajectories = {}

        # Get time vector
        if hasattr(ss, "TDS") and hasattr(ss.TDS, "plt") and ss.TDS.plt is not None:
            if hasattr(ss.TDS.plt, "t"):
                trajectories["time"] = np.array(ss.TDS.plt.t)
            else:
                trajectories["time"] = np.linspace(0, ss.TDS.config.tf, 1000)
        elif hasattr(ss, "dae") and hasattr(ss.dae, "t"):
            trajectories["time"] = (
                np.array(ss.dae.t) if hasattr(ss.dae.t, "__len__") else np.array([ss.dae.t])
            )
        else:
            trajectories["time"] = np.linspace(0, 5.0, 1000)

        # Get delta
        if hasattr(ss.GENCLS, "delta") and hasattr(ss.GENCLS.delta, "v"):
            delta_data = ss.GENCLS.delta.v
            if hasattr(delta_data, "ndim") and delta_data.ndim == 2:
                trajectories["delta"] = delta_data[:, gen_idx]
            else:
                trajectories["delta"] = np.array(delta_data).flatten()
        else:
            trajectories["delta"] = np.zeros_like(trajectories["time"])

        # Get omega
        if hasattr(ss.GENCLS, "omega") and hasattr(ss.GENCLS.omega, "v"):
            omega_data = ss.GENCLS.omega.v
            if hasattr(omega_data, "ndim") and omega_data.ndim == 2:
                trajectories["omega"] = omega_data[:, gen_idx]
            else:
                trajectories["omega"] = np.array(omega_data).flatten()
        else:
            trajectories["omega"] = np.ones_like(trajectories["time"])

        # Get voltage (from generator bus)
        if hasattr(ss, "Bus") and hasattr(ss.Bus, "v") and hasattr(ss.Bus.v, "v"):
            if hasattr(ss.GENCLS, "bus") and hasattr(ss.GENCLS.bus, "v"):
                gen_bus = safe_get_array_value(ss.GENCLS.bus.v, gen_idx)
                bus_indices = ss.Bus.idx.v if hasattr(ss.Bus.idx, "v") else []
                if hasattr(bus_indices, "__iter__"):
                    try:
                        bus_idx = list(bus_indices).index(gen_bus)
                        v_data = ss.Bus.v.v
                        if hasattr(v_data, "ndim") and v_data.ndim == 2:
                            trajectories["voltage"] = v_data[:, bus_idx]
                        else:
                            trajectories["voltage"] = np.array(v_data).flatten()
                    except (ValueError, IndexError):
                        trajectories["voltage"] = np.ones_like(trajectories["time"])
                else:
                    trajectories["voltage"] = np.ones_like(trajectories["time"])
            else:
                trajectories["voltage"] = np.ones_like(trajectories["time"])
        else:
            trajectories["voltage"] = np.ones_like(trajectories["time"])

        # Get Pe (electrical power) - approximate from power balance
        trajectories["Pe"] = trajectories.get("Pe", np.full_like(trajectories["time"], 0.8))

    # Ensure all arrays have same length
    time = trajectories.get("time", np.array([]))

    # Safely check if time is empty or unsized
    try:
        time_len = len(time) if hasattr(time, "__len__") else 0
        if time_len == 0:
            # If time is scalar or empty, create default time array
            if hasattr(ss, "TDS") and hasattr(ss.TDS, "config"):
                tf = ss.TDS.config.tf if hasattr(ss.TDS.config, "tf") else 5.0
                time = np.linspace(0, tf, 1000)
                trajectories["time"] = time
                time_len = len(time)
            else:
                return trajectories
    except (TypeError, AttributeError):
        # If len() fails, try to convert to array
        try:
            time = np.array([time]) if np.isscalar(time) else np.array(time)
            trajectories["time"] = time
            time_len = len(time) if hasattr(time, "__len__") else 0
            if time_len == 0:
                return trajectories
        except Exception:
            return trajectories

    # Ensure time is a 1D array
    if not isinstance(time, np.ndarray):
        time = np.array(time)
    if time.ndim == 0:
        time = np.array([time])
        trajectories["time"] = time
        time_len = 1

    # Add derived features
    delta = trajectories.get("delta", np.zeros_like(time))
    omega = trajectories.get("omega", np.ones_like(time))

    # Ensure delta and omega are arrays with correct shape
    if not isinstance(delta, np.ndarray):
        delta = np.array(delta)
    if delta.ndim == 0:
        delta = np.array([delta])

    # Safely check delta length
    try:
        delta_len = len(delta) if hasattr(delta, "__len__") else 0
    except (TypeError, AttributeError):
        delta_len = 0

    if delta_len != time_len:
        if delta_len == 0:
            delta = np.zeros_like(time)
        else:
            delta = np.resize(delta, time_len) if time_len > 0 else np.array([])

    if not isinstance(omega, np.ndarray):
        omega = np.array(omega)
    if omega.ndim == 0:
        omega = np.array([omega])

    # Safely check omega length
    try:
        omega_len = len(omega) if hasattr(omega, "__len__") else 0
    except (TypeError, AttributeError):
        omega_len = 0

    if omega_len != time_len:
        if omega_len == 0:
            omega = np.ones_like(time)
        else:
            omega = np.resize(omega, time_len) if time_len > 0 else np.array([])

    # Get initial values for deviations
    try:
        delta_len = len(delta) if hasattr(delta, "__len__") else 0
        delta0 = delta[0] if delta_len > 0 else 0.0
    except (TypeError, IndexError, AttributeError):
        delta0 = float(delta) if np.isscalar(delta) else 0.0
    omega0 = 1.0  # Synchronous speed

    # Delta deviation
    trajectories["delta_dev"] = delta - delta0

    # Omega deviation
    trajectories["omega_dev"] = omega - omega0

    # Normalized time (for fault period) - will be set based on fault timing
    # For now, normalize to [0, 1] over simulation time
    try:
        if time_len > 0 and time[-1] > time[0]:
            trajectories["normalized_time"] = (time - time[0]) / (time[-1] - time[0])
        else:
            trajectories["normalized_time"] = np.zeros_like(time)
    except (IndexError, TypeError):
        trajectories["normalized_time"] = np.zeros_like(time)

    return trajectories
