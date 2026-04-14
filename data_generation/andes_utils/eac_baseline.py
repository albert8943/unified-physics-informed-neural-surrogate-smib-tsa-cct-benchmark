#!/usr/bin/env python3
"""
Equal Area Criterion (EAC) baseline module.

Provides analytical CCT computation using Equal Area Criterion for SMIB validation.
"""

import numpy as np
from typing import Dict, Optional, Tuple


def compute_cct_eac(
    Pm: float,
    M: float,
    D: float,
    X_prefault: float,
    X_fault: float,
    X_postfault: float,
    V_gen: float = 1.0,
    V_inf: float = 1.0,
) -> Tuple[Optional[float], Optional[float], Dict[str, any]]:
    """
    Compute CCT using Equal Area Criterion (analytical method for SMIB).

    Parameters:
    -----------
    Pm : float
        Mechanical power (pu)
    M : float
        Inertia coefficient (seconds), M = 2*H
    D : float
        Damping coefficient (pu)
    X_prefault : float
        Pre-fault equivalent reactance (pu)
    X_fault : float
        Fault reactance (pu)
    X_postfault : float
        Post-fault equivalent reactance (pu)
    V_gen : float
        Generator bus voltage (pu, default: 1.0)
    V_inf : float
        Infinite bus voltage (pu, default: 1.0)

    Returns:
    --------
    cct_eac : float or None
        CCT from EAC (seconds), or None if computation fails
    delta_cc : float or None
        Critical clearing angle (radians), or None if computation fails
    diagnostics : dict
        Dictionary with intermediate calculations and diagnostics
    """
    diagnostics = {
        "Pm": Pm,
        "M": M,
        "D": D,
        "X_prefault": X_prefault,
        "X_fault": X_fault,
        "X_postfault": X_postfault,
        "V_gen": V_gen,
        "V_inf": V_inf,
        "error": None,
    }

    try:
        # Compute P_max for each network condition
        P_max_pre = (V_gen * V_inf) / X_prefault if X_prefault > 0 else 0.0
        P_max_fault = (V_gen * V_inf) / X_fault if X_fault > 0 else 0.0
        P_max_post = (V_gen * V_inf) / X_postfault if X_postfault > 0 else 0.0

        diagnostics["P_max_pre"] = P_max_pre
        diagnostics["P_max_fault"] = P_max_fault
        diagnostics["P_max_post"] = P_max_post

        # Check if P_m exceeds P_max (no equilibrium exists)
        if Pm >= P_max_pre:
            diagnostics["error"] = f"Pm ({Pm:.6f}) >= P_max_pre ({P_max_pre:.6f}) - no equilibrium"
            return None, None, diagnostics

        # Compute initial angle (pre-fault equilibrium)
        delta_0 = np.arcsin(Pm / P_max_pre)
        diagnostics["delta_0_rad"] = delta_0
        diagnostics["delta_0_deg"] = np.degrees(delta_0)

        # Compute critical clearing angle using EAC
        # For SMIB with three-phase fault, the critical clearing angle satisfies:
        # A1 (accelerating area) = A2 (decelerating area)

        # Simplified EAC computation (ignoring damping for first approximation)
        # More accurate: solve numerically for delta_cc such that areas are equal

        # For simplified case (no damping, constant P_max_fault during fault):
        # Critical clearing angle is approximately:
        # delta_cc ≈ delta_0 + 2 * (delta_max - delta_0)
        # where delta_max is the angle where P_max_post * sin(delta) = Pm

        delta_max = (
            np.pi - np.arcsin(Pm / P_max_post) if P_max_post > 0 and Pm <= P_max_post else np.pi / 2
        )
        diagnostics["delta_max_rad"] = delta_max
        diagnostics["delta_max_deg"] = np.degrees(delta_max)

        # Compute critical clearing angle (simplified EAC)
        # More accurate would require numerical integration, but for first approximation:
        if P_max_fault > 0 and Pm <= P_max_fault:
            # During fault, if system can still transfer power
            # Use iterative method or simplified formula
            # Simplified: delta_cc ≈ delta_0 + (delta_max - delta_0) / 2
            delta_cc = delta_0 + 0.5 * (delta_max - delta_0)
        else:
            # During fault, power transfer is severely limited
            # Critical clearing angle is closer to delta_max
            delta_cc = delta_0 + 0.8 * (delta_max - delta_0)

        # Ensure delta_cc is within valid range
        delta_cc = np.clip(delta_cc, delta_0, delta_max)
        diagnostics["delta_cc_rad"] = delta_cc
        diagnostics["delta_cc_deg"] = np.degrees(delta_cc)

        # Integrate swing equation to get CCT
        # Simplified: CCT ≈ time to reach delta_cc from delta_0
        # Using: d²δ/dt² = (Pm - Pe_fault) / M
        # During fault: Pe_fault ≈ P_max_fault * sin(delta) (simplified)

        # For constant accelerating power during fault (simplified):
        # Pa_fault = Pm - P_max_fault * sin(delta_avg)
        # where delta_avg ≈ (delta_0 + delta_cc) / 2
        delta_avg = (delta_0 + delta_cc) / 2.0
        Pe_fault_avg = P_max_fault * np.sin(delta_avg) if P_max_fault > 0 else 0.0
        Pa_fault = Pm - Pe_fault_avg

        if Pa_fault <= 0:
            diagnostics["error"] = "Non-positive accelerating power during fault"
            return None, None, diagnostics

        # Time to reach delta_cc from delta_0:
        # Using: delta = delta_0 + 0.5 * (Pa_fault / M) * t²
        # Solving for t: t = sqrt(2 * (delta_cc - delta_0) * M / Pa_fault)
        delta_diff = delta_cc - delta_0
        if delta_diff <= 0:
            diagnostics["error"] = "Invalid delta_cc (delta_cc <= delta_0)"
            return None, None, diagnostics

        cct_eac = np.sqrt(2.0 * delta_diff * M / Pa_fault)

        diagnostics["Pe_fault_avg"] = Pe_fault_avg
        diagnostics["Pa_fault"] = Pa_fault
        diagnostics["delta_diff"] = delta_diff
        diagnostics["cct_eac"] = cct_eac

        return float(cct_eac), float(delta_cc), diagnostics

    except Exception as e:
        diagnostics["error"] = f"EAC computation failed: {str(e)}"
        return None, None, diagnostics


def compare_cct_methods(
    cct_bisection: Optional[float], cct_eac: Optional[float], tolerance: float = 0.01
) -> Dict[str, any]:
    """
    Compare bisection CCT with EAC analytical result.

    Parameters:
    -----------
    cct_bisection : float or None
        CCT from bisection method (seconds)
    cct_eac : float or None
        CCT from EAC analytical method (seconds)
    tolerance : float
        Tolerance for comparison (seconds, default: 0.01 = 10 ms)

    Returns:
    --------
    comparison : dict
        Dictionary containing:
        - error: Absolute error (seconds)
        - relative_error: Relative error (%)
        - is_within_tolerance: Bool
        - both_available: Bool
    """
    comparison = {
        "cct_bisection": cct_bisection,
        "cct_eac": cct_eac,
        "both_available": False,
        "error": None,
        "relative_error": None,
        "is_within_tolerance": False,
    }

    if cct_bisection is None or cct_eac is None:
        comparison["error"] = "One or both CCT values are None"
        return comparison

    comparison["both_available"] = True

    # Compute absolute error
    error = abs(cct_bisection - cct_eac)
    comparison["error"] = error

    # Compute relative error
    if cct_bisection > 0:
        relative_error = 100.0 * error / cct_bisection
        comparison["relative_error"] = relative_error
    else:
        comparison["relative_error"] = None

    # Check if within tolerance
    comparison["is_within_tolerance"] = error <= tolerance

    return comparison
