"""
Dispatch Policy Module for Multimachine Systems.

Implements participation-factor-based redispatch strategies:
- Inertia-weighted participation (for GENCLS classical models)
- Iterative redispatch with loss recalculation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False

from .slack_selection import map_gencls_to_staticgen


def calculate_participation_factors(ss, slack_idx: int) -> Dict[int, float]:
    """
    Calculate inertia-weighted participation factors.

    Formula: ρ_i = M_i / Σ M_j (excluding slack)

    This is the ONLY appropriate method for GENCLS classical models because:
    - GENCLS has no governor dynamics → no droop (R) available
    - GENCLS has no P_max limits in model → headroom-based not directly applicable
    - M (inertia coefficient) is directly available in GENCLS

    Parameters:
    -----------
    ss : ANDES system object
        ANDES system (must have GENCLS model)
    slack_idx : int
        GENCLS index of slack generator (excluded from participation)

    Returns:
    --------
    dict
        Dictionary mapping generator index to participation factor {gen_idx: ρ_i}
        Participation factors sum to 1.0 (excluding slack)
    """
    if not hasattr(ss, "GENCLS") or ss.GENCLS.n == 0:
        raise ValueError("No GENCLS generators found in system")

    # Get M values for all generators
    M_values = {}
    M_total = 0.0

    for gen_idx in range(ss.GENCLS.n):
        if gen_idx != slack_idx:
            M_i = float(ss.GENCLS.M.v[gen_idx])  # M = 2*H for 60 Hz systems
            M_values[gen_idx] = M_i
            M_total += M_i

    # Calculate participation factors
    participation_factors = {}
    if M_total > 0:
        for gen_idx, M_i in M_values.items():
            participation_factors[gen_idx] = M_i / M_total
    else:
        # Fallback: equal participation if all M are zero (shouldn't happen)
        n_participating = len(M_values)
        if n_participating > 0:
            for gen_idx in M_values.keys():
                participation_factors[gen_idx] = 1.0 / n_participating

    return participation_factors


def calculate_system_losses(ss) -> float:
    """
    Calculate system losses from power flow solution.

    Formula: P_loss = Σ P_gen - Σ P_load

    Parameters:
    -----------
    ss : ANDES system object
        ANDES system (must have run power flow)

    Returns:
    --------
    float
        Total system losses (pu)
    """
    total_gen = 0.0
    total_load = 0.0

    # Sum generator outputs
    if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
        for gen_idx in range(ss.GENCLS.n):
            # Get actual power output from StaticGen (linked via GENCLS.gen)
            if hasattr(ss.GENCLS, "gen") and ss.GENCLS.gen.v is not None:
                static_gen_idx = int(ss.GENCLS.gen.v[gen_idx])
                if hasattr(ss, "StaticGen") and static_gen_idx < ss.StaticGen.n:
                    if hasattr(ss.StaticGen, "P") and ss.StaticGen.P.v is not None:
                        total_gen += float(ss.StaticGen.P.v[static_gen_idx])

    # Sum load consumption
    if hasattr(ss, "PQ") and ss.PQ.n > 0:
        for load_idx in range(ss.PQ.n):
            if hasattr(ss.PQ, "p0") and ss.PQ.p0.v is not None:
                total_load += float(ss.PQ.p0.v[load_idx])

    return total_gen - total_load


def apply_dispatch_policy(
    ss,
    slack_idx: int,
    base_setpoints: Dict[int, float],
    delta_load: float,
    delta_losses_estimated: float = 0.0,
) -> Dict[int, float]:
    """
    Apply inertia-weighted dispatch policy (Variant A2).

    For non-slack generators: ΔP_i = ρ_i × (ΔP_load + ΔP_loss_estimated)
    Slack absorbs remainder: ΔP_slack = (ΔP_load + ΔP_loss_estimated) - Σ ΔP_i

    Parameters:
    -----------
    ss : ANDES system object
        ANDES system
    slack_idx : int
        GENCLS index of slack generator
    base_setpoints : dict
        Base generator setpoints {gen_idx: P_base}
    delta_load : float
        Change in total load (pu)
    delta_losses_estimated : float
        Estimated change in losses (pu, default: 0.0)

    Returns:
    --------
    dict
        New generator setpoints {gen_idx: P_new}
    """
    # Calculate participation factors
    participation_factors = calculate_participation_factors(ss, slack_idx)

    # Total power change to distribute
    delta_total = delta_load + delta_losses_estimated

    # Calculate new setpoints
    new_setpoints = {}
    total_redispatch = 0.0

    # Apply dispatch to non-slack generators
    for gen_idx, rho_i in participation_factors.items():
        delta_P_i = rho_i * delta_total
        P_base = base_setpoints.get(gen_idx, 0.0)
        new_setpoints[gen_idx] = P_base + delta_P_i
        total_redispatch += delta_P_i

    # Slack absorbs remainder
    P_slack_base = base_setpoints.get(slack_idx, 0.0)
    delta_P_slack = delta_total - total_redispatch
    new_setpoints[slack_idx] = P_slack_base + delta_P_slack

    return new_setpoints


def apply_dispatch_policy_iterative(
    ss,
    slack_idx: int,
    alpha_base: float,
    alpha_new: float,
    max_iterations: int = 5,
    tolerance: float = 0.01,
    loss_estimation_factor: float = 0.03,
    verbose: bool = False,
) -> Tuple[bool, int, Dict[str, float]]:
    """
    Apply iterative dispatch policy with loss recalculation.

    Iterative process:
    1. Estimate losses (first iteration: loss_estimation_factor × ΔP_load)
    2. Apply dispatch policy
    3. Solve power flow
    4. Calculate actual losses
    5. Check slack deviation
    6. If not converged, redistribute residual and iterate

    Parameters:
    -----------
    ss : ANDES system object
        ANDES system (must be loaded and have base loads)
    slack_idx : int
        GENCLS index of slack generator
    alpha_base : float
        Base load multiplier
    alpha_new : float
        New load multiplier
    max_iterations : int
        Maximum iterations (default: 5)
    tolerance : float
        Slack deviation tolerance (pu, default: 0.01)
    loss_estimation_factor : float
        Loss estimation factor for first iteration (default: 0.03 = 3%)
    verbose : bool
        Whether to print progress (default: False)

    Returns:
    --------
    tuple : (converged, n_iterations, stats)
        converged : bool
            True if converged within tolerance
        n_iterations : int
            Number of iterations performed
        stats : dict
            Statistics: {'slack_deviation', 'power_balance', 'losses', 'participation_factors'}
    """
    # Get base load
    base_load = 0.0
    if hasattr(ss, "PQ") and ss.PQ.n > 0:
        for load_idx in range(ss.PQ.n):
            if hasattr(ss.PQ, "p0") and ss.PQ.p0.v is not None:
                base_load += float(ss.PQ.p0.v[load_idx])

    delta_load = (alpha_new - alpha_base) * base_load

    # Get base generator setpoints
    base_setpoints = {}
    if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
        for gen_idx in range(ss.GENCLS.n):
            if hasattr(ss.GENCLS, "gen") and ss.GENCLS.gen.v is not None:
                static_gen_idx = int(ss.GENCLS.gen.v[gen_idx])
                if hasattr(ss, "StaticGen") and static_gen_idx < ss.StaticGen.n:
                    if hasattr(ss.StaticGen, "P") and ss.StaticGen.P.v is not None:
                        base_setpoints[gen_idx] = float(ss.StaticGen.P.v[static_gen_idx])

    # First iteration: estimate losses
    delta_losses_estimated = loss_estimation_factor * abs(delta_load)

    participation_factors = calculate_participation_factors(ss, slack_idx)

    for iteration in range(max_iterations):
        if verbose:
            print(
                f"  Iteration {iteration + 1}/{max_iterations}: delta_losses_estimated = {delta_losses_estimated:.6f} pu"
            )

        # Apply dispatch
        new_setpoints = apply_dispatch_policy(
            ss, slack_idx, base_setpoints, delta_load, delta_losses_estimated
        )

        # Update generator setpoints in ANDES
        for gen_idx, P_new in new_setpoints.items():
            if gen_idx != slack_idx:  # Don't set slack (it's automatic)
                static_gen_idx = map_gencls_to_staticgen(ss, gen_idx)
                if static_gen_idx is not None and hasattr(ss, "StaticGen"):
                    ss.StaticGen.alter("P", idx=static_gen_idx, value=P_new)

        # Solve power flow
        try:
            ss.PFlow.run()
            if not (hasattr(ss.PFlow, "converged") and ss.PFlow.converged):
                if verbose:
                    print(f"  Power flow failed to converge at iteration {iteration + 1}")
                return False, iteration + 1, {"error": "power_flow_failed"}
        except Exception as e:
            if verbose:
                print(f"  Power flow error at iteration {iteration + 1}: {e}")
            return False, iteration + 1, {"error": str(e)}

        # Calculate actual losses
        P_loss_actual = calculate_system_losses(ss)

        # Get slack power
        static_gen_slack_idx = map_gencls_to_staticgen(ss, slack_idx)
        if static_gen_slack_idx is not None and hasattr(ss, "StaticGen"):
            P_slack = float(ss.StaticGen.P.v[static_gen_slack_idx])
        else:
            P_slack = new_setpoints.get(slack_idx, 0.0)

        # Calculate slack deviation
        P_slack_target = new_setpoints.get(slack_idx, 0.0)
        slack_deviation = abs(P_slack - P_slack_target)

        if verbose:
            print(
                f"    P_slack = {P_slack:.6f} pu, target = {P_slack_target:.6f} pu, deviation = {slack_deviation:.6f} pu"
            )

        # Check convergence
        if slack_deviation < tolerance:
            # Calculate final statistics
            power_balance = calculate_system_losses(ss)
            stats = {
                "slack_deviation": slack_deviation,
                "power_balance": power_balance,
                "losses": P_loss_actual,
                "participation_factors": participation_factors,
                "n_iterations": iteration + 1,
            }
            if verbose:
                print(f"  ✓ Converged after {iteration + 1} iterations")
            return True, iteration + 1, stats

        # Update loss estimate for next iteration
        # Use actual losses plus residual slack deviation
        residual = P_slack - P_slack_target
        delta_losses_estimated = P_loss_actual + residual

    # Not converged
    stats = {
        "slack_deviation": slack_deviation,
        "power_balance": calculate_system_losses(ss),
        "losses": P_loss_actual,
        "participation_factors": participation_factors,
        "n_iterations": max_iterations,
    }
    if verbose:
        print(
            f"  ✗ Not converged after {max_iterations} iterations (deviation: {slack_deviation:.6f} pu)"
        )
    return False, max_iterations, stats


def enforce_limits_and_renormalize(
    ss, participation_factors: Dict[int, float], slack_idx: int
) -> Dict[int, float]:
    """
    Enforce generator limits and re-normalize participation factors.

    If a generator hits P_max or P_min, exclude it from participation
    and re-normalize remaining factors.

    Parameters:
    -----------
    ss : ANDES system object
        ANDES system
    participation_factors : dict
        Current participation factors {gen_idx: ρ_i}
    slack_idx : int
        GENCLS index of slack generator

    Returns:
    --------
    dict
        Updated participation factors (re-normalized if limits hit)
    """
    # TODO: Implement generator limit checking
    # For now, return original factors
    # This should check P_max/P_min from StaticGen or GENCLS
    # and exclude limited generators, then re-normalize

    return participation_factors
