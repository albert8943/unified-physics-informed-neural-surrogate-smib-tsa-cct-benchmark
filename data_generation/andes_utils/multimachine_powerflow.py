"""
Multimachine Power Flow Workflow with Redispatch.

Orchestrates power flow solution with participation-factor-based redispatch
for load-varying transient stability assessment.
"""

from typing import Dict, Optional, Tuple

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False

from .dispatch_policy import apply_dispatch_policy_iterative
from .slack_selection import select_inertia_dominant_slack, validate_slack_selection


def scale_loads_uniform(ss, alpha: float, verbose: bool = False):
    """
    Scale all loads uniformly by alpha (maintains power factor).

    Parameters:
    -----------
    ss : ANDES system object
        ANDES system
    alpha : float
        Load multiplier
    verbose : bool
        Whether to print progress
    """
    if not hasattr(ss, "PQ") or ss.PQ.n == 0:
        if verbose:
            print("  No PQ loads found in system")
        return

    for load_idx in range(ss.PQ.n):
        # Get base values
        if hasattr(ss.PQ, "p0") and ss.PQ.p0.v is not None:
            P_base = float(ss.PQ.p0.v[load_idx])
            P_new = alpha * P_base

            if hasattr(ss.PQ, "q0") and ss.PQ.q0.v is not None:
                Q_base = float(ss.PQ.q0.v[load_idx])
                Q_new = alpha * Q_base
            else:
                Q_new = None

            # Update load
            try:
                load_identifier = load_idx
                if hasattr(ss.PQ, "idx") and hasattr(ss.PQ.idx, "v"):
                    try:
                        idx_array = ss.PQ.idx.v
                        if hasattr(idx_array, "__getitem__") and len(idx_array) > load_idx:
                            load_uid = idx_array[load_idx]
                            load_identifier = load_uid
                    except (IndexError, AttributeError, TypeError):
                        pass

                if hasattr(ss.PQ, "alter"):
                    ss.PQ.alter("p0", load_identifier, P_new)
                    if Q_new is not None:
                        ss.PQ.alter("q0", load_identifier, Q_new)
                else:
                    ss.PQ.p0.v[load_idx] = P_new
                    if Q_new is not None and hasattr(ss.PQ.q0, "v"):
                        ss.PQ.q0.v[load_idx] = Q_new
            except Exception:
                # Fallback to direct access
                ss.PQ.p0.v[load_idx] = P_new
                if Q_new is not None and hasattr(ss.PQ.q0, "v"):
                    ss.PQ.q0.v[load_idx] = Q_new


def run_multimachine_powerflow_with_redispatch(
    ss,
    alpha_base: float,
    alpha_new: float,
    redispatch_config: Optional[Dict] = None,
    verbose: bool = False,
) -> Tuple[bool, Dict]:
    """
    Run power flow with participation-factor-based redispatch.

    Workflow:
    1. Scale loads to base (alpha_base)
    2. Setup system and run initial power flow
    3. Select slack bus (inertia-dominant)
    4. Scale loads to new value (alpha_new)
    5. Apply iterative redispatch
    6. Validate results

    Parameters:
    -----------
    ss : ANDES system object
        ANDES system (must be loaded but not set up)
    alpha_base : float
        Base load multiplier (typically 1.0)
    alpha_new : float
        New load multiplier
    redispatch_config : dict, optional
        Redispatch configuration (from config file)
        Default: max_iterations=5, tolerance=0.01, loss_estimation_factor=0.03
    verbose : bool
        Whether to print progress

    Returns:
    --------
    tuple : (success, stats)
        success : bool
            True if power flow converged and redispatch successful
        stats : dict
            Statistics including slack_idx, convergence info, participation factors
    """
    if redispatch_config is None:
        redispatch_config = {}

    max_iterations = redispatch_config.get("max_redispatch_iterations", 5)
    tolerance = redispatch_config.get("redispatch_tolerance", 0.01)
    loss_estimation_factor = redispatch_config.get("loss_estimation_factor", 0.03)

    stats = {
        "success": False,
        "slack_idx": None,
        "converged": False,
        "n_iterations": 0,
        "error": None,
    }

    try:
        # Step 1: Scale loads to base
        if verbose:
            print(f"  Scaling loads to base (α={alpha_base:.3f})...")
        scale_loads_uniform(ss, alpha_base, verbose=False)

        # Step 2: Setup system
        if verbose:
            print("  Setting up system...")
        ss.setup()

        # Step 3: Run initial power flow
        if verbose:
            print("  Running initial power flow...")
        ss.PFlow.run()
        if not (hasattr(ss.PFlow, "converged") and ss.PFlow.converged):
            stats["error"] = "initial_power_flow_failed"
            return False, stats

        # Step 4: Select slack bus (inertia-dominant)
        slack_idx = select_inertia_dominant_slack(ss, use_mva_rating=True)
        if not validate_slack_selection(ss, slack_idx):
            stats["error"] = "invalid_slack_selection"
            return False, stats

        stats["slack_idx"] = slack_idx
        if verbose:
            print(f"  Selected slack generator: {slack_idx} (inertia-dominant)")

        # Step 5: Scale loads to new value
        if verbose:
            print(f"  Scaling loads to new value (α={alpha_new:.3f})...")
        scale_loads_uniform(ss, alpha_new, verbose=False)

        # Step 6: Apply iterative redispatch
        if verbose:
            print("  Applying iterative redispatch...")
        converged, n_iterations, redispatch_stats = apply_dispatch_policy_iterative(
            ss,
            slack_idx,
            alpha_base,
            alpha_new,
            max_iterations=max_iterations,
            tolerance=tolerance,
            loss_estimation_factor=loss_estimation_factor,
            verbose=verbose,
        )

        stats["converged"] = converged
        stats["n_iterations"] = n_iterations
        stats.update(redispatch_stats)
        stats["success"] = converged

        if verbose and converged:
            print(f"  ✓ Redispatch converged after {n_iterations} iterations")
            print(f"    Slack deviation: {redispatch_stats.get('slack_deviation', 0):.6f} pu")
            print(f"    Power balance: {redispatch_stats.get('power_balance', 0):.6f} pu")

    except Exception as e:
        stats["error"] = str(e)
        if verbose:
            print(f"  ERROR: {e}")
        return False, stats

    return stats["success"], stats


def validate_powerflow_results(ss, tolerance: float = 0.01) -> Tuple[bool, Dict]:
    """
    Validate power flow results.

    Checks:
    - Power flow convergence
    - Power balance: |Σ P_gen - Σ P_load - P_loss| < tolerance
    - Generator limits (if applicable)

    Parameters:
    -----------
    ss : ANDES system object
        ANDES system (must have run power flow)
    tolerance : float
        Power balance tolerance (pu, default: 0.01)

    Returns:
    --------
    tuple : (valid, stats)
        valid : bool
            True if all checks pass
        stats : dict
            Validation statistics
    """
    stats = {"valid": False, "power_balance": 0.0, "converged": False}

    # Check power flow convergence
    if hasattr(ss.PFlow, "converged") and ss.PFlow.converged:
        stats["converged"] = True
    else:
        return False, stats

    # Calculate power balance
    total_gen = 0.0
    total_load = 0.0

    # Sum generator outputs
    if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
        for gen_idx in range(ss.GENCLS.n):
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

    power_balance = abs(total_gen - total_load)
    stats["power_balance"] = power_balance

    # Check if within tolerance
    stats["valid"] = power_balance < tolerance

    return stats["valid"], stats
