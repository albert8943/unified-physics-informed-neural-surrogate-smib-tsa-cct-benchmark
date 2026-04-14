"""
CCT Binary Search Utility.

This module provides functions to estimate Critical Clearing Time (CCT)
using binary search with a trajectory prediction model.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import torch

from .stability_checker import check_stability


def estimate_cct_binary_search(
    trajectory_model,
    delta0: float,
    omega0: float,
    H: float,
    D: float,
    Pm: float,
    Xprefault: float,
    Xfault: float,
    Xpostfault: float,
    tf: float,
    t_eval: np.ndarray,
    low: float = 0.10,
    high: float = 0.50,
    tolerance: float = 0.01,
    max_iterations: int = 50,
    delta_threshold: float = np.pi,
    omega_threshold: float = 1.5,
    stability_mode: str = "global_max",
    final_window_seconds: float = 0.25,
    persistence_window_seconds: float = 0.25,
    persistence_violation_fraction: float = 0.9,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[float, dict]:
    """
    Estimate Critical Clearing Time (CCT) using binary search with trajectory model.

    Performs binary search to find the maximum fault clearing time (tc) that
    results in a stable system trajectory.

    Parameters:
    -----------
    trajectory_model : TrajectoryPredictionPINN
        Trained trajectory prediction model
    delta0 : float
        Initial rotor angle (rad)
    omega0 : float
        Initial rotor speed (pu)
    H : float
        Inertia constant (seconds)
    D : float
        Damping coefficient (pu)
    Pm : float
        Mechanical power (pu). **REQUIRED** - Must match training data range (typically 0.4-0.9)
    Xprefault : float
        Pre-fault reactance (pu)
    Xfault : float
        Fault reactance (pu)
    Xpostfault : float
        Post-fault reactance (pu)
    tf : float
        Fault start time (s)
    t_eval : np.ndarray
        Time points for trajectory evaluation
    low : float
        Lower bound for binary search (minimum fault clearing time, s)
        Default: 0.10
    high : float
        Upper bound for binary search (maximum fault clearing time, s)
        Default: 0.50
    tolerance : float
        Convergence tolerance for binary search (s)
        Default: 0.01
    max_iterations : int
        Maximum number of binary search iterations
        Default: 50
    delta_threshold : float
        Maximum allowed absolute rotor angle for stability (radians)
        Default: π
    omega_threshold : float
        Reserved for speed-based stability checks; ``check_stability`` currently
        classifies using rotor angle only (this argument is ignored there).
        Default: 1.5 pu
    stability_mode : str
        Passed to :func:`utils.stability_checker.check_stability`:
        ``global_max`` | ``terminal`` | ``final_window`` | ``persistence_fraction``.
    final_window_seconds : float
        Trailing time window (s) for ``final_window`` mode.
    persistence_window_seconds : float
        Sliding window (s) for ``persistence_fraction`` mode.
    persistence_violation_fraction : float
        Required fraction of samples with |δ| ≥ ``delta_threshold`` in that window.
    device : str
        Device ('cpu' or 'cuda')
    verbose : bool
        Print progress information

    Returns:
    --------
    tuple : (cct_estimate, info_dict)
        cct_estimate : float
            Estimated Critical Clearing Time (s)
        info_dict : dict
            Dictionary containing:
            - 'iterations': Number of iterations performed
            - 'converged': True if final bracket width <= tolerance
            - 'stable_tc': Maximum stable tc found
            - 'unstable_tc': Minimum unstable tc found
            - 'final_stability': Stability at final estimate
    """
    # Validate inputs.
    if low >= high:
        raise ValueError(f"Lower bound ({low}) must be less than upper bound ({high})")
    if tolerance <= 0:
        raise ValueError(f"Tolerance ({tolerance}) must be positive")
    if low < 0:
        raise ValueError(f"Lower bound ({low}) must be non-negative")

    # Ensure fault clearing time is after fault start time
    if low < tf:
        low = tf
        if verbose:
            warnings.warn(f"Adjusted lower bound to fault start time: {low}")

    # Initialize search bounds
    low_bound = low
    high_bound = high
    stable_tc = None
    unstable_tc = None

    # Binary search
    iteration = 0

    if verbose:
        print(f"Starting binary search for CCT: [{low_bound:.4f}, {high_bound:.4f}]")

    while (high_bound - low_bound) > tolerance and iteration < max_iterations:
        iteration += 1

        # Midpoint
        tc_mid = (low_bound + high_bound) / 2.0

        if verbose:
            print(f"Iteration {iteration}: Testing tc = {tc_mid:.4f} s")

        # Predict trajectory with this clearing time
        delta_pred = None
        omega_pred = None
        try:
            delta_pred, omega_pred = trajectory_model.predict(
                t=t_eval,
                delta0=delta0,
                omega0=omega0,
                H=H,
                D=D,
                Pm=Pm,
                Xprefault=Xprefault,
                Xfault=Xfault,
                Xpostfault=Xpostfault,
                tf=tf,
                tc=tc_mid,
                device=device,
            )
        except Exception as e:
            if verbose:
                print(f"  Error predicting trajectory: {e}")
            # Assume unstable if prediction fails
            is_stable = False
        else:
            # Check stability
            is_stable = check_stability(
                delta=delta_pred,
                omega=omega_pred,
                delta_threshold=delta_threshold,
                omega_threshold=omega_threshold,
                stability_mode=stability_mode,
                time=t_eval,
                final_window_seconds=final_window_seconds,
                persistence_window_seconds=persistence_window_seconds,
                persistence_violation_fraction=persistence_violation_fraction,
            )

        if verbose:
            stability_str = "STABLE" if is_stable else "UNSTABLE"
            if delta_pred is not None and omega_pred is not None:
                max_delta = float(np.abs(np.asarray(delta_pred)).max())
                max_omega_dev = float(np.abs(np.asarray(omega_pred) - 1.0).max())
                print(
                    f"  Stability: {stability_str} | "
                    f"max_delta={max_delta:.4f} rad, "
                    f"max_omega_dev={max_omega_dev:.4f} pu"
                )
            else:
                print(f"  Stability: {stability_str} (no trajectory)")

        # Update bounds
        if is_stable:
            low_bound = tc_mid
            stable_tc = tc_mid
        else:
            high_bound = tc_mid
            unstable_tc = tc_mid

    # Convergence: bracket width within tolerance (do not infer from iteration count alone,
    # or we mark False when tolerance is met on the max_iterations-th step).
    bracket_width = float(high_bound - low_bound)
    converged = bracket_width <= tolerance
    if not converged:
        warnings.warn(
            f"Binary search did not meet tolerance: bracket width {bracket_width:.6g} s > "
            f"{tolerance} after {iteration} iteration(s) (max {max_iterations})."
        )

    # Final estimate (use midpoint or maximum stable)
    if stable_tc is not None:
        if unstable_tc is not None:
            cct_estimate = (stable_tc + unstable_tc) / 2.0
        else:
            cct_estimate = stable_tc
    elif unstable_tc is not None:
        # All tested values were unstable
        cct_estimate = unstable_tc
        warnings.warn("All tested clearing times resulted in unstable trajectories")
    else:
        # Edge case: no stability information
        cct_estimate = (low_bound + high_bound) / 2.0
        warnings.warn("Could not determine stability, using midpoint")

    # Verify final estimate
    try:
        delta_final, omega_final = trajectory_model.predict(
            t=t_eval,
            delta0=delta0,
            omega0=omega0,
            H=H,
            D=D,
            Pm=Pm,
            Xprefault=Xprefault,
            Xfault=Xfault,
            Xpostfault=Xpostfault,
            tf=tf,
            tc=cct_estimate,
            device=device,
        )
        final_stability = check_stability(
            delta=delta_final,
            omega=omega_final,
            delta_threshold=delta_threshold,
            omega_threshold=omega_threshold,
            stability_mode=stability_mode,
            time=t_eval,
            final_window_seconds=final_window_seconds,
            persistence_window_seconds=persistence_window_seconds,
            persistence_violation_fraction=persistence_violation_fraction,
        )
    except Exception:
        final_stability = None

    info_dict = {
        "iterations": iteration,
        "converged": converged,
        "bracket_width": bracket_width,
        "stable_tc": stable_tc,
        "unstable_tc": unstable_tc,
        "final_stability": final_stability,
        "bounds": (low_bound, high_bound),
    }

    if verbose:
        print(f"\nCCT Estimate: {cct_estimate:.4f} s")
        print(f"Converged: {converged}")
        print(f"Stable tc: {stable_tc:.4f} s" if stable_tc is not None else "Stable tc: None")
        print(
            f"Unstable tc: {unstable_tc:.4f} s" if unstable_tc is not None else "Unstable tc: None"
        )

    return cct_estimate, info_dict


def estimate_cct_batch(
    trajectory_model,
    parameters: dict,
    t_eval: np.ndarray,
    low: float = 0.10,
    high: float = 0.50,
    tolerance: float = 0.01,
    device: str = "cpu",
    verbose: bool = False,
    stability_mode: str = "global_max",
    final_window_seconds: float = 0.25,
    persistence_window_seconds: float = 0.25,
    persistence_violation_fraction: float = 0.9,
    delta_threshold: float = np.pi,
    omega_threshold: float = 1.5,
    max_iterations: int = 50,
) -> Tuple[np.ndarray, list]:
    """
    Estimate CCT for a batch of parameter combinations.

    Parameters:
    -----------
    trajectory_model : TrajectoryPredictionPINN
        Trained trajectory prediction model
    parameters : dict
        Dictionary with keys: 'delta0', 'omega0', 'H', 'D', 'Pm', 'Xprefault',
        'Xfault', 'Xpostfault', 'tf'
        Values can be scalars or arrays (for batch processing)
        **Note:** 'Pm' is REQUIRED - must match training data range (typically 0.4-0.9)
    t_eval : np.ndarray
        Time points for trajectory evaluation
    low : float
        Lower bound for binary search
    high : float
        Upper bound for binary search
    tolerance : float
        Convergence tolerance
    device : str
        Device ('cpu' or 'cuda')
    verbose : bool
        Print progress information

    Returns:
    --------
    tuple : (cct_estimates, info_list)
        cct_estimates : np.ndarray
            Array of CCT estimates
        info_list : list
            List of info dictionaries for each estimate
    """
    # Extract parameters.
    delta0 = parameters["delta0"]
    omega0 = parameters["omega0"]
    H = parameters["H"]
    D = parameters["D"]
    Pm = parameters.get("Pm", None)
    if Pm is None:
        raise ValueError(
            "Pm (mechanical power) is REQUIRED in parameters dict. "
            "The model was trained with varying Pm (0.4-0.9 pu). "
            "You must provide the correct Pm value for accurate CCT estimation."
        )
    Xprefault = parameters["Xprefault"]
    Xfault = parameters["Xfault"]
    Xpostfault = parameters["Xpostfault"]
    tf = parameters["tf"]

    # Determine batch size
    param_arrays = [delta0, omega0, H, D, Pm, Xprefault, Xfault, Xpostfault, tf]
    param_arrays = [np.asarray(p) for p in param_arrays]

    # Check if batch processing
    shapes = [p.shape for p in param_arrays if p.ndim > 0]
    if shapes:
        batch_size = shapes[0][0] if shapes[0] else 1
    else:
        batch_size = 1

    # Process each case
    cct_estimates = []
    info_list = []

    for i in range(batch_size):
        if verbose and batch_size > 1:
            print(f"Processing case {i+1}/{batch_size}")

        # Extract parameters for this case
        case_params = {
            "delta0": delta0[i] if hasattr(delta0, "__len__") else delta0,
            "omega0": omega0[i] if hasattr(omega0, "__len__") else omega0,
            "H": H[i] if hasattr(H, "__len__") else H,
            "D": D[i] if hasattr(D, "__len__") else D,
            "Pm": Pm[i] if hasattr(Pm, "__len__") else Pm,
            "Xprefault": Xprefault[i] if hasattr(Xprefault, "__len__") else Xprefault,
            "Xfault": Xfault[i] if hasattr(Xfault, "__len__") else Xfault,
            "Xpostfault": Xpostfault[i] if hasattr(Xpostfault, "__len__") else Xpostfault,
            "tf": tf[i] if hasattr(tf, "__len__") else tf,
        }

        cct, info = estimate_cct_binary_search(
            trajectory_model=trajectory_model,
            delta0=case_params["delta0"],
            omega0=case_params["omega0"],
            H=case_params["H"],
            D=case_params["D"],
            Pm=case_params["Pm"],
            Xprefault=case_params["Xprefault"],
            Xfault=case_params["Xfault"],
            Xpostfault=case_params["Xpostfault"],
            tf=case_params["tf"],
            t_eval=t_eval,
            low=low,
            high=high,
            tolerance=tolerance,
            max_iterations=max_iterations,
            delta_threshold=delta_threshold,
            omega_threshold=omega_threshold,
            stability_mode=stability_mode,
            final_window_seconds=final_window_seconds,
            persistence_window_seconds=persistence_window_seconds,
            persistence_violation_fraction=persistence_violation_fraction,
            device=device,
            verbose=verbose and batch_size == 1,  # Only verbose for single case
        )

        cct_estimates.append(cct)
        info_list.append(info)

    return np.array(cct_estimates), info_list
