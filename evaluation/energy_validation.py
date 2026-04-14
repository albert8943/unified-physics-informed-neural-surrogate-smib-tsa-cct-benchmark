"""
Energy-Based Validation.

Validates physics consistency using transient energy function.
"""

import numpy as np
from typing import Dict, Optional, Tuple


def compute_transient_energy(
    delta: np.ndarray,
    omega: np.ndarray,
    time: np.ndarray,
    M: float,
    D: float,
    Pm: float,
    Pe_func: callable,
) -> np.ndarray:
    """
    Compute transient energy function.

    Transient energy: V(δ, ω) = ½Mω² - Pm(δ - δ₀) + ∫Pe(δ)dδ

    Parameters:
    -----------
    delta : np.ndarray
        Rotor angle trajectory
    omega : np.ndarray
        Rotor speed deviation trajectory
    time : np.ndarray
        Time array
    M : float
        Inertia constant (M = 2*H)
    D : float
        Damping coefficient
    Pm : float
        Mechanical power
    Pe_func : callable
        Function Pe(t) that returns electrical power at time t

    Returns:
    --------
    energy : np.ndarray
        Transient energy trajectory
    """
    n = len(delta)
    energy = np.zeros(n)

    delta0 = delta[0]
    omega0 = omega[0]

    for i in range(n):
        t = time[i]
        Pe = Pe_func(t)

        # Kinetic energy: ½Mω²
        kinetic = 0.5 * M * (omega[i] - 1.0) ** 2

        # Potential energy: -Pm(δ - δ₀) + ∫Pe(δ)dδ
        # Approximate integral using trapezoidal rule
        if i == 0:
            potential = -Pm * (delta[i] - delta0)
        else:
            # Approximate ∫Pe(δ)dδ using trapezoidal rule
            integral_term = 0.0
            for j in range(1, i + 1):
                Pe_j = Pe_func(time[j])
                Pe_jm1 = Pe_func(time[j - 1])
                integral_term += 0.5 * (Pe_j + Pe_jm1) * (delta[j] - delta[j - 1])

            potential = -Pm * (delta[i] - delta0) + integral_term

        energy[i] = kinetic + potential

    return energy


def compute_energy_margin(
    energy_trajectory: np.ndarray,
    critical_energy: Optional[float] = None,
) -> float:
    """
    Compute energy margin.

    Energy margin = Critical energy - Current energy

    Parameters:
    -----------
    energy_trajectory : np.ndarray
        Energy trajectory
    critical_energy : float, optional
        Critical energy (if None, uses max energy)

    Returns:
    --------
    margin : float
        Energy margin
    """
    if critical_energy is None:
        critical_energy = np.max(energy_trajectory)

    current_energy = energy_trajectory[-1]
    margin = critical_energy - current_energy

    return margin


def validate_energy_conservation(
    delta_true: np.ndarray,
    omega_true: np.ndarray,
    delta_pred: np.ndarray,
    omega_pred: np.ndarray,
    time: np.ndarray,
    M: float,
    D: float,
    Pm: float,
    Pe_func: callable,
) -> Dict:
    """
    Validate energy conservation for PINN predictions.

    Parameters:
    -----------
    delta_true : np.ndarray
        True delta trajectory
    omega_true : np.ndarray
        True omega trajectory
    delta_pred : np.ndarray
        Predicted delta trajectory
    omega_pred : np.ndarray
        Predicted omega trajectory
    time : np.ndarray
        Time array
    M : float
        Inertia constant
    D : float
        Damping coefficient
    Pm : float
        Mechanical power
    Pe_func : callable
        Function Pe(t)

    Returns:
    --------
    results : dict
        Energy validation results
    """
    # Compute energy trajectories
    energy_true = compute_transient_energy(delta_true, omega_true, time, M, D, Pm, Pe_func)
    energy_pred = compute_transient_energy(delta_pred, omega_pred, time, M, D, Pm, Pe_func)

    # Compute energy margin
    margin_true = compute_energy_margin(energy_true)
    margin_pred = compute_energy_margin(energy_pred)

    # Compute metrics
    energy_rmse = np.sqrt(np.mean((energy_true - energy_pred) ** 2))
    energy_mae = np.mean(np.abs(energy_true - energy_pred))
    margin_error = np.abs(margin_true - margin_pred)
    margin_relative_error = margin_error / (np.abs(margin_true) + 1e-10) * 100

    # Energy correlation
    energy_correlation = np.corrcoef(energy_true, energy_pred)[0, 1]

    results = {
        "energy_rmse": float(energy_rmse),
        "energy_mae": float(energy_mae),
        "energy_correlation": float(energy_correlation),
        "margin_error": float(margin_error),
        "margin_relative_error": float(margin_relative_error),
        "margin_true": float(margin_true),
        "margin_pred": float(margin_pred),
    }

    return results
