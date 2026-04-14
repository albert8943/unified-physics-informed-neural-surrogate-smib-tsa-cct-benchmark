"""
Post-hoc swing-equation residual diagnostics for predicted trajectories.

We use the same SMIB conventions as ``pinn.core.PhysicsInformedLoss`` / training:

- Inertia coefficient ``M = 2 H`` (``H`` in seconds).
- Rotor speed ``omega`` in **per-unit** with 1.0 at synchronous speed.
- Angle–speed identity used in training:
    dδ/dt = ω_syn (ω - 1),   ω_syn = 2π f_n.

Differentiating the swing equation
``M d²δ/dt² + D dδ/dt = P_m - P_e`` and substituting ``dδ/dt = ω_syn(ω-1)`` gives the
equivalent **first-order ω residual** (single time derivative, better conditioned on a
fixed time grid):

    r_f(t) = dω/dt - (1 / (M ω_syn)) · ( P_m - P_e(t) - D ω_syn (ω - 1) ).

This is proportional to the acceleration-form residual
``M d²δ/dt² + D dδ/dt - (P_m - P_e)`` with factor ``1 / (M ω_syn)`` when the angle–speed
relation holds.

``dω/dt`` is computed with ``numpy.gradient`` along the simulation times in the CSV.
``P_e`` is the ANDES trace (``Pe`` column), matching ``pe_direct_7`` training inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

TimeMask = Literal["all", "postfault"]


@dataclass(frozen=True)
class SwingResidualStats:
    mean_abs: float
    std_abs: float
    max_abs: float
    n_points: int


def _derivative_wrt_time(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if y.shape != t.shape:
        raise ValueError(f"y and t must have same shape; got {y.shape} vs {t.shape}")
    if y.size < 2:
        return np.zeros_like(y, dtype=np.float64)
    return np.gradient(y, t, edge_order=1)


def compute_swing_residual_rf(
    t: np.ndarray,
    omega: np.ndarray,
    H: float,
    D: float,
    Pm: float,
    Pe: np.ndarray,
    fn_hz: float = 60.0,
) -> np.ndarray:
    """
    Equivalent swing residual in ω-form (see module docstring).

    Parameters
    ----------
    t, omega, Pe
        Same length; ``t`` strictly increasing after masking/dedup.
    H, D, Pm
        Scenario parameters (``D``, ``Pm``, ``Pe`` in per-unit power; ``H`` in seconds).
    """
    t = np.asarray(t, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    Pe = np.asarray(Pe, dtype=np.float64)
    if Pe.shape != t.shape:
        raise ValueError(f"Pe shape {Pe.shape} does not match t {t.shape}")

    M = 2.0 * float(H)
    omega_syn = 2.0 * np.pi * float(fn_hz)
    domega_dt = _derivative_wrt_time(omega, t)
    rhs = (float(Pm) - Pe - float(D) * omega_syn * (omega - 1.0)) / (M * omega_syn + 1e-12)
    return (domega_dt - rhs).astype(np.float64)


def apply_time_mask(
    t: np.ndarray,
    tf: float,
    mask: TimeMask,
) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    if mask == "all":
        return np.ones_like(t, dtype=bool)
    if mask == "postfault":
        return t >= float(tf) - 1e-9
    raise ValueError(f"Unknown mask: {mask}")


def stats_abs(x: np.ndarray) -> SwingResidualStats:
    ax = np.abs(np.asarray(x, dtype=np.float64))
    ax = ax[np.isfinite(ax)]
    if ax.size == 0:
        return SwingResidualStats(
            mean_abs=float("nan"), std_abs=float("nan"), max_abs=float("nan"), n_points=0
        )
    return SwingResidualStats(
        mean_abs=float(np.mean(ax)),
        std_abs=float(np.std(ax)),
        max_abs=float(np.max(ax)),
        n_points=int(ax.size),
    )


def aggregate_pooled(abs_rf_list: list[np.ndarray]) -> SwingResidualStats:
    if not abs_rf_list:
        return SwingResidualStats(
            mean_abs=float("nan"), std_abs=float("nan"), max_abs=float("nan"), n_points=0
        )
    cat = np.concatenate([np.abs(np.asarray(a, dtype=np.float64)) for a in abs_rf_list])
    cat = cat[np.isfinite(cat)]
    return stats_abs(cat)


def aggregate_mean_of_scenario_means(per_scenario: list[SwingResidualStats]) -> Tuple[float, float]:
    vals = [s.mean_abs for s in per_scenario if s.n_points > 0 and np.isfinite(s.mean_abs)]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))
