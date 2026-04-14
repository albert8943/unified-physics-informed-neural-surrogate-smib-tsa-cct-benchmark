"""
Stability Checker Utility for Trajectory Analysis.

This module provides functions to determine system stability from trajectory data.
Used for CCT estimation via binary search.

Multimachine TSA: use Center of Inertia (COI) reference. Stability is judged from
relative rotor angles (δ_i − δ_COI) and relative speeds (ω_i − ω_COI), consistent
with run_kundur_fault_expt and standard transient stability practice.
"""

from typing import List, Optional, Union

import numpy as np
import torch


def _persistence_unstable(
    t: np.ndarray,
    delta_abs: np.ndarray,
    *,
    window_seconds: float,
    violation_threshold_rad: float,
    violation_fraction: float,
) -> bool:
    """
    True if there exists a time interval [t0, t0 + window_seconds] containing
    samples such that at least ``violation_fraction`` of those samples satisfy
    delta_abs >= violation_threshold_rad.
    """
    if window_seconds <= 0:
        raise ValueError("persistence window_seconds must be positive")
    if violation_fraction < 0 or violation_fraction > 1.0:
        raise ValueError("violation_fraction must be in [0, 1]")
    if violation_fraction <= 0:
        # (nv / n) >= 0 holds for every non-empty window — would mark everything unstable.
        return False
    n = int(t.size)
    if n == 0:
        return False
    for i in range(n):
        t0 = float(t[i])
        t1 = t0 + float(window_seconds)
        mask = (t >= t0) & (t <= t1)
        if not np.any(mask):
            continue
        sub = delta_abs[mask]
        nv = int(np.count_nonzero(sub >= float(violation_threshold_rad)))
        if (nv / float(sub.size)) >= float(violation_fraction) - 1e-15:
            return True
    return False


def stability_angle_metric_rad(
    delta: Union[np.ndarray, torch.Tensor],
    *,
    mode: str = "global_max",
    time: Optional[Union[np.ndarray, torch.Tensor]] = None,
    final_window_seconds: float = 0.25,
    persistence_window_seconds: float = 0.25,
    persistence_violation_fraction: float = 0.9,
    violation_threshold_rad: float = np.pi,
) -> float:
    """
    Scalar rotor-angle feature (radians) compared to ``delta_threshold`` in
    :func:`check_stability`. Lower is better; stable if metric < threshold.

    Parameters
    ----------
    delta
        Rotor angle trajectory (rad), 1D.
    mode
        - ``global_max``: max |δ(t)| over the whole series (legacy default).
        - ``terminal``: |δ(t_last)| only (ignores mid-horizon spikes).
        - ``final_window``: max |δ(t)| over samples with
          t >= t_end - ``final_window_seconds`` (robust to terminal sampling phase).
        - ``persistence_fraction``: **unstable** if some interval of width
          ``persistence_window_seconds`` contains at least a fraction
          ``persistence_violation_fraction`` of samples with
          |δ| >= ``violation_threshold_rad`` (same rad limit as ``delta_threshold``
          in :func:`check_stability`). Returns ``0.0`` if stable, ``inf`` if not,
          so ``check_stability`` still uses ``metric < delta_threshold``.
        Aliases with hyphens (e.g. ``final-window``) are accepted.
    time
        Time axis (s), same length as ``delta``. Required for ``final_window``
        when ``final_window_seconds`` > 0; required for ``persistence_fraction``.
        Should be non-decreasing (sorted ascending).
    final_window_seconds
        Width of the trailing time window (s) for ``final_window``. If <= 0,
        falls back to terminal sample only.
    persistence_window_seconds
        Length of the sliding time window (s) for ``persistence_fraction``.
    persistence_violation_fraction
        Fraction in ``[0, 1]`` of samples inside a window that must satisfy
        |δ| >= ``violation_threshold_rad`` to count as a persistence hit.
    violation_threshold_rad
        Angle magnitude threshold (rad) for counting violations in
        ``persistence_fraction``. Typically set to ``delta_threshold`` from
        :func:`check_stability`.
    """
    if isinstance(delta, torch.Tensor):
        delta = delta.detach().cpu().numpy()
    delta = np.asarray(delta, dtype=np.float64).flatten()
    if delta.size == 0:
        return float("inf")

    mode_norm = mode.lower().replace("-", "_")
    if mode_norm == "global_max":
        return float(np.abs(delta).max())
    if mode_norm == "terminal":
        return float(np.abs(delta[-1]))

    if mode_norm == "final_window":
        if final_window_seconds <= 0:
            return float(np.abs(delta[-1]))
        if time is None:
            raise ValueError(
                "stability mode 'final_window' requires ``time`` when final_window_seconds > 0"
            )
        if isinstance(time, torch.Tensor):
            time = time.detach().cpu().numpy()
        t = np.asarray(time, dtype=np.float64).flatten()
        if t.shape != delta.shape:
            raise ValueError(f"time shape {t.shape} must match delta shape {delta.shape}")
        t_end = float(t[-1])
        mask = t >= t_end - float(final_window_seconds)
        if not np.any(mask):
            return float(np.abs(delta[-1]))
        return float(np.abs(delta[mask]).max())

    if mode_norm == "persistence_fraction":
        if float(persistence_window_seconds) <= 0:
            raise ValueError("persistence_window_seconds must be positive for persistence_fraction")
        if time is None:
            raise ValueError("stability mode 'persistence_fraction' requires ``time``")
        if isinstance(time, torch.Tensor):
            time = time.detach().cpu().numpy()
        t = np.asarray(time, dtype=np.float64).flatten()
        if t.shape != delta.shape:
            raise ValueError(f"time shape {t.shape} must match delta shape {delta.shape}")
        order = np.argsort(t, kind="mergesort")
        t = t[order]
        d_abs = np.abs(delta[order])
        unstable = _persistence_unstable(
            t,
            d_abs,
            window_seconds=float(persistence_window_seconds),
            violation_threshold_rad=float(violation_threshold_rad),
            violation_fraction=float(persistence_violation_fraction),
        )
        return float("inf") if unstable else 0.0

    raise ValueError(
        f"Unknown stability mode {mode!r}; use 'global_max', 'terminal', "
        f"'final_window', or 'persistence_fraction'"
    )


def check_stability(
    delta: Union[np.ndarray, torch.Tensor],
    omega: Union[np.ndarray, torch.Tensor],
    delta_threshold: float = np.pi,
    omega_threshold: float = 1.5,
    *,
    stability_mode: str = "global_max",
    time: Optional[Union[np.ndarray, torch.Tensor]] = None,
    final_window_seconds: float = 0.25,
    persistence_window_seconds: float = 0.25,
    persistence_violation_fraction: float = 0.9,
) -> bool:
    """
    Check if a trajectory is stable based on rotor angle and speed.

    A trajectory is considered stable if the chosen angle metric is strictly below
    ``delta_threshold`` (radians). For ``persistence_fraction``, the metric is
    ``0.0`` (stable) or ``inf`` (unstable), so this comparison still applies.
    Speed is not used (see module docstring).

    Parameters:
    -----------
    delta : np.ndarray or torch.Tensor
        Rotor angle trajectory (radians)
    omega : np.ndarray or torch.Tensor
        Rotor speed trajectory (per unit)
    delta_threshold : float
        Maximum allowed absolute rotor angle for stability (radians)
        Default: π (180 degrees)
    omega_threshold : float
        Reserved for future speed-based checks; **currently ignored** (stability
        uses rotor angle only).
    stability_mode : str
        ``global_max`` | ``terminal`` | ``final_window`` | ``persistence_fraction`` —
        passed to :func:`stability_angle_metric_rad`.
    time : array, optional
        Time (s), same length as ``delta``; required for ``final_window`` when
        ``final_window_seconds`` > 0, and for ``persistence_fraction``.
    final_window_seconds : float
        Trailing window width for ``final_window`` mode.
    persistence_window_seconds : float
        Sliding window length (s) for ``persistence_fraction``.
    persistence_violation_fraction : float
        Required fraction of samples in a window with |δ| ≥ ``delta_threshold``
        for ``persistence_fraction``.

    Returns:
    --------
    bool
        True if trajectory is stable, False otherwise
    """
    # Convert to numpy if torch tensors (omega kept for API compatibility).
    if isinstance(omega, torch.Tensor):
        _ = omega.detach().cpu().numpy()

    metric = stability_angle_metric_rad(
        delta,
        mode=stability_mode,
        time=time,
        final_window_seconds=final_window_seconds,
        persistence_window_seconds=persistence_window_seconds,
        persistence_violation_fraction=persistence_violation_fraction,
        violation_threshold_rad=float(delta_threshold),
    )
    return bool(metric < float(delta_threshold))


def check_stability_batch(
    delta_batch: Union[np.ndarray, torch.Tensor],
    omega_batch: Union[np.ndarray, torch.Tensor],
    delta_threshold: float = np.pi,
    omega_threshold: float = 1.5,
    *,
    stability_mode: str = "global_max",
    time: Optional[Union[np.ndarray, torch.Tensor]] = None,
    final_window_seconds: float = 0.25,
    persistence_window_seconds: float = 0.25,
    persistence_violation_fraction: float = 0.9,
) -> np.ndarray:
    """
    Check stability for a batch of trajectories.

    Parameters:
    -----------
    delta_batch : np.ndarray or torch.Tensor
        Batch of rotor angle trajectories, shape (batch_size, time_steps)
    omega_batch : np.ndarray or torch.Tensor
        Batch of rotor speed trajectories, shape (batch_size, time_steps)
    delta_threshold : float
        Maximum allowed absolute rotor angle for stability (radians)
    omega_threshold : float
        Maximum allowed absolute speed deviation from 1.0 pu for stability

    Returns:
    --------
    np.ndarray
        Boolean array of shape (batch_size,) indicating stability of each trajectory
    """
    # Convert to numpy if torch tensors.
    if isinstance(delta_batch, torch.Tensor):
        delta_batch = delta_batch.detach().cpu().numpy()
    if isinstance(omega_batch, torch.Tensor):
        omega_batch = omega_batch.detach().cpu().numpy()

    # Ensure 2D arrays
    delta_batch = np.asarray(delta_batch)
    omega_batch = np.asarray(omega_batch)

    if delta_batch.ndim == 1:
        delta_batch = delta_batch.reshape(1, -1)
    if omega_batch.ndim == 1:
        omega_batch = omega_batch.reshape(1, -1)

    n_batch = delta_batch.shape[0]
    metrics = np.empty(n_batch, dtype=np.float64)
    for i in range(n_batch):
        t_i = time
        if isinstance(time, np.ndarray) and time.ndim == 2:
            t_i = time[i]
        elif isinstance(time, torch.Tensor) and time.ndim == 2:
            t_i = time[i].detach().cpu().numpy()
        metrics[i] = stability_angle_metric_rad(
            delta_batch[i],
            mode=stability_mode,
            time=t_i,
            final_window_seconds=final_window_seconds,
            persistence_window_seconds=persistence_window_seconds,
            persistence_violation_fraction=persistence_violation_fraction,
            violation_threshold_rad=float(delta_threshold),
        )
    is_stable = metrics < float(delta_threshold)

    return is_stable


def check_stability_multimachine_coi(
    delta_per_gen: Union[List[np.ndarray], np.ndarray],
    omega_per_gen: Union[List[np.ndarray], np.ndarray],
    M_vals: Union[List[float], np.ndarray],
    delta_threshold: float = np.pi,
    omega_threshold: float = 1.5,
    *,
    stability_mode: str = "global_max",
    time: Optional[Union[np.ndarray, torch.Tensor]] = None,
    final_window_seconds: float = 0.25,
    persistence_window_seconds: float = 0.25,
    persistence_violation_fraction: float = 0.9,
) -> bool:
    """
    Check multimachine transient stability using COI (Center of Inertia) reference.

    Standard for multimachine TSA (e.g. run_kundur_fault_expt, Kundur 2-area):
    - δ_COI(t) = Σ(M_i δ_i(t)) / Σ M_i
    - δ_rel_i(t) = δ_i(t) − δ_COI(t)  (relative rotor angle)
    # - ω_COI(t) = Σ(M_i ω_i(t)) / Σ M_i   # Omega commented out: focus on rotor angle only
    # - ω_rel_i(t) = ω_i(t) − ω_COI(t)
    Unstable if any |δ_rel_i| > delta_threshold (default 180°).
    # or any |ω_rel_i| > omega_threshold  # Omega commented out

    Parameters:
    -----------
    delta_per_gen : list of np.ndarray or np.ndarray of shape (n_gen, n_time)
        Rotor angle (rad) per generator
    omega_per_gen : list of np.ndarray or np.ndarray of shape (n_gen, n_time)
        Rotor speed (pu) per generator
    M_vals : list or array of length n_gen
        Inertia constants M_i (e.g. from GENCLS.M.v)
    delta_threshold : float
        Max allowed |δ_rel| for stability (rad). Default π (180°)
    omega_threshold : float
        Max allowed |ω_rel| for stability (pu). Default 1.5

    Returns:
    --------
    bool
        True if stable (all relative angles and speeds within limits), False otherwise
    """
    if isinstance(delta_per_gen, np.ndarray):
        delta_per_gen = [delta_per_gen[i] for i in range(delta_per_gen.shape[0])]
    if isinstance(omega_per_gen, np.ndarray):
        omega_per_gen = [omega_per_gen[i] for i in range(omega_per_gen.shape[0])]
    M_vals = np.asarray(M_vals).flatten()
    n_gen = len(delta_per_gen)
    if n_gen == 0 or len(M_vals) < n_gen:
        return False
    M_vals = M_vals[:n_gen]
    M_sum = M_vals.sum()
    if M_sum <= 0:
        return False
    # Stack to (n_gen, n_time)
    delta = np.array([np.asarray(d).flatten() for d in delta_per_gen])
    omega = np.array([np.asarray(o).flatten() for o in omega_per_gen])
    n_time = delta.shape[1]
    if n_time == 0:
        return False
    # δ_COI(t); omega not used for stability (focus on rotor angle only)
    delta_coi = np.sum(M_vals[:, np.newaxis] * delta, axis=0) / M_sum
    # omega_coi = np.sum(M_vals[:, np.newaxis] * omega, axis=0) / M_sum  # Omega commented out
    # δ_rel_i(t)
    delta_rel = delta - delta_coi
    # Per-time-step worst generator: v(t) = max_i |δ_rel_i(t)|
    v = np.abs(delta_rel).max(axis=0)
    metric = stability_angle_metric_rad(
        v,
        mode=stability_mode,
        time=time,
        final_window_seconds=final_window_seconds,
        persistence_window_seconds=persistence_window_seconds,
        persistence_violation_fraction=persistence_violation_fraction,
        violation_threshold_rad=float(delta_threshold),
    )
    return bool(metric < float(delta_threshold))
