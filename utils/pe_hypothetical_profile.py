"""
Build Pe(t) for a hypothetical clearing time tc_hyp from a reference trajectory.

Used for Pe-input (pe_direct_7) CCT binary search: the network was trained on
Pe sequences from simulations with clearing time tc_ref. For each candidate
tc_hyp we map evaluation times into reference time so that pre-fault, fault-on,
and post-fault segments stay aligned (fault duration stretched/compressed, post-
fault shifted).
"""

from __future__ import annotations

import numpy as np


def pe_profile_for_hypothetical_clearing(
    t_eval: np.ndarray,
    t_ref: np.ndarray,
    pe_ref: np.ndarray,
    tf: float,
    tc_ref: float,
    tc_hyp: float,
) -> np.ndarray:
    """
    Pe at times t_eval when the fault clears at tc_hyp instead of tc_ref.

    Mapping (reference time tau for each evaluation time t):
    - t < tf: tau = t (pre-fault)
    - tf <= t < tc_hyp: tau = tf + (t - tf) * (tc_ref - tf) / (tc_hyp - tf)
    - t >= tc_hyp: tau = tc_ref + (t - tc_hyp) (post-fault shifted)

    Parameters
    ----------
    t_eval : array
        Times (s) where the trajectory model is evaluated (e.g. binary search grid).
    t_ref, pe_ref : array
        Reference simulation time and Pe (same length, sorted by t_ref).
    tf : float
        Fault inception time (s).
    tc_ref : float
        Clearing time used in the reference CSV (s).
    tc_hyp : float
        Candidate clearing time for this forward pass (s).

    Returns
    -------
    pe_hyp : array, shape like t_eval
        Electrical power (pu) consistent with the piecewise mapping.
    """
    t_eval = np.asarray(t_eval, dtype=np.float64).reshape(-1)
    t_ref = np.asarray(t_ref, dtype=np.float64).reshape(-1)
    pe_ref = np.asarray(pe_ref, dtype=np.float64).reshape(-1)
    if t_ref.shape != pe_ref.shape:
        raise ValueError("t_ref and pe_ref must have the same shape")
    if len(t_ref) < 2:
        raise ValueError("Reference trajectory must contain at least two samples")

    # Avoid zero fault-on duration in the hypothesis (caller should keep tc_hyp > tf).
    eps = 1e-9
    if tc_hyp <= tf + eps:
        tc_hyp = tf + eps
    dur_fault_ref = max(tc_ref - tf, eps)
    dur_fault_hyp = max(tc_hyp - tf, eps)

    tau = np.empty_like(t_eval, dtype=np.float64)
    pre = t_eval < tf
    fault = (t_eval >= tf) & (t_eval < tc_hyp)
    post = t_eval >= tc_hyp

    tau[pre] = t_eval[pre]
    u = (t_eval[fault] - tf) / dur_fault_hyp
    tau[fault] = tf + u * dur_fault_ref
    tau[post] = tc_ref + (t_eval[post] - tc_hyp)

    t_min, t_max = float(t_ref[0]), float(t_ref[-1])
    tau = np.clip(tau, t_min, t_max)
    out = np.interp(tau, t_ref, pe_ref, left=pe_ref[0], right=pe_ref[-1])
    return np.asarray(out, dtype=np.float64)
