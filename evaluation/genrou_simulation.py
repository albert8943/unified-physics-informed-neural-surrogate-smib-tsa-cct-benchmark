"""
Run ANDES transient simulation on the SMIB GENROU case for one scenario.

Used by PINN and ML GENROU validation so both surrogates see identical
GENROU reference trajectories and Pe(t) profiles.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False


def extract_pe_from_genrou(ss_genrou) -> np.ndarray:
    """Extract Pe(t) from GENROU simulation results (plotter or algebraic fallback)."""
    if hasattr(ss_genrou.TDS, "plt") and hasattr(ss_genrou.TDS.plt, "GENROU"):
        pe_t = ss_genrou.TDS.plt.GENROU.Pe[:, 0]
        return pe_t
    if hasattr(ss_genrou, "GENROU"):
        pe_t = ss_genrou.GENROU.Pe.v
        return pe_t
    return np.array([])


def run_genrou_trajectory(
    scenario: Dict[str, Any],
    case_path: str,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load GENROU case, apply scenario parameters, run TDS, return time series.

    Parameters
    ----------
    scenario : dict
        Keys: H, D, Pm, tf, tc, optional scenario_id (ignored here).
    case_path : str
        Resolved path to GENROU JSON (ANDES case file).

    Returns
    -------
    dict or None
        Keys: ``time``, ``delta``, ``omega`` (pu deviation), ``Pe`` — same length.
        None if simulation or extraction failed.
    """
    if not ANDES_AVAILABLE:
        raise ImportError("ANDES is required for GENROU simulation")

    ss_genrou = andes.load(case_path, default_config=True, no_output=True)

    if not hasattr(ss_genrou, "GENROU") or ss_genrou.GENROU.n == 0:
        print("  ⚠️  Warning: No GENROU machine in case; skipping")
        return None

    M = 2 * float(scenario.get("H", 6.0))
    ss_genrou.GENROU.M.v[0] = M
    ss_genrou.GENROU.D.v[0] = float(scenario.get("D", 1.0))
    Pm_value = float(scenario.get("Pm", 0.8))

    if hasattr(ss_genrou, "PV") and ss_genrou.PV.n > 0:
        if hasattr(ss_genrou.PV, "p0") and hasattr(ss_genrou.PV.p0, "v"):
            if len(ss_genrou.PV.p0.v) > 0:
                ss_genrou.PV.p0.v[0] = Pm_value

    if hasattr(ss_genrou.GENROU, "tm0"):
        ss_genrou.GENROU.tm0.v[0] = Pm_value

    if hasattr(ss_genrou, "Fault") and ss_genrou.Fault.n > 0:
        tf = float(scenario.get("tf", 1.0))
        tc = float(scenario.get("tc", 1.2))
        if hasattr(ss_genrou.Fault, "tf"):
            ss_genrou.Fault.tf.v[0] = tf
        elif hasattr(ss_genrou.Fault, "t1"):
            ss_genrou.Fault.t1.v[0] = tf

        if hasattr(ss_genrou.Fault, "tc"):
            ss_genrou.Fault.tc.v[0] = tc
        elif hasattr(ss_genrou.Fault, "t2"):
            ss_genrou.Fault.t2.v[0] = tc

    ss_genrou.PFlow.run()

    ss_genrou.TDS.config.criteria = 0
    ss_genrou.TDS.config.dt = 0.002
    try:
        ss_genrou.TDS.run(tf=5.0)
    except Exception as e:
        print(f"  ⚠️  Warning: TDS failed: {e}")
        return None

    genrou_time = None
    genrou_delta = None
    genrou_omega = None

    if hasattr(ss_genrou, "dae") and hasattr(ss_genrou.dae, "ts") and ss_genrou.dae.ts is not None:
        if hasattr(ss_genrou.dae.ts, "t") and ss_genrou.dae.ts.t is not None:
            genrou_time = np.array(ss_genrou.dae.ts.t)

        if hasattr(ss_genrou.GENROU, "delta") and hasattr(ss_genrou.GENROU.delta, "a"):
            delta_a = ss_genrou.GENROU.delta.a
            if hasattr(ss_genrou.dae.ts, "x") and ss_genrou.dae.ts.x is not None:
                if len(delta_a) > 0:
                    delta_idx = delta_a[0]
                    if delta_idx < ss_genrou.dae.ts.x.shape[1]:
                        genrou_delta = np.array(ss_genrou.dae.ts.x[:, delta_idx])

        if hasattr(ss_genrou.GENROU, "omega") and hasattr(ss_genrou.GENROU.omega, "a"):
            omega_a = ss_genrou.GENROU.omega.a
            if hasattr(ss_genrou.dae.ts, "x") and ss_genrou.dae.ts.x is not None:
                if len(omega_a) > 0:
                    omega_idx = omega_a[0]
                    if omega_idx < ss_genrou.dae.ts.x.shape[1]:
                        genrou_omega = np.array(ss_genrou.dae.ts.x[:, omega_idx]) - 1.0

    if (
        (genrou_delta is None or len(genrou_delta) == 0)
        and hasattr(ss_genrou, "TDS")
        and hasattr(ss_genrou.TDS, "plt")
        and ss_genrou.TDS.plt is not None
    ):
        if genrou_time is None or len(genrou_time) == 0:
            if hasattr(ss_genrou.TDS.plt, "t") and ss_genrou.TDS.plt.t is not None:
                time_plt = ss_genrou.TDS.plt.t
                if time_plt is not None and len(time_plt) > 0:
                    genrou_time = np.array(time_plt)

        if genrou_delta is None or len(genrou_delta) == 0:
            if hasattr(ss_genrou.TDS.plt, "GENROU") and hasattr(ss_genrou.TDS.plt.GENROU, "delta"):
                delta_plt = ss_genrou.TDS.plt.GENROU.delta
                if delta_plt is not None:
                    genrou_delta = np.array(delta_plt)
                    if genrou_delta.ndim > 1 and genrou_delta.shape[1] > 0:
                        genrou_delta = genrou_delta[:, 0]

        if genrou_omega is None or len(genrou_omega) == 0:
            if hasattr(ss_genrou.TDS.plt, "GENROU") and hasattr(ss_genrou.TDS.plt.GENROU, "omega"):
                omega_plt = ss_genrou.TDS.plt.GENROU.omega
                if omega_plt is not None:
                    genrou_omega = np.array(omega_plt) - 1.0
                    if genrou_omega.ndim > 1 and genrou_omega.shape[1] > 0:
                        genrou_omega = genrou_omega[:, 0]

    if genrou_time is None or len(genrou_time) == 0:
        tf = ss_genrou.TDS.config.tf if hasattr(ss_genrou.TDS, "config") else 5.0
        dt = (
            ss_genrou.TDS.config.dt
            if (hasattr(ss_genrou.TDS, "config") and hasattr(ss_genrou.TDS.config, "dt"))
            else 0.002
        )
        if genrou_delta is not None and len(genrou_delta) > 0:
            n_points = len(genrou_delta)
        else:
            n_points = int(tf / dt) + 1
        genrou_time = np.linspace(0, tf, n_points)

    if genrou_delta is None or len(genrou_delta) == 0:
        print(f"  ⚠️  Warning: Could not extract delta trajectory")
        return None

    if genrou_omega is None or len(genrou_omega) == 0:
        print(f"  ⚠️  Warning: Could not extract omega trajectory")
        return None

    pe_t = None
    if hasattr(ss_genrou, "dae") and hasattr(ss_genrou.dae, "ts") and ss_genrou.dae.ts is not None:
        if hasattr(ss_genrou.GENROU, "Pe") and hasattr(ss_genrou.GENROU.Pe, "a"):
            pe_a = ss_genrou.GENROU.Pe.a
            if hasattr(ss_genrou.dae.ts, "y") and ss_genrou.dae.ts.y is not None:
                if len(pe_a) > 0:
                    pe_idx = pe_a[0]
                    if pe_idx < ss_genrou.dae.ts.y.shape[1]:
                        pe_t = np.array(ss_genrou.dae.ts.y[:, pe_idx])

    if pe_t is None or len(pe_t) == 0:
        pe_t = extract_pe_from_genrou(ss_genrou)

    if pe_t is None or len(pe_t) == 0:
        if hasattr(ss_genrou.GENROU, "Pe") and hasattr(ss_genrou.GENROU.Pe, "v"):
            pe_val = (
                ss_genrou.GENROU.Pe.v[0]
                if len(ss_genrou.GENROU.Pe.v) > 0
                else float(scenario.get("Pm", 0.8))
            )
            pe_t = np.full_like(genrou_time, pe_val)
        else:
            pe_t = np.full_like(genrou_time, float(scenario.get("Pm", 0.8)))

    return {
        "time": genrou_time,
        "delta": genrou_delta,
        "omega": genrou_omega,
        "Pe": pe_t,
    }
