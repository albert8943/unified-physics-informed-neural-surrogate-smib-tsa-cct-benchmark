"""
Unified ANDES Simulation Core with Uniform Load Scaling.

This module provides a unified function for running ANDES simulations with uniform
load scaling (alpha multiplier) that works for both SMIB and multimachine systems.
"""

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False
    print("Warning: ANDES not available. Some functions may not work.")


def run_andes_simulation_with_uniform_load_scaling(
    case_file: str,
    alpha: float,  # Uniform multiplier for all loads (0.4-1.2 typical)
    load_q_alpha: Optional[
        float
    ] = None,  # Optional: Q scaling (default: same as alpha for uniform scaling)
    H: Optional[float] = None,  # Single value or per-machine
    D: Optional[float] = None,  # Single value or per-machine
    fault_start_time: float = 1.0,
    fault_clearing_time: Optional[float] = None,
    fault_bus: int = 3,
    fault_reactance: float = 0.0001,
    simulation_time: float = 5.0,
    time_step: Optional[float] = None,
    validate: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Unified ANDES simulation with uniform load scaling (alpha multiplier).
    Works for both SMIB and multimachine systems.

    Sequence (per ANDES Manual):
        1. Load system
        2. Get base load values for ALL loads in system
        3. Apply uniform scaling: P_load_i' = alpha × P_load_i_base
        4. Apply uniform scaling: Q_load_i' = (alpha if load_q_alpha is None else load_q_alpha) × Q_load_i_base
           - Default: Q scaled by same alpha as P (maintains power factor, industrial standard)
           - Optional: Independent Q scaling via load_q_alpha (for special studies)
        5. Run power flow
        6. Extract Pm for each generator (for physics loss)
        7. Set H, D AFTER PF, BEFORE TDS.init()
        8. Initialize TDS
        9. Run TDS
        10. Extract trajectories

    Parameters:
    -----------
    case_file : str
        Path to ANDES case file
    alpha : float
        Uniform multiplier for all loads (e.g., 0.7 = 70% of base load)
        - SMIB: Scales single load
        - Multimachine: Scales all loads uniformly
    load_q_alpha : float, optional
        Optional Q scaling multiplier. If None, uses same alpha as P (uniform scaling).
        Use this only for special studies where independent Q scaling is needed.
    H : float, optional
        Inertia constant H (seconds). If None, uses case file value.
        For multimachine: single value applied to all machines, or can be extended to per-machine.
    D : float, optional
        Damping coefficient D (pu). If None, uses case file value.
        For multimachine: single value applied to all machines, or can be extended to per-machine.
    fault_start_time : float
        Fault start time (seconds, default: 1.0)
    fault_clearing_time : float, optional
        Fault clearing time (seconds). If None, uses case file value.
    fault_bus : int
        Fault bus number (default: 3)
    fault_reactance : float
        Fault reactance (pu, default: 0.0001)
    simulation_time : float
        Total simulation time (seconds, default: 5.0)
    time_step : float, optional
        Time step for TDS (seconds). If None, uses ANDES default.
    validate : bool
        Whether to validate power flow convergence and system state (default: True)
    verbose : bool
        Whether to print progress messages (default: True)

    Returns:
    --------
    dict : Dictionary with:
        - 'time': Time array
        - 'delta': Rotor angle trajectory (array or dict for multimachine)
        - 'omega': Rotor speed trajectory (array or dict for multimachine)
        - 'Pe': Electrical power trajectory (array or dict for multimachine)
        - 'alpha': Uniform load multiplier (for model input)
        - 'P_base': Base load value(s) - single for SMIB, dict for multimachine
        - 'P_actual': Actual load value(s) after scaling
        - 'Q_base': Base reactive load value(s)
        - 'Q_actual': Actual reactive load value(s) after scaling
        - 'P_total': Total system load (sum of all loads) - for reference
        - 'Pm': Mechanical power (single value or dict for multimachine, for physics loss)
        - 'H': Inertia constant H (single value or dict for multimachine)
        - 'D': Damping coefficient D (single value or dict for multimachine)
        - 'metadata': Additional metadata (convergence status, etc.)
    """
    if not ANDES_AVAILABLE:
        raise RuntimeError("ANDES is not available. Cannot run simulation.")

    # Load system with setup=False to allow parameter modification
    if verbose:
        print(f"[LOAD] Loading system from {case_file}...")
    ss = andes.load(case_file, setup=False, no_output=True, default_config=True)

    # Get base load values for ALL loads in system
    base_loads = {}
    if hasattr(ss, "PQ") and ss.PQ.n > 0:
        for load_idx in range(ss.PQ.n):
            base_p = float(ss.PQ.p0.v[load_idx]) if hasattr(ss.PQ, "p0") else 0.0
            base_q = (
                float(ss.PQ.q0.v[load_idx])
                if hasattr(ss.PQ, "q0") and hasattr(ss.PQ.q0, "v")
                else 0.0
            )
            base_loads[load_idx] = {"p0": base_p, "q0": base_q}

        if verbose:
            print(
                f"[LOAD] Found {ss.PQ.n} load(s). Base values: "
                f"P={[b['p0'] for b in base_loads.values()]}, "
                f"Q={[b['q0'] for b in base_loads.values()]}"
            )
    else:
        warnings.warn("No PQ load found in system. Load scaling will be skipped.")
        base_loads = {}

    # Apply uniform scaling to ALL loads (P and Q by same alpha - industrial standard)
    q_alpha = alpha if load_q_alpha is None else load_q_alpha
    actual_loads = {}
    P_total = 0.0
    Q_total = 0.0

    for load_idx in range(ss.PQ.n):
        base_p = base_loads[load_idx]["p0"]
        base_q = base_loads[load_idx]["q0"]

        # Scale by alpha
        new_p0 = alpha * base_p
        # Default: Q scaled by same alpha as P (maintains power factor)
        # Optional: Independent Q scaling via load_q_alpha (for special studies)
        new_q0 = q_alpha * base_q

        actual_loads[load_idx] = {"p0": new_p0, "q0": new_q0}
        P_total += new_p0
        Q_total += new_q0

        # Apply scaling using ANDES alter() method
        try:
            # Try to get UID first
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
                ss.PQ.alter("p0", load_identifier, new_p0)
                ss.PQ.alter("q0", load_identifier, new_q0)
            else:
                # Fallback to direct access
                ss.PQ.p0.v[load_idx] = new_p0
                if hasattr(ss.PQ, "q0") and hasattr(ss.PQ.q0, "v"):
                    ss.PQ.q0.v[load_idx] = new_q0
        except Exception as e:
            # Fallback to direct access
            ss.PQ.p0.v[load_idx] = new_p0
            if hasattr(ss.PQ, "q0") and hasattr(ss.PQ.q0, "v"):
                ss.PQ.q0.v[load_idx] = new_q0
            if verbose:
                print(f"[WARNING] alter() failed for load {load_idx}, using direct access: {e}")

        if verbose:
            print(
                f"[LOAD] Load {load_idx}: P={new_p0:.6f} pu (α×{base_p:.6f}), "
                f"Q={new_q0:.6f} pu ({q_alpha}×{base_q:.6f})"
            )

    # Setup system
    if verbose:
        print("[SETUP] Setting up system...")
    ss.setup()

    # Run power flow
    if verbose:
        print("[PFLOW] Running power flow...")
    ss.PFlow.run()

    # Validate power flow convergence
    if validate:
        if not (hasattr(ss.PFlow, "converged") and ss.PFlow.converged):
            raise RuntimeError(
                f"Power flow failed to converge for alpha={alpha}. "
                f"Check system parameters and load level."
            )

    # Extract generator parameters after power flow
    # For SMIB: single generator, for multimachine: multiple generators
    Pm_dict = {}
    H_dict = {}
    D_dict = {}

    if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
        for gen_idx in range(ss.GENCLS.n):
            Pm_actual = float(ss.GENCLS.tm0.v[gen_idx])
            M_actual = float(ss.GENCLS.M.v[gen_idx])
            D_actual = float(ss.GENCLS.D.v[gen_idx])
            H_actual = M_actual / 2.0

            Pm_dict[gen_idx] = Pm_actual
            H_dict[gen_idx] = H_actual
            D_dict[gen_idx] = D_actual

    # Set H and D AFTER power flow, BEFORE TDS.init()
    if H is not None:
        if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
            for gen_idx in range(ss.GENCLS.n):
                # Convert H to M: M = 2H
                M = 2.0 * H
                ss.GENCLS.M.v[gen_idx] = M
                if verbose:
                    print(f"[GEN] Generator {gen_idx}: Set H={H:.4f} s (M={M:.4f} s)")

    if D is not None:
        if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
            for gen_idx in range(ss.GENCLS.n):
                ss.GENCLS.D.v[gen_idx] = D
                if verbose:
                    print(f"[GEN] Generator {gen_idx}: Set D={D:.4f} pu")

    # Configure fault
    if hasattr(ss, "Fault") and ss.Fault.n > 0:
        ss.Fault.tf.v[0] = fault_start_time
        if fault_clearing_time is not None:
            ss.Fault.tc.v[0] = fault_clearing_time
        ss.Fault.bus.v[0] = fault_bus
        ss.Fault.xf.v[0] = fault_reactance
    else:
        raise RuntimeError("No fault model found in system")

    # Configure TDS
    ss.TDS.config.tf = simulation_time
    if time_step is not None:
        ss.TDS.config.h = time_step
    ss.TDS.config.criteria = 0  # Disable early stopping

    # Disable Toggles (e.g. Kundur line trip at t=2s) so only the fault event occurs
    if hasattr(ss, "Toggle") and ss.Toggle.n > 0:
        try:
            for _i in range(ss.Toggle.n):
                try:
                    if hasattr(ss.Toggle, "u") and hasattr(ss.Toggle.u, "v"):
                        ss.Toggle.u.v[_i] = 0
                    else:
                        ss.Toggle.alter("u", _i + 1, 0)
                except Exception:
                    ss.Toggle.alter("u", _i + 1, 0)
        except Exception:
            pass

    # Initialize TDS (CRITICAL: Must be AFTER H, D are set)
    if verbose:
        print("[TDS] Initializing TDS...")
    try:
        ss.TDS.init()
    except Exception as e:
        if verbose:
            print(f"[TDS] Warning: TDS.init() raised exception: {e}")

    # Run TDS
    if verbose:
        print(f"[TDS] Running simulation (t=0 to {simulation_time} s)...")
    ss.TDS.run()

    # Extract trajectories
    # For SMIB: single generator, for multimachine: multiple generators
    if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
        # Extract time array - use ANDES 1.10.0 compatible method
        # Priority 1: ss.dae.ts.t (recommended for ANDES 1.10.0)
        if hasattr(ss, "dae") and hasattr(ss.dae, "ts") and ss.dae.ts is not None:
            if hasattr(ss.dae.ts, "t") and ss.dae.ts.t is not None:
                time_array = np.array(ss.dae.ts.t)
            else:
                # Fallback: reconstruct from simulation parameters
                tf = ss.TDS.config.tf if hasattr(ss.TDS.config, "tf") else simulation_time
                h = ss.TDS.config.h if hasattr(ss.TDS.config, "h") else (time_step or 0.01)
                n_points = int(tf / h) + 1
                time_array = np.linspace(0, tf, n_points)
        # Priority 2: TDS plotter (fallback)
        elif hasattr(ss, "TDS") and hasattr(ss.TDS, "plt") and ss.TDS.plt is not None:
            if hasattr(ss.TDS.plt, "t") and ss.TDS.plt.t is not None:
                time_array = np.array(ss.TDS.plt.t)
            else:
                # Fallback: reconstruct from simulation parameters
                tf = ss.TDS.config.tf if hasattr(ss.TDS.config, "tf") else simulation_time
                h = ss.TDS.config.h if hasattr(ss.TDS.config, "h") else (time_step or 0.01)
                n_points = int(tf / h) + 1
                time_array = np.linspace(0, tf, n_points)
        else:
            # Last resort: reconstruct from simulation parameters
            tf = ss.TDS.config.tf if hasattr(ss.TDS.config, "tf") else simulation_time
            h = ss.TDS.config.h if hasattr(ss.TDS.config, "h") else (time_step or 0.01)
            n_points = int(tf / h) + 1
            time_array = np.linspace(0, tf, n_points)

        if ss.GENCLS.n == 1:
            # SMIB: single generator
            # Extract trajectories using ANDES 1.10.0 compatible methods
            # Priority 1: Use ss.dae.ts (recommended for ANDES 1.10.0)
            if hasattr(ss, "dae") and hasattr(ss.dae, "ts") and ss.dae.ts is not None:
                # Extract delta and omega from state variables (x)
                if hasattr(ss.GENCLS.delta, "a") and hasattr(ss.dae.ts, "x"):
                    delta_idx = ss.GENCLS.delta.a[0]
                    delta = (
                        np.array(ss.dae.ts.x[:, delta_idx])
                        if delta_idx < ss.dae.ts.x.shape[1]
                        else np.array([])
                    )
                else:
                    delta = np.array([])

                if hasattr(ss.GENCLS.omega, "a") and hasattr(ss.dae.ts, "x"):
                    omega_idx = ss.GENCLS.omega.a[0]
                    omega = (
                        np.array(ss.dae.ts.x[:, omega_idx])
                        if omega_idx < ss.dae.ts.x.shape[1]
                        else np.array([])
                    )
                else:
                    omega = np.array([])

                # Extract Pe from algebraic variables (y)
                if hasattr(ss.GENCLS.Pe, "a") and hasattr(ss.dae.ts, "y"):
                    pe_idx = ss.GENCLS.Pe.a[0]
                    Pe = (
                        np.array(ss.dae.ts.y[:, pe_idx])
                        if pe_idx < ss.dae.ts.y.shape[1]
                        else np.array([])
                    )
                else:
                    Pe = np.array([])
            # Priority 2: Use TDS plotter (fallback)
            elif hasattr(ss, "TDS") and hasattr(ss.TDS, "plt") and ss.TDS.plt is not None:
                delta = (
                    np.array(ss.TDS.plt.GENCLS.delta[:, 0])
                    if hasattr(ss.TDS.plt, "GENCLS") and hasattr(ss.TDS.plt.GENCLS, "delta")
                    else np.array([])
                )
                omega = (
                    np.array(ss.TDS.plt.GENCLS.omega[:, 0])
                    if hasattr(ss.TDS.plt, "GENCLS") and hasattr(ss.TDS.plt.GENCLS, "omega")
                    else np.array([])
                )
                Pe = (
                    np.array(ss.TDS.plt.GENCLS.Pe[:, 0])
                    if hasattr(ss.TDS.plt, "GENCLS") and hasattr(ss.TDS.plt.GENCLS, "Pe")
                    else np.array([])
                )
            else:
                # Last resort: use final values (not ideal, but better than error)
                delta = (
                    np.array([ss.GENCLS.delta.v[0]])
                    if hasattr(ss.GENCLS.delta, "v")
                    else np.array([])
                )
                omega = (
                    np.array([ss.GENCLS.omega.v[0]])
                    if hasattr(ss.GENCLS.omega, "v")
                    else np.array([])
                )
                Pe = np.array([ss.GENCLS.Pe.v[0]]) if hasattr(ss.GENCLS.Pe, "v") else np.array([])

            result = {
                "time": time_array,
                "delta": delta,
                "omega": omega,
                "Pe": Pe,
                "alpha": alpha,
                "P_base": base_loads[0]["p0"] if len(base_loads) > 0 else 0.0,
                "P_actual": actual_loads[0]["p0"] if len(actual_loads) > 0 else 0.0,
                "Q_base": base_loads[0]["q0"] if len(base_loads) > 0 else 0.0,
                "Q_actual": actual_loads[0]["q0"] if len(actual_loads) > 0 else 0.0,
                "P_total": P_total,
                "Pm": Pm_dict[0] if len(Pm_dict) > 0 else None,
                "H": H_dict[0] if len(H_dict) > 0 else None,
                "D": D_dict[0] if len(D_dict) > 0 else None,
                "metadata": {
                    "converged": ss.PFlow.converged if hasattr(ss.PFlow, "converged") else None,
                    "num_loads": ss.PQ.n if hasattr(ss, "PQ") else 0,
                    "num_generators": ss.GENCLS.n,
                },
            }
        else:
            # Multimachine: multiple generators
            delta_dict = {}
            omega_dict = {}
            Pe_dict = {}

            for gen_idx in range(ss.GENCLS.n):
                # Extract trajectories using ANDES 1.10.0 compatible methods
                # Priority 1: Use ss.dae.ts (recommended for ANDES 1.10.0)
                if hasattr(ss, "dae") and hasattr(ss.dae, "ts") and ss.dae.ts is not None:
                    if (
                        hasattr(ss.GENCLS.delta, "a")
                        and hasattr(ss.dae.ts, "x")
                        and gen_idx < len(ss.GENCLS.delta.a)
                    ):
                        delta_idx = ss.GENCLS.delta.a[gen_idx]
                        delta_dict[gen_idx] = (
                            np.array(ss.dae.ts.x[:, delta_idx])
                            if delta_idx < ss.dae.ts.x.shape[1]
                            else np.array([])
                        )
                    else:
                        delta_dict[gen_idx] = np.array([])

                    if (
                        hasattr(ss.GENCLS.omega, "a")
                        and hasattr(ss.dae.ts, "x")
                        and gen_idx < len(ss.GENCLS.omega.a)
                    ):
                        omega_idx = ss.GENCLS.omega.a[gen_idx]
                        omega_dict[gen_idx] = (
                            np.array(ss.dae.ts.x[:, omega_idx])
                            if omega_idx < ss.dae.ts.x.shape[1]
                            else np.array([])
                        )
                    else:
                        omega_dict[gen_idx] = np.array([])

                    if (
                        hasattr(ss.GENCLS.Pe, "a")
                        and hasattr(ss.dae.ts, "y")
                        and gen_idx < len(ss.GENCLS.Pe.a)
                    ):
                        pe_idx = ss.GENCLS.Pe.a[gen_idx]
                        Pe_dict[gen_idx] = (
                            np.array(ss.dae.ts.y[:, pe_idx])
                            if pe_idx < ss.dae.ts.y.shape[1]
                            else np.array([])
                        )
                    else:
                        Pe_dict[gen_idx] = np.array([])
                # Priority 2: Use TDS plotter (fallback)
                elif hasattr(ss, "TDS") and hasattr(ss.TDS, "plt") and ss.TDS.plt is not None:
                    delta_dict[gen_idx] = (
                        np.array(ss.TDS.plt.GENCLS.delta[:, gen_idx])
                        if hasattr(ss.TDS.plt, "GENCLS") and hasattr(ss.TDS.plt.GENCLS, "delta")
                        else np.array([])
                    )
                    omega_dict[gen_idx] = (
                        np.array(ss.TDS.plt.GENCLS.omega[:, gen_idx])
                        if hasattr(ss.TDS.plt, "GENCLS") and hasattr(ss.TDS.plt.GENCLS, "omega")
                        else np.array([])
                    )
                    Pe_dict[gen_idx] = (
                        np.array(ss.TDS.plt.GENCLS.Pe[:, gen_idx])
                        if hasattr(ss.TDS.plt, "GENCLS") and hasattr(ss.TDS.plt.GENCLS, "Pe")
                        else np.array([])
                    )
                else:
                    # Last resort: use final values (not ideal, but better than error)
                    delta_dict[gen_idx] = (
                        np.array([ss.GENCLS.delta.v[gen_idx]])
                        if hasattr(ss.GENCLS.delta, "v") and gen_idx < len(ss.GENCLS.delta.v)
                        else np.array([])
                    )
                    omega_dict[gen_idx] = (
                        np.array([ss.GENCLS.omega.v[gen_idx]])
                        if hasattr(ss.GENCLS.omega, "v") and gen_idx < len(ss.GENCLS.omega.v)
                        else np.array([])
                    )
                    Pe_dict[gen_idx] = (
                        np.array([ss.GENCLS.Pe.v[gen_idx]])
                        if hasattr(ss.GENCLS.Pe, "v") and gen_idx < len(ss.GENCLS.Pe.v)
                        else np.array([])
                    )

            result = {
                "time": time_array,
                "delta": delta_dict,
                "omega": omega_dict,
                "Pe": Pe_dict,
                "alpha": alpha,
                "P_base": {idx: base_loads[idx]["p0"] for idx in base_loads.keys()},
                "P_actual": {idx: actual_loads[idx]["p0"] for idx in actual_loads.keys()},
                "Q_base": {idx: base_loads[idx]["q0"] for idx in base_loads.keys()},
                "Q_actual": {idx: actual_loads[idx]["q0"] for idx in actual_loads.keys()},
                "P_total": P_total,
                "Pm": Pm_dict,
                "H": H_dict,
                "D": D_dict,
                "metadata": {
                    "converged": ss.PFlow.converged if hasattr(ss.PFlow, "converged") else None,
                    "num_loads": ss.PQ.n if hasattr(ss, "PQ") else 0,
                    "num_generators": ss.GENCLS.n,
                },
            }
    else:
        raise RuntimeError("No generator model (GENCLS) found in system")

    return result
