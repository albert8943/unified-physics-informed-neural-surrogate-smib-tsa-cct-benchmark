"""
Parameter Sweep Generator for PINN Training Data.

This module generates diverse training data by varying system parameters
(H, D, fault scenarios, line configurations) and running ANDES simulations.

Supports task-specific data generation strategies:
- Trajectory prediction: Full factorial or LHS sampling
- Parameter estimation: Decorrelated sampling (low H-D correlation)
- CCT estimation: Boundary-focused sampling near stability limits
"""

import itertools
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import progress tracker if available
try:
    from utils.progress_tracker import DataGenerationTracker

    PROGRESS_TRACKER_AVAILABLE = True
except ImportError:
    PROGRESS_TRACKER_AVAILABLE = False
    DataGenerationTracker = None

# When set (e.g. ANDES_DEBUG_TDS=1), show [DEBUG], [PHASE 3], and repeated TDS diagnostics.
# Default: only important messages (progress, CCT, errors); no repeated state dumps.
DEBUG_TDS = os.environ.get("ANDES_DEBUG_TDS", "").strip().lower() in ("1", "true", "yes")

# Suppress ANDES warnings about typical limits (these are informational, not errors)
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress specific ANDES warnings about vf range limits
warnings.filterwarnings("ignore", message=".*vf range.*")
warnings.filterwarnings("ignore", message=".*typical.*limit.*")

# Disable tqdm progress bars to avoid ipywidgets dependency issues
# This prevents ANDES from trying to use tqdm_notebook which requires ipywidgets
os.environ["TQDM_DISABLE"] = "1"
# Also set environment variable to force tqdm to use standard output instead of notebook
os.environ["TQDM_NOTEBOOK"] = "0"

# Try to patch tqdm to disable notebook mode before ANDES imports it
# IMPORTANT: We need to preserve tqdm.write() method which ANDES uses
try:
    import tqdm

    # Store original tqdm class
    _original_tqdm = tqdm.tqdm

    # Create a wrapper class that disables progress bars but preserves write()
    class _DisabledTqdm(_original_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    # Preserve the write static method (ANDES uses tqdm.write() to print messages)
    if hasattr(_original_tqdm, "write"):
        # Copy the write method (it's a staticmethod)
        _DisabledTqdm.write = _original_tqdm.write
    else:
        # Fallback: create a write method that prints to stdout
        def _write_method(s, file=None, end="\n"):
            if file is None:
                import sys

                file = sys.stdout
            file.write(s + end)
            file.flush()

        _DisabledTqdm.write = staticmethod(_write_method)

    # Replace tqdm.tqdm with our disabled version
    tqdm.tqdm = _DisabledTqdm
except ImportError:
    pass

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False
    print("Warning: ANDES not available. Some functions may not work.")

from .andes_extractor import (
    extract_complete_dataset,
    extract_pe_trajectories,
    extract_trajectories,
    extract_system_reactances,
)
from .sampling_strategies import (
    boundary_focused_sample,
    correlation_analysis,
    decorrelated_sample,
    filter_extreme_combinations,
    full_factorial_sample,
    latin_hypercube_sample,
    sobol_sequence_sample,
    validate_sample_quality,
)

# Import redispatch functions if available
try:
    from .andes_utils.multimachine_powerflow import (
        run_multimachine_powerflow_with_redispatch,
    )

    REDISPATCH_AVAILABLE = True
except ImportError:
    REDISPATCH_AVAILABLE = False
    run_multimachine_powerflow_with_redispatch = None


def diagnose_pe_storage_locations(
    ss, gen_idx: int, expected_pe: float, verbose: bool = True
) -> Dict:
    """
    Phase 0: Comprehensive diagnostic function to trace where Pe is stored.

    This function checks ALL possible locations where Pe might be stored after power flow,
    to understand where TDS.init() actually reads from.

    Args:
        ss: ANDES system object
        gen_idx: Generator index to check
        expected_pe: Expected Pe value (for comparison)
        verbose: Whether to print diagnostic output

    Returns:
        Dictionary with diagnostic results:
        - 'pe_locations': Dict mapping location name to Pe value (or None if not found)
        - 'pe_found': List of locations where Pe was found
        - 'pe_correct': List of locations where Pe matches expected value
        - 'recommendation': Suggested fix based on findings
    """
    results = {
        "pe_locations": {},
        "pe_found": [],
        "pe_correct": [],
        "recommendation": None,
    }

    if verbose:
        print("\n" + "=" * 80)
        print("[PHASE 0] COMPREHENSIVE Pe STORAGE LOCATION DIAGNOSTIC")
        print("=" * 80)
        print(f"Generator index: {gen_idx}")
        print(f"Expected Pe: {expected_pe:.6f} pu")
        print("-" * 80)

    # Location 1: ss.dae.y (algebraic variables) - via Pe.a index
    try:
        if hasattr(ss, "dae") and hasattr(ss.dae, "y") and ss.dae.y is not None:
            if hasattr(ss.GENCLS, "Pe") and hasattr(ss.GENCLS.Pe, "a"):
                pe_a = ss.GENCLS.Pe.a
                if (
                    hasattr(pe_a, "__getitem__")
                    and hasattr(pe_a, "__len__")
                    and len(pe_a) > gen_idx
                ):
                    pe_a_idx = int(pe_a[gen_idx])
                    if pe_a_idx < len(ss.dae.y):
                        pe_from_dae_y = float(ss.dae.y[pe_a_idx])
                        results["pe_locations"]["dae.y[Pe.a]"] = pe_from_dae_y
                        results["pe_found"].append("dae.y[Pe.a]")
                        if abs(pe_from_dae_y - expected_pe) < 0.01:
                            results["pe_correct"].append("dae.y[Pe.a]")
                        if verbose:
                            match_str = "[OK]" if abs(pe_from_dae_y - expected_pe) < 0.01 else "[X]"
                            print(
                                f"[1] ss.dae.y[pe_a_idx={pe_a_idx}]: {pe_from_dae_y:.6f} pu"
                                f"{match_str}"
                            )
                    else:
                        if verbose:
                            print(
                                f"[1] ss.dae.y[Pe.a]: pe_a_idx={pe_a_idx} >="
                                f"len(ss.dae.y)={len(ss.dae.y)} (INVALID)"
                            )
                else:
                    if verbose:
                        pe_a_len = len(pe_a) if hasattr(pe_a, "__len__") else "N/A"
                        print(f"[1] ss.dae.y[Pe.a]: pe_a is empty or invalid (len={pe_a_len})")
            else:
                if verbose:
                    print(f"[1] ss.dae.y[Pe.a]: Pe.a not available")
        else:
            if verbose:
                print(f"[1] ss.dae.y[Pe.a]: ss.dae.y is None or doesn't exist")
    except Exception as e:
        if verbose:
            print(f"[1] ss.dae.y[Pe.a]: ERROR - {e}")

    # Location 2: ss.dae.x (state variables) - might contain Pe if it's a state variable
    try:
        if hasattr(ss, "dae") and hasattr(ss.dae, "x") and ss.dae.x is not None:
            # Pe is typically algebraic, not state, but check anyway
            if verbose:
                print(f"[2] ss.dae.x: Pe is algebraic, not state (skipping)")
    except Exception as e:
        if verbose:
            print(f"[2] ss.dae.x: ERROR - {e}")

    # Location 3: ss.GENCLS.Pe.v (direct attribute access)
    try:
        if hasattr(ss.GENCLS, "Pe") and hasattr(ss.GENCLS.Pe, "v"):
            pe_v = ss.GENCLS.Pe.v
            if hasattr(pe_v, "__getitem__") and hasattr(pe_v, "__len__"):
                if len(pe_v) > gen_idx:
                    pe_from_v = float(pe_v[gen_idx])
                    results["pe_locations"]["GENCLS.Pe.v"] = pe_from_v
                    results["pe_found"].append("GENCLS.Pe.v")
                    if abs(pe_from_v - expected_pe) < 0.01:
                        results["pe_correct"].append("GENCLS.Pe.v")
                    if verbose:
                        match_str = "[OK]" if abs(pe_from_v - expected_pe) < 0.01 else "[X]"
                        print(f"[3] ss.GENCLS.Pe.v[{gen_idx}]: {pe_from_v:.6f} pu {match_str}")
                elif len(pe_v) > 0:
                    pe_from_v = float(pe_v[0])
                    results["pe_locations"]["GENCLS.Pe.v[0]"] = pe_from_v
                    results["pe_found"].append("GENCLS.Pe.v[0]")
                    if abs(pe_from_v - expected_pe) < 0.01:
                        results["pe_correct"].append("GENCLS.Pe.v[0]")
                    if verbose:
                        match_str = "[OK]" if abs(pe_from_v - expected_pe) < 0.01 else "[X]"
                        print(f"[3] ss.GENCLS.Pe.v[0]: {pe_from_v:.6f} pu {match_str}")
                else:
                    if verbose:
                        print(f"[3] ss.GENCLS.Pe.v: Empty array (length 0)")
            else:
                try:
                    pe_from_v = float(pe_v)
                    results["pe_locations"]["GENCLS.Pe.v"] = pe_from_v
                    results["pe_found"].append("GENCLS.Pe.v")
                    if abs(pe_from_v - expected_pe) < 0.01:
                        results["pe_correct"].append("GENCLS.Pe.v")
                    if verbose:
                        match_str = "[OK]" if abs(pe_from_v - expected_pe) < 0.01 else "[X]"
                        print(f"[3] ss.GENCLS.Pe.v (scalar): {pe_from_v:.6f} pu {match_str}")
                except (ValueError, TypeError):
                    if verbose:
                        print(f"[3] ss.GENCLS.Pe.v: Cannot convert to float")
        else:
            if verbose:
                print(f"[3] ss.GENCLS.Pe.v: Not available")
    except Exception as e:
        if verbose:
            print(f"[3] ss.GENCLS.Pe.v: ERROR - {e}")

    # Location 4: ss.GENCLS.get('Pe') (official ANDES method)
    try:
        if hasattr(ss.GENCLS, "get"):
            pe_get = ss.GENCLS.get("Pe")
            if pe_get is not None:
                if isinstance(pe_get, np.ndarray):
                    if pe_get.ndim == 1 and len(pe_get) > gen_idx:
                        pe_from_get = float(pe_get[gen_idx])
                        results["pe_locations"]["GENCLS.get(Pe)"] = pe_from_get
                        results["pe_found"].append("GENCLS.get(Pe)")
                        if abs(pe_from_get - expected_pe) < 0.01:
                            results["pe_correct"].append("GENCLS.get(Pe)")
                        if verbose:
                            match_str = "[OK]" if abs(pe_from_get - expected_pe) < 0.01 else "[X]"
                            print(
                                f"[4] ss.GENCLS.get('Pe')[{gen_idx}]: {pe_from_get:.6f} pu"
                                f"{match_str}"
                            )
                    elif pe_get.ndim == 1 and len(pe_get) > 0:
                        pe_from_get = float(pe_get[0])
                        results["pe_locations"]["GENCLS.get(Pe)[0]"] = pe_from_get
                        results["pe_found"].append("GENCLS.get(Pe)[0]")
                        if abs(pe_from_get - expected_pe) < 0.01:
                            results["pe_correct"].append("GENCLS.get(Pe)[0]")
                        if verbose:
                            match_str = "[OK]" if abs(pe_from_get - expected_pe) < 0.01 else "[X]"
                            print(f"[4] ss.GENCLS.get('Pe')[0]: {pe_from_get:.6f} pu {match_str}")
                else:
                    try:
                        pe_from_get = float(pe_get)
                        results["pe_locations"]["GENCLS.get(Pe)"] = pe_from_get
                        results["pe_found"].append("GENCLS.get(Pe)")
                        if abs(pe_from_get - expected_pe) < 0.01:
                            results["pe_correct"].append("GENCLS.get(Pe)")
                        if verbose:
                            match_str = "[OK]" if abs(pe_from_get - expected_pe) < 0.01 else "[X]"
                            print(
                                f"[4] ss.GENCLS.get('Pe') (scalar): {pe_from_get:.6f} pu"
                                f"{match_str}"
                            )
                    except (ValueError, TypeError):
                        if verbose:
                            print(f"[4] ss.GENCLS.get('Pe'): Cannot convert to float")
            else:
                if verbose:
                    print(f"[4] ss.GENCLS.get('Pe'): Returns None")
        else:
            if verbose:
                print(f"[4] ss.GENCLS.get('Pe'): get() method not available")
    except Exception as e:
        if verbose:
            print(f"[4] ss.GENCLS.get('Pe'): ERROR - {e}")

    # Location 5: ss.PFlow.Pe (if power flow stores Pe directly)
    try:
        if hasattr(ss, "PFlow"):
            if hasattr(ss.PFlow, "Pe"):
                pe_from_pf = ss.PFlow.Pe
                if isinstance(pe_from_pf, (int, float)):
                    results["pe_locations"]["PFlow.Pe"] = float(pe_from_pf)
                    results["pe_found"].append("PFlow.Pe")
                    if abs(float(pe_from_pf) - expected_pe) < 0.01:
                        results["pe_correct"].append("PFlow.Pe")
                    if verbose:
                        match_str = "[OK]" if abs(float(pe_from_pf) - expected_pe) < 0.01 else "[X]"
                        print(f"[5] ss.PFlow.Pe: {float(pe_from_pf):.6f} pu {match_str}")
                else:
                    if verbose:
                        print(f"[5] ss.PFlow.Pe: Not a scalar value")
            else:
                if verbose:
                    print(f"[5] ss.PFlow.Pe: Not available")
        else:
            if verbose:
                print(f"[5] ss.PFlow.Pe: PFlow not available")
    except Exception as e:
        if verbose:
            print(f"[5] ss.PFlow.Pe: ERROR - {e}")

    # Location 6: ss.PFlow.solution (power flow solution object)
    try:
        if hasattr(ss, "PFlow") and hasattr(ss.PFlow, "solution"):
            solution = ss.PFlow.solution
            if solution is not None:
                if verbose:
                    solution_dir = [x for x in dir(solution) if not x.startswith("_")][:10]
                    print(f"[6] ss.PFlow.solution: Type={type(solution)}, dir={solution_dir}")
                    # Try to find Pe in solution object
                    if hasattr(solution, "Pe"):
                        pe_from_solution = solution.Pe
                        if isinstance(pe_from_solution, (int, float)):
                            results["pe_locations"]["PFlow.solution.Pe"] = float(pe_from_solution)
                            results["pe_found"].append("PFlow.solution.Pe")
                            if abs(float(pe_from_solution) - expected_pe) < 0.01:
                                results["pe_correct"].append("PFlow.solution.Pe")
                            if verbose:
                                match_str = (
                                    "[OK]"
                                    if abs(float(pe_from_solution) - expected_pe) < 0.01
                                    else "[X]"
                                )
                                print(
                                    f"[6] ss.PFlow.solution.Pe: {float(pe_from_solution):.6f} pu"
                                    f"{match_str}"
                                )
            else:
                if verbose:
                    print(f"[6] ss.PFlow.solution: None")
        else:
            if verbose:
                print(f"[6] ss.PFlow.solution: Not available")
    except Exception as e:
        if verbose:
            print(f"[6] ss.PFlow.solution: ERROR - {e}")

    # Location 7: Case file default (via case file modifier)
    try:
        if CASE_FILE_MODIFIER_AVAILABLE and get_default_pm_from_case_file is not None:
            # This would require case file path, skip for now
            if verbose:
                print(f"[7] Case file default: Requires case file path (skipped in diagnostic)")
        else:
            if verbose:
                print(f"[7] Case file default: Case file modifier not available")
    except Exception as e:
        if verbose:
            print(f"[7] Case file default: ERROR - {e}")

    # Summary and recommendation
    if verbose:
        print("-" * 80)
        print("SUMMARY:")
        print(f"  Locations where Pe was found: {len(results['pe_found'])}")
        if results["pe_found"]:
            print(f"    Found in: {', '.join(results['pe_found'])}")
        print(f"  Locations where Pe is correct: {len(results['pe_correct'])}")
        if results["pe_correct"]:
            print(f"    Correct in: {', '.join(results['pe_correct'])}")
        else:
            print(f"    [X] No locations have correct Pe value!")

        # Generate recommendation
        if results["pe_correct"]:
            results["recommendation"] = (
                f"Pe is correctly stored in: {', '.join(results['pe_correct'])}. TDS.init() should"
                f"read from one of these."
            )
        elif results["pe_found"]:
            wrong_locs = [loc for loc in results["pe_found"] if loc not in results["pe_correct"]]
            results["recommendation"] = (
                f"Pe found but wrong in: {', '.join(wrong_locs)}. Need to fix these locations or"
                f"modify case file."
            )
        else:
            results[
                "recommendation"
            ] = "Pe not found in any location. Power flow may not have stored it correctly. Check power flow convergence."

        print(f"  Recommendation: {results['recommendation']}")
        print("=" * 80 + "\n")

    return results


def check_extreme_parameter_combination(
    H: float,
    D: float,
    Load_P: Optional[float] = None,
    Pm: Optional[float] = None,
    H_range: Optional[Tuple[float, float]] = None,
    D_range: Optional[Tuple[float, float]] = None,
    load_range: Optional[Tuple[float, float]] = None,
    verbose: bool = False,
    # Priority 1: Absolute thresholds based on physical reality
    H_min_absolute: float = 2.0,  # seconds - minimum realistic H for most generators
    D_min_absolute: float = 0.5,  # pu - minimum realistic D (natural + control damping)
    D_min_critical: float = 0.3,  # pu - critical minimum (natural damping alone)
    Load_max_absolute: float = 0.9,  # pu - maximum safe loading (may cause voltage issues)
) -> Tuple[bool, str]:
    """
    Check if parameter combination is extreme and may cause convergence issues.

    Uses BOTH absolute thresholds (physical reality) and normalized thresholds (relative to range).

    Extreme combinations:
    - High load + low H + low D = unstable (may not converge)
    - Low load + high H + high D = very stable (less informative)

    Args:
        H: Inertia constant (seconds)
        D: Damping coefficient (pu)
        Load_P: Load active power (pu) - optional
        Pm: Mechanical power (pu) - optional
        H_range: (H_min, H_max) tuple for normalization
        D_range: (D_min, D_max) tuple for normalization
        load_range: (load_min, load_max) tuple for normalization
        verbose: Whether to print warnings
        H_min_absolute: Absolute minimum H (seconds) - default 2.0s
        D_min_absolute: Absolute minimum D (pu) - default 0.5pu
        D_min_critical: Critical minimum D (pu) - default 0.3pu (unrealistic)
        Load_max_absolute: Absolute maximum load (pu) - default 0.9pu

    Returns:
        Tuple of (is_extreme, warning_message)
        - is_extreme: True if combination is extreme
        - warning_message: Description of the issue
    """
    warning_msg = ""
    is_extreme = False
    P_value = Load_P if Load_P is not None else Pm

    # PRIORITY 1: Check absolute thresholds FIRST (physical reality)
    # These are independent of input ranges and based on power system engineering practice

    # Check absolute H threshold
    if H < H_min_absolute:
        is_extreme = True
        warning_msg = (
            f"Extreme combination detected (absolute): H={H:.2f}s < {H_min_absolute}s. "
            f"This is unrealistic for most generators (except very small units). "
            f"May cause instability or convergence issues."
        )
        return is_extreme, warning_msg

    # Check absolute D threshold (critical minimum)
    if D < D_min_critical:
        is_extreme = True
        warning_msg = (
            f"Extreme combination detected (absolute): D={D:.2f}pu < {D_min_critical}pu. "
            f"This is unrealistic (natural damping alone is ~0.1-0.3pu). "
            f"May cause oscillations or convergence issues."
        )
        return is_extreme, warning_msg

    # Check absolute load threshold
    if P_value is not None and P_value > Load_max_absolute:
        is_extreme = True
        warning_msg = (
            f"Extreme combination detected (absolute): Load={P_value:.3f}pu >"
            f"{Load_max_absolute}pu."
            f"This may cause voltage instability even without faults. "
            f"May cause power flow convergence failures."
        )
        return is_extreme, warning_msg

    # Check if D is below recommended minimum (warning, not rejection)
    if D < D_min_absolute:
        if verbose:
            warning_msg = (
                f"Warning: D={D:.2f}pu < {D_min_absolute}pu (recommended minimum). "
                f"Natural damping (~0.1-0.3pu) + "
                f"control damping (~0.2-0.5pu) typically gives D > 0.5pu. "
                f"Low damping may cause oscillations."
            )

    # Normalize parameters to [0, 1] range if ranges are provided (for relative checks)
    H_norm = None
    D_norm = None
    Load_norm = None

    if H_range is not None:
        H_min, H_max = H_range
        if H_max > H_min:
            H_norm = (H - H_min) / (H_max - H_min)

    if D_range is not None:
        D_min, D_max = D_range
        if D_max > D_min:
            D_norm = (D - D_min) / (D_max - D_min)

    if Load_P is not None and load_range is not None:
        load_min, load_max = load_range
        if load_max > load_min:
            Load_norm = (Load_P - load_min) / (load_max - load_min)
    elif Pm is not None and load_range is not None:
        # Use Pm as proxy for load if load_range is provided
        load_min, load_max = load_range
        if load_max > load_min:
            Load_norm = (Pm - load_min) / (load_max - load_min)

    # Check for extreme combinations using normalized thresholds (relative to range)
    # This catches combinations that are extreme relative to the specified range
    if H_norm is not None and D_norm is not None and Load_norm is not None:
        # High load (>= 0.8) + Low H (<= 0.3) + Low D (<= 0.3) = unstable
        # Only check if absolute thresholds didn't already flag it
        if not is_extreme and Load_norm >= 0.8 and H_norm <= 0.3 and D_norm <= 0.3:
            is_extreme = True
            warning_msg = (
                f"Extreme combination detected (normalized): High load (P={P_value:.3f} pu, "
                f"norm={Load_norm:.2f}) + Low H (H={H:.2f} s, norm={H_norm:.2f}) + "
                f"Low D (D={D:.2f} pu, norm={D_norm:.2f}). "
                f"This combination may cause power flow convergence failures or instability."
            )
        # Low load (<= 0.2) + High H (>= 0.8) + High D (>= 0.8) = very stable (less informative)
        elif Load_norm <= 0.2 and H_norm >= 0.8 and D_norm >= 0.8:
            is_extreme = False  # Not rejected, but less informative
            if verbose:
                warning_msg = (
                    f"Very stable combination: Low load (P={P_value:.3f} pu, "
                    f"norm={Load_norm:.2f}) + High H (H={H:.2f} s, norm={H_norm:.2f}) + "
                    f"High D (D={D:.2f} pu, norm={D_norm:.2f}). "
                    f"This combination is very stable but less informative for training."
                )

    return is_extreme, warning_msg


# Import stability checker for automatic stability labeling (SMIB and multimachine COI)
try:
    from utils.stability_checker import (
        check_stability,
        check_stability_multimachine_coi,
    )

    STABILITY_CHECKER_AVAILABLE = True
except ImportError:
    STABILITY_CHECKER_AVAILABLE = False
    check_stability = None
    check_stability_multimachine_coi = None

# Import CCT finder for CCT-based sampling
try:
    from .andes_utils.cct_finder import find_cct, find_main_generator_index

    CCT_FINDER_AVAILABLE = True
except ImportError:
    CCT_FINDER_AVAILABLE = False
    find_cct = None
    find_main_generator_index = None


def _multimachine_stability_from_df(df, n_gen, delta_threshold=np.pi):
    """Compute multimachine stability from dataframe using COI-relative angles.
    Returns True/False if computable, None otherwise (use other fallback)."""
    if n_gen <= 0:
        return None
    delta_cols = ["delta_" + str(ii) for ii in range(n_gen)]
    if not all(c in df.columns for c in delta_cols):
        return None
    M = None
    if "param_M" in df.columns and pd.notna(df["param_M"].iloc[0]):
        M = float(df["param_M"].iloc[0])
    elif "param_H" in df.columns and pd.notna(df["param_H"].iloc[0]):
        M = 2.0 * float(df["param_H"].iloc[0])
    if M is None or M <= 0:
        return None
    M_vals = np.full(n_gen, M)
    M_sum = float(np.sum(M_vals))
    delta = np.array([df[c].values for c in delta_cols])
    delta_coi = np.sum(M_vals[:, np.newaxis] * delta, axis=0) / M_sum
    delta_rel = delta - delta_coi
    max_delta_rel = np.abs(delta_rel).max()
    return bool(max_delta_rel < delta_threshold)


# Import case file modifier for operating point variation
try:
    from .andes_utils.case_file_modifier import (
        modify_case_file_generator_setpoint,
        get_default_pm_from_case_file,
        get_default_m_from_case_file,
        get_default_d_from_case_file,
        modify_case_file_multiple_generators,
        modify_case_file_load_setpoint,
        get_default_load_from_case_file,
        check_smib_has_load,
        add_pq_load_to_smib_case,
    )
    from .andes_utils.verification_helpers import (
        verify_generator_setpoints,
        verify_power_flow_converged,
        verify_power_balance,
    )

    CASE_FILE_MODIFIER_AVAILABLE = True
except ImportError:
    CASE_FILE_MODIFIER_AVAILABLE = False
    modify_case_file_generator_setpoint = None
    get_default_pm_from_case_file = None
    get_default_m_from_case_file = None
    get_default_d_from_case_file = None
    modify_case_file_multiple_generators = None
    modify_case_file_load_setpoint = None
    get_default_load_from_case_file = None
    check_smib_has_load = None
    add_pq_load_to_smib_case = None
    verify_generator_setpoints = None
    verify_power_flow_converged = None
    verify_power_balance = None


def generate_parameter_sweep(
    case_file: str,
    output_dir: str = "data",
    H_range: Optional[Tuple[float, float, int]] = None,
    D_range: Optional[Tuple[float, float, int]] = None,
    Pm_range: Optional[Tuple[float, float, int]] = None,
    alpha_range: Optional[
        Tuple[float, float, int]
    ] = None,  # Unified: alpha multiplier for all systems
    load_q_alpha_range: Optional[
        Tuple[float, float, int]
    ] = None,  # Optional: independent Q scaling
    # DEPRECATED: load_range, use_load_variation - use alpha_range instead
    load_range: Optional[Tuple[float, float, int]] = None,
    use_load_variation: bool = False,
    fault_clearing_times: Optional[List[float]] = None,
    fault_locations: Optional[List[int]] = None,
    simulation_time: float = 5.0,
    time_step: float = 0.001,
    verbose: bool = True,
    sampling_strategy: str = "full_factorial",
    task: str = "trajectory",
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    validate_quality: bool = True,
    use_cct_based_sampling: bool = False,
    Pm: Optional[float] = None,
    n_samples_per_combination: int = 5,
    cct_offsets: Optional[List[float]] = None,
    fault_start_time: float = 1.0,
    fault_bus: int = 3,
    fault_reactance: float = 0.0001,
    use_pe_as_input: bool = False,
    filter_extreme_combinations: bool = False,
    base_load: Optional[Dict[str, float]] = None,  # Optional: {"Pload": 0.5, "Qload": 0.2}
    case_default_pm: Optional[
        List[float]
    ] = None,  # Pm from case (per-gen); used when tm0 stays 0 after PF
    use_redispatch: bool = False,  # NEW: Enable participation-factor-based redispatch
    redispatch_config: Optional[Dict] = None,  # NEW: Redispatch configuration
    addfile: Optional[str] = None,  # Optional DYR for PSS/E raw; e.g. "kundur/kundur_gencls.dyr"
    skip_fault: bool = False,  # If True, run TDS without fault (validate base case only; add fault later)
) -> pd.DataFrame:
    """
    Generate diverse training data by sweeping system parameters.

    Parameters:
    -----------
    case_file : str
        Path to ANDES case file (e.g., "smib/SMIB.json")
    output_dir : str
        Directory to save generated datasets
    H_range : tuple, optional
        (min, max, num_points) for inertia constant H (seconds).
        **Note**: Prefer using M_range in config (M is what's stored in case file).
        M_range is automatically converted to H_range internally (H = M/2 for 60 Hz).
        If None, extracts default M from case file and uses H = M/2 (single value).
        Falls back to (2.0, 10.0, 5) if case file extraction fails.
    D_range : tuple, optional
        (min, max, num_points) for damping coefficient D (pu).
        If None, extracts default D from case file (single value).
        Falls back to (0.5, 3.0, 5) if case file extraction fails.
    Pm_range : tuple, optional
        (min, max, num_points) for mechanical power Pm (pu).
        If provided, generator setpoint will be varied.
        Mutually exclusive with load_range (use use_load_variation=True for load variation).
    alpha_range : tuple, optional
        (min, max, num_points) for uniform load multiplier alpha.
        When provided, all loads are scaled uniformly: P' = alpha × P_base, Q' = alpha × Q_base.
        Works for both SMIB and multimachine systems.
        Typical range: (0.4, 1.2, 10) for 40% to 120% of base load.
        NOTE: 3D sampling = (H, D, alpha) only. Q is scaled by same alpha (maintains power factor).
    load_q_alpha_range : tuple, optional
        (min, max, num_points) for independent reactive power scaling multiplier.
        If None, Q is scaled by same alpha as P (uniform scaling, maintains power factor - industrial standard).
        Only use for special studies where independent Q scaling is needed.
    load_range : tuple, optional
        DEPRECATED: Use alpha_range instead. Kept for backward compatibility.
        If provided and alpha_range is None, will be converted to alpha_range (requires base load value).
    use_load_variation : bool
        DEPRECATED: Use alpha_range instead. Kept for backward compatibility.
        If True and load_range is provided, will use load variation mode.
    fault_clearing_times : list, optional
        List of fault clearing times in ABSOLUTE time (seconds), not durations.
        Must be >= fault_start_time. Following smib_albert_cct.py approach.
        Example: If fault_start_time=1.0, use [1.15, 1.18, 1.20, 1.22, 1.25]
    fault_locations : list, optional
        List of bus indices for fault locations. If None, uses default.
    simulation_time : float
        Total simulation time (seconds)
    time_step : float
        Time step for simulation (seconds)
    verbose : bool
        Print progress information
    sampling_strategy : str
        Sampling strategy: 'full_factorial', 'latin_hypercube', 'sobol', 'boundary_focused'
    task : str
        Task type: 'trajectory', 'parameter_estimation', 'cct', 'all'
    n_samples : int, optional
        Number of samples for non-factorial strategies. If None, uses full factorial.
    seed : int, optional
        Random seed for reproducibility
    validate_quality : bool
        Whether to validate data quality (correlation, coverage)

    Returns:
    --------
    pd.DataFrame : Combined dataset from all parameter combinations

    **Stability Labels** (automatically added):
    - `is_stable` (bool): Trajectory-based stability classification
      - True if max(|delta|) < π (rotor angle only; omega-based checks commented out)
      - Based on actual trajectory behavior (primary method)
    - `is_stable_from_cct` (bool, optional): CCT-based stability classification
      - True if clearing_time < CCT (only when CCT-based sampling is used)
      - Based on physics (clearing time relative to CCT)
      - NaN if CCT information not available

    **Note**: Both stability labels are provided when CCT-based sampling is used,
    allowing validation of consistency between trajectory-based and physics-based stability.
    """
    if not ANDES_AVAILABLE:
        raise ImportError("ANDES is not available. Cannot generate parameter sweep.")

    # Validate parameter ranges against realistic power system values
    # Warn if ranges are outside typical bounds but allow execution
    if H_range is not None:
        H_min, H_max = H_range[0], H_range[1]
        if H_min < 1.5 or H_max > 12.0:
            if verbose:
                print(
                    f"[WARNING] H_range ({H_min:.2f} to {H_max:.2f} s) is outside typical bounds "
                    f"(1.5 to 12.0 s). Typical range: 2.0-10.0 s for small to large generators."
                )
        if H_min < 0 or H_max < H_min:
            raise ValueError(
                f"Invalid H_range: min={H_min:.2f}, max={H_max:.2f}. Must have min >= 0 and max > min."
            )

    if D_range is not None:
        D_min, D_max = D_range[0], D_range[1]
        if D_min < 0.3 or D_max > 4.0:
            if verbose:
                print(
                    f"[WARNING] D_range ({D_min:.2f} to {D_max:.2f} pu) is outside typical bounds "
                    f"(0.3 to 4.0 pu). Typical range: 0.5-3.0 pu for low to high damping."
                )
        if D_min < 0 or D_max < D_min:
            raise ValueError(
                f"Invalid D_range: min={D_min:.2f}, max={D_max:.2f}. Must have min >= 0 and max > min."
            )

    if alpha_range is not None:
        alpha_min, alpha_max = alpha_range[0], alpha_range[1]
        if alpha_min < 0.3 or alpha_max > 1.3:
            if verbose:
                print(
                    f"[WARNING] alpha_range ({alpha_min:.2f} to {alpha_max:.2f}) is outside typical bounds "
                    f"(0.3 to 1.3). Typical range: 0.4-1.2 (40%-120% of base load). "
                    f"Values > 1.2 may cause instability even without faults."
                )
        if alpha_min < 0 or alpha_max < alpha_min:
            raise ValueError(
                f"Invalid alpha_range: min={alpha_min:.2f}, max={alpha_max:.2f}. Must have min >= 0 and max > min."
            )

    # Backward compatibility: Convert load_range to alpha_range if needed
    # If alpha_range is not provided but load_range is, we need base load value to convert
    # For now, we'll use load_range directly but treat it as absolute values
    # TODO: In future, get base load from case file and convert load_range to alpha_range
    if alpha_range is None and load_range is not None and use_load_variation:
        # For backward compatibility, treat load_range as absolute values
        # This maintains existing behavior but is deprecated
        if verbose:
            print(
                "[WARNING] Using deprecated load_range parameter. "
                "Please migrate to alpha_range for unified approach."
            )
        # We'll use load_range as-is for now, but mark it for conversion
        # In the simulation loop, we'll need to get base load and convert to alpha
        pass

    # Determine if we're using load variation (alpha_range or legacy load_range)
    use_load_mode = (alpha_range is not None) or (use_load_variation and load_range is not None)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize flag for 3D sampling (will be set in Sobol/LHS section if applicable)
    use_3d_sampling = False

    # Task-specific parameter generation
    if task == "parameter_estimation":
        # For parameter estimation, use decorrelated sampling
        # Parameter estimation uses 2D sampling (H, D only), not 3D
        use_3d_sampling = False

        if sampling_strategy == "full_factorial":
            sampling_strategy = "latin_hypercube"  # Prefer LHS for better decorrelation
        if n_samples is None:
            n_samples = H_range[2] * D_range[2]  # Use similar number of samples

        # Generate decorrelated H-D pairs
        bounds = [H_range[:2], D_range[:2]]
        HD_samples, corr_info = decorrelated_sample(
            n_samples=n_samples,
            bounds=bounds,
            target_correlation=0.0,
            tolerance=0.1,
            seed=seed,
        )

        H_values = HD_samples[:, 0]
        D_values = HD_samples[:, 1]

        if verbose:
            print(f"Parameter estimation mode: Generated {len(H_values)} decorrelated (H, D) pairs")
            print(f"Max H-D correlation: {corr_info['max_correlation']:.4f}")

        # Validate correlation if requested
        if validate_quality:
            validation = validate_sample_quality(
                HD_samples, bounds, min_correlation=-0.3, max_correlation=0.3
            )
            if not validation["correlation_ok"]:
                print(
                    f"Warning: H-D correlation "
                    f"({validation['max_correlation']:.4f}) exceeds target range"
                )

    elif task == "cct" and sampling_strategy == "boundary_focused":
        # For CCT estimation with boundary focus
        # Boundary-focused sampling uses 2D sampling (H, D only), not 3D
        use_3d_sampling = False
        use_3d_load_sampling = False

        if n_samples is None:
            n_samples = H_range[2] * D_range[2]

        bounds = [H_range[:2], D_range[:2]]
        HD_samples = boundary_focused_sample(
            n_samples=n_samples, bounds=bounds, boundary_fraction=0.4, seed=seed
        )

        H_values = HD_samples[:, 0]
        D_values = HD_samples[:, 1]

        if verbose:
            print(f"CCT estimation mode: Generated {len(H_values)} boundary-focused (H, D) pairs")

    elif sampling_strategy in ["latin_hypercube", "sobol"]:
        # Use advanced sampling strategies
        if n_samples is None:
            n_samples = H_range[2] * D_range[2]

        # Check if we should do 3D sampling (H, D, Pm) or (H, D, alpha) or 2D sampling (H, D only)
        # 3D sampling is used when:
        #   - Pm_range is provided (not None) and has a grid size > 1, OR
        #   - alpha_range is provided (for H, D, alpha sampling) OR
        #   - use_load_variation=True and load_range is provided (backward compatibility)
        use_3d_sampling = False
        use_3d_alpha_sampling = False
        use_3d_load_sampling = False  # Initialize for backward compatibility
        alpha_values_3d = None  # Initialize for 3D alpha sampling
        load_values_3d = None  # For backward compatibility

        # Check for alpha_range (new unified approach)
        if alpha_range is not None:
            alpha_min, alpha_max, alpha_n = alpha_range
            if alpha_n > 1:
                use_3d_sampling = True
                use_3d_alpha_sampling = True
        # Backward compatibility: Check for load_range
        elif use_load_variation and load_range is not None:
            # 3D sampling with load variation: (H, D, Load) - DEPRECATED
            load_min, load_max, load_n = load_range
            if load_n > 1:
                use_3d_sampling = True
                use_3d_load_sampling = True
        elif Pm_range is not None:
            # 3D sampling with Pm variation: (H, D, Pm)
            Pm_min, Pm_max, Pm_n = Pm_range
            # Always use 3D sampling for Pm variation (even if Pm_n=1)
            # This ensures proper Sobol/LHS sampling across (H, D, Pm) space
            use_3d_sampling = True

        if use_3d_sampling:
            if use_3d_alpha_sampling:
                # 3D sampling with alpha (unified approach): Sample (H, D, alpha) together for better coverage
                # This ensures exactly n_samples unique (H, D, alpha) combinations
                alpha_min, alpha_max, alpha_n = alpha_range
                bounds = [H_range[:2], D_range[:2], (alpha_min, alpha_max)]

                # Perform 3D sampling with alpha
                if sampling_strategy == "latin_hypercube":
                    HDA_samples = latin_hypercube_sample(n_samples, bounds, seed=seed)
                elif sampling_strategy == "sobol":
                    HDA_samples = sobol_sequence_sample(n_samples, bounds, seed=seed)

                # Filter extreme combinations if requested
                if filter_extreme_combinations:
                    HDA_samples_filtered, n_filtered = filter_extreme_combinations(
                        HDA_samples,
                        bounds,
                        H_idx=0,
                        D_idx=1,
                        Load_idx=2,  # alpha is in Load_idx position
                        verbose=verbose,
                        H_min_absolute=2.0,  # seconds
                        D_min_absolute=0.5,  # pu
                        D_min_critical=0.3,  # pu
                        Load_max_absolute=1.2,  # alpha max (120% of base load)
                    )
                    if n_filtered > 0:
                        if verbose:
                            print(
                                f"[FILTER] Filtered out {n_filtered} extreme combinations. "
                                f"Remaining: {len(HDA_samples_filtered)} samples"
                            )
                        HDA_samples = HDA_samples_filtered
                        if len(HDA_samples) < n_samples * 0.8:
                            if verbose:
                                print(
                                    f"[WARNING] Many samples filtered ({n_filtered}/{n_samples}). "
                                    f"Consider increasing n_samples or adjusting parameter ranges."
                                )

                H_values = HDA_samples[:, 0]
                D_values = HDA_samples[:, 1]
                alpha_values_3d = HDA_samples[:, 2]
                if verbose:
                    print(
                        f"Using {sampling_strategy} sampling (3D with alpha): "
                        f"Generated {len(H_values)} unique (H, D, alpha) combinations"
                    )
            elif use_3d_load_sampling:
                # 3D sampling with load variation: Sample (H, D, Load) together for better coverage (DEPRECATED)
                # This ensures exactly n_samples unique (H, D, Load) combinations
                load_min, load_max, load_n = load_range
                bounds = [H_range[:2], D_range[:2], (load_min, load_max)]

                # 3D sampling with load (backward compatibility)
                if sampling_strategy == "latin_hypercube":
                    HDL_samples = latin_hypercube_sample(n_samples, bounds, seed=seed)
                elif sampling_strategy == "sobol":
                    HDL_samples = sobol_sequence_sample(n_samples, bounds, seed=seed)

                # Filter extreme combinations if requested
                if filter_extreme_combinations:
                    HDL_samples_filtered, n_filtered = filter_extreme_combinations(
                        HDL_samples,
                        bounds,
                        H_idx=0,
                        D_idx=1,
                        Load_idx=2,
                        verbose=verbose,
                        H_min_absolute=2.0,  # seconds
                        D_min_absolute=0.5,  # pu
                        D_min_critical=0.3,  # pu
                        Load_max_absolute=0.9,  # pu
                    )
                    if n_filtered > 0:
                        if verbose:
                            print(
                                f"[FILTER] Filtered out {n_filtered} extreme combinations. "
                                f"Remaining: {len(HDL_samples_filtered)} samples"
                            )
                        HDL_samples = HDL_samples_filtered
                        if len(HDL_samples) < n_samples * 0.8:
                            if verbose:
                                print(
                                    f"[WARNING] Many samples filtered ({n_filtered}/{n_samples}). "
                                    f"Consider increasing n_samples or adjusting parameter ranges."
                                )

                H_values = HDL_samples[:, 0]
                D_values = HDL_samples[:, 1]
                load_values_3d = HDL_samples[:, 2]
                if verbose:
                    print(
                        f"Using {sampling_strategy} sampling (3D with load): "
                        f"Generated {len(H_values)} unique (H, D, Load) combinations"
                    )

                # Validate quality if requested (only for H-D correlation, not including Load)
                if validate_quality and task == "parameter_estimation":
                    HD_samples = HDL_samples[:, :2]  # Extract H-D pairs for validation
                    validation = validate_sample_quality(
                        HD_samples,
                        bounds[:2],
                        min_correlation=-0.3,
                        max_correlation=0.3,
                    )
                    if verbose:
                        print(
                            f"Data quality - Max correlation: {validation['max_correlation']:.4f}, "
                            f"Coverage: {validation['coverage_score']:.4f}"
                        )
            else:
                # 3D sampling with Pm variation: Sample (H, D, Pm) together for better coverage
                # This ensures exactly n_samples unique (H, D, Pm) combinations
                bounds = [H_range[:2], D_range[:2], (Pm_min, Pm_max)]

                if sampling_strategy == "latin_hypercube":
                    HDP_samples = latin_hypercube_sample(n_samples, bounds, seed=seed)
                elif sampling_strategy == "sobol":
                    HDP_samples = sobol_sequence_sample(n_samples, bounds, seed=seed)

                # Filter extreme combinations if requested
                if filter_extreme_combinations:
                    HDP_samples_filtered, n_filtered = filter_extreme_combinations(
                        HDP_samples,
                        bounds,
                        H_idx=0,
                        D_idx=1,
                        Load_idx=2,  # Use Pm as proxy for Load
                        verbose=verbose,
                        # Priority 1: Use absolute thresholds based on physical reality
                        H_min_absolute=2.0,  # seconds
                        D_min_absolute=0.5,  # pu
                        D_min_critical=0.3,  # pu
                        Load_max_absolute=0.9,  # pu
                    )
                    if n_filtered > 0:
                        if verbose:
                            print(
                                f"[FILTER] Filtered out {n_filtered} extreme combinations. "
                                f"Remaining: {len(HDP_samples_filtered)} samples"
                            )
                        HDP_samples = HDP_samples_filtered
                        # If too many samples were filtered, generate more to compensate
                        if len(HDP_samples) < n_samples * 0.8:  # Less than 80% remaining
                            if verbose:
                                print(
                                    f"[WARNING] Many samples filtered ({n_filtered}/{n_samples}). "
                                    f"Consider increasing n_samples or adjusting parameter ranges."
                                )

                H_values = HDP_samples[:, 0]
                D_values = HDP_samples[:, 1]
                Pm_values = HDP_samples[:, 2]

                if verbose:
                    print(
                        f"Using {sampling_strategy} sampling (3D with Pm): "
                        f"Generated {len(H_values)} unique (H, D, Pm) combinations"
                    )

                # Validate quality if requested (only for H-D correlation, not including Pm)
                if validate_quality and task == "parameter_estimation":
                    HD_samples = HDP_samples[:, :2]  # Extract H-D pairs for validation
                    validation = validate_sample_quality(
                        HD_samples,
                        bounds[:2],
                        min_correlation=-0.3,
                        max_correlation=0.3,
                    )
                    if verbose:
                        print(
                            f"Data quality - Max correlation: {validation['max_correlation']:.4f}, "
                            f"Coverage: {validation['coverage_score']:.4f}"
                        )
        else:
            # 2D sampling: Sample (H, D) only, Pm will be handled separately
            # When count is 1 for both H and D, use single (H,D) from case/default — no sampling
            if H_range[2] == 1 and D_range[2] == 1:
                H_values = np.linspace(H_range[0], H_range[1], 1)
                D_values = np.linspace(D_range[0], D_range[1], 1)
                if verbose:
                    print("Using single (H, D) from case/default (count=1); no (H,D) sampling.")
            else:
                bounds = [H_range[:2], D_range[:2]]

                if sampling_strategy == "latin_hypercube":
                    HD_samples = latin_hypercube_sample(n_samples, bounds, seed=seed)
                elif sampling_strategy == "sobol":
                    HD_samples = sobol_sequence_sample(n_samples, bounds, seed=seed)

                H_values = HD_samples[:, 0]
                D_values = HD_samples[:, 1]

                if verbose:
                    print(
                        f"Using {sampling_strategy} sampling (2D): "
                        f"Generated {len(H_values)} (H, D) pairs"
                    )

                # Validate quality if requested
                if validate_quality and task == "parameter_estimation":
                    validation = validate_sample_quality(
                        HD_samples, bounds, min_correlation=-0.3, max_correlation=0.3
                    )
                    if verbose:
                        print(
                            f"Data quality - Max correlation: {validation['max_correlation']:.4f}, "
                            f"Coverage: {validation['coverage_score']:.4f}"
                        )

    else:
        # Default: full factorial (original behavior)
        # Full factorial uses 2D sampling (H, D only), not 3D
        use_3d_sampling = False
        use_3d_load_sampling = False

        H_values = np.linspace(H_range[0], H_range[1], H_range[2])
        D_values = np.linspace(D_range[0], D_range[1], D_range[2])

    # Convert H to M (M = 2*H for 60 Hz system)
    M_values = 2.0 * H_values

    # Generate Pm values if Pm_range is provided
    # Note: If 3D sampling was used above, Pm_values is already set from the 3D samples
    if not use_3d_sampling:
        if Pm_range is not None:
            Pm_min, Pm_max, Pm_n = Pm_range
            Pm_values = np.linspace(Pm_min, Pm_max, Pm_n)
            if Pm is not None:
                if verbose:
                    print(
                        f"Warning: Both Pm_range and Pm provided. "
                        f"Pm_range will be used, Pm={Pm} will be ignored."
                    )
        else:
            # Use single Pm value (either provided or will be extracted later)
            Pm_values = [Pm] if Pm is not None else [None]  # None will trigger extraction
    # else: Pm_values already set from 3D sampling above

    # Generate alpha values if alpha_range is provided (NEW: Unified load variation)
    # NOTE: 3D sampling = (H, D, alpha) only. Q is scaled by same alpha (maintains power factor).
    # Industrial standard: Uniform (P,Q) scaling maintains power factor.
    alpha_values = None
    load_values = None  # For backward compatibility
    load_q_values = None
    if alpha_range is not None:
        # If 3D sampling was used with alpha, alpha_values_3d is already set
        if use_3d_sampling and use_3d_alpha_sampling:
            # Use the 3D sampled alpha values (H, D, alpha)
            alpha_values = alpha_values_3d
            if verbose:
                print(
                    f"Using 3D sampled alpha variation: {len(alpha_values)} alpha values "
                    f"(range: {alpha_min:.3f} - {alpha_max:.3f})"
                )
                print(
                    f"  NOTE: Q will be scaled by same alpha (uniform scaling, maintains power factor)."
                )
        else:
            # Full factorial or 2D sampling: generate alpha values linearly
            alpha_min, alpha_max, alpha_n = alpha_range
            if alpha_n == 1:
                # Count=1: use default load level from case (no scaling, alpha=1.0)
                alpha_values = np.array([1.0])
                if verbose:
                    print("Using single alpha (count=1): default load level from case (alpha=1.0).")
            else:
                alpha_values = np.linspace(alpha_min, alpha_max, alpha_n)
                if verbose:
                    print(
                        f"Using full factorial alpha variation: {len(alpha_values)} alpha values "
                        f"(range: {alpha_min:.3f} - {alpha_max:.3f})"
                    )
                    print(
                        f"  NOTE: Q will be scaled by same alpha (uniform scaling, maintains power factor)."
                    )
        # When using alpha variation, Pm will be determined by power flow
        # (generator adjusts to meet scaled load)
        Pm_values = [None] * len(alpha_values)  # Will be extracted after power flow
    elif use_load_variation and load_range is not None:
        # If 3D sampling was used with load, load_values_3d is already set
        if use_3d_sampling and use_3d_load_sampling:
            # Use the 3D sampled load values (H, D, P only - NOT including Q)
            load_values = load_values_3d

            # Generate Q values (fixed or from range)
            # IMPORTANT: Q is NOT included in 3D sampling - it's handled separately
            # This means Q is either fixed (typically 0) or sampled independently (not correlated with H, D, P)
            if load_q_range is not None:
                q_min, q_max, q_n = load_q_range
                # For 3D sampling, we can either:
                # 1. Use fixed Q for all samples (recommended, matches industrial standard)
                # 2. Sample Q independently (NOT ideal - would need 4D sampling for proper correlation)
                # For now, use fixed Q or sample Q independently
                if q_n > 1:
                    # Sample Q independently (NOT correlated with H, D, P - not ideal but acceptable)
                    # NOTE: This is NOT true 4D sampling - Q is sampled independently
                    q_samples = np.linspace(q_min, q_max, q_n)
                    # Repeat Q samples to match load_values length
                    load_q_values = np.tile(q_samples, (len(load_values) // q_n + 1))[
                        : len(load_values)
                    ]
                else:
                    load_q_values = np.full(len(load_values), q_min)
            else:
                # Use fixed Q (typically 0 for SMIB - matches industrial standard)
                load_q_values = np.zeros(len(load_values))

            if verbose:
                q_status = (
                    "Q varies independently"
                    if load_q_range is not None and load_q_range[2] > 1
                    else "Q=0 (fixed)"
                )
                print(
                    f"Using 3D sampled load variation: {len(load_values)} load levels "
                    f"(P: {load_range[0]:.3f} - {load_range[1]:.3f} pu, {q_status})"
                )
                print(
                    f"  NOTE: 3D sampling = (H, D, P) only. Q is handled separately. "
                    f"For true 4D sampling (H, D, P, Q), would need extended implementation."
                )
        else:
            # Full factorial or 2D sampling: generate load values linearly
            load_min, load_max, load_n = load_range
            load_values = np.linspace(load_min, load_max, load_n)

            if load_q_range is not None:
                q_min, q_max, q_n = load_q_range
                load_q_values = np.linspace(q_min, q_max, q_n)
            else:
                # Use fixed Q (typically 0 for SMIB)
                load_q_values = np.zeros(len(load_values))

            if verbose:
                q_status = (
                    f"Q: {load_q_range[0]:.3f} - {load_q_range[1]:.3f} pu"
                    if load_q_range is not None
                    else "Q=0 (fixed)"
                )
                print(
                    f"Using full factorial load variation: {len(load_values)} load levels "
                    f"(P: {load_min:.3f} - {load_max:.3f} pu, {q_status})"
                )
                print(
                    f"  NOTE: Q is handled separately from (H, D, P) combinations. "
                    f"Industrial standard: Vary P only, keep Q=0 for SMIB transient stability."
                )

        # When using load variation, Pm will be determined by power flow
        # (generator adjusts to meet load)
        Pm_values = [None] * len(load_values)  # Will be extracted after power flow
    elif use_load_variation and load_range is None:
        raise ValueError(
            "use_load_variation=True requires load_range to be provided. "
            "Please specify load_range=(min, max, n_points)."
        )

    # For full factorial with fault_clearing_times, use all combinations
    # For other strategies, we'll use the sampled H-D pairs with all fault_clearing_times

    # Get case file path
    if not os.path.isabs(case_file):
        case_path = andes.get_case(case_file)
    else:
        case_path = case_file

    if not os.path.exists(case_path):
        raise FileNotFoundError("Case file not found: {case_path}")

    # Resolve addfile path (e.g. PSS/E DYR for GENCLS when case_file is .raw)
    addfile_path = None
    if addfile:
        addfile_path = andes.get_case(addfile) if not os.path.isabs(addfile) else addfile
        if not os.path.exists(addfile_path):
            addfile_path = None  # ignore if missing

    # CRITICAL: Get default values from case file when H_range or D_range are None
    # This allows using case file defaults instead of function defaults
    default_pm = None
    default_m = None
    default_d = None
    default_h = None

    if CASE_FILE_MODIFIER_AVAILABLE:
        try:
            default_pm = get_default_pm_from_case_file(case_path, gen_idx=0)
            if verbose and default_pm is not None:
                print(f"[CASE FILE] Default Pm from case file: {default_pm:.6f} pu")
        except Exception as e:
            if verbose:
                print(f"[WARNING] Could not get default Pm from case file: {e}")

        # Extract default M and D if H_range or D_range are None
        if H_range is None:
            try:
                default_m = get_default_m_from_case_file(case_path, gen_idx=0)
                if default_m is not None:
                    # Convert M to H (H = M/2 for 60 Hz systems) for internal processing
                    default_h = default_m / 2.0
                    if verbose:
                        print(
                            f"[CASE FILE] Default M from case file: {default_m:.6f} s (H = {default_h:.6f} s)"
                        )
                    # Use single value: create a range with one point
                    H_range = (default_h, default_h, 1)
                else:
                    if verbose:
                        print(
                            f"[WARNING] Could not get default M from case file, using function default"
                        )
                    # Fallback: Use default M range (4.0 to 20.0) converted to H (2.0 to 10.0)
                    H_range = (2.0, 10.0, 5)  # Fallback to function default
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Could not get default M from case file: {e}")
                H_range = (2.0, 10.0, 5)  # Fallback to function default

        if D_range is None:
            try:
                default_d = get_default_d_from_case_file(case_path, gen_idx=0)
                if default_d is not None:
                    if verbose:
                        print(f"[CASE FILE] Default D from case file: {default_d:.6f} pu")
                    # Use single value: create a range with one point
                    D_range = (default_d, default_d, 1)
                else:
                    if verbose:
                        print(
                            f"[WARNING] Could not get default D from case file, using function default"
                        )
                    D_range = (0.5, 3.0, 5)  # Fallback to function default
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Could not get default D from case file: {e}")
                D_range = (0.5, 3.0, 5)  # Fallback to function default
    else:
        # Case file modifier not available - use function defaults
        if H_range is None:
            H_range = (2.0, 10.0, 5)
        if D_range is None:
            D_range = (0.5, 3.0, 5)

    # Store original case path
    original_case_path = case_path

    # Flag to suppress repeated expected warnings (case file modification fallback)
    _case_modify_warning_shown = False

    # NEW: Check for load device and add if missing (for load variation, including alpha_range)
    if use_load_mode and CASE_FILE_MODIFIER_AVAILABLE and check_smib_has_load is not None:
        has_load, load_model_type, load_idx = check_smib_has_load(case_path)
        if not has_load:
            # No load device found - add one automatically
            if verbose:
                print(
                    f"[LOAD CHECK] No load device found in case file. "
                    f"Adding PQ load to bus 3 with default values (P=0.5 pu, Q=0.2 pu)."
                )
            try:
                if add_pq_load_to_smib_case is not None:
                    case_path = add_pq_load_to_smib_case(
                        case_path=case_path,
                        bus_idx=3,  # Default bus for SMIB
                        p0=0.5,  # Default active power (matches ANDES manual strategy)
                        q0=0.2,  # Default reactive power (matches ANDES manual strategy)
                    )
                    original_case_path = case_path  # Update original path
                    if verbose:
                        print(f"[LOAD CHECK] Added PQ load to case file: {case_path}")
                else:
                    if verbose:
                        print(
                            f"[WARNING] add_pq_load_to_smib_case not available. "
                            f"Load variation may fail if case has no load device."
                        )
            except Exception as e:
                if verbose:
                    print(
                        f"[WARNING] Could not add load to case file: {e}. "
                        f"Load variation may fail if case has no load device."
                    )
        else:
            if verbose:
                print(
                    f"[LOAD CHECK] Found existing {load_model_type} load device "
                    f"(index {load_idx}). Will modify using ANDES alter() method."
                )

    # Default fault location (bus 3 for SMIB)
    if fault_locations is None:
        fault_locations = [3]

    # Validate CCT-based sampling requirements FIRST (before handling fault_clearing_times)
    if use_cct_based_sampling:
        if not CCT_FINDER_AVAILABLE:
            raise ImportError(
                "CCT finder not available. Cannot use CCT-based sampling. "
                "Please ensure andes_utils.cct_finder is available."
            )
        if task != "trajectory":
            if verbose:
                print(
                    "Warning: CCT-based sampling is only supported for trajectory task. Disabling."
                )
            use_cct_based_sampling = False

    # Handle fault_clearing_times based on sampling mode
    # When use_cct_based_sampling=True: clearing times = CCT ± offsets only; no fallback.
    #   If CCT is not found for an operating point, that point is skipped (no data).
    # When use_cct_based_sampling=False: fault_clearing_times is REQUIRED
    if not use_cct_based_sampling:
        # Fixed clearing times mode: fault_clearing_times is REQUIRED
        if fault_clearing_times is None:
            # Set default: absolute clearing times (fault_start_time + duration)
            default_durations = [0.15, 0.18, 0.20, 0.22, 0.25]  # Durations in seconds
            fault_clearing_times = [fault_start_time + d for d in default_durations]

        # Validate clearing times (following smib_albert_cct.py validation approach)
        # All clearing times must be absolute times >= fault_start_time
        if fault_clearing_times is not None and len(fault_clearing_times) > 0:
            invalid_times = [tc for tc in fault_clearing_times if tc < fault_start_time]
            if invalid_times:
                raise ValueError(
                    f"Invalid clearing times: {invalid_times}. "
                    f"All clearing times must be >= fault_start_time ({fault_start_time:.3f}s). "
                    f"Clearing times are ABSOLUTE times (not durations). "
                    f"If you provided durations, add fault_start_time to each value."
                )

            # Warn if any clearing times are too close to simulation end
            too_late = [tc for tc in fault_clearing_times if tc >= simulation_time - 0.5]
            if too_late and verbose:
                print(
                    f"Warning: Some clearing times {too_late} are too close to "
                    f"simulation end ({simulation_time:.3f}s). "
                    f"System may not have enough time to recover. "
                    f"Recommendation: clearing_time <= simulation_time - 0.5"
                )
    else:
        # CCT-based sampling mode: no fallback. Clearing times = CCT ± offsets only.
        # Operating points for which CCT is not found are skipped (no data).
        if verbose:
            print(
                "[INFO] CCT-based sampling: clearing times = CCT ± offsets only. "
                "Operating points where CCT is not found use fault_clearing_times as fallback (if in config), else skipped."
            )

    # Handle pairing of M, D, and operating point values based on sampling strategy
    # NEW: Support both Pm variation and load variation (including alpha_range)
    if use_load_mode:
        # Load variation mode: Create (M, D, Load) or (M, D, alpha) combinations
        # Pm will be determined by power flow (generator adjusts to meet load)
        if len(M_values) != len(D_values):
            raise ValueError(
                f"M_values and D_values must have same length for {sampling_strategy} sampling. "
                f"Got {len(M_values)} and {len(D_values)}"
            )

        # Check if using alpha (new unified approach) or load (backward compatibility)
        if alpha_values is not None:
            # Using alpha_range: Create (M, D, alpha) combinations
            # Load_P will be computed from alpha × base_load
            if use_3d_sampling and use_3d_alpha_sampling:
                # 3D sampling already provided (H, D, alpha) triplets - just convert H to M
                if len(M_values) != len(alpha_values):
                    raise ValueError(
                        f"M_values, D_values, and alpha_values must have same length "
                        f"for 3D {sampling_strategy} sampling with alpha variation. "
                        f"Got {len(M_values)}, {len(D_values)}, {len(alpha_values)}"
                    )
                # Create (M, D, alpha) combinations - use special format for alpha
                # Store alpha as 3rd element, None as 4th to indicate alpha mode
                M_D_Load_combinations = [
                    (
                        M,
                        D,
                        alpha,
                        None,
                    )  # alpha stored as 3rd element, None indicates alpha mode
                    for M, D, alpha in zip(M_values, D_values, alpha_values)
                ]
                if verbose:
                    print(
                        f"Using 3D sampled alpha variation: {len(M_D_Load_combinations)} (M, D, alpha)"
                        f"combinations"
                    )
            else:
                # Full factorial or 2D sampling: Create all combinations
                M_D_pairs = list(zip(M_values, D_values))
                M_D_Load_combinations = [
                    (
                        M,
                        D,
                        alpha,
                        None,
                    )  # alpha stored as 3rd element, None indicates alpha mode
                    for M, D in M_D_pairs
                    for alpha in alpha_values
                ]
                if verbose:
                    print(
                        f"[DEBUG] Using alpha variation (full factorial): "
                        f"{len(M_D_Load_combinations)} (M, D, alpha) combinations. "
                        f"alpha_values={alpha_values}, M_values={len(M_values)}, D_values={len(D_values)}"
                    )
            # Store alpha values for later use (will be paired with M, D in the loop)
            # We'll need to zip alpha_values with M_D_pairs
            if use_3d_sampling and use_3d_alpha_sampling:
                # Already paired from 3D sampling
                alpha_values_paired = alpha_values
            else:
                # Need to pair alpha with each (M, D) pair
                alpha_values_paired = [
                    alpha for M, D in zip(M_values, D_values) for alpha in alpha_values
                ]
        elif use_3d_sampling and use_3d_load_sampling:
            # 3D sampling already provided (H, D, Load) triplets - just convert H to M
            # M_values, D_values, and load_values are already paired correctly from 3D sampling
            if len(M_values) != len(load_values):
                raise ValueError(
                    f"M_values, D_values, and load_values must have same length "
                    f"for 3D {sampling_strategy} sampling with load variation. "
                    f"Got {len(M_values)}, {len(D_values)}, {len(load_values)}"
                )

            # Create combinations directly from 3D samples
            if load_q_values is not None and len(load_q_values) == len(load_values):
                # Q values provided
                M_D_Load_combinations = [
                    (M, D, load_p, load_q)
                    for M, D, load_p, load_q in zip(M_values, D_values, load_values, load_q_values)
                ]
            else:
                # Fixed Q (typically 0)
                M_D_Load_combinations = [
                    (
                        M,
                        D,
                        load_p,
                        (
                            load_q_values[0]
                            if load_q_values is not None and len(load_q_values) > 0
                            else 0.0
                        ),
                    )
                    for M, D, load_p in zip(M_values, D_values, load_values)
                ]

            if verbose:
                print(
                    f"Using 3D sampled load variation: {len(M_D_Load_combinations)} (M, D, Load)"
                    f"combinations"
                )
        else:
            # Full factorial or 2D sampling: Create all combinations
            # Create combinations: (M, D, Load_P, Load_Q)
            M_D_pairs = list(zip(M_values, D_values))
            if load_q_values is not None and len(load_q_values) == len(load_values):
                # Both P and Q vary
                M_D_Load_combinations = [
                    (M, D, load_p, load_q)
                    for M, D in M_D_pairs
                    for load_p, load_q in zip(load_values, load_q_values)
                ]
            else:
                # Only P varies, Q is fixed (typically 0)
                M_D_Load_combinations = [
                    (
                        M,
                        D,
                        load_p,
                        (
                            load_q_values[0]
                            if load_q_values is not None and len(load_q_values) > 0
                            else 0.0
                        ),
                    )
                    for M, D in M_D_pairs
                    for load_p in load_values
                ]

            if verbose:
                print(
                    f"Using load variation (full factorial): {len(M_D_Load_combinations)} (M, D,"
                    f"Load) combinations"
                )

        # For compatibility, create M_D_Pm_combinations with None for Pm
        # (Pm will be extracted after power flow)
        M_D_Pm_combinations = [(M, D, None) for M, D, _, _ in M_D_Load_combinations]
    elif use_3d_sampling:
        # 3D sampling already provided (H, D, Pm) triplets - just convert H to M
        # M_values, D_values, and Pm_values are already paired correctly from 3D sampling
        if len(M_values) != len(D_values) or len(M_values) != len(Pm_values):
            raise ValueError(
                f"M_values, D_values, and Pm_values must have same length "
                f"for 3D {sampling_strategy} sampling. "
                f"Got {len(M_values)}, {len(D_values)}, {len(Pm_values)}"
            )
        M_D_Pm_combinations = list(zip(M_values, D_values, Pm_values))
        M_D_Load_combinations = None  # Not using load variation
        if verbose:
            print(
                f"Using 3D sampled (M, D, Pm) combinations: {len(M_D_Pm_combinations)} unique"
                f"triplets"
            )
    elif sampling_strategy == "full_factorial" and task != "parameter_estimation":
        # Full factorial: use all combinations (original behavior)
        if Pm_range is not None:
            # Create all combinations of (M, D, Pm)
            M_D_Pm_combinations = list(itertools.product(M_values, D_values, Pm_values))
        else:
            # Original behavior: only (M, D) pairs, Pm is fixed or extracted
            M_D_Pm_combinations = [
                (M, D, Pm_values[0]) for M, D in itertools.product(M_values, D_values)
            ]
        M_D_Load_combinations = None  # Not using load variation
    else:
        # For other strategies, M and D should be paired
        if len(M_values) != len(D_values):
            raise ValueError(
                f"M_values and D_values must have same length for {sampling_strategy} sampling. "
                f"Got {len(M_values)} and {len(D_values)}"
            )
        # Pair up M and D values, then add Pm
        if use_3d_sampling:
            # For 3D sampling, M, D, and Pm are already paired from the 3D Sobol/LHS samples
            # Just zip them together directly
            M_D_Pm_combinations = list(zip(M_values, D_values, Pm_values))
        elif Pm_range is not None:
            # For 2D sampling with Pm_range, create cartesian product
            # For each (M, D) pair, create combinations with all Pm values
            M_D_pairs = list(zip(M_values, D_values))
            M_D_Pm_combinations = [(M, D, Pm) for M, D in M_D_pairs for Pm in Pm_values]
        else:
            # Original behavior: only (M, D) pairs, Pm is fixed or extracted
            M_D_pairs = list(zip(M_values, D_values))
            M_D_Pm_combinations = [(M, D, Pm_values[0]) for M, D in M_D_pairs]
        M_D_Load_combinations = None  # Not using load variation

    # Store all datasets
    all_datasets = []

    # Calculate total combinations (will be updated if using CCT-based sampling)
    if use_cct_based_sampling:
        # For CCT-based sampling, we'll generate n_samples_per_combination per (M, D, Pm) combination
        total_combinations = (
            len(M_D_Pm_combinations) * n_samples_per_combination * len(fault_locations)
        )
    else:
        total_combinations = (
            len(M_D_Pm_combinations) * len(fault_clearing_times) * len(fault_locations)
        )

    current_combination = 0

    # Track errors for reporting (even when verbose=False)
    error_summary = {
        "no_gencls": 0,
        "gen_param_failed": 0,
        "fault_config_failed": 0,
        "powerflow_failed": 0,
        "powerflow_no_converge": 0,
        "pm_mismatch_after_pf": 0,  # NEW: Pm mismatch after power flow
        "tds_failed": 0,
        "data_extraction_failed": 0,
        "cct_finding_failed": 0,
        "other_errors": 0,
        "first_tds_error": None,  # Store first TDS error for debugging
        "first_tds_error_params": None,  # Store parameters for first TDS error
    }

    # Track data quality metrics
    data_quality_metrics = {
        "total_scenarios": 0,
        "successful_scenarios": 0,
        "rejected_scenarios": 0,
        "power_flow_convergence_rate": 0.0,
        "pm_verification_pass_rate": 0.0,
        "rejection_reasons": {},  # Dict to track reasons for rejection
    }

    # Initialize progress tracker if available
    progress_tracker = None
    if PROGRESS_TRACKER_AVAILABLE and verbose:
        param_ranges = {
            "H": H_range,
            "D": D_range,
        }
        if Pm_range is not None:
            param_ranges["Pm"] = Pm_range
        # When using CCT-based sampling, total_combinations already includes n_samples_per_combination
        # So we use it directly without multiplying again
        total_samples_for_tracker = total_combinations
        progress_tracker = DataGenerationTracker(total_samples_for_tracker, param_ranges)
        progress_tracker.start()
        print("=" * 70)
        print("Generating Trajectory Data")
        print("=" * 70)
        if use_cct_based_sampling:
            n_param_combos = len(M_D_Pm_combinations)
            n_fault_locs = len(fault_locations)
            print(
                f"Target: {n_param_combos} parameter combinations × "
                f"{n_samples_per_combination} trajectories × "
                f"{n_fault_locs} fault location(s) = "
                f"{total_samples_for_tracker} total samples"
            )
        else:
            print(f"Target samples: {total_combinations}")
        print()

    if verbose:
        if use_cct_based_sampling:
            n_param_combos = len(M_D_Pm_combinations)
            print(f"Generating {n_param_combos} unique parameter combinations...")
            print(f"  (Total trajectory simulations: {total_combinations})")
        else:
            print(f"Generating {total_combinations} parameter combinations...")
        print(f"H range: {H_range[0]:.2f} to {H_range[1]:.2f} ({len(H_values)} unique values)")
        print(f"D range: {D_range[0]:.2f} to {D_range[1]:.2f} ({len(D_values)} unique values)")
        if Pm_range is not None:
            print(
                f"Pm range: {Pm_range[0]:.2f} to {Pm_range[1]:.2f} ({len(Pm_values)} unique values)"
            )
        else:
            pm_str = f"fixed at {Pm}" if Pm is not None else "will be extracted from system"
            print(f"Pm: {pm_str}")
        if use_cct_based_sampling:
            print(f"CCT-based sampling: ENABLED")
            print(f"  - Samples per combination: {n_samples_per_combination}")
            print(f"  - Will find CCT for each (M, D, Pm) combination and sample around it")
        else:
            print(f"Fault clearing times: {fault_clearing_times}")
        print(f"Fault locations: {fault_locations}")
        print(f"Task: {task}, Strategy: {sampling_strategy}")

    # Sweep through all parameter combinations
    # Main loop: Iterate over parameter combinations
    # NEW: Support both Pm variation and load variation (including alpha_range)
    if use_load_mode and M_D_Load_combinations is not None:
        # Load variation mode: Iterate over (M, D, Load_P, Load_Q) or (M, D, alpha, None) combinations
        param_combinations = M_D_Load_combinations
        if verbose and DEBUG_TDS:
            print(
                f"[DEBUG] Using load variation mode: {len(param_combinations)} combinations. "
                f"use_load_mode={use_load_mode}, M_D_Load_combinations is not None"
            )
        # use_load_mode is already set correctly above (line 720)
    else:
        # Pm variation mode: Iterate over (M, D, Pm) combinations
        param_combinations = M_D_Pm_combinations
        use_load_mode = False  # Ensure it's False for Pm variation mode
        if verbose and DEBUG_TDS:
            print(
                f"[DEBUG] Using Pm variation mode: {len(param_combinations)} combinations. "
                f"use_load_mode={use_load_mode}, M_D_Load_combinations={M_D_Load_combinations}"
            )

    _tm0_corrected_logged = [False]  # print TDS tm0 correction once per run
    for param_combo in param_combinations:
        if use_load_mode:
            # Check if using alpha (new unified approach) or Load_P (backward compatibility)
            # Alpha combinations: (M, D, alpha, None) where 4th element is None
            # Load combinations: (M, D, Load_P, Load_Q) where 4th element is not None
            if len(param_combo) == 4 and param_combo[3] is None and alpha_values is not None:
                # Alpha variation: param_combo = (M, D, alpha, None)
                M, D, alpha, _ = param_combo
                Load_P = None  # Will be computed from base load and alpha
                Load_Q = None  # Will be computed from base load and alpha
            elif len(param_combo) == 3 and alpha_values is not None:
                # Alpha variation (3-element format): param_combo = (M, D, alpha)
                M, D, alpha = param_combo
                Load_P = None  # Will be computed from base load and alpha
                Load_Q = None  # Will be computed from base load and alpha
            else:
                # Load variation (backward compatibility): param_combo = (M, D, Load_P, Load_Q)
                M, D, Load_P, Load_Q = param_combo
                alpha = None
            Pm_val = None  # Pm will be determined by power flow
        else:
            # Pm variation: param_combo = (M, D, Pm_val)
            M, D, Pm_val = param_combo
            Load_P = None
            Load_Q = None
            alpha = None
        # Initialize CCT information for this (M, D, Pm) combination (will be set if CCT-based sampling is used)
        cct_info_for_this_pair = {
            "cct_absolute": None,
            "cct_duration": None,
            "cct_uncertainty": None,
            "small_delta": None,
        }

        # Determine Pm value to use for this (M, D, Pm) combination
        # If Pm_val is provided (from range or fixed), use it; otherwise extract from system or use default
        Pm_to_use = Pm_val

        # For load variation mode, we need to run power flow first to get the actual Pm
        # For CCT finding, we'll use Load_P to estimate Pm (Pm approx Load_P + losses)
        if use_load_mode:
            # In load variation mode, Pm will be extracted after power flow
            # For CCT finding, we can estimate Pm approx Load_P + losses (typically 3% losses)
            # CRITICAL: If using alpha, compute Load_P from base load BEFORE CCT finding
            # For multimachine (addfile set), use case-native load so CCT matches trajectory operating point.
            if Load_P is None and alpha is not None:
                base_p = None
                base_q = None

                # Multimachine (raw+dyr): prefer case-native load so CCT and trajectory use same load
                if addfile_path:
                    try:
                        temp_ss = andes.load(
                            case_path,
                            setup=False,
                            no_output=True,
                            default_config=True,
                            **(dict(addfile=addfile_path) if addfile_path else {}),
                        )
                        temp_ss.setup()
                        if hasattr(temp_ss, "PQ") and temp_ss.PQ.n > 0:
                            p0 = temp_ss.PQ.p0.v
                            total_p = float(np.sum(p0)) if hasattr(p0, "__len__") else float(p0)
                            first_p = (
                                float(p0[0])
                                if hasattr(p0, "__getitem__") and len(p0) > 0
                                else total_p
                            )
                            base_p = first_p if first_p > 0 else total_p
                            if hasattr(temp_ss.PQ, "q0") and hasattr(temp_ss.PQ.q0, "v"):
                                q0 = temp_ss.PQ.q0.v
                                base_q = (
                                    float(q0[0])
                                    if hasattr(q0, "__getitem__") and len(q0) > 0
                                    else 0.2
                                )
                            else:
                                base_q = 0.2
                            if base_p and base_p > 0 and verbose:
                                print(
                                    f"[DEBUG] CCT: using case-native load (multimachine): "
                                    f"p0={base_p:.6f} pu so CCT matches trajectory."
                                )
                        del temp_ss
                    except Exception as e:
                        if verbose:
                            print(
                                f"[DEBUG] CCT: could not read case-native load: {e}. Using config/case file."
                            )
                        base_p = None
                        base_q = None

                # If not multimachine or case load failed: config (highest priority) then case file
                if base_p is None or base_p == 0.0:
                    if base_load is not None and isinstance(base_load, dict):
                        base_p = base_load.get("Pload", base_load.get("P", None))
                        base_q = base_load.get("Qload", base_load.get("Q", None))
                        if base_p is not None and base_p > 0:
                            if verbose:
                                print(
                                    f"[DEBUG] Using base load from config: p0={base_p:.6f} pu, "
                                    f"q0={base_q or 0:.6f} pu"
                                )
                        else:
                            base_p = None
                            base_q = None

                # If not provided in config, try to read from case file
                if base_p is None or base_p == 0.0:
                    if CASE_FILE_MODIFIER_AVAILABLE and get_default_load_from_case_file is not None:
                        try:
                            result = get_default_load_from_case_file(
                                case_path, load_idx=0, load_model="PQ"
                            )
                            if result is not None:
                                base_p, base_q = result
                                # Check if base_p is valid (> 0)
                                if base_p is not None and base_p > 0:
                                    if verbose:
                                        q_str = f"{base_q:.6f}" if base_q is not None else "N/A"
                                        print(
                                            f"[DEBUG] Read base load from case file: p0={base_p:.6f} pu, "
                                            f"q0={q_str} pu"
                                        )
                                else:
                                    if verbose:
                                        print(
                                            f"[WARNING] Base load p0={base_p} is zero or invalid. "
                                            f"Trying fallback method..."
                                        )
                                    base_p = None  # Try fallback
                        except Exception as e:
                            if verbose:
                                print(
                                    f"[WARNING] Could not read base load from case file: {e}. "
                                    f"Trying fallback method..."
                                )

                # Fallback: Load temporary system to get base load value
                if base_p is None or base_p == 0.0:
                    try:
                        temp_ss = andes.load(
                            case_path,
                            setup=False,
                            no_output=True,
                            default_config=True,
                            **(dict(addfile=addfile_path) if addfile_path else {}),
                        )
                        temp_ss.setup()
                        if hasattr(temp_ss, "PQ") and temp_ss.PQ.n > 0:
                            base_p = (
                                float(temp_ss.PQ.p0.v[0])
                                if hasattr(temp_ss.PQ, "p0")
                                and hasattr(temp_ss.PQ.p0, "v")
                                and hasattr(temp_ss.PQ.p0.v, "__getitem__")
                                and len(temp_ss.PQ.p0.v) > 0
                                else 0.0
                            )
                            base_q = (
                                float(temp_ss.PQ.q0.v[0])
                                if hasattr(temp_ss.PQ, "q0")
                                and hasattr(temp_ss.PQ.q0, "v")
                                and hasattr(temp_ss.PQ.q0.v, "__getitem__")
                                and len(temp_ss.PQ.q0.v) > 0
                                else 0.0
                            )
                            if base_p > 0:
                                if verbose:
                                    print(
                                        f"[DEBUG] Read base load from temporary system: "
                                        f"p0={base_p:.6f} pu, q0={base_q:.6f} pu"
                                    )
                        del temp_ss
                    except Exception as e:
                        if verbose:
                            print(
                                f"[WARNING] Could not compute Load_P from alpha: {e}. "
                                f"Case path: {case_path}. Using fallback."
                            )
                        base_p = None

                # Final check: if base_p is still None or 0, use default
                # NOTE: Default SMIB case file may not have a load, so use standard default
                # Matches ANDES manual strategy: p0=0.5 pu, q0=0.2 pu at bus 3
                if base_p is None or base_p == 0.0:
                    if verbose:
                        print(
                            f"[WARNING] Base load is zero or not found in case file. "
                            f"Using default base_load=0.5 pu (Pload), 0.2 pu (Qload) for alpha scaling. "
                            f"This is normal if the default SMIB case file has no load."
                        )
                    base_p = 0.5  # Default base load for SMIB (matches ANDES manual strategy)
                    base_q = 0.2  # Default reactive power (matches ANDES manual strategy)

                Load_P = alpha * base_p
                if verbose:
                    print(
                        f"[ALPHA SCALING] Computed Load_P={Load_P:.6f} pu "
                        f"(alpha={alpha:.6f} × base_load={base_p:.6f} pu) for CCT finding"
                    )

            if Load_P is not None:
                estimated_losses = 0.03 * Load_P  # Assume 3% losses
                Pm_to_use = Load_P + estimated_losses  # Estimate for CCT finding
                if verbose:
                    print(
                        f"[LOAD VARIATION] Using estimated Pm={Pm_to_use:.6f} pu "
                        f"(Load={Load_P:.6f} pu + "
                        f"losses={estimated_losses:.6f} pu) for CCT finding. "
                        f"Actual Pm will be extracted after power flow."
                    )
            else:
                Pm_to_use = 0.8  # Default fallback
                if verbose:
                    print(
                        f"[WARNING] Load_P is None. Using default Pm={Pm_to_use:.6f} pu "
                        f"for CCT finding (may be inaccurate)"
                    )
        elif Pm_to_use is None:
            # Pm variation mode: Try to extract from a test system load
            try:
                test_ss = andes.load(
                    case_path,
                    setup=False,
                    no_output=True,
                    default_config=True,
                    **(dict(addfile=addfile_path) if addfile_path else {}),
                )
                test_ss.setup()
                if hasattr(test_ss, "GENCLS") and test_ss.GENCLS.n > 0:
                    # SIMPLE: Use find_main_generator_index to get the correct generator
                    # (already imported at module level from .andes_utils.cct_finder)

                    gen_idx = find_main_generator_index(test_ss)

                    # Extract Pm from the main generator
                    if hasattr(test_ss.GENCLS, "tm0") and hasattr(test_ss.GENCLS.tm0, "v"):
                        tm0v = test_ss.GENCLS.tm0.v
                        if hasattr(tm0v, "__getitem__") and len(tm0v) > gen_idx:
                            tm0_val = tm0v[gen_idx]
                        else:
                            tm0_val = tm0v if not hasattr(tm0v, "__len__") else None

                        if tm0_val is not None:
                            # Use absolute value for Pm (handle sign convention)
                            Pm_to_use = abs(float(tm0_val))
                        else:
                            Pm_to_use = None
                    else:
                        Pm_to_use = None
                del test_ss
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Failed to extract Pm from case file: {e}")
                pass

            # Default if extraction failed or Pm is zero/invalid
            if Pm_to_use is None or abs(Pm_to_use) < 1e-6:
                if verbose and Pm_to_use is not None and abs(Pm_to_use) < 1e-6:
                    print(
                        f"[WARNING] Extracted Pm={Pm_to_use:.6f} is zero or near-zero. Using default 0.8 pu."
                    )
                Pm_to_use = 0.8  # Default mechanical power

        # Determine clearing times to use for this (M, D, Pm) combination
        if skip_fault:
            # Run TDS without fault: one scenario per (H, D, alpha); no CCT finding
            clearing_times_for_this_pair = [simulation_time]
            if verbose:
                print(
                    "[INFO] skip_fault=True: running TDS without fault "
                    "(one scenario per combination)."
                )
        elif use_cct_based_sampling:
            # Find CCT for this (M, D, Pm) combination
            if verbose:
                mode_label = "[LOAD VARIATION]" if use_load_mode else "[Pm VARIATION]"
                print(f"\n{mode_label} Finding CCT for M={M:.2f}, D={D:.2f}, Pm={Pm_to_use:.3f}...")

            # Find CCT using bisection (Pm_to_use was already determined above).
            # For load variation: pass alpha and base_load so CCT is found for this load level.
            # Pass addfile so Kundur (raw + dyr) loads with GENCLS.
            # Note: CCT finder uses single-generator stability check; if it fails for multimachine,
            # param_cct_absolute is filled later from stable/unstable boundary (estimated CCT).
            try:
                cct_duration, uncertainty, stable_result, unstable_result = find_cct(
                    case_path=case_path,
                    Pm=Pm_to_use,
                    M=M,
                    D=D,
                    fault_start_time=fault_start_time,
                    fault_bus=fault_bus,
                    fault_reactance=fault_reactance,
                    min_tc=fault_start_time + 0.01,
                    max_tc=simulation_time - 0.5,
                    simulation_time=simulation_time,
                    time_step=time_step,
                    tolerance_initial=0.01,
                    tolerance_final=0.001,
                    max_iterations=50,
                    logger=None,
                    ss=None,
                    reload_system=True,  # Use clean state for reliability
                    alpha=alpha if use_load_mode and alpha is not None else None,
                    base_load=base_load if use_load_mode else None,
                    addfile=addfile_path if addfile_path else None,
                )

                if cct_duration is None:
                    # Fallback: use config fault_clearing_times so we still get trajectory data
                    # when CCT finding fails (e.g. Kundur/multimachine with TDS init issues).
                    # param_cct_absolute will be filled later from stable/unstable boundary (estimated CCT).
                    error_summary["cct_finding_failed"] += 1
                    if verbose:
                        print(
                            f"[CCT] Bisection did not return CCT for this operating point. "
                            f"Using fixed clearing times; CCT will be estimated from stable/unstable boundary in CSV."
                        )
                    if fault_clearing_times and len(fault_clearing_times) > 0:
                        clearing_times_for_this_pair = [
                            max(fault_start_time + 0.01, min(float(tc), simulation_time - 0.5))
                            for tc in fault_clearing_times
                        ]
                        clearing_times_for_this_pair = sorted(
                            list(set(clearing_times_for_this_pair))
                        )
                        if verbose:
                            print(
                                f"[FALLBACK] CCT not found for M={M:.2f}, D={D:.2f}, Pm={Pm_to_use:.3f}. "
                                f"Using {len(clearing_times_for_this_pair)} fixed clearing times from config."
                            )
                    else:
                        if verbose:
                            print(
                                f"[SKIP] CCT not found for M={M:.2f}, D={D:.2f}, Pm={Pm_to_use:.3f}. "
                                f"No fault_clearing_times in config; skipping this operating point."
                            )
                        clearing_times_for_this_pair = []
                else:
                    # Calculate absolute CCT (clearing time, not duration)
                    cct_absolute = fault_start_time + cct_duration

                    # Extract small_delta (ε) from result (following smib_albert_cct.py logic)
                    # small_delta = max(0.004, uncertainty) - ensures minimum spacing of 4ms from boundary
                    if stable_result and "small_delta" in stable_result:
                        small_delta = stable_result["small_delta"]
                    else:
                        # Fallback: calculate from uncertainty (same logic as smib_albert_cct.py)
                        # Minimum offset of 4ms (0.004s) to avoid numerical precision issues at boundary
                        small_delta = max(0.004, uncertainty) if uncertainty is not None else 0.004

                    # Store CCT information for this (M, D) pair (will be added to DataFrame)
                    cct_info_for_this_pair = {
                        "cct_absolute": cct_absolute,
                        "cct_duration": cct_duration,
                        "cct_uncertainty": uncertainty,
                        "small_delta": small_delta,
                    }

                    if verbose:
                        print(
                            f"  [OK] CCT found: {cct_duration:.6f}s (absolute: {cct_absolute:.6f}s)"
                        )
                        print(
                            f"  Uncertainty: ±{uncertainty:.6f}s, small_delta (ε): {small_delta:.6f}s"
                        )

                    # Generate clearing times around CCT using small_delta (following smib_albert_cct.py strategy)
                    if cct_offsets is None:
                        # Auto-generate offsets based on small_delta (proportional to uncertainty)
                        # Strategy: Generate multiple samples around CCT using multiples of small_delta
                        # This ensures offsets are proportional to CCT uncertainty, not fixed values
                        n_stable = max(1, n_samples_per_combination // 2)
                        n_unstable = n_samples_per_combination - n_stable

                        # Stable clearing times (at/before CCT): use negative/zero multiples of small_delta
                        # Range: from -2*small_delta to 0 (at CCT) - includes CCT which is stable
                        if n_stable > 1:
                            stable_multipliers = np.linspace(-2.0, 0.0, n_stable)
                        else:
                            stable_multipliers = [0.0]  # At CCT (stable)

                        # Unstable clearing times (after CCT): use positive multiples of small_delta
                        # Range: from +1*small_delta to +2*small_delta (after CCT, unstable)
                        if n_unstable > 1:
                            unstable_multipliers = np.linspace(1.0, 2.0, n_unstable)
                        else:
                            unstable_multipliers = [1.0]  # Just after CCT (unstable)

                        # Calculate offsets in seconds (multiply small_delta by multipliers)
                        stable_offsets = [mult * small_delta for mult in stable_multipliers]
                        unstable_offsets = [mult * small_delta for mult in unstable_multipliers]
                        offsets = stable_offsets + unstable_offsets
                    else:
                        # User-provided offsets: select exactly n_stable from stable range (<=0) and
                        # n_unstable from unstable range (>0) so clearing times are balanced around CCT
                        n_stable = n_samples_per_combination // 2
                        n_unstable = n_samples_per_combination - n_stable
                        stable_offsets_user = sorted(
                            [o for o in cct_offsets if o <= 0], reverse=True
                        )
                        unstable_offsets_user = sorted([o for o in cct_offsets if o > 0])
                        offsets = (
                            stable_offsets_user[:n_stable] + unstable_offsets_user[:n_unstable]
                        )

                    # Generate clearing times: CCT + offsets
                    # Note: CCT is the maximum stable clearing time, so offset = 0 (at CCT) is stable
                    # Enforce minimum offset of 4ms (0.004s) from CCT boundary to avoid numerical precision issues
                    min_offset_from_boundary = 0.004  # 4ms minimum offset
                    clearing_times_for_this_pair = []
                    for offset in offsets:
                        # Enforce minimum offset from CCT boundary
                        if offset > 0:
                            # Unstable case: ensure at least 4ms after CCT
                            offset = max(offset, min_offset_from_boundary)
                        elif offset < 0:
                            # Stable case: ensure at least 4ms before CCT
                            offset = min(offset, -min_offset_from_boundary)
                        # Use adjusted offset
                        tc = cct_absolute + offset
                        # Ensure clearing time is after fault start and before simulation end.
                        # Use a small margin (2 ms) so that when CCT is near fault_start_time, multiple
                        # offset-based clearing times (e.g. CCT-0.012, CCT-0.008, CCT-0.004) stay distinct
                        # instead of all being clamped to fault_start_time+0.01 (which would collapse to 4 scenarios).
                        tc_min = fault_start_time + 0.002
                        tc = max(tc_min, min(tc, simulation_time - 0.5))
                        clearing_times_for_this_pair.append(tc)

                    # Remove duplicates and sort
                    clearing_times_for_this_pair = sorted(list(set(clearing_times_for_this_pair)))

                    if verbose:
                        print(
                            f"Generated {len(clearing_times_for_this_pair)} clearing times around"
                            f"CCT:"
                        )
                        for tc in clearing_times_for_this_pair:
                            offset_from_cct = tc - cct_absolute
                            # CCT is maximum stable time, so offset <= 0 is stable (offset = 0 means at CCT, which is stable)
                            status = "stable" if offset_from_cct <= 0 else "unstable"
                            print(f"    - {tc:.6f}s (offset: {offset_from_cct:+.6f}s, {status})")
            except Exception as e:
                error_summary["cct_finding_failed"] += 1
                if verbose:
                    print(
                        f"[CCT] Bisection failed for M={M:.2f}, D={D:.2f}, Pm={Pm_to_use:.3f}: {e}. "
                        f"CCT will be estimated from stable/unstable boundary after trajectory generation."
                    )
                if fault_clearing_times and len(fault_clearing_times) > 0:
                    clearing_times_for_this_pair = [
                        max(fault_start_time + 0.01, min(float(tc), simulation_time - 0.5))
                        for tc in fault_clearing_times
                    ]
                    clearing_times_for_this_pair = sorted(list(set(clearing_times_for_this_pair)))
                    if verbose:
                        print(
                            f"[FALLBACK] Using {len(clearing_times_for_this_pair)} fixed clearing times from config."
                        )
                else:
                    if verbose:
                        print(
                            f"[SKIP] Error finding CCT for M={M:.2f}, D={D:.2f}, Pm={Pm_to_use:.3f}: {e}. "
                            f"No fault_clearing_times in config; skipping this operating point."
                        )
                    clearing_times_for_this_pair = []
        else:
            # Use fixed clearing times
            clearing_times_for_this_pair = fault_clearing_times

        # CRITICAL: After CCT finding, ensure we use a completely fresh system for trajectory generation
        # CCT finding may leave residual state (power flow solutions, TDS initial conditions, etc.)
        # By reloading the system here, we ensure clean state for trajectory generation
        # This addresses the issue where TDS.init() may capture stale initial conditions from CCT finding phase
        if use_cct_based_sampling and verbose:
            print(
                "[INFO] CCT finding complete. System will be reloaded for trajectory generation "
                "to ensure clean state and prevent residual state from affecting TDS initialization."
            )

        # Generate trajectories for each clearing time
        # Note: Clearing times are already validated as absolute times >= fault_start_time
        for tc in clearing_times_for_this_pair:
            # Additional safety check (should not be needed if validation above worked)
            if tc < fault_start_time:
                error_summary["other_errors"] += 1
                if verbose:
                    print(
                        f"ERROR: Clearing time {tc:.3f}s < fault_start_time"
                        f"{fault_start_time:.3f}s."
                        f"This should have been caught by validation. Skipping."
                    )
                continue

            for fault_bus_iter in fault_locations:
                current_combination += 1

                try:
                    # CRITICAL: Modify case file for operating point BEFORE loading
                    # NEW: Support both Pm variation and load variation
                    case_path_to_use = original_case_path
                    modified_case_created = False

                    # PHASE 1: Case File Modification for Load Variation Mode
                    # SIMPLIFIED: Modify case file to set generator setpoint to match expected Pm
                    # This ensures TDS.init() reads correct value from case file (it reads from case file, not power flow)
                    # Power flow will still work correctly because we set load via alter() AFTER setup()
                    # The case file generator setpoint is just for TDS.init() to read the correct initial Pe
                    if use_load_mode and CASE_FILE_MODIFIER_AVAILABLE and Pm_to_use is not None:
                        try:
                            if default_pm is None or abs(Pm_to_use - default_pm) > 1e-6:
                                modified_case_path = modify_case_file_generator_setpoint(
                                    case_path=original_case_path,
                                    gen_idx=0,  # For SMIB, generator 0 is the main generator
                                    new_pm=Pm_to_use,  # Set to estimated Pm (Load + losses)
                                )
                                case_path_to_use = modified_case_path
                                modified_case_created = True
                                if verbose:
                                    print(
                                        f"[PHASE 1] [LOAD VARIATION] Modified case file: "
                                        f"Generator setpoint = {Pm_to_use:.6f} pu "
                                        f"(for TDS.init() to read correct Pe). "
                                        f"Load will be set via alter() after setup()."
                                    )
                            else:
                                case_path_to_use = original_case_path
                                modified_case_created = False
                        except Exception as case_modify_err:
                            # Fallback: Continue with original case file
                            if verbose:
                                print(
                                    f"[PHASE 1] [LOAD VARIATION] [WARNING] Case file modification failed: "
                                    f"{case_modify_err}. Continuing with original case file."
                                )
                            case_path_to_use = original_case_path
                            modified_case_created = False

                    # Load variation mode: Use ANDES alter() as PRIMARY method (no case file modification needed)
                    # Load setpoints (p0, q0) are INPUT parameters, so alter() works correctly
                    # Case file modification is only used as fallback if alter() fails

                    # Pm variation mode: Modify generator setpoint (original behavior)
                    if not use_load_mode and CASE_FILE_MODIFIER_AVAILABLE and Pm_to_use is not None:
                        # Pm variation mode: Modify generator setpoint (original behavior)
                        # Check if Pm differs from default (optimization: skip modification if same)
                        if default_pm is None or abs(Pm_to_use - default_pm) > 1e-6:
                            try:
                                modified_case_path = modify_case_file_generator_setpoint(
                                    case_path=original_case_path,
                                    gen_idx=0,  # For SMIB, generator 0 is the main generator
                                    new_pm=Pm_to_use,
                                )
                                case_path_to_use = modified_case_path
                                modified_case_created = True
                                if verbose:
                                    print(
                                        f"[CASE MODIFY] Using modified case file with"
                                        f"Pm={Pm_to_use:.6f} pu"
                                    )
                            except Exception as case_modify_err:
                                # Expected fallback: Some SMIB cases don't have StaticGen, so case file modification fails.
                                # Manual Pm setting works correctly as fallback.
                                error_msg = str(case_modify_err)
                                if "StaticGen" in error_msg or "power setpoint field" in error_msg:
                                    # Expected: GENCLS doesn't have power setpoints, manual setting works
                                    # Suppress verbose warning for expected behavior (only show once per run)
                                    # Note: _case_modify_warning_shown is in outer function scope, accessible directly
                                    if verbose and not _case_modify_warning_shown:
                                        print(
                                            f"[INFO] Case file modification not available (expected"
                                            f"for some SMIB cases)."
                                            f"Using manual Pm setting (works correctly)."
                                        )
                                        _case_modify_warning_shown = True
                                else:
                                    # Unexpected error - show as warning
                                    if verbose:
                                        print(
                                            f"[WARNING] [Pm VARIATION] Could not modify case file"
                                            f"for Pm={Pm_to_use:.6f}: {case_modify_err}."
                                            f"Falling back to manual Pm setting."
                                        )
                                # Fall back to original case file and manual setting
                                case_path_to_use = original_case_path

                    # Load system with setup=False, then call setup() explicitly
                    # Following ANDES tutorial pattern: https://docs.andes.app/en/latest/tutorials/08-parameter-sweeps.html
                    ss = andes.load(
                        case_path_to_use,
                        setup=False,
                        no_output=True,
                        default_config=True,
                        **(dict(addfile=addfile_path) if addfile_path else {}),
                    )

                    # WORKFLOW VERIFICATION: Track workflow order to ensure correct sequence
                    # Expected order: load → setup → M/D → PV.p0 → GENCLS.tm0 → power_flow → TDS
                    workflow_steps = []
                    workflow_steps.append("load")
                    # For multimachine: save tm0 from PF so we can re-apply after TDS.init() if ANDES resets it
                    saved_tm0_from_pf_multimachine = None

                    # NEW: Check if redispatch mode is enabled (must be checked before load scaling)
                    # Initialize to False, then check conditions
                    use_redispatch_mode = False
                    if (
                        use_redispatch
                        and REDISPATCH_AVAILABLE
                        and run_multimachine_powerflow_with_redispatch is not None
                        and alpha is not None
                    ):
                        # Check if system is multimachine (has more than 1 generator)
                        if hasattr(ss, "GENCLS") and ss.GENCLS.n > 1:
                            use_redispatch_mode = True

                    # Use ANDES alter() for load variation BEFORE setup() (following ANDES tutorial pattern)
                    # Tutorial: https://docs.andes.app/en/latest/tutorials/08-parameter-sweeps.html
                    # Load setpoints (p0, q0) are INPUT parameters to power flow, so alter() works correctly
                    # ANDES variable names: p0 (active power), q0 (reactive power)
                    # NEW: If alpha is provided, get base loads and scale by alpha (unified approach)
                    if use_load_mode and alpha is not None:
                        # Unified approach: Get base loads and scale by alpha
                        # CRITICAL: Use proper ANDES case file reading function (handles all formats)
                        base_loads = {}

                        # Multimachine: use case native loads from ss so power flow converges
                        # (config base_load is for SMIB; Kundur etc. need the case's actual P/Q)
                        if (
                            hasattr(ss, "GENCLS")
                            and ss.GENCLS.n > 1
                            and hasattr(ss, "PQ")
                            and ss.PQ.n > 0
                        ):
                            for load_idx in range(ss.PQ.n):
                                try:
                                    base_p = (
                                        float(ss.PQ.p0.v[load_idx])
                                        if hasattr(ss.PQ.p0.v, "__getitem__")
                                        and load_idx < len(ss.PQ.p0.v)
                                        else 0.0
                                    )
                                    base_q = (
                                        float(ss.PQ.q0.v[load_idx])
                                        if hasattr(ss.PQ.q0, "v")
                                        and hasattr(ss.PQ.q0.v, "__getitem__")
                                        and load_idx < len(ss.PQ.q0.v)
                                        else 0.0
                                    )
                                    base_loads[load_idx] = {"p0": base_p, "q0": base_q}
                                except (IndexError, TypeError, AttributeError):
                                    pass
                            if base_loads and verbose:
                                total_p = sum(base_loads[i]["p0"] for i in base_loads)
                                print(
                                    f"[DEBUG] Multimachine: using case native base loads ({len(base_loads)} PQ). Total P={total_p:.4f} pu"
                                )

                        # Check if base_load is provided in config (highest priority for SMIB)
                        if not base_loads and base_load is not None and isinstance(base_load, dict):
                            base_p_config = base_load.get(
                                "Pload", base_load.get("P", None)
                            )  # Support both Pload and P for backward compatibility
                            base_q_config = base_load.get(
                                "Qload", base_load.get("Q", None)
                            )  # Support both Qload and Q for backward compatibility
                            if base_p_config is not None and base_p_config > 0:
                                # Use config value for first load (load_idx=0)
                                base_loads[0] = {
                                    "p0": base_p_config,
                                    "q0": base_q_config,
                                }
                                if verbose:
                                    print(
                                        f"[DEBUG] Using base load from config for load_idx=0: "
                                        f"p0={base_p_config:.6f} pu, q0={base_q_config:.6f} pu"
                                    )

                        # If config base_load not provided or invalid, try to read from case file
                        if 0 not in base_loads:
                            # Try to read all loads from case file using proper function
                            if (
                                CASE_FILE_MODIFIER_AVAILABLE
                                and get_default_load_from_case_file is not None
                            ):
                                load_idx = 0
                                while True:
                                    try:
                                        if verbose and load_idx == 0:
                                            print(
                                                f"[DEBUG] Reading base load from case file: {case_path}, "
                                                f"load_idx={load_idx}"
                                            )
                                        result = get_default_load_from_case_file(
                                            case_path,
                                            load_idx=load_idx,
                                            load_model="PQ",
                                        )
                                        if verbose and load_idx == 0:
                                            print(
                                                f"[DEBUG] get_default_load_from_case_file returned: {result}"
                                            )
                                        if result is not None:
                                            base_p, base_q = result
                                            # Validate: base_p must be > 0 (valid load)
                                            if base_p is not None and base_p > 0:
                                                base_loads[load_idx] = {
                                                    "p0": base_p,
                                                    "q0": base_q,
                                                }
                                                if verbose:
                                                    q_str = (
                                                        f"{base_q:.6f}"
                                                        if base_q is not None
                                                        else "N/A"
                                                    )
                                                    print(
                                                        f"[DEBUG] Read base load from case file: "
                                                        f"p0={base_p:.6f} pu, q0={q_str} pu"
                                                    )
                                                load_idx += 1
                                            else:
                                                if verbose:
                                                    print(
                                                        f"[WARNING] Base load p0={base_p} is zero or invalid "
                                                        f"for load_idx={load_idx}. Trying fallback..."
                                                    )
                                                break  # Invalid load, try fallback
                                        else:
                                            if verbose:
                                                print(
                                                    f"[DEBUG] get_default_load_from_case_file returned None "
                                                    f"for load_idx={load_idx}. No more loads."
                                                )
                                            break  # No more loads
                                    except Exception as e:
                                        if verbose:
                                            print(
                                                f"[WARNING] Error reading base load from case file: {e}. "
                                                f"Trying fallback..."
                                            )
                                        import traceback

                                        if verbose:
                                            traceback.print_exc()
                                        break  # Error, try fallback

                        # Fallback: Load temporary system to get base loads
                        if len(base_loads) == 0:
                            try:
                                if verbose:
                                    print(
                                        f"[DEBUG] Case file read returned no loads. "
                                        f"Trying temporary system load..."
                                    )
                                temp_ss = andes.load(
                                    case_path,
                                    setup=False,
                                    no_output=True,
                                    default_config=True,
                                    **(dict(addfile=addfile_path) if addfile_path else {}),
                                )
                                temp_ss.setup()
                                if hasattr(temp_ss, "PQ") and temp_ss.PQ.n > 0:
                                    for load_idx in range(temp_ss.PQ.n):
                                        base_p = (
                                            float(temp_ss.PQ.p0.v[load_idx])
                                            if hasattr(temp_ss.PQ, "p0")
                                            and hasattr(temp_ss.PQ.p0, "v")
                                            and hasattr(temp_ss.PQ.p0.v, "__getitem__")
                                            and len(temp_ss.PQ.p0.v) > load_idx
                                            else 0.0
                                        )
                                        base_q = (
                                            float(temp_ss.PQ.q0.v[load_idx])
                                            if hasattr(temp_ss.PQ, "q0")
                                            and hasattr(temp_ss.PQ.q0, "v")
                                            and hasattr(temp_ss.PQ.q0.v, "__getitem__")
                                            and len(temp_ss.PQ.q0.v) > load_idx
                                            else 0.0
                                        )
                                        base_loads[load_idx] = {
                                            "p0": base_p,
                                            "q0": base_q,
                                        }
                                del temp_ss
                            except Exception as e:
                                if verbose:
                                    print(
                                        f"[WARNING] Could not read base loads: {e}. "
                                        f"Case path: {case_path}. Using default values."
                                    )

                        # Final check: if base_loads is empty or has zero values, use defaults
                        if len(base_loads) == 0 or all(
                            base_loads[idx]["p0"] == 0.0 for idx in base_loads
                        ):
                            if verbose:
                                print(
                                    f"[WARNING] Base loads are zero or empty. "
                                    f"Using default base_load=0.5 pu (P), 0.2 pu (Q) for alpha scaling."
                                )
                            # Use default values (matches ANDES manual strategy)
                            if hasattr(ss, "PQ") and ss.PQ.n > 0:
                                for load_idx in range(ss.PQ.n):
                                    base_loads[load_idx] = {"p0": 0.5, "q0": 0.2}

                        if len(base_loads) > 0:
                            # CRITICAL: Scale ALL PQ loads uniformly by alpha (strategy from ANDES manual)
                            # This maintains power factor and works for both SMIB and multimachine
                            # Strategy: Read base load from case file, then scale using alter()
                            if hasattr(ss, "PQ") and ss.PQ.n > 0:
                                # Scale all loads uniformly
                                for load_idx in range(ss.PQ.n):
                                    if load_idx in base_loads:
                                        base_p = base_loads[load_idx]["p0"]
                                        base_q = base_loads[load_idx].get("q0")
                                        if base_q is None:
                                            base_q = 0.0  # Avoid None in scaling and format
                                        # Validate: ensure base_p > 0
                                        if base_p == 0.0 or base_p is None:
                                            if verbose:
                                                print(
                                                    f"[WARNING] Base load p0={base_p} is zero for "
                                                    f"load_idx={load_idx}. Using default 0.5 pu."
                                                )
                                            base_p = 0.5  # Default
                                            base_q = 0.2  # Default
                                            base_loads[load_idx] = {
                                                "p0": base_p,
                                                "q0": base_q,
                                            }
                                    else:
                                        # Fallback: use default values if not in base_loads
                                        if verbose:
                                            print(
                                                f"[WARNING] Load {load_idx} not in base_loads. "
                                                f"Using default 0.5 pu (P), 0.2 pu (Q)."
                                            )
                                        base_p = 0.5  # Default
                                        base_q = 0.2  # Default
                                        base_loads[load_idx] = {
                                            "p0": base_p,
                                            "q0": base_q,
                                        }

                                    # Scale by alpha (uniform scaling - maintains power factor)
                                    new_p0 = alpha * base_p
                                    new_q0 = alpha * base_q

                                    # Apply scaling using ANDES alter() method (strategy from manual)
                                    try:
                                        # Try to get UID first, then fallback to index
                                        load_identifier = load_idx
                                        if hasattr(ss.PQ, "idx") and hasattr(ss.PQ.idx, "v"):
                                            try:
                                                idx_array = ss.PQ.idx.v
                                                if (
                                                    hasattr(idx_array, "__getitem__")
                                                    and len(idx_array) > load_idx
                                                ):
                                                    load_uid = idx_array[load_idx]
                                                    load_identifier = load_uid
                                            except (
                                                IndexError,
                                                AttributeError,
                                                TypeError,
                                            ):
                                                pass  # Fallback to index

                                        if hasattr(ss.PQ, "alter"):
                                            ss.PQ.alter("p0", load_identifier, new_p0)
                                            ss.PQ.alter("q0", load_identifier, new_q0)
                                            if verbose and load_idx == 0:
                                                print(
                                                    f"[ALPHA SCALING] alpha={alpha:.6f}: "
                                                    f"Scaled ALL {ss.PQ.n} PQ load(s) uniformly. "
                                                    f"Load 0: P={new_p0:.6f} pu (α×{base_p:.6f}), "
                                                    f"Q={new_q0:.6f} pu (α×{base_q:.6f})"
                                                )
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
                                            print(
                                                f"[WARNING] alter() failed for load {load_idx}, "
                                                f"using direct access: {e}"
                                            )

                                # For backward compatibility, set Load_P to first load's scaled value
                                Load_P = alpha * base_loads[0]["p0"] if 0 in base_loads else None
                                Load_Q = alpha * base_loads[0]["q0"] if 0 in base_loads else None
                            else:
                                warnings.warn("No PQ load found. Cannot apply alpha scaling.")
                                Load_P = None
                                Load_Q = None
                        else:
                            warnings.warn("No PQ load found. Cannot apply alpha scaling.")
                            Load_P = None
                            Load_Q = None

                    # Backward compatibility: Use Load_P directly if provided (not from alpha)
                    if use_load_mode and Load_P is not None and alpha is None:
                        try:
                            # Find load model (PQ is most common)
                            if hasattr(ss, "PQ") and ss.PQ.n > 0:
                                load_idx_int = 0  # For SMIB, typically first load

                                # ANDES alter() can accept either index (int) or UID (string)
                                # Try to get PQ device UID first, then fallback to index
                                load_identifier = load_idx_int  # Default to index

                                # Try to get UID from idx attribute
                                if hasattr(ss.PQ, "idx") and hasattr(ss.PQ.idx, "v"):
                                    try:
                                        idx_array = ss.PQ.idx.v
                                        if (
                                            hasattr(idx_array, "__getitem__")
                                            and len(idx_array) > load_idx_int
                                        ):
                                            load_uid = idx_array[load_idx_int]
                                            load_identifier = load_uid  # Use UID if available
                                    except (IndexError, AttributeError, TypeError):
                                        pass  # Fallback to index

                                if hasattr(ss.PQ, "alter"):
                                    # ANDES built-in method (PRIMARY)
                                    # Try alter with identifier (UID or index)
                                    try:
                                        # ANDES variable: p0 (active power setpoint)
                                        ss.PQ.alter("p0", load_identifier, Load_P)
                                        if Load_Q is not None:
                                            # ANDES variable: q0 (reactive power setpoint)
                                            ss.PQ.alter("q0", load_identifier, Load_Q)
                                        if verbose:
                                            q_str = (
                                                f", q0={Load_Q:.6f}"
                                                if Load_Q is not None
                                                else " (q0=0, fixed)"
                                            )
                                            print(
                                                f"[ANDES ALTER] Set load p0={Load_P:.6f} pu{q_str}"
                                                f"using alter() method"
                                                f"(identifier: {load_identifier})"
                                            )
                                    except (
                                        KeyError,
                                        IndexError,
                                        AttributeError,
                                        ValueError,
                                    ) as alter_error:
                                        # If alter fails, try direct access
                                        if verbose:
                                            print(
                                                f"[DEBUG] alter() failed with {load_identifier},"
                                                f"trying direct access: {alter_error}"
                                            )
                                        # Fallback: Direct access to ANDES variables
                                        # ANDES variable: ss.PQ.p0.v (active power)
                                        ss.PQ.p0.v[load_idx_int] = Load_P
                                        if Load_Q is not None:
                                            # ANDES variable: ss.PQ.q0.v (reactive power)
                                            ss.PQ.q0.v[load_idx_int] = Load_Q
                                        if verbose:
                                            q_str = (
                                                f", q0={Load_Q:.6f}"
                                                if Load_Q is not None
                                                else " (q0=0, fixed)"
                                            )
                                            print(
                                                f"[DIRECT ACCESS] Set load p0={Load_P:.6f}"
                                                f"pu{q_str} using direct access"
                                            )
                                else:
                                    # Fallback: Direct access to ANDES variables
                                    # ANDES variable: ss.PQ.p0.v (active power)
                                    ss.PQ.p0.v[load_idx_int] = Load_P
                                    if Load_Q is not None:
                                        # ANDES variable: ss.PQ.q0.v (reactive power)
                                        ss.PQ.q0.v[load_idx_int] = Load_Q
                                    if verbose:
                                        q_str = (
                                            f", q0={Load_Q:.6f}"
                                            if Load_Q is not None
                                            else " (q0=0, fixed)"
                                        )
                                        print(
                                            f"[DIRECT ACCESS] Set load p0={Load_P:.6f} pu{q_str}"
                                            f"using direct access"
                                        )
                            else:
                                # No PQ load found - try case file modification as fallback
                                if (
                                    CASE_FILE_MODIFIER_AVAILABLE
                                    and modify_case_file_load_setpoint is not None
                                ):
                                    if verbose:
                                        print(
                                            f"[FALLBACK] No PQ load found, using case file"
                                            f"modification for Load P={Load_P:.6f}"
                                        )
                                    try:
                                        modified_case_path = modify_case_file_load_setpoint(
                                            case_path=original_case_path,
                                            load_idx=0,
                                            new_p=Load_P,
                                            new_q=Load_Q,
                                            load_model="PQ",
                                        )
                                        # Reload with modified case file
                                        ss = andes.load(
                                            modified_case_path,
                                            setup=False,
                                            no_output=True,
                                            default_config=True,
                                        )
                                        modified_case_created = True
                                    except Exception as case_modify_err:
                                        if verbose:
                                            print(
                                                f"[WARNING] Could not modify case file for Load"
                                                f"P={Load_P:.6f}: {case_modify_err}"
                                            )
                        except Exception as load_set_err:
                            if verbose:
                                print(
                                    f"[WARNING] Could not set load using alter(): {load_set_err}. "
                                    f"Trying case file modification as fallback."
                                )
                            # Fallback: Try case file modification
                            if (
                                CASE_FILE_MODIFIER_AVAILABLE
                                and modify_case_file_load_setpoint is not None
                            ):
                                try:
                                    modified_case_path = modify_case_file_load_setpoint(
                                        case_path=original_case_path,
                                        load_idx=0,
                                        new_p=Load_P,
                                        new_q=Load_Q,
                                        load_model="PQ",
                                    )
                                    # Reload with modified case file
                                    ss = andes.load(
                                        modified_case_path,
                                        setup=False,
                                        no_output=True,
                                        default_config=True,
                                    )
                                    modified_case_created = True
                                    if verbose:
                                        print(
                                            f"[FALLBACK] Using case file modification for Load"
                                            f"P={Load_P:.6f} pu"
                                        )
                                except Exception as case_modify_err:
                                    if verbose:
                                        print(
                                            f"[ERROR] Both alter() and case file modification"
                                            f"failed for Load P={Load_P:.6f}: {case_modify_err}"
                                        )

                    # Add fault BEFORE setup() when case has no fault (e.g. raw+dyr); ANDES does not allow adding devices after setup()
                    # Skip when skip_fault=True (TDS without fault)
                    if (
                        hasattr(ss, "Fault")
                        and ss.Fault.n == 0
                        and not use_redispatch_mode
                        and not skip_fault
                    ):
                        first_tc = (
                            clearing_times_for_this_pair[0]
                            if clearing_times_for_this_pair
                            else fault_start_time + 0.2
                        )
                        try:
                            ss.add(
                                "Fault",
                                {
                                    "u": 1,
                                    "name": f"F_bus{fault_bus_iter}",
                                    "bus": fault_bus_iter,
                                    "tf": fault_start_time,
                                    "tc": first_tc,
                                    "rf": 0.0,
                                    "xf": fault_reactance,
                                },
                            )
                        except Exception as e_add:
                            if verbose:
                                print(
                                    f"[WARNING] Could not add fault before setup: {e_add}. "
                                    f"Fault will be added per clearing time (may fail after setup)."
                                )

                    # Call setup() explicitly (following ANDES tutorial pattern)
                    # Skip if redispatch already did setup (redispatch does setup internally)
                    if not use_redispatch_mode:
                        ss.setup()

                    # Validate that GENCLS exists and has devices
                    if not hasattr(ss, "GENCLS") or ss.GENCLS.n == 0:
                        error_summary["no_gencls"] += 1
                        if verbose:
                            print(
                                f"Warning: No GENCLS generator found. Skipping combination"
                                f"{current_combination}."
                            )
                        continue

                    # Find main generator index (not infinite bus) - consistent with smib_batch_tds.py
                    # Uses find_main_generator_index() which checks M > 1e6 and generator names
                    if CCT_FINDER_AVAILABLE and find_main_generator_index is not None:
                        gen_idx_to_modify = find_main_generator_index(ss)
                    else:
                        # Fallback: Use first generator if find_main_generator_index not available
                        gen_idx_to_modify = 0
                        if verbose:
                            print(
                                "[WARNING] find_main_generator_index not available. Using gen_idx=0 as fallback."
                            )

                    # Set generator parameters using alter() (ANDES recommended approach)
                    # Multimachine: apply same H (M) and D to ALL generators when one (H,D) is swept (design: uniform H/D).
                    n_gen = getattr(ss.GENCLS, "n", 1)
                    try:
                        if hasattr(ss.GENCLS, "alter"):
                            # Use alter() method (ANDES recommended); set M and D for every generator
                            for gidx in range(n_gen):
                                ss.GENCLS.alter("M", gidx, M)
                                ss.GENCLS.alter("D", gidx, D)
                            workflow_steps.append("M_D_set")  # Track workflow order
                            # Set Pm if case file modification failed (Pm variation mode)
                            if (
                                not modified_case_created
                                and not use_load_mode
                                and Pm_to_use is not None
                            ):
                                # CRITICAL: Set PV.p0 first (for power flow)
                                # This matches the fix in smib_batch_tds.py
                                if hasattr(ss, "PV") and ss.PV.n > 0:
                                    if hasattr(ss.PV, "p0") and hasattr(ss.PV.p0, "v"):
                                        if len(ss.PV.p0.v) > 0:
                                            ss.PV.p0.v[0] = Pm_to_use
                                            workflow_steps.append(
                                                "PV_p0_set"
                                            )  # Track workflow order
                                            if verbose:
                                                print(
                                                    f"[GEN POWER] Set PV.p0 = {Pm_to_use:.6f} pu (P_m variation mode)"
                                                )

                                # CRITICAL: Set GENCLS.tm0 explicitly (does NOT auto-copy from PV.p0)
                                # It must be set manually to match PV.p0
                                # Use direct access (consistent with smib_batch_tds.py and documentation)
                                if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                                    if gen_idx_to_modify < len(ss.GENCLS.tm0.v):
                                        ss.GENCLS.tm0.v[gen_idx_to_modify] = Pm_to_use
                                        workflow_steps.append(
                                            "GENCLS_tm0_set"
                                        )  # Track workflow order
                                        if verbose:
                                            print(
                                                f"[GEN POWER] Set GENCLS.tm0[{gen_idx_to_modify}] = {Pm_to_use:.6f} pu (mechanical power)"
                                            )
                                if hasattr(ss.GENCLS, "P0"):
                                    ss.GENCLS.alter("P0", gen_idx_to_modify, Pm_to_use)
                        else:
                            # Fallback to direct access: set M and D for all generators
                            for gidx in range(n_gen):
                                if gidx < len(ss.GENCLS.M.v):
                                    ss.GENCLS.M.v[gidx] = M
                                if gidx < len(ss.GENCLS.D.v):
                                    ss.GENCLS.D.v[gidx] = D
                            # Set Pm if case file modification failed (Pm variation mode)
                            if (
                                not modified_case_created
                                and not use_load_mode
                                and Pm_to_use is not None
                            ):
                                # CRITICAL: Set PV.p0 first (for power flow)
                                # This matches the fix in smib_batch_tds.py
                                if hasattr(ss, "PV") and ss.PV.n > 0:
                                    if hasattr(ss.PV, "p0") and hasattr(ss.PV.p0, "v"):
                                        if len(ss.PV.p0.v) > 0:
                                            ss.PV.p0.v[0] = Pm_to_use
                                            workflow_steps.append(
                                                "PV_p0_set"
                                            )  # Track workflow order
                                            if verbose:
                                                print(
                                                    f"[GEN POWER] Set PV.p0 = {Pm_to_use:.6f} pu (P_m variation mode)"
                                                )

                                # CRITICAL: Set GENCLS.tm0 explicitly (does NOT auto-copy from PV.p0)
                                # It must be set manually to match PV.p0
                                if hasattr(ss.GENCLS, "tm0"):
                                    ss.GENCLS.tm0.v[gen_idx_to_modify] = Pm_to_use
                                    workflow_steps.append("GENCLS_tm0_set")  # Track workflow order
                                    if verbose:
                                        print(
                                            f"[GEN POWER] Set GENCLS.tm0[{gen_idx_to_modify}] = {Pm_to_use:.6f} pu (mechanical power)"
                                        )
                                if hasattr(ss.GENCLS, "P0"):
                                    ss.GENCLS.P0.v[gen_idx_to_modify] = Pm_to_use

                    except Exception as e:
                        # If alter() fails, try direct access as fallback; set M and D for all generators
                        try:
                            for gidx in range(n_gen):
                                if gidx < len(ss.GENCLS.M.v):
                                    ss.GENCLS.M.v[gidx] = M
                                if gidx < len(ss.GENCLS.D.v):
                                    ss.GENCLS.D.v[gidx] = D
                            # Set Pm if case file modification failed (skip for multimachine + load variation)
                            set_pm_in_exception = (
                                not modified_case_created
                                and Pm_to_use is not None
                                and not (
                                    use_load_mode
                                    and alpha is not None
                                    and hasattr(ss, "GENCLS")
                                    and ss.GENCLS.n > 1
                                )
                            )
                            if set_pm_in_exception:
                                # CRITICAL: Set PV.p0 first (for power flow)
                                # This matches the fix in smib_batch_tds.py
                                if hasattr(ss, "PV") and ss.PV.n > 0:
                                    if hasattr(ss.PV, "p0") and hasattr(ss.PV.p0, "v"):
                                        if len(ss.PV.p0.v) > 0:
                                            ss.PV.p0.v[0] = Pm_to_use
                                            workflow_steps.append(
                                                "PV_p0_set"
                                            )  # Track workflow order
                                            if verbose:
                                                print(
                                                    f"[GEN POWER] Set PV.p0 = {Pm_to_use:.6f} pu (P_m variation mode, exception handler)"
                                                )

                                # CRITICAL: Set GENCLS.tm0 explicitly (does NOT auto-copy from PV.p0)
                                # It must be set manually to match PV.p0
                                if hasattr(ss.GENCLS, "tm0"):
                                    ss.GENCLS.tm0.v[gen_idx_to_modify] = Pm_to_use
                                    workflow_steps.append("GENCLS_tm0_set")  # Track workflow order
                                    if verbose:
                                        print(
                                            f"[GEN POWER] Set GENCLS.tm0[{gen_idx_to_modify}] = {Pm_to_use:.6f} pu (mechanical power)"
                                        )
                                if hasattr(ss.GENCLS, "P0"):
                                    ss.GENCLS.P0.v[gen_idx_to_modify] = Pm_to_use
                        except Exception as e2:
                            error_summary["gen_param_failed"] += 1
                            if verbose:
                                print(
                                    f"Warning: Failed to set generator parameters: {e2}. Skipping"
                                    f"combination {current_combination}."
                                )
                            continue

                    # Configure fault parameters (following smib_albert_cct.py pattern)
                    # Use parameter values (not hardcoded)
                    fault_start_time_actual = fault_start_time  # Use parameter
                    fault_reactance_actual = fault_reactance  # Use parameter
                    fault_bus_actual = fault_bus_iter  # Use loop variable

                    # Check if fault model exists (skip fault config when skip_fault=True)
                    fault_was_added = False
                    if not skip_fault and hasattr(ss, "Fault") and ss.Fault.n > 0:
                        # Modify existing fault using ANDES alter() method (recommended approach)
                        # Find the fault idx (typically 0 for SMIB, but check all faults)
                        fault_idx = None
                        try:
                            # Try to find fault on the specified bus
                            if hasattr(ss.Fault, "bus"):
                                for i in range(ss.Fault.n):
                                    if (
                                        hasattr(ss.Fault.bus, "v")
                                        and ss.Fault.bus.v[i] == fault_bus_actual
                                    ):
                                        fault_idx = i
                                        break
                            # If not found, use first fault (idx=0) as default
                            if fault_idx is None:
                                fault_idx = 0
                        except Exception:
                            fault_idx = 0  # Default to first fault

                        # Use alter() method to modify fault parameters (ANDES recommended approach)
                        try:
                            if hasattr(ss.Fault, "alter"):
                                ss.Fault.alter("tf", fault_idx, fault_start_time_actual)
                                ss.Fault.alter("tc", fault_idx, tc)
                                if hasattr(ss.Fault, "bus"):
                                    ss.Fault.alter("bus", fault_idx, fault_bus_actual)
                                if hasattr(ss.Fault, "xf"):
                                    ss.Fault.alter("xf", fault_idx, fault_reactance_actual)
                                if hasattr(ss.Fault, "rf"):
                                    ss.Fault.alter(
                                        "rf", fault_idx, 0.0
                                    )  # Resistance (0 for bolted fault)
                                if hasattr(ss.Fault, "u"):
                                    ss.Fault.alter("u", fault_idx, 1)  # Enable fault
                            else:
                                # Fallback to direct access if alter() is not available
                                ss.Fault.tf.v[fault_idx] = fault_start_time_actual
                                ss.Fault.tc.v[fault_idx] = tc
                                if hasattr(ss.Fault, "bus"):
                                    ss.Fault.bus.v[fault_idx] = fault_bus_actual
                                if hasattr(ss.Fault, "xf"):
                                    ss.Fault.xf.v[fault_idx] = fault_reactance_actual
                                if hasattr(ss.Fault, "rf"):
                                    ss.Fault.rf.v[fault_idx] = 0.0
                                if hasattr(ss.Fault, "u"):
                                    ss.Fault.u.v[fault_idx] = 1
                        except Exception as e:
                            # If alter() fails, try direct access as fallback
                            try:
                                ss.Fault.tf.v[fault_idx] = fault_start_time_actual
                                ss.Fault.tc.v[fault_idx] = tc
                                if hasattr(ss.Fault, "bus"):
                                    ss.Fault.bus.v[fault_idx] = fault_bus_actual
                                if hasattr(ss.Fault, "xf"):
                                    ss.Fault.xf.v[fault_idx] = fault_reactance_actual
                                if hasattr(ss.Fault, "rf"):
                                    ss.Fault.rf.v[fault_idx] = 0.0
                                if hasattr(ss.Fault, "u"):
                                    ss.Fault.u.v[fault_idx] = 1
                            except Exception as e2:
                                error_summary["fault_config_failed"] += 1
                                if verbose:
                                    print(
                                        f"Warning: Could not modify fault: {e2}. Skipping"
                                        f"combination {current_combination}."
                                    )
                                continue
                    elif not skip_fault:
                        # Add a new fault if it doesn't exist (requires setup())
                        try:
                            ss.add(
                                "Fault",
                                {
                                    "u": 1,  # Enable fault
                                    "name": f"F_bus{fault_bus_actual}",
                                    "bus": fault_bus_actual,
                                    "tf": fault_start_time_actual,
                                    "tc": tc,
                                    "rf": 0.0,  # Resistance (0 for bolted fault)
                                    "xf": fault_reactance_actual,  # Reactance
                                },
                            )
                            fault_was_added = True
                        except Exception as e:
                            error_summary["fault_config_failed"] += 1
                            if verbose:
                                print(
                                    f"Warning: Could not add fault: {e}. Skipping combination"
                                    f"{current_combination}."
                                )
                            continue

                    # If a new fault was added, call setup() again (required after adding devices)
                    if fault_was_added:
                        try:
                            ss.setup()
                        except Exception as e:
                            error_summary["fault_config_failed"] += 1
                            if verbose:
                                print(
                                    f"Warning: Error during setup after adding fault: {e}. Skipping"
                                    f"combination {current_combination}."
                                )
                            continue

                    # CRITICAL: Verify and re-set generator parameters AFTER setup() if needed
                    # M and D parameters were already set before setup(), but we verify they're still correct
                    # Note: Pm is set via case file modification, so it should be correct after setup()
                    gen_idx_after_setup = (
                        gen_idx_to_modify  # Use same index (should still be valid)
                    )
                    try:
                        # Verify M and D are still correct (setup() shouldn't reset them, but we check)
                        needs_reset = False
                        try:
                            M_current = (
                                ss.GENCLS.M.v[gen_idx_after_setup]
                                if hasattr(ss.GENCLS.M.v, "__getitem__")
                                and len(ss.GENCLS.M.v) > gen_idx_after_setup
                                else ss.GENCLS.M.v
                            )
                            D_current = (
                                ss.GENCLS.D.v[gen_idx_after_setup]
                                if hasattr(ss.GENCLS.D.v, "__getitem__")
                                and len(ss.GENCLS.D.v) > gen_idx_after_setup
                                else ss.GENCLS.D.v
                            )
                            M_mismatch = abs(M_current - M) / (abs(M) + 1e-12)
                            D_mismatch = abs(D_current - D) / (abs(D) + 1e-12)
                            if M_mismatch > 0.01 or D_mismatch > 0.01:  # More than 1% difference
                                needs_reset = True
                                if verbose:
                                    print(
                                        f"[WARNING] M or D changed after setup(): "
                                        f"M: requested={M:.6f}, actual={M_current:.6f}"
                                        f"({M_mismatch * 100:.1f}% diff),"
                                        f"D: requested={D:.6f}, actual={D_current:.6f}"
                                        f"({D_mismatch * 100:.1f}% diff)."
                                        f"Attempting to re-set."
                                    )
                        except (IndexError, AttributeError, TypeError):
                            # Can't verify - assume needs reset
                            needs_reset = True

                        # Only re-set M and D if verification failed (Pm is set via case file, so skip)
                        if needs_reset:
                            try:
                                n_gen_reset = getattr(ss.GENCLS, "n", 1)
                                if hasattr(ss.GENCLS, "alter"):
                                    for gidx in range(n_gen_reset):
                                        ss.GENCLS.alter("M", gidx, M)
                                        ss.GENCLS.alter("D", gidx, D)
                                else:
                                    for gidx in range(n_gen_reset):
                                        if gidx < len(ss.GENCLS.M.v):
                                            ss.GENCLS.M.v[gidx] = M
                                        if gidx < len(ss.GENCLS.D.v):
                                            ss.GENCLS.D.v[gidx] = D
                            except Exception as alter_err:
                                if verbose:
                                    print(
                                        f"[WARNING] Could not re-set M/D after setup():"
                                        f"{alter_err}."
                                        f"Continuing - parameters may still be correct."
                                    )
                    except Exception as e:
                        # Non-critical error - log but don't skip
                        if verbose:
                            print(
                                f"[WARNING] Error during parameter verification after setup(): {e}. "
                                f"Continuing - parameters were already set before setup()."
                            )

                    # CRITICAL: Check for extreme parameter combinations before power flow
                    # This helps identify combinations that may cause convergence issues
                    # Convert M to H for the check (H = M / 2 for 60 Hz systems)
                    H_value = M / 2.0 if M is not None else None
                    if H_value is None and H_range is not None:
                        # If M is not available, try to estimate from H_range
                        H_value = (H_range[0] + H_range[1]) / 2.0

                    if use_load_mode and Load_P is not None and H_value is not None:
                        is_extreme, extreme_warning = check_extreme_parameter_combination(
                            H=H_value,
                            D=D,
                            Load_P=Load_P,
                            H_range=H_range[:2] if H_range is not None else None,
                            D_range=D_range[:2] if D_range is not None else None,
                            load_range=load_range[:2] if load_range is not None else None,
                            verbose=verbose,
                        )
                        if is_extreme:
                            # Warn but don't reject - let power flow attempt to converge
                            # If it fails, it will be rejected by power flow convergence check
                            if verbose:
                                print(f"[WARNING] {extreme_warning}")
                                print(
                                    f"         Attempting power flow, but convergence may fail. "
                                    f"Scenario will be rejected if power flow does not converge."
                                )
                    elif Pm_to_use is not None and H_value is not None:
                        # Check for extreme combinations in Pm variation mode
                        is_extreme, extreme_warning = check_extreme_parameter_combination(
                            H=H_value,
                            D=D,
                            Pm=Pm_to_use,
                            H_range=H_range[:2] if H_range is not None else None,
                            D_range=D_range[:2] if D_range is not None else None,
                            load_range=Pm_range[:2] if Pm_range is not None else None,
                            verbose=verbose,
                            # Priority 1: Use absolute thresholds based on physical reality
                            H_min_absolute=2.0,  # seconds
                            D_min_absolute=0.5,  # pu
                            D_min_critical=0.3,  # pu
                            Load_max_absolute=0.9,  # pu
                        )
                        if is_extreme and verbose:
                            print(f"[WARNING] {extreme_warning}")

                    # CRITICAL: For alpha load variation, set constant generator power BEFORE power flow
                    # SMIB only: Generator Pm is CONSTANT (0.9 pu), load varies; infinite bus absorbs.
                    # Multimachine: do NOT set any generator; only loads were scaled - let PF solve.
                    if use_load_mode and alpha is not None:
                        is_multimachine_load_variation = hasattr(ss, "GENCLS") and ss.GENCLS.n > 1
                        if not is_multimachine_load_variation:
                            # Set PV.p0 to constant (0.9 pu) - generator scheduled output (SMIB)
                            pv_setpoint = 0.9  # pu - constant generator output
                            if hasattr(ss, "PV") and ss.PV.n > 0:
                                if hasattr(ss.PV, "p0") and hasattr(ss.PV.p0, "v"):
                                    if len(ss.PV.p0.v) > 0:
                                        ss.PV.p0.v[0] = pv_setpoint
                                        workflow_steps.append("PV_p0_set")  # Track workflow order
                                        if verbose:
                                            print(
                                                f"[GEN POWER] Set PV.p0 = {pv_setpoint:.6f} pu (constant generator output)"
                                            )
                            # CRITICAL: Set GENCLS.tm0 explicitly (does NOT auto-copy from PV.p0)
                            if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                                if gen_idx_after_setup < len(ss.GENCLS.tm0.v):
                                    ss.GENCLS.tm0.v[gen_idx_after_setup] = pv_setpoint
                                    workflow_steps.append("GENCLS_tm0_set")  # Track workflow order
                                    if verbose:
                                        print(
                                            f"[GEN POWER] Set GENCLS.tm0[{gen_idx_after_setup}] = {pv_setpoint:.6f} pu (mechanical power)"
                                        )

                    # CRITICAL: Run power flow FIRST to establish steady-state equilibrium
                    # This must be done BEFORE running TDS
                    # Power flow will use the Pm value set above to establish equilibrium
                    if hasattr(ss, "PFlow"):
                        # CRITICAL: Reset convergence AND initialized flags to force complete re-run
                        # When reusing system (e.g., after CCT finding with different Pm),
                        # we must force ANDES to completely re-solve power flow, not use cached solution
                        if hasattr(ss.PFlow, "converged"):
                            ss.PFlow.converged = False
                        if hasattr(ss.PFlow, "initialized"):
                            ss.PFlow.initialized = False

                        # Also mark TDS as not initialized to force it to re-read power flow solution
                        if hasattr(ss, "TDS") and hasattr(ss.TDS, "initialized"):
                            ss.TDS.initialized = False

                        # For load variation, verify load is set correctly before power flow
                        if use_load_mode and Load_P is not None:
                            if hasattr(ss, "PQ") and ss.PQ.n > 0:
                                try:
                                    load_idx_check = 0
                                    if hasattr(ss.PQ.p0, "v"):
                                        current_load_before_pf = (
                                            ss.PQ.p0.v[load_idx_check]
                                            if hasattr(ss.PQ.p0.v, "__getitem__")
                                            and len(ss.PQ.p0.v) > load_idx_check
                                            else ss.PQ.p0.v
                                        )
                                        if verbose and abs(current_load_before_pf - Load_P) > 1e-6:
                                            print(
                                                f"[DEBUG] Load before power flow:"
                                                f"{current_load_before_pf:.6f} pu"
                                                f"(expected: {Load_P:.6f} pu)"
                                            )
                                except Exception:
                                    pass

                        # Run power flow (will re-solve from scratch)
                        try:
                            # WORKFLOW VERIFICATION: Verify order before power flow
                            expected_before_pf = ["load", "setup", "M_D_set"]
                            if (
                                use_load_mode
                                and alpha is not None
                                and not (hasattr(ss, "GENCLS") and ss.GENCLS.n > 1)
                            ):
                                expected_before_pf.extend(["PV_p0_set", "GENCLS_tm0_set"])
                            missing_steps = [
                                step for step in expected_before_pf if step not in workflow_steps
                            ]
                            # Workflow validation removed for cleaner output
                            # if missing_steps and verbose:
                            #     print(
                            #         f"[WORKFLOW WARNING] Missing steps before power flow: {missing_steps}. "
                            #         f"Current order: {' → '.join(workflow_steps)}"
                            #     )

                            # Skip power flow if redispatch already ran it
                            if not (use_redispatch_mode and REDISPATCH_AVAILABLE):
                                ss.PFlow.run()
                            workflow_steps.append("power_flow")  # Track workflow order
                        except Exception as e:
                            error_summary["powerflow_failed"] += 1
                            if verbose:
                                print(
                                    f"Warning: Power flow failed for M={M:.2f}, D={D:.2f},"
                                    f"tc={tc:.3f}: {e}. Skipping."
                                )
                            continue

                        # CRITICAL: Verify power flow converged using verification helper
                        pf_converged, pf_error = verify_power_flow_converged(ss)

                        # DEBUG: For load variation, verify load and power flow status
                        if use_load_mode and Load_P is not None and verbose:
                            if hasattr(ss, "PQ") and ss.PQ.n > 0:
                                try:
                                    load_idx_check = 0
                                    if hasattr(ss.PQ.p0, "v"):
                                        load_after_pf = (
                                            ss.PQ.p0.v[load_idx_check]
                                            if hasattr(ss.PQ.p0.v, "__getitem__")
                                            and len(ss.PQ.p0.v) > load_idx_check
                                            else ss.PQ.p0.v
                                        )
                                        print(
                                            f"[DEBUG] [LOAD VARIATION] After power flow: "
                                            f"Load={load_after_pf:.6f} pu (expected: {Load_P:.6f}"
                                            f"pu),"
                                            f"converged={pf_converged}"
                                        )
                                except Exception as e:
                                    print(
                                        f"[DEBUG] [LOAD VARIATION] Could not check load after power"
                                        f"flow: {e}"
                                    )

                        if not pf_converged:
                            # Try one more time with reset
                            if hasattr(ss.PFlow, "converged"):
                                ss.PFlow.converged = False
                            if hasattr(ss.PFlow, "initialized"):
                                ss.PFlow.initialized = False
                            try:
                                # Skip power flow if redispatch already ran it
                                if not (use_redispatch_mode and REDISPATCH_AVAILABLE):
                                    ss.PFlow.run()
                            except Exception:
                                pass
                            # Re-verify
                            pf_converged, pf_error = verify_power_flow_converged(ss)
                            if not pf_converged:
                                error_summary["powerflow_no_converge"] += 1
                                data_quality_metrics["rejected_scenarios"] += 1
                                reason = "power_flow_no_converge"
                                data_quality_metrics["rejection_reasons"][reason] = (
                                    data_quality_metrics["rejection_reasons"].get(reason, 0) + 1
                                )
                                if verbose:
                                    print(
                                        f"[REJECT] Power flow did not converge for M={M:.2f},"
                                        f"D={D:.2f}, Pm={Pm_to_use:.3f}, tc={tc:.3f}: {pf_error}."
                                        f"Skipping."
                                    )
                                continue

                        # CRITICAL: For multimachine + load variation, set GENCLS.tm0 from Pe after PF.
                        # ANDES raw/dyr may leave tm0 zero; Pe.v is often empty until TDS - use dae.y as fallback.
                        # TDS needs Pm = Pe in steady state or the swing equation is inconsistent and init can fail.
                        if (
                            use_load_mode
                            and hasattr(ss, "GENCLS")
                            and ss.GENCLS.n > 1
                            and hasattr(ss.GENCLS, "tm0")
                            and hasattr(ss.GENCLS.tm0, "v")
                        ):
                            try:
                                tm0v = ss.GENCLS.tm0.v
                                n_gen = ss.GENCLS.n
                                tm0_len = len(tm0v) if hasattr(tm0v, "__len__") else 1
                                pe_values_set = 0
                                # Method 1: from GENCLS.Pe.v (often empty after PF with raw/dyr)
                                if hasattr(ss.GENCLS, "Pe") and hasattr(ss.GENCLS.Pe, "v"):
                                    Pev = ss.GENCLS.Pe.v
                                    pe_len = len(Pev) if hasattr(Pev, "__len__") else 1
                                    for i in range(min(n_gen, pe_len, tm0_len)):
                                        pe_val = Pev[i] if hasattr(Pev, "__len__") else float(Pev)
                                        if hasattr(tm0v, "__setitem__"):
                                            tm0v[i] = float(pe_val)
                                            pe_values_set += 1
                                # Method 2: from ss.dae.y via GENCLS.Pe.a (populated after power flow)
                                if (
                                    pe_values_set == 0
                                    and hasattr(ss, "dae")
                                    and hasattr(ss.dae, "y")
                                    and ss.dae.y is not None
                                ):
                                    if hasattr(ss.GENCLS, "Pe") and hasattr(ss.GENCLS.Pe, "a"):
                                        pe_a = ss.GENCLS.Pe.a
                                        if hasattr(pe_a, "__len__") and len(pe_a) >= n_gen:
                                            for i in range(min(n_gen, tm0_len)):
                                                pe_a_idx = int(pe_a[i])
                                                if 0 <= pe_a_idx < len(ss.dae.y) and hasattr(
                                                    tm0v, "__setitem__"
                                                ):
                                                    tm0v[i] = float(ss.dae.y[pe_a_idx])
                                                    pe_values_set += 1
                                if pe_values_set > 0:
                                    workflow_steps.append("GENCLS_tm0_from_Pe_multimachine")
                                    # Save so we can re-apply after TDS.init() if ANDES resets tm0
                                    try:
                                        tm0v_save = ss.GENCLS.tm0.v
                                        saved_tm0_from_pf_multimachine = [
                                            float(tm0v_save[i])
                                            if hasattr(tm0v_save, "__getitem__")
                                            and i < len(tm0v_save)
                                            else 0.0
                                            for i in range(ss.GENCLS.n)
                                        ]
                                    except Exception:
                                        saved_tm0_from_pf_multimachine = None
                                    if verbose:
                                        print(
                                            f"[GEN POWER] Set GENCLS.tm0[i] = Pe[i] for {pe_values_set} generators "
                                            "(multimachine steady state Pm = Pe)."
                                        )
                                elif verbose:
                                    print(
                                        "[WARNING] Could not set GENCLS.tm0 from Pe or dae.y after PF "
                                        "(Pe.v empty and dae.y/Pe.a not available)."
                                    )
                                # Fallback: set tm0 from uploaded case (case_default_pm), scaled by alpha
                                if (
                                    pe_values_set == 0
                                    and case_default_pm is not None
                                    and len(case_default_pm) >= n_gen
                                    and hasattr(tm0v, "__setitem__")
                                ):
                                    alpha_scale = float(alpha) if alpha is not None else 1.0
                                    for i in range(min(n_gen, len(case_default_pm), tm0_len)):
                                        tm0v[i] = float(case_default_pm[i]) * alpha_scale
                                    workflow_steps.append("GENCLS_tm0_from_case_default_pm")
                                    if verbose:
                                        print(
                                            f"[GEN POWER] Set GENCLS.tm0 from case_default_pm "
                                            f"(× alpha={alpha_scale:.4f}); uploaded case Pm used."
                                        )
                            except Exception as e_tm0:
                                if verbose:
                                    print(
                                        f"[WARNING] Could not set GENCLS.tm0 from Pe after PF: {e_tm0}"
                                    )

                        # CRITICAL: For SMIB, verify power balance via infinite bus Pm
                        # In SMIB with no load, infinite bus Pm should be -Pm (negative of main generator)
                        # Infinite bus is always PV generator, so check PV.p0 only
                        if (
                            hasattr(ss.GENCLS, "tm0")
                            and hasattr(ss.GENCLS.tm0, "v")
                            and ss.GENCLS.n >= 1
                        ):
                            try:
                                # Get main generator Pm
                                tm0v = ss.GENCLS.tm0.v
                                tm0_main = (
                                    tm0v[gen_idx_after_setup]
                                    if hasattr(tm0v, "__getitem__")
                                    else tm0v
                                )

                                # Get infinite bus power from PV generator
                                inf_bus_pm = 0.0
                                found_inf_bus = False

                                if hasattr(ss, "PV") and ss.PV.n > 0:
                                    if hasattr(ss.PV, "p0") and hasattr(ss.PV.p0, "v"):
                                        pv_p0 = ss.PV.p0.v
                                        # For SMIB: Use the OTHER PV generator (infinite bus, not main gen)
                                        if hasattr(pv_p0, "__len__") and len(pv_p0) > 1:
                                            # CRITICAL FIX: Read from index 1 (infinite bus), not index 0 (main gen)
                                            # We set PV.p0[0] = Pm for main generator
                                            # So infinite bus should be at PV.p0[1]
                                            inf_bus_pm = float(pv_p0[1])
                                            found_inf_bus = True
                                            if verbose:
                                                print(
                                                    f"[DEBUG] Found PV generator (infinite bus): "
                                                    f"PV.p0[1] = {inf_bus_pm:.6f} pu"
                                                )

                                if found_inf_bus:
                                    # Verify power balance: main + infinite ≈ 0
                                    expected_inf_pm = -tm0_main
                                    inf_mismatch_pct = (
                                        100.0
                                        * abs(inf_bus_pm - expected_inf_pm)
                                        / (abs(expected_inf_pm) + 1e-12)
                                    )

                                    # Removed verbose power balance debug for cleaner output

                                    if inf_mismatch_pct > 1.0:
                                        error_summary["power_balance_violated"] += 1
                                        data_quality_metrics["rejected_scenarios"] += 1
                                        reason = "smib_power_balance_violated"
                                        data_quality_metrics["rejection_reasons"][reason] = (
                                            data_quality_metrics["rejection_reasons"].get(reason, 0)
                                            + 1
                                        )
                                        # Only print error if power balance is significantly violated
                                        if verbose and inf_mismatch_pct > 10.0:
                                            print(
                                                f"[REJECT] SMIB power balance violated: "
                                                f"Main Pm={tm0_main:.6f}, Inf (PV) Pm={inf_bus_pm:.6f} "
                                                f"(expected {expected_inf_pm:.6f}, error={inf_mismatch_pct:.2f}%). "
                                                f"Skipping."
                                            )
                                        continue
                                    # Removed success message for cleaner output
                                else:
                                    if verbose:
                                        print(
                                            f"[WARNING] No PV generator found for infinite bus. "
                                            f"Skipping SMIB power balance check."
                                        )
                            except Exception as e:
                                if verbose and DEBUG_TDS:
                                    print(f"[DEBUG] SMIB power balance check failed: {e}")

                        # CRITICAL: Extract or verify Pm after power flow
                        # NEW: For load variation, extract Pm (generator adjusts to meet load)
                        # For Pm variation, verify Pm matches requested value
                        if use_load_mode:
                            # Load variation: Extract Pm from system after power flow
                            # Generator adjusts to meet load, so Pm is determined by power flow
                            try:
                                # DEBUG: Check all generators' Pm values to find the correct one
                                if (
                                    verbose
                                    and hasattr(ss.GENCLS, "tm0")
                                    and hasattr(ss.GENCLS.tm0, "v")
                                ):
                                    tm0v = ss.GENCLS.tm0.v
                                    if hasattr(tm0v, "__len__"):
                                        print(
                                            f"[DEBUG] [LOAD VARIATION] All generators' Pm after"
                                            f"power flow:"
                                        )
                                        for i in range(min(len(tm0v), ss.GENCLS.n)):
                                            print(f"  Generator {i}: tm0.v[{i}] = {tm0v[i]:.6f} pu")

                                # Try to find the generator that matches the load (positive Pm, close to Load_P)
                                # This handles cases where gen_idx_after_setup might point to infinite bus
                                # FIXED: Also try extracting from Pe (electrical power) if tm0.v is zero
                                # In steady state, Pe = Pm, so this is a valid fallback
                                Pm_to_use = None
                                best_gen_idx = gen_idx_after_setup
                                extraction_method = None

                                # METHOD 1: Try extracting from tm0.v (mechanical power setpoint)
                                if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                                    tm0v = ss.GENCLS.tm0.v
                                    if hasattr(tm0v, "__len__"):
                                        # Find generator with Pm closest to Load_P (in absolute value)
                                        # Note: In SMIB, generator may have negative Pm if set up as load
                                        # We look for Pm closest to Load_P in absolute value
                                        best_match = None
                                        best_error = float("inf")
                                        for i in range(min(len(tm0v), ss.GENCLS.n)):
                                            pm_val = float(tm0v[i])
                                            # Calculate error as absolute difference
                                            # Accept both positive and negative Pm (SMIB may have negative)
                                            error = abs(abs(pm_val) - abs(Load_P))
                                            if error < best_error:
                                                best_error = error
                                                best_match = i
                                                Pm_to_use = abs(pm_val)  # Use absolute value for Pm

                                        if best_match is not None:
                                            best_gen_idx = best_match
                                            # CRITICAL: Update gen_idx_after_setup to use the correct generator
                                            # This ensures trajectory extraction uses the right generator
                                            gen_idx_after_setup = best_gen_idx
                                            extraction_method = "tm0.v"
                                            if verbose:
                                                print(
                                                    f"[DEBUG] [LOAD VARIATION] Using Generator"
                                                    f"{best_gen_idx}"
                                                    f"(Pm={Pm_to_use:.6f} pu from tm0.v) -"
                                                    f"better match to Load={Load_P:.6f} pu. "
                                                    f"Updated gen_idx_after_setup to {best_gen_idx}."
                                                )

                                    # Fallback: use gen_idx_after_setup if we couldn't find a better match
                                    if Pm_to_use is None:
                                        pm_fallback = (
                                            tm0v[gen_idx_after_setup]
                                            if hasattr(tm0v, "__getitem__")
                                            and len(tm0v) > gen_idx_after_setup
                                            else (
                                                float(tm0v)
                                                if not hasattr(tm0v, "__len__")
                                                else tm0v[0]
                                            )
                                        )
                                        Pm_to_use = abs(float(pm_fallback))  # Use absolute value
                                        extraction_method = "tm0.v (fallback)"

                                # METHOD 2: Try extracting from P0 if tm0.v failed or is zero
                                if (
                                    (Pm_to_use is None or abs(Pm_to_use) < 0.001)
                                    and hasattr(ss.GENCLS, "P0")
                                    and hasattr(ss.GENCLS.P0, "v")
                                ):
                                    P0v = ss.GENCLS.P0.v
                                    if hasattr(P0v, "__len__"):
                                        # Similar logic for P0 (use absolute value matching)
                                        best_match = None
                                        best_error = float("inf")
                                        for i in range(min(len(P0v), ss.GENCLS.n)):
                                            p0_val = float(P0v[i])
                                            # Calculate error as absolute difference
                                            # Accept both positive and negative P0 (SMIB may have negative)
                                            error = abs(abs(p0_val) - abs(Load_P))
                                            if error < best_error:
                                                best_error = error
                                                best_match = i
                                                Pm_to_use = abs(p0_val)  # Use absolute value for Pm

                                        if best_match is not None:
                                            best_gen_idx = best_match
                                            # CRITICAL: Update gen_idx_after_setup to use the correct generator
                                            gen_idx_after_setup = best_gen_idx
                                            extraction_method = "P0.v"
                                            if verbose:
                                                print(
                                                    f"[DEBUG] [LOAD VARIATION] Using Generator"
                                                    f"{best_gen_idx}"
                                                    f"(Pm={Pm_to_use:.6f} pu from P0.v) -"
                                                    f"better match to Load={Load_P:.6f} pu. "
                                                    f"Updated gen_idx_after_setup to {best_gen_idx}."
                                                )

                                    if Pm_to_use is None or abs(Pm_to_use) < 0.001:
                                        p0_fallback = (
                                            P0v[gen_idx_after_setup]
                                            if hasattr(P0v, "__getitem__")
                                            and len(P0v) > gen_idx_after_setup
                                            else (
                                                float(P0v)
                                                if not hasattr(P0v, "__len__")
                                                else P0v[0]
                                            )
                                        )
                                        if abs(float(p0_fallback)) > 0.001:
                                            Pm_to_use = abs(
                                                float(p0_fallback)
                                            )  # Use absolute value
                                            extraction_method = "P0.v (fallback)"

                                # METHOD 3: Extract from Pe (electrical power) if tm0.v and P0.v are zero
                                # In steady state before fault, Pe = Pm, so this is valid
                                if Pm_to_use is None or abs(Pm_to_use) < 0.001:
                                    if hasattr(ss.GENCLS, "Pe") and hasattr(ss.GENCLS.Pe, "v"):
                                        Pev = ss.GENCLS.Pe.v
                                        if hasattr(Pev, "__len__") and len(Pev) > 0:
                                            # Find generator with Pe closest to Load_P
                                            best_match = None
                                            best_error = float("inf")
                                            for i in range(min(len(Pev), ss.GENCLS.n)):
                                                pe_val = float(Pev[i])
                                                # In steady state, Pe = Pm, so use Pe as Pm
                                                error = abs(abs(pe_val) - abs(Load_P))
                                                if error < best_error:
                                                    best_error = error
                                                    best_match = i
                                                    Pm_to_use = abs(pe_val)  # Use absolute value

                                            if best_match is not None:
                                                best_gen_idx = best_match
                                                gen_idx_after_setup = best_gen_idx
                                                extraction_method = "Pe.v (steady state Pe = Pm)"
                                                if verbose:
                                                    print(
                                                        f"[DEBUG] [LOAD VARIATION] Using Generator"
                                                        f"{best_gen_idx}"
                                                        f"(Pm={Pm_to_use:.6f} pu from Pe.v) -"
                                                        f"extracted from electrical power (Pe = Pm in steady state). "
                                                        f"Updated gen_idx_after_setup to {best_gen_idx}."
                                                    )
                                        elif not hasattr(Pev, "__len__"):
                                            # Scalar Pe value
                                            pe_val = abs(float(Pev))
                                            if pe_val > 0.001:
                                                Pm_to_use = pe_val
                                                extraction_method = (
                                                    "Pe.v (scalar, steady state Pe = Pm)"
                                                )
                                                if verbose:
                                                    print(
                                                        f"[DEBUG] [LOAD VARIATION] Extracted Pm={Pm_to_use:.6f} pu "
                                                        f"from Pe.v (scalar, steady state Pe = Pm)."
                                                    )

                                # METHOD 3b: Use Pm from uploaded case (case_default_pm), scaled by alpha
                                if (
                                    (Pm_to_use is None or abs(Pm_to_use) < 0.001)
                                    and case_default_pm is not None
                                    and len(case_default_pm) > 0
                                ):
                                    idx = (
                                        best_gen_idx
                                        if best_gen_idx is not None
                                        else gen_idx_after_setup
                                    )
                                    idx = min(idx, len(case_default_pm) - 1)
                                    alpha_scale = float(alpha) if alpha is not None else 1.0
                                    Pm_to_use = abs(float(case_default_pm[idx]) * alpha_scale)
                                    extraction_method = "case_default_pm (uploaded case × alpha)"
                                    if verbose:
                                        print(
                                            f"[DEBUG] [LOAD VARIATION] Using Pm from case: "
                                            f"Pm={Pm_to_use:.6f} pu (case_default_pm[{idx}] × alpha={alpha_scale:.4f})."
                                        )

                                # METHOD 4: FALLBACK - Use Load_P + estimated losses as Pm
                                # In load variation mode, if all extraction methods fail, use load value
                                # This is valid because in steady state: Pm ≈ Load + Losses
                                if (
                                    Pm_to_use is None or abs(Pm_to_use) < 0.001
                                ) and Load_P is not None:
                                    # Estimate losses (2-5% typical for SMIB)
                                    estimated_losses = 0.03 * Load_P  # 3% of load as typical losses
                                    Pm_to_use = Load_P + estimated_losses
                                    extraction_method = "Load_P + estimated losses (fallback)"
                                    if verbose:
                                        print(
                                            f"[DEBUG] [LOAD VARIATION] Using fallback: Pm={Pm_to_use:.6f} pu "
                                            f"(Load={Load_P:.6f} pu + losses={estimated_losses:.6f} pu). "
                                            f"This is an estimate since tm0.v, P0.v, and Pe.v are not available."
                                        )

                                # CRITICAL: Check if Pm_to_use is still None or zero (extraction failed)
                                if Pm_to_use is None or abs(Pm_to_use) < 0.001:
                                    # This should not happen if extraction logic is correct,
                                    # but add safety check to prevent errors
                                    error_msg = (
                                        f"Pm extraction failed: Could not extract Pm from any generator. "
                                        f"Tried: tm0.v, P0.v, Pe.v, and Load_P fallback. "
                                        f"GENCLS.tm0, GENCLS.P0, GENCLS.Pe may not be available, "
                                        f"and Load_P is None or zero."
                                    )
                                    if verbose:
                                        print(f"[ERROR] {error_msg}")
                                    raise ValueError(error_msg)

                                # CRITICAL: Reject scenarios where Pm approx 0.0 with non-zero load
                                # This indicates a fundamental power balance violation
                                if abs(Pm_to_use) < 0.001 and abs(Load_P) > 0.01:
                                    error_summary["pm_zero_with_load"] = (
                                        error_summary.get("pm_zero_with_load", 0) + 1
                                    )
                                    data_quality_metrics["rejected_scenarios"] += 1
                                    reason = "pm_zero_with_load"
                                    data_quality_metrics["rejection_reasons"][reason] = (
                                        data_quality_metrics["rejection_reasons"].get(reason, 0) + 1
                                    )
                                    if verbose:
                                        print(
                                            f"[REJECT] Critical power balance violation: "
                                            f"Pm={Pm_to_use:.6f} pu (approx 0) with Load={Load_P:.6f} pu. "
                                            f"This indicates Pm extraction failure or power flow"
                                            f"issue."
                                            f"Rejecting scenario."
                                        )
                                    continue

                                # PRIORITY 2: Verify power balance: Pm approx Load + Losses
                                # Try to use actual losses from power flow when available
                                actual_losses = None
                                if hasattr(ss, "PFlow") and hasattr(ss.PFlow, "losses"):
                                    try:
                                        # Try to get actual losses from power flow
                                        pf_losses = ss.PFlow.losses
                                        if pf_losses is not None and isinstance(
                                            pf_losses, (int, float)
                                        ):
                                            actual_losses = float(pf_losses)
                                    except (AttributeError, TypeError, ValueError):
                                        pass

                                # Use actual losses if available, otherwise estimate
                                if actual_losses is not None and actual_losses >= 0:
                                    losses_to_use = actual_losses
                                    loss_source = "actual (from power flow)"
                                else:
                                    # Fallback: Estimate losses (2-5% typical for SMIB)
                                    # Use load-dependent estimation: 2% base + 1% per 0.1 pu loading above 0.4 pu
                                    base_losses = 0.02 * Pm_to_use
                                    load_dependent_losses = (
                                        max(0, 0.01 * (Pm_to_use - 0.4) / 0.1) * Pm_to_use
                                    )
                                    losses_to_use = base_losses + load_dependent_losses
                                    # Cap at 5% (reasonable maximum for SMIB)
                                    losses_to_use = min(losses_to_use, 0.05 * Pm_to_use)
                                    loss_source = "estimated"

                                power_balance_error = abs(Pm_to_use - Load_P - losses_to_use)

                                # Reject scenarios with large power balance errors (>0.05 pu)
                                # Skip strict rejection for multimachine: PF already balanced the system;
                                # GENCLS.tm0 may be unpopulated (raw/dyr) so we use fallback Pm and loss
                                # estimates can differ, causing a false "violation"
                                is_multimachine = hasattr(ss, "GENCLS") and ss.GENCLS.n > 1
                                if power_balance_error > 0.05 and not is_multimachine:
                                    error_summary["power_balance_violation"] = (
                                        error_summary.get("power_balance_violation", 0) + 1
                                    )
                                    data_quality_metrics["rejected_scenarios"] += 1
                                    reason = "power_balance_violation"
                                    data_quality_metrics["rejection_reasons"][reason] = (
                                        data_quality_metrics["rejection_reasons"].get(reason, 0) + 1
                                    )
                                    if verbose:
                                        print(
                                            f"[REJECT] Power balance violation too large: "
                                            f"Pm={Pm_to_use:.6f} pu, Load={Load_P:.6f} pu, "
                                            f"losses={losses_to_use:.6f} pu ({loss_source}), "
                                            f"error={power_balance_error:.6f} pu (threshold: 0.05"
                                            f"pu)."
                                            f"This indicates power flow convergence issue."
                                            f"Rejecting scenario."
                                        )
                                    continue
                                elif power_balance_error > 0.05 and is_multimachine and verbose:
                                    print(
                                        f"[INFO] Multimachine: power balance check skipped (PF converged). "
                                        f"Pm={Pm_to_use:.6f} pu, Load={Load_P:.6f} pu, "
                                        f"error={power_balance_error:.6f} pu."
                                    )
                                elif (
                                    power_balance_error > 0.02
                                ):  # Warning for moderate errors (tolerance relaxed to 0.02 pu)
                                    # Note: Tolerance relaxed from 0.01 to 0.02 pu to account for:
                                    # - Loss estimation uncertainty (3% is approximate, actual losses may vary)
                                    # - Numerical precision in power flow solution
                                    # - Small discrepancies between estimated and actual losses in SMIB systems
                                    if verbose:
                                        print(
                                            f"[WARNING] Power balance violation in load variation"
                                            f"mode:"
                                            f"Pm={Pm_to_use:.6f} pu, Load={Load_P:.6f} pu, "
                                            f"losses={losses_to_use:.6f} pu ({loss_source}), "
                                            f"error={power_balance_error:.6f} pu (tolerance: 0.02"
                                            f"pu)."
                                            f"This may indicate power flow convergence issue or"
                                            f"loss estimation error."
                                        )
                                elif verbose:
                                    # Only show success message if verbose (reduce noise)
                                    print(
                                        f"[LOAD VARIATION] Power balance OK: Pm={Pm_to_use:.6f} pu"
                                        f"approx"
                                        f"Load={Load_P:.6f} pu + "
                                        f"losses={losses_to_use:.6f} pu ({loss_source}) "
                                        f"(error: {power_balance_error:.6f} pu)"
                                    )
                            except Exception as e:
                                # CRITICAL: Reject scenario if Pm extraction fails completely
                                # Using a default value would create incorrect training data
                                error_summary["pm_extraction_failed"] = (
                                    error_summary.get("pm_extraction_failed", 0) + 1
                                )
                                data_quality_metrics["rejected_scenarios"] += 1
                                reason = "pm_extraction_failed"
                                data_quality_metrics["rejection_reasons"][reason] = (
                                    data_quality_metrics["rejection_reasons"].get(reason, 0) + 1
                                )
                                if verbose:
                                    print(
                                        f"[REJECT] Could not extract Pm after power flow in load "
                                        f"variation mode: {e}. "
                                        f"Rejecting scenario to avoid incorrect training data."
                                    )
                                continue
                        elif Pm_to_use is not None:
                            # Pm variation mode: Verify Pm matches requested value (1% tolerance)
                            # NOTE: This section only runs in Pm variation mode, not load variation mode

                            # CRITICAL: Verify Pm after power flow matches Pm_to_use (for consistency)
                            # Extract Pm from power flow result and verify it matches
                            # This matches the fix in smib_batch_tds.py (lines 1240-1286)
                            Pm_after_pf = Pm_to_use  # Initialize with expected value
                            if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                                if gen_idx_after_setup < len(ss.GENCLS.tm0.v):
                                    Pm_after_pf = float(ss.GENCLS.tm0.v[gen_idx_after_setup])

                                    # CRITICAL: If Pm_after_pf is zero or very small, we might be using the wrong generator
                                    # (e.g., infinite bus). Use find_main_generator_index to get the correct generator.
                                    if abs(Pm_after_pf) < 1e-6 and ss.GENCLS.n > 1:
                                        # Removed verbose warning for cleaner output
                                        # Use find_main_generator_index to get the main generator (not infinite bus)
                                        if (
                                            CCT_FINDER_AVAILABLE
                                            and find_main_generator_index is not None
                                        ):
                                            gen_idx_after_setup = find_main_generator_index(ss)
                                            # Verify this generator has non-zero Pm
                                            if gen_idx_after_setup < len(ss.GENCLS.tm0.v):
                                                Pm_candidate = float(
                                                    ss.GENCLS.tm0.v[gen_idx_after_setup]
                                                )
                                                if abs(Pm_candidate) > 1e-6:
                                                    Pm_after_pf = Pm_candidate
                                                    if verbose:
                                                        sign_note = (
                                                            " (negative - may be sign convention)"
                                                            if Pm_candidate < 0
                                                            else ""
                                                        )
                                                        print(
                                                            f"[FIXED] Using generator {gen_idx_after_setup} (from find_main_generator_index) with Pm={Pm_after_pf:.6f} pu{sign_note}"
                                                        )
                                                else:
                                                    # Fallback: search for generator with non-zero Pm (removed verbose warning)
                                                    for i in range(ss.GENCLS.n):
                                                        # Skip infinite bus (M > 1e6)
                                                        if hasattr(ss.GENCLS, "M") and hasattr(
                                                            ss.GENCLS.M, "v"
                                                        ):
                                                            if i < len(ss.GENCLS.M.v):
                                                                M_candidate = float(
                                                                    ss.GENCLS.M.v[i]
                                                                )
                                                                if M_candidate > 1e6:
                                                                    continue  # Skip infinite bus
                                                        if i < len(ss.GENCLS.tm0.v):
                                                            Pm_candidate = float(ss.GENCLS.tm0.v[i])
                                                            if abs(Pm_candidate) > 1e-6:
                                                                gen_idx_after_setup = i
                                                                Pm_after_pf = Pm_candidate
                                                                if verbose:
                                                                    sign_note = (
                                                                        " (negative - may be sign convention)"
                                                                        if Pm_candidate < 0
                                                                        else ""
                                                                    )
                                                                    print(
                                                                        f"[FIXED] Using generator {gen_idx_after_setup} with Pm={Pm_after_pf:.6f} pu{sign_note}"
                                                                    )
                                                                break
                                        else:
                                            # Fallback: search for generator with non-zero Pm (skip infinite bus)
                                            for i in range(ss.GENCLS.n):
                                                # Skip infinite bus (M > 1e6)
                                                if hasattr(ss.GENCLS, "M") and hasattr(
                                                    ss.GENCLS.M, "v"
                                                ):
                                                    if i < len(ss.GENCLS.M.v):
                                                        M_candidate = float(ss.GENCLS.M.v[i])
                                                        if M_candidate > 1e6:
                                                            continue  # Skip infinite bus
                                                if i < len(ss.GENCLS.tm0.v):
                                                    Pm_candidate = float(ss.GENCLS.tm0.v[i])
                                                    if abs(Pm_candidate) > 1e-6:
                                                        gen_idx_after_setup = i
                                                        Pm_after_pf = Pm_candidate
                                                        if verbose:
                                                            sign_note = (
                                                                " (negative - may be sign convention)"
                                                                if Pm_candidate < 0
                                                                else ""
                                                            )
                                                            print(
                                                                f"[FIXED] Using generator {gen_idx_after_setup} with Pm={Pm_after_pf:.6f} pu{sign_note}"
                                                            )
                                                        break

                                    # Verify it matches Pm_to_use (should be same since we set it)
                                    if (
                                        abs(Pm_after_pf - Pm_to_use) > 1e-3
                                    ):  # Allow small numerical differences
                                        if verbose:
                                            print(
                                                f"[WARNING] Pm mismatch after power flow: {Pm_after_pf:.6f} != {Pm_to_use:.6f} "
                                                f"(diff: {abs(Pm_after_pf - Pm_to_use):.6f} pu). Using Pm_to_use."
                                            )
                                        # Use Pm_to_use (expected value) for consistency
                                        Pm_after_pf = Pm_to_use

                            generator_setpoints = {gen_idx_after_setup: Pm_to_use}
                            all_match, errors = verify_generator_setpoints(
                                ss, generator_setpoints, tolerance=0.01
                            )
                            if not all_match:
                                # Try to fix the mismatch by re-setting Pm and re-running power flow
                                if verbose:
                                    print(
                                        f"[WARNING] [Pm VARIATION] Pm setpoint mismatch after power"
                                        f"flow, attempting to fix:"
                                    )
                                    for error in errors:
                                        print(f"  {error}")

                                # Re-set Pm programmatically (similar to CCT finder approach)
                                try:
                                    if hasattr(ss.GENCLS, "P0") and hasattr(ss.GENCLS.P0, "v"):
                                        if hasattr(ss.GENCLS.P0.v, "__setitem__"):
                                            ss.GENCLS.P0.v[gen_idx_after_setup] = Pm_to_use
                                        else:
                                            ss.GENCLS.P0.v = Pm_to_use
                                    if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                                        if hasattr(ss.GENCLS.tm0.v, "__setitem__"):
                                            ss.GENCLS.tm0.v[gen_idx_after_setup] = Pm_to_use
                                        else:
                                            ss.GENCLS.tm0.v = Pm_to_use

                                    # Re-run power flow with corrected Pm
                                    if hasattr(ss, "PFlow"):
                                        ss.PFlow.initialized = False
                                        # Skip power flow if redispatch already ran it
                                        if not (use_redispatch_mode and REDISPATCH_AVAILABLE):
                                            ss.PFlow.run()

                                    # Re-verify
                                    all_match, errors = verify_generator_setpoints(
                                        ss, generator_setpoints, tolerance=0.01
                                    )
                                except Exception as fix_err:
                                    if verbose:
                                        print(
                                            f"[WARNING] [Pm VARIATION] Failed to fix Pm mismatch:"
                                            f"{fix_err}"
                                        )

                                if not all_match:
                                    # Still mismatched after fix attempt - reject
                                    error_summary["pm_mismatch_after_pf"] = (
                                        error_summary.get("pm_mismatch_after_pf", 0) + 1
                                    )
                                    data_quality_metrics["rejected_scenarios"] += 1
                                    reason = "pm_mismatch_after_pf"
                                    data_quality_metrics["rejection_reasons"][reason] = (
                                        data_quality_metrics["rejection_reasons"].get(reason, 0) + 1
                                    )
                                    if verbose:
                                        print(
                                            f"[REJECT] [Pm VARIATION] Pm setpoint mismatch after"
                                            f"power flow (fix failed):"
                                        )
                                        for error in errors:
                                            print(f"  {error}")
                                    continue

                            # Track successful Pm verification
                            data_quality_metrics[
                                "pm_verification_pass_rate"
                            ] = data_quality_metrics["successful_scenarios"] / max(
                                data_quality_metrics["total_scenarios"], 1
                            )

                        # Note: Pm verification is done above using verify_generator_setpoints()
                        # This ensures tm0 matches requested Pm after power flow
                    else:
                        if verbose:
                            print(
                                f"Warning: PFlow module not available. Skipping combination"
                                f"{current_combination}."
                            )
                        continue

                    # Configure TDS parameters (following ANDES manual: fixt, tstep, shrinkt)
                    # CRITICAL: Set time step BEFORE init() to maximize chance of it being accepted
                    if time_step is not None:
                        # User wants to force a specific time step
                        # Use ANDES's proper configuration parameters: fixt, tstep, shrinkt
                        if hasattr(ss.TDS, "config"):
                            # CRITICAL: Set fixt=1 for fixed step mode (ANDES manual parameter)
                            if hasattr(ss.TDS.config, "fixt"):
                                ss.TDS.config.fixt = 1  # Fixed step mode (1 = fixed, 0 = adaptive)
                            # Also try alternative names for compatibility
                            if hasattr(ss.TDS.config, "fixed_step"):
                                ss.TDS.config.fixed_step = True

                            # CRITICAL: Set tstep (initial/constant step size) - this is the key parameter!
                            # tstep defines the constant step size in fixed mode or starting value in adaptive mode
                            if hasattr(ss.TDS.config, "tstep"):
                                ss.TDS.config.tstep = time_step
                            # Also set h as fallback (some ANDES versions use h instead of tstep)
                            if hasattr(ss.TDS.config, "h"):
                                ss.TDS.config.h = time_step
                            if hasattr(ss.TDS.config, "dt"):
                                ss.TDS.config.dt = time_step

                            # Set shrinkt to prevent step shrinking when Newton iterations fail
                            # shrinkt=0 means don't allow shrinking, shrinkt=1 allows shrinking
                            if hasattr(ss.TDS.config, "shrinkt"):
                                ss.TDS.config.shrinkt = (
                                    0  # Don't allow shrinking for strict fixed step
                                )

                            # Disable automatic time step calculation
                            if hasattr(ss.TDS.config, "auto_h"):
                                ss.TDS.config.auto_h = False
                            if hasattr(ss.TDS.config, "calc_h"):
                                ss.TDS.config.calc_h = False
                            if hasattr(ss.TDS.config, "auto_step"):
                                ss.TDS.config.auto_step = False
                            if hasattr(ss.TDS.config, "adaptive"):
                                ss.TDS.config.adaptive = False
                        # Also try setting in solver if available
                        if hasattr(ss.TDS, "solver") and hasattr(ss.TDS.solver, "h"):
                            ss.TDS.solver.h = time_step
                        # Try setting in TDS directly
                        if hasattr(ss.TDS, "h"):
                            ss.TDS.h = time_step
                    # If time_step is None, ANDES will use its automatic calculation

                    if simulation_time is not None:
                        ss.TDS.config.tf = simulation_time
                    ss.TDS.config.tol = 1e-4

                    # CRITICAL: Disable automatic stability criteria checking
                    # This allows simulation to run to completion so we can extract all data
                    if hasattr(ss.TDS.config, "criteria"):
                        ss.TDS.config.criteria = 0  # 0 = disabled

                    # Enable plotter to store time series data (required for data extraction)
                    if hasattr(ss.TDS.config, "plot"):
                        ss.TDS.config.plot = True
                    if hasattr(ss.TDS.config, "save_plt"):
                        ss.TDS.config.save_plt = True
                    if hasattr(ss.TDS.config, "store"):
                        ss.TDS.config.store = True
                    # CRITICAL: Set save_interval=1 to save every time step (not downsampled to 30 Hz)
                    # This ensures output matches the requested time_step (e.g., 1 ms instead of 33.33 ms)
                    # Must be set BEFORE init() to take effect
                    if hasattr(ss.TDS.config, "save_interval"):
                        ss.TDS.config.save_interval = 1  # Save every time step
                    # Also try alternative configuration names (ANDES version-dependent)
                    if hasattr(ss.TDS.config, "save_step"):
                        ss.TDS.config.save_step = 1
                    if hasattr(ss.TDS.config, "output_step"):
                        ss.TDS.config.output_step = 1

                    # Disable progress bars to avoid tqdm_notebook/ipywidgets dependency issues
                    if hasattr(ss.TDS.config, "no_tqdm"):
                        ss.TDS.config.no_tqdm = True
                    if hasattr(ss.TDS.config, "show_progress"):
                        ss.TDS.config.show_progress = False
                    if hasattr(ss.TDS.config, "progress"):
                        ss.TDS.config.progress = False

                    # Note: Pm verification was done after power flow above
                    # Since Pm is set via case file modification, it should be correct
                    # No need to re-verify before TDS.init() - power flow already verified it

                    # PHASE 0: Comprehensive diagnostic - trace where Pe is stored
                    # This helps understand where TDS.init() actually reads from
                    # Run AFTER power flow has completed and Pm has been extracted

                    # PHASE 0.3: Verify case file generator setpoint
                    # CRITICAL: Always check (not just when verbose) to diagnose Pe issue
                    case_file_pm = None
                    if CASE_FILE_MODIFIER_AVAILABLE and get_default_pm_from_case_file is not None:
                        try:
                            case_file_pm = get_default_pm_from_case_file(
                                original_case_path, gen_idx=0
                            )
                            if case_file_pm is not None:
                                print(
                                    f"[PHASE 0.3] Case file default Pm: {case_file_pm:.6f} pu "
                                    f"(expected: {Pm_to_use:.6f} pu if load variation)"
                                )
                            if case_file_pm is not None and abs(case_file_pm - Pm_to_use) > 0.01:
                                print(
                                    f"[PHASE 0.3] [WARNING] Case file has Pm={case_file_pm:.6f} pu, "
                                    f"but we need Pm={Pm_to_use:.6f} pu. "
                                    f"TDS.init() might read from case file instead of power flow!"
                                )
                        except Exception as case_check_err:
                            print(
                                "[PHASE 0.3] [ERROR] Could not check case file default:",
                                str(case_check_err),
                            )

                    # PHASE 0.1 & 0.2: Comprehensive diagnostic - trace where Pe is stored
                    # Run this AFTER power flow to see where Pe is actually stored
                    # CRITICAL: Always run diagnostic (not just when verbose) to diagnose Pe issue
                    if Pm_to_use is not None:
                        try:
                            diagnose_pe_storage_locations(
                                ss=ss,
                                gen_idx=gen_idx_after_setup,
                                expected_pe=Pm_to_use,
                                verbose=True,  # Always verbose for critical diagnostics
                            )
                        except Exception as diag_err:
                            print(f"[PHASE 0] [ERROR] Diagnostic failed: {diag_err}")
                            import traceback

                            print(f"[PHASE 0] [ERROR] Traceback: {traceback.format_exc()}")

                    # CRITICAL: Verify power flow solution Pe matches Pm BEFORE TDS.init()
                    # If Pe from power flow doesn't match Pm, TDS will use wrong initial conditions
                    # Use official ANDES methods to extract Pe from power flow solution
                    pe_from_pf_before_tds = None
                    try:
                        gen_idx_for_pe_check = gen_idx_after_setup

                        # METHOD 1: Try direct Pe.v access (1D array after power flow)
                        if hasattr(ss.GENCLS, "Pe") and hasattr(ss.GENCLS.Pe, "v"):
                            pe_v = ss.GENCLS.Pe.v
                            if verbose and DEBUG_TDS:
                                print(
                                    f"[DEBUG] Pe extraction Method 1: Pe.v exists,"
                                    f"type={type(pe_v)},"
                                    f"hasattr(__len__)={hasattr(pe_v, '__len__')}"
                                )
                            if hasattr(pe_v, "__getitem__") and hasattr(pe_v, "__len__"):
                                try:
                                    pe_len = len(pe_v) if hasattr(pe_v, "__len__") else 0
                                    if verbose and DEBUG_TDS:
                                        pe_shape = pe_v.shape if hasattr(pe_v, "shape") else "N/A"
                                        print(
                                            f"[DEBUG] Pe extraction Method 1: len(pe_v)={pe_len}, "
                                            f"gen_idx_for_pe_check={gen_idx_for_pe_check}, "
                                            f"shape={pe_shape}"
                                        )
                                    if pe_len == 0:
                                        # Empty array - Pe.v not populated after power flow
                                        if verbose and DEBUG_TDS:
                                            print(
                                                f"[DEBUG] Pe extraction Method 1 SKIPPED: "
                                                f"Pe.v is empty (length 0) after power flow. "
                                                f"This is normal - Pe.v may not be populated until"
                                                f"TDS runs."
                                            )
                                    elif len(pe_v) > gen_idx_for_pe_check:
                                        pe_from_pf_before_tds = float(pe_v[gen_idx_for_pe_check])
                                        if verbose and DEBUG_TDS:
                                            print(
                                                f"[DEBUG] Pe extraction Method 1 SUCCESS: "
                                                f"Pe={pe_from_pf_before_tds:.6f} pu"
                                            )
                                    elif len(pe_v) > 0:
                                        pe_from_pf_before_tds = float(pe_v[0])
                                        if verbose and DEBUG_TDS:
                                            print(
                                                f"[DEBUG] Pe extraction Method 1 SUCCESS (using"
                                                f"index 0):"
                                                f"Pe={pe_from_pf_before_tds:.6f} pu"
                                            )
                                    else:
                                        # Try scalar conversion only if not empty
                                        try:
                                            pe_from_pf_before_tds = float(pe_v)
                                            if verbose and DEBUG_TDS:
                                                print(
                                                    f"[DEBUG] Pe extraction Method 1 SUCCESS"
                                                    f"(scalar):"
                                                    f"Pe={pe_from_pf_before_tds:.6f} pu"
                                                )
                                        except (ValueError, TypeError):
                                            if verbose and DEBUG_TDS:
                                                print(
                                                    f"[DEBUG] Pe extraction Method 1 FAILED: "
                                                    f"Cannot convert to scalar (empty array)"
                                                )
                                except (TypeError, AttributeError, ValueError) as e:
                                    if verbose and DEBUG_TDS:
                                        print(f"[DEBUG] Pe extraction Method 1 FAILED: {e}")
                            else:
                                try:
                                    pe_from_pf_before_tds = float(pe_v)
                                    if verbose and DEBUG_TDS:
                                        print(
                                            f"[DEBUG] Pe extraction Method 1 SUCCESS (direct"
                                            f"float):"
                                            f"Pe={pe_from_pf_before_tds:.6f} pu"
                                        )
                                except (TypeError, ValueError) as e:
                                    if verbose and DEBUG_TDS:
                                        print(
                                            f"[DEBUG] Pe extraction Method 1 FAILED (direct float):"
                                            f"{e}"
                                        )
                        else:
                            if verbose and DEBUG_TDS:
                                print(
                                    f"[DEBUG] Pe extraction Method 1 SKIPPED: "
                                    f"hasattr(GENCLS, 'Pe')={hasattr(ss.GENCLS, 'Pe')}, "
                                    f"hasattr(Pe, 'v')={hasattr(ss.GENCLS, 'Pe') and hasattr(ss.GENCLS.Pe, 'v')}"
                                )

                        # METHOD 2: Try .get() method (official ANDES alternative)
                        if pe_from_pf_before_tds is None and hasattr(ss.GENCLS, "get"):
                            try:
                                pe_get = ss.GENCLS.get("Pe")
                                if pe_get is not None:
                                    if isinstance(pe_get, np.ndarray):
                                        if pe_get.ndim == 1 and len(pe_get) > gen_idx_for_pe_check:
                                            pe_from_pf_before_tds = float(
                                                pe_get[gen_idx_for_pe_check]
                                            )
                                        elif pe_get.ndim == 1 and len(pe_get) > 0:
                                            pe_from_pf_before_tds = float(pe_get[0])
                                        elif pe_get.ndim == 0:
                                            pe_from_pf_before_tds = float(pe_get)
                                    elif not isinstance(pe_get, (list, tuple)):
                                        pe_from_pf_before_tds = float(pe_get)
                            except Exception:
                                pass

                        # METHOD 3: Try accessing via algebraic variable index from power flow solution
                        # After power flow, solution is in ss.dae.y (algebraic variables)
                        if pe_from_pf_before_tds is None:
                            if (
                                hasattr(ss, "dae")
                                and hasattr(ss.dae, "y")
                                and ss.dae.y is not None
                                and hasattr(ss.GENCLS, "Pe")
                                and hasattr(ss.GENCLS.Pe, "a")
                            ):
                                try:
                                    pe_a = ss.GENCLS.Pe.a
                                    if hasattr(pe_a, "__getitem__") and hasattr(pe_a, "__len__"):
                                        if len(pe_a) > gen_idx_for_pe_check:
                                            pe_a_idx = int(pe_a[gen_idx_for_pe_check])
                                            if pe_a_idx < len(ss.dae.y):
                                                pe_from_pf_before_tds = float(ss.dae.y[pe_a_idx])
                                        elif len(pe_a) > 0:
                                            pe_a_idx = int(pe_a[0])
                                            if pe_a_idx < len(ss.dae.y):
                                                pe_from_pf_before_tds = float(ss.dae.y[pe_a_idx])
                                except (
                                    IndexError,
                                    TypeError,
                                    ValueError,
                                    AttributeError,
                                ):
                                    pass
                        # METHOD 4: Fallback - Use Pm as Pe (they should be equal at steady-state)
                        # This is a valid assumption: at steady-state, Pe = Pm (power balance)
                        if pe_from_pf_before_tds is None and Pm_to_use is not None:
                            # At steady-state (power flow solution), Pe should equal Pm
                            pe_from_pf_before_tds = float(Pm_to_use)
                            if verbose and DEBUG_TDS:
                                print(
                                    f"[DEBUG] Pe extraction from power flow failed. "
                                    f"Using Pm={Pm_to_use:.6f} pu as Pe (valid at steady-state:"
                                    f"Pe=Pm)."
                                )
                    except Exception as pe_extract_err:
                        if verbose and DEBUG_TDS:
                            print(f"[DEBUG] Could not extract Pe from power flow: {pe_extract_err}")
                        # Final fallback: use Pm as Pe
                        if pe_from_pf_before_tds is None and Pm_to_use is not None:
                            pe_from_pf_before_tds = float(Pm_to_use)
                            if verbose and DEBUG_TDS:
                                print(
                                    f"[DEBUG] Using Pm={Pm_to_use:.6f} pu as Pe fallback "
                                    f"(valid at steady-state: Pe=Pm)."
                                )

                    if pe_from_pf_before_tds is not None and Pm_to_use is not None:
                        pe_pf_mismatch_before_tds = (
                            100.0
                            * abs(pe_from_pf_before_tds - Pm_to_use)
                            / (abs(Pm_to_use) + 1e-12)
                        )
                        if pe_pf_mismatch_before_tds > 5.0:  # More than 5% difference
                            # CRITICAL ERROR: Power flow Pe doesn't match Pm!
                            mode_label = (
                                "[Pm VARIATION]" if not use_load_mode else "[LOAD VARIATION]"
                            )
                            error_msg = (
                                f"[ERROR] {mode_label} Power flow Pe doesn't match Pm before TDS.init()!\n"
                                f"  Pe from power flow: {pe_from_pf_before_tds:.6f} pu\n"
                                f"  Requested Pm: {Pm_to_use:.6f} pu\n"
                                f"  Mismatch: {pe_pf_mismatch_before_tds:.1f}%\n"
                                f"  This indicates power flow didn't converge to correct equilibrium."
                            )
                            if verbose:
                                print(error_msg)
                            # Skip this configuration
                            continue

                    # CRITICAL: Initialize TDS (power flow is already converged)
                    # Note: Time step must be set BEFORE init() to ensure it's applied
                    # CRITICAL: Reset TDS.initialized RIGHT BEFORE init() to ensure it reads NEW power flow solution
                    # (We reset it before power flow, but ANDES may have re-initialized it during TDS config)
                    if hasattr(ss, "TDS") and hasattr(ss.TDS, "initialized"):
                        ss.TDS.initialized = False  # Force TDS to re-read power flow solution

                    # CRITICAL DEBUG: Capture state RIGHT BEFORE TDS.init() to diagnose P_e extraction issues
                    if verbose and Pm_to_use is not None and DEBUG_TDS:
                        try:
                            # Check Pm value
                            tm0_before_tds_init = None
                            if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                                tm0v = ss.GENCLS.tm0.v
                                if hasattr(tm0v, "__getitem__") and len(tm0v) > gen_idx_after_setup:
                                    tm0_before_tds_init = float(tm0v[gen_idx_after_setup])

                            # Check Pe from power flow (use the value we already extracted)
                            pe_from_pf_debug = pe_from_pf_before_tds
                            # If still None, try one more time with all methods
                            if pe_from_pf_debug is None:
                                # Try direct Pe.v
                                if hasattr(ss.GENCLS, "Pe") and hasattr(ss.GENCLS.Pe, "v"):
                                    try:
                                        pe_v_debug = ss.GENCLS.Pe.v
                                        if hasattr(pe_v_debug, "__getitem__") and hasattr(
                                            pe_v_debug, "__len__"
                                        ):
                                            if len(pe_v_debug) > gen_idx_after_setup:
                                                pe_from_pf_debug = float(
                                                    pe_v_debug[gen_idx_after_setup]
                                                )
                                            elif len(pe_v_debug) > 0:
                                                pe_from_pf_debug = float(pe_v_debug[0])
                                    except Exception:
                                        pass
                                # Try .get() method
                                if pe_from_pf_debug is None and hasattr(ss.GENCLS, "get"):
                                    try:
                                        pe_get_debug = ss.GENCLS.get("Pe")
                                        if pe_get_debug is not None:
                                            if isinstance(pe_get_debug, np.ndarray):
                                                if (
                                                    pe_get_debug.ndim == 1
                                                    and len(pe_get_debug) > gen_idx_after_setup
                                                ):
                                                    pe_from_pf_debug = float(
                                                        pe_get_debug[gen_idx_after_setup]
                                                    )
                                                elif (
                                                    pe_get_debug.ndim == 1 and len(pe_get_debug) > 0
                                                ):
                                                    pe_from_pf_debug = float(pe_get_debug[0])
                                    except Exception:
                                        pass
                                # Try dae.y method
                                if pe_from_pf_debug is None:
                                    if (
                                        hasattr(ss, "dae")
                                        and hasattr(ss.dae, "y")
                                        and ss.dae.y is not None
                                        and hasattr(ss.GENCLS, "Pe")
                                        and hasattr(ss.GENCLS.Pe, "a")
                                    ):
                                        try:
                                            pe_a = ss.GENCLS.Pe.a
                                            if hasattr(pe_a, "__getitem__") and hasattr(
                                                pe_a, "__len__"
                                            ):
                                                if len(pe_a) > gen_idx_after_setup:
                                                    pe_a_idx = int(pe_a[gen_idx_after_setup])
                                                    if pe_a_idx < len(ss.dae.y):
                                                        pe_from_pf_debug = float(ss.dae.y[pe_a_idx])
                                        except Exception:
                                            pass

                            mode_label_debug = (
                                "[Pm VARIATION]" if not use_load_mode else "[LOAD VARIATION]"
                            )
                            # Format values safely (handle None)
                            tm0_str = (
                                f"{tm0_before_tds_init:.6f}"
                                if tm0_before_tds_init is not None
                                else "N/A"
                            )
                            pe_pf_str = (
                                f"{pe_from_pf_debug:.6f}" if pe_from_pf_debug is not None else "N/A"
                            )
                            tds_init_str = (
                                str(getattr(ss.TDS, "initialized", "N/A"))
                                if hasattr(ss, "TDS")
                                else "N/A"
                            )
                            print(
                                f"[DEBUG] {mode_label_debug} State BEFORE TDS.init(): "
                                f"Pm={tm0_str} pu (expected: {Pm_to_use:.6f} pu), "
                                f"Pe_from_PF={pe_pf_str} pu (expected: {Pm_to_use:.6f} pu), "
                                f"TDS.initialized={tds_init_str}"
                            )
                        except Exception as debug_err:
                            if verbose and DEBUG_TDS:
                                print(
                                    f"[DEBUG] Could not capture state before TDS.init():"
                                    f"{debug_err}"
                                )

                    # CRITICAL: Verify power flow solution is in DAE state before TDS.init()
                    # TDS.init() reads from ss.dae.y (algebraic variables) which should contain Pe from power flow
                    has_dae = hasattr(ss, "dae")
                    has_dae_y = hasattr(ss, "dae") and hasattr(ss.dae, "y") if has_dae else False
                    dae_y_not_none = (
                        hasattr(ss, "dae") and hasattr(ss.dae, "y") and ss.dae.y is not None
                        if has_dae
                        else False
                    )
                    if DEBUG_TDS:
                        print(
                            f"[DEBUG] [LOAD VARIATION] Checking DAE state: hasattr(ss, 'dae')={has_dae}, "
                            f"hasattr(ss.dae, 'y')={has_dae_y}, "
                            f"ss.dae.y is not None={dae_y_not_none}"
                        )
                    if hasattr(ss, "dae") and hasattr(ss.dae, "y") and ss.dae.y is not None:
                        try:
                            # Try to extract Pe from DAE state (what TDS.init() will read)
                            has_gencls_pe = hasattr(ss.GENCLS, "Pe")
                            has_pe_a = (
                                hasattr(ss.GENCLS, "Pe") and hasattr(ss.GENCLS.Pe, "a")
                                if has_gencls_pe
                                else False
                            )
                            if DEBUG_TDS:
                                print(
                                    f"[DEBUG] [LOAD VARIATION] DAE state exists. Checking GENCLS.Pe: "
                                    f"hasattr(ss.GENCLS, 'Pe')={has_gencls_pe}, "
                                    f"hasattr(ss.GENCLS.Pe, 'a')={has_pe_a}"
                                )
                            if hasattr(ss.GENCLS, "Pe") and hasattr(ss.GENCLS.Pe, "a"):
                                pe_a = ss.GENCLS.Pe.a
                                if DEBUG_TDS:
                                    print(
                                        f"[DEBUG] [LOAD VARIATION] pe_a type={type(pe_a)}, "
                                        f"hasattr(__getitem__)={hasattr(pe_a, '__getitem__')}, "
                                        f"hasattr(__len__)={hasattr(pe_a, '__len__')}, "
                                        f"len(pe_a)={len(pe_a) if hasattr(pe_a, '__len__') else 'N/A'}, "
                                        f"gen_idx_after_setup={gen_idx_after_setup}"
                                    )
                                if (
                                    hasattr(pe_a, "__getitem__")
                                    and hasattr(pe_a, "__len__")
                                    and len(pe_a) > gen_idx_after_setup
                                ):
                                    pe_a_idx = int(pe_a[gen_idx_after_setup])
                                    if DEBUG_TDS:
                                        print(
                                            f"[DEBUG] [LOAD VARIATION] pe_a_idx={pe_a_idx}, "
                                            f"len(ss.dae.y)={len(ss.dae.y)}, "
                                            f"pe_a_idx < len(ss.dae.y)={pe_a_idx < len(ss.dae.y)}"
                                        )
                                    if pe_a_idx < len(ss.dae.y):
                                        pe_from_dae = float(ss.dae.y[pe_a_idx])
                                        if DEBUG_TDS:
                                            print(
                                                f"[DEBUG] [LOAD VARIATION] Pe in DAE state (before"
                                                f"TDS.init()):"
                                                f"Pe={pe_from_dae:.6f} pu (expected: {Pm_to_use:.6f}"
                                                f"pu)"
                                            )
                                        # If DAE state has wrong Pe, TDS.init() will use it!
                                        if abs(pe_from_dae - Pm_to_use) > 0.01:
                                            if verbose and DEBUG_TDS:
                                                print(
                                                    f"[CRITICAL] [LOAD VARIATION] DAE state has"
                                                    f"wrong Pe={pe_from_dae:.6f} pu"
                                                    f"(expected: {Pm_to_use:.6f} pu). TDS.init()"
                                                    f"will use wrong initial condition!"
                                                    f"Re-running power flow to fix DAE state..."
                                                )
                                            # Re-run power flow to fix DAE state
                                            if hasattr(ss.PFlow, "converged"):
                                                ss.PFlow.converged = False
                                            if hasattr(ss.PFlow, "initialized"):
                                                ss.PFlow.initialized = False
                                            # Skip power flow if redispatch already ran it
                                            if not (use_redispatch_mode and REDISPATCH_AVAILABLE):
                                                ss.PFlow.run()
                                            # Verify again
                                            if pe_a_idx < len(ss.dae.y):
                                                pe_from_dae_after = float(ss.dae.y[pe_a_idx])
                                                if verbose and DEBUG_TDS:
                                                    print(
                                                        f"[DEBUG] [LOAD VARIATION] Pe in DAE state"
                                                        f"(after PF rerun):"
                                                        f"Pe={pe_from_dae_after:.6f} pu (expected:"
                                                        f"{Pm_to_use:.6f} pu)"
                                                    )
                        except Exception as dae_check_err:
                            if DEBUG_TDS:
                                print(
                                    f"[ERROR] Could not check DAE state before TDS.init():"
                                    f"{dae_check_err}"
                                )
                                import traceback

                                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                    else:
                        if DEBUG_TDS:
                            print(
                                "[ERROR] [LOAD VARIATION] DAE state check failed: ss.dae.y is None or doesn't exist!"
                            )

                    # PHASE 3: TDS Configuration Investigation (diagnostics only)
                    if DEBUG_TDS:
                        print("[PHASE 3] Investigating TDS configuration options...")
                        try:
                            if hasattr(ss, "TDS"):
                                tds_attrs = [x for x in dir(ss.TDS) if not x.startswith("_")]
                                print(f"[PHASE 3] TDS attributes: {tds_attrs[:20]}...")  # First 20

                                if hasattr(ss.TDS, "config"):
                                    config_attrs = [
                                        x for x in dir(ss.TDS.config) if not x.startswith("_")
                                    ]
                                    print(
                                        f"[PHASE 3] TDS.config attributes: {config_attrs[:20]}..."
                                    )  # First 20

                                    # Look for initialization-related options
                                    init_related = [
                                        x
                                        for x in config_attrs
                                        if "init" in x.lower()
                                        or "ic" in x.lower()
                                        or "initial" in x.lower()
                                    ]
                                    if init_related:
                                        print(
                                            f"[PHASE 3] Found initialization-related config options:"
                                            f"{init_related}"
                                        )
                        except Exception as tds_inspect_err:
                            print(
                                f"[PHASE 3] [ERROR] Could not inspect TDS config: {tds_inspect_err}"
                            )

                    # Disable Toggles (e.g. line trip at t=2s in Kundur) so only the fault event occurs
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

                    # SIMPLIFIED: Call TDS.init() - it should read correct value from case file (which we modified)
                    # The case file modification ensures TDS.init() reads the correct generator setpoint
                    try:
                        if hasattr(ss.TDS, "init"):
                            if verbose and DEBUG_TDS:
                                print("[TDS] Initializing TDS from power flow solution...")
                            ss.TDS.init()

                            # CRITICAL: Verify Pm after TDS.init() - ANDES may reset tm0
                            # This matches the fix in smib_batch_tds.py (lines 420-440)
                            if (
                                Pm_to_use is not None
                                and hasattr(ss.GENCLS, "tm0")
                                and hasattr(ss.GENCLS.tm0, "v")
                            ):
                                tm0_after_init = None
                                if gen_idx_after_setup < len(ss.GENCLS.tm0.v):
                                    tm0_after_init = float(ss.GENCLS.tm0.v[gen_idx_after_setup])

                                # CRITICAL: If Pm is zero or very small, we might be using the wrong generator
                                # (e.g., infinite bus). Use find_main_generator_index to get the correct generator.
                                if (
                                    tm0_after_init is None or abs(tm0_after_init) < 1e-6
                                ) and ss.GENCLS.n > 1:
                                    if verbose:
                                        print(
                                            f"[WARNING] Generator {gen_idx_after_setup} has Pm≈0 after TDS.init(). "
                                            f"Finding correct generator using find_main_generator_index..."
                                        )
                                    # Use find_main_generator_index to get the main generator (not infinite bus)
                                    if (
                                        CCT_FINDER_AVAILABLE
                                        and find_main_generator_index is not None
                                    ):
                                        gen_idx_after_setup = find_main_generator_index(ss)
                                        # Verify this generator has non-zero Pm
                                        if gen_idx_after_setup < len(ss.GENCLS.tm0.v):
                                            Pm_candidate = float(
                                                ss.GENCLS.tm0.v[gen_idx_after_setup]
                                            )
                                            if abs(Pm_candidate) > 1e-6:
                                                tm0_after_init = Pm_candidate
                                                if verbose:
                                                    sign_note = (
                                                        " (negative - may be sign convention)"
                                                        if Pm_candidate < 0
                                                        else ""
                                                    )
                                                    print(
                                                        f"[FIXED] Using generator {gen_idx_after_setup} (from find_main_generator_index) with Pm={tm0_after_init:.6f} pu{sign_note}"
                                                    )
                                            else:
                                                # Fallback: search for generator with non-zero Pm
                                                if verbose:
                                                    print(
                                                        f"[WARNING] Generator {gen_idx_after_setup} from find_main_generator_index also has Pm≈0. Searching for generator with non-zero Pm..."
                                                    )
                                                for i in range(ss.GENCLS.n):
                                                    # Skip infinite bus (M > 1e6)
                                                    if hasattr(ss.GENCLS, "M") and hasattr(
                                                        ss.GENCLS.M, "v"
                                                    ):
                                                        if i < len(ss.GENCLS.M.v):
                                                            M_candidate = float(ss.GENCLS.M.v[i])
                                                            if M_candidate > 1e6:
                                                                continue  # Skip infinite bus
                                                    if i < len(ss.GENCLS.tm0.v):
                                                        Pm_candidate = float(ss.GENCLS.tm0.v[i])
                                                        if abs(Pm_candidate) > 1e-6:
                                                            gen_idx_after_setup = i
                                                            tm0_after_init = Pm_candidate
                                                            if verbose:
                                                                sign_note = (
                                                                    " (negative - may be sign convention)"
                                                                    if Pm_candidate < 0
                                                                    else ""
                                                                )
                                                                print(
                                                                    f"[FIXED] Using generator {gen_idx_after_setup} with Pm={tm0_after_init:.6f} pu{sign_note}"
                                                                )
                                                            break
                                    else:
                                        # Fallback: search for generator with non-zero Pm (skip infinite bus)
                                        for i in range(ss.GENCLS.n):
                                            # Skip infinite bus (M > 1e6)
                                            if hasattr(ss.GENCLS, "M") and hasattr(
                                                ss.GENCLS.M, "v"
                                            ):
                                                if i < len(ss.GENCLS.M.v):
                                                    M_candidate = float(ss.GENCLS.M.v[i])
                                                    if M_candidate > 1e6:
                                                        continue  # Skip infinite bus
                                            if i < len(ss.GENCLS.tm0.v):
                                                Pm_candidate = float(ss.GENCLS.tm0.v[i])
                                                if abs(Pm_candidate) > 1e-6:
                                                    gen_idx_after_setup = i
                                                    tm0_after_init = Pm_candidate
                                                    if verbose:
                                                        sign_note = (
                                                            " (negative - may be sign convention)"
                                                            if Pm_candidate < 0
                                                            else ""
                                                        )
                                                        print(
                                                            f"[FIXED] Using generator {gen_idx_after_setup} with Pm={tm0_after_init:.6f} pu{sign_note}"
                                                        )
                                                    break

                                # Verify Pm matches expected value (re-set if TDS.init() reset it)
                                if (
                                    tm0_after_init is not None
                                    and abs(tm0_after_init - Pm_to_use) > 1e-6
                                ):
                                    if verbose and DEBUG_TDS:
                                        print(
                                            f"[TDS CRITICAL] TDS.init() reset tm0: {tm0_after_init:.6f} != {Pm_to_use:.6f}. Re-setting..."
                                        )
                                    elif verbose and not _tm0_corrected_logged[0]:
                                        print("TDS: tm0 corrected after init (multimachine).")
                                        _tm0_corrected_logged[0] = True
                                    # Multimachine: re-apply all tm0 from saved PF values (dae.y may be overwritten by init)
                                    if (
                                        ss.GENCLS.n > 1
                                        and saved_tm0_from_pf_multimachine is not None
                                    ):
                                        tm0v = ss.GENCLS.tm0.v
                                        if hasattr(tm0v, "__setitem__"):
                                            for i in range(
                                                min(
                                                    len(saved_tm0_from_pf_multimachine),
                                                    ss.GENCLS.n,
                                                    len(tm0v),
                                                )
                                            ):
                                                tm0v[i] = saved_tm0_from_pf_multimachine[i]
                                            if verbose and DEBUG_TDS:
                                                print(
                                                    "[TDS DEBUG] Re-set all GENCLS.tm0 from saved PF values (multimachine)."
                                                )
                                    elif (
                                        ss.GENCLS.n > 1
                                        and hasattr(ss, "dae")
                                        and hasattr(ss.dae, "y")
                                        and ss.dae.y is not None
                                    ):
                                        if hasattr(ss.GENCLS, "Pe") and hasattr(ss.GENCLS.Pe, "a"):
                                            pe_a = ss.GENCLS.Pe.a
                                            tm0v = ss.GENCLS.tm0.v
                                            if hasattr(pe_a, "__len__") and hasattr(
                                                tm0v, "__setitem__"
                                            ):
                                                n_gen = ss.GENCLS.n
                                                for i in range(min(n_gen, len(pe_a), len(tm0v))):
                                                    pe_a_idx = int(pe_a[i])
                                                    if 0 <= pe_a_idx < len(ss.dae.y):
                                                        tm0v[i] = float(ss.dae.y[pe_a_idx])
                                                if verbose and DEBUG_TDS:
                                                    print(
                                                        "[TDS DEBUG] Re-set all GENCLS.tm0 from dae.y (multimachine)."
                                                    )
                                                # CRITICAL: dae.y after TDS.init() can be wrong (e.g. wrong gen order).
                                                # Always force main generator to Pm_to_use so scenario is consistent.
                                                if gen_idx_after_setup < len(ss.GENCLS.tm0.v):
                                                    ss.GENCLS.tm0.v[gen_idx_after_setup] = Pm_to_use
                                                    if verbose and DEBUG_TDS:
                                                        print(
                                                            "[TDS DEBUG] Forced main generator tm0 to Pm_to_use (dae.y may be wrong)."
                                                        )
                                    else:
                                        # SMIB or no dae.y: re-set only the main generator
                                        if gen_idx_after_setup < len(ss.GENCLS.tm0.v):
                                            ss.GENCLS.tm0.v[gen_idx_after_setup] = Pm_to_use
                                    if hasattr(ss.GENCLS, "M") and hasattr(ss.GENCLS.M, "v"):
                                        for gidx in range(ss.GENCLS.n):
                                            if gidx < len(ss.GENCLS.M.v):
                                                ss.GENCLS.M.v[gidx] = M
                                    if hasattr(ss.GENCLS, "D") and hasattr(ss.GENCLS.D, "v"):
                                        for gidx in range(ss.GENCLS.n):
                                            if gidx < len(ss.GENCLS.D.v):
                                                ss.GENCLS.D.v[gidx] = D
                                    # PV.p0 only for SMIB
                                    if ss.GENCLS.n == 1 and hasattr(ss, "PV") and ss.PV.n > 0:
                                        if hasattr(ss.PV, "p0") and hasattr(ss.PV.p0, "v"):
                                            if len(ss.PV.p0.v) > 0:
                                                ss.PV.p0.v[0] = Pm_to_use
                                    # Re-initialize TDS after fixing parameters
                                    ss.TDS.initialized = False
                                    ss.TDS.init()
                                    if verbose and DEBUG_TDS:
                                        print(
                                            "[TDS DEBUG] Parameters fixed after TDS.init() - re-initialized"
                                        )

                            # Set time step configuration after TDS.init()
                            if time_step is not None:
                                # Re-set the proper ANDES parameters after init() (ANDES may recalculate them)
                                # CRITICAL: Use fixt, tstep, shrinkt (ANDES manual parameters)
                                if hasattr(ss.TDS.config, "fixt"):
                                    ss.TDS.config.fixt = 1  # Fixed step mode
                                if hasattr(ss.TDS.config, "tstep"):
                                    ss.TDS.config.tstep = (
                                        time_step  # CRITICAL: This is the key parameter
                                    )
                                if hasattr(ss.TDS.config, "shrinkt"):
                                    ss.TDS.config.shrinkt = 0  # Don't allow shrinking

                                # Also set h/dt as fallback
                                if hasattr(ss.TDS.config, "h"):
                                    ss.TDS.config.h = time_step
                                if hasattr(ss.TDS.config, "dt"):
                                    ss.TDS.config.dt = time_step

                                # Disable adaptive time stepping if available
                                if hasattr(ss.TDS.config, "fixed_step"):
                                    ss.TDS.config.fixed_step = True
                                if hasattr(ss.TDS.config, "adaptive"):
                                    ss.TDS.config.adaptive = False

                                # Try to set in solver config if available
                                if hasattr(ss.TDS, "solver") and hasattr(ss.TDS.solver, "h"):
                                    ss.TDS.solver.h = time_step
                                # Also try setting in the internal time step storage
                                if hasattr(ss.TDS, "h"):
                                    ss.TDS.h = time_step

                                # CRITICAL: Re-set save_interval AFTER init() as well (ANDES may reset it)
                                if hasattr(ss.TDS.config, "save_interval"):
                                    ss.TDS.config.save_interval = 1
                                if hasattr(ss.TDS.config, "save_step"):
                                    ss.TDS.config.save_step = 1
                                if hasattr(ss.TDS.config, "output_step"):
                                    ss.TDS.config.output_step = 1

                                # VERIFY: Check actual time step after forcing (ANDES may have overridden it)
                                # Check tstep first (ANDES manual parameter), then h as fallback
                                actual_h = None
                                if hasattr(ss.TDS.config, "tstep"):
                                    actual_h = ss.TDS.config.tstep
                                elif hasattr(ss.TDS.config, "h"):
                                    actual_h = ss.TDS.config.h
                                elif hasattr(ss.TDS, "solver") and hasattr(ss.TDS.solver, "h"):
                                    actual_h = ss.TDS.solver.h
                                elif hasattr(ss.TDS, "h"):
                                    actual_h = ss.TDS.h

                                # If time step was overridden, try to force it one more time using proper ANDES parameters
                                if actual_h is not None and abs(actual_h - time_step) > 1e-6:
                                    # Force again more aggressively using ANDES manual parameters
                                    if hasattr(ss.TDS.config, "fixt"):
                                        ss.TDS.config.fixt = 1
                                    if hasattr(ss.TDS.config, "tstep"):
                                        ss.TDS.config.tstep = time_step  # CRITICAL: Use tstep
                                    if hasattr(ss.TDS.config, "shrinkt"):
                                        ss.TDS.config.shrinkt = 0
                                    if hasattr(ss.TDS.config, "h"):
                                        ss.TDS.config.h = time_step
                                    if hasattr(ss.TDS.config, "dt"):
                                        ss.TDS.config.dt = time_step
                                    if hasattr(ss.TDS, "solver") and hasattr(ss.TDS.solver, "h"):
                                        ss.TDS.solver.h = time_step
                                    if hasattr(ss.TDS, "h"):
                                        ss.TDS.h = time_step
                                    # Re-check (check tstep first, then h)
                                    if hasattr(ss.TDS.config, "tstep"):
                                        actual_h = ss.TDS.config.tstep
                                    elif hasattr(ss.TDS.config, "h"):
                                        actual_h = ss.TDS.config.h

                                # Warn if time step was overridden (critical for CCT accuracy)
                                if actual_h is not None and abs(actual_h - time_step) > 1e-6:
                                    # Only warn once per run to avoid spam
                                    if not hasattr(generate_parameter_sweep, "_time_step_warned"):
                                        generate_parameter_sweep._time_step_warned = True
                                        warning_msg = (
                                            f"\n{'=' * 70}\n"
                                            f"[WARNING] ANDES overrode requested time step!\n"
                                            f"{'=' * 70}\n"
                                            f"  Requested: {time_step * 1000:.3f} ms (0.001 s)\n"
                                            f"  Actual:    {actual_h * 1000:.3f} ms ({actual_h:.6f} s)\n"
                                            f"  Difference: {abs(actual_h - time_step) * 1000:.3f} ms\n"
                                            f"\n"
                                            f"  Impact:\n"
                                            f"- CCT accuracy limited to ~{actual_h * 1000:.1f} ms (vs "
                                            f"target 1-2 ms)\n"
                                            f"- Time points per trajectory: "
                                            f"~{int(simulation_time / actual_h)} (vs "
                                            f"{int(simulation_time / time_step)} expected)\n"
                                            f"  - This may affect training data quality\n"
                                            f"\n"
                                            f"  Possible causes:\n"
                                            f"  - ANDES uses frequency-based time step calculation\n"
                                            f"- System frequency: {1.0 / actual_h:.1f} Hz (if"
                                            f"{actual_h * 1000:.1f} ms = 1/(2*freq))\n"
                                            f"- ANDES may require specific configuration to force "
                                            f"custom time step\n"
                                            f"\n"
                                            f"  Recommendation:\n"
                                            f"- Check ANDES documentation for forcing fixed time "
                                            f"step\n"
                                            f"- Consider if {actual_h * 1000:.1f} ms resolution is "
                                            f"acceptable for your use case\n"
                                            f"- For CCT accuracy < 2 ms, time step should be ≤ 1 "
                                            f"ms\n"
                                            f"{'=' * 70}\n"
                                        )
                                        # Print to stderr so it's visible even with stdout suppression
                                        # Use module-level sys import
                                        import sys as _sys_module

                                        print(warning_msg, file=_sys_module.stderr)
                                        if verbose:
                                            print(warning_msg)
                    except Exception as e:
                        error_summary["tds_failed"] += 1
                        error_msg = f"TDS initialization failed: {str(e)}"
                        # Capture more detailed error information
                        if hasattr(e, "__traceback__"):
                            import traceback

                            error_details = "".join(
                                traceback.format_exception(type(e), e, e.__traceback__)
                            )
                        else:
                            error_details = error_msg

                        # Store first TDS error for debugging (even when verbose=False)
                        if error_summary["first_tds_error"] is None:
                            error_summary["first_tds_error"] = error_details
                            error_summary["first_tds_error_params"] = {
                                "M": M,
                                "D": D,
                                "tc": tc,
                                "Pm": Pm_to_use,
                                "fault_bus": fault_bus_actual,
                                "fault_start_time": fault_start_time_actual,
                                "simulation_time": simulation_time,
                                "time_step": time_step,
                                "stage": "TDS_init",
                            }

                        if verbose:
                            print(
                                f"Warning: TDS initialization failed for M={M:.2f}, D={D:.2f},"
                                f"tc={tc:.3f}, bus={fault_bus_actual}: {error_msg}"
                            )
                            print(f"  Error details: {error_details[:500]}")  # First 500 chars
                        continue

                    # Run TDS simulation
                    try:
                        # WORKFLOW VERIFICATION: Verify order before TDS
                        expected_before_tds = ["load", "setup", "M_D_set", "power_flow"]
                        if use_load_mode and alpha is not None:
                            expected_before_tds.extend(["PV_p0_set", "GENCLS_tm0_set"])
                        missing_steps = [
                            step for step in expected_before_tds if step not in workflow_steps
                        ]
                        # Workflow validation removed for cleaner output
                        # if missing_steps and verbose:
                        #     print(
                        #         f"[WORKFLOW WARNING] Missing steps before TDS: {missing_steps}. "
                        #         f"Current order: {' → '.join(workflow_steps)}"
                        #     )
                        if verbose and len(workflow_steps) > 0:
                            # Log successful workflow order verification (only once per combination to avoid spam)
                            if current_combination == 1 or (current_combination % 10 == 0):
                                workflow_str = " → ".join(workflow_steps)
                                print(f"[WORKFLOW VERIFIED] Correct order: {workflow_str} → TDS")

                        # Disable Toggles (e.g. line trip at t=2s in Kundur) so only the fault event occurs
                        if hasattr(ss, "Toggle") and ss.Toggle.n > 0:
                            try:
                                for i in range(ss.Toggle.n):
                                    try:
                                        if hasattr(ss.Toggle, "u") and hasattr(ss.Toggle.u, "v"):
                                            ss.Toggle.u.v[i] = 0
                                        else:
                                            ss.Toggle.alter("u", i + 1, 0)
                                    except Exception:
                                        ss.Toggle.alter("u", i + 1, 0)
                            except Exception:
                                pass

                        ss.TDS.run()
                        workflow_steps.append("TDS")  # Track workflow order
                        # TDS simulation completed
                        if verbose:
                            try:
                                # Try to extract Pe(t=0) from TDS results immediately after run
                                if (
                                    hasattr(ss, "dae")
                                    and hasattr(ss.dae, "ts")
                                    and ss.dae.ts is not None
                                ):
                                    if hasattr(ss.dae.ts, "t") and hasattr(ss.dae.ts, "y"):
                                        time_ts = ss.dae.ts.t
                                        y_ts = ss.dae.ts.y
                                        if len(time_ts) > 0 and y_ts is not None and y_ts.ndim == 2:
                                            # Find earliest time point (should be t approx 0, pre-fault)
                                            t0_idx = np.argmin(time_ts)
                                            t0_val = float(time_ts[t0_idx])

                                            # Try to extract Pe from ts.y
                                            pe_at_t0_from_ts = None
                                            if (
                                                hasattr(ss, "GENCLS")
                                                and hasattr(ss.GENCLS, "Pe")
                                                and hasattr(ss.GENCLS.Pe, "a")
                                            ):
                                                try:
                                                    pea = ss.GENCLS.Pe.a
                                                    if (
                                                        hasattr(pea, "__len__")
                                                        and len(pea) > gen_idx_after_setup
                                                    ):
                                                        pe_idx_debug = int(pea[gen_idx_after_setup])
                                                        if pe_idx_debug < y_ts.shape[1]:
                                                            pe_at_t0_from_ts = float(
                                                                y_ts[t0_idx, pe_idx_debug]
                                                            )
                                                except Exception:
                                                    pass

                                            mode_label_after = (
                                                "[Pm VARIATION]"
                                                if not use_load_mode
                                                else "[LOAD VARIATION]"
                                            )
                                            # Format safely (handle None)
                                            pe_t0_str = (
                                                f"{pe_at_t0_from_ts:.6f}"
                                                if pe_at_t0_from_ts is not None
                                                else "N/A"
                                            )
                                            print(
                                                f"[DEBUG] {mode_label_after} State AFTER TDS.run(): "
                                                f"t0={t0_val:.6f}s, Pe(t=0)={pe_t0_str} pu "
                                                f"(expected: {Pm_to_use:.6f} pu)"
                                            )
                            except Exception as debug_err:
                                if verbose:
                                    print(
                                        f"[DEBUG] Could not check Pe(t=0) after TDS.run():"
                                        f"{debug_err}"
                                    )

                        # CRITICAL: Re-set generator parameters after TDS (it may reset them)
                        # Re-identify generator index after TDS (it may have changed)
                        gen_idx_after_tds = None
                        if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
                            # Find the main generator (not infinite bus) - same logic as before
                            infinite_bus_idx_after_tds = None
                            for i in range(ss.GENCLS.n):
                                is_infinite = False
                                if hasattr(ss.GENCLS, "M") and hasattr(ss.GENCLS.M, "v"):
                                    try:
                                        M_val = (
                                            ss.GENCLS.M.v[i]
                                            if hasattr(ss.GENCLS.M.v, "__getitem__")
                                            and len(ss.GENCLS.M.v) > i
                                            else ss.GENCLS.M.v
                                        )
                                        if (
                                            M_val is not None
                                            and isinstance(M_val, (int, float))
                                            and M_val > 1e6
                                        ):
                                            is_infinite = True
                                    except (IndexError, AttributeError, TypeError):
                                        pass
                                if is_infinite:
                                    infinite_bus_idx_after_tds = i
                                    break

                            # Determine which generator to modify
                            if infinite_bus_idx_after_tds == 0 and ss.GENCLS.n > 1:
                                gen_idx_after_tds = 1
                            else:
                                gen_idx_after_tds = 0

                        # Fallback to original index if re-identification failed
                        if gen_idx_after_tds is None:
                            try:
                                gen_idx_after_tds = gen_idx_after_setup
                            except NameError:
                                gen_idx_after_tds = gen_idx_to_modify

                        # Re-set parameters using direct access (more reliable than alter() after TDS)
                        # CRITICAL: Use direct access, not alter(), because alter() may fail after TDS
                        if Pm_to_use is not None and gen_idx_after_tds is not None:
                            try:
                                # Validate generator index is valid
                                if gen_idx_after_tds >= ss.GENCLS.n:
                                    if verbose:
                                        print(
                                            f"[WARNING] gen_idx_after_tds={gen_idx_after_tds} >= "
                                            f"ss.GENCLS.n={ss.GENCLS.n}, falling back to 0"
                                        )
                                    gen_idx_after_tds = 0

                                # Use direct access (more reliable after TDS)
                                # Check if tm0.v is array or scalar
                                if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                                    tm0v = ss.GENCLS.tm0.v
                                    if hasattr(tm0v, "__getitem__") and hasattr(tm0v, "__len__"):
                                        # Array case
                                        if gen_idx_after_tds < len(tm0v):
                                            ss.GENCLS.tm0.v[gen_idx_after_tds] = Pm_to_use
                                    else:
                                        # Single generator case (scalar)
                                        ss.GENCLS.tm0.v = Pm_to_use

                                # Also set P0 if available
                                if hasattr(ss.GENCLS, "P0") and hasattr(ss.GENCLS.P0, "v"):
                                    p0v = ss.GENCLS.P0.v
                                    if hasattr(p0v, "__getitem__") and hasattr(p0v, "__len__"):
                                        # Array case
                                        if gen_idx_after_tds < len(p0v):
                                            ss.GENCLS.P0.v[gen_idx_after_tds] = Pm_to_use
                                    else:
                                        # Single generator case (scalar)
                                        ss.GENCLS.P0.v = Pm_to_use

                                # Verify after re-setting
                                if hasattr(ss.GENCLS, "tm0") and hasattr(ss.GENCLS.tm0, "v"):
                                    tm0_after_tds = (
                                        ss.GENCLS.tm0.v[gen_idx_after_tds]
                                        if hasattr(ss.GENCLS.tm0.v, "__getitem__")
                                        and len(ss.GENCLS.tm0.v) > gen_idx_after_tds
                                        else ss.GENCLS.tm0.v
                                    )
                                    mismatch_pct = (
                                        100.0
                                        * abs(tm0_after_tds - Pm_to_use)
                                        / (abs(Pm_to_use) + 1e-12)
                                    )
                                    if mismatch_pct > 0.01:
                                        mode_label = (
                                            "[Pm VARIATION]"
                                            if not use_load_mode
                                            else "[LOAD VARIATION]"
                                        )
                                        if verbose:
                                            print(
                                                f"[FIX] {mode_label} Re-set Pm after TDS:"
                                                f"{Pm_to_use:.6f} pu"
                                                f"(was {tm0_after_tds:.6f} pu, {mismatch_pct:.3f}%"
                                                f"difference, gen_idx={gen_idx_after_tds})"
                                            )
                                    # Only show debug message in Pm variation mode (less relevant for load variation)
                                    elif verbose and not use_load_mode:
                                        print(
                                            f"[DEBUG] [Pm VARIATION] Pm after TDS:"
                                            f"{tm0_after_tds:.6f} pu"
                                            f"(matches requested {Pm_to_use:.6f} pu,"
                                            f"gen_idx={gen_idx_after_tds})"
                                        )
                            except Exception as tds_param_err:
                                mode_label = (
                                    "[Pm VARIATION]" if not use_load_mode else "[LOAD VARIATION]"
                                )
                                if verbose:
                                    print(
                                        f"[WARNING] {mode_label} Could not re-set Pm after TDS"
                                        f"(gen_idx={gen_idx_after_tds}): {tds_param_err}"
                                    )
                    except Exception as e:
                        error_summary["tds_failed"] += 1
                        error_msg = str(e)
                        # Capture more detailed error information
                        if hasattr(e, "__traceback__"):
                            import traceback

                            error_details = "".join(
                                traceback.format_exception(type(e), e, e.__traceback__)
                            )
                        else:
                            error_details = error_msg

                        # Store first TDS error for debugging (even when verbose=False)
                        if error_summary["first_tds_error"] is None:
                            error_summary["first_tds_error"] = error_details
                            error_summary["first_tds_error_params"] = {
                                "M": M,
                                "D": D,
                                "tc": tc,
                                "Pm": Pm_to_use,
                                "fault_bus": fault_bus_actual,
                                "fault_start_time": fault_start_time_actual,
                                "simulation_time": simulation_time,
                                "time_step": time_step,
                            }

                        if verbose:
                            print(
                                f"Warning: TDS simulation failed for M={M:.2f}, D={D:.2f},"
                                f"tc={tc:.3f}, bus={fault_bus_actual}: {error_msg}"
                            )
                            print(f"  Error details: {error_details[:500]}")  # First 500 chars
                        continue

                    # Check if simulation was successful
                    # Check multiple indicators of TDS success
                    tds_success = True
                    error_reason = None

                    # Check exit code if available
                    if hasattr(ss, "exit_code") and ss.exit_code != 0:
                        tds_success = False
                        error_reason = f"exit_code={ss.exit_code}"

                    # Check if TDS has time series data (indicates successful run)
                    if tds_success:
                        has_data = False
                        try:
                            if (
                                hasattr(ss, "dae")
                                and hasattr(ss.dae, "ts")
                                and ss.dae.ts is not None
                            ):
                                if hasattr(ss.dae.ts, "t") and ss.dae.ts.t is not None:
                                    if len(ss.dae.ts.t) > 0:
                                        has_data = True
                        except Exception:
                            pass

                        if not has_data:
                            # Try alternative check via TDS plotter
                            try:
                                if (
                                    hasattr(ss, "TDS")
                                    and hasattr(ss.TDS, "plt")
                                    and ss.TDS.plt is not None
                                ):
                                    if hasattr(ss.TDS.plt, "t") and ss.TDS.plt.t is not None:
                                        if len(ss.TDS.plt.t) > 0:
                                            has_data = True
                            except Exception:
                                pass

                        if not has_data:
                            tds_success = False
                            error_reason = "no time series data after TDS.run()"

                    if not tds_success:
                        error_summary["tds_failed"] += 1
                        # Store first TDS failure reason for final error message (even when verbose=False)
                        if error_summary["first_tds_error"] is None and error_reason:
                            error_summary["first_tds_error"] = error_reason
                            error_summary["first_tds_error_params"] = {
                                "M": M,
                                "D": D,
                                "tc": tc,
                                "Pm": Pm_to_use,
                                "fault_bus": fault_bus_actual,
                                "fault_start_time": fault_start_time_actual,
                                "simulation_time": simulation_time,
                                "time_step": time_step,
                            }
                        if verbose:
                            print(
                                f"Warning: TDS simulation failed for M={M:.2f}, D={D:.2f},"
                                f"tc={tc:.3f}, bus={fault_bus_actual}: {error_reason}"
                            )
                        continue

                    # Extract dataset (this requires TDS to have run successfully)
                    try:
                        if use_pe_as_input:
                            # Extract Pe(t) trajectories and other data for Pe-input approach
                            # CRITICAL: Use gen_idx_after_setup (not gen_idx_to_modify) because
                            # gen_idx_after_setup may have been updated to the correct generator
                            # after power flow (e.g., when finding non-infinite bus generator)
                            gen_idx_for_extraction = gen_idx_after_setup
                            n_gen = getattr(ss.GENCLS, "n", 1)
                            # Multimachine: extract Pe for all generators so we can write Pe_0..Pe_{n-1}
                            pe_data = extract_pe_trajectories(
                                ss,
                                gen_idx=None if n_gen > 1 else gen_idx_for_extraction,
                                Pm_actual=Pm_to_use,
                            )
                            trajectories = extract_trajectories(
                                ss, gen_idx=gen_idx_for_extraction, Pm_actual=Pm_to_use
                            )

                            # Build DataFrame with Pe(t) as input (no reactances)
                            # Ensure time vectors match - use the one from trajectories as primary source
                            time_pe = pe_data.get("time", None)
                            time_traj = trajectories.get("time", None)

                            if time_traj is not None:
                                time = time_traj
                            elif time_pe is not None:
                                time = time_pe
                            else:
                                time = np.array([])

                            # Validate time vector length matches trajectory data
                            if len(time) > 0:
                                delta_sample = trajectories.get("delta", None)
                                if delta_sample is not None and hasattr(delta_sample, "__len__"):
                                    if len(delta_sample) != len(time):
                                        warnings.warn(
                                            f"Time vector length mismatch: pe_data time"
                                            f"length={len(time_pe) if time_pe is not None else 0},"
                                            f"trajectories time length={len(time_traj) if time_traj is not None else 0},"
                                            f"delta length={len(delta_sample)}. Using trajectories time vector.",
                                            UserWarning,
                                            stacklevel=2,
                                        )
                                        # Correct time to match delta length - use time_traj if available
                                        if time_traj is not None and len(time_traj) == len(
                                            delta_sample
                                        ):
                                            time = time_traj
                                        elif time_pe is not None and len(time_pe) == len(
                                            delta_sample
                                        ):
                                            time = time_pe
                                        # If neither matches, keep current time (will cause issues downstream)

                            # Initialize Pe_raw using the corrected time vector
                            Pe_raw = pe_data.get("Pe", trajectories.get("Pe", np.zeros_like(time)))

                            # Handle case where Pe might be a dict (multi-machine) or array (single machine)
                            if isinstance(Pe_raw, dict):
                                # Multi-machine case: use the generator index or first machine
                                # Handle type mismatch: gen_idx_to_modify might be int, but dict keys might be int or str
                                pe_key = None
                                if gen_idx_to_modify is not None:
                                    # Try exact match first
                                    if gen_idx_to_modify in Pe_raw:
                                        pe_key = gen_idx_to_modify
                                    else:
                                        # Try all possible type conversions systematically
                                        # Try converting gen_idx_to_modify to int
                                        try:
                                            gen_idx_int = int(float(gen_idx_to_modify))
                                            if gen_idx_int in Pe_raw:
                                                pe_key = gen_idx_int
                                        except (ValueError, TypeError):
                                            pass

                                        # Try converting gen_idx_to_modify to string
                                        if pe_key is None:
                                            try:
                                                gen_idx_str = str(int(float(gen_idx_to_modify)))
                                                if gen_idx_str in Pe_raw:
                                                    pe_key = gen_idx_str
                                            except (ValueError, TypeError):
                                                pass

                                        # Try matching against dict keys with type conversions
                                        if pe_key is None:
                                            for key in Pe_raw.keys():
                                                try:
                                                    # Try converting dict key to match gen_idx_to_modify type
                                                    if isinstance(gen_idx_to_modify, (int, float)):
                                                        key_as_int = int(float(key))
                                                        if (
                                                            abs(
                                                                key_as_int
                                                                - float(gen_idx_to_modify)
                                                            )
                                                            < 1e-6
                                                        ):
                                                            pe_key = key
                                                            break
                                                    elif isinstance(gen_idx_to_modify, str):
                                                        if str(key) == gen_idx_to_modify or int(
                                                            float(key)
                                                        ) == int(float(gen_idx_to_modify)):
                                                            pe_key = key
                                                            break
                                                except (ValueError, TypeError):
                                                    continue

                                if pe_key is not None:
                                    Pe = Pe_raw[pe_key]
                                elif len(Pe_raw) > 0:
                                    # Use first available machine
                                    # Issue warning if gen_idx_to_modify is None (should specify which machine) or if it couldn't be matched
                                    if gen_idx_to_modify is None:
                                        warnings.warn(
                                            f"gen_idx_to_modify is None but Pe is a dictionary"
                                            f"(multi-machine case)."
                                            f"Falling back to first machine's Pe trajectory (key:"
                                            f"{list(Pe_raw.keys())[0]})."
                                            f"This may cause data corruption if Pe trajectory is"
                                            f"paired with wrong machine's parameters."
                                            f"Available keys: {list(Pe_raw.keys())}",
                                            UserWarning,
                                            stacklevel=2,
                                        )
                                    else:
                                        warnings.warn(
                                            f"Could not match gen_idx_to_modify={gen_idx_to_modify} "
                                            f"with any key in Pe dictionary (keys:"
                                            f"{list(Pe_raw.keys())})."
                                            f"Falling back to first machine's Pe trajectory (key:"
                                            f"{list(Pe_raw.keys())[0]})."
                                            f"This may cause data corruption if Pe trajectory is paired with wrong machine's parameters.",
                                            UserWarning,
                                            stacklevel=2,
                                        )
                                    Pe = Pe_raw[list(Pe_raw.keys())[0]]
                                else:
                                    Pe = np.zeros_like(time)
                            else:
                                # Single machine case: Pe is already an array
                                Pe = Pe_raw

                            # Validate and resample Pe to match corrected time vector length
                            if len(time) > 0:
                                # Convert Pe to numpy array if needed
                                Pe = np.asarray(Pe)
                                # Get length safely: handle both arrays and scalars (0-d arrays)
                                try:
                                    Pe_original_len = len(Pe) if hasattr(Pe, "__len__") else 0
                                except TypeError:
                                    # Pe is a scalar (0-d array), len() raises TypeError
                                    Pe_original_len = 0

                                if Pe_original_len == 0:
                                    # If Pe is empty or scalar, create array matching time length
                                    # Check if Pe is a scalar (0-d array) - only scalars are safe to call .item()
                                    if Pe.ndim == 0:
                                        # Pe is a scalar (0-d array), safe to call .item()
                                        try:
                                            Pe = np.full_like(time, Pe.item())
                                        except (ValueError, AttributeError):
                                            # Fallback if .item() fails for any reason
                                            Pe = np.full_like(time, 0.0)
                                    else:
                                        # Pe is an empty array (ndim > 0 but len == 0)
                                        # Cannot call .item() on empty array - raises ValueError
                                        # Use default value 0.0 directly
                                        Pe = np.full_like(time, 0.0)
                                elif Pe_original_len != len(time):
                                    # Get original time vector for Pe from pe_data
                                    time_pe_original = pe_data.get("time", None)
                                    if (
                                        time_pe_original is not None
                                        and len(time_pe_original) == Pe_original_len
                                    ):
                                        # Resample Pe to match corrected time vector
                                        Pe = np.interp(time, time_pe_original, Pe)
                                        warnings.warn(
                                            f"Resampled Pe from length {Pe_original_len} to"
                                            f"{len(time)}"
                                            f"to match corrected time vector length.",
                                            UserWarning,
                                            stacklevel=2,
                                        )
                                    else:
                                        # If we can't resample, pad or truncate to match
                                        if Pe_original_len > len(time):
                                            Pe = Pe[: len(time)]
                                            warnings.warn(
                                                f"Truncated Pe from length {Pe_original_len} to"
                                                f"{len(time)}"
                                                f"to match corrected time vector length.",
                                                UserWarning,
                                                stacklevel=2,
                                            )
                                        else:
                                            # Pad with last value
                                            Pe_padded = np.zeros_like(time)
                                            Pe_padded[:Pe_original_len] = Pe
                                            Pe_padded[Pe_original_len:] = (
                                                Pe[-1] if Pe_original_len > 0 else 0.0
                                            )
                                            Pe = Pe_padded
                                            warnings.warn(
                                                f"Padded Pe from length {Pe_original_len} to"
                                                f"{len(time)}"
                                                f"to match corrected time vector length.",
                                                UserWarning,
                                                stacklevel=2,
                                            )
                                # Final validation: ensure Pe matches time length
                                if len(Pe) != len(time):
                                    # Last resort: create array matching time length
                                    Pe = np.full_like(time, Pe[0] if len(Pe) > 0 else 0.0)
                                    warnings.warn(
                                        f"Force-matched Pe length to {len(time)} (original: {Pe_original_len}).",
                                        UserWarning,
                                        stacklevel=2,
                                    )
                            else:
                                # If time is empty, create empty Pe
                                Pe = np.array([])

                            # Get generator parameters
                            # Convert gen_idx_to_modify to int if it's a string (for array indexing)
                            gen_idx_int = None
                            if gen_idx_to_modify is not None:
                                if isinstance(gen_idx_to_modify, (int, np.integer)):
                                    gen_idx_int = int(gen_idx_to_modify)
                                elif isinstance(gen_idx_to_modify, str):
                                    try:
                                        gen_idx_int = int(gen_idx_to_modify)
                                    except (ValueError, TypeError):
                                        gen_idx_int = None

                            if hasattr(ss.GENCLS, "M") and hasattr(ss.GENCLS.M, "v"):
                                if (
                                    gen_idx_int is not None
                                    and isinstance(ss.GENCLS.M.v, (list, np.ndarray))
                                    and 0 <= gen_idx_int < len(ss.GENCLS.M.v)
                                ):
                                    M_val = ss.GENCLS.M.v[gen_idx_int]
                                elif (
                                    isinstance(ss.GENCLS.M.v, (list, np.ndarray))
                                    and len(ss.GENCLS.M.v) > 0
                                ):
                                    # Fallback to first generator if gen_idx_to_modify is None or invalid
                                    M_val = ss.GENCLS.M.v[0]
                                else:
                                    M_val = ss.GENCLS.M.v
                            else:
                                M_val = 5.7512  # Default

                            if hasattr(ss.GENCLS, "D") and hasattr(ss.GENCLS.D, "v"):
                                if (
                                    gen_idx_int is not None
                                    and isinstance(ss.GENCLS.D.v, (list, np.ndarray))
                                    and 0 <= gen_idx_int < len(ss.GENCLS.D.v)
                                ):
                                    D_val = ss.GENCLS.D.v[gen_idx_int]
                                elif (
                                    isinstance(ss.GENCLS.D.v, (list, np.ndarray))
                                    and len(ss.GENCLS.D.v) > 0
                                ):
                                    # Fallback to first generator if gen_idx_to_modify is None or invalid
                                    D_val = ss.GENCLS.D.v[0]
                                else:
                                    D_val = ss.GENCLS.D.v
                            else:
                                D_val = 1.0  # Default

                            # Create DataFrame with Pe(t) as input
                            # CRITICAL: Use model-truth Pm from extract_trajectories() (not Pm_to_use)
                            # extract_trajectories() already stores model truth in "Pm" and metadata in "Pm_requested"
                            Pm_from_trajectories = trajectories.get("Pm", None)
                            if Pm_from_trajectories is None:
                                # Fallback if extract_trajectories() didn't provide Pm
                                Pm_from_trajectories = np.full_like(time, Pm_to_use)
                            elif not isinstance(Pm_from_trajectories, np.ndarray):
                                # Convert scalar to array if needed
                                Pm_from_trajectories = np.full_like(
                                    time, float(Pm_from_trajectories)
                                )

                            data = {
                                "time": time,
                                "delta": trajectories.get("delta", np.zeros_like(time)),
                                "delta_deg": np.degrees(
                                    trajectories.get("delta", np.zeros_like(time))
                                ),
                                "omega": trajectories.get("omega", np.ones_like(time)),
                                "omega_deviation": trajectories.get("omega", np.ones_like(time))
                                - 1.0,
                                "Pe": Pe,  # Pe(t) as input
                                "Pm": Pm_from_trajectories,  # Use model-truth Pm from extract_trajectories() (for physics loss)
                                "load": (
                                    Load_P
                                    if use_load_mode and Load_P is not None
                                    else Pm_from_trajectories
                                ),  # Load power (for model input)
                                "M": M_val,
                                "D": D_val,
                                "H": M_val / 2.0,
                                "delta0": (
                                    trajectories.get("delta", np.array([0.0]))[0]
                                    if len(trajectories.get("delta", np.array([0.0]))) > 0
                                    else 0.0
                                ),
                                "omega0": (
                                    trajectories.get("omega", np.array([1.0]))[0]
                                    if len(trajectories.get("omega", np.array([1.0]))) > 0
                                    else 1.0
                                ),
                            }

                            # Get fault times (still needed for state labeling)
                            if hasattr(ss, "Fault") and ss.Fault.n > 0:
                                fault_data = ss.Fault.as_df()
                                if len(fault_data) > 0:
                                    data["tf"] = fault_data.iloc[0].get("tf", fault_start_time)
                                    data["tc"] = fault_data.iloc[0].get("tc", tc)
                                else:
                                    data["tf"] = fault_start_time
                                    data["tc"] = tc
                            else:
                                data["tf"] = fault_start_time
                                data["tc"] = tc

                            # Label system states
                            from .andes_extractor import label_system_states

                            data["state"] = label_system_states(time, data["tf"], data["tc"])

                            # Add Pm_requested if it exists (metadata from sampler)
                            if "Pm_requested" in trajectories:
                                data["Pm_requested"] = trajectories["Pm_requested"]

                            # Multimachine: add per-generator Pe_0..Pe_{n-1} (real per-machine Pe(t))
                            if n_gen > 1 and isinstance(Pe_raw, dict):
                                time_pe_orig = pe_data.get("time", time)
                                for i in range(n_gen):
                                    pe_i = Pe_raw.get(i, Pe_raw.get(str(i), np.zeros_like(time)))
                                    pe_i = np.asarray(pe_i)
                                    if len(pe_i) != len(time) and len(time) > 0 and len(pe_i) > 0:
                                        if len(time_pe_orig) == len(pe_i):
                                            pe_i = np.interp(time, time_pe_orig, pe_i)
                                        elif len(pe_i) > len(time):
                                            pe_i = pe_i[: len(time)]
                                        else:
                                            pe_i = np.resize(pe_i, len(time))
                                    data["Pe_" + str(i)] = pe_i

                            # Multimachine: add per-generator Pm_0..Pm_{n-1} from GENCLS.tm0.v (for 9-dim input / per-machine Pm)
                            if (
                                n_gen > 1
                                and hasattr(ss, "GENCLS")
                                and hasattr(ss.GENCLS, "tm0")
                                and hasattr(ss.GENCLS.tm0, "v")
                            ):
                                tm0v = ss.GENCLS.tm0.v
                                for i in range(n_gen):
                                    if hasattr(tm0v, "__len__") and len(tm0v) > i:
                                        pm_i = float(tm0v[i])
                                    else:
                                        pm_i = Pm_to_use if Pm_to_use is not None else 0.8
                                    data["Pm_" + str(i)] = np.full_like(time, pm_i)

                            # Multimachine: add per-generator delta, delta_deg, omega and COI-relative angles
                            if n_gen > 1:
                                for i in range(n_gen):
                                    try:
                                        traj_i = extract_trajectories(
                                            ss, gen_idx=i, Pm_actual=Pm_to_use
                                        )
                                        d_i = traj_i.get("delta", np.zeros_like(time))
                                        o_i = traj_i.get("omega", np.ones_like(time))
                                        if len(d_i) != len(time) and len(time) > 0 and len(d_i) > 0:
                                            t_traj = traj_i.get("time", time)
                                            if len(t_traj) == len(d_i):
                                                d_i = np.interp(time, t_traj, d_i)
                                                o_i = np.interp(time, t_traj, o_i)
                                            else:
                                                d_i = np.resize(d_i, len(time))
                                                o_i = np.resize(o_i, len(time))
                                        data["delta_" + str(i)] = d_i
                                        data["delta_deg_" + str(i)] = np.degrees(d_i)
                                        data["omega_" + str(i)] = o_i
                                    except Exception:
                                        data["delta_" + str(i)] = np.zeros_like(time)
                                        data["delta_deg_" + str(i)] = np.zeros_like(time)
                                        data["omega_" + str(i)] = np.ones_like(time)
                                # COI and relative rotor angles (δ_rel_i = δ_i − δ_COI), same as run_kundur_fault_expt
                                if hasattr(ss.GENCLS, "M") and hasattr(ss.GENCLS.M, "v"):
                                    M_arr = np.asarray(ss.GENCLS.M.v).flatten()
                                    if len(M_arr) >= n_gen:
                                        M_vals = M_arr[:n_gen]
                                        M_sum = float(np.sum(M_vals))
                                        if M_sum > 0:
                                            delta_stack = np.array(
                                                [data["delta_" + str(i)] for i in range(n_gen)]
                                            )
                                            delta_coi = (
                                                np.sum(M_vals[:, np.newaxis] * delta_stack, axis=0)
                                                / M_sum
                                            )
                                            for i in range(n_gen):
                                                d_rel = data["delta_" + str(i)] - delta_coi
                                                data["delta_rel_" + str(i)] = d_rel
                                                data["delta_rel_deg_" + str(i)] = np.degrees(d_rel)
                                else:
                                    for i in range(n_gen):
                                        data["delta_rel_" + str(i)] = data["delta_" + str(i)]
                                        data["delta_rel_deg_" + str(i)] = data[
                                            "delta_deg_" + str(i)
                                        ]

                            df = pd.DataFrame(data)
                        else:
                            # Use existing reactance-based approach
                            df = extract_complete_dataset(
                                ss, gen_idx=gen_idx_to_modify, Pm_actual=Pm_to_use
                            )
                    except Exception as e:
                        error_summary["data_extraction_failed"] += 1
                        if verbose:
                            print(
                                f"Warning: Data extraction failed for M={M:.2f}, D={D:.2f},"
                                f"tc={tc:.3f}: {e}. Skipping."
                            )
                        continue

                    # Validate that we have data
                    if df is None or len(df) == 0:
                        error_summary["data_extraction_failed"] += 1
                        if verbose:
                            print(
                                f"Warning: No data extracted for M={M:.2f}, D={D:.2f}, tc={tc:.3f}."
                                f"Skipping."
                            )
                        continue

                    # VERIFY: Check actual time step from extracted data (first trajectory only, to avoid spam)
                    if time_step is not None and not hasattr(
                        generate_parameter_sweep, "_data_time_step_checked"
                    ):
                        if "time" in df.columns and len(df) > 1:
                            time_values = df["time"].values
                            time_values_sorted = np.sort(np.unique(time_values))
                            if len(time_values_sorted) > 1:
                                time_diffs = np.diff(time_values_sorted)
                                actual_time_step_from_data = np.median(
                                    time_diffs
                                )  # Use median to avoid outliers

                                if abs(actual_time_step_from_data - time_step) > 1e-6:
                                    generate_parameter_sweep._data_time_step_checked = True
                                    warning_msg = (
                                        f"\n{'=' * 70}\n"
                                        f"[VERIFIED] Actual time step from data differs from "
                                        f"requested!\n"
                                        f"{'=' * 70}\n"
                                        f"  Requested: {time_step * 1000:.3f} ms\n"
                                        f"Actual (from data): {actual_time_step_from_data * 1000:.3f} ms\n"
                                        f"Difference: {abs(actual_time_step_from_data - time_step) * 1000:.3f} ms\n"
                                        f"  Time points per trajectory: {len(time_values_sorted)}\n"
                                        f"  Expected: ~{int(simulation_time / time_step)}\n"
                                        f"\n"
                                        f"  This confirms ANDES is using a different time step.\n"
                                        f"CCT accuracy will be limited to "
                                        f"~{actual_time_step_from_data * 1000:.1f} ms resolution.\n"
                                        f"{'=' * 70}\n"
                                    )
                                    # Use module-level sys import
                                    import sys as _sys_module

                                    print(warning_msg, file=_sys_module.stderr)
                                    if verbose:
                                        print(warning_msg)

                    # Add parameter metadata (use fault_bus_actual so stored bus matches simulation)
                    df["param_M"] = M
                    df["param_D"] = D
                    df["param_H"] = M / 2.0
                    df["param_tc"] = tc
                    df["param_tf"] = fault_start_time_actual
                    df["param_fault_bus"] = fault_bus_actual
                    df["scenario_id"] = current_combination

                    # Add Pm to metadata (used for CCT finding, should be stored)
                    df["param_Pm"] = Pm_to_use
                    # Add load to metadata (for load variation mode, used as model input)
                    if use_load_mode and Load_P is not None:
                        df["param_load"] = Load_P

                    # Add CCT information if CCT-based sampling was used and CCT was found
                    if (
                        use_cct_based_sampling
                        and cct_info_for_this_pair["cct_absolute"] is not None
                    ):
                        df["param_cct_absolute"] = cct_info_for_this_pair["cct_absolute"]
                        df["param_cct_duration"] = cct_info_for_this_pair["cct_duration"]
                        df["param_cct_uncertainty"] = cct_info_for_this_pair["cct_uncertainty"]
                        df["param_small_delta"] = cct_info_for_this_pair["small_delta"]
                        df["param_offset_from_cct"] = tc - cct_info_for_this_pair["cct_absolute"]

                        # Add CCT-based stability label (physics-based)
                        # CCT is the maximum stable clearing time, so:
                        # Clearing time <= CCT should be stable, clearing time > CCT should be unstable
                        # Note: offset = 0 means exactly at CCT, which is stable by definition
                        df["is_stable_from_cct"] = tc <= cct_info_for_this_pair["cct_absolute"]
                    else:
                        # Set to NaN if CCT was not found or not using CCT-based sampling
                        df["param_cct_absolute"] = np.nan
                        df["param_cct_duration"] = np.nan
                        df["param_cct_uncertainty"] = np.nan
                        df["param_small_delta"] = np.nan
                        df["param_offset_from_cct"] = np.nan
                        df["is_stable_from_cct"] = np.nan

                    # Add trajectory-based stability label (primary method)
                    # Multimachine: use COI-based relative rotor angle (δ − δ_COI), same as run_kundur_fault_expt / standard TSA
                    n_gen_df = getattr(ss.GENCLS, "n", 1) if hasattr(ss, "GENCLS") else 1
                    if n_gen_df > 1 and "delta_0" in df.columns and "omega_0" in df.columns:
                        try:
                            if (
                                check_stability_multimachine_coi is not None
                                and hasattr(ss.GENCLS, "M")
                                and hasattr(ss.GENCLS.M, "v")
                            ):
                                M_arr = np.asarray(ss.GENCLS.M.v).flatten()
                                if len(M_arr) >= n_gen_df:
                                    delta_per = [
                                        df["delta_" + str(ii)].values for ii in range(n_gen_df)
                                    ]
                                    omega_per = [
                                        df["omega_" + str(ii)].values for ii in range(n_gen_df)
                                    ]
                                    M_vals = M_arr[:n_gen_df]
                                    df["is_stable"] = check_stability_multimachine_coi(
                                        delta_per_gen=delta_per,
                                        omega_per_gen=omega_per,
                                        M_vals=M_vals,
                                        delta_threshold=np.pi,
                                        omega_threshold=1.5,
                                    )
                                else:
                                    # Fallback: compute COI-relative from df (param_M, delta_0..) when possible
                                    _fallback_stable = _multimachine_stability_from_df(df, n_gen_df)
                                    if _fallback_stable is not None:
                                        df["is_stable"] = _fallback_stable
                                    else:
                                        stable_per = []
                                        for ii in range(n_gen_df):
                                            if check_stability is not None:
                                                stable_per.append(
                                                    check_stability(
                                                        df["delta_" + str(ii)].values,
                                                        df["omega_" + str(ii)].values,
                                                        np.pi,
                                                        1.5,
                                                    )
                                                )
                                            else:
                                                max_d = df["delta_" + str(ii)].abs().max()
                                                stable_per.append(max_d < np.pi)
                                        df["is_stable"] = all(stable_per)
                            else:
                                # Fallback: COI-relative from df when possible (same M for all gen)
                                _fallback_stable = _multimachine_stability_from_df(df, n_gen_df)
                                if _fallback_stable is not None:
                                    df["is_stable"] = _fallback_stable
                                else:
                                    stable_per = []
                                    for ii in range(n_gen_df):
                                        max_d = df["delta_" + str(ii)].abs().max()
                                        stable_per.append(max_d < np.pi)
                                    df["is_stable"] = all(stable_per)
                        except Exception:
                            if "delta" in df.columns:
                                if check_stability is not None:
                                    df["is_stable"] = check_stability(
                                        df["delta"].values,
                                        df["omega"].values
                                        if "omega" in df.columns
                                        else np.ones_like(df["delta"].values),
                                        np.pi,
                                        1.5,
                                    )
                                else:
                                    # Stability from rotor angle only (omega commented out)
                                    df["is_stable"] = df["delta"].abs().max() < np.pi
                            else:
                                df["is_stable"] = np.nan
                    elif STABILITY_CHECKER_AVAILABLE and "delta" in df.columns:
                        try:
                            is_stable = check_stability(
                                delta=df["delta"].values,
                                omega=df["omega"].values
                                if "omega" in df.columns
                                else np.ones_like(df["delta"].values),
                                delta_threshold=np.pi,
                                omega_threshold=1.5,
                            )
                            df["is_stable"] = is_stable
                        except Exception:
                            # Stability from rotor angle only (omega commented out)
                            df["is_stable"] = df["delta"].abs().max() < np.pi
                    else:
                        if "delta" in df.columns:
                            # Stability from rotor angle only (omega commented out)
                            df["is_stable"] = df["delta"].abs().max() < np.pi
                        else:
                            df["is_stable"] = np.nan

                    # Validate consistency between trajectory-based and CCT-based stability (if both available)
                    if (
                        use_cct_based_sampling
                        and cct_info_for_this_pair["cct_absolute"] is not None
                    ):
                        if not pd.isna(df["is_stable"].iloc[0]) and not pd.isna(
                            df["is_stable_from_cct"].iloc[0]
                        ):
                            is_consistent = (
                                df["is_stable"].iloc[0] == df["is_stable_from_cct"].iloc[0]
                            )
                            if not is_consistent and verbose:
                                print(
                                    f"[WARNING] Stability inconsistency for M={M:.2f}, D={D:.2f},"
                                    f"tc={tc:.3f}:"
                                    f"trajectory-based={df['is_stable'].iloc[0]}, "
                                    f"CCT-based={df['is_stable_from_cct'].iloc[0]}"
                                )

                    all_datasets.append(df)

                    # Update data quality metrics
                    data_quality_metrics["total_scenarios"] += 1
                    data_quality_metrics["successful_scenarios"] += 1

                    # Clean up temporary case file if it was created
                    if modified_case_created and case_path_to_use != original_case_path:
                        try:
                            if os.path.exists(case_path_to_use):
                                os.remove(case_path_to_use)
                                if verbose:
                                    print(
                                        f"[CLEANUP] Removed temporary case file: {case_path_to_use}"
                                    )
                        except Exception as cleanup_err:
                            if verbose:
                                print(
                                    f"[WARNING] Could not remove temporary case file: {cleanup_err}"
                                )

                    # Update progress tracker
                    if progress_tracker is not None:
                        # Extract metrics from df
                        is_stable_val = (
                            df["is_stable"].iloc[0]
                            if "is_stable" in df.columns and len(df) > 0
                            else None
                        )
                        cct_abs = cct_info_for_this_pair.get("cct_absolute")
                        cct_unc = cct_info_for_this_pair.get("cct_uncertainty")

                        # Extract max angle and max freq dev from trajectory
                        max_angle = None
                        max_freq_dev = None
                        if "delta" in df.columns and len(df) > 0:
                            max_angle_rad = df["delta"].abs().max()
                            max_angle = np.degrees(max_angle_rad)
                        if "omega" in df.columns and len(df) > 0:
                            max_omega_dev_pu = (df["omega"] - 1.0).abs().max()
                            # Convert to Hz (assuming 60 Hz system)
                            max_freq_dev = max_omega_dev_pu * 60.0

                        # Convert M to H for tracking
                        H_val = M / 2.0

                        progress_tracker.update(
                            is_stable=bool(is_stable_val) if is_stable_val is not None else False,
                            cct=cct_abs,
                            cct_uncertainty=cct_unc,
                            max_angle=max_angle,
                            max_freq_dev=max_freq_dev,
                            H=H_val,
                            D=D,
                            Pm=Pm_to_use,
                            cct_found=cct_info_for_this_pair.get("cct_absolute") is not None,
                            simulation_success=True,
                        )

                        # Display progress every sample or every 5 samples for large datasets
                        display_frequency = 1 if total_combinations <= 50 else 5
                        if (
                            current_combination % display_frequency == 0
                            or current_combination == total_combinations - 1
                        ):
                            current_params = {"H": H_val, "D": D, "Pm": Pm_to_use}
                            print(progress_tracker.display_progress(current_params))
                            print()

                    if verbose and current_combination % 10 == 0 and progress_tracker is None:
                        print(
                            f"Progress: {current_combination}/{total_combinations}"
                            f"({100 * current_combination / total_combinations:.1f}%)"
                        )

                except Exception as e:
                    error_summary["other_errors"] += 1

                    # Update tracker for failed simulation
                    if progress_tracker is not None:
                        H_val = M / 2.0
                        progress_tracker.update(
                            is_stable=False,
                            H=H_val,
                            D=D,
                            Pm=Pm_to_use,
                            cct_found=False,
                            simulation_success=False,
                        )

                    if verbose:
                        print(
                            f"Error in combination {current_combination} (M={M:.2f}, D={D:.2f},"
                            f"tc={tc:.3f}): {e}"
                        )
                        import traceback

                        if verbose:
                            traceback.print_exc()
                    continue

    if len(all_datasets) == 0:
        # Print error summary even when verbose=False to help diagnose issues
        # Also print to stderr so it's visible even with stdout suppression
        import sys

        error_msg = "\n" + "=" * 70 + "\n"
        error_msg += "ERROR: No successful simulations. Check parameters and ANDES setup.\n"
        error_msg += "=" * 70 + "\n"
        error_msg += f"Error summary (out of {total_combinations} total attempts):\n"
        error_msg += f"  - No GENCLS found: {error_summary['no_gencls']}\n"
        error_msg += (
            f"  - Generator parameter setting failed: {error_summary['gen_param_failed']}\n"
        )
        error_msg += f"  - Fault configuration failed: {error_summary['fault_config_failed']}\n"
        error_msg += f"  - Power flow failed: {error_summary['powerflow_failed']}\n"
        error_msg += f"  - Power flow did not converge: {error_summary['powerflow_no_converge']}\n"
        error_msg += (
            f"  - Pm mismatch after power flow: {error_summary.get('pm_mismatch_after_pf', 0)}\n"
        )
        error_msg += (
            f"  - Pm zero with non-zero load: {error_summary.get('pm_zero_with_load', 0)}\n"
        )
        error_msg += (
            f"  - Power balance violation: {error_summary.get('power_balance_violation', 0)}\n"
        )
        error_msg += f"  - Pm extraction failed: {error_summary.get('pm_extraction_failed', 0)}\n"
        error_msg += f"  - TDS simulation failed: {error_summary['tds_failed']}\n"
        error_msg += f"  - Data extraction failed: {error_summary['data_extraction_failed']}\n"
        error_msg += f"  - Other errors: {error_summary['other_errors']}\n"
        error_msg += "=" * 70 + "\n"

        # Include first TDS error details if available (helps diagnose the issue)
        if error_summary["first_tds_error"] is not None:
            error_msg += "\nFirst TDS error details (for debugging):\n"
            error_msg += "-" * 70 + "\n"
            if error_summary["first_tds_error_params"] is not None:
                params = error_summary["first_tds_error_params"]
                error_msg += f"Parameters: M={params['M']:.2f}, D={params['D']:.2f}, "
                error_msg += f"Pm={params['Pm']:.3f}, tc={params['tc']:.3f}, "
                error_msg += f"fault_bus={params['fault_bus']}, "
                error_msg += f"fault_start={params['fault_start_time']:.3f}s, "
                error_msg += f"sim_time={params['simulation_time']:.2f}s, "
                error_msg += f"dt={params['time_step']:.4f}s\n"
            error_msg += "\n"
            # Show first 2000 characters of error (full traceback)
            error_details = error_summary["first_tds_error"]
            error_msg += error_details[:2000]
            if len(error_details) > 2000:
                error_msg += f"\n... (truncated, {len(error_details) - 2000} more characters)"
            error_msg += "\n" + "-" * 70 + "\n"

        error_msg += "\nTroubleshooting tips:\n"
        error_msg += "  1. Check that ANDES case file exists and is valid\n"
        error_msg += "  2. Verify GENCLS generator model is present in case file\n"
        error_msg += "  3. Check that fault model exists or can be added\n"
        error_msg += "  4. Try running with verbose=True to see detailed error messages\n"
        error_msg += "  5. Check ANDES installation and version compatibility\n"
        error_msg += "  6. Review the first TDS error details above for specific failure cause\n"

        # Print to stderr so it's visible even with stdout suppression
        # Use module-level sys import
        import sys as _sys_module

        print(error_msg, file=_sys_module.stderr)
        raise ValueError(error_msg)

    # Display final summary if tracker was used
    if progress_tracker is not None:
        print()
        print(progress_tracker.display_summary())
        print()

    # Calculate final data quality metrics
    if data_quality_metrics["total_scenarios"] > 0:
        data_quality_metrics["power_flow_convergence_rate"] = (
            data_quality_metrics["total_scenarios"]
            - error_summary["powerflow_no_converge"]
            - error_summary["powerflow_failed"]
        ) / data_quality_metrics["total_scenarios"]
        data_quality_metrics["pm_verification_pass_rate"] = (
            data_quality_metrics["total_scenarios"] - error_summary.get("pm_mismatch_after_pf", 0)
        ) / data_quality_metrics["total_scenarios"]

        if verbose:
            print("\n" + "=" * 70)
            print("Data Quality Metrics")
            print("=" * 70)
            print(f"Total scenarios attempted: {data_quality_metrics['total_scenarios']}")
            print(f"Successful scenarios: {data_quality_metrics['successful_scenarios']}")
            print(f"Rejected scenarios: {data_quality_metrics['rejected_scenarios']}")
            print(
                f"Power flow convergence rate:"
                f"{data_quality_metrics['power_flow_convergence_rate'] * 100:.1f}%"
            )
            print(
                f"Pm verification pass rate:"
                f"{data_quality_metrics['pm_verification_pass_rate'] * 100:.1f}%"
            )
            if data_quality_metrics["rejection_reasons"]:
                print("\nRejection reasons:")
                for reason, count in data_quality_metrics["rejection_reasons"].items():
                    print(f"  - {reason}: {count}")
            print("=" * 70 + "\n")

    # Combine all datasets
    combined_df = pd.concat(all_datasets, ignore_index=True)

    # When CCT-based sampling was used but bisection failed: fill param_cct_absolute from
    # stable/unstable boundary (estimated CCT) so the CSV has CCT values for analysis/plots
    if (
        use_cct_based_sampling
        and len(combined_df) > 0
        and "scenario_id" in combined_df.columns
        and "param_cct_absolute" in combined_df.columns
        and "param_tc" in combined_df.columns
        and "is_stable" in combined_df.columns
    ):
        scenarios_first = combined_df.groupby("scenario_id").first()
        # Group by (H, D, load or Pm) to get one estimated CCT per operating point
        h_col = "param_H" if "param_H" in scenarios_first.columns else None
        d_col = "param_D" if "param_D" in scenarios_first.columns else None
        load_col = "param_load" if "param_load" in scenarios_first.columns else "param_Pm"
        if h_col and d_col and load_col in scenarios_first.columns:
            nan_cct_count = scenarios_first["param_cct_absolute"].isna().sum()
            if nan_cct_count > 0:
                fault_start = float(
                    scenarios_first["param_tf"].iloc[0]
                    if "param_tf" in scenarios_first.columns
                    and pd.notna(scenarios_first["param_tf"].iloc[0])
                    else fault_start_time
                )
                filled = 0
                for _keys, grp in scenarios_first.groupby([h_col, d_col, load_col], dropna=False):
                    if grp["param_cct_absolute"].notna().all():
                        continue
                    stable_tc = grp.loc[grp["is_stable"], "param_tc"].dropna()
                    unstable_tc = grp.loc[~grp["is_stable"], "param_tc"].dropna()
                    # Estimate CCT so every scenario in this operating point gets a value
                    if len(stable_tc) > 0 and len(unstable_tc) > 0:
                        estimated_cct = (float(stable_tc.max()) + float(unstable_tc.min())) / 2.0
                    elif len(stable_tc) > 0:
                        estimated_cct = float(stable_tc.max())  # lower bound: CCT >= max(stable tc)
                    elif len(unstable_tc) > 0:
                        estimated_cct = float(
                            unstable_tc.min()
                        )  # upper bound: CCT < min(unstable tc)
                    else:
                        continue
                    for sid in grp.index:
                        mask = (combined_df["scenario_id"] == sid) & (
                            combined_df["param_cct_absolute"].isna()
                        )
                        if mask.any():
                            combined_df.loc[mask, "param_cct_absolute"] = estimated_cct
                            combined_df.loc[mask, "param_cct_duration"] = (
                                estimated_cct - fault_start
                            )
                            filled += 1
                if filled > 0 and verbose:
                    print(
                        f"[INFO] Filled param_cct_absolute (estimated from stable/unstable boundary) "
                        f"for {filled} scenario(s) where bisection did not provide CCT."
                    )

    # VERIFY: Report actual time step statistics from generated data
    if time_step is not None and len(combined_df) > 0 and "time" in combined_df.columns:
        # Sample a few scenarios to get time step statistics
        time_step_stats = []
        for scenario_id in combined_df["scenario_id"].unique()[
            : min(5, combined_df["scenario_id"].nunique())
        ]:
            scenario_data = combined_df[combined_df["scenario_id"] == scenario_id]
            if len(scenario_data) > 1:
                time_values = np.sort(scenario_data["time"].unique())
                if len(time_values) > 1:
                    time_diffs = np.diff(time_values)
                    # Use median to avoid outliers from adaptive stepping or filtering
                    median_dt = np.median(time_diffs)
                    time_step_stats.append(median_dt)

        if time_step_stats:
            actual_time_step_median = np.median(time_step_stats)
            actual_time_step_mean = np.mean(time_step_stats)
            actual_time_step_std = np.std(time_step_stats)

            # Only report if significantly different from requested
            if abs(actual_time_step_median - time_step) > 1e-6:
                summary_msg = (
                    f"\n{'=' * 70}\n"
                    f"Time Step Verification Summary\n"
                    f"{'=' * 70}\n"
                    f"  Requested time step: {time_step * 1000:.3f} ms\n"
                    f"  Actual time step (median): {actual_time_step_median * 1000:.3f} ms\n"
                    f"  Actual time step (mean): {actual_time_step_mean * 1000:.3f} ms\n"
                    f"  Std deviation: {actual_time_step_std * 1000:.3f} ms\n"
                    f"  Difference: {abs(actual_time_step_median - time_step) * 1000:.3f} ms\n"
                    f"\n"
                    f"  Impact on data:\n"
                    f"- Time points per trajectory: ~{int(simulation_time / actual_time_step_median)} "
                    f"(expected: ~{int(simulation_time / time_step)})\n"
                    f"  - Total data points: {len(combined_df):,}\n"
                    f"  - CCT accuracy limit: ~{actual_time_step_median * 1000:.1f} ms\n"
                    f"\n"
                    f"  Note: ANDES may override time step based on system frequency.\n"
                    f"  For CCT accuracy < 2 ms, consider checking ANDES documentation\n"
                    f"for forcing fixed time step or accepting {actual_time_step_median * 1000:.1f} ms resolution.\n"
                    f"{'=' * 70}\n"
                )
                if verbose:
                    print(summary_msg)
                # Also print to stderr so it's visible even with stdout suppression
                # Use module-level sys import
                import sys as _sys_module

                print(summary_msg, file=_sys_module.stderr)

    # Save combined dataset with timestamp
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"parameter_sweep_data_{timestamp}.csv"
    combined_df.to_csv(output_file, index=False)

    if verbose:
        print(f"\n[OK] Generated {len(all_datasets)} successful simulations")
        print(f"[OK] Total data points: {len(combined_df)}")
        print(f"[OK] Saved to: {output_file}")

    return combined_df


def generate_focused_sweep(
    case_file: str,
    output_dir: str = "data",
    H_values: Optional[List[float]] = None,
    D_values: Optional[List[float]] = None,
    tc_values: Optional[List[float]] = None,
    fault_start_time: float = 1.0,
    **kwargs,
) -> pd.DataFrame:
    """
    Generate parameter sweep with specific parameter values.

    Parameters:
    -----------
    case_file : str
        Path to ANDES case file
    output_dir : str
        Output directory
    H_values : list, optional
        Specific H values to use
    D_values : list, optional
        Specific D values to use
    tc_values : list, optional
        Specific fault clearing times to use
    fault_start_time : float, optional
        Fault start time in seconds (default: 1.0)
    **kwargs : dict
        Additional arguments passed to generate_parameter_sweep

    Returns:
    --------
    pd.DataFrame : Combined dataset
    """
    if H_values is None:
        H_values = [3.0, 4.0, 5.0, 6.0, 7.0]
    if D_values is None:
        D_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    if tc_values is None:
        tc_values = [0.15, 0.18, 0.20, 0.22, 0.25]

    # Extract use_pe_as_input from kwargs (used later in the function)
    use_pe_as_input = kwargs.get("use_pe_as_input", False)

    M_values = [2.0 * H for H in H_values]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get case file path
    if not os.path.isabs(case_file):
        case_path = andes.get_case(case_file)
    else:
        case_path = case_file

    all_datasets = []
    total_combinations = len(M_values) * len(D_values) * len(tc_values)
    current_combination = 0

    print(f"Generating {total_combinations} focused parameter combinations...")

    for M, D, tc in itertools.product(M_values, D_values, tc_values):
        current_combination += 1

        try:
            # Load system with setup=False, then call setup() explicitly (following smib_albert_cct.py pattern)
            ss = andes.load(case_path, setup=False, no_output=True, default_config=True)
            ss.setup()

            # Validate that GENCLS exists and has devices
            if not hasattr(ss, "GENCLS") or ss.GENCLS.n == 0:
                continue

            # Identify infinite bus and determine which generator to modify
            infinite_bus_idx = None
            for i in range(ss.GENCLS.n):
                is_infinite = False
                if hasattr(ss.GENCLS, "name") and hasattr(ss.GENCLS.name, "v"):
                    try:
                        gen_name = (
                            ss.GENCLS.name.v[i]
                            if hasattr(ss.GENCLS.name.v, "__getitem__")
                            else ss.GENCLS.name.v
                        )
                        if gen_name and (
                            "slack" in str(gen_name).lower() or "infinite" in str(gen_name).lower()
                        ):
                            is_infinite = True
                    except (IndexError, AttributeError, TypeError):
                        pass
                if not is_infinite and hasattr(ss.GENCLS, "M") and hasattr(ss.GENCLS.M, "v"):
                    try:
                        M_val = (
                            ss.GENCLS.M.v[i]
                            if hasattr(ss.GENCLS.M.v, "__getitem__")
                            else ss.GENCLS.M.v
                        )
                        if M_val is not None and isinstance(M_val, (int, float)) and M_val > 1e6:
                            is_infinite = True
                    except (IndexError, AttributeError, TypeError):
                        pass
                if is_infinite:
                    infinite_bus_idx = i
                    break

            gen_idx_to_modify = 0
            if infinite_bus_idx == 0 and ss.GENCLS.n > 1:
                gen_idx_to_modify = 1
            elif infinite_bus_idx is not None and infinite_bus_idx != 0:
                gen_idx_to_modify = 0

            # Set generator parameters: same M and D for all generators (uniform H/D design)
            n_gen_tds = getattr(ss.GENCLS, "n", 1)
            try:
                if hasattr(ss.GENCLS, "alter"):
                    for gidx in range(n_gen_tds):
                        ss.GENCLS.alter("M", gidx, M)
                        ss.GENCLS.alter("D", gidx, D)
                else:
                    for gidx in range(n_gen_tds):
                        if gidx < len(ss.GENCLS.M.v):
                            ss.GENCLS.M.v[gidx] = M
                        if gidx < len(ss.GENCLS.D.v):
                            ss.GENCLS.D.v[gidx] = D
            except Exception:
                try:
                    for gidx in range(n_gen_tds):
                        if gidx < len(ss.GENCLS.M.v):
                            ss.GENCLS.M.v[gidx] = M
                        if gidx < len(ss.GENCLS.D.v):
                            ss.GENCLS.D.v[gidx] = D
                except Exception:
                    continue

            # Set fault parameters
            if hasattr(ss, "Fault") and ss.Fault.n > 0:
                try:
                    if hasattr(ss.Fault, "alter"):
                        ss.Fault.alter("tc", 0, tc)
                    else:
                        ss.Fault.tc.v[0] = tc
                except Exception:
                    continue
            else:
                continue

            # CRITICAL: Run power flow FIRST to establish steady-state equilibrium
            if hasattr(ss, "PFlow"):
                if hasattr(ss.PFlow, "converged"):
                    ss.PFlow.converged = False
                try:
                    ss.PFlow.run()
                except Exception:
                    continue
                if not (hasattr(ss.PFlow, "converged") and ss.PFlow.converged):
                    if hasattr(ss.PFlow, "converged"):
                        ss.PFlow.converged = False
                    try:
                        ss.PFlow.run()
                    except Exception:
                        pass
                    if not (hasattr(ss.PFlow, "converged") and ss.PFlow.converged):
                        continue
            else:
                continue

            # Setup and run TDS
            ss.TDS.config.tf = kwargs.get("simulation_time", 5.0)
            time_step_val = kwargs.get("time_step", 0.001)
            # Use proper ANDES parameters: fixt, tstep, shrinkt
            if hasattr(ss.TDS.config, "fixt"):
                ss.TDS.config.fixt = 1  # Fixed step mode
            if hasattr(ss.TDS.config, "tstep"):
                ss.TDS.config.tstep = time_step_val  # CRITICAL: Use tstep
            if hasattr(ss.TDS.config, "shrinkt"):
                ss.TDS.config.shrinkt = 0  # Don't allow shrinking
            # Also set h as fallback
            if hasattr(ss.TDS.config, "h"):
                ss.TDS.config.h = time_step_val
            ss.TDS.config.tol = 1e-4

            # Disable automatic stability criteria checking
            if hasattr(ss.TDS.config, "criteria"):
                ss.TDS.config.criteria = 0

            # Enable plotter to store time series data
            if hasattr(ss.TDS.config, "plot"):
                ss.TDS.config.plot = True
            if hasattr(ss.TDS.config, "save_plt"):
                ss.TDS.config.save_plt = True
            if hasattr(ss.TDS.config, "store"):
                ss.TDS.config.store = True
            # CRITICAL: Set save_interval=1 to save every time step (not downsampled to 30 Hz)
            if hasattr(ss.TDS.config, "save_interval"):
                ss.TDS.config.save_interval = 1  # Save every time step

            # Disable progress bars to avoid tqdm_notebook/ipywidgets dependency issues
            if hasattr(ss.TDS.config, "no_tqdm"):
                ss.TDS.config.no_tqdm = True
            if hasattr(ss.TDS.config, "show_progress"):
                ss.TDS.config.show_progress = False
            if hasattr(ss.TDS.config, "progress"):
                ss.TDS.config.progress = False

            # Initialize TDS
            if hasattr(ss.TDS, "init"):
                ss.TDS.init()

            # Disable Toggles (e.g. line trip at t=2s in Kundur) so only the fault event occurs
            if hasattr(ss, "Toggle") and ss.Toggle.n > 0:
                try:
                    for i in range(ss.Toggle.n):
                        try:
                            if hasattr(ss.Toggle, "u") and hasattr(ss.Toggle.u, "v"):
                                ss.Toggle.u.v[i] = 0
                            else:
                                ss.Toggle.alter("u", i + 1, 0)
                        except Exception:
                            ss.Toggle.alter("u", i + 1, 0)
                except Exception:
                    pass

            # Run TDS
            try:
                ss.TDS.run()
            except Exception:
                continue

            if hasattr(ss, "exit_code") and ss.exit_code != 0:
                continue

            # Extract dataset
            try:
                if use_pe_as_input:
                    # Extract Pe(t) trajectories and other data for Pe-input approach
                    pe_data = extract_pe_trajectories(
                        ss, gen_idx=gen_idx_to_modify, Pm_actual=Pm_to_use
                    )
                    trajectories = extract_trajectories(
                        ss, gen_idx=gen_idx_to_modify, Pm_actual=Pm_to_use
                    )

                    # Build DataFrame with Pe(t) as input (no reactances)
                    # Ensure time vectors match - use the one from trajectories as primary source
                    time_pe = pe_data.get("time", None)
                    time_traj = trajectories.get("time", None)

                    if time_traj is not None:
                        time = time_traj
                    elif time_pe is not None:
                        time = time_pe
                    else:
                        time = np.array([])

                    # Validate time vector length matches trajectory data
                    if len(time) > 0:
                        delta_sample = trajectories.get("delta", None)
                        if delta_sample is not None and hasattr(delta_sample, "__len__"):
                            if len(delta_sample) != len(time):
                                time_pe_len = len(time_pe) if time_pe is not None else 0
                                time_traj_len = len(time_traj) if time_traj is not None else 0
                                warnings.warn(
                                    f"Time vector length mismatch: pe_data time "
                                    f"length={time_pe_len}, "
                                    f"trajectories time length={time_traj_len}, "
                                    f"delta length={len(delta_sample)}. Using trajectories time vector.",
                                    UserWarning,
                                    stacklevel=2,
                                )
                                # Correct time to match delta length - use time_traj if available
                                if time_traj is not None and len(time_traj) == len(delta_sample):
                                    time = time_traj
                                elif time_pe is not None and len(time_pe) == len(delta_sample):
                                    time = time_pe
                                # If neither matches, keep current time (will cause issues downstream)

                    # Initialize Pe_raw using the corrected time vector
                    Pe_raw = pe_data.get("Pe", trajectories.get("Pe", np.zeros_like(time)))

                    # Handle case where Pe might be a dict (multi-machine) or array (single machine)
                    if isinstance(Pe_raw, dict):
                        # Multi-machine case: use the generator index or first machine
                        # Handle type mismatch: gen_idx_to_modify might be int, but dict keys might be int or str
                        pe_key = None
                        if gen_idx_to_modify is not None:
                            # Try exact match first
                            if gen_idx_to_modify in Pe_raw:
                                pe_key = gen_idx_to_modify
                            else:
                                # Try all possible type conversions systematically
                                # Try converting gen_idx_to_modify to int
                                try:
                                    gen_idx_int = int(float(gen_idx_to_modify))
                                    if gen_idx_int in Pe_raw:
                                        pe_key = gen_idx_int
                                except (ValueError, TypeError):
                                    pass

                                # Try converting gen_idx_to_modify to string
                                if pe_key is None:
                                    try:
                                        gen_idx_str = str(int(float(gen_idx_to_modify)))
                                        if gen_idx_str in Pe_raw:
                                            pe_key = gen_idx_str
                                    except (ValueError, TypeError):
                                        pass

                                # Try matching against dict keys with type conversions
                                if pe_key is None:
                                    for key in Pe_raw.keys():
                                        try:
                                            # Try converting dict key to match gen_idx_to_modify type
                                            if isinstance(gen_idx_to_modify, (int, float)):
                                                key_as_int = int(float(key))
                                                if (
                                                    abs(key_as_int - float(gen_idx_to_modify))
                                                    < 1e-6
                                                ):
                                                    pe_key = key
                                                    break
                                            elif isinstance(gen_idx_to_modify, str):
                                                if str(key) == gen_idx_to_modify or int(
                                                    float(key)
                                                ) == int(float(gen_idx_to_modify)):
                                                    pe_key = key
                                                    break
                                        except (ValueError, TypeError):
                                            continue

                        if pe_key is not None:
                            Pe = Pe_raw[pe_key]
                        elif len(Pe_raw) > 0:
                            # Use first available machine
                            # Issue warning if gen_idx_to_modify is None (should specify which machine) or if it couldn't be matched
                            if gen_idx_to_modify is None:
                                warnings.warn(
                                    f"gen_idx_to_modify is None but Pe is a dictionary"
                                    f"(multi-machine case)."
                                    f"Falling back to first machine's Pe trajectory (key:"
                                    f"{list(Pe_raw.keys())[0]})."
                                    f"This may cause data corruption if Pe trajectory is paired"
                                    f"with wrong machine's parameters."
                                    f"Available keys: {list(Pe_raw.keys())}",
                                    UserWarning,
                                    stacklevel=2,
                                )
                            else:
                                warnings.warn(
                                    f"Could not match gen_idx_to_modify={gen_idx_to_modify} "
                                    f"with any key in Pe dictionary (keys: {list(Pe_raw.keys())}). "
                                    f"Falling back to first machine's Pe trajectory (key:"
                                    f"{list(Pe_raw.keys())[0]})."
                                    f"This may cause data corruption if Pe trajectory is paired with wrong machine's parameters.",
                                    UserWarning,
                                    stacklevel=2,
                                )
                            Pe = Pe_raw[list(Pe_raw.keys())[0]]
                        else:
                            Pe = np.zeros_like(time)
                    else:
                        # Single machine case: Pe is already an array
                        Pe = Pe_raw

                    # Get generator parameters
                    # Convert gen_idx_to_modify to int if it's a string (for array indexing)
                    gen_idx_int = None
                    if gen_idx_to_modify is not None:
                        if isinstance(gen_idx_to_modify, (int, np.integer)):
                            gen_idx_int = int(gen_idx_to_modify)
                        elif isinstance(gen_idx_to_modify, str):
                            try:
                                gen_idx_int = int(gen_idx_to_modify)
                            except (ValueError, TypeError):
                                gen_idx_int = None

                    if hasattr(ss.GENCLS, "M") and hasattr(ss.GENCLS.M, "v"):
                        if (
                            gen_idx_int is not None
                            and isinstance(ss.GENCLS.M.v, (list, np.ndarray))
                            and 0 <= gen_idx_int < len(ss.GENCLS.M.v)
                        ):
                            M_val = ss.GENCLS.M.v[gen_idx_int]
                        elif (
                            isinstance(ss.GENCLS.M.v, (list, np.ndarray)) and len(ss.GENCLS.M.v) > 0
                        ):
                            # Fallback to first generator if gen_idx_to_modify is None or invalid
                            M_val = ss.GENCLS.M.v[0]
                        else:
                            M_val = ss.GENCLS.M.v
                    else:
                        M_val = 5.7512  # Default

                    if hasattr(ss.GENCLS, "D") and hasattr(ss.GENCLS.D, "v"):
                        if (
                            gen_idx_int is not None
                            and isinstance(ss.GENCLS.D.v, (list, np.ndarray))
                            and 0 <= gen_idx_int < len(ss.GENCLS.D.v)
                        ):
                            D_val = ss.GENCLS.D.v[gen_idx_int]
                        elif (
                            isinstance(ss.GENCLS.D.v, (list, np.ndarray)) and len(ss.GENCLS.D.v) > 0
                        ):
                            # Fallback to first generator if gen_idx_to_modify is None or invalid
                            D_val = ss.GENCLS.D.v[0]
                        else:
                            D_val = ss.GENCLS.D.v
                    else:
                        D_val = 1.0  # Default

                    # Create DataFrame with Pe(t) as input
                    # CRITICAL: Use model-truth Pm from extract_trajectories() (not Pm_to_use)
                    # extract_trajectories() already stores model truth in "Pm" and metadata in "Pm_requested"
                    Pm_from_trajectories = trajectories.get("Pm", None)
                    if Pm_from_trajectories is None:
                        # Fallback if extract_trajectories() didn't provide Pm
                        Pm_from_trajectories = np.full_like(time, Pm_to_use)
                    elif not isinstance(Pm_from_trajectories, np.ndarray):
                        # Convert scalar to array if needed
                        Pm_from_trajectories = np.full_like(time, float(Pm_from_trajectories))

                    data = {
                        "time": time,
                        "delta": trajectories.get("delta", np.zeros_like(time)),
                        "delta_deg": np.degrees(trajectories.get("delta", np.zeros_like(time))),
                        "omega": trajectories.get("omega", np.ones_like(time)),
                        "omega_deviation": trajectories.get("omega", np.ones_like(time)) - 1.0,
                        "Pe": Pe,  # Pe(t) as input
                        "Pm": Pm_from_trajectories,  # Use model-truth Pm from extract_trajectories() (for physics loss)
                        "load": (
                            Load_P if use_load_mode and Load_P is not None else Pm_from_trajectories
                        ),  # Load power (for model input)
                        "M": M_val,
                        "D": D_val,
                        "H": M_val / 2.0,
                        "delta0": (
                            trajectories.get("delta", np.array([0.0]))[0]
                            if len(trajectories.get("delta", np.array([0.0]))) > 0
                            else 0.0
                        ),
                        "omega0": (
                            trajectories.get("omega", np.array([1.0]))[0]
                            if len(trajectories.get("omega", np.array([1.0]))) > 0
                            else 1.0
                        ),
                    }

                    # Get fault times from system configuration, not loop variable
                    # fault_start_time is now an explicit parameter
                    fault_start_time_actual = fault_start_time
                    if hasattr(ss, "Fault") and ss.Fault.n > 0:
                        fault_data = ss.Fault.as_df()
                        if len(fault_data) > 0:
                            # Get actual fault clearing time from system configuration
                            tc_from_system = fault_data.iloc[0].get("tc", None)
                            data["tf"] = fault_data.iloc[0].get("tf", fault_start_time_actual)
                            # Prioritize system's tc from fault configuration, use loop variable tc only as fallback
                            # The loop variable tc represents the intended clearing time being swept
                            if tc_from_system is not None:
                                data["tc"] = tc_from_system
                            else:
                                data[
                                    "tc"
                                ] = tc  # Fallback to loop variable (the intended clearing time)
                        else:
                            data["tf"] = fault_start_time_actual
                            data["tc"] = tc  # Loop variable is the intended clearing time
                    else:
                        data["tf"] = fault_start_time_actual
                        data["tc"] = tc  # Loop variable is the intended clearing time

                    # Label system states
                    from .andes_extractor import label_system_states

                    data["state"] = label_system_states(time, data["tf"], data["tc"])

                    # Add Pm_requested if it exists (metadata from sampler)
                    if "Pm_requested" in trajectories:
                        data["Pm_requested"] = trajectories["Pm_requested"]

                    df = pd.DataFrame(data)
                else:
                    # Use existing reactance-based approach
                    df = extract_complete_dataset(
                        ss, gen_idx=gen_idx_to_modify, Pm_actual=Pm_to_use
                    )
            except Exception:
                continue

            if df is None or len(df) == 0:
                continue

            df["param_M"] = M
            df["param_D"] = D
            df["param_H"] = M / 2.0
            df["param_tc"] = tc
            df["param_tf"] = fault_start_time
            df["param_fault_bus"] = fault_bus_actual
            df["scenario_id"] = current_combination

            # Add stability labels (trajectory-based)
            if STABILITY_CHECKER_AVAILABLE and "delta" in df.columns:
                try:
                    is_stable = check_stability(
                        delta=df["delta"].values,
                        omega=df["omega"].values
                        if "omega" in df.columns
                        else np.ones_like(df["delta"].values),
                        delta_threshold=np.pi,
                        omega_threshold=1.5,
                    )
                    df["is_stable"] = is_stable
                except Exception:
                    # Fallback: stability from rotor angle only (omega commented out)
                    df["is_stable"] = df["delta"].abs().max() < np.pi
            else:
                # Fallback: stability from rotor angle only
                if "delta" in df.columns:
                    df["is_stable"] = df["delta"].abs().max() < np.pi
                else:
                    df["is_stable"] = np.nan

            # CCT-based stability not available in this context
            df["is_stable_from_cct"] = np.nan

            all_datasets.append(df)

            if current_combination % 10 == 0:
                print(f"Progress: {current_combination}/{total_combinations}")

        except Exception as e:
            print(f"Error in combination {current_combination}: {e}")
            continue

    if len(all_datasets) == 0:
        raise ValueError("No successful simulations.")

    combined_df = pd.concat(all_datasets, ignore_index=True)

    # Save with timestamp
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"focused_sweep_data_{timestamp}.csv"
    combined_df.to_csv(output_file, index=False)

    print(f"[OK] Generated {len(all_datasets)} successful simulations")
    print(f"[OK] Saved to: {output_file}")

    return combined_df


def generate_trajectory_data(
    case_file: str,
    output_dir: str = "data",
    H_range: Tuple[float, float, int] = (2.0, 10.0, 5),
    D_range: Tuple[float, float, int] = (0.5, 3.0, 5),
    Pm_range: Optional[Tuple[float, float, int]] = None,
    alpha_range: Optional[Tuple[float, float, int]] = None,  # NEW: Unified alpha multiplier
    load_q_alpha_range: Optional[
        Tuple[float, float, int]
    ] = None,  # Optional: independent Q scaling
    # DEPRECATED: load_range, use_load_variation - use alpha_range instead
    load_range: Optional[Tuple[float, float, int]] = None,
    load_q_range: Optional[Tuple[float, float, int]] = None,
    use_load_variation: bool = False,
    fault_clearing_times: Optional[List[float]] = None,
    fault_locations: Optional[List[int]] = None,
    simulation_time: float = 5.0,
    time_step: float = 0.001,
    sampling_strategy: str = "full_factorial",
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    use_cct_based_sampling: bool = False,
    Pm: Optional[float] = None,
    n_samples_per_combination: int = 5,
    cct_offsets: Optional[List[float]] = None,
    fault_start_time: float = 1.0,
    fault_bus: int = 3,
    fault_reactance: float = 0.0001,
    use_pe_as_input: bool = False,
    base_load: Optional[Dict[str, float]] = None,  # Optional: {"P": 0.5, "Q": 0.2}
) -> pd.DataFrame:
    """
    Generate data for trajectory prediction task.

    This function is optimized for forward problem (predicting trajectories
    from parameters). Uses full factorial or LHS sampling for good coverage.

    Parameters:
    -----------
    case_file : str
        Path to ANDES case file
    output_dir : str
        Output directory
    H_range : tuple
        (min, max, num_points) for inertia constant H
    D_range : tuple
        (min, max, num_points) for damping coefficient D
    Pm_range : tuple, optional
        (min, max, num_points) for mechanical power Pm (pu).
        Mutually exclusive with alpha_range/load_range (use alpha_range for load variation).
    alpha_range : tuple, optional
        (min, max, num_points) for uniform load multiplier alpha.
        When provided, all loads are scaled uniformly: P' = alpha × P_base, Q' = alpha × Q_base.
        Works for both SMIB and multimachine systems.
        Typical range: (0.4, 1.2, 10) for 40% to 120% of base load.
        NOTE: 3D sampling = (H, D, alpha) only. Q is scaled by same alpha (maintains power factor).
    load_q_alpha_range : tuple, optional
        (min, max, num_points) for independent reactive power scaling multiplier.
        If None, Q is scaled by same alpha as P (uniform scaling, maintains power factor - industrial standard).
        Only use for special studies where independent Q scaling is needed.
    load_range : tuple, optional
        DEPRECATED: Use alpha_range instead. Kept for backward compatibility.
        (min, max, num_points) for active power load (p0) (pu, positive for consumption).
        Requires use_load_variation=True. When used, generator adjusts to meet load.
        NOTE: 3D sampling = (H, D, P) only. Q is handled separately.
        Industrial standard: Vary P only, keep Q=0 for SMIB transient stability studies.
    load_q_range : tuple, optional
        (min, max, num_points) for reactive power load (q0) (pu, positive for consumption).
        If None, reactive load is set to 0.0 (standard for SMIB studies).
        Only used when use_load_variation=True.
        NOTE: Q is NOT included in 3D sampling - it's handled separately (fixed or independent).
    use_load_variation : bool
        If True, use load variation instead of Pm variation.
        When True, load_range must be provided. Generator will adjust to meet load.
        Uses ANDES built-in ss.PQ.alter() method (primary), case file modification as fallback.
        Default: False (use Pm variation).
    fault_clearing_times : list, optional
        List of fault clearing times in ABSOLUTE time (seconds), not durations.
        Must be >= fault_start_time.

        **When use_cct_based_sampling=False (default):**
        - REQUIRED: Used directly for trajectory generation
        - Default: [fault_start_time + 0.15, fault_start_time + 0.18, ...]
        - Example: If fault_start_time=1.0, use [1.15, 1.18, 1.20, 1.22, 1.25]

        **When use_cct_based_sampling=True:**
        - OPTIONAL: Only used as fallback if CCT finding fails for a (H, D) pair
        - Bisection method automatically finds CCT and generates clearing times around it
        - If None, uses default fallback: [fault_start_time + 0.15, ...]
    fault_locations : list, optional
        List of fault bus indices
    simulation_time : float
        Total simulation time
    time_step : float
        Time step for simulation
    sampling_strategy : str
        'full_factorial' or 'latin_hypercube'
    n_samples : int, optional
        Number of samples for LHS (if not full factorial)
    seed : int, optional
        Random seed
    verbose : bool
        Print progress information
    use_cct_based_sampling : bool
        If True, finds CCT for each (H, D, Pm) combination using bisection,
        then generates trajectories at multiple clearing times around the CCT.
        This ensures both stable and unstable cases for each operating point,
        focusing on the critical boundary region. Default: False (backward compatible)
    Pm : float, optional
        Mechanical power (pu). If None, will attempt to extract from case file.
        Default: None (extract from system)
    n_samples_per_combination : int
        Number of clearing times to generate around CCT (used only if use_cct_based_sampling=True).
        Default: 5
    cct_offsets : list, optional
        Offsets from CCT to use for generating clearing times (in seconds).
        Example: [-0.05, -0.02, 0.0, 0.02, 0.05] for 5 samples.
        If None, auto-generates offsets based on n_samples_per_combination.
        Default: None
    fault_start_time : float
        Fault start time (seconds). Used for CCT finding. Default: 1.0
    fault_bus : int
        Bus where fault occurs. Used for CCT finding. Default: 3
    fault_reactance : float
        Fault reactance (pu). Used for CCT finding. Default: 0.0001

    Returns:
    --------
    pd.DataFrame
        Generated dataset for trajectory prediction
    """
    # Set default fault_clearing_times if not provided and not using CCT-based sampling
    if fault_clearing_times is None and not use_cct_based_sampling:
        fault_clearing_times = [0.15, 0.18, 0.20, 0.22, 0.25]

    return generate_parameter_sweep(
        case_file=case_file,
        output_dir=output_dir,
        H_range=H_range,
        D_range=D_range,
        Pm_range=Pm_range,
        alpha_range=alpha_range,  # NEW: Unified approach
        load_q_alpha_range=load_q_alpha_range,  # Optional: independent Q scaling
        # Backward compatibility:
        load_range=load_range,
        # Note: load_q_range is not passed - use load_q_alpha_range instead
        use_load_variation=use_load_variation,
        fault_clearing_times=fault_clearing_times,
        fault_locations=fault_locations,
        simulation_time=simulation_time,
        time_step=time_step,
        sampling_strategy=sampling_strategy,
        task="trajectory",
        n_samples=n_samples,
        seed=seed,
        verbose=verbose,
        validate_quality=False,  # Not critical for trajectory prediction
        use_cct_based_sampling=use_cct_based_sampling,
        Pm=Pm,
        n_samples_per_combination=n_samples_per_combination,
        cct_offsets=cct_offsets,
        fault_start_time=fault_start_time,
        fault_bus=fault_bus,
        fault_reactance=fault_reactance,
        use_pe_as_input=use_pe_as_input,
        base_load=base_load,  # Pass base_load from config
    )


def generate_parameter_estimation_data(
    case_file: str,
    output_dir: str = "data",
    H_range: Tuple[float, float, int] = (2.0, 10.0, 5),
    D_range: Tuple[float, float, int] = (0.5, 3.0, 5),
    fault_clearing_times: List[float] = [0.15, 0.18, 0.20, 0.22, 0.25],
    fault_locations: Optional[List[int]] = None,
    simulation_time: float = 5.0,
    time_step: float = 0.001,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    target_correlation: float = 0.0,
    correlation_tolerance: float = 0.1,
) -> pd.DataFrame:
    """
    Generate data for parameter estimation task.

    This function ensures low correlation between H and D to help the model
    learn distinct effects of each parameter. Uses decorrelated sampling.

    Parameters:
    -----------
    case_file : str
        Path to ANDES case file
    output_dir : str
        Output directory
    H_range : tuple
        (min, max, num_points) for inertia constant H
    D_range : tuple
        (min, max, num_points) for damping coefficient D
    fault_clearing_times : list
        List of fault clearing times
    fault_locations : list, optional
        List of fault bus indices
    simulation_time : float
        Total simulation time
    time_step : float
        Time step for simulation
    n_samples : int, optional
        Number of (H, D) pairs to generate
    seed : int, optional
        Random seed
    verbose : bool
        Print progress information
    target_correlation : float
        Target H-D correlation (typically 0.0)
    correlation_tolerance : float
        Acceptable deviation from target correlation

    Returns:
    --------
    pd.DataFrame
        Generated dataset for parameter estimation
    """
    if n_samples is None:
        n_samples = H_range[2] * D_range[2]

    return generate_parameter_sweep(
        case_file=case_file,
        output_dir=output_dir,
        H_range=H_range,
        D_range=D_range,
        fault_clearing_times=fault_clearing_times,
        fault_locations=fault_locations,
        simulation_time=simulation_time,
        time_step=time_step,
        sampling_strategy="latin_hypercube",  # Will use decorrelated sampling
        task="parameter_estimation",
        n_samples=n_samples,
        seed=seed,
        verbose=verbose,
        validate_quality=True,  # Critical for parameter estimation
    )


def generate_cct_data(
    case_file: str,
    output_dir: str = "data",
    H_range: Tuple[float, float, int] = (2.0, 10.0, 5),
    D_range: Tuple[float, float, int] = (0.5, 3.0, 5),
    fault_clearing_times: Optional[List[float]] = None,
    fault_locations: Optional[List[int]] = None,
    simulation_time: float = 5.0,
    time_step: float = 0.001,
    n_samples: Optional[int] = None,
    boundary_fraction: float = 0.4,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate data for CCT estimation task.

    This function focuses on boundary cases near stability limits and
    includes both stable and unstable cases for binary classification.

    Parameters:
    -----------
    case_file : str
        Path to ANDES case file
    output_dir : str
        Output directory
    H_range : tuple
        (min, max, num_points) for inertia constant H
    D_range : tuple
        (min, max, num_points) for damping coefficient D
    fault_clearing_times : list, optional
        List of fault clearing times. If None, will generate dense sampling.
    fault_locations : list, optional
        List of fault bus indices
    simulation_time : float
        Total simulation time
    time_step : float
        Time step for simulation
    n_samples : int, optional
        Number of (H, D) pairs to generate
    boundary_fraction : float
        Fraction of samples near boundaries (0.0 to 1.0)
    seed : int, optional
        Random seed
    verbose : bool
        Print progress information

    Returns:
    --------
    pd.DataFrame
        Generated dataset for CCT estimation with stability labels
    """
    if n_samples is None:
        n_samples = H_range[2] * D_range[2]

    # For CCT, generate dense fault clearing time sampling if not provided
    if fault_clearing_times is None:
        # Generate dense sampling around expected CCT range
        fault_clearing_times = np.linspace(0.10, 0.30, 20).tolist()

    return generate_parameter_sweep(
        case_file=case_file,
        output_dir=output_dir,
        H_range=H_range,
        D_range=D_range,
        fault_clearing_times=fault_clearing_times,
        fault_locations=fault_locations,
        simulation_time=simulation_time,
        time_step=time_step,
        sampling_strategy="boundary_focused",
        task="cct",
        n_samples=n_samples,
        seed=seed,
        verbose=verbose,
        validate_quality=False,
    )


def generate_multi_task_data(
    case_file: str,
    output_dir: str = "data",
    tasks: List[str] = ["trajectory", "parameter_estimation"],
    H_range: Tuple[float, float, int] = (2.0, 10.0, 5),
    D_range: Tuple[float, float, int] = (0.5, 3.0, 5),
    fault_clearing_times: List[float] = [0.15, 0.18, 0.20, 0.22, 0.25],
    fault_locations: Optional[List[int]] = None,
    simulation_time: float = 5.0,
    time_step: float = 0.001,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Generate unified data for multiple tasks simultaneously.

    This function generates datasets for trajectory prediction and parameter estimation.
    The same trajectory data can be used for both forward (trajectory prediction) and
    inverse (parameter estimation) problems.

    Note:
        CCT estimation is now performed using binary search with the trajectory model,
        so no separate CCT data generation is needed. The 'cct' task is deprecated
        but kept for backward compatibility.

    Parameters:
    -----------
    case_file : str
        Path to ANDES case file
    output_dir : str
        Output directory
    tasks : list
        List of tasks: 'trajectory', 'parameter_estimation'
        Note: 'cct' is deprecated - CCT is now estimated via binary search
    H_range : tuple
        (min, max, num_points) for inertia constant H
    D_range : tuple
        (min, max, num_points) for damping coefficient D
    fault_clearing_times : list
        List of fault clearing times
    fault_locations : list, optional
        List of fault bus indices
    simulation_time : float
        Total simulation time
    time_step : float
        Time step for simulation
    n_samples : int, optional
        Number of samples per task
    seed : int, optional
        Random seed
    verbose : bool
        Print progress information

    Returns:
    --------
    dict
        Dictionary mapping task names to DataFrames
    """
    results = {}
    task_output_dirs = {}

    # Create separate output directories for each task
    base_output_dir = Path(output_dir)
    for task in tasks:
        task_output_dir = base_output_dir / task
        task_output_dirs[task] = str(task_output_dir)

    if verbose:
        print(f"Generating data for tasks: {tasks}")

    # Generate data for each task
    for task in tasks:
        if verbose:
            print("\n" + "=" * 60)
            print(f"Generating data for task: {task}")
            print("=" * 60)

        if task == "trajectory":
            df = generate_trajectory_data(
                case_file=case_file,
                output_dir=task_output_dirs[task],
                H_range=H_range,
                D_range=D_range,
                fault_clearing_times=fault_clearing_times,
                fault_locations=fault_locations,
                simulation_time=simulation_time,
                time_step=time_step,
                n_samples=n_samples,
                seed=seed,
                verbose=verbose,
            )
        elif task == "parameter_estimation":
            df = generate_parameter_estimation_data(
                case_file=case_file,
                output_dir=task_output_dirs[task],
                H_range=H_range,
                D_range=D_range,
                fault_clearing_times=fault_clearing_times,
                fault_locations=fault_locations,
                simulation_time=simulation_time,
                time_step=time_step,
                n_samples=n_samples,
                seed=seed,
                verbose=verbose,
            )
        elif task == "cct":
            import warnings

            warnings.warn(
                "CCT data generation is deprecated. CCT is now estimated using "
                "binary search with the trajectory model. See utils.cct_binary_search.",
                DeprecationWarning,
                stacklevel=2,
            )
            df = generate_cct_data(
                case_file=case_file,
                output_dir=task_output_dirs[task],
                H_range=H_range,
                D_range=D_range,
                fault_clearing_times=fault_clearing_times,
                fault_locations=fault_locations,
                simulation_time=simulation_time,
                time_step=time_step,
                n_samples=n_samples,
                seed=seed,
                verbose=verbose,
            )
        else:
            raise ValueError(
                f"Unknown task: {task}. Supported tasks: 'trajectory', 'parameter_estimation'"
            )

        results[task] = df

    if verbose:
        print("\n" + "=" * 60)
        print("Multi-task data generation completed!")
        print("=" * 60)
        for task, df in results.items():
            print(f"  {task}: {len(df)} data points")

    return results


def generate_parameter_sweep_multimachine(
    case_file: str,
    output_dir: str = "data",
    num_machines: Optional[int] = None,
    H_ranges: Optional[List[Tuple[float, float, int]]] = None,
    D_ranges: Optional[List[Tuple[float, float, int]]] = None,
    Pm_ranges: Optional[List[Tuple[float, float, int]]] = None,
    fault_clearing_times: Optional[List[float]] = None,
    fault_locations: Optional[List[int]] = None,
    simulation_time: float = 5.0,
    time_step: float = 0.001,
    verbose: bool = True,
    sampling_strategy: str = "full_factorial",
    task: str = "trajectory",
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    validate_quality: bool = True,
    use_cct_based_sampling: bool = False,
    n_samples_per_combination: int = 5,
    cct_offsets: Optional[List[float]] = None,
    fault_start_time: float = 1.0,
    fault_bus: int = 3,
    fault_reactance: float = 0.0001,
    use_pe_as_input: bool = True,
    use_redispatch: bool = False,  # NEW: Enable participation-factor-based redispatch
    redispatch_config: Optional[Dict] = None,  # NEW: Redispatch configuration
    addfile: Optional[str] = None,  # Optional DYR for PSS/E raw; e.g. "kundur/kundur_gencls.dyr"
    alpha_range: Optional[
        Tuple[float, float, int]
    ] = None,  # Load variation: scale all loads by alpha (Pm from PF)
    base_load: Optional[
        Dict[str, float]
    ] = None,  # Optional base load {"Pload": 0.5, "Qload": 0.2} for alpha scaling
    case_default_pm: Optional[
        List[float]
    ] = None,  # Pm from uploaded case (per-gen); used when tm0 stays 0 after PF (e.g. raw+dyr)
    skip_fault: bool = False,  # If True, run TDS without fault (validate base case; add fault later)
) -> pd.DataFrame:
    """
    Generate data for multi-machine systems with Pe_i(t) as input.

    This function extends generate_parameter_sweep to handle multi-machine systems
    by extracting Pe_i(t) for each generator.

    Parameters:
    -----------
    case_file : str
        Path to ANDES case file (e.g., "kundur/kundur.json")
    output_dir : str
        Directory to save generated datasets
    num_machines : int, optional
        Number of machines. If None, detected from case file.
    H_ranges : list of tuples, optional
        List of (min, max, num_points) for each machine's inertia constant H.
        If None, uses same range for all machines.
    D_ranges : list of tuples, optional
        List of (min, max, num_points) for each machine's damping coefficient D.
        If None, uses same range for all machines.
    Pm_ranges : list of tuples, optional
        List of (min, max, num_points) for each machine's mechanical power Pm.
        If None, uses same range for all machines.
    fault_clearing_times : list, optional
        List of fault clearing times
    fault_locations : list, optional
        List of bus indices for fault locations
    simulation_time : float
        Total simulation time (seconds)
    time_step : float
        Time step for simulation (seconds)
    verbose : bool
        Print progress information
    sampling_strategy : str
        Sampling strategy: 'full_factorial', 'latin_hypercube', 'sobol', 'boundary_focused'
    task : str
        Task type: 'trajectory', 'parameter_estimation', 'cct'
    n_samples : int, optional
        Number of samples (for non-full-factorial strategies)
    seed : int, optional
        Random seed for reproducibility
    validate_quality : bool
        Whether to validate data quality
    use_cct_based_sampling : bool
        Whether to use CCT-based sampling
    n_samples_per_combination : int
        Number of samples per parameter combination
    cct_offsets : list, optional
        Offsets from CCT for sampling
    fault_start_time : float
        Fault start time (seconds)
    fault_bus : int
        Default fault bus index
    fault_reactance : float
        Fault reactance (pu)
    use_pe_as_input : bool
        If True, extract Pe_i(t) for each machine as input.
        If False, use reactance-based approach.
    use_redispatch : bool
        If True, use participation-factor-based redispatch instead of simple uniform scaling.
        Default: False (backward compatible)
    redispatch_config : dict, optional
        Redispatch configuration dictionary with:
        - max_redispatch_iterations: int (default: 5)
        - redispatch_tolerance: float (default: 0.01 pu)
        - loss_estimation_factor: float (default: 0.03)
        - participation_factor_method: str (default: "inertia_weighted")

    Returns:
    --------
    pd.DataFrame : Combined dataset with per-machine Pe_i(t) trajectories
    """
    if not ANDES_AVAILABLE:
        raise ImportError("ANDES is not available. Cannot generate multi-machine data.")

    # Load system to detect number of machines and require GENCLS (pipeline is GENCLS-based)
    try:
        case_path_mm = andes.get_case(case_file) if not os.path.isabs(case_file) else case_file
        addfile_path_mm = None
        if addfile:
            addfile_path_mm = andes.get_case(addfile) if not os.path.isabs(addfile) else addfile
        ss = (
            andes.load(case_path_mm, addfile=addfile_path_mm)
            if addfile_path_mm
            else andes.load(case_path_mm)
        )
        if num_machines is None:
            if hasattr(ss, "GENCLS") and ss.GENCLS.n > 0:
                num_machines = ss.GENCLS.n
            else:
                raise ValueError("No GENCLS generators found in case file.")
        else:
            # Validate that the case actually has GENCLS (e.g. config may set num_machines=4 but case has GENROU)
            if not hasattr(ss, "GENCLS") or ss.GENCLS.n == 0:
                genrou_n = getattr(getattr(ss, "GENROU", None), "n", 0) or 0
                raise ValueError(
                    "This pipeline requires a case file with GENCLS (classical) generators. "
                    f"The loaded case has no GENCLS (GENROU count: {genrou_n}). "
                    "The ANDES stock case 'kundur/kundur_full.xlsx' uses GENROU. "
                    "Use a GENCLS-based Kundur case or another GENCLS multimachine case. "
                    "See docs/multimachine_case_studies/KUNDUR_EXPERIMENT_PLAN.md."
                )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load case file {case_file}: {e}") from e

    # Multimachine: focus variables are H, D, alpha. Pm is NOT a sweep variable; when Pload (alpha) changes, Pm follows from power flow.
    if alpha_range is not None:
        Pm_ranges = None  # Pm from PF after load scaling
        if verbose:
            print("[INFO] Multimachine focus: H, D, alpha. Pm from power flow (load).")
    elif addfile:
        # When addfile is set (e.g. Kundur) but alpha_range was not passed, use single load level so we still use load mode.
        alpha_range = (1.0, 1.0, 1)
        Pm_ranges = None
        if verbose:
            print(
                "[INFO] Multimachine: alpha_range defaulted to (1.0, 1.0, 1). Focus: H, D, alpha; Pm from PF."
            )

    if verbose:
        print(f"Multi-machine system detected: {num_machines} generators (GENCLS)")
        print(f"Using Pe(t) as input: {use_pe_as_input}")
        if use_redispatch:
            print("[INFO] REDISPATCH MODE: Participation-factor-based redispatch enabled")
            print("   NOTE: Redispatch integration into data generation loop is in progress.")
            print("   Current implementation uses uniform load scaling.")
            print(
                "   For full redispatch support, modify generate_parameter_sweep() simulation loop."
            )

    # Use same ranges for all machines if not specified
    if H_ranges is None:
        H_ranges = [(2.0, 10.0, 5)] * num_machines
    if D_ranges is None:
        D_ranges = [(0.5, 3.0, 5)] * num_machines
    if Pm_ranges is None and alpha_range is None:
        Pm_ranges = [(0.4, 0.9, 6)] * num_machines

    # Ensure we have ranges for all machines (when not using alpha_range)
    while len(H_ranges) < num_machines:
        H_ranges.append(H_ranges[-1] if H_ranges else (2.0, 10.0, 5))
    while len(D_ranges) < num_machines:
        D_ranges.append(D_ranges[-1] if D_ranges else (0.5, 3.0, 5))
    if Pm_ranges is not None:
        while len(Pm_ranges) < num_machines:
            Pm_ranges.append(Pm_ranges[-1] if Pm_ranges else (0.4, 0.9, 6))

    # Validate that per-machine ranges are provided and warn if they're being ignored
    ranges_differ = False
    if len(H_ranges) > 1 and Pm_ranges is not None:
        # Check if ranges actually differ
        first_H = H_ranges[0]
        first_D = D_ranges[0]
        first_Pm = Pm_ranges[0]
        for i in range(1, num_machines):
            if H_ranges[i] != first_H or D_ranges[i] != first_D or Pm_ranges[i] != first_Pm:
                ranges_differ = True
                break

    if ranges_differ and sampling_strategy != "full_factorial":
        if verbose:
            print("=" * 70)
            print("WARNING: Per-machine parameter ranges differ, but non-factorial")
            print("         sampling strategy only uses first machine's ranges.")
            print("         Machines 1-{} will use ranges from machine 0:".format(num_machines - 1))
            print(
                "         H_range: {}, D_range: {}, Pm_range: {}".format(
                    H_ranges[0], D_ranges[0], Pm_ranges[0]
                )
            )
            print("=" * 70)
        # Use first machine's ranges for non-factorial strategies
        H_range = H_ranges[0]
        D_range = D_ranges[0]
        Pm_range = Pm_ranges[0] if Pm_ranges is not None else None

        df_base = generate_parameter_sweep(
            case_file=case_file,
            output_dir=output_dir,
            H_range=H_range,
            D_range=D_range,
            Pm_range=Pm_range,
            alpha_range=alpha_range,
            base_load=base_load,
            case_default_pm=case_default_pm,
            fault_clearing_times=fault_clearing_times,
            fault_locations=fault_locations,
            simulation_time=simulation_time,
            time_step=time_step,
            verbose=verbose,
            sampling_strategy=sampling_strategy,
            task=task,
            n_samples=n_samples,
            seed=seed,
            validate_quality=validate_quality,
            use_cct_based_sampling=use_cct_based_sampling,
            n_samples_per_combination=n_samples_per_combination,
            cct_offsets=cct_offsets,
            fault_start_time=fault_start_time,
            fault_bus=fault_bus,
            fault_reactance=fault_reactance,
            use_pe_as_input=use_pe_as_input,
            addfile=addfile,
            skip_fault=skip_fault,
        )

        # If use_pe_as_input and per-machine Pe_0..Pe_{n-1} not already in CSV (from sweep)
        if use_pe_as_input and num_machines > 1:
            if f"Pe_0" not in df_base.columns and "Pe" in df_base.columns:
                if verbose:
                    print(
                        "  Per-machine Pe_0..Pe_{} not in data; copying single Pe.".format(
                            num_machines - 1
                        )
                    )
                for i in range(num_machines):
                    df_base[f"Pe_{i}"] = df_base["Pe"].values
            elif f"Pe_0" in df_base.columns and verbose:
                print(
                    "  [OK] Per-machine Pe_0..Pe_{} already in data (from sweep).".format(
                        num_machines - 1
                    )
                )
            elif "Pe" not in df_base.columns:
                if verbose:
                    print(
                        "  [WARNING] No 'Pe' column found. Creating NaN placeholder Pe_0..Pe_{}.".format(
                            num_machines - 1
                        )
                    )
                for i in range(num_machines):
                    df_base[f"Pe_{i}"] = np.nan

        return df_base

    # For full factorial or when ranges are the same, use first machine's ranges
    # NOTE: Full per-machine parameter sweep requires custom simulation loop that
    # modifies all machines' parameters. Current implementation uses first machine's
    # ranges for all machines. This is a known limitation.
    if ranges_differ and verbose:
        print("=" * 70)
        print("NOTE: Per-machine parameter ranges differ, but current implementation")
        print("      uses first machine's ranges for all machines.")
        print("      Full per-machine parameter sweep requires custom simulation loop.")
        print("      Using ranges from machine 0 for all machines:")
        print(
            "      H_range: {}, D_range: {}, Pm_range: {}".format(
                H_ranges[0], D_ranges[0], Pm_ranges[0]
            )
        )
        print("=" * 70)

    H_range = H_ranges[0]
    D_range = D_ranges[0]
    Pm_range = Pm_ranges[0] if Pm_ranges is not None else None

    df_base = generate_parameter_sweep(
        case_file=case_file,
        output_dir=output_dir,
        H_range=H_range,
        D_range=D_range,
        Pm_range=Pm_range,
        alpha_range=alpha_range,
        base_load=base_load,
        case_default_pm=case_default_pm,
        fault_clearing_times=fault_clearing_times,
        fault_locations=fault_locations,
        simulation_time=simulation_time,
        time_step=time_step,
        verbose=verbose,
        sampling_strategy=sampling_strategy,
        task=task,
        n_samples=n_samples,
        seed=seed,
        validate_quality=validate_quality,
        use_cct_based_sampling=use_cct_based_sampling,
        n_samples_per_combination=n_samples_per_combination,
        cct_offsets=cct_offsets,
        fault_start_time=fault_start_time,
        fault_bus=fault_bus,
        fault_reactance=fault_reactance,
        use_pe_as_input=use_pe_as_input,
        addfile=addfile,
        skip_fault=skip_fault,
    )

    # If use_pe_as_input and per-machine Pe_0..Pe_{n-1} not already in CSV (from sweep)
    if use_pe_as_input and num_machines > 1:
        if f"Pe_0" not in df_base.columns and "Pe" in df_base.columns:
            if verbose:
                print(
                    "  Per-machine Pe_0..Pe_{} not in data; copying single Pe.".format(
                        num_machines - 1
                    )
                )
            for i in range(num_machines):
                df_base[f"Pe_{i}"] = df_base["Pe"].values
        elif f"Pe_0" in df_base.columns and verbose:
            print(
                "  [OK] Per-machine Pe_0..Pe_{} already in data (from sweep).".format(
                    num_machines - 1
                )
            )
        elif "Pe" not in df_base.columns:
            if verbose:
                print(
                    "  [WARNING] No 'Pe' column found. Creating NaN placeholder Pe_0..Pe_{}.".format(
                        num_machines - 1
                    )
                )
            for i in range(num_machines):
                df_base[f"Pe_{i}"] = np.nan

    return df_base
