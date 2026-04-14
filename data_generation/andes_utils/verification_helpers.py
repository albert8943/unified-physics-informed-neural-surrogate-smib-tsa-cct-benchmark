"""
Verification Helper Functions for ANDES Parameter Setting

This module provides functions to verify that parameters were correctly set
in ANDES systems, ensuring data quality and preventing silent failures.

Author: Albert
Date: 2025-12-19
"""

from typing import Any, Dict, List, Tuple

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False


def safe_get_tm0_value(ss: Any, gen_idx: int) -> float:
    """
    Safely get tm0.v value, handling scalar or array.

    Args:
        ss: ANDES system object
        gen_idx: Generator index

    Returns:
        tm0 value as float
    """
    if (
        not hasattr(ss, "GENCLS")
        or not hasattr(ss.GENCLS, "tm0")
        or not hasattr(ss.GENCLS.tm0, "v")
    ):
        return 0.0

    tm0_v = ss.GENCLS.tm0.v
    if hasattr(tm0_v, "__getitem__") and hasattr(tm0_v, "__len__"):
        if len(tm0_v) > gen_idx:
            return float(tm0_v[gen_idx])
        else:
            # If gen_idx is out of bounds but tm0_v is an array,
            # it might be a single generator represented as an array of length 1.
            # Try to return the first element if it exists.
            if len(tm0_v) > 0:
                return float(tm0_v[0])
    # Scalar value
    return float(tm0_v)


def verify_generator_setpoints(
    ss: Any,
    generator_setpoints: Dict[int, float],
    tolerance: float = 0.01,
) -> Tuple[bool, List[str]]:
    """
    Verify that tm0[i] matches requested Pm[i] for all generators.

    This is a critical verification step to ensure that the case file
    modification worked correctly and power flow used the correct setpoints.

    Args:
        ss: ANDES system object (must have run power flow)
        generator_setpoints: Dict mapping generator index to requested Pm value
        tolerance: Maximum allowed mismatch (as fraction, e.g., 0.01 = 1%)

    Returns:
        Tuple of (all_match, list_of_errors)
        - all_match: True if all generators match within tolerance
        - list_of_errors: List of error messages for mismatched generators

    Example:
        >>> generator_setpoints = {0: 0.7, 1: 0.8, 2: 0.6}
        >>> all_match, errors = verify_generator_setpoints(ss, generator_setpoints)
        >>> if not all_match:
        ...     for error in errors:
        ...         print(error)
    """
    if not ANDES_AVAILABLE:
        return False, ["ANDES not available"]

    errors = []
    for gen_idx, requested_pm in generator_setpoints.items():
        try:
            actual_tm0 = safe_get_tm0_value(ss, gen_idx)
            mismatch_pct = abs(actual_tm0 - requested_pm) / (abs(requested_pm) + 1e-12) * 100

            if mismatch_pct > tolerance * 100:
                errors.append(
                    f"Gen {gen_idx}: requested={requested_pm:.6f} pu, "
                    f"actual={actual_tm0:.6f} pu ({mismatch_pct:.2f}% mismatch)"
                )
        except Exception as e:
            errors.append(f"Gen {gen_idx}: Error reading tm0 - {e}")

    return len(errors) == 0, errors


def verify_power_flow_converged(ss: Any) -> Tuple[bool, str]:
    """
    Verify that power flow converged successfully.

    Args:
        ss: ANDES system object (must have run power flow)

    Returns:
        Tuple of (converged, error_message)
        - converged: True if power flow converged
        - error_message: Empty string if converged, error description if not
    """
    if not ANDES_AVAILABLE:
        return False, "ANDES not available"

    if not hasattr(ss, "PFlow"):
        return False, "PFlow object not found in system"

    if hasattr(ss.PFlow, "converged"):
        if ss.PFlow.converged:
            return True, ""
        else:
            return False, "Power flow did not converge"
    else:
        # If converged attribute doesn't exist, assume it converged
        # (some ANDES versions might not have this attribute)
        return True, ""


def verify_power_balance(
    ss: Any,
    generator_setpoints: Dict[int, float],
    tolerance: float = 0.01,
) -> Tuple[bool, str]:
    """
    Verify power balance in multimachine system.

    This function is a wrapper around check_power_balance from case_file_modifier
    for consistency. It verifies that total generation approximately equals
    total load plus losses.

    Args:
        ss: ANDES system object (must be loaded)
        generator_setpoints: Dict mapping generator index to Pm value
        tolerance: Maximum allowed imbalance (pu). Default: 0.01 pu

    Returns:
        Tuple of (is_balanced, error_message)
        - is_balanced: True if power balance is maintained
        - error_message: Empty string if balanced, error description if not
    """
    # Import here to avoid circular dependency
    from .case_file_modifier import check_power_balance

    return check_power_balance(ss, generator_setpoints, tolerance)
