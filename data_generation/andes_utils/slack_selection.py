"""
Slack Bus Selection Module for Multimachine Systems.

Implements inertia-dominant slack selection philosophy:
slack_idx = argmax_i (H_i × S_base,i)

Where:
- H_i = Inertia constant (H = M/2 for 60 Hz systems)
- S_base,i = Generator MVA rating
"""

from typing import Optional

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False


def get_generator_mva_rating(ss, gen_idx: int, use_mva_rating: bool = True) -> float:
    """
    Extract MVA rating for a generator.

    Fallback logic:
    1. Try StaticGen.Sn (MVA rating) - primary source
    2. Fallback to GENCLS.Sn if available
    3. Fallback to system base (typically 100 MVA) if not specified

    Parameters:
    -----------
    ss : ANDES system object
        ANDES system
    gen_idx : int
        GENCLS generator index
    use_mva_rating : bool
        If False, return 1.0 (use H_i alone without S_base)

    Returns:
    --------
    float
        MVA rating (or 1.0 if use_mva_rating=False)
    """
    if not use_mva_rating:
        return 1.0

    # Try to get MVA rating from StaticGen (primary source)
    # GENCLS generators are linked to StaticGen via 'gen' field
    if hasattr(ss, "GENCLS") and gen_idx < ss.GENCLS.n:
        # Get StaticGen index from GENCLS.gen field
        if hasattr(ss.GENCLS, "gen") and ss.GENCLS.gen.v is not None:
            static_gen_idx = ss.GENCLS.gen.v[gen_idx]

            # Try StaticGen.Sn
            if hasattr(ss, "StaticGen") and static_gen_idx < ss.StaticGen.n:
                if hasattr(ss.StaticGen, "Sn") and ss.StaticGen.Sn.v is not None:
                    return float(ss.StaticGen.Sn.v[static_gen_idx])

        # Fallback: Try GENCLS.Sn if available
        if hasattr(ss.GENCLS, "Sn") and ss.GENCLS.Sn.v is not None:
            return float(ss.GENCLS.Sn.v[gen_idx])

    # Final fallback: Use system base MVA (typically 100 MVA)
    if hasattr(ss, "config") and hasattr(ss.config, "base_mva"):
        return float(ss.config.base_mva)

    # Default: 100 MVA (typical base)
    return 100.0


def get_generator_inertia_contribution(ss, gen_idx: int, use_mva_rating: bool = True) -> float:
    """
    Calculate inertia contribution H_i × S_base,i for a generator.

    Parameters:
    -----------
    ss : ANDES system object
        ANDES system
    gen_idx : int
        GENCLS generator index
    use_mva_rating : bool
        Whether to include MVA rating in calculation

    Returns:
    --------
    float
        Inertia contribution (H_i × S_base,i or just H_i if use_mva_rating=False)
    """
    if not hasattr(ss, "GENCLS") or gen_idx >= ss.GENCLS.n:
        raise ValueError(f"Invalid generator index: {gen_idx} (total generators: {ss.GENCLS.n})")

    # Get M (inertia coefficient) from GENCLS
    M_i = float(ss.GENCLS.M.v[gen_idx])  # M = 2*H for 60 Hz systems
    H_i = M_i / 2.0  # Convert to H

    if use_mva_rating:
        S_base_i = get_generator_mva_rating(ss, gen_idx, use_mva_rating=True)
        return H_i * S_base_i
    else:
        return H_i


def select_inertia_dominant_slack(ss, use_mva_rating: bool = True) -> int:
    """
    Select slack generator based on inertia-dominant criterion.

    Criterion: slack_idx = argmax_i (H_i × S_base,i)

    Parameters:
    -----------
    ss : ANDES system object
        ANDES system (must be loaded but not necessarily set up)
    use_mva_rating : bool
        Whether to include MVA rating in calculation

    Returns:
    --------
    int
        GENCLS index of selected slack generator
    """
    if not hasattr(ss, "GENCLS") or ss.GENCLS.n == 0:
        raise ValueError("No GENCLS generators found in system")

    max_contribution = -1.0
    slack_idx = 0

    for gen_idx in range(ss.GENCLS.n):
        contribution = get_generator_inertia_contribution(ss, gen_idx, use_mva_rating)
        if contribution > max_contribution:
            max_contribution = contribution
            slack_idx = gen_idx

    return slack_idx


def validate_slack_selection(ss, slack_idx: int) -> bool:
    """
    Validate that selected slack generator is valid.

    Parameters:
    -----------
    ss : ANDES system object
        ANDES system
    slack_idx : int
        GENCLS index of slack generator

    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    if not hasattr(ss, "GENCLS"):
        return False

    if slack_idx < 0 or slack_idx >= ss.GENCLS.n:
        return False

    return True


def map_gencls_to_staticgen(ss, gencls_idx: int) -> Optional[int]:
    """
    Map GENCLS index to corresponding StaticGen index.

    ANDES uses StaticGen components for slack buses, not GENCLS directly.
    The mapping is via GENCLS[gencls_idx].gen → StaticGen[static_gen_idx].

    Parameters:
    -----------
    ss : ANDES system object
        ANDES system
    gencls_idx : int
        GENCLS generator index

    Returns:
    --------
    int or None
        StaticGen index if found, None otherwise
    """
    if not hasattr(ss, "GENCLS") or gencls_idx >= ss.GENCLS.n:
        return None

    if not hasattr(ss.GENCLS, "gen") or ss.GENCLS.gen.v is None:
        return None

    static_gen_idx = int(ss.GENCLS.gen.v[gencls_idx])

    if hasattr(ss, "StaticGen") and static_gen_idx < ss.StaticGen.n:
        return static_gen_idx

    return None
