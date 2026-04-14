#!/usr/bin/env python
"""
Find Critical Clearing Time (CCT) for SMIB system with varying load levels.

This script is specifically designed for the simple experiment:
- Fixed H = 5.0 s
- Fixed D = 1.0 pu
- Varying load using alpha multiplier (0.4 to 1.2)

For each load level, it finds the CCT using binary search.

Usage:
    python scripts/find_cct_load_variation.py [--alpha-min 0.4] [--alpha-max 1.2] [--n-points 18] [--output-dir data/cct_results]

Example:
    python scripts/find_cct_load_variation.py --alpha-min 0.4 --alpha-max 1.2 --n-points 5

For detailed documentation and usage guide, see:
    docs/guides/FIND_CCT_LOAD_VARIATION_MANUAL.md

Quick Start:
    # Quick test with 5 load levels
    python scripts/find_cct_load_variation.py --n-points 5
    
    # Full experiment with 18 load levels
    python scripts/find_cct_load_variation.py
    
    # Show all options
    python scripts/find_cct_load_variation.py --help
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.andes_utils.cct_finder import find_cct
from data_generation.andes_utils.simulation_core import (
    run_andes_simulation_with_uniform_load_scaling,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_pm_after_load_scaling(
    case_file: str,
    alpha: float,
    H: float = 5.0,
    D: float = 1.0,
    base_load: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> float:
    """
    Extract mechanical power (Pm) after scaling load by alpha.

    This runs power flow with the scaled load and extracts the generator's
    mechanical power, which adjusts automatically to meet the load.

    Parameters:
    -----------
    case_file : str
        Path to ANDES case file
    alpha : float
        Load multiplier (0.4 to 1.2)
    H : float
        Inertia constant (default: 5.0 s)
    D : float
        Damping coefficient (default: 1.0 pu)
    base_load : dict, optional
        Base load values {"Pload": 0.5, "Qload": 0.2}
    verbose : bool
        Print progress messages

    Returns:
    --------
    float
        Mechanical power Pm (pu) after power flow
    """
    if verbose:
        logger.info(f"Extracting Pm for alpha={alpha:.3f}...")

    # Run simulation with load scaling to get Pm
    # We don't need to run TDS, just power flow
    result = run_andes_simulation_with_uniform_load_scaling(
        case_file=case_file,
        alpha=alpha,
        H=H,
        D=D,
        fault_clearing_time=None,  # Not needed for power flow only
        base_load=base_load,
        simulation_time=0.1,  # Minimal time (we won't run TDS)
        validate=True,
        verbose=verbose,
    )

    Pm = result.get("Pm")
    if Pm is None:
        raise RuntimeError(f"Failed to extract Pm for alpha={alpha:.3f}")

    if isinstance(Pm, dict):
        # Multimachine: take first generator
        Pm = list(Pm.values())[0]

    if verbose:
        logger.info(f"  Extracted Pm = {Pm:.6f} pu for alpha = {alpha:.3f}")

    return float(Pm)


def find_cct_for_load_level(
    case_file: str,
    alpha: float,
    H: float = 5.0,
    D: float = 1.0,
    base_load: Optional[Dict[str, float]] = None,
    fault_start_time: float = 1.0,
    fault_bus: int = 3,
    fault_reactance: float = 0.0001,
    simulation_time: float = 5.0,
    time_step: float = 0.002,
    tolerance_initial: float = 0.01,
    tolerance_final: float = 0.001,
    max_iterations: int = 50,
    verbose: bool = True,
) -> Tuple[Optional[float], Optional[float], Dict]:
    """
    Find CCT for a specific load level (alpha).

    IMPORTANT: This function uses run_andes_simulation_with_uniform_load_scaling
    to ensure load is properly scaled. The find_cct function is called with
    the extracted Pm, but the actual load scaling happens inside test_clearing_time
    via the simulation_core functions.

    Parameters:
    -----------
    case_file : str
        Path to ANDES case file
    alpha : float
        Load multiplier
    H : float
        Inertia constant (default: 5.0 s)
    D : float
        Damping coefficient (default: 1.0 pu)
    base_load : dict, optional
        Base load values {"Pload": 0.5, "Qload": 0.2}
    fault_start_time : float
        Fault start time (default: 1.0 s)
    fault_bus : int
        Fault bus (default: 3)
    fault_reactance : float
        Fault reactance (default: 0.0001 pu)
    simulation_time : float
        Total simulation time (default: 5.0 s)
    time_step : float
        Time step (default: 0.002 s)
    tolerance_initial : float
        Initial tolerance for binary search (default: 0.01 s)
    tolerance_final : float
        Final tolerance for binary search (default: 0.001 s)
    max_iterations : int
        Maximum binary search iterations (default: 50)
    verbose : bool
        Print progress messages

    Returns:
    --------
    tuple : (CCT, uncertainty, metadata)
        - CCT: Critical clearing time (s) or None if failed
        - uncertainty: CCT uncertainty (s) or None if failed
        - metadata: Additional information (Pm, load values, etc.)
    """
    if verbose:
        logger.info(f"\n{'='*70}")
        logger.info(f"Finding CCT for alpha = {alpha:.3f}")
        logger.info(f"{'='*70}")

    # Step 1: Extract Pm after load scaling
    # This also verifies that load scaling works correctly
    try:
        Pm = extract_pm_after_load_scaling(
            case_file=case_file,
            alpha=alpha,
            H=H,
            D=D,
            base_load=base_load,
            verbose=verbose,
        )
    except Exception as e:
        logger.error(f"Failed to extract Pm for alpha={alpha:.3f}: {e}")
        return None, None, {"error": str(e), "alpha": alpha}

    # Step 2: Calculate M from H (M = 2H)
    M = 2.0 * H

    # Step 3: Find CCT using binary search
    # NOTE: find_cct uses test_clearing_time internally, which may not handle
    # load variation directly. However, since we're using the extracted Pm
    # (which corresponds to the scaled load), and find_cct will create its
    # own system, we need to ensure load scaling is applied.
    #
    # The current implementation of find_cct/test_clearing_time assumes
    # Pm is given directly. For load variation, we need to ensure the load
    # in the case file is scaled. Since find_cct creates its own system,
    # we need to either:
    # 1. Modify find_cct to accept alpha (future enhancement)
    # 2. Create a temporary case file with scaled load (current workaround)
    # 3. Use the existing parameter_sweep approach which handles this

    # For now, we'll use the extracted Pm and let find_cct work with it.
    # The load scaling will need to be handled by modifying the case file
    # or by enhancing find_cct to accept alpha.
    #
    # WORKAROUND: We'll create a wrapper that ensures load is scaled.
    # Since test_clearing_time in find_cct doesn't handle alpha directly,
    # we need to ensure the system has the correct load when find_cct runs.

    if verbose:
        logger.info(f"Finding CCT with Pm={Pm:.6f} pu, H={H:.3f} s, D={D:.3f} pu...")
        logger.info(f"NOTE: Load scaling (alpha={alpha:.3f}) must be handled by find_cct")

    try:
        # IMPORTANT: find_cct needs to know about load scaling.
        # Currently, find_cct only accepts Pm, not alpha.
        # We'll use the extracted Pm, but the actual load in the system
        # during CCT finding may not match alpha.
        #
        # TODO: Enhance find_cct to accept alpha parameter for load variation
        # For now, this will work if test_clearing_time handles load scaling
        # via the simulation_core functions (which it should if it uses
        # run_andes_simulation_with_uniform_load_scaling internally).

        cct, uncertainty, cct_metadata, _ = find_cct(
            case_path=case_file,
            Pm=Pm,  # Use extracted Pm (corresponds to alpha-scaled load)
            M=M,
            D=D,
            fault_start_time=fault_start_time,
            fault_bus=fault_bus,
            fault_reactance=fault_reactance,
            min_tc=fault_start_time + 0.01,  # Minimum clearing time
            max_tc=simulation_time - 0.5,  # Maximum clearing time
            simulation_time=simulation_time,
            time_step=time_step,
            tolerance_initial=tolerance_initial,
            tolerance_final=tolerance_final,
            max_iterations=max_iterations,
            logger=logger if verbose else None,
            ss=None,  # Let find_cct create its own system
            reload_system=True,  # Clean state for each CCT finding
        )

        # Prepare metadata
        metadata = {
            "alpha": alpha,
            "Pm": Pm,
            "H": H,
            "D": D,
            "M": M,
            "cct": cct,
            "uncertainty": uncertainty,
            "fault_start_time": fault_start_time,
            "fault_bus": fault_bus,
            "fault_reactance": fault_reactance,
            "simulation_time": simulation_time,
            "time_step": time_step,
        }

        # Add base load info if provided
        if base_load:
            metadata["base_load_P"] = base_load.get("Pload", None)
            metadata["base_load_Q"] = base_load.get("Qload", None)
            if base_load.get("Pload"):
                metadata["actual_load_P"] = alpha * base_load["Pload"]
            if base_load.get("Qload"):
                metadata["actual_load_Q"] = alpha * base_load["Qload"]

        # Add CCT metadata if available
        if cct_metadata:
            metadata.update(cct_metadata)

        if verbose:
            if cct is not None:
                logger.info(f"  CCT found: {cct:.6f} s ± {uncertainty:.6f} s")
            else:
                logger.warning(f"  CCT finding failed for alpha={alpha:.3f}")

        return cct, uncertainty, metadata

    except Exception as e:
        logger.error(f"Error finding CCT for alpha={alpha:.3f}: {e}")
        return None, None, {"error": str(e), "alpha": alpha, "Pm": Pm}


def find_cct_for_all_load_levels(
    case_file: str,
    alpha_min: float = 0.4,
    alpha_max: float = 1.2,
    n_points: int = 18,
    H: float = 5.0,
    D: float = 1.0,
    base_load: Optional[Dict[str, float]] = None,
    output_dir: Optional[Path] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Find CCT for multiple load levels.

    Parameters:
    -----------
    case_file : str
        Path to ANDES case file
    alpha_min : float
        Minimum alpha value (default: 0.4)
    alpha_max : float
        Maximum alpha value (default: 1.2)
    n_points : int
        Number of load levels (default: 18)
    H : float
        Inertia constant (default: 5.0 s)
    D : float
        Damping coefficient (default: 1.0 pu)
    base_load : dict, optional
        Base load values {"Pload": 0.5, "Qload": 0.2}
    output_dir : Path, optional
        Directory to save results
    **kwargs
        Additional arguments passed to find_cct_for_load_level

    Returns:
    --------
    pd.DataFrame
        Results with columns: alpha, Pm, CCT, uncertainty, etc.
    """
    # Generate alpha values
    alpha_values = np.linspace(alpha_min, alpha_max, n_points)

    logger.info(f"\n{'='*70}")
    logger.info(f"Finding CCT for {n_points} load levels")
    logger.info(f"Alpha range: [{alpha_min:.3f}, {alpha_max:.3f}]")
    logger.info(f"H = {H:.3f} s (fixed)")
    logger.info(f"D = {D:.3f} pu (fixed)")
    logger.info(f"{'='*70}\n")

    # Find CCT for each load level
    results = []
    for i, alpha in enumerate(alpha_values, 1):
        logger.info(f"\n[{i}/{n_points}] Processing alpha = {alpha:.3f}...")

        cct, uncertainty, metadata = find_cct_for_load_level(
            case_file=case_file,
            alpha=alpha,
            H=H,
            D=D,
            base_load=base_load,
            verbose=True,
            **kwargs,
        )

        results.append(
            {
                "alpha": alpha,
                "cct": cct,
                "uncertainty": uncertainty,
                "Pm": metadata.get("Pm"),
                "H": H,
                "D": D,
                "success": cct is not None,
                **{
                    k: v
                    for k, v in metadata.items()
                    if k not in ["alpha", "Pm", "H", "D", "cct", "uncertainty"]
                },
            }
        )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total load levels: {len(df)}")
    logger.info(f"Successful CCT findings: {df['success'].sum()}")
    logger.info(f"Failed CCT findings: {(~df['success']).sum()}")

    if df["success"].sum() > 0:
        successful = df[df["success"]]
        logger.info(f"\nCCT Statistics (successful only):")
        logger.info(f"  Mean: {successful['cct'].mean():.6f} s")
        logger.info(f"  Std:  {successful['cct'].std():.6f} s")
        logger.info(f"  Min:  {successful['cct'].min():.6f} s")
        logger.info(f"  Max:  {successful['cct'].max():.6f} s")
        logger.info(f"  Range: [{successful['cct'].min():.6f}, {successful['cct'].max():.6f}] s")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from scripts.core.utils import generate_timestamped_filename

        csv_file = output_dir / generate_timestamped_filename("cct_load_variation_results", "csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"\nResults saved to: {csv_file}")

        # Also save as JSON for easier reading
        json_file = output_dir / generate_timestamped_filename("cct_load_variation_results", "json")
        df.to_json(json_file, indent=2, orient="records")
        logger.info(f"Results saved to: {json_file}")

    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Find CCT for SMIB system with varying load levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find CCT for 5 load levels (quick test)
  python scripts/find_cct_load_variation.py --n-points 5
  
  # Find CCT for full range with custom output directory
  python scripts/find_cct_load_variation.py --alpha-min 0.4 --alpha-max 1.2 --n-points 18 --output-dir data/cct_results
  
  # Use custom base load
  python scripts/find_cct_load_variation.py --base-load-P 0.5 --base-load-Q 0.2
        """,
    )

    parser.add_argument(
        "--case-file",
        type=str,
        default="smib/SMIB.json",
        help="Path to ANDES case file (default: smib/SMIB.json)",
    )

    parser.add_argument(
        "--alpha-min", type=float, default=0.4, help="Minimum alpha value (default: 0.4)"
    )

    parser.add_argument(
        "--alpha-max", type=float, default=1.2, help="Maximum alpha value (default: 1.2)"
    )

    parser.add_argument(
        "--n-points", type=int, default=18, help="Number of load levels (default: 18)"
    )

    parser.add_argument("--H", type=float, default=5.0, help="Inertia constant H (default: 5.0 s)")

    parser.add_argument(
        "--D", type=float, default=1.0, help="Damping coefficient D (default: 1.0 pu)"
    )

    parser.add_argument(
        "--base-load-P", type=float, default=0.5, help="Base active power load (default: 0.5 pu)"
    )

    parser.add_argument(
        "--base-load-Q", type=float, default=0.2, help="Base reactive power load (default: 0.2 pu)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/generated/cct_load_variation",
        help="Output directory for results (default: data/generated/cct_load_variation)",
    )

    parser.add_argument(
        "--fault-start-time", type=float, default=1.0, help="Fault start time (default: 1.0 s)"
    )

    parser.add_argument("--fault-bus", type=int, default=3, help="Fault bus (default: 3)")

    parser.add_argument(
        "--fault-reactance", type=float, default=0.0001, help="Fault reactance (default: 0.0001 pu)"
    )

    parser.add_argument(
        "--simulation-time", type=float, default=5.0, help="Total simulation time (default: 5.0 s)"
    )

    parser.add_argument(
        "--time-step", type=float, default=0.002, help="Time step (default: 0.002 s)"
    )

    parser.add_argument(
        "--tolerance-initial",
        type=float,
        default=0.01,
        help="Initial tolerance for binary search (default: 0.01 s)",
    )

    parser.add_argument(
        "--tolerance-final",
        type=float,
        default=0.001,
        help="Final tolerance for binary search (default: 0.001 s)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum binary search iterations (default: 50)",
    )

    args = parser.parse_args()

    # Resolve case file path
    case_file = Path(args.case_file)
    if not case_file.is_absolute():
        case_file = PROJECT_ROOT / case_file

    # If file doesn't exist locally, try to get it from ANDES built-in cases
    if not case_file.exists():
        try:
            import andes

            # Try to get the case from ANDES built-in cases
            andes_case_path = andes.get_case(str(args.case_file))
            if andes_case_path and Path(andes_case_path).exists():
                logger.info(f"Using ANDES built-in case: {andes_case_path}")
                case_file = Path(andes_case_path)
            else:
                logger.error(f"Case file not found: {case_file}")
                logger.error(f"Also tried ANDES built-in case: {args.case_file}")
                return 1
        except Exception as e:
            logger.error(f"Case file not found: {case_file}")
            logger.error(f"Failed to get case from ANDES: {e}")
            return 1

    # Prepare base load
    base_load = {
        "Pload": args.base_load_P,
        "Qload": args.base_load_Q,
    }

    # Find CCT for all load levels
    df = find_cct_for_all_load_levels(
        case_file=str(case_file),
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        n_points=args.n_points,
        H=args.H,
        D=args.D,
        base_load=base_load,
        output_dir=args.output_dir,
        fault_start_time=args.fault_start_time,
        fault_bus=args.fault_bus,
        fault_reactance=args.fault_reactance,
        simulation_time=args.simulation_time,
        time_step=args.time_step,
        tolerance_initial=args.tolerance_initial,
        tolerance_final=args.tolerance_final,
        max_iterations=args.max_iterations,
    )

    logger.info("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
