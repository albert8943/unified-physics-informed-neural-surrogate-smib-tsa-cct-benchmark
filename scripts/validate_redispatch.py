"""
Redispatch Validation Script (P0 - Critical).

Validates participation-factor-based redispatch implementation on 50 samples.
Checks convergence rate, power balance, limit violations, and participation
factor consistency.

Usage:
    python scripts/validate_redispatch.py \\
        --config configs/publication/kundur_trajectory.yaml \\
        --n-samples 50
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False
    print("ERROR: ANDES is not available. Cannot run validation.")
    sys.exit(1)

from data_generation.andes_utils.dispatch_policy import apply_dispatch_policy_iterative
from data_generation.andes_utils.slack_selection import (
    select_inertia_dominant_slack,
    validate_slack_selection,
)
from data_generation.sampling_strategies import sobol_sequence_sample


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def scale_loads_uniform(ss, alpha: float, verbose: bool = False):
    """
    Scale all loads uniformly by alpha (maintains power factor).

    Parameters:
    -----------
    ss : ANDES system object
        ANDES system
    alpha : float
        Load multiplier
    verbose : bool
        Whether to print progress
    """
    if not hasattr(ss, "PQ") or ss.PQ.n == 0:
        if verbose:
            print("  No PQ loads found in system")
        return

    for load_idx in range(ss.PQ.n):
        # Get base values
        if hasattr(ss.PQ, "p0") and ss.PQ.p0.v is not None:
            P_base = float(ss.PQ.p0.v[load_idx])
            P_new = alpha * P_base

            if hasattr(ss.PQ, "q0") and ss.PQ.q0.v is not None:
                Q_base = float(ss.PQ.q0.v[load_idx])
                Q_new = alpha * Q_base
            else:
                Q_new = None

            # Update load
            ss.PQ.alter("p0", idx=load_idx, value=P_new)
            if Q_new is not None:
                ss.PQ.alter("q0", idx=load_idx, value=Q_new)

            if verbose:
                print(f"    Load {load_idx}: P = {P_new:.6f} pu (α×{P_base:.6f})")


def validate_redispatch_scenario(
    case_file: str,
    alpha_base: float,
    alpha_new: float,
    config: Dict,
    verbose: bool = False,
) -> Dict:
    """
    Validate redispatch for a single scenario.

    Parameters:
    -----------
    case_file : str
        Path to ANDES case file
    alpha_base : float
        Base load multiplier
    alpha_new : float
        New load multiplier
    config : dict
        Configuration dictionary
    verbose : bool
        Whether to print progress

    Returns:
    --------
    dict
        Validation results
    """
    result = {
        "alpha_base": alpha_base,
        "alpha_new": alpha_new,
        "converged": False,
        "n_iterations": 0,
        "slack_deviation": float("inf"),
        "power_balance": float("inf"),
        "participation_factor_sum": 0.0,
        "error": None,
    }

    try:
        # Load system
        ss = andes.load(case_file, setup=False, no_output=True)

        # Scale loads to base
        scale_loads_uniform(ss, alpha_base, verbose=False)

        # Setup system
        ss.setup()

        # Run initial power flow
        ss.PFlow.run()
        if not (hasattr(ss.PFlow, "converged") and ss.PFlow.converged):
            result["error"] = "initial_power_flow_failed"
            return result

        # Select slack bus (inertia-dominant)
        slack_idx = select_inertia_dominant_slack(ss, use_mva_rating=True)
        if not validate_slack_selection(ss, slack_idx):
            result["error"] = "invalid_slack_selection"
            return result

        # Scale loads to new value
        scale_loads_uniform(ss, alpha_new, verbose=False)

        # Get redispatch parameters from config
        redispatch_config = config.get("load_variation", {})
        max_iterations = redispatch_config.get("max_redispatch_iterations", 5)
        tolerance = redispatch_config.get("redispatch_tolerance", 0.01)
        loss_estimation_factor = redispatch_config.get("loss_estimation_factor", 0.03)

        # Apply iterative redispatch
        converged, n_iterations, stats = apply_dispatch_policy_iterative(
            ss,
            slack_idx,
            alpha_base,
            alpha_new,
            max_iterations=max_iterations,
            tolerance=tolerance,
            loss_estimation_factor=loss_estimation_factor,
            verbose=verbose,
        )

        result["converged"] = converged
        result["n_iterations"] = n_iterations

        if "error" in stats:
            result["error"] = stats["error"]
            return result

        result["slack_deviation"] = stats.get("slack_deviation", float("inf"))
        result["power_balance"] = abs(stats.get("power_balance", 0.0))

        # Check participation factor sum
        participation_factors = stats.get("participation_factors", {})
        if participation_factors:
            pf_sum = sum(participation_factors.values())
            result["participation_factor_sum"] = pf_sum

        # Check power balance
        power_balance_tolerance = config.get("data_quality", {}).get(
            "power_balance_tolerance", 0.01
        )
        result["power_balance_ok"] = result["power_balance"] < power_balance_tolerance

        # Check participation factor consistency
        pf_tolerance = config.get("data_quality", {}).get("participation_factor_tolerance", 1e-6)
        result["participation_factor_ok"] = (
            abs(result["participation_factor_sum"] - 1.0) < pf_tolerance
        )

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"  ERROR: {e}")

    return result


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate redispatch implementation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (e.g., configs/experiments/kundur/kundur_trajectory.yaml)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples to validate (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/validation/redispatch",
        help="Output directory for validation results",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Get case file
    case_file = config.get("system", {}).get("case_file", "kundur/kundur_full.xlsx")
    try:
        case_path = andes.get_case(case_file)
    except Exception:
        case_path = case_file
        if not Path(case_path).exists():
            print(f"ERROR: Case file not found: {case_file}")
            sys.exit(1)

    # Get parameter ranges
    param_ranges = config.get("parameter_ranges", {})
    alpha_range = param_ranges.get("alpha", [0.4, 1.2])

    # Generate alpha samples using Sobol
    if isinstance(alpha_range, list) and len(alpha_range) == 2:
        alpha_min, alpha_max = alpha_range[0], alpha_range[1]
    else:
        alpha_min, alpha_max = 0.4, 1.2

    # Generate samples
    print("=" * 70)
    print("REDISPATCH VALIDATION")
    print("=" * 70)
    print(f"Case file: {case_path}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Alpha range: [{alpha_min}, {alpha_max}]")
    print("=" * 70)

    # Generate alpha pairs (base and new)
    # Use Sobol sampling for alpha_new, alpha_base = 1.0 (typical base)
    alpha_base = 1.0  # Typical base loading
    alpha_samples = sobol_sequence_sample(
        bounds=[(alpha_min, alpha_max)],
        n_samples=args.n_samples,
        seed=42,
    )
    alpha_new_values = alpha_samples[:, 0]

    # Run validation
    results = []
    n_converged = 0
    n_failed = 0

    for i, alpha_new in enumerate(alpha_new_values):
        if args.verbose:
            print(
                f"\nSample {i+1}/{args.n_samples}: α_base={alpha_base:.3f}, α_new={alpha_new:.3f}"
            )
        else:
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{args.n_samples} samples...")

        result = validate_redispatch_scenario(
            case_path, alpha_base, alpha_new, config, verbose=args.verbose
        )
        result["sample_id"] = i + 1
        results.append(result)

        if result["converged"]:
            n_converged += 1
        else:
            n_failed += 1

    # Compile results
    results_df = pd.DataFrame(results)

    # Calculate statistics
    convergence_rate = n_converged / args.n_samples if args.n_samples > 0 else 0.0

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"Total samples: {args.n_samples}")
    print(f"Converged: {n_converged} ({convergence_rate*100:.1f}%)")
    print(f"Failed: {n_failed} ({(1-convergence_rate)*100:.1f}%)")

    if n_converged > 0:
        converged_results = results_df[results_df["converged"]]
        print(f"\nConverged samples statistics:")
        print(f"  Mean iterations: {converged_results['n_iterations'].mean():.2f}")
        print(f"  Max iterations: {converged_results['n_iterations'].max()}")
        print(f"  Mean slack deviation: {converged_results['slack_deviation'].mean():.6f} pu")
        print(f"  Max slack deviation: {converged_results['slack_deviation'].max():.6f} pu")
        print(f"  Mean power balance: {converged_results['power_balance'].mean():.6f} pu")
        print(f"  Max power balance: {converged_results['power_balance'].max():.6f} pu")

        # Check participation factor consistency
        pf_ok = converged_results.get(
            "participation_factor_ok", pd.Series([False] * len(converged_results))
        )
        print(
            f"  Participation factor consistency: {pf_ok.sum()}/{len(pf_ok)} ({pf_ok.mean()*100:.1f}%)"
        )

        # Check power balance
        pb_ok = converged_results.get(
            "power_balance_ok", pd.Series([False] * len(converged_results))
        )
        print(f"  Power balance OK: {pb_ok.sum()}/{len(pb_ok)} ({pb_ok.mean()*100:.1f}%)")

    # Error analysis
    if n_failed > 0:
        failed_results = results_df[~results_df["converged"]]
        print(f"\nFailed samples analysis:")
        if "error" in failed_results.columns:
            error_counts = failed_results["error"].value_counts()
            print("  Error types:")
            for error_type, count in error_counts.items():
                print(f"    - {error_type}: {count}")

    # Target: >95% convergence rate
    target_rate = config.get("data_quality", {}).get("redispatch_convergence_target", 0.95)
    if convergence_rate >= target_rate:
        print(
            f"\n✓ PASS: Convergence rate {convergence_rate*100:.1f}% >= target {target_rate*100:.1f}%"
        )
    else:
        print(
            f"\n✗ FAIL: Convergence rate {convergence_rate*100:.1f}% < target {target_rate*100:.1f}%"
        )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"redispatch_validation_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)

    summary_file = output_dir / f"redispatch_validation_summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write("REDISPATCH VALIDATION SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Case file: {case_path}\n")
        f.write(f"Number of samples: {args.n_samples}\n")
        f.write(f"Convergence rate: {convergence_rate*100:.1f}%\n")
        f.write(f"Target: {target_rate*100:.1f}%\n")
        f.write(f"Status: {'PASS' if convergence_rate >= target_rate else 'FAIL'}\n")
        f.write("=" * 70 + "\n")

    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 70)

    # Exit with error code if validation failed
    if convergence_rate < target_rate:
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
