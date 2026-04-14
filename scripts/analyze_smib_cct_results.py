#!/usr/bin/env python
"""
Analyze SMIB CCT experiment results (Load Variation or P_m Variation).

Usage:
    python scripts/analyze_smib_cct_results.py [data_file_path] [--variation-mode load|pm] [--plot] [--output-dir OUTPUT_DIR]
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def detect_variation_mode(df: pd.DataFrame) -> str:
    """Detect variation mode from data columns."""
    if "param_Pm" in df.columns or "pm_values" in df.columns:
        return "pm"
    elif "param_alpha" in df.columns or "alpha_values" in df.columns:
        return "load"
    else:
        # Try to infer from scenario data
        if "scenario_id" in df.columns:
            scenarios = df.groupby("scenario_id").first()
            if "param_Pm" in scenarios.columns:
                return "pm"
            elif "param_alpha" in scenarios.columns:
                return "load"
    return "unknown"


def analyze_results(
    data_path: Path, variation_mode: str = None, plot: bool = False, output_dir: Path = None
):
    """Analyze experiment results."""
    # Detect variation mode if not provided
    df_temp = pd.read_csv(data_path, nrows=100)  # Read sample to detect mode
    detected_mode = detect_variation_mode(df_temp)

    if variation_mode is None:
        variation_mode = detected_mode

    mode_name = "P_m Variation" if variation_mode == "pm" else "Load Variation"
    print("=" * 70)
    print(f"SMIB CCT {mode_name} - Results Analysis")
    print("=" * 70)
    print()

    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[OK] Loaded {len(df):,} data points")
    print()

    # Basic statistics
    print("=" * 70)
    print("1. BASIC STATISTICS")
    print("=" * 70)
    print(f"Total data points: {len(df):,}")

    if "scenario_id" in df.columns:
        n_scenarios = df["scenario_id"].nunique()
        print(f"Unique scenarios: {n_scenarios}")
    else:
        print("[WARNING] No scenario_id column found")
        n_scenarios = 0

    # Parameter coverage based on variation mode
    if variation_mode == "pm":
        pm_col = None
        for col in ["param_Pm", "pm_values", "Pm"]:
            if col in df.columns:
                pm_col = col
                break

        if pm_col:
            n_pm_levels = df[pm_col].nunique()
            pm_values = sorted(df[pm_col].unique())
            print(f"Unique P_m levels: {n_pm_levels}")
            print(f"P_m range: [{pm_values[0]:.6f}, {pm_values[-1]:.6f}] pu")
            print(
                f"P_m values: {[f'{p:.6f}' for p in pm_values[:10]]}{'...' if len(pm_values) > 10 else ''}"
            )

            # Check for normalized P_m
            if "pm_normalized" in df.columns or "param_Pm_normalized" in df.columns:
                norm_col = (
                    "pm_normalized" if "pm_normalized" in df.columns else "param_Pm_normalized"
                )
                pm_norm = sorted(df[norm_col].unique())
                print(f"P_m normalized (fraction of P_max): [{pm_norm[0]:.3f}, {pm_norm[-1]:.3f}]")
        else:
            print("[WARNING] No P_m column found (param_Pm, pm_values, or Pm)")
    else:
        if "param_alpha" in df.columns:
            n_load_levels = df["param_alpha"].nunique()
            alphas = sorted(df["param_alpha"].unique())
            print(f"Unique load levels (alpha): {n_load_levels}")
            print(f"Alpha range: [{alphas[0]:.3f}, {alphas[-1]:.3f}]")
            print(
                f"Alpha values: {[f'{a:.3f}' for a in alphas[:10]]}{'...' if len(alphas) > 10 else ''}"
            )
        else:
            print("[WARNING] No param_alpha column found")

    print()

    # Stability analysis
    print("=" * 70)
    print("2. STABILITY ANALYSIS")
    print("=" * 70)

    if "is_stable" in df.columns:
        stable_count = df["is_stable"].sum()
        total = len(df)
        unstable_count = total - stable_count

        print(f"Stable trajectories: {stable_count:,} ({100*stable_count/total:.1f}%)")
        print(f"Unstable trajectories: {unstable_count:,} ({100*unstable_count/total:.1f}%)")
        print(f"Expected: ~54% stable, ~46% unstable")

        if stable_count == 0:
            print("\n[CRITICAL] All trajectories are unstable!")
            print("   This indicates a problem with clearing time generation.")
        elif stable_count / total < 0.3:
            print("\n[WARNING] Too few stable trajectories (< 30%)")
            print("   Expected 40-60% stable for CCT-based sampling")
    else:
        print("[WARNING] No is_stable column found")

    # Per-scenario stability
    if "scenario_id" in df.columns and "is_stable" in df.columns:
        scenario_stability = df.groupby("scenario_id")["is_stable"].first()
        stable_scenarios = scenario_stability.sum()
        print(
            f"\nStable scenarios: {stable_scenarios}/{n_scenarios} ({100*stable_scenarios/n_scenarios:.1f}%)"
        )
        print(
            f"Unstable scenarios: {n_scenarios-stable_scenarios}/{n_scenarios} ({100*(n_scenarios-stable_scenarios)/n_scenarios:.1f}%)"
        )

    print()

    # CCT and clearing time analysis
    print("=" * 70)
    print("3. CCT AND CLEARING TIME ANALYSIS")
    print("=" * 70)

    if "scenario_id" in df.columns:
        scenarios = df.groupby("scenario_id").first()

        # Find CCT column
        cct_col = None
        for col in ["param_cct_absolute", "cct", "param_cct"]:
            if col in scenarios.columns:
                cct_col = col
                break

        if cct_col:
            cct_values = scenarios[cct_col].dropna()
            print(f"CCT statistics (from {cct_col}):")
            print(f"  Count: {len(cct_values)}")
            print(f"  Range: [{cct_values.min():.3f}s, {cct_values.max():.3f}s]")
            print(f"  Mean: {cct_values.mean():.3f}s")
            print(f"  Std: {cct_values.std():.3f}s")
            print(f"  Median: {cct_values.median():.3f}s")

            if "clearing_time" in scenarios.columns:
                scenarios["cct_diff"] = scenarios["clearing_time"] - scenarios[cct_col]
                diff_values = scenarios["cct_diff"].dropna()

                print(f"\nClearing time - CCT (offset) statistics:")
                print(f"  Min offset: {diff_values.min():.6f}s")
                print(f"  Max offset: {diff_values.max():.6f}s")
                print(f"  Mean offset: {diff_values.mean():.6f}s")
                print(f"  Median offset: {diff_values.median():.6f}s")

                stable_offsets = (diff_values < 0).sum()
                unstable_offsets = (diff_values >= 0).sum()

                print(f"\nOffset distribution:")
                print(
                    f"  Negative offsets (should be stable): {stable_offsets} ({100*stable_offsets/len(diff_values):.1f}%)"
                )
                print(
                    f"  Zero/Positive offsets (unstable): {unstable_offsets} ({100*unstable_offsets/len(diff_values):.1f}%)"
                )

                if stable_offsets == 0:
                    print("\n[CRITICAL] All clearing times are >= CCT!")
                    print("   Expected: Some clearing times should be < CCT (stable cases)")
                    print("   Possible causes:")
                    print("     1. CCT finding is finding values that are too low")
                    print("     2. Offsets are not being applied correctly")
                    print(
                        "     3. CCT is at the boundary where even negative offsets result in instability"
                    )
                elif stable_offsets < len(diff_values) * 0.3:
                    print("\n[WARNING] Too few stable offsets (< 30%)")
                    print("   Expected ~40-60% stable offsets for balanced dataset")
        else:
            print("[WARNING] No CCT column found (param_cct_absolute, cct, or param_cct)")

        if "clearing_time" in scenarios.columns:
            print(f"\nClearing time statistics:")
            print(
                f"  Range: [{scenarios['clearing_time'].min():.3f}s, {scenarios['clearing_time'].max():.3f}s]"
            )
            print(f"  Mean: {scenarios['clearing_time'].mean():.3f}s")
            print(f"  Unique values: {scenarios['clearing_time'].nunique()}")

    print()

    # Trajectory quality
    print("=" * 70)
    print("4. TRAJECTORY QUALITY ANALYSIS")
    print("=" * 70)

    if "scenario_id" in df.columns:
        if "delta_deg" in df.columns:
            max_angles = df.groupby("scenario_id")["delta_deg"].max()
            print(f"Rotor angle (delta) statistics:")
            print(f"  Min max angle: {max_angles.min():.1f}°")
            print(f"  Max max angle: {max_angles.max():.1f}°")
            print(f"  Mean max angle: {max_angles.mean():.1f}°")
            print(f"  Threshold: 180° (loss of synchronism)")

            above_threshold = (max_angles > 180).sum()
            print(
                f"  Scenarios above 180°: {above_threshold}/{len(max_angles)} ({100*above_threshold/len(max_angles):.1f}%)"
            )

            if above_threshold == len(max_angles):
                print("\n[CRITICAL] All scenarios exceed 180 degrees (all unstable)")
            elif above_threshold > len(max_angles) * 0.5:
                print(
                    f"\n⚠️  WARNING: {100*above_threshold/len(max_angles):.1f}% scenarios exceed 180°"
                )

        if "omega_deviation" in df.columns:
            max_omega = df.groupby("scenario_id")["omega_deviation"].max()
            print(f"\nFrequency deviation (omega) statistics:")
            print(f"  Min max deviation: {max_omega.min():.2f} Hz")
            print(f"  Max max deviation: {max_omega.max():.2f} Hz")
            print(f"  Mean max deviation: {max_omega.mean():.2f} Hz")
            print(f"  Threshold: 0.5 Hz (typical limit)")

            above_threshold = (max_omega > 0.5).sum()
            print(
                f"  Scenarios above 0.5 Hz: {above_threshold}/{len(max_omega)} ({100*above_threshold/len(max_omega):.1f}%)"
            )

            if above_threshold == len(max_omega):
                print("\n[CRITICAL] All scenarios exceed 0.5 Hz (all unstable)")

    print()

    # Parameter coverage
    print("=" * 70)
    print("5. PARAMETER COVERAGE")
    print("=" * 70)

    if "scenario_id" in df.columns:
        scenarios = df.groupby("scenario_id").first()

        if "param_H" in scenarios.columns:
            h_values = scenarios["param_H"].unique()
            print(f"H (inertia) values: {sorted(h_values)}")
            print(f"  Expected: [5.0] (fixed)")

        if "param_D" in scenarios.columns:
            d_values = scenarios["param_D"].unique()
            print(f"D (damping) values: {sorted(d_values)}")
            print(f"  Expected: [1.0] (fixed)")

        if variation_mode == "pm":
            pm_col = None
            for col in ["param_Pm", "pm_values", "Pm"]:
                if col in scenarios.columns:
                    pm_col = col
                    break

            if pm_col:
                pm_values = sorted(scenarios[pm_col].unique())
                print(f"P_m (mechanical power) values: {len(pm_values)} unique")
                print(f"  Range: [{pm_values[0]:.6f}, {pm_values[-1]:.6f}] pu")

                # Check for normalized values
                if (
                    "pm_normalized" in scenarios.columns
                    or "param_Pm_normalized" in scenarios.columns
                ):
                    norm_col = (
                        "pm_normalized"
                        if "pm_normalized" in scenarios.columns
                        else "param_Pm_normalized"
                    )
                    pm_norm = sorted(scenarios[norm_col].unique())
                    print(
                        f"  Normalized (fraction of P_max): [{pm_norm[0]:.3f}, {pm_norm[-1]:.3f}]"
                    )
        else:
            if "param_alpha" in scenarios.columns:
                alpha_values = sorted(scenarios["param_alpha"].unique())
                print(f"Alpha (load multiplier) values: {len(alpha_values)} unique")
                print(f"  Range: [{alpha_values[0]:.3f}, {alpha_values[-1]:.3f}]")
                print(f"  Expected: 5 values from [0.4, 1.2] (for test)")
                if len(alpha_values) < 5:
                    print(f"  [WARNING] Only {len(alpha_values)} load levels (expected 5 for test)")

    print()

    # Summary and recommendations
    print("=" * 70)
    print("6. SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)

    issues = []
    if "is_stable" in df.columns:
        stable_ratio = df["is_stable"].sum() / len(df)
        if stable_ratio == 0:
            issues.append("ALL trajectories are unstable (0% stable)")
        elif stable_ratio < 0.3:
            issues.append(
                f"Too few stable trajectories ({100*stable_ratio:.1f}% stable, expected 40-60%)"
            )

    if (
        "scenario_id" in df.columns
        and "param_cct_absolute" in df.groupby("scenario_id").first().columns
    ):
        scenarios = df.groupby("scenario_id").first()
        if "clearing_time" in scenarios.columns:
            cct_diff = scenarios["clearing_time"] - scenarios["param_cct_absolute"]
            if (cct_diff < 0).sum() == 0:
                issues.append("All clearing times are >= CCT (no stable cases)")

    if issues:
        print("\n[CRITICAL] ISSUES DETECTED:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

        print("\n[RECOMMENDATIONS]:")
        print("   1. Check CCT finding algorithm - CCT values may be too low")
        print("   2. Verify clearing time offsets are applied correctly")
        print("   3. Check if offsets [-0.004, -0.002, 0.0, 0.002, 0.004] are appropriate")
        print("   4. Consider using wider offsets (e.g., [-0.01, -0.005, 0.0, 0.005, 0.01])")
        print("   5. Verify CCT finding tolerance settings (tolerance_final=0.001s)")
        print("   6. Check if system is more stable than CCT finding suggests")
    else:
        print("\n[OK] No critical issues detected")
        print("   Data quality appears acceptable")

    print()
    print("=" * 70)

    # Generate plots if requested
    if plot and MATPLOTLIB_AVAILABLE:
        if variation_mode == "pm":
            plot_pm_variation_results(df, output_dir)
        else:
            print("[INFO] Plotting for load variation mode not yet implemented in this script")
            print("       Use examples/smib_batch_tds.py --plot for load variation plots")


def plot_pm_variation_results(df: pd.DataFrame, output_dir: Path = None):
    """Plot CCT vs P_m results (publication-ready)."""
    if not MATPLOTLIB_AVAILABLE:
        print("[WARNING] Matplotlib not available. Skipping plots.")
        return

    if "scenario_id" not in df.columns:
        print("[WARNING] No scenario_id column found. Cannot generate plots.")
        return

    scenarios = df.groupby("scenario_id").first()

    # Find P_m column
    pm_col = None
    for col in ["param_Pm", "pm_values", "Pm"]:
        if col in scenarios.columns:
            pm_col = col
            break

    if pm_col is None:
        print("[WARNING] No P_m column found. Cannot generate CCT vs P_m plot.")
        return

    # Find CCT column
    cct_col = None
    for col in ["param_cct_absolute", "cct", "param_cct"]:
        if col in scenarios.columns:
            cct_col = col
            break

    if cct_col is None:
        print("[WARNING] No CCT column found. Cannot generate CCT vs P_m plot.")
        return

    # Filter valid data
    valid_mask = scenarios[cct_col].notna() & scenarios[pm_col].notna()
    pm_data = scenarios.loc[valid_mask, pm_col].values
    cct_data = scenarios.loc[valid_mask, cct_col].values

    if len(pm_data) == 0:
        print("[WARNING] No valid CCT-P_m pairs found.")
        return

    # Sort by P_m
    sort_idx = np.argsort(pm_data)
    pm_data = pm_data[sort_idx]
    cct_data = cct_data[sort_idx]

    # Publication-quality figure settings
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 1.5,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "lines.linewidth": 2,
            "lines.markersize": 8,
        }
    )

    # Try to compute EAC baseline if parameters are available
    eac_data = None
    eac_errors = []
    try:
        from data_generation.andes_utils.eac_baseline import compute_cct_eac, compare_cct_methods
        from data_generation.andes_utils.data_extractor import extract_network_reactances

        # Extract network parameters for EAC computation
        # Try to get from scenarios or use defaults
        H_values = (
            scenarios.get("param_H", scenarios.get("H", [5.0])).values
            if "param_H" in scenarios.columns or "H" in scenarios.columns
            else [5.0]
        )
        D_values = (
            scenarios.get("param_D", scenarios.get("D", [1.0])).values
            if "param_D" in scenarios.columns or "D" in scenarios.columns
            else [1.0]
        )
        M_values = [2.0 * h for h in H_values] if len(H_values) > 0 else [10.0]

        # Default network reactances (can be extracted from data if available)
        X_prefault = 0.5  # Default, should be extracted from data
        X_fault = 0.0001
        X_postfault = 0.5

        # Compute EAC CCT for each P_m value
        eac_cct_list = []
        for i, pm_val in enumerate(pm_data):
            if i < len(M_values):
                M = M_values[i] if i < len(M_values) else M_values[0]
                D = D_values[i] if i < len(D_values) else D_values[0]
            else:
                M = M_values[0]
                D = D_values[0]

            cct_eac, _, _ = compute_cct_eac(
                Pm=pm_val,
                M=M,
                D=D,
                X_prefault=X_prefault,
                X_fault=X_fault,
                X_postfault=X_postfault,
                V_gen=1.0,
                V_inf=1.0,
            )

            if cct_eac is not None:
                eac_cct_list.append(cct_eac)
                # Compare with bisection
                if i < len(cct_data):
                    comparison = compare_cct_methods(cct_data[i], cct_eac, tolerance=0.01)
                    if comparison.get("both_available", False):
                        eac_errors.append(comparison.get("error", None))
            else:
                eac_cct_list.append(np.nan)

        eac_data = np.array(eac_cct_list)

    except Exception as e:
        print(f"[INFO] EAC baseline computation skipped: {e}")
        eac_data = None

    # Create figure
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    # Plot CCT vs P_m (bisection)
    ax.plot(
        pm_data,
        cct_data,
        "o-",
        color="#2ca02c",
        linewidth=2,
        markersize=6,
        label="Bisection CCT",
        zorder=3,
    )

    # Plot EAC baseline if available
    if eac_data is not None and not np.all(np.isnan(eac_data)):
        valid_eac = ~np.isnan(eac_data)
        if np.any(valid_eac):
            ax.plot(
                pm_data[valid_eac],
                eac_data[valid_eac],
                "s--",
                color="#d62728",
                linewidth=2,
                markersize=5,
                label="EAC Analytical CCT",
                zorder=2,
                alpha=0.7,
            )

            # Add error statistics
            if eac_errors and len(eac_errors) > 0:
                valid_errors = [e for e in eac_errors if e is not None]
                if valid_errors:
                    mean_error = np.mean(valid_errors)
                    max_error = np.max(valid_errors)
                    ax.text(
                        0.05,
                        0.88,
                        f"EAC Error: {mean_error*1000:.1f} ms (max: {max_error*1000:.1f} ms)",
                        transform=ax.transAxes,
                        fontsize=9,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
                    )

    # Compute sensitivity (dCCT/dP_m) using finite differences
    if len(pm_data) > 1:
        sensitivity = np.gradient(cct_data, pm_data)
        # Add sensitivity annotation
        mean_sensitivity = np.mean(sensitivity)
        y_pos = 0.95 if eac_data is None else 0.80
        ax.text(
            0.05,
            y_pos,
            f"dCCT/dP_m ≈ {mean_sensitivity:.3f} s/pu",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Labels and title
    ax.set_xlabel("Mechanical Power P_m (pu)", fontsize=12)
    ax.set_ylabel("Critical Clearing Time CCT (s)", fontsize=12)
    ax.set_title("CCT vs Mechanical Power (Generator Stress Study)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best")

    # Improve spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    # Save plot
    if output_dir is None:
        output_dir = Path("examples/data/pinn_training/analysis")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as PDF (publication-ready)
    pdf_path = output_dir / "cct_vs_pm_analysis.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"[PLOT] Saved CCT vs P_m plot: {pdf_path}")

    # Also save as PNG
    png_path = output_dir / "cct_vs_pm_analysis.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"[PLOT] Saved CCT vs P_m plot: {png_path}")

    plt.close()

    # Sensitivity analysis plot
    if len(pm_data) > 1:
        fig, ax = plt.subplots(figsize=(7.0, 5.5))

        # Plot sensitivity
        pm_mid = (pm_data[:-1] + pm_data[1:]) / 2
        sensitivity = np.gradient(cct_data, pm_data)

        ax.plot(
            pm_mid,
            sensitivity[:-1] if len(sensitivity) > len(pm_mid) else sensitivity,
            "s-",
            color="#d62728",
            linewidth=2,
            markersize=6,
            label="dCCT/dP_m",
        )
        ax.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.3)

        ax.set_xlabel("Mechanical Power P_m (pu)", fontsize=12)
        ax.set_ylabel("Sensitivity dCCT/dP_m (s/pu)", fontsize=12)
        ax.set_title("CCT Sensitivity to Mechanical Power", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best")

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        plt.tight_layout()

        sensitivity_path = output_dir / "cct_sensitivity_pm.pdf"
        plt.savefig(
            sensitivity_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"[PLOT] Saved sensitivity plot: {sensitivity_path}")

        plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze SMIB CCT experiment results (Load Variation or P_m Variation)"
    )
    parser.add_argument(
        "data_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to data file (CSV)",
    )
    parser.add_argument(
        "--variation-mode",
        type=str,
        choices=["load", "pm", "auto"],
        default="auto",
        help="Variation mode: 'load', 'pm', or 'auto' (detect from data)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate publication-ready plots",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: examples/data/pinn_training/analysis)",
    )

    args = parser.parse_args()

    # Determine data path
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        # Find latest trajectory data file in test directory
        test_dir = PROJECT_ROOT / "data" / "generated" / "smib_cct_load_variation_test"
        if test_dir.exists():
            data_files = list(test_dir.glob("trajectory_data_*.csv"))
            if data_files:
                data_path = max(data_files, key=lambda p: p.stat().st_mtime)
            else:
                print("[ERROR] No trajectory data files found in test directory")
                print(f"   Expected in: {test_dir}")
                return 1
        else:
            print("[ERROR] Test directory not found or no data path provided")
            print(f"   Expected: {test_dir}")
            print("   Usage: python scripts/analyze_smib_cct_results.py [data_file_path]")
            return 1

    if not data_path.exists():
        print(f"[ERROR] Data file not found: {data_path}")
        return 1

    variation_mode = None if args.variation_mode == "auto" else args.variation_mode
    output_dir = Path(args.output_dir) if args.output_dir else None

    analyze_results(data_path, variation_mode=variation_mode, plot=args.plot, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
