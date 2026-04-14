#!/usr/bin/env python
"""
Comprehensive Dataset Analysis Script for Publication

This script performs comprehensive analysis of generated trajectory and CCT data,
including statistical analysis, visualization, and generation of publication-ready figures.

Usage:
    python scripts/analyze_data.py [data_file.csv] [--output-dir OUTPUT_DIR] [--format FORMAT]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set matplotlib backend before importing pyplot (important for headless environments)
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Import project's timestamping utility
try:
    from scripts.core.utils import generate_timestamped_filename
except ImportError:
    # Fallback if workflow utils not available
    from datetime import datetime

    def generate_timestamped_filename(base_name: str, extension: str, **kwargs) -> str:
        """Fallback timestamped filename generator."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.{extension}"


def _analysis_figure_filename(base_name: str, extension: str, *, stable_paper_names: bool) -> str:
    """Timestamped analysis export name, or fixed ``{base}.{ext}`` for manuscript bundles."""
    if stable_paper_names:
        return f"{base_name}.{extension}"
    return generate_timestamped_filename(base_name, extension)


# Set publication-quality plotting style
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("colorblind")
PUBLICATION_DPI = 300
FIGURE_SIZE_SINGLE = (3.5, 2.5)  # Single column width (inches)
FIGURE_SIZE_DOUBLE = (7.0, 5.0)  # Double column width (inches)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load trajectory data from CSV file."""
    print(f"📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def _operating_point_col(df: pd.DataFrame) -> str:
    """Column for operating point: param_load (multimachine) or param_Pm (SMIB)."""
    if "param_load" in df.columns and df["param_load"].notna().any():
        return "param_load"
    return "param_Pm"


def _operating_point_label(col: str) -> str:
    """Label for operating point column in plots/reports."""
    return "Load" if col == "param_load" else "Pm (pu)"


def _has_param_space(df: pd.DataFrame) -> bool:
    """True if df has param_H, param_D, and an operating-point column (param_Pm or param_load)."""
    op_col = _operating_point_col(df)
    return all(c in df.columns for c in ["param_H", "param_D"]) and op_col in df.columns


def compute_dataset_statistics(df: pd.DataFrame) -> Dict:
    """Compute comprehensive dataset statistics."""
    print("\n📊 Computing dataset statistics...")

    stats_dict = {}

    # Basic counts
    stats_dict["total_rows"] = len(df)
    stats_dict["total_columns"] = len(df.columns)

    op_col = _operating_point_col(df)
    if "scenario_id" in df.columns and _has_param_space(df):
        stats_dict["scenarios_per_combo"] = (
            df.groupby(["param_H", "param_D", op_col])["scenario_id"].nunique().mean()
        )
    elif "scenario_id" in df.columns:
        stats_dict["scenarios_per_combo"] = None

    if "scenario_id" in df.columns:
        stats_dict["total_scenarios"] = df["scenario_id"].nunique()

    # Parameter combinations (param_H, param_D, and param_Pm or param_load)
    if _has_param_space(df):
        param_combos = df[["param_H", "param_D", op_col]].drop_duplicates()
        stats_dict["unique_combinations"] = len(param_combos)
        stats_dict["operating_point_label"] = _operating_point_label(op_col)

        # Parameter ranges
        stats_dict["H_range"] = (df["param_H"].min(), df["param_H"].max())
        stats_dict["D_range"] = (df["param_D"].min(), df["param_D"].max())
        stats_dict["Pm_range"] = (df[op_col].min(), df[op_col].max())

        # Parameter statistics
        stats_dict["H_mean"] = df["param_H"].mean()
        stats_dict["H_std"] = df["param_H"].std()
        stats_dict["D_mean"] = df["param_D"].mean()
        stats_dict["D_std"] = df["param_D"].std()
        stats_dict["Pm_mean"] = df[op_col].mean()
        stats_dict["Pm_std"] = df[op_col].std()

    # Stability distribution
    if "is_stable" in df.columns:
        if "scenario_id" in df.columns:
            scenario_stability = df.groupby("scenario_id")["is_stable"].first()
            stats_dict["stable_scenarios"] = scenario_stability.sum()
            stats_dict["unstable_scenarios"] = (
                len(scenario_stability) - stats_dict["stable_scenarios"]
            )
            stats_dict["stability_ratio"] = (
                stats_dict["stable_scenarios"] / len(scenario_stability) * 100
            )
        else:
            stats_dict["stable_rows"] = df["is_stable"].sum()
            stats_dict["unstable_rows"] = (~df["is_stable"]).sum()

    # CCT statistics
    if "param_cct_absolute" in df.columns:
        cct_values = df["param_cct_absolute"].dropna().unique()
        if len(cct_values) > 0:
            stats_dict["unique_cct_values"] = len(cct_values)
            stats_dict["cct_mean"] = cct_values.mean()
            stats_dict["cct_std"] = cct_values.std()
            stats_dict["cct_min"] = cct_values.min()
            stats_dict["cct_max"] = cct_values.max()
            stats_dict["cct_range"] = stats_dict["cct_max"] - stats_dict["cct_min"]

            # CCT percentiles
            stats_dict["cct_q25"] = np.percentile(cct_values, 25)
            stats_dict["cct_q50"] = np.percentile(cct_values, 50)
            stats_dict["cct_q75"] = np.percentile(cct_values, 75)
            stats_dict["cct_q95"] = np.percentile(cct_values, 95)

    # Trajectory statistics
    if "scenario_id" in df.columns:
        traj_lengths = df.groupby("scenario_id").size()
        stats_dict["traj_length_mean"] = traj_lengths.mean()
        stats_dict["traj_length_std"] = traj_lengths.std()
        stats_dict["traj_length_min"] = traj_lengths.min()
        stats_dict["traj_length_max"] = traj_lengths.max()

    # Time statistics
    if "time" in df.columns:
        stats_dict["time_min"] = df["time"].min()
        stats_dict["time_max"] = df["time"].max()
        stats_dict["time_range"] = stats_dict["time_max"] - stats_dict["time_min"]

    return stats_dict


def compute_trajectory_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-trajectory statistics."""
    print("\n📈 Computing trajectory-level statistics...")

    if "scenario_id" not in df.columns:
        print("   ⚠️  No scenario_id column found, skipping trajectory statistics")
        return pd.DataFrame()

    traj_stats = []

    for scenario_id in df["scenario_id"].unique():
        traj = df[df["scenario_id"] == scenario_id]

        stats_row = {"scenario_id": scenario_id}

        # Extract parameters
        op_col = _operating_point_col(df)
        if "param_H" in traj.columns:
            stats_row["H"] = traj["param_H"].iloc[0]
        if "param_D" in traj.columns:
            stats_row["D"] = traj["param_D"].iloc[0]
        if op_col in traj.columns:
            stats_row["Pm"] = traj[op_col].iloc[0]
        if "param_cct_absolute" in traj.columns:
            stats_row["CCT"] = traj["param_cct_absolute"].iloc[0]
        if "param_tc" in traj.columns:
            stats_row["clearing_time"] = traj["param_tc"].iloc[0]

        # Delta (rotor angle) statistics
        if "delta" in traj.columns:
            delta = traj["delta"].values
            stats_row["delta_max"] = np.max(np.abs(delta))
            stats_row["delta_max_deg"] = np.degrees(stats_row["delta_max"])
            stats_row["delta_mean"] = np.mean(delta)
            stats_row["delta_std"] = np.std(delta)

            # First swing peak
            if "time" in traj.columns:
                time = traj["time"].values
                # Find first maximum (within first 2 seconds)
                first_2s = time <= (time[0] + 2.0) if len(time) > 0 else np.array([])
                if np.any(first_2s):
                    delta_first_2s = np.abs(delta[first_2s])
                    first_peak_idx = np.argmax(delta_first_2s)
                    stats_row["first_swing_peak"] = delta_first_2s[first_peak_idx]
                    stats_row["first_swing_time"] = time[first_2s][first_peak_idx]

        # [Rotor angle focus: omega statistics commented out]
        # if "omega" in traj.columns:
        #     omega = traj["omega"].values
        #     omega_dev = omega - 1.0  # Deviation from nominal
        #     stats_row["omega_max_dev"] = np.max(np.abs(omega_dev))
        #     stats_row["omega_mean"] = np.mean(omega)
        #     stats_row["omega_std"] = np.std(omega)

        # Stability
        if "is_stable" in traj.columns:
            stats_row["is_stable"] = traj["is_stable"].iloc[0]

        traj_stats.append(stats_row)

    traj_stats_df = pd.DataFrame(traj_stats)
    print(f"   Computed statistics for {len(traj_stats_df)} trajectories")
    return traj_stats_df


def compute_cct_correlations(df: pd.DataFrame) -> Dict:
    """Compute correlations between CCT and parameters."""
    print("\n🔗 Computing CCT correlations...")

    if "param_cct_absolute" not in df.columns:
        print("   ⚠️  No CCT data found, skipping correlations")
        return {}

    op_col = _operating_point_col(df)
    if op_col not in df.columns or not _has_param_space(df):
        print("   ⚠️  Missing parameter columns for CCT correlation")
        return {}

    # Get unique parameter combinations with CCT (param_H, param_D, operating point, CCT)
    cct_data = df[["param_H", "param_D", op_col, "param_cct_absolute"]].drop_duplicates().dropna()

    if len(cct_data) == 0:
        print("   ⚠️  No valid CCT data found")
        return {}

    if len(cct_data) < 2:
        print(
            f"⚠️ Insufficient data for correlation (only {len(cct_data)} sample(s), need at least"
            f"2)"
        )
        print("   ⚠️  Correlation analysis requires multiple parameter combinations")
        return {}

    correlations = {}
    param_list = ["param_H", "param_D", op_col]

    # Pearson correlations
    for param in param_list:
        if param in cct_data.columns:
            corr, p_value = stats.pearsonr(cct_data[param], cct_data["param_cct_absolute"])
            correlations[param] = {
                "pearson_r": corr,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }

    # Spearman correlations (non-parametric)
    for param in param_list:
        if param in cct_data.columns:
            corr, p_value = stats.spearmanr(cct_data[param], cct_data["param_cct_absolute"])
            correlations[param]["spearman_rho"] = corr
            correlations[param]["spearman_p"] = p_value

    return correlations


def generate_parameter_space_figures(
    df: pd.DataFrame,
    output_dir: Path,
    formats: List[str] = ["png", "pdf"],
    *,
    stable_paper_names: bool = False,
) -> Dict[str, Path]:
    """Generate parameter space coverage figures.

    If ``stable_paper_names`` is True, write ``parameter_*.pdf`` (no timestamps) for
    manuscript bundles (see ``scripts/plot_benchmark_ensemble_paper_figures.py``).
    """
    print("\n📐 Generating parameter space figures...")

    figure_paths = {}
    # Ensure output directory exists and is writable
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.exists():
        raise RuntimeError(f"Cannot create output directory: {output_dir}")

    if not _has_param_space(df):
        print(
            "   ⚠️  Missing parameter columns (need param_H, param_D, and param_Pm or param_load)"
        )
        return figure_paths

    op_col = _operating_point_col(df)
    op_label = _operating_point_label(op_col)
    param_combos = df[["param_H", "param_D", op_col]].drop_duplicates()

    # Figure 1: 3D scatter plot (2D projections)
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE_DOUBLE)

    # H vs D
    axes[0].scatter(
        param_combos["param_H"],
        param_combos["param_D"],
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[0].set_xlabel("Inertia Constant H (s)", fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Damping Coefficient D (pu)", fontweight="bold", fontsize=10)
    axes[0].set_title("(a) H vs D", fontweight="bold", fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # H vs operating point (Pm or Load)
    axes[1].scatter(
        param_combos["param_H"],
        param_combos[op_col],
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[1].set_xlabel("Inertia Constant H (s)", fontweight="bold", fontsize=10)
    axes[1].set_ylabel(op_label, fontweight="bold", fontsize=10)
    axes[1].set_title(f"(b) H vs {op_label.split(' ')[0]}", fontweight="bold", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # D vs operating point
    axes[2].scatter(
        param_combos["param_D"],
        param_combos[op_col],
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[2].set_xlabel("Damping Coefficient D (pu)", fontweight="bold", fontsize=10)
    axes[2].set_ylabel(op_label, fontweight="bold", fontsize=10)
    axes[2].set_title(f"(c) D vs {op_label.split(' ')[0]}", fontweight="bold", fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    for fmt in formats:
        filename = _analysis_figure_filename(
            "parameter_space_coverage", fmt, stable_paper_names=stable_paper_names
        )
        path = output_dir / filename
        try:
            plt.savefig(path, dpi=PUBLICATION_DPI, bbox_inches="tight", format=fmt)
            import time

            time.sleep(0.1)  # Allow file system to sync
            if path.exists() and path.stat().st_size > 0:
                print(f"   ✓ Saved: {path.name} ({path.stat().st_size} bytes)")
            else:
                print(f"   ❌ Failed to save: {path.name}")
                print(f"      Path: {path.absolute()}")
        except Exception as e:
            print(f"   ❌ Error saving {path.name}: {e}")
            import traceback

            traceback.print_exc()
        if fmt == formats[0]:
            figure_paths["parameter_space_coverage"] = path
    plt.close()  # Close after all formats are saved

    # Figure 1b: 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        param_combos["param_H"],
        param_combos["param_D"],
        param_combos[op_col],
        c=param_combos[op_col],
        cmap="viridis",
        alpha=0.7,
        s=100,
        edgecolors="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Inertia Constant H (s)", fontweight="bold", fontsize=11, labelpad=10)
    ax.set_ylabel("Damping Coefficient D (pu)", fontweight="bold", fontsize=11, labelpad=10)
    ax.set_zlabel(op_label, fontweight="bold", fontsize=11, labelpad=10)
    ax.set_title("3D Parameter Space Coverage", fontweight="bold", fontsize=12, pad=20)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label(op_label, fontweight="bold", fontsize=10)

    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    for fmt in formats:
        filename = _analysis_figure_filename(
            "parameter_space_coverage_3d", fmt, stable_paper_names=stable_paper_names
        )
        path = output_dir / filename
        try:
            plt.savefig(path, dpi=PUBLICATION_DPI, bbox_inches="tight", format=fmt)
            import time

            time.sleep(0.1)  # Allow file system to sync
            if path.exists() and path.stat().st_size > 0:
                print(f"   ✓ Saved: {path.name} ({path.stat().st_size} bytes)")
            else:
                print(f"   ❌ Failed to save: {path.name}")
                print(f"      Path: {path.absolute()}")
        except Exception as e:
            print(f"   ❌ Error saving {path.name}: {e}")
            import traceback

            traceback.print_exc()
        if fmt == formats[0]:
            figure_paths["parameter_space_coverage_3d"] = path
    plt.close()  # Close after all formats are saved

    # Figure 2: Parameter distributions
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE_DOUBLE)

    for idx, (param, label) in enumerate(
        [("param_H", "H (s)"), ("param_D", "D (pu)"), (op_col, op_label)]
    ):
        unique_vals = param_combos[param].values
        axes[idx].hist(unique_vals, bins=15, alpha=0.7, edgecolor="black", linewidth=1)
        axes[idx].set_xlabel(label, fontweight="bold", fontsize=10)
        axes[idx].set_ylabel("Frequency", fontweight="bold", fontsize=10)
        axes[idx].set_title(f"({chr(97+idx)}) {label} Distribution", fontweight="bold", fontsize=11)
        axes[idx].grid(True, alpha=0.3, axis="y")

        # Add statistics text
        mean_val = np.mean(unique_vals)
        std_val = np.std(unique_vals)
        axes[idx].axvline(
            mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}"
        )
        axes[idx].legend(fontsize=8)

    plt.tight_layout()

    for fmt in formats:
        filename = _analysis_figure_filename(
            "parameter_distributions", fmt, stable_paper_names=stable_paper_names
        )
        path = output_dir / filename
        try:
            plt.savefig(path, dpi=PUBLICATION_DPI, bbox_inches="tight", format=fmt)
            import time

            time.sleep(0.1)  # Allow file system to sync
            if path.exists() and path.stat().st_size > 0:
                print(f"   ✓ Saved: {path.name} ({path.stat().st_size} bytes)")
            else:
                print(f"   ❌ Failed to save: {path.name}")
        except Exception as e:
            print(f"   ❌ Error saving {path.name}: {e}")
        if fmt == formats[0]:
            figure_paths["parameter_distributions"] = path
    plt.close()  # Close after all formats are saved

    return figure_paths


def generate_trajectory_figures(
    df: pd.DataFrame, output_dir: Path, formats: List[str] = ["png", "pdf"], n_examples: int = 3
) -> Dict[str, Path]:
    """Generate representative trajectory figures."""
    print("\n📈 Generating trajectory figures...")

    figure_paths = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    if "scenario_id" not in df.columns or "time" not in df.columns:
        print("   ⚠️  Missing required columns for trajectory plots")
        return figure_paths

    op_col = _operating_point_col(df)
    op_label_short = "Load" if op_col == "param_load" else "Pm"

    # Select representative trajectories
    stable_scenarios = (
        df[df["is_stable"] == True]["scenario_id"].unique()[:n_examples]
        if "is_stable" in df.columns
        else df["scenario_id"].unique()[:n_examples]
    )
    unstable_scenarios = (
        df[df["is_stable"] == False]["scenario_id"].unique()[:n_examples]
        if "is_stable" in df.columns
        else []
    )

    # Figure 1: Stable trajectories
    if len(stable_scenarios) > 0:
        # Increase height slightly to accommodate multi-line titles with parameter info
        fig, axes = plt.subplots(
            len(stable_scenarios), 2, figsize=(FIGURE_SIZE_DOUBLE[0], 3.5 * len(stable_scenarios))
        )
        if len(stable_scenarios) == 1:
            axes = axes.reshape(1, -1)

        for idx, scenario_id in enumerate(stable_scenarios):
            traj = df[df["scenario_id"] == scenario_id].sort_values("time")
            time = traj["time"].values
            delta = traj["delta"].values if "delta" in traj.columns else None
            # omega = traj["omega"].values if "omega" in traj.columns else None  # [Rotor angle focus: commented out]

            # Extract system parameters
            H = traj["param_H"].iloc[0] if "param_H" in traj.columns else None
            D = traj["param_D"].iloc[0] if "param_D" in traj.columns else None
            Pm = traj[op_col].iloc[0] if op_col in traj.columns else None
            tc = traj["param_tc"].iloc[0] if "param_tc" in traj.columns else None
            cct = (
                traj["param_cct_absolute"].iloc[0] if "param_cct_absolute" in traj.columns else None
            )
            offset = (
                traj["param_offset_from_cct"].iloc[0]
                if "param_offset_from_cct" in traj.columns
                else None
            )
            tf = 1.0  # Default fault start time

            # Build parameter info string
            param_info_parts = []
            if H is not None:
                param_info_parts.append(f"H={H:.2f}s")
            if D is not None:
                param_info_parts.append(f"D={D:.2f}pu")
            if Pm is not None:
                param_info_parts.append(f"{op_label_short}={Pm:.2f}")
            param_info = ", ".join(param_info_parts)

            # Build timing info string
            timing_info_parts = []
            if tc is not None:
                timing_info_parts.append(f"tc={tc:.3f}s")
            if cct is not None and offset is not None:
                timing_info_parts.append(f"CCT={cct:.3f}s")
                timing_info_parts.append(f"offset={offset*1000:+.1f}ms")
            timing_info = ", ".join(timing_info_parts)

            # Plot delta (rotor angle only)
            if delta is not None:
                axes[idx, 0].plot(time, np.degrees(delta), "b-", linewidth=2, label="Rotor Angle")
                if tc:
                    axes[idx, 0].axvline(
                        tc, color="orange", linestyle="--", linewidth=1.5, label="Fault Clear"
                    )
                axes[idx, 0].axvline(
                    tf, color="red", linestyle="--", linewidth=1.5, label="Fault Start"
                )
                axes[idx, 0].set_xlabel("Time (s)", fontweight="bold", fontsize=10)
                axes[idx, 0].set_ylabel("Rotor Angle (deg)", fontweight="bold", fontsize=10)

                # Enhanced title with system parameters
                title = f"Scenario {scenario_id} - Stable"
                if param_info:
                    title += f"\n({param_info})"
                if timing_info:
                    title += f"\n{timing_info}"
                axes[idx, 0].set_title(title, fontweight="bold", fontsize=10)
                axes[idx, 0].grid(True, alpha=0.3)
                axes[idx, 0].legend(fontsize=8)

            # [Rotor angle focus: omega plot commented out]
            # if omega is not None:
            #     axes[idx, 1].plot(time, omega, "g-", linewidth=2, label="Rotor Speed")
            #     axes[idx, 1].axhline(1.0, color="k", linestyle=":", linewidth=1, alpha=0.5, label="Nominal")
            #     if tc:
            #         axes[idx, 1].axvline(tc, color="orange", linestyle="--", linewidth=1.5, label="Fault Clear")
            #     axes[idx, 1].axvline(tf, color="red", linestyle="--", linewidth=1.5, label="Fault Start")
            #     axes[idx, 1].set_xlabel("Time (s)", fontweight="bold", fontsize=10)
            #     axes[idx, 1].set_ylabel("Rotor Speed (pu)", fontweight="bold", fontsize=10)
            #     axes[idx, 1].set_title(title, fontweight="bold", fontsize=10)
            #     axes[idx, 1].grid(True, alpha=0.3)
            #     axes[idx, 1].legend(fontsize=8)

        plt.tight_layout()

        for fmt in formats:
            filename = generate_timestamped_filename("stable_trajectories", fmt)
            path = output_dir / filename
            try:
                plt.savefig(path, dpi=PUBLICATION_DPI, bbox_inches="tight", format=fmt)
                import time

                time.sleep(0.1)  # Allow file system to sync
                if path.exists() and path.stat().st_size > 0:
                    print(f"   ✓ Saved: {path.name} ({path.stat().st_size} bytes)")
                else:
                    print(f"   ❌ Failed to save: {path.name}")
            except Exception as e:
                print(f"   ❌ Error saving {path.name}: {e}")
            if fmt == formats[0]:
                figure_paths["stable_trajectories"] = path
        plt.close()  # Close after all formats are saved

    # Figure 2: Unstable trajectories
    if len(unstable_scenarios) > 0:
        # Increase height slightly to accommodate multi-line titles with parameter info
        fig, axes = plt.subplots(
            len(unstable_scenarios),
            2,
            figsize=(FIGURE_SIZE_DOUBLE[0], 3.5 * len(unstable_scenarios)),
        )
        if len(unstable_scenarios) == 1:
            axes = axes.reshape(1, -1)

        for idx, scenario_id in enumerate(unstable_scenarios):
            traj = df[df["scenario_id"] == scenario_id].sort_values("time")
            time = traj["time"].values
            delta = traj["delta"].values if "delta" in traj.columns else None
            # omega = traj["omega"].values if "omega" in traj.columns else None  # [Rotor angle focus: commented out]

            # Extract system parameters
            H = traj["param_H"].iloc[0] if "param_H" in traj.columns else None
            D = traj["param_D"].iloc[0] if "param_D" in traj.columns else None
            Pm = traj[op_col].iloc[0] if op_col in traj.columns else None
            tc = traj["param_tc"].iloc[0] if "param_tc" in traj.columns else None
            cct = (
                traj["param_cct_absolute"].iloc[0] if "param_cct_absolute" in traj.columns else None
            )
            offset = (
                traj["param_offset_from_cct"].iloc[0]
                if "param_offset_from_cct" in traj.columns
                else None
            )
            tf = 1.0

            # Build parameter info string
            param_info_parts = []
            if H is not None:
                param_info_parts.append(f"H={H:.2f}s")
            if D is not None:
                param_info_parts.append(f"D={D:.2f}pu")
            if Pm is not None:
                param_info_parts.append(f"{op_label_short}={Pm:.2f}")
            param_info = ", ".join(param_info_parts)

            # Build timing info string
            timing_info_parts = []
            if tc is not None:
                timing_info_parts.append(f"tc={tc:.3f}s")
            if cct is not None and offset is not None:
                timing_info_parts.append(f"CCT={cct:.3f}s")
                timing_info_parts.append(f"offset={offset*1000:+.1f}ms")
            timing_info = ", ".join(timing_info_parts)

            if delta is not None:
                axes[idx, 0].plot(time, np.degrees(delta), "r-", linewidth=2, label="Rotor Angle")
                if tc:
                    axes[idx, 0].axvline(
                        tc, color="orange", linestyle="--", linewidth=1.5, label="Fault Clear"
                    )
                axes[idx, 0].axvline(
                    tf, color="red", linestyle="--", linewidth=1.5, label="Fault Start"
                )
                axes[idx, 0].set_xlabel("Time (s)", fontweight="bold", fontsize=10)
                axes[idx, 0].set_ylabel("Rotor Angle (deg)", fontweight="bold", fontsize=10)

                # Enhanced title with system parameters
                title = f"Scenario {scenario_id} - Unstable"
                if param_info:
                    title += f"\n({param_info})"
                if timing_info:
                    title += f"\n{timing_info}"
                axes[idx, 0].set_title(title, fontweight="bold", fontsize=10)
                axes[idx, 0].grid(True, alpha=0.3)
                axes[idx, 0].legend(fontsize=8)

            # [Rotor angle focus: omega plot commented out]
            # if omega is not None:
            #     axes[idx, 1].plot(time, omega, "orange", linewidth=2, label="Rotor Speed")
            #     axes[idx, 1].axhline(1.0, color="k", linestyle=":", linewidth=1, alpha=0.5, label="Nominal")
            #     if tc:
            #         axes[idx, 1].axvline(tc, color="orange", linestyle="--", linewidth=1.5, label="Fault Clear")
            #     axes[idx, 1].axvline(tf, color="red", linestyle="--", linewidth=1.5, label="Fault Start")
            #     axes[idx, 1].set_xlabel("Time (s)", fontweight="bold", fontsize=10)
            #     axes[idx, 1].set_ylabel("Rotor Speed (pu)", fontweight="bold", fontsize=10)
            #     axes[idx, 1].set_title(title, fontweight="bold", fontsize=10)
            #     axes[idx, 1].grid(True, alpha=0.3)
            #     axes[idx, 1].legend(fontsize=8)

        plt.tight_layout()

        for fmt in formats:
            filename = generate_timestamped_filename("unstable_trajectories", fmt)
            path = output_dir / filename
            try:
                plt.savefig(path, dpi=PUBLICATION_DPI, bbox_inches="tight", format=fmt)
                import time

                time.sleep(0.1)  # Allow file system to sync
                if path.exists() and path.stat().st_size > 0:
                    print(f"   ✓ Saved: {path.name} ({path.stat().st_size} bytes)")
                else:
                    print(f"   ❌ Failed to save: {path.name}")
            except Exception as e:
                print(f"   ❌ Error saving {path.name}: {e}")
            if fmt == formats[0]:
                figure_paths["unstable_trajectories"] = path
        plt.close()  # Close after all formats are saved

    return figure_paths


def generate_cct_figures(
    df: pd.DataFrame,
    output_dir: Path,
    formats: List[str] = ["png", "pdf"],
    *,
    stable_paper_names: bool = False,
) -> Dict[str, Path]:
    """Generate CCT analysis figures.

    If ``stable_paper_names`` is True, write ``cct_*.pdf`` (no timestamps) for
    manuscript bundles.
    """
    print("\n⏱️  Generating CCT analysis figures...")

    figure_paths = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    if "param_cct_absolute" not in df.columns:
        print("   ⚠️  No CCT data found, skipping CCT figures")
        return figure_paths

    if not _has_param_space(df):
        print("   ⚠️  Missing param_H, param_D, or operating point column for CCT figures")
        return figure_paths

    op_col = _operating_point_col(df)
    # Get unique parameter combinations with CCT (param_H, param_D, operating point, CCT)
    cct_data = df[["param_H", "param_D", op_col, "param_cct_absolute"]].drop_duplicates().dropna()

    if len(cct_data) == 0:
        print("   ⚠️  No valid CCT data found")
        return figure_paths

    if len(cct_data) < 2:
        print(
            f"⚠️ Insufficient data for CCT correlation plots (only {len(cct_data)} sample(s), need"
            f"at least 2)"
        )
        print("   ⚠️  Skipping CCT vs parameters figure (correlation requires multiple samples)")
        # Still generate CCT distribution if we have at least 1 sample
        if len(cct_data) == 1:
            # Generate simple CCT distribution for single sample
            fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE_SINGLE)
            cct_value = cct_data["param_cct_absolute"].values[0]
            ax.axvline(
                cct_value, color="red", linestyle="--", linewidth=2, label=f"CCT: {cct_value:.3f} s"
            )
            ax.set_xlabel("Critical Clearing Time (s)", fontweight="bold", fontsize=10)
            ax.set_title("CCT (Single Sample)", fontweight="bold", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()

            for fmt in formats:
                filename = _analysis_figure_filename(
                    "cct_distribution", fmt, stable_paper_names=stable_paper_names
                )
                path = output_dir / filename
                try:
                    plt.savefig(path, dpi=PUBLICATION_DPI, bbox_inches="tight", format=fmt)
                    import time

                    time.sleep(0.1)  # Allow file system to sync
                    if path.exists() and path.stat().st_size > 0:
                        print(f"   ✓ Saved: {path.name} ({path.stat().st_size} bytes)")
                    else:
                        print(f"   ❌ Failed to save: {path.name}")
                except Exception as e:
                    print(f"   ❌ Error saving {path.name}: {e}")
                if fmt == formats[0]:
                    figure_paths["cct_distribution"] = path
            plt.close()  # Close after all formats are saved

        return figure_paths

    # Figure 1: CCT distribution
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_DOUBLE)

    # Histogram
    cct_values = cct_data["param_cct_absolute"].values
    axes[0].hist(cct_values, bins=20, alpha=0.7, edgecolor="black", linewidth=1)
    axes[0].axvline(
        np.mean(cct_values),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(cct_values):.3f} s",
    )
    axes[0].set_xlabel("Critical Clearing Time (s)", fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Frequency", fontweight="bold", fontsize=10)
    axes[0].set_title("(a) CCT Distribution", fontweight="bold", fontsize=11)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].legend(fontsize=9)

    # Box plot
    axes[1].boxplot(cct_values, vert=True, patch_artist=True)
    axes[1].set_ylabel("Critical Clearing Time (s)", fontweight="bold", fontsize=10)
    axes[1].set_title("(b) CCT Box Plot", fontweight="bold", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    for fmt in formats:
        filename = _analysis_figure_filename(
            "cct_distribution", fmt, stable_paper_names=stable_paper_names
        )
        path = output_dir / filename
        try:
            plt.savefig(path, dpi=PUBLICATION_DPI, bbox_inches="tight", format=fmt)
            import time

            time.sleep(0.1)  # Allow file system to sync
            if path.exists() and path.stat().st_size > 0:
                print(f"   ✓ Saved: {path.name} ({path.stat().st_size} bytes)")
            else:
                print(f"   ❌ Failed to save: {path.name}")
        except Exception as e:
            print(f"   ❌ Error saving {path.name}: {e}")
        if fmt == formats[0]:
            figure_paths["cct_distribution"] = path
    plt.close()  # Close after all formats are saved

    # Figure 2: CCT vs Parameters
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE_DOUBLE)
    param_list_cct = ["param_H", "param_D", op_col]

    for idx, param in enumerate(param_list_cct):
        if param in cct_data.columns:
            axes[idx].scatter(
                cct_data[param],
                cct_data["param_cct_absolute"],
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )

            # Add trend line
            x = cct_data[param].values.reshape(-1, 1)
            y = cct_data["param_cct_absolute"].values
            model = LinearRegression()
            model.fit(x, y)
            x_trend = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
            y_trend = model.predict(x_trend)
            axes[idx].plot(x_trend, y_trend, "r--", linewidth=2, label="Trend")

            # Compute correlation
            corr, p_val = stats.pearsonr(cct_data[param], cct_data["param_cct_absolute"])
            label = param.replace("param_", "")
            axes[idx].set_xlabel(f"{label}", fontweight="bold", fontsize=10)
            axes[idx].set_ylabel("CCT (s)", fontweight="bold", fontsize=10)
            axes[idx].set_title(
                f"({chr(97+idx)}) CCT vs {label}\n(r={corr:.3f}, p={p_val:.3f})",
                fontweight="bold",
                fontsize=10,
            )
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend(fontsize=8)

    plt.tight_layout()

    for fmt in formats:
        filename = _analysis_figure_filename(
            "cct_vs_parameters", fmt, stable_paper_names=stable_paper_names
        )
        path = output_dir / filename
        try:
            plt.savefig(path, dpi=PUBLICATION_DPI, bbox_inches="tight", format=fmt)
            import time

            time.sleep(0.1)  # Allow file system to sync
            if path.exists() and path.stat().st_size > 0:
                print(f"   ✓ Saved: {path.name} ({path.stat().st_size} bytes)")
            else:
                print(f"   ❌ Failed to save: {path.name}")
        except Exception as e:
            print(f"   ❌ Error saving {path.name}: {e}")
        if fmt == formats[0]:
            figure_paths["cct_vs_parameters"] = path
    plt.close()  # Close after all formats are saved

    return figure_paths


def generate_summary_report(
    stats_dict: Dict,
    traj_stats_df: pd.DataFrame,
    cct_correlations: Dict,
    output_dir: Path,
) -> Path:
    """Generate a summary report in text format."""
    print("\n📝 Generating summary report...")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_filename = generate_timestamped_filename("analysis_summary_report", "txt")
    report_path = output_dir / report_filename

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("DATASET ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Dataset Overview
        f.write("1. DATASET OVERVIEW\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total rows: {stats_dict.get('total_rows', 'N/A'):,}\n")
        f.write(f"Total columns: {stats_dict.get('total_columns', 'N/A')}\n")
        if "total_scenarios" in stats_dict:
            f.write(f"Total scenarios (trajectories): {stats_dict['total_scenarios']:,}\n")
        if "unique_combinations" in stats_dict:
            f.write(f"Unique parameter combinations: {stats_dict['unique_combinations']}\n")
        f.write("\n")

        # Parameter Ranges
        f.write("2. PARAMETER RANGES\n")
        f.write("-" * 70 + "\n")
        if "H_range" in stats_dict:
            f.write(
                f"H (Inertia): [{stats_dict['H_range'][0]:.3f}, {stats_dict['H_range'][1]:.3f}] s\n"
            )
            f.write(
                f"  Mean: {stats_dict.get('H_mean', 0):.3f} ± {stats_dict.get('H_std', 0):.3f} s\n"
            )
        if "D_range" in stats_dict:
            f.write(
                f"D (Damping): [{stats_dict['D_range'][0]:.3f}, {stats_dict['D_range'][1]:.3f}] pu\n"
            )
            f.write(
                f"  Mean: {stats_dict.get('D_mean', 0):.3f} ± {stats_dict.get('D_std', 0):.3f} pu\n"
            )
        if "Pm_range" in stats_dict:
            op_label = stats_dict.get("operating_point_label", "Pm (pu)")
            f.write(
                f"{op_label}: [{stats_dict['Pm_range'][0]:.3f}, {stats_dict['Pm_range'][1]:.3f}]\n"
            )
            f.write(f"  Mean: {stats_dict.get('Pm_mean', 0):.3f} ± {stats_dict.get('Pm_std', 0)}\n")
        f.write("\n")

        # Stability Distribution
        f.write("3. STABILITY DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        if "stable_scenarios" in stats_dict:
            f.write(f"Stable scenarios: {stats_dict['stable_scenarios']}\n")
            f.write(f"Unstable scenarios: {stats_dict['unstable_scenarios']}\n")
            if "stability_ratio" in stats_dict:
                f.write(f"Stability ratio: {stats_dict['stability_ratio']:.1f}% stable\n")
        f.write("\n")

        # CCT Statistics
        f.write("4. CCT STATISTICS\n")
        f.write("-" * 70 + "\n")
        if "cct_mean" in stats_dict:
            f.write(f"Unique CCT values: {stats_dict.get('unique_cct_values', 'N/A')}\n")
            f.write(
                f"Mean CCT: {stats_dict['cct_mean']:.6f} ± {stats_dict.get('cct_std', 0):.6f} s\n"
            )
            f.write(
                f"CCT range: [{stats_dict.get('cct_min', 0):.6f}, {stats_dict.get('cct_max', 0):.6f}] s\n"
            )
            f.write(f"  Q25: {stats_dict.get('cct_q25', 0):.6f} s\n")
            f.write(f"  Q50 (median): {stats_dict.get('cct_q50', 0):.6f} s\n")
            f.write(f"  Q75: {stats_dict.get('cct_q75', 0):.6f} s\n")
            f.write(f"  Q95: {stats_dict.get('cct_q95', 0):.6f} s\n")
        f.write("\n")

        # CCT Correlations
        f.write("5. CCT CORRELATIONS\n")
        f.write("-" * 70 + "\n")
        for param, corr_data in cct_correlations.items():
            label = param.replace("param_", "")
            f.write(f"{label}:\n")
            f.write(f"  Pearson r: {corr_data.get('pearson_r', 0):.4f}\n")
            f.write(f"  p-value: {corr_data.get('p_value', 1):.4f}\n")
            f.write(f"  Significant: {corr_data.get('significant', False)}\n")
            if "spearman_rho" in corr_data:
                f.write(f"  Spearman ρ: {corr_data['spearman_rho']:.4f}\n")
            f.write("\n")

        # Trajectory Statistics Summary
        if len(traj_stats_df) > 0:
            f.write("6. TRAJECTORY STATISTICS SUMMARY\n")
            f.write("-" * 70 + "\n")
            if "delta_max_deg" in traj_stats_df.columns:
                f.write(f"Max rotor angle (δ_max):\n")
                f.write(f"  Mean: {traj_stats_df['delta_max_deg'].mean():.2f}°\n")
                f.write(f"  Std: {traj_stats_df['delta_max_deg'].std():.2f}°\n")
                f.write(f"  Max: {traj_stats_df['delta_max_deg'].max():.2f}°\n")
            # [Rotor angle focus: omega summary commented out]
            # if "omega_max_dev" in traj_stats_df.columns:
            #     f.write(f"Max speed deviation (|ω-1|_max):\n")
            #     f.write(f"  Mean: {traj_stats_df['omega_max_dev'].mean():.4f} pu\n")
            #     f.write(f"  Std: {traj_stats_df['omega_max_dev'].std():.4f} pu\n")
            #     f.write(f"  Max: {traj_stats_df['omega_max_dev'].max():.4f} pu\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("Report generated successfully.\n")
        f.write("=" * 70 + "\n")

    print(f"   ✅ Report saved to: {report_path}")
    return report_path


def main():
    """Main data analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Comprehensive dataset analysis for publication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest data file
  python scripts/analyze_data.py
  
  # Analyze specific file
  python scripts/analyze_data.py data/generated/quick_test/parameter_sweep_data_*.csv
  
  # Generate PDF figures too
  python scripts/analyze_data.py --format png pdf
        """,
    )
    parser.add_argument(
        "data_file",
        nargs="?",
        type=str,
        help="Path to data CSV file (default: latest in data/common/). "
        "Can be from any source (local, Colab, etc.)",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["quick", "moderate", "comprehensive"],
        default=None,
        help="Data level to search if data_file not specified (default: quick)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to search for data files (alternative to --level)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/analysis",
        help="Output directory for figures and reports (default: results/analysis)",
    )
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        default=["png"],
        help="Figure formats (default: png). Use --format png pdf for both formats.",
    )

    args = parser.parse_args()

    # Determine data file
    if args.data_file:
        data_path_str = args.data_file

        # Check if user provided placeholder (common mistake)
        if "YYYYMMDD_HHMMSS" in data_path_str or "YYYYMMDD" in data_path_str:
            print(f"❌ Error: Placeholder detected in path: {data_path_str}")
            print(f"\n💡 The path contains 'YYYYMMDD_HHMMSS' which is a placeholder.")
            print(f"   Replace it with the actual timestamp from your generated file.")
            print(f"\n   Example:")
            print(f"   ❌ Wrong: data/generated/quick_test/parameter_sweep_data_YYYYMMDD_HHMMSS.csv")
            print(
                f"✅ Right:"
                f"data/common/trajectory_data_1000_H2-10_D0.5-3_abc12345_20251205_170908.csv"
            )
            print(f"   ✅ Or use: data/common/trajectory_data_*.csv (wildcard)")
            print(f"\n   Or use a glob pattern to find the latest file:")
            print(f"   ✅ Right: data/generated/quick_test/parameter_sweep_data_*.csv")
            sys.exit(1)

        # Try to resolve the path (handles glob patterns)
        data_path = Path(data_path_str)

        # Handle glob patterns
        if "*" in str(data_path) or "?" in str(data_path):
            matches = list(data_path.parent.glob(data_path.name))
            if matches:
                data_path = max(matches, key=lambda p: p.stat().st_mtime)
                print(f"📂 Using latest matching file: {data_path.name}")
            else:
                print(f"❌ No files found matching pattern: {data_path_str}")
                sys.exit(1)
        else:
            data_path = Path(data_path_str)
    else:
        # Use data_utils to find file (supports --level and --data-dir)
        from scripts.core.data_utils import find_data_file

        try:
            data_path = find_data_file(
                data_path=None,
                data_dir=args.data_dir,
                level=args.level,
                pattern="parameter_sweep_data_*.csv",
            )
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print(f"\n💡 Available files in {data_path.parent}:")
        all_csv = list(data_path.parent.glob("*.csv"))
        if all_csv:
            for f in sorted(all_csv, key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
                print(f"   - {f.name}")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 70)
    print(f"Data file: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Figure formats: {', '.join(args.format)} (default: png only)")
    print("=" * 70)

    # Load data
    df = load_data(data_path)

    # Compute statistics
    stats_dict = compute_dataset_statistics(df)
    traj_stats_df = compute_trajectory_statistics(df)
    cct_correlations = compute_cct_correlations(df)

    # Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    param_figures = generate_parameter_space_figures(df, figures_dir, args.format)
    traj_figures = generate_trajectory_figures(df, figures_dir, args.format)
    cct_figures = generate_cct_figures(df, figures_dir, args.format)

    # Generate summary report
    report_path = generate_summary_report(stats_dict, traj_stats_df, cct_correlations, output_dir)

    # Save trajectory statistics
    if len(traj_stats_df) > 0:
        traj_stats_filename = generate_timestamped_filename("trajectory_statistics", "csv")
        traj_stats_path = output_dir / traj_stats_filename
        traj_stats_df.to_csv(traj_stats_path, index=False)
        print(f"\n💾 Trajectory statistics saved to: {traj_stats_path}")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"✅ Figures saved to: {figures_dir}")
    print(f"   - Parameter space: {len(param_figures)} figures")
    print(f"   - Trajectories: {len(traj_figures)} figures")
    print(f"   - CCT analysis: {len(cct_figures)} figures")
    print(f"✅ Summary report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
