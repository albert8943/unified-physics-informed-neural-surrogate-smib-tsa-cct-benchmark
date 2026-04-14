#!/usr/bin/env python
"""
Analyze multimachine (Kundur 2-area) trajectory data before training.

Similar to scripts/analyze_smib_cct_results.py for SMIB: reports basic statistics,
stability distribution, CCT/clearing time, trajectory quality (max rotor angle;
omega/frequency deviation analysis is commented out for rotor-angle-only focus),
parameter coverage, and recommendations. Use this after
generate_multimachine_data.py and before preprocess_data.py / training.

Usage:
    python scripts/analyze_multimachine_data.py [data_file.csv]
    python scripts/analyze_multimachine_data.py data/multimachine/kundur/parameter_sweep_data_*.csv --plot --output-dir results/multimachine_analysis
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_data(data_path: Path) -> pd.DataFrame:
    """Load multimachine parameter_sweep CSV."""
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Rows: {len(df):,}  Columns: {len(df.columns)}")
    return df


def _fill_missing_cct_from_boundary(
    df: pd.DataFrame, fault_start_time: float = 1.0
) -> Tuple[pd.DataFrame, int]:
    """Fill missing param_cct_absolute from stable/unstable boundary per operating point.

    Groups by (param_H, param_D, param_load or param_Pm). For each group with any NaN CCT,
    estimates CCT from max(stable tc) and/or min(unstable tc), then fills all rows
    for those scenarios. Returns (df, number of scenarios filled).
    """
    if (
        "scenario_id" not in df.columns
        or "param_tc" not in df.columns
        or "is_stable" not in df.columns
    ):
        return df, 0
    if "param_cct_absolute" not in df.columns:
        df["param_cct_absolute"] = np.nan
    if "param_cct_duration" not in df.columns:
        df["param_cct_duration"] = np.nan
    scenarios_first = df.groupby("scenario_id").first()
    h_col = "param_H" if "param_H" in scenarios_first.columns else None
    d_col = "param_D" if "param_D" in scenarios_first.columns else None
    load_col = "param_load" if "param_load" in scenarios_first.columns else "param_Pm"
    if not h_col or not d_col or load_col not in scenarios_first.columns:
        return df, 0
    nan_count = scenarios_first["param_cct_absolute"].isna().sum()
    if nan_count == 0:
        return df, 0
    fault_start = fault_start_time
    if "param_tf" in scenarios_first.columns and pd.notna(scenarios_first["param_tf"]).any():
        fault_start = float(scenarios_first["param_tf"].dropna().iloc[0])
    filled = 0
    for _keys, grp in scenarios_first.groupby([h_col, d_col, load_col], dropna=False):
        if grp["param_cct_absolute"].notna().all():
            continue
        stable_tc = grp.loc[grp["is_stable"], "param_tc"].dropna()
        unstable_tc = grp.loc[~grp["is_stable"], "param_tc"].dropna()
        if len(stable_tc) > 0 and len(unstable_tc) > 0:
            estimated_cct = (float(stable_tc.max()) + float(unstable_tc.min())) / 2.0
        elif len(stable_tc) > 0:
            estimated_cct = float(stable_tc.max())
        elif len(unstable_tc) > 0:
            estimated_cct = float(unstable_tc.min())
        else:
            continue
        for sid in grp.index:
            if pd.isna(grp.loc[sid, "param_cct_absolute"]):
                mask = df["scenario_id"] == sid
                df.loc[mask, "param_cct_absolute"] = estimated_cct
                df.loc[mask, "param_cct_duration"] = estimated_cct - fault_start
                filled += 1
    return df, filled


def analyze_multimachine_data(
    data_path: Path,
    output_dir: Path = None,
    plot: bool = False,
    num_machines: int = None,
    n_sample_trajectories: int = 5,
    simulation_time_s: float = 5.0,
) -> None:
    """Run full analysis and optionally generate plots and report file."""
    df = load_data(data_path)

    # Fill missing param_cct_absolute from stable/unstable boundary so every scenario has a CCT
    df, n_filled = _fill_missing_cct_from_boundary(df, fault_start_time=1.0)
    if n_filled > 0:
        print(
            f"  Filled param_cct_absolute for {n_filled} scenario(s) from stable/unstable boundary."
        )
        try:
            df.to_csv(data_path, index=False)
            print(f"  Saved updated CSV to {data_path}")
        except OSError as e:
            print(f"  [WARNING] Could not save CSV: {e}")

    if output_dir is None:
        output_dir = data_path.parent / "analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(df)
    has_scenario = "scenario_id" in df.columns
    n_scenarios = df["scenario_id"].nunique() if has_scenario else 0

    # ----- 1. BASIC STATISTICS -----
    print()
    print("=" * 70)
    print("1. BASIC STATISTICS")
    print("=" * 70)
    print(f"Total rows (time points): {n_rows:,}")
    print(f"Unique scenarios: {n_scenarios}")
    if n_scenarios > 0:
        rows_per_scenario = n_rows / n_scenarios
        print(f"Rows per scenario (approx): {rows_per_scenario:.0f}")
    # Infer num_machines from delta/Pe columns if not provided
    delta_cols = [
        c for c in df.columns if c.startswith("delta_") and c != "delta_deg" and c != "delta0"
    ]
    pe_cols = [c for c in df.columns if c.startswith("Pe_") or c == "Pe"]
    if num_machines is None and (delta_cols or pe_cols):
        num_machines = max(
            len([c for c in delta_cols if c.replace("delta_", "").isdigit()]) or 0,
            len([c for c in pe_cols if c == "Pe" or (c.startswith("Pe_") and c[3:].isdigit())])
            or 1,
        )
        if num_machines == 0:
            num_machines = 1
    if num_machines is not None:
        print(f"Machines (inferred or provided): {num_machines}")

    # ----- 2. STABILITY ANALYSIS -----
    print()
    print("=" * 70)
    print("2. STABILITY ANALYSIS")
    print("=" * 70)
    if "is_stable" not in df.columns:
        print("[WARNING] No 'is_stable' column found.")
    else:
        if has_scenario:
            scenario_stability = df.groupby("scenario_id")["is_stable"].first()
            stable_scenarios = int(scenario_stability.sum())
            unstable_scenarios = len(scenario_stability) - stable_scenarios
            stable_pct = (
                100.0 * stable_scenarios / len(scenario_stability) if len(scenario_stability) else 0
            )
            print(f"Stable scenarios:   {stable_scenarios} ({stable_pct:.1f}%)")
            print(f"Unstable scenarios: {unstable_scenarios} ({100 - stable_pct:.1f}%)")
        else:
            stable_rows = df["is_stable"].sum()
            print(f"Stable rows:   {stable_rows} ({100*stable_rows/n_rows:.1f}%)")
            print(f"Unstable rows: {n_rows - stable_rows}")

        # Class balance warning (same as SMIB)
        if has_scenario:
            ratio = stable_scenarios / n_scenarios if n_scenarios else 0
            if ratio == 0:
                print("\n[CRITICAL] All scenarios are unstable (0% stable).")
                print("   Consider: increase CCT offsets below CCT, or check CCT finding.")
            elif ratio < 0.3:
                print(
                    f"\n[WARNING] Class imbalance: only {ratio*100:.1f}% stable (target often 30-70%)."
                )

    # ----- 3. CCT AND CLEARING TIME -----
    print()
    print("=" * 70)
    print("3. CCT AND CLEARING TIME")
    print("=" * 70)
    if has_scenario:
        scenarios = df.groupby("scenario_id").first()
        tc_col = "param_tc" if "param_tc" in scenarios.columns else "tc"
        if tc_col in scenarios.columns:
            tc = scenarios[tc_col].dropna()
            if len(tc) > 0:
                print(
                    f"Clearing time (param_tc): min={tc.min():.3f}s  max={tc.max():.3f}s  mean={tc.mean():.3f}s"
                )
        cct_col = None
        for col in ["param_cct_absolute", "param_cct_duration", "cct"]:
            if col in scenarios.columns and scenarios[col].notna().any():
                cct_col = col
                break
        if cct_col:
            cct = scenarios[cct_col].dropna()
            if len(cct) > 0:
                print(
                    f"CCT ({cct_col}): min={cct.min():.3f}s  max={cct.max():.3f}s  mean={cct.mean():.3f}s"
                )
                if tc_col in scenarios.columns:
                    offset = scenarios[tc_col] - scenarios[cct_col]
                    offset = offset.dropna()
                    if len(offset) > 0:
                        print(
                            f"Offset (tc - CCT): min={offset.min():.4f}s  max={offset.max():.4f}s"
                        )
        else:
            # Estimate CCT from stable/unstable boundary when no param_cct_absolute
            estimated_cct = None
            if "is_stable" in scenarios.columns and tc_col in scenarios.columns:
                stable_tc = scenarios.loc[scenarios["is_stable"] == True, tc_col].dropna()
                unstable_tc = scenarios.loc[scenarios["is_stable"] == False, tc_col].dropna()
                if len(stable_tc) > 0 and len(unstable_tc) > 0:
                    max_stable_tc = float(stable_tc.max())
                    min_unstable_tc = float(unstable_tc.min())
                    estimated_cct = (max_stable_tc + min_unstable_tc) / 2.0
                    print("No CCT column; estimated from stable/unstable boundary:")
                    print(f"  Max stable clearing time:   {max_stable_tc:.3f}s")
                    print(f"  Min unstable clearing time: {min_unstable_tc:.3f}s")
                    print(f"  Estimated CCT (midpoint):  {estimated_cct:.3f}s")
                else:
                    print("No CCT column with data found (param_cct_absolute may be empty).")
                    if len(stable_tc) > 0:
                        estimated_cct = float(stable_tc.max())
                        print(
                            f"  (Max stable tc = {estimated_cct:.3f}s; no unstable tc to bound CCT.)"
                        )
                    elif len(unstable_tc) > 0:
                        print(
                            f"  (Min unstable tc = {float(unstable_tc.min()):.3f}s; no stable tc to bound CCT.)"
                        )
            else:
                print("No CCT column with data found (param_cct_absolute may be empty).")

    # ----- 4. TRAJECTORY QUALITY -----
    print()
    print("=" * 70)
    print("4. TRAJECTORY QUALITY")
    print("=" * 70)
    if has_scenario:
        # Use COI-relative angles for multimachine (max |δ_rel|); else single delta_deg
        max_angles = _max_angle_per_scenario(df)
        if max_angles is not None:
            print("Rotor angle (max |delta - delta_COI| or delta_deg) per scenario:")
            print(
                f"  Min max: {max_angles.min():.1f} deg  Max max: {max_angles.max():.1f} deg  Mean max: {max_angles.mean():.1f} deg"
            )
            above_180 = (max_angles > 180).sum()
            print(
                f"  Scenarios with max angle > 180 deg: {above_180}/{len(max_angles)} ({100*above_180/len(max_angles):.1f}%)"
            )
            if above_180 == len(max_angles):
                print("  [CRITICAL] All scenarios exceed 180 deg (loss of synchronism).")
        # [Focus on rotor angle only: omega/frequency deviation analysis commented out]
        # if "omega_deviation" in df.columns:
        #     max_omega = df.groupby("scenario_id")["omega_deviation"].max()
        #     max_omega_hz = np.abs(max_omega) / (2 * np.pi) if max_omega.abs().max() < 10 else np.abs(max_omega)
        #     print(f"Frequency deviation (omega_deviation) per scenario:")
        #     print(f"  Min max: {max_omega.min():.4f}  Max max: {max_omega.max():.4f} pu")
        #     print(f"  Approx max deviation: {max_omega_hz.max():.2f} Hz (if rad/s, divide by 2*pi)")
        #     above_05 = (np.abs(max_omega) > 0.5 * 2 * np.pi).sum()
        #     if above_05 > 0:
        #         print(f"  Scenarios with |omega_dev| > ~0.5 Hz: {above_05}/{len(max_omega)}")

    # ----- 5. PARAMETER COVERAGE -----
    print()
    print("=" * 70)
    print("5. PARAMETER COVERAGE")
    print("=" * 70)
    if has_scenario:
        scenarios = df.groupby("scenario_id").first()

        # Per-machine param columns (e.g. param_H_0, param_H_1, ...)
        def _per_unit_cols(prefix):
            return sorted(
                [
                    c
                    for c in scenarios.columns
                    if c.startswith(prefix + "_") and c[len(prefix) + 1 :].isdigit()
                ],
                key=lambda x: int(x.split("_")[-1]),
            )

        for param_prefix, label_prefix in [
            ("param_H", "H (inertia)"),
            ("param_D", "D (damping)"),
            ("param_Pm", "Pm (mech. power)"),
            ("param_load", "Load"),
        ]:
            per_unit = _per_unit_cols(param_prefix)
            if per_unit:
                for col in per_unit:
                    v = scenarios[col].dropna()
                    if len(v) > 0:
                        unit_idx = col.split("_")[-1]
                        print(
                            f"  {label_prefix} gen {unit_idx}: [{v.min():.3f}, {v.max():.3f}]  unique={v.nunique()}"
                        )
                vals = [scenarios[c].dropna() for c in per_unit if scenarios[c].notna().any()]
                if vals:
                    agg_min = min(x.min() for x in vals)
                    agg_max = max(x.max() for x in vals)
                    agg_unique = max(x.nunique() for x in vals)
                    print(
                        f"  {label_prefix} (aggregated): [{agg_min:.3f}, {agg_max:.3f}]  max_unique={agg_unique}"
                    )
            elif param_prefix in scenarios.columns:
                v = scenarios[param_prefix].dropna()
                if len(v) > 0:
                    print(f"  {label_prefix}: [{v.min():.3f}, {v.max():.3f}]  unique={v.nunique()}")

    # ----- 6. SUMMARY AND RECOMMENDATIONS -----
    print()
    print("=" * 70)
    print("6. SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)
    issues = []
    if "is_stable" in df.columns and has_scenario:
        stable_ratio = df.groupby("scenario_id")["is_stable"].first().mean()
        if stable_ratio == 0:
            issues.append("All scenarios unstable; check CCT offsets or CCT finding.")
        elif stable_ratio < 0.3:
            issues.append(
                f"Low stable ratio ({100*stable_ratio:.1f}%); consider more clearing times below CCT."
            )
    max_angles = _max_angle_per_scenario(df) if has_scenario else None
    if max_angles is not None and (max_angles > 180).all():
        issues.append("All scenarios exceed 180 deg; data is all unstable.")

    if issues:
        print("Issues detected:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("No critical issues detected. Data may be suitable for preprocessing and training.")

    print()
    print("=" * 70)

    # ----- REPORT FILE -----
    # Recompute estimated_cct for report if we have stable+unstable but no CCT column
    report_estimated_cct = None
    if has_scenario and "is_stable" in df.columns and "param_tc" in df.columns:
        scenarios = df.groupby("scenario_id").first()
        tc_col = "param_tc" if "param_tc" in scenarios.columns else "tc"
        has_cct = any(
            scenarios[col].notna().any()
            for col in ["param_cct_absolute", "param_cct_duration", "cct"]
            if col in scenarios.columns
        )
        if not has_cct and tc_col in scenarios.columns:
            stable_tc = scenarios.loc[scenarios["is_stable"] == True, tc_col].dropna()
            unstable_tc = scenarios.loc[scenarios["is_stable"] == False, tc_col].dropna()
            if len(stable_tc) > 0 and len(unstable_tc) > 0:
                report_estimated_cct = (float(stable_tc.max()) + float(unstable_tc.min())) / 2.0
            elif len(stable_tc) > 0:
                report_estimated_cct = float(stable_tc.max())

    report_path = output_dir / "multimachine_data_analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Multimachine Data Analysis Report\n")
        f.write(f"Data file: {data_path}\n")
        f.write(f"Rows: {n_rows}  Scenarios: {n_scenarios}\n")
        if "is_stable" in df.columns and has_scenario:
            ss = df.groupby("scenario_id")["is_stable"].first()
            f.write(f"Stable scenarios: {int(ss.sum())}  Unstable: {len(ss) - int(ss.sum())}\n")
        if report_estimated_cct is not None:
            f.write(
                f"Estimated CCT (from stable/unstable boundary): {report_estimated_cct:.3f} s\n"
            )
        f.write("\nFigures (generated with --plot):\n")
        f.write("  - multimachine_stability_pie.png       Stable vs unstable scenario share\n")
        f.write(
            "  - multimachine_max_angle_hist.png      Max rotor angle per scenario (180 deg = loss of sync)\n"
        )
        # f.write("  - multimachine_max_omega_deviation_hist.png  Max frequency deviation per scenario\n")  # omega: commented (rotor angle focus)
        f.write("  - multimachine_parameter_coverage.png  H, D, Pm, load distribution\n")
        f.write(
            "  - multimachine_sample_trajectories_stable.png   Stable trajectories (CCT in title)\n"
        )
        f.write(
            "  - multimachine_sample_trajectories_unstable.png Unstable trajectories (CCT in title)\n"
        )
        f.write(
            "  - multimachine_cct_distribution.png    CCT histogram and box (when param_cct_absolute exists)\n"
        )
        f.write(
            "  - multimachine_cct_vs_parameters.png   CCT vs load/H/D (when param_cct_absolute exists)\n"
        )
        f.write("  - trajectory_statistics.csv  Per-scenario stats (max angle, params)\n")
        if issues:
            f.write("\nRecommendations:\n")
            for issue in issues:
                f.write(f"  - {issue}\n")
            f.write(
                "  - Enable load sweep in config: alpha_range: [0.7, 1.2, 6] for varied operating points.\n"
            )
            f.write(
                "  - Ensure additional_clearing_time_offsets include negative values (below CCT) for stable cases.\n"
            )
            f.write(
                "  - Check CCT finding: if many failures, relax tolerance or fault/reactance.\n"
            )
        f.write("\nNext: preprocess_data.py then training.\n")
    print(f"Report saved: {report_path}")

    # ----- TRAJECTORY STATISTICS CSV (align with SMIB analyze_data.py) -----
    if has_scenario and "scenario_id" in df.columns:
        max_angles_series = _max_angle_per_scenario(df)
        traj_stats = []
        for sid in df["scenario_id"].unique():
            traj = df[df["scenario_id"] == sid]
            row = {"scenario_id": sid}
            if "is_stable" in traj.columns:
                row["is_stable"] = traj["is_stable"].iloc[0]
            if max_angles_series is not None and sid in max_angles_series.index:
                row["max_delta_deg"] = max_angles_series[sid]
            elif "delta_deg" in traj.columns:
                row["max_delta_deg"] = traj["delta_deg"].max()
            # [Rotor angle focus: omega stats commented out]
            # if "omega_deviation" in traj.columns:
            #     row["max_omega_deviation"] = traj["omega_deviation"].abs().max()
            param_cols = ["param_H", "param_D", "param_Pm", "param_load", "param_tc"]
            param_cols += [
                c
                for c in traj.columns
                if c.startswith("param_H_")
                or c.startswith("param_D_")
                or c.startswith("param_Pm_")
                or c.startswith("param_load_")
            ]
            for col in param_cols:
                if col in traj.columns:
                    row[col] = traj[col].iloc[0]
            traj_stats.append(row)
        if traj_stats:
            stats_df = pd.DataFrame(traj_stats)
            stats_path = output_dir / "trajectory_statistics.csv"
            stats_df.to_csv(stats_path, index=False)
            print(f"Trajectory statistics saved: {stats_path}")

    # ----- PLOTS -----
    if plot and MATPLOTLIB_AVAILABLE and has_scenario:
        _plot_stability_pie(df, output_dir)
        _plot_trajectory_quality(df, output_dir)
        # _plot_frequency_deviation(df, output_dir)  # omega: commented (rotor angle focus)
        _plot_parameter_coverage(df, output_dir)
        _plot_sample_trajectories(
            df,
            output_dir,
            n_per_type=n_sample_trajectories,
            max_time_s=simulation_time_s,
        )
        if "param_cct_absolute" in df.columns and df["param_cct_absolute"].notna().any():
            _plot_cct_distribution(df, output_dir)
            _plot_cct_vs_parameters(df, output_dir)
        print("Figures saved in:", output_dir)


def _plot_stability_pie(df: pd.DataFrame, output_dir: Path) -> None:
    """Stability distribution pie chart."""
    scenario_stability = df.groupby("scenario_id")["is_stable"].first()
    stable = scenario_stability.sum()
    unstable = len(scenario_stability) - stable
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        [stable, unstable],
        labels=["Stable", "Unstable"],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
    )
    ax.set_title("Stability distribution (scenarios)")
    fig.tight_layout()
    fig.savefig(output_dir / "multimachine_stability_pie.png", dpi=150, bbox_inches="tight")
    plt.close()


def _max_angle_per_scenario(df: pd.DataFrame):
    """Per-scenario max rotor angle (deg). Uses COI-relative (delta_rel_deg_*) for multimachine, else delta_deg."""
    if "scenario_id" not in df.columns:
        return None
    rel_cols = sorted(
        [
            c
            for c in df.columns
            if c.startswith("delta_rel_deg_") and c[len("delta_rel_deg_") :].isdigit()
        ],
        key=lambda x: int(x.split("_")[-1]),
    )
    if rel_cols:
        # Multimachine: max over time and over generators of |δ_rel_deg|
        per_scenario = df.groupby("scenario_id")
        max_angle = per_scenario[rel_cols].apply(lambda g: np.abs(g[rel_cols]).max().max())
        return max_angle
    if "delta_deg" in df.columns:
        return df.groupby("scenario_id")["delta_deg"].max()
    return None


def _plot_trajectory_quality(df: pd.DataFrame, output_dir: Path) -> None:
    """Max rotor angle per scenario (COI-relative for multimachine, else delta_deg)."""
    max_angle = _max_angle_per_scenario(df)
    if max_angle is None:
        return
    max_angle = max_angle.reindex(df["scenario_id"].unique()).dropna()
    if len(max_angle) == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(max_angle, bins=min(50, len(max_angle)), color="steelblue", edgecolor="white")
    ax.axvline(180, color="red", linestyle="--", label="180 deg")
    ax.set_xlabel("Max rotor angle (deg)")
    ax.set_ylabel("Number of scenarios")
    ax.set_title("Max rotor angle per scenario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "multimachine_max_angle_hist.png", dpi=150, bbox_inches="tight")
    plt.close()


# [Rotor angle focus: omega/frequency deviation plot commented out]
# def _plot_frequency_deviation(df: pd.DataFrame, output_dir: Path) -> None:
#     """Max |omega_deviation| per scenario (indicator of frequency swing severity)."""
#     if "omega_deviation" not in df.columns:
#         return
#     scenarios = df.groupby("scenario_id")
#     max_omega = scenarios["omega_deviation"].apply(lambda x: np.abs(x).max())
#     fig, ax = plt.subplots(figsize=(5, 3))
#     ax.hist(max_omega, bins=min(50, len(max_omega)), color="coral", edgecolor="white")
#     ax.set_xlabel("Max |omega deviation| (pu)")
#     ax.set_ylabel("Number of scenarios")
#     ax.set_title("Max frequency deviation per scenario")
#     fig.tight_layout()
#     fig.savefig(output_dir / "multimachine_max_omega_deviation_hist.png", dpi=150, bbox_inches="tight")
#     plt.close()


# Colors for generators (match reference figure: gen1–gen4)
_GEN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _get_delta_columns_for_plot(df: pd.DataFrame):
    """Return (list of column names for delta, use_relative_label, in_radians).
    Column naming aligned with run_kundur_fault_expt / plot_kundur_publication_figures when present.
    Prefer: delta_rel_gen1_rad, ... (fault-expt style) -> rad; then delta_deg_0, ...; delta_0, ...; else single delta_deg/delta.
    """
    # Fault-expt style: delta_rel_gen1_rad, delta_rel_gen2_rad, ... (same as tds_trajectories.csv from run_kundur_fault_expt)
    delta_rel_gen = sorted(
        [c for c in df.columns if c.startswith("delta_rel_gen") and c.endswith("_rad")],
        key=lambda x: int(x.replace("delta_rel_gen", "").replace("_rad", "") or 0),
    )
    if delta_rel_gen:
        return delta_rel_gen, True, True
    delta_gen_rad = sorted(
        [
            c
            for c in df.columns
            if c.startswith("delta_gen") and not c.startswith("delta_rel_") and c.endswith("_rad")
        ],
        key=lambda x: int(x.replace("delta_gen", "").replace("_rad", "") or 0),
    )
    if delta_gen_rad:
        return delta_gen_rad, False, True
    # COI-relative from parameter_sweep: delta_rel_deg_0, delta_rel_deg_1, ... (prefer for multimachine TSA)
    delta_rel_deg = sorted(
        [
            c
            for c in df.columns
            if c.startswith("delta_rel_deg_") and c[len("delta_rel_deg_") :].isdigit()
        ],
        key=lambda x: int(x.split("_")[-1]),
    )
    if delta_rel_deg:
        return delta_rel_deg, True, False
    # Per-generator delta_deg_0, delta_deg_1, ...
    deg_cols = sorted(
        [
            c
            for c in df.columns
            if c.startswith("delta_deg_")
            and not c.startswith("delta_rel_")
            and c[len("delta_deg_") :].isdigit()
        ],
        key=lambda x: int(x.split("_")[-1]),
    )
    if deg_cols:
        return deg_cols, True, False
    # Per-generator delta_0, delta_1, ... (radians; exclude delta_deg_*)
    rad_cols = sorted(
        [
            c
            for c in df.columns
            if c.startswith("delta_")
            and c != "delta"
            and not c.startswith("delta_deg_")
            and c.split("_")[-1].isdigit()
            and len(c.split("_")) == 2
        ],
        key=lambda x: int(x.split("_")[-1]),
    )
    if rad_cols:
        return rad_cols, True, True
    # Single delta_deg or delta
    if "delta_deg" in df.columns:
        return ["delta_deg"], False, False
    if "delta" in df.columns:
        return ["delta"], False, True
    return [], False, False


def _plot_sample_trajectories(
    df: pd.DataFrame,
    output_dir: Path,
    n_per_type: int = 5,
    random_state: int = 42,
    max_time_s: float = 5.0,
) -> None:
    """Plot stable and unstable rotor angle trajectories in two separate figures.
    Each subplot title includes CCT when available. Time axis runs from 0 to max_time_s.
    Saves:
    - multimachine_sample_trajectories_stable.png
    - multimachine_sample_trajectories_unstable.png
    """
    time_col = "time_s" if "time_s" in df.columns else "time"
    if "scenario_id" not in df.columns or time_col not in df.columns:
        return
    delta_cols, use_relative_label, in_radians = _get_delta_columns_for_plot(df)
    if not delta_cols:
        return
    to_deg = np.rad2deg if in_radians else (lambda x: x)
    rng = np.random.default_rng(random_state)
    stable_pool = (
        df[df["is_stable"] == True]["scenario_id"].unique()
        if "is_stable" in df.columns
        else np.array([])
    )
    unstable_pool = (
        df[df["is_stable"] == False]["scenario_id"].unique()
        if "is_stable" in df.columns
        else np.array([])
    )
    n_stable = min(n_per_type, len(stable_pool))
    n_unstable = min(n_per_type, len(unstable_pool))
    if stable_pool.size or unstable_pool.size:
        print(
            f"Sample trajectories: up to {n_per_type} per type "
            f"(data: {len(stable_pool)} stable, {len(unstable_pool)} unstable -> plotting {n_stable} stable, {n_unstable} unstable)"
        )
    stable_ids = list(rng.choice(stable_pool, size=n_stable, replace=False) if n_stable else [])
    unstable_ids = list(
        rng.choice(unstable_pool, size=n_unstable, replace=False) if n_unstable else []
    )
    scenarios_first = df.groupby("scenario_id").first()

    # Estimated CCT from stable/unstable boundary; used as fallback when a scenario has no CCT
    # so every subplot title can show a CCT for the reader
    estimated_cct = None
    if "is_stable" in df.columns and "param_tc" in df.columns:
        scenarios = df.groupby("scenario_id").first()
        tc_col = "param_tc"
        stable_tc = scenarios.loc[scenarios["is_stable"], tc_col].dropna()
        unstable_tc = scenarios.loc[~scenarios["is_stable"], tc_col].dropna()
        if len(stable_tc) > 0 and len(unstable_tc) > 0:
            estimated_cct = (float(stable_tc.max()) + float(unstable_tc.min())) / 2.0
        elif len(stable_tc) > 0:
            estimated_cct = float(stable_tc.max())

    def _draw_one_figure(scenario_ids, label_type, suptitle, filename):
        if not scenario_ids:
            return
        n_rows = len(scenario_ids)
        fig, axes = plt.subplots(n_rows, 1, figsize=(6, 2.2 * n_rows))
        if n_rows == 1:
            axes = [axes]
        for idx, sid in enumerate(scenario_ids):
            traj = df[df["scenario_id"] == sid].sort_values(time_col)
            t = traj[time_col].values
            ax = axes[idx]
            fault_start = None
            if "tf" in traj.columns and pd.notna(traj["tf"].iloc[0]):
                fault_start = float(traj["tf"].iloc[0])
            elif "param_tf" in traj.columns and pd.notna(traj["param_tf"].iloc[0]):
                fault_start = float(traj["param_tf"].iloc[0])
            else:
                fault_start = 1.0
            fault_clear = None
            if "param_tc" in traj.columns and pd.notna(traj["param_tc"].iloc[0]):
                fault_clear = float(traj["param_tc"].iloc[0])
            elif "tc" in traj.columns and pd.notna(traj["tc"].iloc[0]):
                fault_clear = float(traj["tc"].iloc[0])
            fault_bus = None
            if "param_fault_bus" in traj.columns and pd.notna(traj["param_fault_bus"].iloc[0]):
                try:
                    fault_bus = int(traj["param_fault_bus"].iloc[0])
                except (ValueError, TypeError):
                    fault_bus = None
            cct_s = None
            if "param_cct_absolute" in traj.columns and pd.notna(
                traj["param_cct_absolute"].iloc[0]
            ):
                cct_s = float(traj["param_cct_absolute"].iloc[0])
            elif "param_cct_duration" in traj.columns and pd.notna(
                traj["param_cct_duration"].iloc[0]
            ):
                cct_s = float(traj["param_cct_duration"].iloc[0])
            if cct_s is None and estimated_cct is not None:
                cct_s = estimated_cct  # Use estimated CCT so title/legend always show a CCT
            for i, col in enumerate(delta_cols):
                y = traj[col].values
                if in_radians:
                    y = to_deg(y)
                if col.startswith("delta_rel_gen") or col.startswith("delta_gen"):
                    gen_label = (
                        col.replace("delta_rel_", "").replace("delta_", "").replace("_rad", "")
                    )
                else:
                    gen_label = (
                        f"gen{i + 1}"
                        if len(delta_cols) > 1
                        else (
                            r"$\delta - \delta_{\mathrm{COI}}$ (deg)"
                            if use_relative_label
                            else "delta (deg)"
                        )
                    )
                ax.plot(t, y, color=_GEN_COLORS[i % len(_GEN_COLORS)], lw=1.5, label=gen_label)
            ax.axhline(
                180, color="gray", linestyle="-.", linewidth=1, alpha=0.8, label="_nolegend_"
            )
            ax.axhline(
                -180, color="gray", linestyle="-.", linewidth=1, alpha=0.8, label="_nolegend_"
            )
            loc_str = f" (bus {fault_bus})" if fault_bus is not None else ""
            if fault_start is not None:
                ax.axvline(
                    fault_start,
                    color="#c0392b",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                    label=f"fault on{loc_str}",
                )
            if fault_clear is not None:
                ax.axvline(
                    fault_clear,
                    color="#27ae60",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.9,
                    label=f"fault clear{loc_str}",
                )
            ax.set_ylabel(
                r"$\delta - \delta_{\mathrm{COI}}$ (deg)"
                if use_relative_label
                else "Rotor angle (deg)"
            )
            ax.set_xlabel("Time (s)" if idx == n_rows - 1 else "")
            ax.set_ylim(-360, 360)
            ax.set_xlim(0, max_time_s)
            # Place legend outside plot area (right); CCT is in subplot title only
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize=7,
                framealpha=0.95,
                borderaxespad=0,
            )
            # Title: id, Stable/Unstable, CCT (for this scenario), H, D, tc, load/Pm
            info_parts = [f"id={sid}", label_type]
            if cct_s is not None:
                info_parts.append(f"CCT={cct_s:.3f}s")
            if sid in scenarios_first.index:
                row = scenarios_first.loc[sid]
                if "param_H" in row and pd.notna(row.get("param_H")):
                    info_parts.append("H={:.2f}".format(float(row["param_H"])))
                if "param_D" in row and pd.notna(row.get("param_D")):
                    info_parts.append("D={:.2f}".format(float(row["param_D"])))
                if "param_tc" in row and pd.notna(row.get("param_tc")):
                    info_parts.append("tc={:.3f}s".format(float(row["param_tc"])))
                if "param_load" in row and pd.notna(row.get("param_load")):
                    info_parts.append("load={:.2f}".format(float(row["param_load"])))
                elif "param_Pm" in row and pd.notna(row.get("param_Pm")):
                    info_parts.append("Pm={:.2f}".format(float(row["param_Pm"])))
            ax.set_title("  |  ".join(info_parts), fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--")
        fig.suptitle(suptitle, fontsize=10)
        # Leave right margin for legends placed outside axes (bbox_to_anchor=(1.02, 1))
        fig.tight_layout(rect=[0, 0, 0.78, 0.97])
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    _draw_one_figure(
        stable_ids,
        "Stable",
        "Time-domain: relative rotor angle - stable sample trajectories"
        if use_relative_label
        else "Stable sample trajectories (rotor angle)",
        "multimachine_sample_trajectories_stable.png",
    )
    _draw_one_figure(
        unstable_ids,
        "Unstable",
        "Time-domain: relative rotor angle - unstable sample trajectories"
        if use_relative_label
        else "Unstable sample trajectories (rotor angle)",
        "multimachine_sample_trajectories_unstable.png",
    )


def _plot_parameter_coverage(df: pd.DataFrame, output_dir: Path) -> None:
    """Parameter ranges: per-generator and per-load when available, plus aggregated."""
    if "scenario_id" not in df.columns:
        return
    scenarios = df.groupby("scenario_id").first()

    def _per_unit_cols(prefix):
        return sorted(
            [
                c
                for c in scenarios.columns
                if c.startswith(prefix + "_") and c[len(prefix) + 1 :].isdigit()
            ],
            key=lambda x: int(x.split("_")[-1]),
        )

    # Build list of (column, label) for plotting: per-unit then single/aggregated
    plot_items = []  # (col, label)
    for param_prefix, label_prefix in [
        ("param_H", "H"),
        ("param_D", "D"),
        ("param_Pm", "Pm"),
        ("param_load", "load"),
    ]:
        per_unit = _per_unit_cols(param_prefix)
        if per_unit:
            for col in per_unit:
                unit_idx = col.split("_")[-1]
                plot_items.append((col, f"{label_prefix}_{unit_idx}"))
            # aggregated as last of this group (min/max over all units)
            plot_items.append((per_unit, f"{label_prefix} (agg)"))
        elif param_prefix in scenarios.columns and scenarios[param_prefix].notna().any():
            plot_items.append((param_prefix, label_prefix))

    if not plot_items:
        return

    # Flatten: for aggregated we pass list of cols, we'll compute combined series
    n_axes = len(plot_items)
    n_cols = min(4, n_axes)
    n_rows = (n_axes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.8 * n_rows))
    if n_axes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for idx, (col_or_cols, label) in enumerate(plot_items):
        if idx >= len(axes):
            break
        ax = axes[idx]
        if isinstance(col_or_cols, list):
            v = pd.concat([scenarios[c].dropna() for c in col_or_cols], ignore_index=True)
        else:
            v = scenarios[col_or_cols].dropna()
        if len(v) > 0:
            ax.hist(v, bins=min(30, v.nunique() or 1), color="steelblue", edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Parameter coverage (per generator/load and aggregated)")
    fig.tight_layout()
    fig.savefig(output_dir / "multimachine_parameter_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_cct_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """CCT distribution (histogram and box) when param_cct_absolute exists."""
    if "param_cct_absolute" not in df.columns or "scenario_id" not in df.columns:
        return
    scenarios = df.groupby("scenario_id").first()
    cct = scenarios["param_cct_absolute"].dropna()
    if len(cct) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].hist(cct, bins=min(30, len(cct)), color="steelblue", edgecolor="white")
    axes[0].axvline(
        cct.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {cct.mean():.3f} s"
    )
    axes[0].set_xlabel("CCT (s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("CCT distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].boxplot(cct, vert=True, patch_artist=True)
    axes[1].set_ylabel("CCT (s)")
    axes[1].set_title("CCT box plot")
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "multimachine_cct_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_cct_vs_parameters(df: pd.DataFrame, output_dir: Path) -> None:
    """CCT vs load (or Pm), H, and D when param_cct_absolute exists."""
    if "param_cct_absolute" not in df.columns or "scenario_id" not in df.columns:
        return
    scenarios = df.groupby("scenario_id").first()
    op_col = (
        "param_load"
        if ("param_load" in scenarios.columns and scenarios["param_load"].notna().any())
        else "param_Pm"
    )
    param_cols = ["param_H", "param_D", op_col]
    if not all(c in scenarios.columns for c in param_cols):
        return
    cct_data = scenarios[param_cols + ["param_cct_absolute"]].dropna()
    if len(cct_data) < 2:
        return
    labels = {"param_H": "H (s)", "param_D": "D (pu)", "param_load": "Load", "param_Pm": "Pm (pu)"}
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    for idx, col in enumerate(param_cols):
        axes[idx].scatter(
            cct_data[col],
            cct_data["param_cct_absolute"],
            alpha=0.6,
            s=25,
            edgecolors="black",
            linewidth=0.5,
        )
        axes[idx].set_xlabel(labels.get(col, col))
        axes[idx].set_ylabel("CCT (s)")
        axes[idx].set_title(f"CCT vs {labels.get(col, col)}")
        axes[idx].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "multimachine_cct_vs_parameters.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multimachine (Kundur) trajectory data before training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "data_file",
        nargs="?",
        type=str,
        default=None,
        help="Path to parameter_sweep_data_*.csv (or glob pattern)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for report and figures (default: <data_dir>/analysis)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate stability and trajectory quality figures",
    )
    parser.add_argument(
        "--num-machines",
        type=int,
        default=None,
        help="Number of machines (optional; inferred from columns if not set)",
    )
    parser.add_argument(
        "--sample-trajectories",
        type=int,
        default=5,
        metavar="N",
        help="Max number of stable and unstable sample trajectories to plot (default: 5)",
    )
    parser.add_argument(
        "--simulation-time",
        type=float,
        default=5.0,
        metavar="S",
        help="Full simulation time in seconds; trajectory plot x-axis 0 to S (default: 5.0)",
    )
    args = parser.parse_args()

    # Resolve data file
    if args.data_file:
        data_path = Path(args.data_file)
        if "*" in str(data_path):
            matches = list(data_path.parent.glob(data_path.name))
            if not matches:
                print(f"No files matching: {args.data_file}")
                sys.exit(1)
            data_path = max(matches, key=lambda p: p.stat().st_mtime)
        else:
            data_path = Path(args.data_file)
    else:
        # Default: latest in data/multimachine/kundur
        default_dir = PROJECT_ROOT / "data" / "multimachine" / "kundur"
        if not default_dir.exists():
            print("No data file given and data/multimachine/kundur not found.")
            print(
                "Usage: python scripts/analyze_multimachine_data.py <path_to_parameter_sweep_data_*.csv>"
            )
            sys.exit(1)
        matches = list(default_dir.glob("parameter_sweep_data_*.csv"))
        if not matches:
            print(f"No parameter_sweep_data_*.csv in {default_dir}")
            sys.exit(1)
        data_path = max(matches, key=lambda p: p.stat().st_mtime)
        print(f"Using latest file: {data_path.name}")

    if not data_path.exists():
        print(f"File not found: {data_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    analyze_multimachine_data(
        data_path=data_path,
        output_dir=output_dir,
        plot=args.plot,
        num_machines=args.num_machines,
        n_sample_trajectories=args.sample_trajectories,
        simulation_time_s=args.simulation_time,
    )


if __name__ == "__main__":
    main()
