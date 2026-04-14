"""
Generate publication-quality figures from Kundur power flow and TDS results.

Reads CSV/outputs from outputs/kundur_simulations/exp_YYYYMMDD_HHMMSS/ (or --exp-dir).
By default saves PNG only (for experiments and cross-checking). Uncomment the PDF/EPS
blocks in each plot function when preparing final figures for journal submission.

Usage:
  python scripts/plot_kundur_publication_figures.py
  python scripts/plot_kundur_publication_figures.py --exp-dir outputs/kundur_simulations/exp_20260213_104547
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Publication-style defaults (aligned with visualization/publication_figures.py)
matplotlib.rcParams.update(
    {
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.linewidth": 1.2,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
    }
)
# Colorblind-friendly
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = PROJECT_ROOT / "outputs" / "kundur_simulations"


def _get_fault_info(
    exp_dir: Path,
) -> tuple[Optional[float], Optional[float], Optional[int]]:
    """Read fault_start_t_s, fault_clear_t_s, and fault_bus from run_info.txt if present.
    Returns (start, clear, fault_bus) or (None, None, None)."""
    run_info = exp_dir / "run_info.txt"
    if not run_info.exists():
        return None, None, None
    fault_start = fault_clear = None
    fault_bus: Optional[int] = None
    try:
        for line in run_info.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("fault_start_t_s:"):
                fault_start = float(line.split(":", 1)[1].strip())
            elif line.startswith("fault_clear_t_s:"):
                fault_clear = float(line.split(":", 1)[1].strip())
            elif line.startswith("fault_bus:"):
                fault_bus = int(line.split(":", 1)[1].strip())
    except (ValueError, OSError):
        pass
    return fault_start, fault_clear, fault_bus


def _draw_fault_lines(
    ax,
    fault_start: Optional[float],
    fault_clear: Optional[float],
    fault_bus: Optional[int] = None,
    legend_on_ax: bool = False,
) -> None:
    """Draw vertical lines at fault application and clearing time. Call for each subplot in TDS figures."""
    if fault_start is None and fault_clear is None:
        return
    loc_str = f" (bus {fault_bus})" if fault_bus is not None else ""
    if fault_start is not None:
        ax.axvline(
            fault_start,
            color="#c0392b",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            label=("fault on" + loc_str) if legend_on_ax else "",
        )
    if fault_clear is not None:
        ax.axvline(
            fault_clear,
            color="#27ae60",
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            label=("fault clear" + loc_str) if legend_on_ax else "",
        )
    if legend_on_ax and (fault_start is not None or fault_clear is not None):
        ax.legend(loc="upper right", framealpha=0.9)


def _add_cct_to_legend(ax, cct_s: float, **legend_kw) -> None:
    """Add a 'CCT = X.XXX s' entry to the axis legend (e.g. for stable/unstable comparison figures).
    Any legend_kw (e.g. bbox_to_anchor, loc) are passed to ax.legend() to preserve placement."""
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(facecolor="none", edgecolor="none"))
    labels.append(f"CCT = {cct_s:.3f} s")
    kwargs = {"loc": "upper right", "framealpha": 0.9, **legend_kw}
    ax.legend(handles=handles, labels=labels, **kwargs)


def get_latest_exp_dir(base: Path) -> Optional[Path]:
    """Return latest exp_* directory under base, or None."""
    if not base.exists():
        return None
    dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("exp_")],
        reverse=True,
    )
    return dirs[0] if dirs else None


def plot_pf_voltage_profile(exp_dir: Path, out_dir: Path, dpi: int = 300) -> None:
    """Bar chart: bus voltage magnitude (pu) vs bus."""
    csv_path = exp_dir / "power_flow_bus.csv"
    if not csv_path.exists():
        print(f"  Skip PF voltage: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    if "v0" not in df.columns:
        print("  Skip PF voltage: no 'v0' column")
        return
    bus_label = df["name"].astype(str).values if "name" in df.columns else np.arange(1, len(df) + 1)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = np.arange(len(df))
    ax.bar(x, df["v0"], color=COLORS[0], edgecolor="gray", linewidth=0.5)
    ax.axhline(1.05, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bus_label)
    ax.set_xlabel("Bus number")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title("Power flow: bus voltage profile")
    ax.set_ylim(0.9, 1.12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    # PNG for experiments and cross-checking
    fig.savefig(out_dir / "pf_voltage_profile.png", dpi=dpi, bbox_inches="tight")
    # Uncomment for final journal submission (vector formats):
    # fig.savefig(out_dir / "pf_voltage_profile.pdf", bbox_inches="tight")
    # fig.savefig(out_dir / "pf_voltage_profile.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pf_voltage_profile.png -> {out_dir}")


def plot_pf_voltage_angle(exp_dir: Path, out_dir: Path, dpi: int = 300) -> None:
    """Bar chart: bus voltage angle (deg) vs bus."""
    csv_path = exp_dir / "power_flow_bus.csv"
    if not csv_path.exists() or "a0" not in pd.read_csv(csv_path, nrows=1).columns:
        print(f"  Skip PF angle: {csv_path} or a0 not found")
        return
    df = pd.read_csv(csv_path)
    bus_label = df["name"].astype(str).values if "name" in df.columns else np.arange(1, len(df) + 1)
    a0_deg = np.rad2deg(df["a0"])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = np.arange(len(df))
    ax.bar(x, a0_deg, color=COLORS[1], edgecolor="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(bus_label)
    ax.set_xlabel("Bus number")
    ax.set_ylabel("Voltage angle (deg)")
    ax.set_title("Power flow: bus voltage angle")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "pf_voltage_angle.png", dpi=dpi, bbox_inches="tight")
    # Uncomment for final journal submission (vector formats):
    # fig.savefig(out_dir / "pf_voltage_angle.pdf", bbox_inches="tight")
    # fig.savefig(out_dir / "pf_voltage_angle.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pf_voltage_angle.png -> {out_dir}")


def plot_pf_voltage_profile_and_angle(exp_dir: Path, out_dir: Path, dpi: int = 300) -> None:
    """One figure: (a) voltage profile (pu), (b) voltage angle (deg) — journal style."""
    csv_path = exp_dir / "power_flow_bus.csv"
    if not csv_path.exists():
        print(f"  Skip PF combined: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    if "v0" not in df.columns or "a0" not in df.columns:
        print("  Skip PF combined: need 'v0' and 'a0' columns")
        return
    bus_label = df["name"].astype(str).values if "name" in df.columns else np.arange(1, len(df) + 1)
    a0_deg = np.rad2deg(df["a0"])
    x = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2), sharex=True)
    # (a) Voltage magnitude
    ax1.bar(x, df["v0"], color=COLORS[0], edgecolor="gray", linewidth=0.5)
    ax1.axhline(1.05, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax1.axhline(0.95, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bus_label)
    ax1.set_xlabel("Bus number")
    ax1.set_ylabel("Voltage (pu)")
    ax1.set_ylim(0.9, 1.12)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax1.set_title("(a) Voltage profile", fontsize=11)
    # (b) Voltage angle
    ax2.bar(x, a0_deg, color=COLORS[1], edgecolor="gray", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bus_label)
    ax2.set_xlabel("Bus number")
    ax2.set_ylabel("Voltage angle (deg)")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax2.set_title("(b) Voltage angle", fontsize=11)

    fig.suptitle("Power flow: bus voltage", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "pf_voltage_profile_and_angle.png", dpi=dpi, bbox_inches="tight")
    # Uncomment for final journal submission (vector formats):
    # fig.savefig(out_dir / "pf_voltage_profile_and_angle.pdf", bbox_inches="tight")
    # fig.savefig(out_dir / "pf_voltage_profile_and_angle.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pf_voltage_profile_and_angle.png -> {out_dir}")


def plot_pf_bus_power(exp_dir: Path, out_dir: Path, dpi: int = 300) -> None:
    """Bar chart: load power (P and Q) at load buses only — original style (loads only, positive magnitude)."""
    csv_path = exp_dir / "power_flow_bus.csv"
    if not csv_path.exists():
        print(f"  Skip PF bus power: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    if "P_pu" not in df.columns or "Q_pu" not in df.columns:
        print(
            "  Skip PF bus power: no 'P_pu'/'Q_pu' columns (re-run run_kundur_expt.py to export bus P, Q)"
        )
        return
    # Only buses with load (negative P or non-zero Q typical for loads)
    load_mask = (df["P_pu"] < 0) | (df["Q_pu"].abs() > 1e-6)
    df_load = df.loc[load_mask].copy()
    if df_load.empty:
        print("  Skip PF bus power: no load buses (P_pu < 0 or |Q_pu| > 0)")
        return
    bus_label = (
        df_load["name"].astype(str).values
        if "name" in df_load.columns
        else np.arange(1, len(df_load) + 1)
    )
    x = np.arange(len(df_load))
    # Show load power as positive magnitude
    P_load = np.abs(df_load["P_pu"].values)
    Q_load = df_load["Q_pu"].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3), sharex=True)
    ax1.bar(x, P_load, color=COLORS[0], edgecolor="gray", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bus_label)
    ax1.set_xlabel("Bus number")
    ax1.set_ylabel("Load active power P (pu)")
    ax1.set_title("(a) Load active power", fontsize=11)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax2.bar(x, Q_load, color=COLORS[1], edgecolor="gray", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bus_label)
    ax2.set_xlabel("Bus number")
    ax2.set_ylabel("Load reactive power Q (pu)")
    ax2.set_title("(b) Load reactive power", fontsize=11)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)
    fig.suptitle("Power flow: load power per bus", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "pf_load_power_per_bus.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pf_load_power_per_bus.png -> {out_dir}")


def plot_tds_Pe_trajectories(exp_dir: Path, out_dir: Path, dpi: int = 300) -> None:
    """One subplot per generator: electrical power Pe(t) (pu) vs time (s), same layout as delta/omega."""
    csv_path = exp_dir / "tds_trajectories.csv"
    if not csv_path.exists():
        print(f"  Skip TDS Pe: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    if "time_s" not in df.columns:
        print("  Skip TDS Pe: no 'time_s' column")
        return
    t = df["time_s"].values
    pe_cols = [c for c in df.columns if c.startswith("Pe_gen")]
    if not pe_cols:
        print("  Skip TDS Pe: no Pe_gen* columns")
        return
    n_gen = len(pe_cols)
    fig, axes = plt.subplots(n_gen, 1, figsize=(5.5, 1.8 * n_gen), sharex=True)
    if n_gen == 1:
        axes = [axes]
    fault_start, fault_clear, fault_bus = _get_fault_info(exp_dir)
    for i, col in enumerate(pe_cols):
        ax = axes[i]
        label = col.replace("Pe_", "").replace("_pu", "")
        ax.plot(t, df[col], color=COLORS[i % len(COLORS)], label=label)
        ax.set_ylabel("Pe (pu)")
        ax.set_title(label, fontsize=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.6)
        _draw_fault_lines(ax, fault_start, fault_clear, fault_bus, legend_on_ax=(i == 0))
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Time-domain: generator electrical power", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "tds_Pe_trajectories.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved tds_Pe_trajectories.png -> {out_dir}")


def plot_tds_delta_trajectories(
    exp_dir: Path,
    out_dir: Path,
    dpi: int = 300,
    cct_s: Optional[float] = None,
    filename_suffix: Optional[str] = None,
) -> None:
    """Single plot: all generators' relative rotor angle (δ − δ_COI) in degrees; fallback to absolute δ.
    CSV stores angles in radians; we convert to degrees for plotting and set y-axis limit ±360 deg.
    If cct_s is set, add 'CCT = X.XXX s' to the legend. If filename_suffix is set, save as
    tds_delta_trajectories_{filename_suffix}.png (for stable/unstable comparison)."""
    csv_path = exp_dir / "tds_trajectories.csv"
    if not csv_path.exists():
        print(f"  Skip TDS delta: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    if "time_s" not in df.columns:
        return
    # Prefer relative rotor angles (standard in fault/transient studies)
    delta_rel_cols = [c for c in df.columns if c.startswith("delta_rel_gen")]
    delta_abs_cols = [
        c for c in df.columns if c.startswith("delta_gen") and not c.startswith("delta_rel_")
    ]
    delta_cols = delta_rel_cols if delta_rel_cols else delta_abs_cols
    delta_cols = sorted(delta_cols)  # ensure gen1, gen2, ... order for correct indexing
    if not delta_cols:
        print(
            "  Skip TDS delta: no delta_rel_gen* or delta_gen* columns (re-run experiment to export delta)"
        )
        return
    use_relative = bool(delta_rel_cols)
    t = df["time_s"].values
    fault_start, fault_clear, fault_bus = _get_fault_info(exp_dir)
    # CSV angles are in radians; convert to degrees for plot. Y-axis: deg, limit ±360.
    delta_rad = df[delta_cols].values
    delta_deg = np.rad2deg(delta_rad)

    # --- Single figure: all generator rotor angles on one axes (stable/unstable each get one figure) ---
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for i, col in enumerate(delta_cols):
        label = col.replace("delta_rel_", "").replace("delta_", "").replace("_rad", "")
        ax.plot(t, delta_deg[:, i], color=COLORS[i % len(COLORS)], label=label)
    ax.set_ylabel(r"$\delta - \delta_{\mathrm{COI}}$ (deg)" if use_relative else r"$\delta$ (deg)")
    ax.set_ylim(-360, 360)
    ax.set_xlim(0, 5)
    ax.set_xlabel("Time (s)")
    ax.set_title(
        "Time-domain: relative rotor angle" if use_relative else "Time-domain: rotor angle",
        fontsize=12,
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    _draw_fault_lines(ax, fault_start, fault_clear, fault_bus, legend_on_ax=True)
    ax.axhline(
        180,
        color="gray",
        linestyle="-.",
        linewidth=1,
        alpha=0.8,
        label="critical angle (±180°)",
    )
    ax.axhline(-180, color="gray", linestyle="-.", linewidth=1, alpha=0.8)
    # Legend below x-axis: enough offset so "Time (s)" and tick labels are fully visible
    _legend_kw = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, -0.58),
        "ncol": 2,
        "framealpha": 0.95,
        "fontsize": 9,
    }
    handles, labels = ax.get_legend_handles_labels()
    if cct_s is not None:
        handles.append(mpatches.Patch(facecolor="none", edgecolor="none"))
        labels.append(f"CCT = {cct_s:.3f} s")
    if fault_clear is not None:
        handles.append(mpatches.Patch(facecolor="none", edgecolor="none"))
        labels.append(f"Fault clearing time: {fault_clear:.3f} s")
    ax.legend(handles=handles, labels=labels, **_legend_kw)
    # Leave margin so y-axis tick labels (±250/360) and title aren't clipped; bottom for x-axis + legend
    fig.tight_layout(rect=(0.03, 0.32, 0.97, 0.96), pad=1.1)
    basename = (
        "tds_delta_trajectories.png"
        if not filename_suffix
        else f"tds_delta_trajectories_{filename_suffix}.png"
    )
    fig.savefig(out_dir / basename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {basename} -> {out_dir}")

    # --- (Commented) Original: one subplot per generator ---
    # n_gen = len(delta_cols)
    # fig, axes = plt.subplots(n_gen, 1, figsize=(5.5, 1.8 * n_gen), sharex=True)
    # if n_gen == 1:
    #     axes = [axes]
    # for i, col in enumerate(delta_cols):
    #     ax = axes[i]
    #     label = col.replace("delta_rel_", "").replace("delta_", "").replace("_rad", "")
    #     ax.plot(t, delta_deg[:, i], color=COLORS[i % len(COLORS)], label=label)
    #     ax.set_ylabel(
    #         r"$\delta - \delta_{\mathrm{COI}}$ (deg)" if use_relative else r"$\delta$ (deg)"
    #     )
    #     ax.set_ylim(-360, 360)
    #     ax.set_title(label, fontsize=10)
    #     ax.legend(loc="upper right", framealpha=0.9)
    #     ax.grid(True, linestyle="--", alpha=0.6)
    #     _draw_fault_lines(ax, fault_start, fault_clear, fault_bus, legend_on_ax=(i == 0))
    #     ax.axhline(180, color="gray", linestyle="-.", linewidth=1, alpha=0.8, label=("critical angle (±180°)" if i == 0 else ""))
    #     ax.axhline(-180, color="gray", linestyle="-.", linewidth=1, alpha=0.8)
    #     if i == 0:
    #         ax.legend(loc="upper right", framealpha=0.9)
    # if cct_s is not None:
    #     _add_cct_to_legend(axes[0], cct_s)
    # for ax in axes:
    #     ax.set_xlim(0, 5)
    # axes[-1].set_xlabel("Time (s)")
    # fig.suptitle(
    #     "Time-domain: relative rotor angle" if use_relative else "Time-domain: rotor angle",
    #     y=1.02,
    #     fontsize=12,
    # )
    # fig.tight_layout()
    # basename = "tds_delta_trajectories.png" if not filename_suffix else f"tds_delta_trajectories_{filename_suffix}.png"
    # fig.savefig(out_dir / basename, dpi=dpi, bbox_inches="tight")
    # plt.close(fig)
    # print(f"  Saved {basename} -> {out_dir}")


def plot_tds_omega_trajectories(exp_dir: Path, out_dir: Path, dpi: int = 300) -> None:
    """One subplot per generator: relative rotor speed (ω − ω_COI) in pu; fallback to absolute ω."""
    csv_path = exp_dir / "tds_trajectories.csv"
    if not csv_path.exists():
        print(f"  Skip TDS omega: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    if "time_s" not in df.columns:
        return
    # Prefer relative rotor speed (COI-referred, common in fault/transient studies)
    omega_rel_cols = [c for c in df.columns if c.startswith("omega_rel_gen")]
    omega_abs_cols = [
        c for c in df.columns if c.startswith("omega_gen") and not c.startswith("omega_rel_")
    ]
    omega_cols = omega_rel_cols if omega_rel_cols else omega_abs_cols
    if not omega_cols:
        print(
            "  Skip TDS omega: no omega_rel_gen* or omega_gen* columns (re-run experiment to export omega)"
        )
        return
    use_relative = bool(omega_rel_cols)
    t = df["time_s"].values
    n_gen = len(omega_cols)
    fig, axes = plt.subplots(n_gen, 1, figsize=(5.5, 1.8 * n_gen), sharex=True)
    if n_gen == 1:
        axes = [axes]
    fault_start, fault_clear, fault_bus = _get_fault_info(exp_dir)
    for i, col in enumerate(omega_cols):
        ax = axes[i]
        label = col.replace("omega_rel_", "").replace("omega_", "").replace("_pu", "")
        ax.plot(t, df[col], color=COLORS[i % len(COLORS)], label=label)
        ax.set_ylabel(
            r"$\omega - \omega_{\mathrm{COI}}$ (pu)" if use_relative else r"$\omega$ (pu)"
        )
        ax.set_title(label, fontsize=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.6)
        _draw_fault_lines(ax, fault_start, fault_clear, fault_bus, legend_on_ax=(i == 0))
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "Time-domain: relative rotor speed" if use_relative else "Time-domain: rotor speed",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "tds_omega_trajectories.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved tds_omega_trajectories.png -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Kundur PF/TDS publication figures from an exp directory."
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=None,
        help=f"Experiment directory (default: latest under {DEFAULT_BASE})",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG output")
    args = parser.parse_args()

    exp_dir = args.exp_dir
    if exp_dir is None:
        exp_dir = get_latest_exp_dir(DEFAULT_BASE)
        if exp_dir is None:
            print(f"No exp_* directory found under {DEFAULT_BASE}. Run run_kundur_expt.py first.")
            return
        print(f"Using latest exp dir: {exp_dir}")
    else:
        exp_dir = Path(exp_dir)
        if not exp_dir.is_dir():
            print(f"Not a directory: {exp_dir}")
            return

    out_dir = exp_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output figures -> {out_dir}\n")

    # Combined PF figure (voltage profile + angle in one) — journal style
    plot_pf_voltage_profile_and_angle(exp_dir, out_dir, dpi=args.dpi)
    plot_pf_bus_power(exp_dir, out_dir, dpi=args.dpi)
    # Separate PF figures (commented out; use combined figure for journal)
    # plot_pf_voltage_profile(exp_dir, out_dir, dpi=args.dpi)
    # plot_pf_voltage_angle(exp_dir, out_dir, dpi=args.dpi)
    plot_tds_Pe_trajectories(exp_dir, out_dir, dpi=args.dpi)
    plot_tds_delta_trajectories(exp_dir, out_dir, dpi=args.dpi)
    plot_tds_omega_trajectories(exp_dir, out_dir, dpi=args.dpi)

    print(
        "\nDone. PNG only (for experiments). Uncomment PDF/EPS in script for final journal figures."
    )


if __name__ == "__main__":
    main()
