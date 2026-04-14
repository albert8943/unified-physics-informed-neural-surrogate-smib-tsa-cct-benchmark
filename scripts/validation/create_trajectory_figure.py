#!/usr/bin/env python
"""
Create Trajectory Comparison Figure for GENROU Validation.

Generates publication-ready trajectory comparison plots showing PINN predictions
vs GENROU ground truth for representative scenarios.

Usage:
    python scripts/validation/create_trajectory_figure.py \
        --pinn-model validation/delta_weight_sweep/exp_delta20.0_omega40.0/exp_20260116_014607/pinn/model/best_model_20260116_020404.pth \
        --genrou-case test_cases/SMIB_genrou.json \
        --results outputs/publication/genrou_validation/exp_YYYYMMDD_HHMMSS/results/genrou_validation_results.json \
        --output-dir outputs/publication/genrou_validation/exp_YYYYMMDD_HHMMSS/figures \
        --n-scenarios 3 \
        --dpi 300
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix encoding for Windows
if sys.platform == "win32":
    try:
        import io

        if not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Set publication-quality defaults
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.linewidth"] = 1.5
matplotlib.rcParams["lines.linewidth"] = 2.0
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.alpha"] = 0.3

from evaluation.genrou_validation import validate_pinn_on_genrou
from scripts.core.utils import load_config


def select_representative_scenarios(results: List[Dict], n_scenarios: int = 3) -> List[Dict]:
    """
    Select representative scenarios (best, worst, medium performance).

    Parameters:
    -----------
    results : list
        List of validation results
    n_scenarios : int
        Number of scenarios to select

    Returns:
    --------
    selected : list
        Selected scenarios with indices
    """
    # Sort by delta_r2
    sorted_results = sorted(
        results,
        key=lambda x: x.get("delta_r2", -999) if not np.isnan(x.get("delta_r2", -999)) else -999,
    )

    selected = []
    if len(sorted_results) >= n_scenarios:
        # Best, middle, worst
        indices = [0, len(sorted_results) // 2, len(sorted_results) - 1]
        for idx in indices[:n_scenarios]:
            selected.append(
                {
                    "index": idx,
                    "result": sorted_results[idx],
                    "performance": "best"
                    if idx == 0
                    else ("worst" if idx == len(sorted_results) - 1 else "medium"),
                }
            )
    else:
        # Use all available
        for idx, result in enumerate(sorted_results):
            selected.append(
                {
                    "index": idx,
                    "result": result,
                    "performance": "all",
                }
            )

    return selected


def extract_trajectories_for_scenario(
    scenario: Dict,
    pinn_model_path: str,
    genrou_case_file: str,
) -> Optional[Dict]:
    """
    Extract trajectories for a single scenario by re-running validation.

    Parameters:
    -----------
    scenario : dict
        Scenario parameters
    pinn_model_path : str
        Path to PINN model
    genrou_case_file : str
        Path to GENROU case file

    Returns:
    --------
    trajectories : dict or None
        Dictionary with time, genrou_delta, genrou_omega, pinn_delta, pinn_omega
    """
    try:
        # Import here to avoid circular imports
        from evaluation.genrou_validation import validate_pinn_on_genrou

        # Run validation for this single scenario
        results = validate_pinn_on_genrou(
            pinn_model_path=pinn_model_path,
            genrou_case_file=genrou_case_file,
            test_scenarios=[scenario],
            device="cpu",
        )

        if len(results) == 0:
            return None

        # Note: The current validation function doesn't return trajectories
        # This is a placeholder - trajectories would need to be extracted during validation
        # For now, we'll create a note that trajectories need to be saved during validation
        return {
            "scenario": scenario,
            "note": "Trajectories need to be extracted during validation. This requires modifying genrou_validation.py to save trajectory data.",
        }
    except Exception as e:
        print(f"  ⚠️  Warning: Could not extract trajectories: {e}")
        return None


def create_trajectory_figure_from_data(
    trajectory_data: List[Dict],
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """
    Create trajectory comparison figure from trajectory data.

    Parameters:
    -----------
    trajectory_data : list
        List of trajectory dictionaries
    output_dir : Path
        Output directory
    dpi : int
        Figure resolution
    """
    n_scenarios = len(trajectory_data)

    # Create figure: n_scenarios rows × 2 columns (delta, omega)
    fig, axes = plt.subplots(n_scenarios, 2, figsize=(12, 4 * n_scenarios))

    if n_scenarios == 1:
        axes = axes.reshape(1, -1)

    for i, traj_data in enumerate(trajectory_data):
        scenario = traj_data["scenario"]
        time = traj_data.get("time", np.array([]))
        genrou_delta = traj_data.get("genrou_delta", np.array([]))
        genrou_omega = traj_data.get("genrou_omega", np.array([]))
        pinn_delta = traj_data.get("pinn_delta", np.array([]))
        pinn_omega = traj_data.get("pinn_omega", np.array([]))
        tf = scenario.get("tf", 1.0)
        tc = scenario.get("tc", 1.2)

        # Delta plot
        ax = axes[i, 0]
        if len(time) > 0 and len(genrou_delta) > 0:
            ax.plot(time, genrou_delta, label="GENROU (Ground Truth)", linewidth=2, color="#1f77b4")
        if len(time) > 0 and len(pinn_delta) > 0:
            ax.plot(
                time,
                pinn_delta,
                label="PINN (Prediction)",
                linewidth=2,
                linestyle="--",
                color="#ff7f0e",
            )
        ax.axvline(
            x=tf, color="green", linestyle=":", linewidth=1.5, alpha=0.7, label="Fault Start"
        )
        ax.axvline(
            x=tc, color="orange", linestyle=":", linewidth=1.5, alpha=0.7, label="Fault Clear"
        )
        ax.set_xlabel("Time (s)", fontweight="bold")
        ax.set_ylabel("Rotor Angle δ (rad)", fontweight="bold")
        ax.set_title(
            f"Scenario {scenario.get('scenario_id', i+1)} - Rotor Angle", fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Omega plot
        ax = axes[i, 1]
        if len(time) > 0 and len(genrou_omega) > 0:
            ax.plot(time, genrou_omega, label="GENROU (Ground Truth)", linewidth=2, color="#1f77b4")
        if len(time) > 0 and len(pinn_omega) > 0:
            ax.plot(
                time,
                pinn_omega,
                label="PINN (Prediction)",
                linewidth=2,
                linestyle="--",
                color="#ff7f0e",
            )
        ax.axvline(
            x=tf, color="green", linestyle=":", linewidth=1.5, alpha=0.7, label="Fault Start"
        )
        ax.axvline(
            x=tc, color="orange", linestyle=":", linewidth=1.5, alpha=0.7, label="Fault Clear"
        )
        ax.set_xlabel("Time (s)", fontweight="bold")
        ax.set_ylabel("Rotor Speed ω (pu)", fontweight="bold")
        ax.set_title(
            f"Scenario {scenario.get('scenario_id', i+1)} - Rotor Speed", fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_file = output_dir / "trajectory_comparison.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"✓ Saved: {output_file}")

    output_file_pdf = output_dir / "trajectory_comparison.pdf"
    plt.savefig(output_file_pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"✓ Saved: {output_file_pdf}")

    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create trajectory comparison figure")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to GENROU validation results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for figures",
    )
    parser.add_argument(
        "--pinn-model",
        type=str,
        help="Path to PINN model (required if extracting trajectories)",
    )
    parser.add_argument(
        "--genrou-case",
        type=str,
        help="Path to GENROU case file (required if extracting trajectories)",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=3,
        help="Number of representative scenarios to plot (default: 3)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution (default: 300)",
    )
    parser.add_argument(
        "--trajectory-data",
        type=str,
        help="Path to trajectory data JSON (if trajectories were saved during validation)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GENROU VALIDATION - TRAJECTORY FIGURE")
    print("=" * 70)

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r") as f:
        results = json.load(f)

    print(f"✓ Loaded {len(results)} validation results")

    # Select representative scenarios
    selected = select_representative_scenarios(results, n_scenarios=args.n_scenarios)
    print(f"✓ Selected {len(selected)} representative scenarios")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if trajectory data file exists
    if args.trajectory_data and Path(args.trajectory_data).exists():
        # Load trajectory data
        with open(args.trajectory_data, "r") as f:
            trajectory_data = json.load(f)
        print(f"✓ Loaded trajectory data from: {args.trajectory_data}")
        create_trajectory_figure_from_data(trajectory_data, output_dir, dpi=args.dpi)
    else:
        # Note: Trajectories need to be extracted during validation
        # For now, create a placeholder note
        print("\n⚠️  Note: Trajectory data not provided.")
        print(
            "   To create trajectory comparison figure, trajectories must be saved during validation."
        )
        print("   This requires modifying genrou_validation.py to save trajectory data.")
        print("   Alternatively, re-run validation with trajectory saving enabled.")
        print("\n   Creating placeholder note...")

        note_file = output_dir / "TRAJECTORY_FIGURE_NOTE.md"
        with open(note_file, "w") as f:
            f.write(
                """# Trajectory Comparison Figure

## Status: Requires Trajectory Data

To create the trajectory comparison figure, trajectory data must be extracted during validation.

### Option 1: Modify Validation Script
Modify `evaluation/genrou_validation.py` to save trajectory data (time, delta, omega) for each scenario.

### Option 2: Re-run Selected Scenarios
Re-run validation for selected scenarios and save trajectories.

### Selected Scenarios for Figure:
"""
            )
            for sel in selected:
                scenario = sel["result"]["scenario"]
                f.write(
                    f"- Scenario {scenario.get('scenario_id', 'N/A')}: {sel['performance']} performance\n"
                )
                f.write(f"  - R² Delta: {sel['result'].get('delta_r2', 'N/A'):.4f}\n")
                f.write(
                    f"  - Parameters: H={scenario.get('H', 'N/A')}, D={scenario.get('D', 'N/A')}, Pm={scenario.get('Pm', 'N/A')}\n"
                )

        print(f"✓ Note saved to: {note_file}")


if __name__ == "__main__":
    main()
