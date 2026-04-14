"""
Model Approach Comparison Utility

Compare Pe Input vs Reactance Approaches.

This script compares the performance of Pe(t) input approach vs reactance-based approach
on the same test data.

Usage:
    python scripts/comparison/compare_approaches.py \
        --data-dir data \
        --model-pe-input model_pe.pth \
        --model-reactance model_reactance.pth
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pinn.trajectory_prediction import TrajectoryPredictionPINN, TrajectoryPredictionPINN_PeInput
from utils.metrics import compute_trajectory_metrics


def main():
    parser = argparse.ArgumentParser(description="Compare Pe Input vs Reactance Approaches")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing test data",
    )
    parser.add_argument(
        "--model-pe-input",
        type=str,
        required=True,
        help="Path to Pe input model checkpoint",
    )
    parser.add_argument(
        "--model-reactance",
        type=str,
        required=True,
        help="Path to reactance-based model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/comparison",
        help="Output directory for comparison results",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")

    args = parser.parse_args()

    # Setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Comparing Pe Input vs Reactance Approaches")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Load test data
    print("\nLoading test data...")
    test_data = load_test_data(args.data_dir)

    # Load models
    print("\nLoading models...")
    model_pe = load_pe_input_model(args.model_pe_input, device)
    model_reactance = load_reactance_model(args.model_reactance, device)

    # Evaluate both models
    print("\nEvaluating models...")
    results_pe = evaluate_model(model_pe, test_data, use_pe_input=True, device=device)
    results_reactance = evaluate_model(
        model_reactance, test_data, use_pe_input=False, device=device
    )

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print("\nPe Input Approach:")
    print(f"  Delta RMSE: {results_pe['delta_rmse']:.6f}")
    print(f"  Omega RMSE: {results_pe['omega_rmse']:.6f}")
    print(f"  Delta MAE: {results_pe['delta_mae']:.6f}")
    print(f"  Omega MAE: {results_pe['omega_mae']:.6f}")
    print(f"  Delta R²: {results_pe['delta_r2']:.6f}")
    print(f"  Omega R²: {results_pe['omega_r2']:.6f}")

    print("\nReactance-Based Approach:")
    print(f"  Delta RMSE: {results_reactance['delta_rmse']:.6f}")
    print(f"  Omega RMSE: {results_reactance['omega_rmse']:.6f}")
    print(f"  Delta MAE: {results_reactance['delta_mae']:.6f}")
    print(f"  Omega MAE: {results_reactance['omega_mae']:.6f}")
    print(f"  Delta R²: {results_reactance['delta_r2']:.6f}")
    print(f"  Omega R²: {results_reactance['omega_r2']:.6f}")

    # Save comparison report
    report_path = output_dir / "comparison_report.txt"
    save_comparison_report(results_pe, results_reactance, report_path)

    print(f"\n✓ Comparison report saved to: {report_path}")


def load_test_data(data_dir: str) -> pd.DataFrame:
    """Load test data."""
    data_path = Path(data_dir)
    if data_path.is_dir():
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        data_file = csv_files[0]
    else:
        data_file = data_path

    return pd.read_csv(data_file)


def load_pe_input_model(model_path: str, device: torch.device) -> TrajectoryPredictionPINN_PeInput:
    """Load Pe input model."""
    model = TrajectoryPredictionPINN_PeInput(input_dim=7, hidden_dims=[64, 64, 64, 64]).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_reactance_model(model_path: str, device: torch.device) -> TrajectoryPredictionPINN:
    """Load reactance-based model."""
    model = TrajectoryPredictionPINN(input_dim=11, hidden_dims=[64, 64, 64, 64]).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_model(
    model, test_data: pd.DataFrame, use_pe_input: bool, device: torch.device
) -> dict:
    """Evaluate model on test data."""
    all_delta_pred = []
    all_omega_pred = []
    all_delta_true = []
    all_omega_true = []

    # Evaluate on sample scenarios
    scenarios = test_data["scenario_id"].unique()[:10]  # Limit for speed

    for scenario_id in scenarios:
        scenario_data = test_data[test_data["scenario_id"] == scenario_id]

        t = scenario_data["time"].values
        delta0 = scenario_data["delta0"].iloc[0]
        omega0 = scenario_data["omega0"].iloc[0]
        H = scenario_data["H"].iloc[0]
        D = scenario_data["D"].iloc[0]
        Pm = scenario_data["Pm"].iloc[0]

        delta_true = scenario_data["delta"].values
        omega_true = scenario_data["omega"].values

        with torch.no_grad():
            if use_pe_input:
                Pe = scenario_data["Pe"].values
                delta_pred, omega_pred = model.predict(
                    t=t, delta0=delta0, omega0=omega0, H=H, D=D, Pm=Pm, Pe=Pe, device=str(device)
                )
            else:
                Xprefault = scenario_data["Xprefault"].iloc[0]
                Xfault = scenario_data["Xfault"].iloc[0]
                Xpostfault = scenario_data["Xpostfault"].iloc[0]
                tf = scenario_data["tf"].iloc[0]
                tc = scenario_data["tc"].iloc[0]
                delta_pred, omega_pred = model.predict(
                    t=t,
                    delta0=delta0,
                    omega0=omega0,
                    H=H,
                    D=D,
                    Pm=Pm,
                    Xprefault=Xprefault,
                    Xfault=Xfault,
                    Xpostfault=Xpostfault,
                    tf=tf,
                    tc=tc,
                    device=str(device),
                )

        all_delta_pred.extend(delta_pred)
        all_omega_pred.extend(omega_pred)
        all_delta_true.extend(delta_true)
        all_omega_true.extend(omega_true)

    # Compute metrics
    delta_rmse = np.sqrt(mean_squared_error(all_delta_true, all_delta_pred))
    omega_rmse = np.sqrt(mean_squared_error(all_omega_true, all_omega_pred))
    delta_mae = mean_absolute_error(all_delta_true, all_delta_pred)
    omega_mae = mean_absolute_error(all_omega_true, all_omega_pred)
    delta_r2 = r2_score(all_delta_true, all_delta_pred)
    omega_r2 = r2_score(all_omega_true, all_omega_pred)

    return {
        "delta_rmse": delta_rmse,
        "omega_rmse": omega_rmse,
        "delta_mae": delta_mae,
        "omega_mae": omega_mae,
        "delta_r2": delta_r2,
        "omega_r2": omega_r2,
    }


def save_comparison_report(results_pe: dict, results_reactance: dict, path: Path):
    """Save comparison report to file."""
    with open(path, "w") as f:
        f.write("Pe Input vs Reactance Approach Comparison\n")
        f.write("=" * 70 + "\n\n")

        f.write("Pe Input Approach:\n")
        f.write(f"  Delta RMSE: {results_pe['delta_rmse']:.6f}\n")
        f.write(f"  Omega RMSE: {results_pe['omega_rmse']:.6f}\n")
        f.write(f"  Delta MAE: {results_pe['delta_mae']:.6f}\n")
        f.write(f"  Omega MAE: {results_pe['omega_mae']:.6f}\n")
        f.write(f"  Delta R²: {results_pe['delta_r2']:.6f}\n")
        f.write(f"  Omega R²: {results_pe['omega_r2']:.6f}\n\n")

        f.write("Reactance-Based Approach:\n")
        f.write(f"  Delta RMSE: {results_reactance['delta_rmse']:.6f}\n")
        f.write(f"  Omega RMSE: {results_reactance['omega_rmse']:.6f}\n")
        f.write(f"  Delta MAE: {results_reactance['delta_mae']:.6f}\n")
        f.write(f"  Omega MAE: {results_reactance['omega_mae']:.6f}\n")
        f.write(f"  Delta R²: {results_reactance['delta_r2']:.6f}\n")
        f.write(f"  Omega R²: {results_reactance['omega_r2']:.6f}\n\n")

        f.write("Differences:\n")
        f.write(f"  Delta RMSE: {results_pe['delta_rmse'] - results_reactance['delta_rmse']:.6f}\n")
        f.write(f"  Omega RMSE: {results_pe['omega_rmse'] - results_reactance['omega_rmse']:.6f}\n")


if __name__ == "__main__":
    main()
