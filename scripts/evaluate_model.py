#!/usr/bin/env python
"""
Model Evaluation Script

Evaluate trained PINN models on test data and generate performance metrics.

Usage:
    python scripts/evaluate_model.py --model-path outputs/models/common/trajectory/model_*.pth --data-path data/common/trajectory_data_*.csv
"""

import argparse
import re
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

from pinn.checkpoint_layout import infer_architecture_from_state_dict
from pinn.trajectory_prediction import TrajectoryPredictionPINN, TrajectoryPredictionPINN_PeInput
from utils.normalization import normalize_value, denormalize_tensor
from scripts.core.utils import generate_timestamped_filename


def load_model_and_scalers(model_path: Path):
    """Load trained model and scalers. Infers backbone (sequential vs residual) from weights."""
    model_dir = model_path.parent

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    mc = dict(checkpoint.get("model_config") or {})

    use_residual, input_dim, hidden_dims = infer_architecture_from_state_dict(state_dict)
    if not hidden_dims:
        hidden_dims = [256, 256, 128, 128] if input_dim in (7, 9) else [64, 64, 64, 64]

    linear_keys = [k for k in state_dict if re.match(r"network\.\d+\.weight", k)]
    nums = (
        sorted(set(int(re.search(r"network\.(\d+)\.weight", k).group(1)) for k in linear_keys))
        if linear_keys
        else []
    )
    use_dropout_seq = bool(nums and 3 in nums)
    if mc.get("dropout") is not None:
        dropout = float(mc["dropout"])
    elif use_residual:
        # Dropout layers carry no tensors; default matches SMIB parity YAML (0.1)
        dropout = 0.1
    else:
        dropout = 0.1 if use_dropout_seq else 0.0

    if input_dim == 7:
        input_method = "pe_direct_7"
    elif input_dim == 9:
        input_method = "pe_direct"
    else:
        input_method = "reactance"
    print(
        f"  Detected: input_dim={input_dim}, hidden_dims={hidden_dims}, input_method={input_method}, "
        f"use_residual={use_residual}, dropout={dropout}"
    )

    if input_method in ("pe_direct", "pe_direct_7"):
        model = TrajectoryPredictionPINN_PeInput(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=mc.get("activation", "tanh"),
            use_residual=use_residual,
            dropout=dropout,
            use_standardization=mc.get("use_standardization", True),
        )
    else:
        model = TrajectoryPredictionPINN(
            input_dim=11,
            hidden_dims=hidden_dims,
            activation=mc.get("activation", "tanh"),
            use_residual=use_residual,
            dropout=dropout,
            use_standardization=mc.get("use_standardization", True),
        )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("[OK] Model loaded")

    # Scalers: from checkpoint first, then scalers.pkl, else will fit from data
    scalers = checkpoint.get("scalers", {}) if isinstance(checkpoint, dict) else {}
    if not scalers and (model_dir / "scalers.pkl").exists():
        with open(model_dir / "scalers.pkl", "rb") as f:
            scalers = pickle.load(f)
        print(f"[OK] Scalers loaded from: {model_dir / 'scalers.pkl'}")
    elif scalers:
        print(f"[OK] Scalers loaded from checkpoint: {list(scalers.keys())}")
    else:
        print("[WARNING] Scalers not found. Will fit from data.")
    return model, scalers, input_method


def fit_scalers_from_data(df: pd.DataFrame) -> dict:
    """Fit scalers from data if not provided."""
    scalers = {}

    # Sample for efficiency
    sample_scenarios = df["scenario_id"].unique()[: min(100, len(df["scenario_id"].unique()))]
    sample_data = df[df["scenario_id"].isin(sample_scenarios)]

    # Fit scalers
    scalers["time"] = StandardScaler().fit(sample_data["time"].values.reshape(-1, 1))
    scalers["delta"] = StandardScaler().fit(sample_data["delta"].values.reshape(-1, 1))
    scalers["omega"] = StandardScaler().fit(sample_data["omega"].values.reshape(-1, 1))

    scenario_data = df.groupby("scenario_id").first().reset_index()
    H_col = "param_H" if "param_H" in scenario_data.columns else "H"
    D_col = "param_D" if "param_D" in scenario_data.columns else "D"
    Pm_col = "param_Pm" if "param_Pm" in scenario_data.columns else "Pm"
    tc_col = "param_tc" if "param_tc" in scenario_data.columns else "tc"

    scalers["H"] = StandardScaler().fit(scenario_data[H_col].values.reshape(-1, 1))
    scalers["D"] = StandardScaler().fit(scenario_data[D_col].values.reshape(-1, 1))
    scalers["Pm"] = StandardScaler().fit(scenario_data[Pm_col].values.reshape(-1, 1))
    if "Xprefault" in scenario_data.columns:
        scalers["Xprefault"] = StandardScaler().fit(
            scenario_data["Xprefault"].values.reshape(-1, 1)
        )
    if "Xfault" in scenario_data.columns:
        scalers["Xfault"] = StandardScaler().fit(scenario_data["Xfault"].values.reshape(-1, 1))
    if "Xpostfault" in scenario_data.columns:
        scalers["Xpostfault"] = StandardScaler().fit(
            scenario_data["Xpostfault"].values.reshape(-1, 1)
        )
    scalers["tf"] = StandardScaler().fit(scenario_data["tf"].values.reshape(-1, 1))
    scalers["tc"] = StandardScaler().fit(scenario_data[tc_col].values.reshape(-1, 1))
    if "Pe" in sample_data.columns:
        scalers["Pe"] = StandardScaler().fit(sample_data["Pe"].values.reshape(-1, 1))

    # REVERT: Use separate scalers for delta0/omega0 (like training code and December experiments)
    # December experiments achieved R² Delta = 0.881 using separate scalers.
    # Separate scalers create beneficial structure: tight initial condition space → wide trajectory space
    # Fit delta0/omega0 scalers on initial conditions only (scenario_data)
    scalers["delta0"] = StandardScaler().fit(scenario_data["delta0"].values.reshape(-1, 1))
    scalers["omega0"] = StandardScaler().fit(scenario_data["omega0"].values.reshape(-1, 1))

    return scalers


def evaluate_scenario(
    model,
    scenario_data: pd.DataFrame,
    scalers: dict,
    device: str = "cpu",
    input_method: str = "reactance",
):
    """Evaluate model on a single scenario. input_method: 'pe_direct' (9-dim) or 'reactance' (11-dim)."""
    model.eval()
    scenario_data = scenario_data.sort_values("time")

    # Extract raw values
    t_raw = scenario_data["time"].values.astype(np.float32)
    delta_true = scenario_data["delta"].values.astype(np.float32)
    omega_true = scenario_data["omega"].values.astype(np.float32)

    row = scenario_data.iloc[0]
    delta0_raw = float(row.get("delta0", delta_true[0]))
    omega0_raw = float(row.get("omega0", omega_true[0]))
    H_raw = float(row.get("param_H", row.get("H", 5.0)))
    D_raw = float(row.get("param_D", row.get("D", 1.0)))
    Pm_raw = float(row.get("param_Pm", row.get("Pm", 0.8)))
    Xprefault_raw = float(row.get("Xprefault", 0.5))
    Xfault_raw = float(row.get("Xfault", 0.0001))
    Xpostfault_raw = float(row.get("Xpostfault", 0.5))
    tf_raw = float(row.get("tf", 1.0))
    tc_raw = float(row.get("tc", row.get("param_tc", 1.2)))

    # Normalize
    from utils.normalization import normalize_array

    t_norm = torch.tensor(
        normalize_array(t_raw, scalers["time"]), dtype=torch.float32, device=device
    )
    delta0_norm = torch.tensor(
        [normalize_value(delta0_raw, scalers["delta0"])], dtype=torch.float32, device=device
    )
    omega0_norm = torch.tensor(
        [normalize_value(omega0_raw, scalers["omega0"])], dtype=torch.float32, device=device
    )
    H_norm = torch.tensor(
        [normalize_value(H_raw, scalers["H"])], dtype=torch.float32, device=device
    )
    D_norm = torch.tensor(
        [normalize_value(D_raw, scalers["D"])], dtype=torch.float32, device=device
    )
    Pm_norm = torch.tensor(
        [normalize_value(Pm_raw, scalers["Pm"])], dtype=torch.float32, device=device
    )
    tf_norm = torch.tensor(
        [normalize_value(tf_raw, scalers["tf"])], dtype=torch.float32, device=device
    )
    tc_norm = torch.tensor(
        [normalize_value(tc_raw, scalers["tc"])], dtype=torch.float32, device=device
    )

    if input_method in ("pe_direct", "pe_direct_7"):
        # 7-D: [t, δ₀, ω₀, H, D, Pm, Pe(t)] — 9-D adds tf, tc (match model.input_dim)
        Pe_col = "Pe" if "Pe" in scenario_data.columns else "Pe_0"
        Pe_raw = (
            scenario_data[Pe_col].values.astype(np.float32)
            if Pe_col in scenario_data.columns
            else np.full_like(t_raw, Pm_raw)
        )
        Pe_norm = torch.tensor(
            normalize_array(Pe_raw, scalers.get("Pe", scalers["Pm"])),
            dtype=torch.float32,
            device=device,
        )
        pe_dim = int(getattr(model, "input_dim", 9))
        stack_7 = [
            t_norm,
            delta0_norm.expand(len(t_norm)),
            omega0_norm.expand(len(t_norm)),
            H_norm.expand(len(t_norm)),
            D_norm.expand(len(t_norm)),
            Pm_norm.expand(len(t_norm)),
            Pe_norm,
        ]
        if pe_dim == 7:
            inputs = torch.stack(stack_7, dim=1)
        else:
            inputs = torch.stack(
                stack_7
                + [
                    tf_norm.expand(len(t_norm)),
                    tc_norm.expand(len(t_norm)),
                ],
                dim=1,
            )
    else:
        Xprefault_norm = torch.tensor(
            [normalize_value(Xprefault_raw, scalers["Xprefault"])],
            dtype=torch.float32,
            device=device,
        )
        Xfault_norm = torch.tensor(
            [normalize_value(Xfault_raw, scalers["Xfault"])], dtype=torch.float32, device=device
        )
        Xpostfault_norm = torch.tensor(
            [normalize_value(Xpostfault_raw, scalers["Xpostfault"])],
            dtype=torch.float32,
            device=device,
        )
        # 11-dim: [t, δ₀, ω₀, H, D, Pm, Xprefault, Xfault, Xpostfault, tf, tc]
        inputs = torch.stack(
            [
                t_norm,
                delta0_norm.expand(len(t_norm)),
                omega0_norm.expand(len(t_norm)),
                H_norm.expand(len(t_norm)),
                D_norm.expand(len(t_norm)),
                Pm_norm.expand(len(t_norm)),
                Xprefault_norm.expand(len(t_norm)),
                Xfault_norm.expand(len(t_norm)),
                Xpostfault_norm.expand(len(t_norm)),
                tf_norm.expand(len(t_norm)),
                tc_norm.expand(len(t_norm)),
            ],
            dim=1,
        )

    # Predict
    with torch.no_grad():
        outputs = model(inputs)
        delta_pred_norm = outputs[:, 0].cpu().numpy()
        omega_pred_norm = outputs[:, 1].cpu().numpy()

    # Denormalize
    delta_pred = denormalize_tensor(
        torch.tensor(delta_pred_norm), scalers["delta"], device="cpu"
    ).numpy()
    omega_pred = denormalize_tensor(
        torch.tensor(omega_pred_norm), scalers["omega"], device="cpu"
    ).numpy()

    # Compute errors
    delta_error = np.abs(delta_pred - delta_true)
    omega_error = np.abs(omega_pred - omega_true)

    return {
        "time": t_raw,
        "delta_true": delta_true,
        "omega_true": omega_true,
        "delta_pred": delta_pred,
        "omega_pred": omega_pred,
        "delta_error": delta_error,
        "omega_error": omega_error,
        "delta_rmse": np.sqrt(np.mean(delta_error**2)),
        "omega_rmse": np.sqrt(np.mean(omega_error**2)),
        "delta_mae": np.mean(delta_error),
        "omega_mae": np.mean(omega_error),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained PINN model on your data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/evaluate_model.py \\
    --model-path outputs/training/best_model_20251205_120000.pth \\
    --data-path data/common/trajectory_data_1000_H2-10_D0.5-3_abc12345_20251205_170908.csv

  # Evaluate on specific scenarios
  python scripts/evaluate_model.py \\
    --model-path outputs/training/best_model_20251205_120000.pth \\
    --data-path data/common/trajectory_data_1000_H2-10_D0.5-3_abc12345_20251205_170908.csv \\
    --n-scenarios 5
        """,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model file (.pth)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to data CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Directory to save evaluation results (default: results/evaluation)",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=10,
        help="Number of scenarios to evaluate (default: 10)",
    )
    parser.add_argument(
        "--max-scenarios-per-plot",
        type=int,
        default=10,
        help="Max scenarios to show in each evaluation figure, stable or unstable (default: 10, use 5+ for at least 5 when available)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cpu/cuda/auto, default: auto)",
    )

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Validate paths (resolve globs to latest match if needed)
    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    if "*" in str(model_path):
        matches = list(model_path.parent.glob(model_path.name))
        if not matches:
            print(f"Error: No model file matching: {model_path}")
            sys.exit(1)
        model_path = max(matches, key=lambda p: p.stat().st_mtime)
    if "*" in str(data_path):
        matches = list(data_path.parent.glob(data_path.name))
        if not matches:
            print(f"Error: No data file matching: {data_path}")
            sys.exit(1)
        data_path = max(matches, key=lambda p: p.stat().st_mtime)

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Device: {device}")

    # Load model and scalers
    model, scalers, input_method = load_model_and_scalers(model_path)
    model = model.to(device)

    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows, {len(df['scenario_id'].unique())} scenarios")

    # Fit scalers if not loaded
    if not scalers:
        print("Fitting scalers from data...")
        scalers = fit_scalers_from_data(df)
        print("Scalers fitted")

    # Use all scenarios if path is a held-out split (test_data_*.csv or val_data_*.csv);
    # otherwise split with stratify by is_stable
    scenarios = df["scenario_id"].unique()
    if "test_data" in str(data_path) or "val_data" in str(data_path):
        test_scenarios_list = list(scenarios)
        if args.n_scenarios is not None:
            test_scenarios_list = test_scenarios_list[: args.n_scenarios]
        test_data = df
    else:
        from sklearn.model_selection import train_test_split

        stratify_vec = None
        if "is_stable" in df.columns:
            scenario_stability = df.groupby("scenario_id")["is_stable"].first()
            stratify_vec = scenario_stability.reindex(scenarios).values
            if np.any(pd.isna(stratify_vec)) or len(np.unique(stratify_vec)) < 2:
                stratify_vec = None
        try:
            train_scenarios, test_scenarios = train_test_split(
                scenarios, test_size=0.15, random_state=42, stratify=stratify_vec
            )
        except ValueError:
            train_scenarios, test_scenarios = train_test_split(
                scenarios, test_size=0.15, random_state=42
            )
        test_data = df[df["scenario_id"].isin(test_scenarios)]
        test_scenarios_list = list(test_scenarios)[: (args.n_scenarios or len(test_scenarios))]

    # Evaluate on test scenarios
    print(f"\nEvaluating on {len(test_scenarios_list)} test scenarios...")

    results = []
    all_metrics = {"delta_rmse": [], "omega_rmse": [], "delta_mae": [], "omega_mae": []}

    for scenario_id in test_scenarios_list:
        scenario_data = test_data[test_data["scenario_id"] == scenario_id]
        result = evaluate_scenario(
            model, scenario_data, scalers, str(device), input_method=input_method
        )
        result["scenario_id"] = scenario_id
        # Classify stable vs unstable for separate delta-only plots
        if "is_stable" in scenario_data.columns and pd.notna(scenario_data["is_stable"].iloc[0]):
            result["is_stable"] = bool(scenario_data["is_stable"].iloc[0])
        else:
            max_angle_rad = np.abs(result["delta_true"]).max()
            result["is_stable"] = max_angle_rad < np.pi
        results.append(result)

        all_metrics["delta_rmse"].append(result["delta_rmse"])
        all_metrics["omega_rmse"].append(result["omega_rmse"])
        all_metrics["delta_mae"].append(result["delta_mae"])
        all_metrics["omega_mae"].append(result["omega_mae"])

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Number of scenarios evaluated: {len(results)}")
    print(f"\nDelta (Rotor Angle) Metrics:")
    print(
        f"RMSE: {np.mean(all_metrics['delta_rmse']):.6f} ± {np.std(all_metrics['delta_rmse']):.6f}"
        f"rad"
    )
    print(
        f"  MAE:  {np.mean(all_metrics['delta_mae']):.6f} ± {np.std(all_metrics['delta_mae']):.6f} rad"
    )
    print(f"\nOmega (Rotor Speed) Metrics:")
    print(
        f"RMSE: {np.mean(all_metrics['omega_rmse']):.6f} ± {np.std(all_metrics['omega_rmse']):.6f}"
        f"pu"
    )
    print(
        f"  MAE:  {np.mean(all_metrics['omega_mae']):.6f} ± {np.std(all_metrics['omega_mae']):.6f} pu"
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_df = pd.DataFrame(
        {
            "scenario_id": [r["scenario_id"] for r in results],
            "delta_rmse": all_metrics["delta_rmse"],
            "omega_rmse": all_metrics["omega_rmse"],
            "delta_mae": all_metrics["delta_mae"],
            "omega_mae": all_metrics["omega_mae"],
        }
    )
    metrics_path = output_dir / generate_timestamped_filename("evaluation_metrics", "csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")

    # Delta-only plots: one PNG for stable trajectories, one for unstable
    if len(results) > 0:
        stable_results = [r for r in results if r.get("is_stable", False)]
        unstable_results = [r for r in results if not r.get("is_stable", False)]
        n_plot = max(5, args.max_scenarios_per_plot)  # show at least 5 scenarios when available

        def _draw_delta_figure(subset, title_suffix, filename):
            if subset:
                subs = subset[:n_plot]
                n_rows = len(subs)
                fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3 * n_rows))
                if n_rows == 1:
                    axes = [axes]
                for idx, result in enumerate(subs):
                    ax = axes[idx]
                    ax.plot(
                        result["time"],
                        np.degrees(result["delta_true"]),
                        "b-",
                        label="True",
                        linewidth=2,
                    )
                    ax.plot(
                        result["time"],
                        np.degrees(result["delta_pred"]),
                        "r--",
                        label="Predicted",
                        linewidth=2,
                    )
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Rotor Angle (deg)")
                    ax.set_title(f"Scenario {result['scenario_id']} - Delta")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            else:
                # Always write both files: placeholder when no scenarios of this type in test set
                fig, ax = plt.subplots(1, 1, figsize=(6, 2))
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    f"No {title_suffix} in the test set.",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
            fig.suptitle(f"Rotor angle (delta) — {title_suffix}", fontsize=11)
            plt.tight_layout()
            path = output_dir / filename
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  {path.name}")

        _draw_delta_figure(stable_results, "stable trajectories", "evaluation_plots_stable.png")
        _draw_delta_figure(
            unstable_results, "unstable trajectories", "evaluation_plots_unstable.png"
        )
        print("Plots saved: evaluation_plots_stable.png, evaluation_plots_unstable.png")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
