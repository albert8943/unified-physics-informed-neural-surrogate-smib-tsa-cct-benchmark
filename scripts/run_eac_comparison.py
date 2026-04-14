#!/usr/bin/env python
"""
EAC Comparison Analysis Script.

Runs comprehensive EAC comparison analysis with damping effect analysis:
- PINN vs ANDES (ground truth)
- EAC vs ANDES (shows EAC limitations)
- Quantitative damping effect analysis
- Generates comparison figures

Usage:
    python scripts/run_eac_comparison.py \
        --pinn-model outputs/experiments/exp_XXX/model/best_model.pth \
        --data-path data/generated/test_data.csv \
        --output-dir outputs/eac_comparison
"""

import argparse
import glob
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import torch
from evaluation.cct_comparison import (
    compare_cct_estimation_with_damping_analysis,
    check_damping_distribution,
)
from scripts.core.utils import (
    load_config,
    generate_experiment_id,
    create_experiment_directory,
)
from visualization.publication_figures import (
    plot_eac_cct_comparison,
    plot_eac_error_by_damping,
    plot_eac_error_by_category,
)
from utils.cct_binary_search import estimate_cct_binary_search
from datetime import datetime


def generate_eac_experiment_summary(
    experiment_dir: Path,
    experiment_id: str,
    model_path: Path,
    data_path: Path,
    results: Dict,
    scenarios: List[Dict],
    damping_dist: Dict,
    true_cct: Optional[np.ndarray] = None,
) -> Path:
    """
    Generate EAC comparison experiment summary markdown file.

    Parameters
    ----------
    experiment_dir : Path
        Experiment output directory
    experiment_id : str
        Experiment ID (e.g., exp_20260120_143022)
    model_path : Path
        Path to PINN model used
    data_path : Path
        Path to test data used
    results : Dict
        Comparison results dictionary
    scenarios : List[Dict]
        List of test scenarios
    damping_dist : Dict
        Damping distribution statistics
    true_cct : np.ndarray, optional
        Ground truth CCT values

    Returns
    -------
    Path
        Path to generated summary markdown file
    """
    summary_file = experiment_dir / f"EAC_COMPARISON_SUMMARY_{experiment_id}.md"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Extract results
    pinn_results = results.get("PINN", {})
    eac_results = results.get("EAC", {})
    tds_results = results.get("TDS", {})

    pinn_pred = pinn_results.get("cct_predictions", np.array([]))
    eac_pred = eac_results.get("cct_predictions", np.array([]))

    # Extract metrics
    pinn_metrics = pinn_results.get("metrics", {})
    eac_metrics = eac_results.get("metrics", {})
    eac_success_rate = results.get("eac_success_rate", {})
    eac_correlation = results.get("eac_damping_correlation", {})

    # Count valid predictions
    pinn_valid = np.sum(~np.isnan(pinn_pred)) if len(pinn_pred) > 0 else 0
    eac_valid = np.sum(~np.isnan(eac_pred)) if len(eac_pred) > 0 else 0

    # Generate markdown content
    md_content = f"""# EAC Comparison Analysis Summary

**Experiment ID**: `{experiment_id}`  
**Date**: {date_str}  
**Generated**: {timestamp}

---

## 📊 Executive Summary

This experiment compares PINN (Physics-Informed Neural Network) CCT estimation with the classical Equal Area Criterion (EAC) method. EAC assumes no damping (D=0), while PINN models are trained with realistic damping values, making this a **capability demonstration** rather than a direct accuracy comparison.

### Key Findings

"""

    # Add key findings based on results
    if pinn_metrics and true_cct is not None:
        mae_ms = pinn_metrics.get("cct_mae_ms", "N/A")
        rmse_ms = pinn_metrics.get("cct_rmse_ms", "N/A")
        md_content += f"- **PINN Performance**: MAE = {mae_ms:.2f} ms, RMSE = {rmse_ms:.2f} ms (vs ANDES ground truth)\n"

    if eac_success_rate:
        success_pct = eac_success_rate.get("success_rate", 0) * 100
        md_content += f"- **EAC Success Rate**: {success_pct:.1f}% ({eac_success_rate.get('successful', 0)}/{eac_success_rate.get('total', 0)} scenarios)\n"

    if eac_correlation and "error" not in eac_correlation:
        corr = eac_correlation.get("correlation", 0)
        p_val = eac_correlation.get("p_value", 1.0)
        significant = eac_correlation.get("significant", False)
        md_content += f"- **EAC Error vs Damping Correlation**: r = {corr:.3f} (p = {p_val:.3e}, {'significant' if significant else 'not significant'})\n"
        if significant and corr > 0:
            md_content += (
                "  - ⚠️ **EAC error increases with damping** - demonstrates EAC limitation\n"
            )

    md_content += f"""
---

## 🔧 Experiment Configuration

### Model Information

- **PINN Model**: `{model_path}`
- **Model Type**: Trajectory Prediction PINN
- **Device**: CPU/CUDA (auto-detected)

### Data Information

- **Test Data**: `{data_path}`
- **Number of Scenarios**: {len(scenarios)}
- **Ground Truth CCT Available**: {"✅ Yes" if true_cct is not None and np.sum(~np.isnan(true_cct)) > 0 else "❌ No"}

### Damping Distribution

- **Total Scenarios**: {damping_dist.get("total", 0)}
- **Damping Range**: {damping_dist.get("min", 0):.3f} - {damping_dist.get("max", 0):.3f} pu
- **Mean Damping**: {damping_dist.get("mean", 0):.3f} pu
- **Median Damping**: {damping_dist.get("median", 0):.3f} pu

**Damping Categories**:
- D < 0.3 pu: {damping_dist.get("d_lt_0.3", 0)} scenarios
- 0.3 ≤ D < 0.5 pu: {damping_dist.get("d_0.3_to_0.5", 0)} scenarios
- 0.5 ≤ D < 1.5 pu: {damping_dist.get("d_0.5_to_1.5", 0)} scenarios
- D ≥ 1.5 pu: {damping_dist.get("d_ge_1.5", 0)} scenarios

---

## 📈 Results

### PINN vs ANDES (Ground Truth)

"""

    if pinn_metrics and true_cct is not None:
        mae_ms = pinn_metrics.get("cct_mae_ms", "N/A")
        rmse_ms = pinn_metrics.get("cct_rmse_ms", "N/A")
        max_err_ms = pinn_metrics.get("cct_max_error_ms", "N/A")

        md_content += f"""
| Metric | Value |
|--------|-------|
| **MAE** | {mae_ms:.3f} ms |
| **RMSE** | {rmse_ms:.3f} ms |
| **Max Error** | {max_err_ms:.3f} ms |
| **Valid Predictions** | {pinn_valid}/{len(scenarios)} |

"""

        # Performance assessment
        if isinstance(mae_ms, (int, float)):
            if mae_ms < 10:
                md_content += "✅ **Excellent performance** - MAE < 10 ms\n\n"
            elif mae_ms < 50:
                md_content += "✅ **Good performance** - MAE < 50 ms\n\n"
            elif mae_ms < 100:
                md_content += "⚠️ **Moderate performance** - MAE < 100 ms\n\n"
            else:
                md_content += (
                    "⚠️ **High error** - MAE > 100 ms (check model training or predictions)\n\n"
                )
    else:
        md_content += f"""
- **Valid Predictions**: {pinn_valid}/{len(scenarios)}
- **Metrics**: Not computed (no ground truth CCT available)

"""

    md_content += "### EAC vs ANDES (Ground Truth)\n\n"

    if eac_metrics and true_cct is not None:
        eac_mae_ms = eac_metrics.get("cct_mae_ms", "N/A")
        eac_rmse_ms = eac_metrics.get("cct_rmse_ms", "N/A")
        eac_max_err_ms = eac_metrics.get("cct_max_error_ms", "N/A")

        md_content += f"""
| Metric | Value |
|--------|-------|
| **MAE** | {eac_mae_ms:.3f} ms |
| **RMSE** | {eac_rmse_ms:.3f} ms |
| **Max Error** | {eac_max_err_ms:.3f} ms |
| **Valid Predictions** | {eac_valid}/{len(scenarios)} |

**Note**: EAC assumes D=0, so higher errors are expected with realistic damping values.

"""
    else:
        md_content += f"""
- **Valid Predictions**: {eac_valid}/{len(scenarios)}
- **Metrics**: Not computed (no ground truth CCT available or all EAC calculations failed)

"""

    # EAC Success Rate
    if eac_success_rate:
        success_pct = eac_success_rate.get("success_rate", 0) * 100
        md_content += f"""
### EAC Calculation Success Rate

- **Successful**: {eac_success_rate.get("successful", 0)}/{eac_success_rate.get("total", 0)} scenarios ({success_pct:.1f}%)
- **Failed**: {eac_success_rate.get("failed", 0)} scenarios

"""
        if success_pct < 50:
            md_content += "⚠️ **Warning**: Low EAC success rate - check reactance values or parameter ranges\n\n"

    # Damping Correlation Analysis
    if eac_correlation and "error" not in eac_correlation:
        corr = eac_correlation.get("correlation", 0)
        p_val = eac_correlation.get("p_value", 1.0)
        r_squared = eac_correlation.get("r_squared", 0)
        significant = eac_correlation.get("significant", False)

        md_content += f"""
### EAC Error vs Damping Correlation

| Metric | Value |
|--------|-------|
| **Correlation (r)** | {corr:.3f} |
| **R²** | {r_squared:.3f} |
| **P-value** | {p_val:.3e} |
| **Significant** | {"✅ Yes" if significant else "❌ No"} |

"""
        if significant and corr > 0.3:
            md_content += f"**Finding**: EAC error shows {'strong' if corr > 0.7 else 'moderate' if corr > 0.5 else 'weak'} positive correlation with damping (r = {corr:.3f}). This demonstrates that EAC becomes less accurate as damping increases, which is expected since EAC assumes D=0.\n\n"

    md_content += """---

## 📁 Output Files

All results are saved in this experiment directory:

- **Comparison Results**: `results/comparison_results.json`
- **Damping Distribution**: `results/damping_distribution.json`
- **Figures**:
  - `results/figures/eac_cct_comparison.png` - CCT comparison scatter plot
  - `results/figures/eac_error_by_damping.png` - Error vs damping analysis
  - `results/figures/eac_error_by_category.png` - Error by damping category

---

## 🔍 Interpretation Notes

### EAC Limitations

1. **Damping Assumption**: EAC assumes D=0 (no damping), while your PINN model is trained with realistic damping (D = 0.5-2.5 pu)
2. **Fair Comparison**: This is a **capability demonstration**, not a direct accuracy comparison
3. **Expected Behavior**: EAC error should increase with damping - this is normal and demonstrates the limitation

### PINN Advantages

1. **Handles Damping**: PINN can account for realistic damping effects
2. **Physics-Informed**: Incorporates swing equation physics
3. **Robust**: Works across wide parameter ranges

### Publication Strategy

- **DO**: Emphasize PINN's capability to handle damping that EAC cannot
- **DO**: Quantify EAC limitations with correlation/regression analysis
- **DO NOT**: Claim PINN is "more accurate" than EAC (unfair comparison due to assumption mismatch)

---

## 🔗 Related Documentation

- [EAC Comparison Guide](../../docs/04_evaluation/eac_comparison.md)
- [EAC Publication Strategy](../../docs/06_publication/eac_publication_strategy.md)
- [EAC Commands Reference](../../docs/04_evaluation/eac_comparison_commands.md)

---

**Last Updated**: {timestamp}  
**Generated Automatically**: This file was generated by the EAC comparison analysis script.

""".format(
        timestamp=timestamp
    )

    # Write to file
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    return summary_file


def load_pinn_model(model_path: Path, config_path: Optional[Path] = None, device: str = "cpu"):
    """
    Load trained PINN model.

    Parameters:
    -----------
    model_path : Path
        Path to model checkpoint
    config_path : Path, optional
        Path to config file
    device : str
        Device to use

    Returns:
    --------
    model : nn.Module
        Loaded model
    config : dict
        Model configuration
    scalers : dict
        Data scalers
    """
    from pinn.trajectory_prediction import (
        TrajectoryPredictionPINN,
        TrajectoryPredictionPINN_PeInput,
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get config
    if config_path and config_path.exists():
        config = load_config(config_path)
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # Try to infer from model path
        exp_dir = model_path.parent.parent
        config_path = exp_dir / "config.yaml"
        if config_path.exists():
            config = load_config(config_path)
        else:
            raise ValueError(f"Could not find config for model: {model_path}")

    model_config = config.get("model", {})
    input_method = model_config.get("input_method", "reactance")

    # Get scalers from checkpoint if available
    scalers = checkpoint.get("scalers", None)

    # Build model
    if input_method == "pe_direct" or model_config.get("use_pe_as_input", False):
        model = TrajectoryPredictionPINN_PeInput(
            input_dim=9,
            hidden_dims=model_config.get("hidden_dims", [64, 64, 64, 64]),
            activation=model_config.get("activation", "tanh"),
            use_residual=model_config.get("use_residual", False),
            dropout=model_config.get("dropout", 0.0),
            use_standardization=model_config.get("use_standardization", True),
        )
    else:
        model = TrajectoryPredictionPINN(
            input_dim=model_config.get("input_dim", 11),
            hidden_dims=model_config.get("hidden_dims", [64, 64, 64, 64]),
            activation=model_config.get("activation", "tanh"),
            use_residual=model_config.get("use_residual", False),
            dropout=model_config.get("dropout", 0.0),
            use_standardization=model_config.get("use_standardization", True),
        )

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)
    return model, config, scalers


def extract_test_scenarios_from_data(
    data_path: Path, max_scenarios: Optional[int] = None
) -> Tuple[List[Dict], np.ndarray]:
    """
    Extract unique test scenarios from data file.

    Parameters:
    -----------
    data_path : Path
        Path to test data CSV
    max_scenarios : int, optional
        Maximum number of scenarios to use

    Returns:
    --------
    scenarios : list
        List of scenario dictionaries with parameters
    true_cct : np.ndarray
        True CCT values (if available in data, otherwise None)
    """
    data = pd.read_csv(data_path)

    # Check if data has required columns
    if "scenario_id" not in data.columns:
        raise ValueError("Data file must contain 'scenario_id' column")

    # Get unique scenarios
    scenario_ids = sorted(data["scenario_id"].unique())
    if len(scenario_ids) == 0:
        raise ValueError("No scenarios found in data file")

    if max_scenarios:
        scenario_ids = scenario_ids[:max_scenarios]

    scenarios = []
    true_cct_list = []

    for scenario_id in scenario_ids:
        scenario_data = data[data["scenario_id"] == scenario_id].iloc[0]

        # Extract parameters
        # Handle M and H: prefer M if available, otherwise use H
        M = scenario_data.get("M", None)
        H = scenario_data.get("H", None)
        if M is not None:
            # M is available, compute H from M
            H_computed = M / 2.0
            H_final = H if H is not None else H_computed
        elif H is not None:
            # H is available, compute M from H
            M = 2.0 * H
            H_final = H
        else:
            # Neither available, use defaults
            H_final = 6.0
            M = 12.0

        # Extract reactances - may not be in data
        # Since you're using Pe input method, reactance is not needed for PINN/ML models
        # BUT EAC (Equal Area Criterion) is an analytical method that uses power-angle relationship:
        #   Pe = (V1*V2/X)*sin(δ)
        # So EAC needs reactance to calculate Pmax = V1*V2/X and find initial angle δ0

        # Try to infer Xprefault from Pe0, delta0, and voltages if available
        # From: Pe = (V1*V2/X)*sin(δ)  =>  X = (V1*V2/Pe)*sin(δ)
        # Also need to ensure: Pm = (V1*V2/X)*sin(δ0) for EAC to work
        Xprefault = scenario_data.get("Xprefault", None)
        V1 = scenario_data.get("V0", scenario_data.get("V1", 1.05))  # Generator voltage
        V2 = 1.0  # Infinite bus voltage (typically constant)
        Pm = scenario_data.get("Pm", scenario_data.get("Pm_requested", 0.8))

        if Xprefault is None:
            # Try to calculate from Pe0 and delta0 (pre-fault steady-state)
            Pe0 = scenario_data.get("Pe0", scenario_data.get("Pe", None))
            delta0 = scenario_data.get("delta0", None)

            if (
                Pe0 is not None
                and delta0 is not None
                and abs(Pe0) > 1e-6
                and abs(np.sin(delta0)) > 1e-6
            ):
                # Calculate: X = (V1*V2/Pe)*sin(δ)
                Xprefault = (V1 * V2 / Pe0) * np.sin(delta0)
                # Ensure reasonable value (typical range: 0.1-2.0 pu for SMIB)
                if Xprefault < 0.1 or Xprefault > 2.0:
                    # Recalculate from Pm and delta0 to ensure EAC validity
                    # From: Pm = (V1*V2/X)*sin(δ0)  =>  X = (V1*V2/Pm)*sin(δ0)
                    if delta0 is not None and abs(np.sin(delta0)) > 1e-6 and abs(Pm) > 1e-6:
                        Xprefault = (V1 * V2 / Pm) * np.sin(delta0)
                        # Clamp to reasonable range
                        Xprefault = max(0.1, min(2.0, Xprefault))
                    else:
                        # Use default for SMIB system
                        Xprefault = 0.5
            else:
                # Try to calculate from Pm and delta0 directly
                delta0 = scenario_data.get("delta0", None)
                if delta0 is not None and abs(np.sin(delta0)) > 1e-6 and abs(Pm) > 1e-6:
                    # From: Pm = (V1*V2/X)*sin(δ0)  =>  X = (V1*V2/Pm)*sin(δ0)
                    Xprefault = (V1 * V2 / Pm) * np.sin(delta0)
                    # Clamp to reasonable range
                    Xprefault = max(0.1, min(2.0, Xprefault))
                else:
                    # Use default for SMIB system
                    Xprefault = 0.5

        # Validate that Xprefault results in valid Pm/Pmax_pre ratio for EAC
        # EAC requires: |Pm/Pmax_pre| <= 1.0, where Pmax_pre = (V1*V2)/Xprefault
        Pmax_pre = (V1 * V2) / Xprefault if Xprefault > 0 else float("inf")
        if Pmax_pre > 0 and abs(Pm / Pmax_pre) > 1.0:
            # Adjust Xprefault to ensure valid ratio
            # We want: |Pm/Pmax_pre| <= 1.0  =>  |Pm * Xprefault / (V1*V2)| <= 1.0
            #  =>  Xprefault <= (V1*V2) / |Pm|
            Xprefault_max = (V1 * V2) / abs(Pm) if abs(Pm) > 1e-6 else 2.0
            # Use a value slightly smaller to ensure ratio < 1.0
            Xprefault = min(Xprefault, Xprefault_max * 0.99)
            # Ensure still in reasonable range
            Xprefault = max(0.1, min(2.0, Xprefault))

        # Xfault is typically very small (bolted fault)
        Xfault = scenario_data.get("Xfault", scenario_data.get("fault_reactance", 0.0001))

        # Xpostfault is typically same as Xprefault (no line tripping in standard SMIB)
        Xpostfault = scenario_data.get("Xpostfault", Xprefault)

        # V1 and V2 are already defined above during Xprefault inference
        # Store them in scenario for EAC calculation consistency
        scenario = {
            "scenario_id": scenario_id,
            "Pm": scenario_data.get("Pm", scenario_data.get("Pm_requested", 0.8)),
            "H": H_final,
            "M": M,
            "D": scenario_data.get("D", 1.0),
            "Xprefault": Xprefault,
            "Xfault": Xfault,
            "Xpostfault": Xpostfault,
            "V1": V1,
            "V2": V2,
        }

        # Try to get CCT from data (may be in metadata or computed)
        # Note: 'tc' is clearing_time, not CCT - don't use it as ground truth
        cct = scenario_data.get("cct", scenario_data.get("CCT", None))
        if cct is None:
            # Try param_cct_absolute if available (from CCT-based sampling in smib_batch_tds)
            cct = scenario_data.get("param_cct_absolute", None)
        if cct is None:
            # Try cct_absolute (alternative column name)
            cct = scenario_data.get("cct_absolute", None)
        # Note: We don't use 'tc' (clearing_time) as CCT because it's just the
        # clearing time used for that trajectory, not the critical clearing time

        scenarios.append(scenario)
        true_cct_list.append(cct if cct is not None else np.nan)

    true_cct = np.array(true_cct_list)
    return scenarios, true_cct


def estimate_cct_pinn_for_scenarios(
    model,
    scenarios: List[Dict],
    data_path: Path,
    device: str = "cpu",
    simulation_time: float = 5.0,
) -> np.ndarray:
    """
    Estimate CCT for all scenarios using PINN binary search.

    Parameters:
    -----------
    model : nn.Module
        Trained PINN model
    scenarios : list
        List of scenario dictionaries with parameters
    data_path : Path
        Path to data file (for extracting initial conditions)
    device : str
        Device to use
    simulation_time : float
        Simulation time for trajectory evaluation

    Returns:
    --------
    cct_predictions : np.ndarray
        PINN CCT predictions
    """
    data = pd.read_csv(data_path)
    cct_predictions = []

    print(f"Estimating CCT for {len(scenarios)} scenarios using PINN...")

    # Time array for trajectory evaluation
    t_eval = np.linspace(0, simulation_time, 500)

    for i, scenario in enumerate(scenarios):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(scenarios)}")

        scenario_id = scenario["scenario_id"]
        scenario_data = data[data["scenario_id"] == scenario_id]

        # Check if scenario data exists
        if len(scenario_data) == 0:
            print(f"  ⚠️  No data found for scenario {scenario_id}, skipping")
            cct_predictions.append(np.nan)
            continue

        try:
            # Extract initial conditions from first row of scenario data
            first_row = scenario_data.iloc[0]
            delta0 = first_row.get("delta0", first_row.get("delta", 0.0))
            omega0 = first_row.get("omega0", first_row.get("omega", 1.0))
            tf = first_row.get("tf", first_row.get("fault_start_time", 1.0))

            # Extract parameters
            # H and M should already be in scenario from extract_test_scenarios_from_data
            H = scenario.get("H", 6.0)
            D = scenario.get("D", 1.0)
            Pm = scenario.get("Pm", 0.8)
            Xprefault = scenario.get("Xprefault", 0.5)
            Xfault = scenario.get("Xfault", 0.0001)
            Xpostfault = scenario.get("Xpostfault", 0.5)

            # Use binary search to estimate CCT
            cct, info = estimate_cct_binary_search(
                trajectory_model=model,
                delta0=float(delta0),
                omega0=float(omega0),
                H=float(H),
                D=float(D),
                Pm=float(Pm),
                Xprefault=float(Xprefault),
                Xfault=float(Xfault),
                Xpostfault=float(Xpostfault),
                tf=float(tf),
                t_eval=t_eval,
                low=tf + 0.001,  # Clearing time must be after fault start
                high=tf + 1.0,  # Reasonable upper bound
                tolerance=0.001,
                max_iterations=20,
                device=device,
                verbose=False,
            )
            if cct is not None:
                cct_predictions.append(cct)
            else:
                cct_predictions.append(np.nan)
        except Exception as e:
            print(f"  ⚠️  Failed to estimate CCT for scenario {scenario_id}: {e}")
            cct_predictions.append(np.nan)

    return np.array(cct_predictions)


def main():
    """Main function for EAC comparison analysis."""
    parser = argparse.ArgumentParser(description="Run EAC comparison analysis")
    parser.add_argument(
        "--pinn-model",
        type=str,
        required=True,
        help="Path to trained PINN model checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to test data CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eac_comparison",
        help="Output directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (optional, will try to infer from model path)",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Maximum number of scenarios to analyze (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, or cuda)",
    )

    args = parser.parse_args()

    # Generate experiment ID and create experiment directory structure
    experiment_id = generate_experiment_id()
    base_output_dir = Path(args.output_dir)

    # Create experiment directory structure following project conventions
    dirs = create_experiment_directory(base_output_dir, experiment_id)

    # Use experiment directories
    output_dir = dirs["root"]  # Experiment root directory
    figures_dir = dirs["results"] / "figures"  # Figures in results/figures/
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Handle wildcards in model path (for PowerShell/Windows compatibility)
    model_path_str = args.pinn_model
    if "*" in model_path_str or "?" in model_path_str:
        # Expand wildcards
        matches = glob.glob(model_path_str)
        if not matches:
            print(f"✗ No files found matching pattern: {model_path_str}")
            sys.exit(1)
        if len(matches) > 1:
            print(f"⚠️  Multiple files found matching pattern: {model_path_str}")
            print(f"   Using first match: {matches[0]}")
        model_path_str = matches[0]
        print(f"   Expanded to: {model_path_str}")

    print("=" * 70)
    print("EAC COMPARISON ANALYSIS")
    print("=" * 70)
    print(f"Experiment ID: {experiment_id}")
    print(f"Output Directory: {output_dir}")
    print(f"PINN Model: {model_path_str}")
    print(f"Test Data: {args.data_path}")
    print(f"Device: {device}")

    # Load PINN model
    print("\n" + "=" * 70)
    print("LOADING PINN MODEL")
    print("=" * 70)

    model_path = Path(model_path_str)
    config_path = Path(args.config) if args.config else None

    try:
        model, config, scalers = load_pinn_model(model_path, config_path, device)
        print("✓ PINN model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load PINN model: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Extract test scenarios
    print("\n" + "=" * 70)
    print("EXTRACTING TEST SCENARIOS")
    print("=" * 70)

    # Handle wildcards in data path (for PowerShell/Windows compatibility)
    data_path_str = args.data_path
    if "*" in data_path_str or "?" in data_path_str:
        # Expand wildcards
        matches = glob.glob(data_path_str)
        if not matches:
            print(f"✗ No files found matching pattern: {data_path_str}")
            sys.exit(1)
        if len(matches) > 1:
            print(f"⚠️  Multiple files found matching pattern: {data_path_str}")
            print(f"   Using first match: {matches[0]}")
        data_path_str = matches[0]
        print(f"   Expanded to: {data_path_str}")

    data_path = Path(data_path_str)

    try:
        scenarios, true_cct = extract_test_scenarios_from_data(data_path, args.max_scenarios)
        print(f"✓ Extracted {len(scenarios)} test scenarios")

        # Check if reactance was inferred (for Pe input method users)
        if len(scenarios) > 0:
            first_scenario = scenarios[0]
            if "Xprefault" in first_scenario:
                # Check if it's a calculated value (not default)
                sample_data = pd.read_csv(data_path, nrows=1)
                has_xprefault_col = "Xprefault" in sample_data.columns
                has_pe0 = "Pe0" in sample_data.columns or "Pe" in sample_data.columns
                has_delta0 = "delta0" in sample_data.columns

                if not has_xprefault_col and has_pe0 and has_delta0:
                    print("ℹ️  Reactance inferred from Pe0 and delta0 (Pe input method detected)")
                elif not has_xprefault_col:
                    print(
                        "⚠️  Using default reactance values (0.5, 0.0001, 0.5) - EAC may be inaccurate"
                    )

        # Check if we have CCT ground truth
        valid_cct_mask = ~np.isnan(true_cct)
        n_valid_cct = np.sum(valid_cct_mask)
        if n_valid_cct == 0:
            print(
                "⚠️  Warning: No CCT ground truth found in data. "
                "Will need to compute using ANDES TDS (this may take time)."
            )
        else:
            print(f"✓ Found CCT ground truth for {n_valid_cct}/{len(scenarios)} scenarios")
    except Exception as e:
        print(f"✗ Failed to extract test scenarios: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Check damping distribution
    print("\n" + "=" * 70)
    print("CHECKING DAMPING DISTRIBUTION")
    print("=" * 70)
    damping_dist = check_damping_distribution(scenarios)
    print(f"Total scenarios: {damping_dist['total']}")
    print(f"Damping range: {damping_dist['min']:.3f} - {damping_dist['max']:.3f} pu")
    print(f"Mean damping: {damping_dist['mean']:.3f} pu")
    print(f"Median damping: {damping_dist['median']:.3f} pu")
    print(f"\nDamping distribution:")
    print(f"  D < 0.3 pu: {damping_dist.get('d_lt_0.3', 0)} scenarios")
    d_lt_05 = damping_dist.get("d_lt_0.3", 0) + damping_dist.get("d_0.3_to_0.5", 0)
    print(f"  D < 0.5 pu: {d_lt_05} scenarios")
    print(f"  0.5 ≤ D < 1.5 pu: {damping_dist.get('d_0.5_to_1.5', 0)} scenarios")
    print(f"  D ≥ 1.5 pu: {damping_dist.get('d_ge_1.5', 0)} scenarios")

    # Save damping distribution to results directory
    with open(dirs["results"] / "damping_distribution.json", "w") as f:
        json.dump(damping_dist, f, indent=2)
    print(f"✓ Saved damping distribution to: {dirs['results'] / 'damping_distribution.json'}")

    # Run comparison analysis
    print("\n" + "=" * 70)
    print("RUNNING CCT COMPARISON")
    print("=" * 70)

    # Estimate CCT using PINN
    print("\nEstimating CCT using PINN...")
    pinn_cct_predictions = estimate_cct_pinn_for_scenarios(model, scenarios, data_path, device)

    # Run comprehensive comparison
    print("\nRunning EAC comparison analysis...")
    try:
        results = compare_cct_estimation_with_damping_analysis(
            test_scenarios=scenarios,
            pinn_model=None,  # We'll add PINN predictions separately
            true_cct=true_cct,
        )

        # Update with PINN predictions
        if "PINN" not in results:
            results["PINN"] = {}
        results["PINN"]["cct_predictions"] = pinn_cct_predictions

        # Compute PINN metrics if true CCT available
        if true_cct is not None:
            from utils.metrics import compute_cct_metrics

            valid_mask = ~(np.isnan(pinn_cct_predictions) | np.isnan(true_cct))
            if np.sum(valid_mask) > 0:
                pinn_metrics = compute_cct_metrics(
                    cct_pred=pinn_cct_predictions[valid_mask],
                    cct_true=true_cct[valid_mask],
                )
                results["PINN"]["metrics"] = pinn_metrics
            else:
                print("⚠️  No valid PINN predictions or true CCT for metrics computation")
        else:
            print("⚠️  No ground truth CCT available - PINN metrics not computed")

        print("✓ Comparison analysis complete")
    except Exception as e:
        print(f"✗ Comparison analysis failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for method_name, method_results in results.items():
        results_json[method_name] = {}
        for key, value in method_results.items():
            if isinstance(value, np.ndarray):
                results_json[method_name][key] = value.tolist()
            elif isinstance(value, np.generic):
                results_json[method_name][key] = float(value)
            elif isinstance(value, dict):
                # Recursively convert dict values
                results_json[method_name][key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        results_json[method_name][key][k] = v.tolist()
                    elif isinstance(v, np.generic):
                        results_json[method_name][key][k] = float(v)
                    else:
                        results_json[method_name][key][k] = v
            else:
                results_json[method_name][key] = value

    with open(dirs["results"] / "comparison_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"✓ Saved results to: {dirs['results'] / 'comparison_results.json'}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if "EAC" in results and "metrics" in results["EAC"]:
        eac_metrics = results["EAC"]["metrics"]
        print("\nEAC vs ANDES:")
        # Use correct metric keys: cct_mae_ms, cct_rmse_ms, cct_max_error_ms
        mae_ms = eac_metrics.get("cct_mae_ms", eac_metrics.get("cct_mae", "N/A"))
        rmse_ms = eac_metrics.get("cct_rmse_ms", eac_metrics.get("cct_rmse", "N/A"))
        max_err_ms = eac_metrics.get("cct_max_error_ms", eac_metrics.get("cct_max_error", "N/A"))
        if isinstance(mae_ms, (int, float)):
            mae_ms = f"{mae_ms:.3f}"
        if isinstance(rmse_ms, (int, float)):
            rmse_ms = f"{rmse_ms:.3f}"
        if isinstance(max_err_ms, (int, float)):
            max_err_ms = f"{max_err_ms:.3f}"
        print(f"  MAE: {mae_ms} ms")
        print(f"  RMSE: {rmse_ms} ms")
        print(f"  Max Error: {max_err_ms} ms")

    if "PINN" in results and "metrics" in results["PINN"]:
        pinn_metrics = results["PINN"]["metrics"]
        print("\nPINN vs ANDES:")
        # Use correct metric keys: cct_mae_ms, cct_rmse_ms, cct_max_error_ms
        mae_ms = pinn_metrics.get("cct_mae_ms", pinn_metrics.get("cct_mae", "N/A"))
        rmse_ms = pinn_metrics.get("cct_rmse_ms", pinn_metrics.get("cct_rmse", "N/A"))
        max_err_ms = pinn_metrics.get("cct_max_error_ms", pinn_metrics.get("cct_max_error", "N/A"))
        if isinstance(mae_ms, (int, float)):
            mae_ms = f"{mae_ms:.3f}"
        if isinstance(rmse_ms, (int, float)):
            rmse_ms = f"{rmse_ms:.3f}"
        if isinstance(max_err_ms, (int, float)):
            max_err_ms = f"{max_err_ms:.3f}"
        print(f"  MAE: {mae_ms} ms")
        print(f"  RMSE: {rmse_ms} ms")
        print(f"  Max Error: {max_err_ms} ms")
    elif "PINN" in results:
        print("\nPINN Predictions:")
        pinn_pred = results["PINN"]["cct_predictions"]
        valid_pred = ~np.isnan(pinn_pred)
        print(f"  Valid predictions: {np.sum(valid_pred)}/{len(pinn_pred)}")
        if np.sum(valid_pred) > 0:
            print(
                f"  Range: {np.min(pinn_pred[valid_pred]):.4f} - {np.max(pinn_pred[valid_pred]):.4f} s"
            )
            print(f"  Mean: {np.mean(pinn_pred[valid_pred]):.4f} s")
        print("  ⚠️  No ground truth CCT available - metrics not computed")

    if "EAC" in results:
        eac_pred = results["EAC"]["cct_predictions"]
        valid_eac = ~np.isnan(eac_pred)
        print(f"\nEAC Predictions:")
        print(f"  Valid predictions: {np.sum(valid_eac)}/{len(eac_pred)}")
        if np.sum(valid_eac) == 0:
            print("  ⚠️  All EAC calculations failed - check parameter ranges")
        elif true_cct is not None:
            if "eac_damping_correlation" in results:
                corr = results["eac_damping_correlation"]
                if "error" not in corr:
                    print("\nEAC Error vs Damping Correlation:")
                    print(f"  Correlation: r = {corr.get('correlation', 'N/A'):.3f}")
                    print(f"  P-value: p = {corr.get('p_value', 'N/A'):.3e}")
                    print(f"  R²: {corr.get('r_squared', 'N/A'):.3f}")
                    print(f"  Significant: {corr.get('significant', 'N/A')}")
                else:
                    print(f"\nEAC Correlation Analysis: {corr.get('error', 'Failed')}")

            if "eac_success_rate" in results:
                success = results["eac_success_rate"]
                print(f"\nEAC Success Rate: {success.get('success_rate', 0) * 100:.1f}%")
                print(f"  Successful: {success.get('successful', 0)}/{success.get('total', 0)}")
        else:
            print("  ⚠️  No ground truth CCT available - EAC analysis limited")

    # Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    if "EAC" in results and "PINN" in results:
        # Figure 1: CCT Comparison Scatter Plot
        print("\nGenerating CCT comparison scatter plot...")
        try:
            eac_pred = results["EAC"]["cct_predictions"]
            pinn_pred = results["PINN"]["cct_predictions"]
            valid_mask = ~(np.isnan(true_cct) | np.isnan(eac_pred) | np.isnan(pinn_pred))

            if np.sum(valid_mask) > 0:
                _ = plot_eac_cct_comparison(
                    true_cct=true_cct[valid_mask],
                    pinn_predictions=pinn_pred[valid_mask],
                    eac_predictions=eac_pred[valid_mask],
                    output_dir=figures_dir,
                )
                print("✓ CCT comparison plot generated")
            else:
                print("⚠️  No valid data for CCT comparison plot")
        except Exception as e:
            print(f"⚠️  Failed to generate CCT comparison plot: {e}")
            import traceback

            traceback.print_exc()

        # Figure 2: Error vs Damping
        print("\nGenerating error vs damping plot...")
        try:
            eac_pred = results["EAC"]["cct_predictions"]
            pinn_pred = results["PINN"]["cct_predictions"]
            damping_values = np.array([s["D"] for s in scenarios])

            if true_cct is not None:
                valid_mask = ~(np.isnan(true_cct) | np.isnan(eac_pred))
            else:
                print("⚠️  No ground truth CCT available - skipping error vs damping plot")
                valid_mask = np.array([])

            if np.sum(valid_mask) > 0:
                eac_errors = (
                    np.abs(eac_pred[valid_mask] - true_cct[valid_mask]) * 1000
                )  # Convert to ms
                damping_eac = damping_values[valid_mask]

                pinn_errors = None
                pinn_damping = None
                if np.sum(~np.isnan(pinn_pred)) > 0:
                    pinn_valid_mask = ~(np.isnan(true_cct) | np.isnan(pinn_pred))
                    if np.sum(pinn_valid_mask) > 0:
                        pinn_errors = (
                            np.abs(pinn_pred[pinn_valid_mask] - true_cct[pinn_valid_mask]) * 1000
                        )
                        pinn_damping = damping_values[pinn_valid_mask]

                _ = plot_eac_error_by_damping(
                    eac_errors=eac_errors,
                    damping_values=damping_eac,
                    pinn_errors=pinn_errors,
                    pinn_damping_values=pinn_damping,
                    output_dir=figures_dir,
                )
                print("✓ Error vs damping plot generated")
            else:
                print("⚠️  No valid data for error vs damping plot")
        except Exception as e:
            print(f"⚠️  Failed to generate error vs damping plot: {e}")

        # Figure 3: Error by Category (only if we have ground truth)
        if true_cct is not None:
            print("\nGenerating error by category plot...")
            try:
                if "eac_metrics_by_category" in results:
                    eac_metrics_cat = results["eac_metrics_by_category"]
                    pinn_metrics_cat = results.get("pinn_metrics_by_category", None)

                    _ = plot_eac_error_by_category(
                        eac_metrics_by_category=eac_metrics_cat,
                        pinn_metrics_by_category=pinn_metrics_cat,
                        output_dir=figures_dir,
                    )
                    print("✓ Error by category plot generated")
                else:
                    print("⚠️  No category metrics available")
            except Exception as e:
                print(f"⚠️  Failed to generate error by category plot: {e}")
                import traceback

                traceback.print_exc()
        else:
            print("\n⚠️  Skipping error by category plot - no ground truth CCT available")

    # Generate experiment summary markdown file
    print("\n" + "=" * 70)
    print("GENERATING EXPERIMENT SUMMARY")
    print("=" * 70)
    try:
        # Get actual data path (after wildcard expansion)
        # data_path was already expanded earlier in the script
        summary_file = generate_eac_experiment_summary(
            experiment_dir=output_dir,
            experiment_id=experiment_id,
            model_path=model_path,
            data_path=data_path,  # Use the already expanded data_path from extract_test_scenarios_from_data
            results=results,
            scenarios=scenarios,
            damping_dist=damping_dist,
            true_cct=true_cct,
        )
        print(f"✓ Experiment summary saved to: {summary_file}")
    except Exception as e:
        print(f"⚠️  Failed to generate experiment summary: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Experiment ID: {experiment_id}")
    print(f"Results saved to: {dirs['results']}")
    print(f"Figures saved to: {figures_dir}")
    print(f"Experiment root: {output_dir}")


if __name__ == "__main__":
    main()
