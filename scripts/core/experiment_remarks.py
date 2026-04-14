"""
Generate experiment remarks/summary markdown file.

This module creates a comprehensive markdown document summarizing experiment results,
configuration, and key findings.
"""

import json
import shlex
import torch
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime


def _format_invocation_section(summary: Dict, experiment_dir: Path) -> str:
    """
    Markdown block: exact command used for this run (from experiment_summary invocation_latest).

    Populated by run_complete_experiment.py for every new experiment; older summaries may omit it.
    """
    inv = summary.get("invocation_latest")
    if not inv:
        return (
            "### Command line (replay this experiment)\n\n"
            "_No `invocation_latest` block in this summary (experiment may pre-date automatic "
            "logging). If `RERUN.md` or `run_invocations.jsonl` exists in this folder, use those; "
            "otherwise re-run with current `scripts/run_complete_experiment.py` to capture the "
            "command._\n\n"
        )

    ts = inv.get("timestamp", "N/A")
    cwd = inv.get("cwd", "")
    pr = inv.get("project_root", "")
    posix = inv.get("command_posix", "")
    win = inv.get("command_windows_cmd", "")

    cd_bash = f"cd {shlex.quote(pr)}" if pr else "# cd <repository-root>"
    cd_cmd = f'cd /d "{pr}"' if pr else "REM cd <repository-root>"

    lines = [
        "### Command line (replay this experiment)",
        "",
        f"- **Recorded at:** {ts}",
        f"- **Shell cwd when launched:** `{cwd}`",
        f"- **Repository root (use for relative paths in command):** `{pr}`",
        "",
        "**Unix / Git Bash / WSL**",
        "",
        "```bash",
        cd_bash,
        posix or "# (command not recorded)",
        "```",
        "",
        "**Windows CMD**",
        "",
        "```cmd",
        cd_cmd,
        win or "REM (command not recorded)",
        "```",
        "",
        "**Machine-readable:** `experiment_summary.json` → `invocation_latest` (`argv`, "
        "`command_posix`, `command_windows_cmd`).",
        "",
    ]

    exp_resolved = experiment_dir.resolve()
    for key, label in (
        ("rerun_md", "Formatted copy-paste file"),
        ("invocations_log", "Append-only history (all partial runs to this folder)"),
    ):
        p = inv.get(key)
        if not p:
            continue
        try:
            rel = Path(p).resolve().relative_to(exp_resolved)
            lines.append(f"- **{label}:** `{rel}`")
        except (ValueError, OSError):
            lines.append(f"- **{label}:** `{p}`")

    lines.append("")
    return "\n".join(lines)


def generate_experiment_remarks(
    experiment_dir: Path,
    experiment_id: str,
    config: Dict,
    summary: Dict,
    analysis_report_path: Optional[Path] = None,
    comparison_results_path: Optional[Path] = None,
) -> Path:
    """
    Generate experiment remarks markdown file.

    Parameters
    ----------
    experiment_dir : Path
        Experiment output directory
    experiment_id : str
        Experiment ID (e.g., exp_20260105_183401)
    config : Dict
        Experiment configuration
    summary : Dict
        Experiment summary dictionary
    analysis_report_path : Path, optional
        Path to analysis summary report text file
    comparison_results_path : Path, optional
        Path to comparison results JSON file

    Returns
    -------
    Path
        Path to generated remarks markdown file
    """
    remarks_file = experiment_dir / f"EXPERIMENT_REMARKS_{experiment_id}.md"

    # Extract information from summary
    timestamp = summary.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    date_str = timestamp.split("T")[0] if "T" in timestamp else timestamp

    # Extract configuration details
    data_config = config.get("data", {}).get("generation", {})
    data_validation_config = config.get("data", {}).get("validation", {})
    data_preprocessing_config = config.get("data", {}).get("preprocessing", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    loss_config = config.get("loss", {})
    evaluation_config = config.get("evaluation", {})

    # Extract reproducibility information
    reproducibility = summary.get("reproducibility", {})
    data_info = summary.get("data", {})

    # Extract results
    pinn_results = summary.get("pinn", {})
    ml_baseline_results = summary.get("ml_baseline", {})
    comparison_results = summary.get("comparison", {})

    # Extract PINN metrics
    pinn_eval = pinn_results.get("evaluation", {})
    pinn_metrics = pinn_eval.get("metrics", {}) if isinstance(pinn_eval, dict) else {}
    if isinstance(pinn_metrics, dict) and "metrics" in pinn_metrics:
        pinn_metrics = pinn_metrics["metrics"]

    # Extract ML baseline metrics
    ml_metrics = {}
    if ml_baseline_results:
        for model_type, model_info in ml_baseline_results.items():
            # Try evaluation.metrics first, then evaluation directly, then metrics directly
            if "evaluation" in model_info:
                eval_data = model_info["evaluation"]
                if isinstance(eval_data, dict):
                    if "metrics" in eval_data:
                        ml_metrics[model_type] = eval_data["metrics"]
                    elif any(
                        key.startswith(("delta_", "omega_", "rmse", "mae", "r2"))
                        for key in eval_data.keys()
                    ):
                        # Metrics are directly in evaluation dict
                        ml_metrics[model_type] = eval_data
            # Also check if metrics are directly in model_info
            elif "metrics" in model_info:
                ml_metrics[model_type] = model_info["metrics"]

    # Read analysis report if available
    analysis_text = ""
    if analysis_report_path and analysis_report_path.exists():
        try:
            with open(analysis_report_path, "r", encoding="utf-8") as f:
                analysis_text = f.read()
        except Exception:
            analysis_text = ""

    # Read comparison results if available
    comparison_data = None
    if comparison_results_path and comparison_results_path.exists():
        try:
            with open(comparison_results_path, "r", encoding="utf-8") as f:
                comparison_data = json.load(f)
        except Exception:
            comparison_data = None

    # Generate markdown content
    md_content = f"""# Experiment Remarks: {experiment_id}

**Date**: {date_str}  
**Experiment ID**: {experiment_id}  
**Status**: {'✅ Completed' if pinn_results else '⚠️ Partial'}  
**Config Path**: {summary.get('config_path', 'N/A')}

"""

    md_content += _format_invocation_section(summary, experiment_dir)

    md_content += """
---

## 🔬 Reproducibility Information

"""

    # Add reproducibility section
    if reproducibility:
        md_content += _format_reproducibility(reproducibility)
    else:
        md_content += "*Reproducibility information not available.*\n\n"

    # Format experiment overview with actual values
    hidden_dims = model_config.get("hidden_dims", [])
    epochs = training_config.get("epochs", "N/A")
    lambda_data = loss_config.get("lambda_data", "N/A")
    lambda_physics = loss_config.get("lambda_physics", "N/A")
    lambda_ic = loss_config.get("lambda_ic", "N/A")
    use_normalized_loss = loss_config.get("use_normalized_loss", False)
    sampling_strategy = data_config.get("sampling_strategy", "N/A")
    n_samples = data_config.get("n_samples", "N/A")

    # Format sampling strategy safely
    sampling_strategy_str = (
        sampling_strategy.upper() if isinstance(sampling_strategy, str) else str(sampling_strategy)
    )

    md_content += f"""---
    
## 📊 Executive Summary

### Experiment Overview

This experiment was run with the following configuration:
- **Model Architecture**: {_format_list(hidden_dims)}
- **Training Epochs**: {epochs}
- **Loss Weights**: λ_data={lambda_data}, λ_physics={lambda_physics}, λ_ic={lambda_ic}
- **Normalized Loss**: {'✅ Enabled' if use_normalized_loss else '❌ Disabled'}
- **Sampling Strategy**: {sampling_strategy_str}
- **Number of Samples**: {n_samples}

---

## 📈 Dataset Overview

"""

    # Add data file information
    if data_info:
        md_content += _format_data_files(data_info)
        md_content += "\n"

    # Add analysis report if available
    if analysis_text:
        md_content += _extract_analysis_summary(analysis_text)
    else:
        md_content += "*Analysis report not available.*\n\n"

    md_content += """
---

## 🎯 Model Performance

"""

    # Add Performance Comparison Table if both PINN and ML baseline metrics are available
    if pinn_metrics and ml_metrics:
        md_content += _format_performance_comparison_table(pinn_metrics, ml_metrics)
        md_content += "\n---\n\n### Detailed Performance Assessment\n\n"

    md_content += "#### PINN Model Results\n\n"

    if pinn_metrics:
        md_content += _format_pinn_metrics(pinn_metrics)
        # Add performance assessment
        md_content += _format_performance_assessment("PINN", pinn_metrics)
    else:
        md_content += "*PINN evaluation results not available.*\n\n"

    # Add training info
    pinn_training = pinn_results.get("training", {})
    # If training info doesn't have best_epoch, extract from training_history
    if pinn_training and "best_epoch" not in pinn_training and "training_history" in pinn_training:
        history = pinn_training.get("training_history", {})
        if isinstance(history, dict) and "train_losses" in history and "val_losses" in history:
            train_losses = history.get("train_losses", [])
            val_losses = history.get("val_losses", [])
            if train_losses and val_losses:
                best_val_loss = min(val_losses)
                best_epoch = val_losses.index(best_val_loss)
                final_train_loss = train_losses[-1] if train_losses else None
                pinn_training.update(
                    {
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                        "final_train_loss": final_train_loss,
                        "total_epochs": len(train_losses),
                    }
                )
    if pinn_training:
        md_content += _format_training_info("PINN", pinn_training, experiment_dir)

    # Extract ML baseline training info for both detailed analysis and comparison
    ml_training_info = {}
    if ml_baseline_results:
        for model_type, model_info in ml_baseline_results.items():
            training_data = None
            history = None

            # Check if training info exists
            if "training" in model_info:
                training_data = model_info["training"]
                # Check if training_history is inside training dict
                if isinstance(training_data, dict) and "training_history" in training_data:
                    history = training_data.get("training_history")
                elif (
                    isinstance(training_data, dict)
                    and "train_losses" in training_data
                    and "val_losses" in training_data
                ):
                    # Training data itself contains the history
                    history = training_data
            elif "training_history" in model_info:
                # Training history at top level
                history = model_info["training_history"]

            # If we have training data with all required fields, use it
            if training_data and isinstance(training_data, dict):
                if "best_epoch" in training_data or "best_val_loss" in training_data:
                    ml_training_info[model_type] = training_data
                    continue

            # Extract training info from training_history if available
            if (
                history
                and isinstance(history, dict)
                and "train_losses" in history
                and "val_losses" in history
            ):
                train_losses = history.get("train_losses", [])
                val_losses = history.get("val_losses", [])
                if train_losses and val_losses:
                    # Calculate best epoch and losses
                    best_val_loss = min(val_losses)
                    best_epoch = val_losses.index(best_val_loss)
                    final_train_loss = train_losses[-1] if train_losses else None
                    ml_training_info[model_type] = {
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                        "final_train_loss": final_train_loss,
                        "total_epochs": len(train_losses),
                        "training_history": history,  # Keep history for detailed analysis
                    }

    # Add ML Baseline training info right after PINN training (before comparison table)
    if ml_training_info:
        for model_type, ml_training in ml_training_info.items():
            # Format model name to match the format used in ML Baseline Results section
            model_display_name = model_type.replace("_", " ").title()
            md_content += _format_training_info(
                f"ML Baseline ({model_display_name})", ml_training, experiment_dir
            )

    # Add training comparison table if both PINN and ML baseline training info available
    if pinn_training and ml_training_info:
        md_content += _format_training_comparison_table(
            pinn_training, ml_training_info, experiment_dir
        )
        # Add ML Baseline Training Issue section if detected
        ml_training_issue = _detect_ml_baseline_training_issue(ml_training_info, experiment_dir)
        if ml_training_issue:
            md_content += ml_training_issue

    md_content += """
### ML Baseline Results

"""

    if ml_metrics:
        for model_type, metrics in ml_metrics.items():
            md_content += f"#### {model_type.replace('_', ' ').title()}\n\n"
            md_content += _format_ml_metrics(metrics)
            # Add performance assessment
            md_content += _format_performance_assessment(f"ML Baseline ({model_type})", metrics)
            # Note: Training info is already added above in the training section

        # Add summary if both models available
        if pinn_metrics and ml_metrics:
            md_content += "\n#### Summary\n\n"
            md_content += _format_performance_summary(pinn_metrics, ml_metrics)
    else:
        md_content += "*ML baseline evaluation results not available.*\n\n"

    md_content += """
---

## 🔍 Model Comparison

"""

    if comparison_data:
        md_content += _format_comparison_results(comparison_data)
    elif comparison_results:
        md_content += _format_comparison_from_summary(comparison_results)
    else:
        md_content += "*Comparison results not available.*\n\n"

    md_content += """
---

## ⚠️ Overfitting Analysis

"""

    # Add overfitting analysis section
    overfitting_analysis = _format_overfitting_analysis(
        pinn_training, ml_training_info, model_config, training_config, experiment_dir
    )
    if overfitting_analysis:
        md_content += overfitting_analysis
    else:
        md_content += "*Overfitting analysis not available (training history required).*\n\n"

    md_content += """
---

## 📝 Key Findings

### Strengths

"""

    # Auto-generate findings based on metrics
    findings = _generate_findings(pinn_metrics, ml_metrics, comparison_data or comparison_results)
    md_content += findings

    md_content += """
---

## ⚙️ Configuration Details

### Data Generation

"""
    md_content += _format_data_config(
        data_config, data_validation_config, data_preprocessing_config
    )

    md_content += """
### Model Architecture

"""
    md_content += _format_model_config(model_config)

    md_content += """
### Training Configuration

"""
    md_content += _format_training_config(training_config, loss_config)

    md_content += """
### Evaluation Configuration

"""
    md_content += _format_evaluation_config(evaluation_config)

    md_content += """
---

## 📁 Output Files

- **Experiment Summary**: `experiment_summary.json`
- **Configuration**: `config.yaml`
- **Analysis Report**: `analysis/analysis_summary_report_*.txt`
- **Comparison Results**: `comparison/comparison_results.json`
- **PINN Model**: `pinn/model/best_model_*.pth`
- **ML Baseline Model**: `ml_baseline/*/model/best_model_*.pth`

---

## 🔗 Related Documentation

- [Complete Experiment Guide](../../docs/05_experiments/complete_experiment.md)
- [Model Evaluation Guide](../../docs/04_evaluation/model_evaluation.md)
- [Troubleshooting Guide](../../docs/09_troubleshooting/common_errors.md)

---

"""
    last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_content += f"**Last Updated**: {last_updated}  \n"
    md_content += (
        "**Generated Automatically**: This file was generated by the experiment workflow.\n"
    )

    # Write to file
    with open(remarks_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    return remarks_file


def _format_list(items):
    """Format list as string."""
    if not items:
        return "N/A"
    return " × ".join(map(str, items))


def _extract_analysis_summary(analysis_text: str) -> str:
    """Extract key information from analysis report."""
    lines = analysis_text.split("\n")
    summary_lines = []
    in_section = False
    current_section = None

    for line in lines:
        # Detect section headers
        if "DATASET OVERVIEW" in line:
            in_section = True
            current_section = "Dataset Overview"
            summary_lines.append(f"\n### {current_section}\n\n")
        elif "PARAMETER RANGES" in line:
            in_section = True
            current_section = "Parameter Ranges"
            summary_lines.append(f"\n### {current_section}\n\n")
        elif "STABILITY DISTRIBUTION" in line:
            in_section = True
            current_section = "Stability Distribution"
            summary_lines.append(f"\n### {current_section}\n\n")
        elif "CCT STATISTICS" in line:
            in_section = True
            current_section = "CCT Statistics"
            summary_lines.append(f"\n### {current_section}\n\n")
        elif "CCT CORRELATIONS" in line:
            in_section = True
            current_section = "CCT Correlations"
            summary_lines.append(f"\n### {current_section}\n\n")
        elif "TRAJECTORY STATISTICS" in line:
            in_section = True
            current_section = "Trajectory Statistics"
            summary_lines.append(f"\n### {current_section}\n\n")
        elif (
            in_section
            and line.strip()
            and not line.strip().startswith("=")
            and not line.strip().startswith("-")
        ):
            # Format data lines nicely
            formatted_line = line.strip()
            if formatted_line:
                # Convert to markdown list items if appropriate
                if ":" in formatted_line and not formatted_line.startswith("-"):
                    parts = formatted_line.split(":", 1)
                    if len(parts) == 2:
                        summary_lines.append(f"- **{parts[0].strip()}**: {parts[1].strip()}\n")
                    else:
                        summary_lines.append(f"{formatted_line}\n")
                else:
                    summary_lines.append(f"{formatted_line}\n")
        elif line.strip().startswith("=") and in_section:
            in_section = False
            current_section = None

    return "".join(summary_lines) if summary_lines else analysis_text[:500] + "\n\n"


def _format_pinn_metrics(metrics: Dict) -> str:
    """Format PINN metrics."""
    content = ""

    if "rmse_delta" in metrics:
        content += f"- **RMSE Delta**: {metrics['rmse_delta']:.4f} rad\n"
    if "rmse_omega" in metrics:
        content += f"- **RMSE Omega**: {metrics['rmse_omega']:.4f} pu\n"
    if "mae_delta" in metrics:
        content += f"- **MAE Delta**: {metrics['mae_delta']:.4f} rad\n"
    if "mae_omega" in metrics:
        content += f"- **MAE Omega**: {metrics['mae_omega']:.4f} pu\n"
    if "r2_delta" in metrics:
        r2_d = metrics["r2_delta"]
        status = "✅" if r2_d > 0.5 else "⚠️" if r2_d > 0 else "❌"
        content += f"- **R² Delta**: {r2_d:.4f} {status}\n"
    if "r2_omega" in metrics:
        r2_o = metrics["r2_omega"]
        status = "✅" if r2_o > 0.5 else "⚠️" if r2_o > 0 else "❌"
        content += f"- **R² Omega**: {r2_o:.4f} {status}\n"

    return content + "\n"


def _format_ml_metrics(metrics: Dict) -> str:
    """Format ML baseline metrics."""
    content = ""

    # Handle both naming conventions: rmse_delta/delta_rmse, etc.
    rmse_d = metrics.get("rmse_delta") or metrics.get("delta_rmse")
    rmse_o = metrics.get("rmse_omega") or metrics.get("omega_rmse")
    mae_d = metrics.get("mae_delta") or metrics.get("delta_mae")
    mae_o = metrics.get("mae_omega") or metrics.get("omega_mae")
    r2_d = metrics.get("r2_delta") or metrics.get("delta_r2")
    r2_o = metrics.get("r2_omega") or metrics.get("omega_r2")
    stability = metrics.get("stability_accuracy") or metrics.get(
        "stability_classification_accuracy"
    )

    if rmse_d is not None:
        content += f"- **RMSE Delta**: {rmse_d:.4f} rad\n"
    if rmse_o is not None:
        content += f"- **RMSE Omega**: {rmse_o:.4f} pu\n"
    if mae_d is not None:
        content += f"- **MAE Delta**: {mae_d:.4f} rad\n"
    if mae_o is not None:
        content += f"- **MAE Omega**: {mae_o:.4f} pu\n"
    if r2_d is not None:
        status = "✅" if r2_d > 0.5 else "⚠️" if r2_d > 0 else "❌"
        content += f"- **R² Delta**: {r2_d:.4f} {status}\n"
    if r2_o is not None:
        status = "✅" if r2_o > 0.5 else "⚠️" if r2_o > 0 else "❌"
        content += f"- **R² Omega**: {r2_o:.4f} {status}\n"

    return content + "\n"


def _format_training_info(
    model_name: str, training_info: Dict, experiment_dir: Optional[Path] = None
) -> str:
    """Format training information with detailed loss analysis."""
    content = f"\n#### {model_name} Training\n\n"

    if "best_epoch" in training_info:
        content += f"- **Best Epoch**: {training_info['best_epoch']}\n"
    if "best_val_loss" in training_info:
        content += f"- **Best Validation Loss**: {training_info['best_val_loss']:.4f}\n"
    if "final_train_loss" in training_info:
        content += f"- **Final Training Loss**: {training_info['final_train_loss']:.4f}\n"

    # Add detailed loss analysis if training history is available
    loss_analysis = _analyze_training_losses(model_name, training_info, experiment_dir)
    if loss_analysis:
        content += "\n" + loss_analysis

    return content + "\n"


def _analyze_training_losses(
    model_name: str, training_info: Dict, experiment_dir: Optional[Path] = None
) -> str:
    """Analyze training losses to detect overfitting and provide recommendations."""
    content = ""

    # Try to load training history from checkpoint
    train_losses = []
    val_losses = []

    if experiment_dir:
        # Try to find checkpoint file
        if "pinn" in model_name.lower():
            checkpoint_files = list(experiment_dir.glob("pinn/best_model_*.pth"))
        else:
            # ML baseline - extract model type from model_name if available
            model_type = None
            if "(" in model_name and ")" in model_name:
                # Extract model type from "ML Baseline (standard_nn)" format
                model_type = model_name.split("(")[1].split(")")[0].strip()

            if model_type:
                checkpoint_files = list(
                    experiment_dir.glob(f"ml_baseline/{model_type}/best_model_*.pth")
                )
                if not checkpoint_files:
                    checkpoint_files = list(
                        experiment_dir.glob(f"ml_baseline/{model_type}/model.pth")
                    )
            else:
                # Try all ML baseline models
                checkpoint_files = list(experiment_dir.glob("ml_baseline/*/best_model_*.pth"))
                if not checkpoint_files:
                    checkpoint_files = list(experiment_dir.glob("ml_baseline/*/model.pth"))

        if checkpoint_files:
            try:
                checkpoint = torch.load(
                    str(checkpoint_files[0]), map_location="cpu", weights_only=False
                )
                history = checkpoint.get("training_history", {})
                train_losses = history.get("train_losses", [])
                val_losses = history.get("val_losses", [])
            except Exception:
                pass

    # If no history from checkpoint, try to extract from training_info
    if not train_losses and "training_history" in training_info:
        history = training_info.get("training_history", {})
        train_losses = history.get("train_losses", [])
        val_losses = history.get("val_losses", [])

    if not train_losses or not val_losses:
        return ""  # No history available

    # Analyze losses
    best_epoch = training_info.get("best_epoch", len(val_losses) - 1)
    best_val_loss = training_info.get("best_val_loss", min(val_losses) if val_losses else 0)
    final_train_loss = training_info.get(
        "final_train_loss", train_losses[-1] if train_losses else 0
    )

    # Calculate metrics
    initial_train_loss = train_losses[0] if train_losses else 0
    initial_val_loss = val_losses[0] if val_losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0

    # Loss reduction
    train_reduction = (
        ((initial_train_loss - final_train_loss) / initial_train_loss * 100)
        if initial_train_loss > 0
        else 0
    )
    val_reduction = (
        ((initial_val_loss - best_val_loss) / initial_val_loss * 100) if initial_val_loss > 0 else 0
    )

    # Overfitting detection
    if best_epoch < len(train_losses) and best_epoch < len(val_losses):
        train_loss_at_best = train_losses[best_epoch]
        val_loss_at_best = val_losses[best_epoch]
        overfitting_gap = val_loss_at_best - train_loss_at_best
        overfitting_ratio = (
            (overfitting_gap / train_loss_at_best * 100) if train_loss_at_best > 0 else 0
        )
    else:
        overfitting_gap = final_val_loss - final_train_loss
        overfitting_ratio = (
            (overfitting_gap / final_train_loss * 100) if final_train_loss > 0 else 0
        )

    # Build analysis content
    content += "**Training Loss Analysis:**\n\n"

    # Loss progression
    content += "**Loss Progression:**\n"
    content += f"- Initial Training Loss: {initial_train_loss:.2f}\n"
    content += f"- Initial Validation Loss: {initial_val_loss:.2f}\n"
    content += f"- Final Training Loss: {final_train_loss:.2f} ({train_reduction:.1f}% reduction)\n"
    content += f"- Best Validation Loss: {best_val_loss:.2f} at epoch {best_epoch} ({val_reduction:.1f}% reduction)\n"
    content += f"- Final Validation Loss: {final_val_loss:.2f}\n\n"

    # Overfitting detection
    content += "**Overfitting Analysis:**\n"
    if best_epoch < len(train_losses) and best_epoch < len(val_losses):
        content += f"- Training Loss at Best Epoch: {train_losses[best_epoch]:.2f}\n"
        content += f"- Validation Loss at Best Epoch: {val_losses[best_epoch]:.2f}\n"
        content += f"- Gap (Val - Train): {overfitting_gap:.2f} ({overfitting_ratio:.1f}% higher)\n"

    # Determine overfitting severity
    if overfitting_ratio > 50:
        content += f"- ⚠️ **Severe Overfitting Detected**: Validation loss is {overfitting_ratio:.1f}% higher than training loss\n"
    elif overfitting_ratio > 30:
        content += f"- ⚠️ **Moderate Overfitting Detected**: Validation loss is {overfitting_ratio:.1f}% higher than training loss\n"
    elif overfitting_ratio > 10:
        content += f"- ✅ **Mild Overfitting**: Validation loss is {overfitting_ratio:.1f}% higher (acceptable)\n"
    else:
        content += (
            f"- ✅ **Good Generalization**: Training and validation losses are well-balanced\n"
        )

    content += "\n"

    # Recommendations
    if overfitting_ratio > 30:
        content += "**Recommendations to Reduce Overfitting:**\n"
        content += "- Reduce early stopping patience (try 10-20 epochs instead of 200)\n"
        content += "- Increase regularization: add dropout (0.1-0.3) or increase weight_decay (1e-4 to 1e-3)\n"
        content += "- Reduce model capacity: use smaller hidden dimensions\n"
        content += "- Check data splits: ensure train/val distributions are similar\n"
        content += "- Monitor training curves: stop when validation loss plateaus or increases\n\n"

    # Loss progression summary
    if len(val_losses) > 10:
        content += "**Loss Progression Summary (every 50 epochs):**\n"
        step = max(1, len(val_losses) // 10)
        for i in range(0, len(val_losses), step):
            if i < len(train_losses) and i < len(val_losses):
                content += f"- Epoch {i}: Train={train_losses[i]:.2f}, Val={val_losses[i]:.2f}\n"
        content += "\n"

    return content


def _format_training_comparison_table(
    pinn_training: Dict, ml_training_info: Dict, experiment_dir: Optional[Path] = None
) -> str:
    """Format training comparison table between PINN and ML baseline."""
    content = "\n---\n\n### Training Comparison Table\n\n"
    content += "| Training Metric | PINN Model | ML Baseline (Standard NN) | Notes |\n"
    content += "|----------------|------------|--------------------------|-------|\n"

    # Extract metrics
    pinn_epochs = pinn_training.get("total_epochs", pinn_training.get("epochs", "N/A"))
    pinn_best_epoch = pinn_training.get("best_epoch", "N/A")
    pinn_best_val = pinn_training.get("best_val_loss", "N/A")
    pinn_final_train = pinn_training.get("final_train_loss", "N/A")

    # Get first ML baseline (usually standard_nn)
    ml_model_type = list(ml_training_info.keys())[0] if ml_training_info else "standard_nn"
    ml_training = ml_training_info.get(ml_model_type, {})
    ml_epochs = ml_training.get("total_epochs", ml_training.get("epochs", "N/A"))
    ml_best_epoch = ml_training.get("best_epoch", "N/A")
    ml_best_val = ml_training.get("best_val_loss", "N/A")
    ml_final_train = ml_training.get("final_train_loss", "N/A")

    # Format values
    def fmt_val(val, decimals=2):
        if isinstance(val, (int, float)):
            return f"{val:.{decimals}f}" if decimals > 0 else str(int(val))
        return str(val)

    # Total epochs
    content += f"| **Total Epochs Trained** | {pinn_epochs} | {ml_epochs} | "
    if isinstance(pinn_epochs, (int, float)) and isinstance(ml_epochs, (int, float)):
        if ml_epochs < pinn_epochs:
            content += "ML Baseline trained fewer epochs |\n"
        else:
            content += "Both models trained similar epochs |\n"
    else:
        content += " |\n"

    # Best epoch
    content += f"| **Best Epoch** | {pinn_best_epoch} | {ml_best_epoch} | "
    if isinstance(pinn_best_epoch, int) and isinstance(ml_best_epoch, int):
        if ml_best_epoch == 0:
            content += "⚠️ ML Baseline best at start (epoch 0) |\n"
        elif isinstance(pinn_epochs, int) and pinn_best_epoch > pinn_epochs * 0.8:
            content += "PINN best late in training |\n"
        else:
            content += " |\n"
    else:
        content += " |\n"

    # Best validation loss
    content += f"| **Best Validation Loss** | {fmt_val(pinn_best_val)} | {fmt_val(ml_best_val)} | "
    if isinstance(pinn_best_val, (int, float)) and isinstance(ml_best_val, (int, float)):
        if ml_best_val < pinn_best_val * 0.1:  # ML baseline much lower
            content += "⚠️ ML Baseline has much lower validation loss (different loss scales) |\n"
        else:
            content += " |\n"
    else:
        content += " |\n"

    # Final training loss
    content += (
        f"| **Final Training Loss** | {fmt_val(pinn_final_train)} | {fmt_val(ml_final_train)} | "
    )
    if isinstance(pinn_final_train, (int, float)) and isinstance(ml_final_train, (int, float)):
        if ml_final_train < pinn_final_train * 0.1:  # ML baseline much lower
            content += "⚠️ ML Baseline has much lower final loss (different loss scales) |\n"
        else:
            content += " |\n"
    else:
        content += " |\n"

    # Loss reduction
    if experiment_dir:
        # Try to calculate loss reduction from training history
        try:
            checkpoint_files = list(experiment_dir.glob("pinn/best_model_*.pth"))
            if checkpoint_files:
                checkpoint = torch.load(
                    str(checkpoint_files[0]), map_location="cpu", weights_only=False
                )
                history = checkpoint.get("training_history", {})
                pinn_train_losses = history.get("train_losses", [])
                if pinn_train_losses:
                    pinn_reduction = (
                        (
                            (pinn_train_losses[0] - pinn_train_losses[-1])
                            / pinn_train_losses[0]
                            * 100
                        )
                        if pinn_train_losses[0] > 0
                        else 0
                    )
                    content += f"| **Loss Reduction** | {pinn_reduction:.1f}% | "

                    # ML baseline loss reduction
                    ml_checkpoint_files = list(
                        experiment_dir.glob("ml_baseline/*/best_model_*.pth")
                    )
                    if not ml_checkpoint_files:
                        ml_checkpoint_files = list(experiment_dir.glob("ml_baseline/*/model.pth"))
                    if ml_checkpoint_files:
                        ml_checkpoint = torch.load(
                            str(ml_checkpoint_files[0]), map_location="cpu", weights_only=False
                        )
                        ml_history = ml_checkpoint.get("training_history", {})
                        ml_train_losses = ml_history.get("train_losses", [])
                        if ml_train_losses:
                            ml_reduction = (
                                (
                                    (ml_train_losses[0] - ml_train_losses[-1])
                                    / ml_train_losses[0]
                                    * 100
                                )
                                if ml_train_losses[0] > 0
                                else 0
                            )
                            content += f"{ml_reduction:.1f}% | Percentage reduction from first to last epoch |\n"
                        else:
                            content += "N/A | |\n"
                    else:
                        content += "N/A | |\n"
                else:
                    content += "| **Loss Reduction** | N/A | N/A | |\n"
            else:
                content += "| **Loss Reduction** | N/A | N/A | |\n"
        except Exception:
            content += "| **Loss Reduction** | N/A | N/A | |\n"
    else:
        content += "| **Loss Reduction** | N/A | N/A | |\n"

    content += "\n**Training Observations:**\n"

    # Add observations
    if isinstance(pinn_best_epoch, int) and isinstance(ml_best_epoch, int):
        if ml_best_epoch == 0:
            content += "- **ML Baseline**: ⚠️ **Training Issue Detected** - Best validation loss at epoch 0, suggesting overfitting or training instability\n"
        if isinstance(pinn_epochs, int) and pinn_best_epoch > pinn_epochs * 0.8:
            content += "- **PINN**: Required full training, best performance late in training\n"

    if isinstance(pinn_best_val, (int, float)) and isinstance(ml_best_val, (int, float)):
        if ml_best_val < pinn_best_val * 0.1:
            content += "- **Loss Scale Difference**: ML Baseline losses are ~10-20x smaller than PINN losses (different loss functions/scales)\n"

    content += "\n"

    return content


def _format_performance_comparison_table(pinn_metrics: Dict, ml_metrics: Dict) -> str:
    """Format performance comparison table with target ranges and winners."""
    content = "### Performance Comparison Table\n\n"
    content += "| Metric | PINN Model | ML Baseline (Standard NN) | Target Range | Winner |\n"
    content += "|--------|------------|---------------------------|--------------|--------|\n"

    # Get first ML baseline metrics (usually standard_nn)
    ml_model_type = (
        list(ml_metrics.keys())[0] if isinstance(ml_metrics, dict) and ml_metrics else None
    )
    ml_metric = (
        ml_metrics.get(ml_model_type, {})
        if ml_model_type
        else ml_metrics
        if isinstance(ml_metrics, dict)
        else {}
    )

    # Helper function to get metric value (handles both naming conventions)
    def get_metric(metrics_dict, key1, key2=None):
        """Get metric value handling both naming conventions."""
        if key2 is None:
            # Generate alternative key name
            if "rmse_delta" in key1:
                key2 = "delta_rmse"
            elif "rmse_omega" in key1:
                key2 = "omega_rmse"
            elif "mae_delta" in key1:
                key2 = "delta_mae"
            elif "mae_omega" in key1:
                key2 = "omega_mae"
            elif "r2_delta" in key1:
                key2 = "delta_r2"
            elif "r2_omega" in key1:
                key2 = "omega_r2"
            elif "stability_accuracy" in key1:
                key2 = "stability_classification_accuracy"
        val = metrics_dict.get(key1)
        if val is None and key2:
            val = metrics_dict.get(key2)
        return val

    # Helper function to format with status
    def fmt_with_status(value, good_threshold, excellent_threshold, reverse=False):
        if not isinstance(value, (int, float)):
            return f"{value} ⚠️"
        if reverse:  # For R², higher is better
            status = (
                "✅" if value >= excellent_threshold else "✅" if value >= good_threshold else "⚠️"
            )
        else:  # For RMSE/MAE, lower is better
            status = (
                "✅" if value <= excellent_threshold else "✅" if value <= good_threshold else "⚠️"
            )
        return f"{value:.4f} {status}"

    # RMSE Delta
    pinn_rmse_d = get_metric(pinn_metrics, "rmse_delta", "delta_rmse")
    ml_rmse_d = (
        get_metric(ml_metric, "rmse_delta", "delta_rmse") if isinstance(ml_metric, dict) else "N/A"
    )
    pinn_fmt = (
        fmt_with_status(pinn_rmse_d, 0.4, 0.1, False)
        if isinstance(pinn_rmse_d, (int, float))
        else f"{pinn_rmse_d} ⚠️"
    )
    ml_fmt = (
        fmt_with_status(ml_rmse_d, 0.4, 0.1, False)
        if isinstance(ml_rmse_d, (int, float))
        else f"{ml_rmse_d} ⚠️"
    )
    winner = (
        "ML Baseline"
        if isinstance(pinn_rmse_d, (int, float))
        and isinstance(ml_rmse_d, (int, float))
        and ml_rmse_d < pinn_rmse_d
        else "PINN"
        if isinstance(pinn_rmse_d, (int, float)) and isinstance(ml_rmse_d, (int, float))
        else "N/A"
    )
    content += f"| **RMSE Delta** | {pinn_fmt} | {ml_fmt} | <0.4 rad (good), <0.1 rad (excellent) | {winner} |\n"

    # RMSE Omega
    pinn_rmse_o = get_metric(pinn_metrics, "rmse_omega", "omega_rmse")
    ml_rmse_o = (
        get_metric(ml_metric, "rmse_omega", "omega_rmse") if isinstance(ml_metric, dict) else "N/A"
    )
    pinn_fmt = (
        fmt_with_status(pinn_rmse_o, 0.015, 0.010, False)
        if isinstance(pinn_rmse_o, (int, float))
        else f"{pinn_rmse_o} ⚠️"
    )
    ml_fmt = (
        fmt_with_status(ml_rmse_o, 0.015, 0.010, False)
        if isinstance(ml_rmse_o, (int, float))
        else f"{ml_rmse_o} ⚠️"
    )
    winner = (
        "PINN"
        if isinstance(pinn_rmse_o, (int, float))
        and isinstance(ml_rmse_o, (int, float))
        and pinn_rmse_o < ml_rmse_o
        else "ML Baseline"
        if isinstance(pinn_rmse_o, (int, float)) and isinstance(ml_rmse_o, (int, float))
        else "N/A"
    )
    content += f"| **RMSE Omega** | {pinn_fmt} | {ml_fmt} | <0.015 pu (good), <0.010 pu (excellent) | {winner} |\n"

    # MAE Delta
    pinn_mae_d = get_metric(pinn_metrics, "mae_delta", "delta_mae")
    ml_mae_d = (
        get_metric(ml_metric, "mae_delta", "delta_mae") if isinstance(ml_metric, dict) else "N/A"
    )
    pinn_fmt = (
        fmt_with_status(pinn_mae_d, 0.3, 0.15, False)
        if isinstance(pinn_mae_d, (int, float))
        else f"{pinn_mae_d} ⚠️"
    )
    ml_fmt = (
        fmt_with_status(ml_mae_d, 0.3, 0.15, False)
        if isinstance(ml_mae_d, (int, float))
        else f"{ml_mae_d} ⚠️"
    )
    winner = (
        "ML Baseline"
        if isinstance(pinn_mae_d, (int, float))
        and isinstance(ml_mae_d, (int, float))
        and ml_mae_d < pinn_mae_d
        else "PINN"
        if isinstance(pinn_mae_d, (int, float)) and isinstance(ml_mae_d, (int, float))
        else "N/A"
    )
    content += f"| **MAE Delta** | {pinn_fmt} | {ml_fmt} | <0.3 rad (good), <0.15 rad (excellent) | {winner} |\n"

    # MAE Omega
    pinn_mae_o = get_metric(pinn_metrics, "mae_omega", "omega_mae")
    ml_mae_o = (
        get_metric(ml_metric, "mae_omega", "omega_mae") if isinstance(ml_metric, dict) else "N/A"
    )
    pinn_fmt = (
        fmt_with_status(pinn_mae_o, 0.012, 0.008, False)
        if isinstance(pinn_mae_o, (int, float))
        else f"{pinn_mae_o} ⚠️"
    )
    ml_fmt = (
        fmt_with_status(ml_mae_o, 0.012, 0.008, False)
        if isinstance(ml_mae_o, (int, float))
        else f"{ml_mae_o} ⚠️"
    )
    winner = (
        "PINN"
        if isinstance(pinn_mae_o, (int, float))
        and isinstance(ml_mae_o, (int, float))
        and pinn_mae_o < ml_mae_o
        else "ML Baseline"
        if isinstance(pinn_mae_o, (int, float)) and isinstance(ml_mae_o, (int, float))
        else "N/A"
    )
    content += f"| **MAE Omega** | {pinn_fmt} | {ml_fmt} | <0.012 pu (good), <0.008 pu (excellent) | {winner} |\n"

    # R² Delta
    pinn_r2_d = get_metric(pinn_metrics, "r2_delta", "delta_r2")
    ml_r2_d = (
        get_metric(ml_metric, "r2_delta", "delta_r2") if isinstance(ml_metric, dict) else "N/A"
    )
    pinn_fmt = (
        fmt_with_status(pinn_r2_d, 0.8, 0.95, True)
        if isinstance(pinn_r2_d, (int, float))
        else f"{pinn_r2_d} ⚠️"
    )
    ml_fmt = (
        fmt_with_status(ml_r2_d, 0.8, 0.95, True)
        if isinstance(ml_r2_d, (int, float))
        else f"{ml_r2_d} ⚠️"
    )
    winner = (
        "ML Baseline"
        if isinstance(pinn_r2_d, (int, float))
        and isinstance(ml_r2_d, (int, float))
        and ml_r2_d > pinn_r2_d
        else "PINN"
        if isinstance(pinn_r2_d, (int, float)) and isinstance(ml_r2_d, (int, float))
        else "N/A"
    )
    content += (
        f"| **R² Delta** | {pinn_fmt} | {ml_fmt} | >0.8 (good), >0.95 (excellent) | {winner} |\n"
    )

    # R² Omega
    pinn_r2_o = get_metric(pinn_metrics, "r2_omega", "omega_r2")
    ml_r2_o = (
        get_metric(ml_metric, "r2_omega", "omega_r2") if isinstance(ml_metric, dict) else "N/A"
    )
    pinn_fmt = (
        fmt_with_status(pinn_r2_o, 0.6, 0.85, True)
        if isinstance(pinn_r2_o, (int, float))
        else f"{pinn_r2_o} ⚠️"
    )
    ml_fmt = (
        fmt_with_status(ml_r2_o, 0.6, 0.85, True)
        if isinstance(ml_r2_o, (int, float))
        else f"{ml_r2_o} ⚠️"
    )
    winner = (
        "PINN"
        if isinstance(pinn_r2_o, (int, float))
        and isinstance(ml_r2_o, (int, float))
        and pinn_r2_o > ml_r2_o
        else "ML Baseline"
        if isinstance(pinn_r2_o, (int, float)) and isinstance(ml_r2_o, (int, float))
        else "N/A"
    )
    content += (
        f"| **R² Omega** | {pinn_fmt} | {ml_fmt} | >0.6 (good), >0.85 (excellent) | {winner} |\n"
    )

    # Stability Classification (if available)
    ml_stability = (
        get_metric(ml_metric, "stability_accuracy", "stability_classification_accuracy")
        if isinstance(ml_metric, dict)
        else "N/A"
    )
    if isinstance(ml_stability, (int, float)):
        status = "✅" if ml_stability >= 0.99 else "✅" if ml_stability >= 0.90 else "⚠️"
        content += f"| **Stability Classification** | N/A | {ml_stability*100:.1f}% {status} | >90% (good), >99% (excellent) | ML Baseline |\n"
    else:
        content += (
            f"| **Stability Classification** | N/A | N/A | >90% (good), >99% (excellent) | N/A |\n"
        )

    content += "\n**Legend:**\n"
    content += "- ✅ = Meets target range\n"
    content += "- ⚠️ = Below target range\n"
    content += "- N/A = Not available\n\n"

    return content


def _format_performance_assessment(model_name: str, metrics: Dict) -> str:
    """Format detailed performance assessment for a model."""
    content = "**Performance Assessment:**\n"

    # Handle both naming conventions
    rmse_d = metrics.get("rmse_delta") or metrics.get("delta_rmse")
    rmse_o = metrics.get("rmse_omega") or metrics.get("omega_rmse")
    mae_d = metrics.get("mae_delta") or metrics.get("delta_mae")
    mae_o = metrics.get("mae_omega") or metrics.get("omega_mae")
    r2_d = metrics.get("r2_delta") or metrics.get("delta_r2")
    r2_o = metrics.get("r2_omega") or metrics.get("omega_r2")
    stability = metrics.get("stability_accuracy") or metrics.get(
        "stability_classification_accuracy"
    )

    # Delta assessment
    delta_good = True
    if isinstance(rmse_d, (int, float)) and rmse_d > 0.4:
        delta_good = False
    if isinstance(r2_d, (int, float)) and r2_d < 0.8:
        delta_good = False

    if delta_good and isinstance(rmse_d, (int, float)) and isinstance(r2_d, (int, float)):
        if rmse_d <= 0.1 and r2_d >= 0.95:
            content += f"- ✅ **Delta prediction is excellent**: RMSE ({rmse_d:.4f} rad) and R² ({r2_d:.4f}) exceed target ranges\n"
        else:
            content += f"- ✅ **Delta prediction is good**: RMSE ({rmse_d:.4f} rad) and R² ({r2_d:.4f}) meet target ranges\n"
    else:
        content += f"- ⚠️ **Delta prediction needs improvement**: "
        if isinstance(rmse_d, (int, float)):
            content += f"RMSE ({rmse_d:.4f} rad)"
        if isinstance(r2_d, (int, float)):
            content += (
                f" and R² ({r2_d:.4f})" if isinstance(rmse_d, (int, float)) else f"R² ({r2_d:.4f})"
            )
        content += " are below target ranges\n"

    # Omega assessment
    omega_good = True
    if isinstance(rmse_o, (int, float)) and rmse_o > 0.015:
        omega_good = False
    if isinstance(r2_o, (int, float)) and r2_o < 0.6:
        omega_good = False

    if omega_good and isinstance(rmse_o, (int, float)) and isinstance(r2_o, (int, float)):
        if rmse_o <= 0.010 and r2_o >= 0.85:
            mae_o_str = f"{mae_o:.4f}" if isinstance(mae_o, (int, float)) else "N/A"
            content += f"- ✅ **Omega prediction is excellent**: RMSE ({rmse_o:.4f} pu), MAE ({mae_o_str} pu), and R² ({r2_o:.4f}) exceed target ranges\n"
        else:
            mae_o_str = f"{mae_o:.4f}" if isinstance(mae_o, (int, float)) else "N/A"
            content += f"- ✅ **Omega prediction is acceptable**: RMSE ({rmse_o:.4f} pu), MAE ({mae_o_str} pu), and R² ({r2_o:.4f}) are within good ranges\n"
    else:
        rmse_o_str = f"{rmse_o:.4f}" if isinstance(rmse_o, (int, float)) else "N/A"
        mae_o_str = f"{mae_o:.4f}" if isinstance(mae_o, (int, float)) else "N/A"
        r2_o_str = f"{r2_o:.4f}" if isinstance(r2_o, (int, float)) else "N/A"
        content += f"- ⚠️ **Omega prediction needs improvement**: RMSE ({rmse_o_str} pu), MAE ({mae_o_str} pu), and R² ({r2_o_str}) are below target ranges\n"

    # Stability classification (if available)
    if isinstance(stability, (int, float)):
        if stability >= 0.99:
            content += (
                f"- ✅ **Stability classification**: Excellent accuracy ({stability*100:.1f}%)\n"
            )
        elif stability >= 0.90:
            content += f"- ✅ **Stability classification**: Good accuracy ({stability*100:.1f}%)\n"
        else:
            content += (
                f"- ⚠️ **Stability classification**: Needs improvement ({stability*100:.1f}%)\n"
            )

    # Overall assessment
    if delta_good and omega_good:
        content += f"- ✅ **Overall**: Strong performance with both delta and omega predictions meeting targets\n"
    elif delta_good or omega_good:
        content += f"- ⚠️ **Overall**: Mixed performance with {'omega' if omega_good else 'delta'} prediction meeting targets but {'delta' if omega_good else 'omega'} prediction requiring optimization\n"
    else:
        content += f"- ⚠️ **Overall**: Performance needs improvement for both delta and omega predictions\n"

    content += "\n"
    return content


def _format_performance_summary(pinn_metrics: Dict, ml_metrics: Dict) -> str:
    """Format performance summary comparing PINN and ML baseline."""
    content = ""

    # Get first ML baseline metrics
    ml_model_type = (
        list(ml_metrics.keys())[0] if isinstance(ml_metrics, dict) and ml_metrics else None
    )
    ml_metric = (
        ml_metrics.get(ml_model_type, {})
        if ml_model_type
        else ml_metrics
        if isinstance(ml_metrics, dict)
        else {}
    )

    # Handle both naming conventions
    def get_metric_val(metrics_dict, key1, key2=None):
        """Get metric value handling both naming conventions."""
        if key2 is None:
            if "delta" in key1:
                key2 = (
                    key1.replace("rmse_delta", "delta_rmse")
                    .replace("mae_delta", "delta_mae")
                    .replace("r2_delta", "delta_r2")
                )
            elif "omega" in key1:
                key2 = (
                    key1.replace("rmse_omega", "omega_rmse")
                    .replace("mae_omega", "omega_mae")
                    .replace("r2_omega", "omega_r2")
                )
        return metrics_dict.get(key1) or metrics_dict.get(key2) if key2 else metrics_dict.get(key1)

    pinn_r2_d = get_metric_val(pinn_metrics, "r2_delta", "delta_r2") or 0
    ml_r2_d = (
        get_metric_val(ml_metric, "r2_delta", "delta_r2") if isinstance(ml_metric, dict) else 0
    )
    pinn_rmse_d = get_metric_val(pinn_metrics, "rmse_delta", "delta_rmse") or 0
    ml_rmse_d = (
        get_metric_val(ml_metric, "rmse_delta", "delta_rmse") if isinstance(ml_metric, dict) else 0
    )

    pinn_r2_o = get_metric_val(pinn_metrics, "r2_omega", "omega_r2") or 0
    ml_r2_o = (
        get_metric_val(ml_metric, "r2_omega", "omega_r2") if isinstance(ml_metric, dict) else 0
    )
    pinn_rmse_o = get_metric_val(pinn_metrics, "rmse_omega", "omega_rmse") or 0
    ml_rmse_o = (
        get_metric_val(ml_metric, "rmse_omega", "omega_rmse") if isinstance(ml_metric, dict) else 0
    )

    # Delta comparison
    if isinstance(pinn_r2_d, (int, float)) and isinstance(ml_r2_d, (int, float)):
        if ml_r2_d > pinn_r2_d:
            content += f"- **Delta (Rotor Angle) Prediction**: ML Baseline significantly outperforms PINN (R²: {ml_r2_d:.4f} vs {pinn_r2_d:.4f}, RMSE: {ml_rmse_d:.4f} vs {pinn_rmse_d:.4f} rad)\n"
        else:
            content += f"- **Delta (Rotor Angle) Prediction**: PINN outperforms ML Baseline (R²: {pinn_r2_d:.4f} vs {ml_r2_d:.4f}, RMSE: {pinn_rmse_d:.4f} vs {ml_rmse_d:.4f} rad)\n"

    # Omega comparison
    if isinstance(pinn_r2_o, (int, float)) and isinstance(ml_r2_o, (int, float)):
        if pinn_r2_o > ml_r2_o:
            content += f"- **Omega (Rotor Speed) Prediction**: PINN slightly outperforms ML Baseline (R²: {pinn_r2_o:.4f} vs {ml_r2_o:.4f}, RMSE: {pinn_rmse_o:.4f} vs {ml_rmse_o:.4f} pu)\n"
        else:
            content += f"- **Omega (Rotor Speed) Prediction**: ML Baseline outperforms PINN (R²: {ml_r2_o:.4f} vs {pinn_r2_o:.4f}, RMSE: {ml_rmse_o:.4f} vs {pinn_rmse_o:.4f} pu)\n"

    # Overall winner
    delta_winner = (
        "ML Baseline"
        if isinstance(pinn_r2_d, (int, float))
        and isinstance(ml_r2_d, (int, float))
        and ml_r2_d > pinn_r2_d
        else "PINN"
    )
    omega_winner = (
        "PINN"
        if isinstance(pinn_r2_o, (int, float))
        and isinstance(ml_r2_o, (int, float))
        and pinn_r2_o > ml_r2_o
        else "ML Baseline"
    )

    if delta_winner == omega_winner:
        content += f"- **Overall Winner**: {delta_winner} performs better overall\n"
    else:
        content += f"- **Overall Winner**: Mixed results - {delta_winner} better for delta, {omega_winner} better for omega\n"

    content += "\n"
    return content


def _detect_ml_baseline_training_issue(
    ml_training_info: Dict, experiment_dir: Optional[Path] = None
) -> str:
    """Detect and format ML baseline training issues."""
    content = ""

    # Get first ML baseline (usually standard_nn)
    ml_model_type = list(ml_training_info.keys())[0] if ml_training_info else None
    if not ml_model_type:
        return ""

    ml_training = ml_training_info.get(ml_model_type, {})
    ml_best_epoch = ml_training.get("best_epoch", None)
    ml_best_val = ml_training.get("best_val_loss", None)

    # Check if best epoch is 0 (training issue)
    issue_detected = False
    if isinstance(ml_best_epoch, int) and ml_best_epoch == 0:
        issue_detected = True

    # Check validation loss progression if checkpoint available
    if experiment_dir and isinstance(ml_best_epoch, int) and ml_best_epoch == 0:
        try:
            ml_checkpoint_files = list(
                experiment_dir.glob(f"ml_baseline/{ml_model_type}/best_model_*.pth")
            )
            if not ml_checkpoint_files:
                ml_checkpoint_files = list(
                    experiment_dir.glob(f"ml_baseline/{ml_model_type}/model.pth")
                )
            if ml_checkpoint_files:
                ml_checkpoint = torch.load(
                    str(ml_checkpoint_files[0]), map_location="cpu", weights_only=False
                )
                ml_history = ml_checkpoint.get("training_history", {})
                ml_val_losses = ml_history.get("val_losses", [])
                if len(ml_val_losses) > 1:
                    initial_val_loss = ml_val_losses[0]
                    final_val_loss = ml_val_losses[-1]
                    if isinstance(initial_val_loss, (int, float)) and isinstance(
                        final_val_loss, (int, float)
                    ):
                        if final_val_loss > initial_val_loss * 1.5:  # Increased by 50% or more
                            issue_detected = True
        except Exception:
            pass

    if not issue_detected:
        return ""

    content += "\n### ⚠️ ML Baseline Training Issue\n\n"
    content += "**Problem Identified:**\n"
    content += "The ML Baseline model shows signs of **training instability** or **overfitting**:\n"

    if isinstance(ml_best_epoch, int) and ml_best_epoch == 0:
        content += (
            f"- Validation loss best at the **initial epoch** (epoch 0), not after training\n"
        )
        if isinstance(ml_best_val, (int, float)):
            content += f"- Best validation loss: {ml_best_val:.2f} at epoch 0\n"

    # Try to get validation loss progression
    if experiment_dir:
        try:
            ml_checkpoint_files = list(
                experiment_dir.glob(f"ml_baseline/{ml_model_type}/best_model_*.pth")
            )
            if not ml_checkpoint_files:
                ml_checkpoint_files = list(
                    experiment_dir.glob(f"ml_baseline/{ml_model_type}/model.pth")
                )
            if ml_checkpoint_files:
                ml_checkpoint = torch.load(
                    str(ml_checkpoint_files[0]), map_location="cpu", weights_only=False
                )
                ml_history = ml_checkpoint.get("training_history", {})
                ml_val_losses = ml_history.get("val_losses", [])
                if len(ml_val_losses) > 1:
                    initial_val_loss = ml_val_losses[0]
                    final_val_loss = ml_val_losses[-1]
                    if isinstance(initial_val_loss, (int, float)) and isinstance(
                        final_val_loss, (int, float)
                    ):
                        if final_val_loss > initial_val_loss:
                            content += f"- Validation loss **increased dramatically** from {initial_val_loss:.2f} (epoch 0) to {final_val_loss:.2f} (epoch {len(ml_val_losses)-1})\n"
                            content += "- Model performance **degraded** as training progressed\n"
        except Exception:
            pass

    content += "\n**Possible Causes:**\n"
    content += "1. **Early stopping patience too high** - Set to 200 epochs (should be 10-20 for ML baselines) ⚠️ **PRIMARY ISSUE**\n"
    content += "2. **ML Baseline Lambda_IC may be too high** - Check if lambda_IC matches PINN's value (10.0) for fair comparison\n"
    content += "3. **Learning rate too high** - Model may be overshooting optimal weights\n"
    content += "4. **Data normalization issues** - Validation set may have different distribution\n"
    content += (
        "5. **Model capacity too large** - Architecture may be overfitting to training data\n"
    )

    content += "\n**Recommendations:**\n"
    content += "1. ✅ **Reduce early stopping patience** from 200 to 10-20 epochs (current: 200 is too high) - **HIGHEST PRIORITY**\n"
    content += "2. ✅ **Ensure lambda_IC matches PINN** (10.0) for fair comparison - May help stability while still enforcing IC constraints\n"
    content += "3. ✅ **Lower learning rate** to 0.0005 or add learning rate scheduling\n"
    content += "4. ✅ **Add dropout** (0.1-0.2) or reduce model capacity if overfitting persists\n"
    content += "5. ✅ **Check data splits** - Ensure train/val distributions are similar\n"
    content += "6. ✅ **Monitor training curves** - Stop training when validation loss plateaus or increases\n"

    content += "\n**Note:** Despite the training issue, the ML Baseline may still achieve good test performance if the model at epoch 0 (best validation) was saved and used for evaluation.\n\n"

    return content


def _format_comparison_results(comparison_data: Dict) -> str:
    """Format comparison results from JSON file."""
    content = ""

    if "delta_comparison" in comparison_data:
        delta_comp = comparison_data["delta_comparison"]
        content += "### Delta (Rotor Angle) Comparison\n\n"
        content += f"- **ML Baseline RMSE**: {delta_comp.get('ml_baseline', {}).get('mean', 'N/A'):.4f} rad\n"
        content += f"- **PINN RMSE**: {delta_comp.get('pinn', {}).get('mean', 'N/A'):.4f} rad\n"

        improvement = delta_comp.get("improvement", {})
        if improvement:
            rel_improve = improvement.get("relative_percent", 0)
            if rel_improve > 0:
                content += f"- **PINN Improvement**: {rel_improve:.1f}% better ✅\n"
            else:
                content += f"- **PINN Performance**: {abs(rel_improve):.1f}% worse ⚠️\n"

        stat_test = delta_comp.get("statistical_test", {})
        if stat_test:
            p_val = stat_test.get("p_value", 1.0)
            sig = "✅" if p_val < 0.05 else "⚠️"
            content += f"- **Statistical Significance**: p = {p_val:.4f} {sig}\n"
        content += "\n"

    if "omega_comparison" in comparison_data:
        omega_comp = comparison_data["omega_comparison"]
        content += "### Omega (Rotor Speed) Comparison\n\n"
        content += f"- **ML Baseline RMSE**: {omega_comp.get('ml_baseline', {}).get('mean', 'N/A'):.4f} pu\n"
        content += f"- **PINN RMSE**: {omega_comp.get('pinn', {}).get('mean', 'N/A'):.4f} pu\n"

        improvement = omega_comp.get("improvement", {})
        if improvement:
            rel_improve = improvement.get("relative_percent", 0)
            if rel_improve > 0:
                content += f"- **PINN Improvement**: {rel_improve:.1f}% better ✅\n"
            else:
                content += f"- **PINN Performance**: {abs(rel_improve):.1f}% worse ⚠️\n"

        stat_test = omega_comp.get("statistical_test", {})
        if stat_test:
            p_val = stat_test.get("p_value", 1.0)
            sig = "✅" if p_val < 0.05 else "⚠️"
            content += f"- **Statistical Significance**: p = {p_val:.4f} {sig}\n"
        content += "\n"

    return content


def _format_comparison_from_summary(comparison_results: Dict) -> str:
    """Format comparison results from summary dictionary."""
    # Similar to _format_comparison_results but handles dict structure from summary
    return _format_comparison_results(comparison_results)


def _generate_findings(pinn_metrics: Dict, ml_metrics: Dict, comparison: Optional[Dict]) -> str:
    """Auto-generate key findings based on metrics."""
    findings = ""

    # Compare PINN vs ML baseline
    if comparison:
        delta_comp = comparison.get("delta_comparison", {})
        omega_comp = comparison.get("omega_comparison", {})

        if delta_comp and omega_comp:
            delta_improve = delta_comp.get("improvement", {}).get("relative_percent", 0)
            omega_improve = omega_comp.get("improvement", {}).get("relative_percent", 0)

            if delta_improve > 0 and omega_improve > 0:
                findings += (
                    "- ✅ **PINN outperforms ML baseline** in both delta and omega predictions\n"
                )
            elif delta_improve < 0 and omega_improve < 0:
                findings += "- ❌ **PINN underperforms ML baseline** in both metrics\n"
            else:
                findings += "- ⚠️ **Mixed results**: PINN shows better performance in one metric but worse in the other\n"

    # Check R² values
    if pinn_metrics:
        r2_d = pinn_metrics.get("r2_delta", 0)
        r2_o = pinn_metrics.get("r2_omega", 0)

        if r2_d > 0.7:
            findings += "- ✅ **Excellent delta prediction** (R² > 0.7)\n"
        elif r2_d > 0.5:
            findings += "- ⚠️ **Moderate delta prediction** (0.5 < R² < 0.7)\n"
        elif r2_d < 0:
            findings += "- ❌ **Poor delta prediction** (negative R² - worse than mean predictor)\n"

        if r2_o > 0.7:
            findings += "- ✅ **Excellent omega prediction** (R² > 0.7)\n"
        elif r2_o > 0.5:
            findings += "- ⚠️ **Moderate omega prediction** (0.5 < R² < 0.7)\n"
        elif r2_o < 0:
            findings += "- ❌ **Poor omega prediction** (negative R² - worse than mean predictor)\n"

    if not findings:
        findings += "- *No automatic findings generated. Please review metrics manually.*\n"

    findings += "\n### Areas for Improvement\n\n"
    findings += "- *Add specific recommendations based on results*\n"

    return findings


def _format_data_config(
    data_config: Dict, validation_config: Dict = None, preprocessing_config: Dict = None
) -> str:
    """Format data generation configuration."""
    content = ""

    # Case file
    if "case_file" in data_config:
        content += f"- **Case File**: `{data_config['case_file']}`\n"

    # Input method
    if "use_pe_as_input" in data_config:
        content += f"- **Use Pe as Input**: {'✅ Enabled' if data_config['use_pe_as_input'] else '❌ Disabled'}\n"

    if "parameter_ranges" in data_config:
        ranges = data_config["parameter_ranges"]
        content += "\n**Parameter Ranges:**\n"
        for param, values in ranges.items():
            if isinstance(values, list) and len(values) >= 2:
                content += f"- {param}: [{values[0]}, {values[1]}]"
                if len(values) > 2:
                    content += f" (samples: {values[2]})"
                content += "\n"
        content += "\n"

    if "sampling_strategy" in data_config:
        content += f"- **Sampling Strategy**: {data_config['sampling_strategy'].upper()}\n"
    if "n_samples" in data_config:
        content += f"- **Number of Samples**: {data_config['n_samples']}\n"
    if "n_samples_per_combination" in data_config:
        content += f"- **Samples per Combination**: {data_config['n_samples_per_combination']}\n"
    if "use_cct_based_sampling" in data_config:
        content += f"- **CCT-based Sampling**: {'✅ Enabled' if data_config['use_cct_based_sampling'] else '❌ Disabled'}\n"

    # Fault configuration
    if "fault" in data_config:
        fault = data_config["fault"]
        content += "\n**Fault Configuration:**\n"
        if "start_time" in fault:
            content += f"- **Start Time**: {fault['start_time']} s\n"
        if "bus" in fault:
            content += f"- **Fault Bus**: {fault['bus']}\n"
        if "reactance" in fault:
            content += f"- **Fault Reactance**: {fault['reactance']} pu\n"
        content += "\n"

    # Simulation settings
    if "simulation_time" in data_config:
        content += f"- **Simulation Time**: {data_config['simulation_time']} s\n"
    if "time_step" in data_config:
        content += f"- **Time Step**: {data_config['time_step']} s\n"

    # CCT offsets
    if "additional_clearing_time_offsets" in data_config:
        offsets = data_config["additional_clearing_time_offsets"]
        if offsets:
            content += f"- **CCT Offsets**: {offsets}\n"

    # Validation configuration
    if validation_config:
        content += "\n**Validation Settings:**\n"
        if "physics_validation" in validation_config:
            content += f"- **Physics Validation**: {'✅ Enabled' if validation_config['physics_validation'] else '❌ Disabled'}\n"
        if "strict_validation" in validation_config:
            content += f"- **Strict Validation**: {'✅ Enabled' if validation_config['strict_validation'] else '❌ Disabled'}\n"
        content += "\n"

    # Preprocessing configuration
    if preprocessing_config:
        content += "\n**Preprocessing Settings:**\n"
        if "train_ratio" in preprocessing_config:
            content += f"- **Train/Val/Test Split**: {preprocessing_config.get('train_ratio', 0.7):.1%} / {preprocessing_config.get('val_ratio', 0.15):.1%} / {preprocessing_config.get('test_ratio', 0.15):.1%}\n"
        if "stratify_by" in preprocessing_config:
            content += f"- **Stratification**: {preprocessing_config['stratify_by']}\n"
        if "filter_angles" in preprocessing_config:
            content += f"- **Angle Filtering**: {'✅ Enabled' if preprocessing_config['filter_angles'] else '❌ Disabled'}\n"
            if (
                preprocessing_config.get("filter_angles")
                and "max_angle_deg" in preprocessing_config
            ):
                content += f"- **Max Angle**: {preprocessing_config['max_angle_deg']}°\n"
            if (
                preprocessing_config.get("filter_angles")
                and "stability_threshold_deg" in preprocessing_config
            ):
                content += f"- **Stability Threshold**: {preprocessing_config['stability_threshold_deg']}°\n"
        content += "\n"

    return content


def _format_model_config(model_config: Dict) -> str:
    """Format model architecture configuration."""
    content = ""

    if "hidden_dims" in model_config:
        content += f"- **Architecture**: {' × '.join(map(str, model_config['hidden_dims']))}\n"
    if "activation" in model_config:
        content += f"- **Activation**: {model_config['activation']}\n"
    if "dropout" in model_config:
        content += f"- **Dropout**: {model_config['dropout']}\n"
    if "use_residual" in model_config:
        content += f"- **Residual Connections**: {'✅ Enabled' if model_config['use_residual'] else '❌ Disabled'}\n"

    return content + "\n"


def _format_reproducibility(reproducibility: Dict) -> str:
    """Format reproducibility information."""
    content = ""

    if "git_commit" in reproducibility:
        content += f"- **Git Commit**: `{reproducibility['git_commit']}`\n"
    if "git_branch" in reproducibility:
        content += f"- **Git Branch**: `{reproducibility['git_branch']}`\n"
    if "random_seed" in reproducibility:
        content += f"- **Random Seed**: {reproducibility['random_seed']}\n"
    if "python_version" in reproducibility:
        content += f"- **Python Version**: {reproducibility['python_version']}\n"

    if "package_versions" in reproducibility:
        content += "\n**Package Versions:**\n"
        pkg_versions = reproducibility["package_versions"]
        for pkg, version in pkg_versions.items():
            content += f"- **{pkg}**: {version}\n"

    return content + "\n"


def _format_data_files(data_info: Dict) -> str:
    """Format data file information."""
    content = "### Data Files\n\n"

    if "source" in data_info:
        source = data_info["source"]
        status = "♻️ Reused" if source == "reused" else "🆕 Generated"
        content += f"- **Data Source**: {status}\n"

    if "train_path" in data_info:
        train_path = Path(data_info["train_path"])
        content += f"- **Training Data**: `{train_path.name}`\n"
    if "val_path" in data_info:
        val_path = Path(data_info["val_path"])
        content += f"- **Validation Data**: `{val_path.name}`\n"
    if "test_path" in data_info:
        test_path = Path(data_info["test_path"])
        content += f"- **Test Data**: `{test_path.name}`\n"

    return content + "\n"


def _format_evaluation_config(evaluation_config: Dict) -> str:
    """Format evaluation configuration."""
    content = ""

    if "run_baselines" in evaluation_config:
        content += f"- **Run ML Baselines**: {'✅ Enabled' if evaluation_config['run_baselines'] else '❌ Disabled'}\n"
    if "metrics" in evaluation_config:
        metrics = evaluation_config["metrics"]
        if isinstance(metrics, list):
            content += f"- **Evaluation Metrics**: {', '.join(metrics).upper()}\n"

    return content + "\n"


def _analyze_overfitting_metrics(train_losses: list, val_losses: list, best_epoch: int) -> Dict:
    """Analyze overfitting from training and validation losses."""
    if not train_losses or not val_losses:
        return {}

    # Calculate metrics
    initial_train = train_losses[0]
    initial_val = val_losses[0]
    final_train = train_losses[-1]
    final_val = val_losses[-1]

    # Loss at best epoch
    if best_epoch < len(train_losses) and best_epoch < len(val_losses):
        train_at_best = train_losses[best_epoch]
        val_at_best = val_losses[best_epoch]
    else:
        train_at_best = final_train
        val_at_best = final_val

    # Overfitting gap
    gap = val_at_best - train_at_best
    gap_percent = (gap / train_at_best * 100) if train_at_best > 0 else 0

    # Loss reduction
    train_reduction = (
        ((initial_train - final_train) / initial_train * 100) if initial_train > 0 else 0
    )
    val_reduction = ((initial_val - val_at_best) / initial_val * 100) if initial_val > 0 else 0

    # Determine severity
    if gap_percent > 50:
        severity = "SEVERE"
        severity_icon = "🔴"
    elif gap_percent > 30:
        severity = "MODERATE"
        severity_icon = "🟡"
    elif gap_percent > 10:
        severity = "MILD"
        severity_icon = "🟢"
    else:
        severity = "NONE"
        severity_icon = "✅"

    # Check if validation loss increases after best epoch
    val_increases = False
    if best_epoch < len(val_losses) - 1:
        val_after_best = val_losses[best_epoch + 1 :]
        if any(v > val_at_best for v in val_after_best):
            val_increases = True
            max_val_after = max(val_after_best)
            increase_percent = (
                ((max_val_after - val_at_best) / val_at_best * 100) if val_at_best > 0 else 0
            )
        else:
            increase_percent = 0
    else:
        increase_percent = 0

    return {
        "initial_train": initial_train,
        "initial_val": initial_val,
        "final_train": final_train,
        "final_val": final_val,
        "train_at_best": train_at_best,
        "val_at_best": val_at_best,
        "best_epoch": best_epoch,
        "gap": gap,
        "gap_percent": gap_percent,
        "severity": severity,
        "severity_icon": severity_icon,
        "train_reduction": train_reduction,
        "val_reduction": val_reduction,
        "val_increases_after_best": val_increases,
        "val_increase_percent": increase_percent,
        "total_epochs": len(train_losses),
    }


def _format_overfitting_analysis(
    pinn_training: Optional[Dict],
    ml_training_info: Dict,
    model_config: Dict,
    training_config: Dict,
    experiment_dir: Optional[Path] = None,
) -> str:
    """Format comprehensive overfitting analysis for all models."""
    content = ""
    has_analysis = False

    # Analyze PINN
    pinn_analysis = None
    if pinn_training and isinstance(pinn_training, dict) and len(pinn_training) > 0:
        train_losses, val_losses = _extract_training_history("PINN", pinn_training, experiment_dir)
        if train_losses and val_losses:
            best_epoch = pinn_training.get("best_epoch", len(val_losses) - 1)
            pinn_analysis = _analyze_overfitting_metrics(train_losses, val_losses, best_epoch)
            has_analysis = True

    # Analyze ML Baseline
    ml_analyses = {}
    if ml_training_info and isinstance(ml_training_info, dict) and len(ml_training_info) > 0:
        for model_type, ml_training in ml_training_info.items():
            if ml_training and isinstance(ml_training, dict):
                train_losses, val_losses = _extract_training_history(
                    f"ML Baseline ({model_type})", ml_training, experiment_dir
                )
                if train_losses and val_losses:
                    best_epoch = ml_training.get("best_epoch", len(val_losses) - 1)
                    ml_analyses[model_type] = _analyze_overfitting_metrics(
                        train_losses, val_losses, best_epoch
                    )
                    has_analysis = True

    if not has_analysis:
        return ""

    # Format PINN analysis
    if pinn_analysis:
        content += "### PINN Model Overfitting Analysis\n\n"
        content += _format_single_model_overfitting(
            "PINN", pinn_analysis, model_config, training_config
        )
        content += "\n"

    # Format ML Baseline analyses
    if ml_analyses:
        for model_type, ml_analysis in ml_analyses.items():
            model_display_name = model_type.replace("_", " ").title()
            content += f"### ML Baseline ({model_display_name}) Overfitting Analysis\n\n"
            content += _format_single_model_overfitting(
                f"ML Baseline ({model_display_name})", ml_analysis, model_config, training_config
            )
            content += "\n"

    # Add overall recommendations
    content += "### Overall Recommendations\n\n"

    # Check if any model has severe overfitting
    severe_models = []
    if pinn_analysis and pinn_analysis.get("severity") == "SEVERE":
        severe_models.append("PINN")
    for model_type, ml_analysis in ml_analyses.items():
        if ml_analysis.get("severity") == "SEVERE":
            severe_models.append(f"ML Baseline ({model_type})")

    if severe_models:
        content += f"**⚠️ Critical**: Severe overfitting detected in {', '.join(severe_models)}\n\n"
        content += "**Immediate Actions Required:**\n"
        content += "1. **Add Dropout**: Increase from 0.0 to 0.2-0.3\n"
        content += "2. **Increase Weight Decay**: Change from 1e-5 to 1e-4 or 1e-3\n"
        content += "3. **Reduce Early Stopping Patience**: PINN: 50-100 epochs, ML Baseline: 10-20 epochs\n"
        content += "4. **Consider Reducing Model Capacity**: Use smaller hidden dimensions if overfitting persists\n\n"
    else:
        # Check for moderate overfitting
        moderate_models = []
        if pinn_analysis and pinn_analysis.get("severity") == "MODERATE":
            moderate_models.append("PINN")
        for model_type, ml_analysis in ml_analyses.items():
            if ml_analysis.get("severity") == "MODERATE":
                moderate_models.append(f"ML Baseline ({model_type})")

        if moderate_models:
            content += (
                f"**⚠️ Warning**: Moderate overfitting detected in {', '.join(moderate_models)}\n\n"
            )
            content += "**Recommended Actions:**\n"
            content += "1. **Add Dropout**: Set to 0.1-0.2\n"
            content += "2. **Increase Weight Decay**: Change to 1e-4\n"
            content += "3. **Monitor Training Curves**: Stop when validation loss plateaus\n\n"
        else:
            content += "✅ **No significant overfitting detected**. Current regularization settings appear adequate.\n\n"

    return content


def _format_single_model_overfitting(
    model_name: str, analysis: Dict, model_config: Dict, training_config: Dict
) -> str:
    """Format overfitting analysis for a single model."""
    content = ""

    severity = analysis.get("severity", "NONE")
    severity_icon = analysis.get("severity_icon", "✅")

    content += f"**Overfitting Status**: {severity_icon} **{severity}**\n\n"

    content += "**Metrics:**\n"
    content += f"- Validation Gap: {analysis['gap']:.4f} ({analysis['gap_percent']:.1f}% higher than training loss)\n"
    content += f"- Best Epoch: {analysis['best_epoch']}\n"
    content += f"- Training Loss at Best: {analysis['train_at_best']:.2f}\n"
    content += f"- Validation Loss at Best: {analysis['val_at_best']:.2f}\n"
    content += f"- Loss Reduction: Training {analysis['train_reduction']:.1f}%, Validation {analysis['val_reduction']:.1f}%\n\n"

    if analysis.get("val_increases_after_best"):
        content += f"⚠️ **Warning**: Validation loss increased by {analysis['val_increase_percent']:.1f}% after best epoch\n\n"

    # Configuration issues
    dropout = float(model_config.get("dropout", 0.0))
    weight_decay = float(training_config.get("weight_decay", 1e-5))
    patience = training_config.get("early_stopping_patience", None)
    if patience is not None:
        patience = int(patience)

    issues = []
    if dropout == 0.0 and severity in ["MODERATE", "SEVERE"]:
        issues.append(f"❌ Dropout is 0.0 (should be 0.1-0.3 for regularization)")
    if weight_decay < 1e-4 and severity in ["MODERATE", "SEVERE"]:
        issues.append(f"❌ Weight decay is too low ({weight_decay}, should be 1e-4 to 1e-3)")
    if patience and patience > 100 and "pinn" in model_name.lower():
        issues.append(f"⚠️ Early stopping patience is high ({patience}, recommended: 50-100)")
    if patience and patience > 20 and "ml" in model_name.lower():
        issues.append(
            f"❌ Early stopping patience is too high ({patience}, should be 10-20 for ML baselines)"
        )

    if issues:
        content += "**Configuration Issues:**\n"
        for issue in issues:
            content += f"- {issue}\n"
        content += "\n"

    return content


def _extract_training_history(
    model_name: str, training_info: Dict, experiment_dir: Optional[Path] = None
) -> Tuple[list, list]:
    """Extract training and validation losses from training info or checkpoint."""
    train_losses = []
    val_losses = []

    # Try to get from training_info first
    if "training_history" in training_info:
        history = training_info.get("training_history", {})
        train_losses = history.get("train_losses", [])
        val_losses = history.get("val_losses", [])
        if train_losses and val_losses:
            return train_losses, val_losses

    # Try to load from checkpoint
    if experiment_dir:
        try:
            if "pinn" in model_name.lower():
                checkpoint_files = list(experiment_dir.glob("pinn/best_model_*.pth"))
            else:
                # ML baseline - extract model type
                model_type = None
                if "(" in model_name and ")" in model_name:
                    model_type = model_name.split("(")[1].split(")")[0].strip()

                if model_type:
                    checkpoint_files = list(
                        experiment_dir.glob(f"ml_baseline/{model_type}/best_model_*.pth")
                    )
                    if not checkpoint_files:
                        checkpoint_files = list(
                            experiment_dir.glob(f"ml_baseline/{model_type}/model.pth")
                        )
                else:
                    checkpoint_files = list(experiment_dir.glob("ml_baseline/*/best_model_*.pth"))
                    if not checkpoint_files:
                        checkpoint_files = list(experiment_dir.glob("ml_baseline/*/model.pth"))

            if checkpoint_files:
                checkpoint = torch.load(
                    str(checkpoint_files[0]), map_location="cpu", weights_only=False
                )
                history = checkpoint.get("training_history", {})
                train_losses = history.get("train_losses", [])
                val_losses = history.get("val_losses", [])
        except Exception:
            pass

    return train_losses, val_losses


def _format_training_config(training_config: Dict, loss_config: Dict) -> str:
    """Format training configuration."""
    content = ""

    if "epochs" in training_config:
        content += f"- **Epochs**: {training_config['epochs']}\n"
    if "learning_rate" in training_config:
        content += f"- **Learning Rate**: {training_config['learning_rate']}\n"
    if "weight_decay" in training_config:
        content += f"- **Weight Decay**: {training_config['weight_decay']}\n"
    if "early_stopping_patience" in training_config:
        content += f"- **Early Stopping Patience**: {training_config['early_stopping_patience']}\n"

    content += "\n**Loss Configuration:**\n"
    if "lambda_data" in loss_config:
        content += f"- λ_data: {loss_config['lambda_data']}\n"
    if "lambda_physics" in loss_config:
        content += f"- λ_physics: {loss_config['lambda_physics']}\n"
    if "lambda_ic" in loss_config:
        content += f"- λ_ic: {loss_config['lambda_ic']}\n"
    if "use_normalized_loss" in loss_config:
        content += f"- **Normalized Loss**: {'✅ Enabled' if loss_config['use_normalized_loss'] else '❌ Disabled'}\n"
    if "scale_to_norm" in loss_config:
        content += f"- **Scale to Norm**: {loss_config['scale_to_norm']}\n"

    return content + "\n"
