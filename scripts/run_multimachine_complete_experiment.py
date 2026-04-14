#!/usr/bin/env python
"""
Multimachine complete experiment: same output layout as SMIB (outputs/complete_experiments/exp_*).

Runs: preprocess (stratify) -> train PINN + ML baseline -> compare (ANDES vs ML vs PINN)
with stable and unstable trajectory plots (model_comparison_delta_only_*.png and overlaid).

Usage:
  python scripts/run_multimachine_complete_experiment.py --data-path "data/multimachine/kundur/exp_*/parameter_sweep_data_*.csv"
  python scripts/run_multimachine_complete_experiment.py --data-path "data/multimachine/kundur/exp_20260220_180602/parameter_sweep_data_20260220_183010.csv" --config configs/publication/kundur_2area.yaml
  python scripts/run_multimachine_complete_experiment.py --data-path "data/multimachine/kundur/exp_*/parameter_sweep_data_*.csv" --skip-ml-baseline-training --ml-baseline-model-path path/to/model.pth

  # Preprocess + train PINN only + evaluate + comparison figures (no ML baseline):
  python scripts/run_multimachine_complete_experiment.py --data-path "data/multimachine/kundur/exp_*/parameter_sweep_data_*.csv" --skip-ml-baseline-training

  # Full pipeline from data generation (generate -> store -> preprocess -> train -> evaluate -> plots):
  python scripts/run_multimachine_complete_experiment.py --generate-data --data-output-dir data/multimachine/kundur --processed-dir data/multimachine/kundur/processed --skip-ml-baseline-training

Shared data (many experiments):
  --processed-dir DIR        Common folder for train/val/test. Preprocess once into DIR; then all experiments use --processed-dir DIR --skip-preprocessing (and omit --data-path). See docs/guides/MANY_EXPERIMENTS_SHARED_DATA.md.

Skip options:
  --skip-preprocessing       Use existing train/val/test (from exp dir or --processed-dir)
  --skip-training            Skip PINN training (use --pinn-model-path or existing exp pinn/model/)
  --pinn-model-path PATH     Path to existing PINN (optional when --skip-training)
  --skip-ml-baseline-training  Skip ML baseline training
  --ml-baseline-model-path PATH  Path to existing ML model (optional when skipping ML training)
  --skip-comparison          Skip comparison step (no trajectory figures)
  --no-evaluate              Skip PINN evaluation (no evaluation_plots_stable/unstable.png)

PINN type (default: multimachine 4-generator):
  By default this script trains the 4-generator Multimachine PINN, evaluates with
  evaluate_multimachine_model.py, and writes trajectory overlays + publication figures to
  comparison/figures. Requires --data-path when using --skip-preprocessing (raw CSV for eval).

  --smib-pinn                Use single-angle "SMIB view" instead: train TrajectoryPredictionPINN,
                             evaluate with evaluate_model.py, compare one equivalent rotor angle.
  --num-machines N           Number of generators (default: from config or 4). Used for multimachine PINN.
"""

import argparse
import io
import subprocess
import sys
import time
from pathlib import Path

# Fix Unicode encoding for Windows (match run_complete_experiment.py)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from data_generation.preprocessing import preprocess_data
from scripts.core.utils import (
    generate_experiment_id,
    generate_timestamped_filename,
    load_config,
    load_json,
    save_config,
    save_json,
)
from scripts.core.training import train_model
from scripts.compare_models import generate_delta_only_plots_pinn_only
from evaluation.baselines.ml_baselines import MLBaselineTrainer


def _create_dirs(base_dir: Path, experiment_id: str) -> dict:
    """Create outputs/complete_experiments/exp_<id> structure (mirror SMIB)."""
    exp_dir = base_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        "root": exp_dir,
        "processed": exp_dir / "processed",
        "pinn_model": exp_dir / "pinn" / "model",
        "ml_baseline": exp_dir / "ml_baseline",
        "ml_baseline_model": exp_dir / "ml_baseline" / "standard_nn" / "model",
        "comparison": exp_dir / "comparison",
        "comparison_figures": exp_dir / "comparison" / "figures",
        "evaluation": exp_dir / "evaluation",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _resolve_data_path(data_path_str: str) -> Path:
    """Resolve optional glob to latest file."""
    path = Path(data_path_str)
    if "*" in path.name:
        matches = list(path.parent.glob(path.name))
        if not matches:
            raise FileNotFoundError(f"No file matching: {data_path_str}")
        path = max(matches, key=lambda p: p.stat().st_mtime)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return path


def _find_latest_parameter_sweep_csv(directory: Path) -> Path:
    """Return path to latest parameter_sweep_data_*.csv under directory (including exp_* subdirs)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Not a directory: {directory}")
    candidates = list(directory.glob("parameter_sweep_data_*.csv"))
    for sub in directory.iterdir():
        if sub.is_dir() and sub.name.startswith("exp_"):
            candidates.extend(sub.glob("parameter_sweep_data_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No parameter_sweep_data_*.csv under {directory}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_data_generation(
    config_path: Path, data_output_dir: Path, skip_analysis: bool = True
) -> Path:
    """Run generate_multimachine_data.py; return path to generated parameter_sweep_data_*.csv."""
    print("\n" + "=" * 70)
    print("STEP 0: DATA GENERATION (Kundur multimachine)")
    print("=" * 70)
    print(f"  Config: {config_path}")
    print(f"  Output: {data_output_dir}")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_multimachine_data.py"),
        "--config",
        str(config_path),
        "--output",
        str(data_output_dir),
        "--skip-analysis" if skip_analysis else "",
    ]
    cmd = [c for c in cmd if c != ""]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Data generation failed with exit code {result.returncode}")
    csv_path = _find_latest_parameter_sweep_csv(data_output_dir)
    print(f"  Generated data: {csv_path}")
    return csv_path


def _preprocess_and_save(
    data_path: Path,
    processed_dir: Path,
    config: dict,
) -> tuple[Path, Path, Path]:
    """Preprocess multimachine CSV (map to SMIB, optional filter, split with stratify_by_stability)."""
    preprocess_cfg = config.get("data", {}).get("preprocessing", {})
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data):,} rows, {data['scenario_id'].nunique()} scenarios")
    result = preprocess_data(
        data=data,
        normalize=False,
        apply_feature_engineering=False,
        split=True,
        train_ratio=preprocess_cfg.get("train_ratio", 0.7),
        val_ratio=preprocess_cfg.get("val_ratio", 0.15),
        test_ratio=preprocess_cfg.get("test_ratio", 0.15),
        stratify_by="scenario_id",
        stratify_by_stability=preprocess_cfg.get("stratify_by_stability", True),
        random_state=config.get("reproducibility", {}).get("random_seed", 42),
        filter_angles=preprocess_cfg.get("filter_angles", False),
        max_angle_deg=preprocess_cfg.get("max_angle_deg", 360.0),
        stability_threshold_deg=preprocess_cfg.get("stability_threshold_deg", 180.0),
    )
    train_data = result["train_data"]
    val_data = result["val_data"]
    test_data = result["test_data"]
    train_path = processed_dir / generate_timestamped_filename("train_data", "csv")
    val_path = processed_dir / generate_timestamped_filename("val_data", "csv")
    test_path = processed_dir / generate_timestamped_filename("test_data", "csv")
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)
    print(f"  Train: {train_path.name}  Val: {val_path.name}  Test: {test_path.name}")
    return train_path, val_path, test_path


def _train_ml_baseline_multimachine(
    train_path: Path,
    val_path: Path,
    dirs: dict,
    config: dict,
) -> Path | None:
    """Train ML baseline (standard_nn) and save to dirs['ml_baseline_model']. Returns path to model.pth."""
    model_dir = dirs["ml_baseline_model"]
    model_dir.mkdir(parents=True, exist_ok=True)

    model_config = config.get("model", {})
    ml_cfg = config.get("ml_baseline", {})
    loss_cfg = config.get("loss", {})
    train_cfg = config.get("training", {})

    input_method = model_config.get("input_method", "pe_direct")
    use_pe = config.get("data", {}).get("generation", {}).get("use_pe_as_input", True)
    if use_pe or input_method == "pe_direct":
        ml_input_method = ml_cfg.get("input_method", "pe_direct")
    else:
        ml_input_method = ml_cfg.get("input_method", input_method)

    dropout = float(ml_cfg.get("dropout", model_config.get("dropout", 0.0)))
    hidden_dims = ml_cfg.get("hidden_dims") or model_config.get("hidden_dims", [256, 256, 128, 128])
    if isinstance(hidden_dims, (list, tuple)):
        hidden_dims = list(hidden_dims)

    trainer = MLBaselineTrainer(
        model_type="standard_nn",
        model_config={
            "hidden_dims": hidden_dims,
            "activation": model_config.get("activation", "tanh"),
            "dropout": dropout,
        },
    )

    scale_to_norm = list(loss_cfg.get("scale_to_norm", [20.0, 40.0]))
    if ml_cfg.get("scale_to_norm") is not None:
        scale_to_norm = list(ml_cfg["scale_to_norm"])
    unstable_weight = float(ml_cfg.get("unstable_weight", 1.0))
    use_fixed_target_scale = bool(ml_cfg.get("use_fixed_target_scale", False))

    train_loader, val_loader, scalers = trainer.prepare_data(
        data_path=train_path,
        input_method=ml_input_method,
        val_data_path=val_path if val_path and val_path.exists() else None,
        scale_to_norm=scale_to_norm,
        unstable_weight=unstable_weight,
        use_fixed_target_scale=use_fixed_target_scale,
    )

    epochs = train_cfg.get("epochs", 300)
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 5e-5))
    patience = train_cfg.get("ml_baseline_early_stopping_patience") or train_cfg.get(
        "early_stopping_patience", 100
    )
    lambda_ic = float(loss_cfg.get("lambda_ic", 10.0))
    pre_fault_weight = float(ml_cfg.get("pre_fault_weight", 1.0))
    lambda_steady_state = float(ml_cfg.get("lambda_steady_state", 0.0))
    use_ic_over_prefault = bool(ml_cfg.get("use_ic_over_prefault", False))
    two_phase_training = bool(ml_cfg.get("two_phase_training", False))
    phase1_epochs = int(ml_cfg.get("phase1_epochs", 0))
    phase2_pre_fault_weight = ml_cfg.get("phase2_pre_fault_weight")
    if phase2_pre_fault_weight is not None:
        phase2_pre_fault_weight = float(phase2_pre_fault_weight)

    start = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        early_stopping_patience=patience,
        lambda_ic=lambda_ic,
        scale_to_norm=scale_to_norm,
        pre_fault_weight=pre_fault_weight,
        lambda_steady_state=lambda_steady_state,
        use_ic_over_prefault=use_ic_over_prefault,
        two_phase_training=two_phase_training,
        phase1_epochs=phase1_epochs,
        phase2_pre_fault_weight=phase2_pre_fault_weight,
    )
    elapsed = time.time() - start

    model_filename = generate_timestamped_filename("best_model", "pth")
    model_path = model_dir / model_filename
    compat_path = model_dir / "model.pth"
    checkpoint = {
        "model_state_dict": trainer.model.state_dict(),
        "model_type": "standard_nn",
        "model_config": trainer.model_config,
        "scalers": scalers,
        "input_method": ml_input_method,
        "training_history": history,
    }
    torch.save(checkpoint, model_path)
    torch.save(checkpoint, compat_path)

    history_path = model_dir / generate_timestamped_filename("training_history", "json")
    save_json(history or {}, history_path)

    print(f"  ML baseline saved: {model_path.name} ({elapsed:.1f}s)")
    return compat_path


def _run_compare_models(
    ml_model_path: Path,
    pinn_model_path: Path,
    test_path: Path,
    comparison_dir: Path,
    n_examples: int = 5,
    pinn_config_path: Path | None = None,
) -> bool:
    """Run compare_models.py subprocess; comparison figures go to comparison/figures."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "compare_models.py"),
        "--ml-baseline-model",
        str(ml_model_path),
        "--pinn-model",
        str(pinn_model_path),
        "--test-data",
        str(test_path),
        "--test-split-path",
        str(test_path),
        "--output-dir",
        str(comparison_dir),
        "--n-examples",
        str(n_examples),
    ]
    if pinn_config_path and pinn_config_path.exists():
        cmd.extend(["--pinn-config", str(pinn_config_path)])
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.stdout and result.stdout.strip():
        print(result.stdout)
    if result.stderr and result.stderr.strip():
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def _run_andes_only_trajectory_plots(
    test_path: Path,
    output_dir: Path,
    n_examples: int = 5,
) -> bool:
    """Generate ANDES-only stable/unstable trajectory figures via analyze_multimachine_data.py."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "analyze_multimachine_data.py"),
        str(test_path),
        "--output-dir",
        str(output_dir),
        "--plot",
        "--sample-trajectories",
        str(n_examples),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.stdout and result.stdout.strip():
        print(result.stdout)
    if result.stderr and result.stderr.strip():
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def _run_evaluate_ml_baseline(
    model_path: Path,
    test_path: Path,
    output_dir: Path,
) -> dict:
    """Run evaluate_ml_baseline.py subprocess; return metrics dict if metrics.json exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_ml_baseline.py"),
        "--model-path",
        str(model_path),
        "--test-data",
        str(test_path),
        "--test-split-path",
        str(test_path),
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.stdout and result.stdout.strip():
        print(result.stdout)
    if result.stderr and result.stderr.strip():
        print(result.stderr, file=sys.stderr)
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        try:
            return load_json(metrics_file)
        except Exception:
            return {}
    return {}


def _build_pinn_predictions(test_data: pd.DataFrame, model_path: Path, device: str = "cpu") -> dict:
    """Run PINN on each test scenario; return {scenario_id: (time, delta, omega)}."""
    from scripts.evaluate_model import (
        load_model_and_scalers,
        evaluate_scenario,
        fit_scalers_from_data,
    )

    model, scalers, input_method = load_model_and_scalers(model_path)
    model = model.to(device)
    if not scalers:
        scalers = fit_scalers_from_data(test_data)
    pinn_predictions = {}
    for scenario_id in test_data["scenario_id"].unique():
        scenario_df = test_data[test_data["scenario_id"] == scenario_id].sort_values("time")
        res = evaluate_scenario(model, scenario_df, scalers, device, input_method=input_method)
        sid = int(scenario_id) if scenario_id is not None else None
        if sid is not None:
            pinn_predictions[sid] = (
                res["time"],
                res["delta_pred"],
                res["omega_pred"],
            )
    return pinn_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run multimachine complete experiment (outputs/complete_experiments/exp_*)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to raw parameter_sweep_data_*.csv (globs supported). Not needed if --skip-preprocessing and --processed-dir set.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Common folder for train/val/test CSVs. If set: preprocess writes here (or read here when --skip-preprocessing). Use for many experiments sharing one dataset.",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Run data generation first (Kundur multimachine); store under --data-output-dir, then run full pipeline.",
    )
    parser.add_argument(
        "--data-output-dir",
        type=str,
        default="data/multimachine/kundur",
        help="Where to store generated data (exp_YYYYMMDD_HHMMSS/parameter_sweep_data_*.csv). Used when --generate-data (default: data/multimachine/kundur).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/publication/kundur_2area.yaml",
        help="Config YAML (default: kundur_2area.yaml)",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="outputs/complete_experiments",
        help="Base directory for exp_<id> (default: outputs/complete_experiments)",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Resume or use this existing experiment directory (e.g. outputs/complete_experiments/exp_20260303_130302). Use with --resume to continue training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume PINN training from latest checkpoint (use with --experiment-dir to resume a specific run)",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Use existing processed CSVs in exp dir (must run once without this first)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip PINN training; use existing PINN (from exp dir or --pinn-model-path) for comparison",
    )
    parser.add_argument(
        "--pinn-model-path",
        type=str,
        default=None,
        help="Path to existing PINN checkpoint (e.g. when --skip-training; may use glob best_model_*.pth)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=5,
        help="Number of stable and unstable scenarios each for comparison figure (default: 5)",
    )
    parser.add_argument(
        "--skip-ml-baseline-training",
        action="store_true",
        help="Skip ML baseline training (use existing model with --ml-baseline-model-path if needed)",
    )
    parser.add_argument(
        "--ml-baseline-model-path",
        type=str,
        default=None,
        help="Path to existing ML baseline model (when --skip-ml-baseline-training)",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip comparison step (do not generate ANDES vs ML vs PINN figures)",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip PINN evaluation step (evaluation_plots_stable/unstable.png in evaluation/)",
    )
    parser.add_argument(
        "--smib-pinn",
        action="store_true",
        help="Use single-angle SMIB-style PINN (TrajectoryPredictionPINN + evaluate_model.py). Default is 4-generator multimachine PINN.",
    )
    parser.add_argument(
        "--num-machines",
        type=int,
        default=None,
        help="Number of generators (default: from config or 4). Used for multimachine PINN (default pipeline).",
    )
    args = parser.parse_args()
    args.evaluate = not args.no_evaluate
    # Default pipeline is multimachine (4-gen); --smib-pinn switches to single-angle
    args.multimachine_pinn = not args.smib_pinn

    # num_machines for multimachine PINN (from config or default)
    num_machines = args.num_machines

    # Where to read/write processed data: common folder or per-experiment
    if args.processed_dir:
        processed_dir = Path(args.processed_dir)
        if not processed_dir.is_absolute():
            processed_dir = PROJECT_ROOT / processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using shared processed data dir: {processed_dir}")
    else:
        processed_dir = None  # will use exp dir below

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path) if config_path.exists() else {}
    config["_config_path"] = str(config_path)
    if num_machines is None:
        num_machines = (
            config.get("model", {}).get("num_machines")
            or config.get("data", {}).get("generation", {}).get("num_machines")
            or 4
        )
    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
        if not experiment_dir.is_absolute():
            experiment_dir = PROJECT_ROOT / experiment_dir
        output_base = experiment_dir.parent
        experiment_id = experiment_dir.name
        dirs = _create_dirs(output_base, experiment_id)
        print("Using existing experiment dir:", dirs["root"])
    else:
        output_base = PROJECT_ROOT / args.output_base
        experiment_id = generate_experiment_id()
        dirs = _create_dirs(output_base, experiment_id)
    save_config(config, dirs["root"] / "config.yaml")
    print(f"Experiment ID: {experiment_id}")
    print(f"Output: {dirs['root']}")

    # Optional Step 0: Generate data and get path to raw CSV
    raw_data_path = None
    if args.generate_data:
        data_output_dir = Path(args.data_output_dir)
        if not data_output_dir.is_absolute():
            data_output_dir = PROJECT_ROOT / data_output_dir
        data_output_dir.mkdir(parents=True, exist_ok=True)
        raw_data_path = _run_data_generation(config_path, data_output_dir, skip_analysis=True)
    elif args.data_path:
        raw_data_path = _resolve_data_path(args.data_path)

    if args.multimachine_pinn and raw_data_path is None:
        print(
            "[ERROR] The default multimachine PINN pipeline requires --data-path (raw parameter_sweep_data CSV) for training and evaluation. Provide --data-path or run without --skip-preprocessing."
        )
        sys.exit(1)

    # Step 1: Preprocess or load existing train/val/test
    if args.skip_preprocessing:
        use_dir = processed_dir if processed_dir is not None else dirs["processed"]
        train_files = sorted(use_dir.glob("train_data_*.csv"))
        test_files = sorted(use_dir.glob("test_data_*.csv"))
        if not train_files or not test_files:
            print(
                "[ERROR] No train_data_*.csv or test_data_*.csv in",
                use_dir,
                ". Run without --skip-preprocessing (and set --data-path or --generate-data) first.",
            )
            sys.exit(1)
        train_path = train_files[-1]
        test_path = test_files[-1]
        val_files = sorted(use_dir.glob("val_data_*.csv"))
        val_path = val_files[-1] if val_files else None
        print(f"Using existing splits: {train_path.name}, {test_path.name}")
    else:
        if raw_data_path is None:
            print(
                "[ERROR] Need --data-path or --generate-data when not using --skip-preprocessing."
            )
            sys.exit(1)
        out_dir = processed_dir if processed_dir is not None else dirs["processed"]
        print("\n" + "=" * 70)
        print("PREPROCESSING (stratify by stability)")
        if processed_dir:
            print("(writing train/val/test to shared folder for reuse by other experiments)")
        print("=" * 70)
        train_path, val_path, test_path = _preprocess_and_save(raw_data_path, out_dir, config)

    # Step 2: Train PINN
    if not args.skip_training:
        if args.multimachine_pinn:
            print("\n" + "=" * 70)
            print("TRAINING: Multimachine PINN (4-generator)")
            print("=" * 70)
            raw_dir = str(raw_data_path.parent)
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "training" / "train_multimachine_pe_input.py"),
                "--data-dir",
                raw_dir,
                "--num-machines",
                str(num_machines),
                "--output-dir",
                str(dirs["pinn_model"]),
                "--config",
                str(config_path),
            ]
            if config.get("training", {}).get("epochs"):
                cmd.extend(["--epochs", str(config["training"]["epochs"])])
            if config.get("training", {}).get("batch_size"):
                cmd.extend(["--batch-size", str(config["training"]["batch_size"])])
            if args.resume:
                cmd.append("--resume")
            print("  (Training can take hours; progress streams below.)")
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                print(
                    "[ERROR] train_multimachine_pe_input.py failed (exit code %d). See stderr above for details."
                    % result.returncode
                )
                sys.exit(result.returncode)
            print("  Model saved to:", dirs["pinn_model"] / "best_model.pth")
        else:
            print("\n" + "=" * 70)
            print("TRAINING: Single-machine PINN view")
            print("=" * 70)
            train_model(
                config=config,
                data_path=train_path,
                output_dir=dirs["pinn_model"],
                seed=config.get("reproducibility", {}).get("random_seed"),
                use_common_repository=False,
            )
    else:
        print("\n[OK] Skipping PINN training (--skip-training)")

    # Step 3: Train ML baseline (same as SMIB: ANDES vs ML vs PINN)
    ml_model_path = None
    if args.skip_ml_baseline_training and args.ml_baseline_model_path:
        p = Path(args.ml_baseline_model_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            ml_model_path = p
            print(f"\n[OK] Using existing ML baseline: {ml_model_path}")
        else:
            print(f"[WARNING] ML baseline path not found: {p}")
    elif not args.skip_ml_baseline_training:
        print("\n" + "=" * 70)
        print("TRAINING: ML baseline (standard_nn)")
        print("=" * 70)
        ml_model_path = _train_ml_baseline_multimachine(train_path, val_path, dirs, config)
    if not args.skip_ml_baseline_training and ml_model_path is None:
        ml_model_path = (
            dirs["ml_baseline_model"] / "model.pth"
            if (dirs["ml_baseline_model"] / "model.pth").exists()
            else None
        )

    # Resolve PINN model path (CLI path overrides exp dir; relative paths vs PROJECT_ROOT)
    pinn_model_path = None
    if args.pinn_model_path:
        p = Path(args.pinn_model_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if "*" in p.name:
            matches = list(p.parent.glob(p.name))
            pinn_model_path = max(matches, key=lambda x: x.stat().st_mtime) if matches else None
        else:
            pinn_model_path = p if p.exists() else None
        if pinn_model_path:
            print(f"\n[OK] Using existing PINN: {pinn_model_path}")
    if pinn_model_path is None:
        pinn_files = list(dirs["pinn_model"].glob("best_model*.pth"))
        if pinn_files:
            pinn_model_path = max(pinn_files, key=lambda p: p.stat().st_mtime)

    # Step 3.5: PINN evaluation (evaluation_plots_stable.png, evaluation_plots_unstable.png)
    if args.evaluate and pinn_model_path is not None:
        if args.multimachine_pinn:
            print("\n" + "=" * 70)
            print("EVALUATION: Multimachine PINN (ANDES vs PINN overlays + publication figures)")
            print("=" * 70)
            dirs["comparison_figures"].mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "evaluate_multimachine_model.py"),
                    "--model-path",
                    str(pinn_model_path),
                    "--data-path",
                    str(raw_data_path),
                    "--num-machines",
                    str(num_machines),
                    "--output-dir",
                    str(dirs["comparison_figures"]),
                    "--test-ratio",
                    "0.15",
                    "--dpi",
                    "300",
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.stdout and result.stdout.strip():
                print(result.stdout)
            if result.stderr and result.stderr.strip():
                print(result.stderr, file=sys.stderr)
            if result.returncode == 0:
                print(
                    "  Trajectory overlays and publication figures saved to:",
                    dirs["comparison_figures"],
                )
            else:
                print(
                    "  [WARNING] evaluate_multimachine_model.py returned non-zero; check output above."
                )
        else:
            print("\n" + "=" * 70)
            print("EVALUATION: PINN on test set (stable + unstable figures)")
            print("=" * 70)
            dirs["evaluation"].mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "evaluate_model.py"),
                    "--model-path",
                    str(pinn_model_path),
                    "--data-path",
                    str(test_path),
                    "--output-dir",
                    str(dirs["evaluation"]),
                    "--n-scenarios",
                    "9999",
                    "--max-scenarios-per-plot",
                    "10",
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.stdout and result.stdout.strip():
                print(result.stdout)
            if result.stderr and result.stderr.strip():
                print(result.stderr, file=sys.stderr)
            if result.returncode == 0:
                print("  Evaluation figures saved to:", dirs["evaluation"])
            else:
                print("  [WARNING] evaluate_model.py returned non-zero; check output above.")

    # Step 3.6: ML baseline evaluation (metrics and figures per model type)
    ml_eval_results: dict = {}
    if ml_model_path is not None and test_path is not None:
        eval_dir = dirs["ml_baseline"] / "standard_nn" / "evaluation"
        print("\n" + "=" * 70)
        print("EVALUATION: ML baseline (standard_nn) on test set")
        print("=" * 70)
        ml_eval_results["standard_nn"] = _run_evaluate_ml_baseline(
            ml_model_path, test_path, eval_dir
        )
        if ml_eval_results["standard_nn"]:
            print("  ML evaluation saved to:", eval_dir)

    # Step 4: Comparison (ANDES vs ML vs PINN, or ANDES vs PINN if no ML, or ANDES-only trajectories)
    if args.skip_comparison:
        print("\n[OK] Skipping comparison (--skip-comparison).")
    elif pinn_model_path is None and ml_model_path is None:
        print("\n" + "=" * 70)
        print("TRAJECTORIES: ANDES only (stable + unstable sample trajectories)")
        print("=" * 70)
        ok = _run_andes_only_trajectory_plots(
            test_path,
            dirs["comparison_figures"],
            n_examples=args.n_examples,
        )
        if ok:
            print("  ANDES trajectory figures saved to:", dirs["comparison_figures"])
        else:
            print("  [WARNING] analyze_multimachine_data.py returned an error; check output above.")
    elif pinn_model_path is None:
        print("[WARNING] No PINN model found; skipping model comparison.")
    elif ml_model_path is not None:
        print("\n" + "=" * 70)
        print("COMPARISON: ANDES vs ML baseline vs PINN (stable + unstable)")
        print("=" * 70)
        ok = _run_compare_models(
            ml_model_path,
            pinn_model_path,
            test_path,
            dirs["comparison"],
            n_examples=args.n_examples,
            pinn_config_path=dirs["root"] / "config.yaml",
        )
        if ok:
            print("  Comparison figures saved to:", dirs["comparison_figures"])
        else:
            print("  [WARNING] compare_models.py returned an error; check output above.")
    else:
        if args.multimachine_pinn:
            # Comparison figures produced by evaluate_multimachine_model in Step 3.5; if we skipped eval, run it now
            if not args.evaluate:
                print("\n" + "=" * 70)
                print(
                    "COMPARISON: Multimachine PINN vs ANDES (trajectory overlays + publication figures)"
                )
                print("=" * 70)
                dirs["comparison_figures"].mkdir(parents=True, exist_ok=True)
                result = subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "evaluate_multimachine_model.py"),
                        "--model-path",
                        str(pinn_model_path),
                        "--data-path",
                        str(raw_data_path),
                        "--num-machines",
                        str(num_machines),
                        "--output-dir",
                        str(dirs["comparison_figures"]),
                        "--test-ratio",
                        "0.15",
                        "--dpi",
                        "300",
                    ],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if result.stdout and result.stdout.strip():
                    print(result.stdout)
                if result.stderr and result.stderr.strip():
                    print(result.stderr, file=sys.stderr)
                if result.returncode == 0:
                    print("  Figures saved to:", dirs["comparison_figures"])
                else:
                    print("  [WARNING] evaluate_multimachine_model.py returned non-zero.")
            else:
                print(
                    "\n[OK] Multimachine PINN comparison figures already saved in Step 3.5 (evaluation)."
                )
        else:
            print("\n" + "=" * 70)
            print("COMPARISON: ANDES vs PINN only (stable + unstable)")
            print("=" * 70)
            test_data = pd.read_csv(test_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pinn_predictions = _build_pinn_predictions(test_data, pinn_model_path, device)
            fig_path = generate_delta_only_plots_pinn_only(
                test_data=test_data,
                pinn_predictions=pinn_predictions,
                output_dir=dirs["comparison_figures"],
                n_examples=args.n_examples,
            )
            if fig_path:
                print(f"  Saved: {fig_path}")

    # Experiment summary (align with SMIB run_complete_experiment)
    summary = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_path": str(config.get("_config_path", "")),
        "data": {
            "train_path": str(train_path) if train_path else None,
            "val_path": str(val_path) if val_path else None,
            "test_path": str(test_path) if test_path else None,
        },
        "pinn": {"model_path": str(pinn_model_path) if pinn_model_path else None},
        "ml_baseline": {
            "model_path": str(ml_model_path) if ml_model_path else None,
            "evaluation": ml_eval_results,
        },
    }
    summary_file = dirs["root"] / "experiment_summary.json"
    save_json(summary, summary_file)

    print("\n[OK] Multimachine complete experiment finished.")
    print(f"  Results: {dirs['root']}")


if __name__ == "__main__":
    main()
