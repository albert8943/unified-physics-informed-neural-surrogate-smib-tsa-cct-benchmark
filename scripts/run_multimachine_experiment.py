#!/usr/bin/env python
"""
Multimachine (Kundur) experiment pipeline — single entry point without changing SMIB scripts.

Runs: (optional) data gen → multimachine analysis → preprocess (with delta_0→delta mapping) → train.

Usage:
    # Use existing multimachine CSV (recommended: no SMIB code touched)
    python scripts/run_multimachine_experiment.py --data-path "data/multimachine/kundur/exp_*/parameter_sweep_data_*.csv"

    # Preprocess + train only (skip analysis)
    python scripts/run_multimachine_experiment.py --data-path "data/multimachine/kundur/exp_20260220_180602/parameter_sweep_data_20260220_183010.csv" --skip-analysis

    # Full pipeline including data generation
    python scripts/run_multimachine_experiment.py --config configs/publication/kundur_2area.yaml

    # Train the multi-machine PINN (Pe input) instead of single-machine view
    python scripts/run_multimachine_experiment.py --data-path "data/multimachine/kundur/exp_*/parameter_sweep_data_*.csv" --use-multimachine-pinn --num-machines 4
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from data_generation.preprocessing import preprocess_data, split_dataset
from scripts.core.training import train_model
from scripts.core.utils import generate_timestamped_filename, save_json, load_config


def _find_latest_multimachine_csv(data_dir: Path) -> Path:
    """Latest parameter_sweep_data_*.csv under data_dir (including exp_* subdirs)."""
    data_dir = Path(data_dir)
    candidates = list(data_dir.glob("parameter_sweep_data_*.csv"))
    for sub in data_dir.iterdir():
        if sub.is_dir() and sub.name.startswith("exp_"):
            candidates.extend(sub.glob("parameter_sweep_data_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No parameter_sweep_data_*.csv under {data_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_multimachine_analysis(data_path: Path, output_dir: Path) -> None:
    """Run analyze_multimachine_data.py on data_path; write report/figures to output_dir."""
    script = PROJECT_ROOT / "scripts" / "analyze_multimachine_data.py"
    if not script.exists():
        print(f"[WARNING] {script.name} not found; skipping multimachine analysis.")
        return
    out = output_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(script), str(data_path), "--plot", "--output-dir", str(out)],
        cwd=str(PROJECT_ROOT),
        check=False,
    )


def _preprocess_and_save(
    data_path: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    filter_angles: bool = False,
    max_angle_deg: float = 360.0,
    stability_threshold_deg: float = 180.0,
    stratify_by_stability: bool = True,
) -> tuple[Path, Path, Path]:
    """Load CSV, run preprocess_data (multimachine→SMIB mapping, optional angle filter, split with both stable/unstable in train/val/test), save."""
    print("\n" + "=" * 70)
    print("PREPROCESSING (multimachine → SMIB-style, angle filter if enabled, split)")
    print("=" * 70)
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data):,} rows, {data['scenario_id'].nunique()} scenarios")
    if filter_angles:
        print(
            f"Rotor angle filtering: max_angle_deg={max_angle_deg}, stability_threshold_deg={stability_threshold_deg}"
        )
    if stratify_by_stability:
        print(
            "Split will stratify by is_stable so train/val/test include both stable and unstable scenarios."
        )
    result = preprocess_data(
        data=data,
        normalize=False,
        apply_feature_engineering=False,
        split=True,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_by="scenario_id",
        stratify_by_stability=stratify_by_stability,
        random_state=random_state,
        filter_angles=filter_angles,
        max_angle_deg=max_angle_deg,
        stability_threshold_deg=stability_threshold_deg,
    )
    train_data = result["train_data"]
    val_data = result["val_data"]
    test_data = result["test_data"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / generate_timestamped_filename("train_data", "csv")
    val_path = output_dir / generate_timestamped_filename("val_data", "csv")
    test_path = output_dir / generate_timestamped_filename("test_data", "csv")
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)
    print(f"  Train: {train_path.name}  Val: {val_path.name}  Test: {test_path.name}")
    return train_path, val_path, test_path


def main():
    parser = argparse.ArgumentParser(
        description="Run multimachine (Kundur) experiment: analysis → preprocess → train",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/publication/kundur_2area.yaml",
        help="YAML config for data gen and training (default: kundur_2area.yaml)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to parameter_sweep_data_*.csv (or directory to search). If not set, uses --data-dir.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/multimachine/kundur",
        help="Directory to search for latest parameter_sweep_data_*.csv when --data-path not set",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output (analysis, processed, model). Default: data/multimachine/kundur/multimachine_exp_<timestamp>",
    )
    parser.add_argument(
        "--skip-data-generation", action="store_true", help="Do not run generate_multimachine_data"
    )
    parser.add_argument(
        "--skip-analysis", action="store_true", help="Skip analyze_multimachine_data"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Use existing train_data_*.csv in output-dir",
    )
    parser.add_argument(
        "--skip-training", action="store_true", help="Only run analysis and/or preprocessing"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after training (same as SMIB workflow)",
    )
    parser.add_argument(
        "--use-multimachine-pinn",
        action="store_true",
        help="Train MultimachinePINN (Pe input) via training/train_multimachine_pe_input.py instead of single-machine PINN",
    )
    parser.add_argument(
        "--num-machines", type=int, default=4, help="For --use-multimachine-pinn (default: 4)"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    args = parser.parse_args()

    # Resolve data path
    if args.data_path:
        data_path = Path(args.data_path)
        if "*" in str(data_path):
            parent = data_path.parent
            matches = list(parent.glob(data_path.name))
            if not matches:
                print(f"No files matching {args.data_path}")
                sys.exit(1)
            data_path = max(matches, key=lambda p: p.stat().st_mtime)
        if not data_path.exists():
            print(f"Data path not found: {data_path}")
            sys.exit(1)
    else:
        data_dir = PROJECT_ROOT / args.data_dir
        try:
            data_path = _find_latest_multimachine_csv(data_dir)
        except FileNotFoundError as e:
            print(e)
            print("Generate data first: python scripts/generate_multimachine_data.py")
            sys.exit(1)

    # Output base
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "data" / "multimachine" / "kundur" / f"multimachine_exp_{ts}"
    )
    output_base = output_base if output_base.is_absolute() else PROJECT_ROOT / output_base
    analysis_dir = output_base / "analysis"
    processed_dir = output_base / "processed"
    model_dir = output_base / "model"

    print("=" * 70)
    print("MULTIMACHINE EXPERIMENT")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Output: {output_base}")

    # Optional: data generation (creates new exp_* and CSV)
    if not args.skip_data_generation:
        print("\n[INFO] Data generation: run manually if needed:")
        print("  python scripts/generate_multimachine_data.py --config", args.config)
        print("  Then re-run this script with --data-path pointing to the new CSV.\n")

    # Step 1: Multimachine analysis (optional)
    if not args.skip_analysis:
        _run_multimachine_analysis(data_path, output_base)

    # Step 2: Preprocess (map multimachine→SMIB, split)
    if args.skip_preprocessing:
        train_files = list(processed_dir.glob("train_data_*.csv"))
        if not train_files:
            print(
                f"[ERROR] No train_data_*.csv in {processed_dir}. Run without --skip-preprocessing first."
            )
            sys.exit(1)
        train_path = sorted(train_files)[-1]
        test_path = (
            sorted(processed_dir.glob("test_data_*.csv"))[-1]
            if list(processed_dir.glob("test_data_*.csv"))
            else None
        )
        val_path = (
            sorted(processed_dir.glob("val_data_*.csv"))[-1]
            if list(processed_dir.glob("val_data_*.csv"))
            else None
        )
    else:
        config_path = PROJECT_ROOT / args.config
        config = load_config(config_path) if config_path.exists() else {}
        preprocess_cfg = config.get("data", {}).get("preprocessing", {})
        train_path, val_path, test_path = _preprocess_and_save(
            data_path,
            processed_dir,
            train_ratio=preprocess_cfg.get("train_ratio", 0.7),
            val_ratio=preprocess_cfg.get("val_ratio", 0.15),
            test_ratio=preprocess_cfg.get("test_ratio", 0.15),
            random_state=config.get("reproducibility", {}).get("random_seed", 42),
            filter_angles=preprocess_cfg.get("filter_angles", False),
            max_angle_deg=preprocess_cfg.get("max_angle_deg", 360.0),
            stability_threshold_deg=preprocess_cfg.get("stability_threshold_deg", 180.0),
            stratify_by_stability=preprocess_cfg.get("stratify_by_stability", True),
        )

    # Step 3: Training
    if args.skip_training:
        print("\n[OK] Skipping training (--skip-training). Done.")
        return

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path) if config_path.exists() else {}
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs

    if args.use_multimachine_pinn:
        print("\n" + "=" * 70)
        print("TRAINING: Multimachine PINN (Pe input)")
        print("=" * 70)
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "training" / "train_multimachine_pe_input.py"),
            "--data-dir",
            str(processed_dir),
            "--output-dir",
            str(model_dir),
            "--num-machines",
            str(args.num_machines),
            "--config",
            args.config,
        ]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    else:
        print("\n" + "=" * 70)
        print("TRAINING: Single-machine PINN (equivalent view of multimachine data)")
        print("=" * 70)
        train_model(
            config=config,
            data_path=train_path,
            output_dir=model_dir,
            seed=config.get("reproducibility", {}).get("random_seed"),
            use_common_repository=False,
        )

    # Step 4: Optional evaluation (same as SMIB: evaluate on test set)
    if args.evaluate and not args.skip_training:
        eval_dir = output_base / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        if args.use_multimachine_pinn:
            print("\n" + "=" * 70)
            print("EVALUATION: Multimachine PINN")
            print("=" * 70)
            best_model = model_dir / "best_model.pth"
            if best_model.exists():
                subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "evaluate_multimachine_model.py"),
                        "--model-path",
                        str(best_model),
                        "--data-path",
                        str(data_path),
                        "--num-machines",
                        str(args.num_machines),
                        "--output-dir",
                        str(eval_dir),
                    ],
                    cwd=str(PROJECT_ROOT),
                    check=False,
                )
            else:
                print(f"  [WARNING] No {best_model}; skip evaluation.")
        else:
            print("\n" + "=" * 70)
            print("EVALUATION: Single-machine PINN (evaluate_model.py)")
            print("=" * 70)
            train_files = list(model_dir.glob("best_model*.pth"))
            if train_files and test_path and test_path.exists():
                subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "evaluate_model.py"),
                        "--model-path",
                        str(sorted(train_files)[-1]),
                        "--data-path",
                        str(test_path),
                        "--output-dir",
                        str(eval_dir),
                    ],
                    cwd=str(PROJECT_ROOT),
                    check=False,
                )
            else:
                print("  [WARNING] No best_model*.pth or test_data; skip evaluation.")

    print("\n[OK] Multimachine experiment finished.")
    print(f"  Analysis: {analysis_dir}")
    print(f"  Processed: {processed_dir}")
    print(f"  Model: {model_dir}")
    if args.evaluate and not args.skip_training:
        print(f"  Evaluation: {output_base / 'evaluation'}")


if __name__ == "__main__":
    main()
