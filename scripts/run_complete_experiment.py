#!/usr/bin/env python
"""
Complete Experiment Workflow Script.

Orchestrates the entire experiment pipeline:
1. Data generation/reuse
2. Data analysis (ANDES)
3. PINN training
4. ML baseline training
5. Model evaluation
6. Model comparison
7. Combined analysis

Usage:
    # Full workflow
    python scripts/run_complete_experiment.py --config configs/experiments/comprehensive.yaml

    # Reuse existing data
    python scripts/run_complete_experiment.py --config config.yaml --skip-data-generation --data-dir path/to/data

    # Reuse existing models
    python scripts/run_complete_experiment.py --config config.yaml \
        --skip-pinn-training --pinn-model-path outputs/experiments/exp_XXX/model/best_model_*.pth \
        --skip-ml-baseline-training --ml-baseline-model-path outputs/ml_baselines/exp_XXX/standard_nn/model.pth

CLI --random-seed overrides reproducibility.random_seed in the loaded YAML
(PINN and ML baseline training).

Reproducibility artifacts (written under each experiment folder, e.g. exp_*/):
    - RERUN.md — copy-paste commands for Unix and Windows CMD
    - run_invocations.jsonl — append-only log of every invocation (argv, cwd, time)
    - experiment_summary.json — includes invocation_latest after a successful run
"""

import argparse
import glob
import json
import shlex
import subprocess
import sys
import io
import time
import shutil
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

import pandas as pd
import torch
from scripts.core.utils import (
    generate_experiment_id,
    generate_timestamped_filename,
    load_config,
    save_config,
    get_git_commit,
    get_git_branch,
    get_package_versions,
    save_json,
    load_json,
)
from scripts.core.data_generation import generate_training_data
from scripts.core.training import train_model
from scripts.core.evaluation import evaluate_model
from scripts.core.experiment_remarks import generate_experiment_remarks
from scripts.core.experiment_tracker import record_complete_experiment
from scripts.core.experiment_cleanup import (
    cleanup_experiment_directory,
    prepare_clean_experiment_directory,
)
from evaluation.baselines.ml_baselines import MLBaselineTrainer
from data_generation.preprocessing import split_dataset


def _resolve_paired_val_test_paths(train_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Pick val/test CSVs that belong to the same preprocessed generation as ``train_path``.

    If the directory contains multiple ``train_data_*`` / ``val_data_*`` / ``test_data_*``
    timestamps, taking ``sorted(...)[-1]`` for each role independently can mix splits
    (e.g. newest train with an older val). PINN and ML already share one ``train_path``;
    this keeps the **triple** (train, val, test) aligned so both models and evaluation
    see one consistent dataset.
    """
    parent = train_path.parent
    name = train_path.name
    if "train_data_" not in name:
        return None, None

    val_paired = parent / name.replace("train_data_", "val_data_", 1)
    test_paired = parent / name.replace("train_data_", "test_data_", 1)
    val_path: Optional[Path] = val_paired if val_paired.exists() else None
    test_path: Optional[Path] = test_paired if test_paired.exists() else None

    if val_path is None:
        val_files = list(parent.glob("val_data_*.csv"))
        if val_files:
            val_path = sorted(val_files)[-1]
            print(
                f"⚠️  Warning: expected paired val file missing ({val_paired.name}). "
                f"Using latest in folder: {val_path.name}"
            )
    if test_path is None:
        test_files = list(parent.glob("test_data_*.csv"))
        if test_files:
            test_path = sorted(test_files)[-1]
            print(
                f"⚠️  Warning: expected paired test file missing ({test_paired.name}). "
                f"Using latest in folder: {test_path.name}"
            )
    return val_path, test_path


def _angle_filter_requires_new_split(
    filter_stats: Optional[dict], filtered_nrows: int, combined_nrows: int
) -> bool:
    """
    True if angle filtering removed points or scenarios (or truncated trajectories).

    When False and the original train/val/test files already have disjoint scenarios,
    we keep those files instead of merging and calling split_dataset again so
    reusing preprocessed CSVs preserves the frozen test set (e.g. publication repro).
    """
    if filtered_nrows != combined_nrows:
        return True
    if not filter_stats:
        return False
    if filter_stats.get("points_removed", 0) > 0:
        return True
    if filter_stats.get("scenarios_removed", 0) > 0:
        return True
    if filter_stats.get("scenarios_truncated", 0) > 0:
        return True
    return False


def _collect_invocation_metadata() -> Dict:
    """Capture argv and cwd so runs can be replayed from the experiment folder."""
    argv = [str(x) for x in sys.argv]
    cwd = str(Path.cwd().resolve())
    pr = str(PROJECT_ROOT.resolve())
    try:
        command_posix = shlex.join(argv)
    except AttributeError:
        command_posix = " ".join(shlex.quote(a) for a in argv)
    command_windows_cmd = subprocess.list2cmdline(argv)
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cwd": cwd,
        "project_root": pr,
        "argv": argv,
        "command_posix": command_posix,
        "command_windows_cmd": command_windows_cmd,
    }


def _save_experiment_invocation_artifacts(experiment_root: Path, meta: Dict) -> None:
    """
    Persist how this run was started: append JSONL history and refresh RERUN.md.

    Files:
        run_invocations.jsonl — one JSON object per line (full history for this folder)
        RERUN.md — latest command, formatted for bash and Windows CMD
    """
    experiment_root.mkdir(parents=True, exist_ok=True)
    log_path = experiment_root / "run_invocations.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    pr = meta["project_root"]
    cwd = meta["cwd"]
    posix = meta["command_posix"]
    win = meta["command_windows_cmd"]
    ts = meta["timestamp"]

    md_lines = [
        "# Rerun / reproduce this experiment",
        "",
        f"Last recorded invocation: **{ts}**",
        "",
        "Paths in the command are relative to the **repository root** below. If your shell was "
        "elsewhere when you ran, `cd` to the repo root first.",
        "",
        f"- **Repository root:** `{pr}`",
        f"- **cwd when this line was logged:** `{cwd}`",
        "",
        "## Unix / Git Bash / WSL",
        "",
        "```bash",
        f"cd {shlex.quote(pr)}",
        posix,
        "```",
        "",
        "## Windows Command Prompt",
        "",
        "```cmd",
        f'cd /d "{pr}"',
        win,
        "```",
        "",
        "## History",
        "",
        "Every time you run `run_complete_experiment.py` targeting this folder (same "
        "`--output-dir` / `--experiment-id`), a line is **appended** to "
        "`run_invocations.jsonl` (`argv`, `cwd`, `timestamp`). Use it to recover exact "
        "partial runs (e.g. PINN-only vs ML-only).",
        "",
        "## Manual notes",
        "",
        "You can add free-form notes in `EXPERIMENT_REMARKS_<id>.md` (generated at end of "
        "run) or edit this file below.",
        "",
        "---",
        "",
    ]
    rerun_path = experiment_root / "RERUN.md"
    with open(rerun_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def _create_unified_directory(base_dir: Path, experiment_id: str) -> Dict[str, Path]:
    """
    Create unified experiment directory structure.

    Parameters:
    -----------
    base_dir : Path
        Base output directory
    experiment_id : str
        Experiment ID (e.g., exp_20251215_120000)

    Returns:
    --------
    dirs : dict
        Dictionary with paths to all subdirectories
    """
    exp_dir = base_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    dirs = {
        "root": exp_dir,
        # NOTE: "data" directory removed - data is saved only to common repository (data/processed/)
        # This prevents large data files from being duplicated in each experiment directory
        "analysis": exp_dir / "analysis",
        "figures": exp_dir
        / "figures",  # Common figures directory (shared between PINN and ML baseline)
        "pinn": exp_dir / "pinn",
        "pinn_model": exp_dir / "pinn" / "model",
        "pinn_results": exp_dir
        / "pinn"
        / "results",  # For evaluation results (like run_experiment.py)
        "pinn_figures": exp_dir
        / "pinn"
        / "results"
        / "figures",  # Where figures are actually saved
        "ml_baseline": exp_dir / "ml_baseline",
        "comparison": exp_dir / "comparison",
        "comparison_figures": exp_dir / "comparison" / "figures",
        # Note: ml_baseline model directories are created per model_type in _train_ml_baseline
    }

    # Create all subdirectories (excluding data directory - saved only to common repository)
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def _expand_wildcard_path(path_str: str) -> Optional[Path]:
    """
    Expand wildcard pattern in file path to find latest matching file.

    Parameters:
    -----------
    path_str : str
        Path string, may contain wildcards (e.g., best_model_*.pth)

    Returns:
    --------
    path : Path or None
        Path to latest matching file, or None if no match
    """
    if "*" not in path_str:
        path = Path(path_str)
        return path if path.exists() else None

    # Find all matching files
    matches = list(glob.glob(str(path_str)))
    if not matches:
        return None

    # Return the latest file (by modification time)
    return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))


def _get_global_processed_dir(experiment_id: Optional[str] = None) -> Path:
    """
    Get the global processed data directory path.

    Parameters:
    -----------
    experiment_id : str, optional
        Experiment ID for creating timestamped folder. If None, uses current timestamp.

    Returns:
    --------
    processed_dir : Path
        Path to global processed data directory (data/processed/exp_YYYYMMDD_HHMMSS/)
    """
    global_processed_base = PROJECT_ROOT / "data" / "processed"

    if experiment_id:
        # Use experiment_id to create folder (e.g., exp_20251217_131235)
        processed_dir = global_processed_base / experiment_id
    else:
        # Fallback: use current timestamp
        from scripts.core.utils import generate_experiment_id

        exp_id = generate_experiment_id()
        processed_dir = global_processed_base / exp_id

    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def _get_global_models_dir(experiment_id: Optional[str] = None) -> Path:
    """
    Get the global models directory path for storing trained models.

    Parameters:
    -----------
    experiment_id : str, optional
        Experiment ID for creating timestamped folder. If None, uses current timestamp.

    Returns:
    --------
    models_dir : Path
        Path to global models directory (outputs/models/exp_YYYYMMDD_HHMMSS/)
    """
    global_models_base = PROJECT_ROOT / "outputs" / "models"

    if experiment_id:
        # Use experiment_id to create folder (e.g., exp_20251217_131235)
        models_dir = global_models_base / experiment_id
    else:
        # Fallback: use current timestamp
        from scripts.core.utils import generate_experiment_id

        exp_id = generate_experiment_id()
        models_dir = global_models_base / exp_id

    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _handle_data_generation(
    config: Dict,
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    experiment_id: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Handle data generation or reuse.

    Returns:
    --------
    train_path, val_path, test_path : tuple of Path or None
    """
    if args.skip_data_generation:
        # Reuse existing data
        if args.data_path:
            data_path = Path(args.data_path)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            print(f"[OK] Using existing data: {data_path}")

            # Check if it's preprocessed
            if "train_data_" in data_path.name:
                train_path = data_path
                val_path, test_path = _resolve_paired_val_test_paths(train_path)

                # FIX: Check if angle filtering should be applied to preprocessed data
                preprocess_config = config.get("data", {}).get("preprocessing", {})
                filter_angles = preprocess_config.get("filter_angles", False)

                if filter_angles and test_path:
                    # Apply angle filtering to preprocessed data
                    print(f"\n📊 Re-applying angle filtering to preprocessed data...")
                    print(f"   Max angle: {preprocess_config.get('max_angle_deg', 360.0)}°")
                    stability_threshold = preprocess_config.get("stability_threshold_deg", 180.0)
                    print(f"   Stability threshold: {stability_threshold}°")

                    from utils.angle_filter import filter_trajectory_by_angle

                    # Load all three splits
                    train_data = pd.read_csv(train_path)
                    val_data = pd.read_csv(val_path) if val_path else None
                    test_data = pd.read_csv(test_path)

                    # Check for overlapping scenarios in original data (warn if found)
                    train_scenarios_orig = set(train_data["scenario_id"].unique())
                    val_scenarios_orig = (
                        set(val_data["scenario_id"].unique()) if val_data is not None else set()
                    )
                    test_scenarios_orig = set(test_data["scenario_id"].unique())

                    overlaps_orig = (
                        (train_scenarios_orig & val_scenarios_orig)
                        | (train_scenarios_orig & test_scenarios_orig)
                        | (val_scenarios_orig & test_scenarios_orig)
                    )
                    if overlaps_orig:
                        print(
                            f"⚠️ Warning: Original preprocessed data has overlapping scenarios: "
                            f"{len(overlaps_orig)} scenarios"
                        )
                        print(f"   This will be fixed by re-splitting after angle filtering.")

                    # Combine all data for filtering (needed for proper scenario-level filtering)
                    all_data = (
                        pd.concat([train_data, val_data, test_data], ignore_index=True)
                        if val_data is not None
                        else pd.concat([train_data, test_data], ignore_index=True)
                    )

                    # Note: If original files had overlapping scenarios, we'll have duplicate rows
                    # but split_dataset handles this correctly by splitting on unique scenario_ids
                    print(
                        f"[INFO] Combined data: {len(all_data):,} rows, "
                        f"{all_data['scenario_id'].nunique()} unique scenarios"
                    )

                    # Apply angle filtering
                    max_angle_deg = preprocess_config.get("max_angle_deg", 360.0)
                    stability_threshold_deg = preprocess_config.get(
                        "stability_threshold_deg", 180.0
                    )

                    filtered_data, filter_stats = filter_trajectory_by_angle(
                        data=all_data,
                        max_angle_deg=max_angle_deg,
                        stability_threshold_deg=stability_threshold_deg,
                    )

                    if filter_stats:
                        print(f"\n[INFO] Angle filtering statistics:")
                        print(
                            f"Points: {filter_stats.get('original_points', 0):,} →"
                            f"{filter_stats.get('filtered_points', 0):,}"
                        )
                        print(
                            f"Scenarios: {filter_stats.get('original_scenarios', 0)} →"
                            f"{filter_stats.get('filtered_scenarios', 0)}"
                        )
                        print(
                            f"Max angle: {filter_stats.get('max_angle_before', 0):.1f}° →"
                            f"{filter_stats.get('max_angle_after', 0):.1f}°"
                        )

                    must_resplit = bool(overlaps_orig) or _angle_filter_requires_new_split(
                        filter_stats, len(filtered_data), len(all_data)
                    )

                    if not must_resplit:
                        print(
                            "\n[OK] Angle filtering did not remove rows or scenarios; "
                            "train/val/test splits are already disjoint — "
                            "keeping original preprocessed files (frozen split preserved)."
                        )
                    else:
                        train_ratio = preprocess_config.get("train_ratio", 0.7)
                        val_ratio = preprocess_config.get("val_ratio", 0.15)
                        test_ratio = preprocess_config.get("test_ratio", 0.15)
                        random_state = config.get("reproducibility", {}).get("random_seed", 42)
                        stratify_by = preprocess_config.get("stratify_by", "scenario_id")

                        train_data_filtered, val_data_filtered, test_data_filtered = split_dataset(
                            filtered_data,
                            train_ratio=train_ratio,
                            val_ratio=val_ratio,
                            test_ratio=test_ratio,
                            stratify_by=stratify_by,
                            random_state=random_state,
                        )

                        train_scenarios_new = set(train_data_filtered["scenario_id"].unique())
                        val_scenarios_new = set(val_data_filtered["scenario_id"].unique())
                        test_scenarios_new = set(test_data_filtered["scenario_id"].unique())

                        overlaps_new = (
                            (train_scenarios_new & val_scenarios_new)
                            | (train_scenarios_new & test_scenarios_new)
                            | (val_scenarios_new & test_scenarios_new)
                        )
                        if overlaps_new:
                            print(
                                f"❌ ERROR: Re-split still has overlapping scenarios: "
                                f"{len(overlaps_new)} scenarios"
                            )
                            print(
                                f"   This indicates a bug in split_dataset. Please report this issue."
                            )
                        else:
                            print(f"\n✓ Split verification: No overlapping scenarios (correct)")

                        processed_dir = _get_global_processed_dir(experiment_id)

                        train_filename = generate_timestamped_filename("train_data", "csv")
                        val_filename = generate_timestamped_filename("val_data", "csv")
                        test_filename = generate_timestamped_filename("test_data", "csv")

                        train_path = processed_dir / train_filename
                        val_path = processed_dir / val_filename
                        test_path = processed_dir / test_filename

                        train_data_filtered.to_csv(train_path, index=False)
                        val_data_filtered.to_csv(val_path, index=False)
                        test_data_filtered.to_csv(test_path, index=False)

                        print(f"\n[OK] Re-preprocessed data with angle filtering:")
                        print(
                            f"Train: {train_path.name} ({len(train_data_filtered):,} rows,"
                            f"{len(train_data_filtered['scenario_id'].unique())} scenarios)"
                        )
                        print(
                            f"Val: {val_path.name} ({len(val_data_filtered):,} rows,"
                            f"{len(val_data_filtered['scenario_id'].unique())} scenarios)"
                        )
                        print(
                            f"Test: {test_path.name} ({len(test_data_filtered):,} rows,"
                            f"{len(test_data_filtered['scenario_id'].unique())} scenarios)"
                        )

                return train_path, val_path, test_path
            else:
                # Full data, check if preprocessing with angle filtering is needed
                preprocess_config = config.get("data", {}).get("preprocessing", {})
                filter_angles = preprocess_config.get("filter_angles", False)

                if filter_angles:
                    # Apply preprocessing with angle filtering
                    print(f"\n📊 Preprocessing raw data with angle filtering...")
                    from data_generation.preprocessing import preprocess_data

                    processed_dir = _get_global_processed_dir(experiment_id)

                    train_ratio = preprocess_config.get("train_ratio", 0.7)
                    val_ratio = preprocess_config.get("val_ratio", 0.15)
                    test_ratio = preprocess_config.get("test_ratio", 0.15)
                    random_state = config.get("reproducibility", {}).get("random_seed", 42)
                    stratify_by = preprocess_config.get("stratify_by", "scenario_id")
                    max_angle_deg = preprocess_config.get("max_angle_deg", 360.0)
                    stability_threshold_deg = preprocess_config.get(
                        "stability_threshold_deg", 180.0
                    )

                    data = pd.read_csv(data_path)
                    preprocess_result = preprocess_data(
                        data=data,
                        normalize=False,
                        apply_feature_engineering=False,
                        split=True,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                        stratify_by=stratify_by,
                        random_state=random_state,
                        filter_angles=True,
                        max_angle_deg=max_angle_deg,
                        stability_threshold_deg=stability_threshold_deg,
                    )
                    train_data = preprocess_result["train_data"]
                    val_data = preprocess_result["val_data"]
                    test_data = preprocess_result["test_data"]
                    filter_stats = preprocess_result.get("filter_stats")

                    if filter_stats:
                        print(f"\n[INFO] Angle filtering statistics:")
                        print(
                            f"Points: {filter_stats.get('original_points', 0):,} →"
                            f"{filter_stats.get('filtered_points', 0):,}"
                        )
                        print(
                            f"Scenarios: {filter_stats.get('original_scenarios', 0)} →"
                            f"{filter_stats.get('filtered_scenarios', 0)}"
                        )
                        print(
                            f"Max angle: {filter_stats.get('max_angle_before', 0):.1f}° →"
                            f"{filter_stats.get('max_angle_after', 0):.1f}°"
                        )

                    # Verify the split is correct (no overlaps)
                    train_scenarios = set(train_data["scenario_id"].unique())
                    val_scenarios = set(val_data["scenario_id"].unique())
                    test_scenarios = set(test_data["scenario_id"].unique())

                    overlaps = (
                        (train_scenarios & val_scenarios)
                        | (train_scenarios & test_scenarios)
                        | (val_scenarios & test_scenarios)
                    )
                    if overlaps:
                        print(
                            f"❌ ERROR: Preprocessing created overlapping scenarios: "
                            f"{len(overlaps)} scenarios"
                        )
                        print(
                            f"This indicates a bug in preprocess_data or split_dataset. Please"
                            f"report this issue."
                        )
                        print(
                            f"   Overlapping scenario IDs: {sorted(list(overlaps))[:10]}..."
                            if len(overlaps) > 10
                            else f"   Overlapping scenario IDs: {sorted(list(overlaps))}"
                        )
                    else:
                        print(f"\n✓ Split verification: No overlapping scenarios (correct)")

                    # Save preprocessed splits
                    train_filename = generate_timestamped_filename("train_data", "csv")
                    val_filename = generate_timestamped_filename("val_data", "csv")
                    test_filename = generate_timestamped_filename("test_data", "csv")

                    train_path = processed_dir / train_filename
                    val_path = processed_dir / val_filename
                    test_path = processed_dir / test_filename

                    train_data.to_csv(train_path, index=False)
                    val_data.to_csv(val_path, index=False)
                    test_data.to_csv(test_path, index=False)

                    print(f"\n[OK] Data preprocessed and split (saved to global location)")
                    print(f"     Location: {processed_dir}")
                    print(
                        f"Train: {train_path.name} ({len(train_data):,} rows,"
                        f"{len(train_data['scenario_id'].unique())} scenarios)"
                    )
                    print(
                        f"Val: {val_path.name} ({len(val_data):,} rows,"
                        f"{len(val_data['scenario_id'].unique())} scenarios)"
                    )
                    print(
                        f"Test: {test_path.name} ({len(test_data):,} rows,"
                        f"{len(test_data['scenario_id'].unique())} scenarios)"
                    )

                    return train_path, val_path, test_path
                else:
                    # No filtering needed, return raw data path
                    return data_path, None, None

        if args.data_dir:
            data_dir = Path(args.data_dir)

            # Look for preprocessed files first (check both data_dir and preprocessed subdirectory)
            # This matches the behavior of run_experiment.py
            train_files = list(data_dir.glob("train_data_*.csv"))
            processed_dir = data_dir / "processed"
            if processed_dir.exists():
                train_files.extend(list(processed_dir.glob("train_data_*.csv")))

            if train_files:
                train_path = sorted(train_files)[-1]
                val_path, test_path = _resolve_paired_val_test_paths(train_path)

                if test_path:
                    print(f"[OK] Found preprocessed data:")
                    print(f"     Train: {train_path.name}")
                    if val_path:
                        print(f"     Val: {val_path.name}")
                    print(f"     Test: {test_path.name}")

                    # FIX: Check if angle filtering should be applied to preprocessed data
                    preprocess_config = config.get("data", {}).get("preprocessing", {})
                    filter_angles = preprocess_config.get("filter_angles", False)

                    if filter_angles:
                        # Apply angle filtering to preprocessed data
                        print(f"\n📊 Re-applying angle filtering to preprocessed data...")
                        print(f"   Max angle: {preprocess_config.get('max_angle_deg', 360.0)}°")
                        print(
                            f"Stability threshold:"
                            f"{preprocess_config.get('stability_threshold_deg', 180.0)}°"
                        )

                        from data_generation.preprocessing import preprocess_data
                        from utils.angle_filter import filter_trajectory_by_angle

                        # Load all three splits
                        train_data = pd.read_csv(train_path)
                        val_data = pd.read_csv(val_path) if val_path else None
                        test_data = pd.read_csv(test_path)

                        # Check for overlapping scenarios in original data (warn if found)
                        train_scenarios_orig = set(train_data["scenario_id"].unique())
                        val_scenarios_orig = (
                            set(val_data["scenario_id"].unique()) if val_data is not None else set()
                        )
                        test_scenarios_orig = set(test_data["scenario_id"].unique())

                        overlaps_orig = (
                            (train_scenarios_orig & val_scenarios_orig)
                            | (train_scenarios_orig & test_scenarios_orig)
                            | (val_scenarios_orig & test_scenarios_orig)
                        )
                        if overlaps_orig:
                            print(
                                f"⚠️ Warning: Original preprocessed data has overlapping "
                                f"scenarios: {len(overlaps_orig)} scenarios"
                            )
                            print(f"   This will be fixed by re-splitting after angle filtering.")

                        # Combine all data for filtering (needed for proper scenario-level filtering)
                        all_data = (
                            pd.concat([train_data, val_data, test_data], ignore_index=True)
                            if val_data is not None
                            else pd.concat([train_data, test_data], ignore_index=True)
                        )

                        # Note: If original files had overlapping scenarios, we'll have duplicate rows
                        # but split_dataset handles this correctly by splitting on unique scenario_ids
                        print(
                            f"[INFO] Combined data: {len(all_data):,} rows, "
                            f"{all_data['scenario_id'].nunique()} unique scenarios"
                        )

                        # Apply angle filtering
                        max_angle_deg = preprocess_config.get("max_angle_deg", 360.0)
                        stability_threshold_deg = preprocess_config.get(
                            "stability_threshold_deg", 180.0
                        )

                        filtered_data, filter_stats = filter_trajectory_by_angle(
                            data=all_data,
                            max_angle_deg=max_angle_deg,
                            stability_threshold_deg=stability_threshold_deg,
                        )

                        if filter_stats:
                            print(f"\n[INFO] Angle filtering statistics:")
                            print(
                                f"Points: {filter_stats.get('original_points', 0):,} →"
                                f"{filter_stats.get('filtered_points', 0):,}"
                            )
                            print(
                                f"Scenarios: {filter_stats.get('original_scenarios', 0)} →"
                                f"{filter_stats.get('filtered_scenarios', 0)}"
                            )
                            print(
                                f"Max angle: {filter_stats.get('max_angle_before', 0):.1f}° →"
                                f"{filter_stats.get('max_angle_after', 0):.1f}°"
                            )

                        must_resplit = bool(overlaps_orig) or _angle_filter_requires_new_split(
                            filter_stats, len(filtered_data), len(all_data)
                        )

                        if not must_resplit:
                            print(
                                "\n[OK] Angle filtering did not remove rows or scenarios; "
                                "train/val/test splits are already disjoint — "
                                "keeping original preprocessed files (frozen split preserved)."
                            )
                        else:
                            train_ratio = preprocess_config.get("train_ratio", 0.7)
                            val_ratio = preprocess_config.get("val_ratio", 0.15)
                            test_ratio = preprocess_config.get("test_ratio", 0.15)
                            random_state = config.get("reproducibility", {}).get("random_seed", 42)
                            stratify_by = preprocess_config.get("stratify_by", "scenario_id")

                            (
                                train_data_filtered,
                                val_data_filtered,
                                test_data_filtered,
                            ) = split_dataset(
                                filtered_data,
                                train_ratio=train_ratio,
                                val_ratio=val_ratio,
                                test_ratio=test_ratio,
                                stratify_by=stratify_by,
                                random_state=random_state,
                            )

                            train_scenarios_new = set(train_data_filtered["scenario_id"].unique())
                            val_scenarios_new = set(val_data_filtered["scenario_id"].unique())
                            test_scenarios_new = set(test_data_filtered["scenario_id"].unique())

                            overlaps_new = (
                                (train_scenarios_new & val_scenarios_new)
                                | (train_scenarios_new & test_scenarios_new)
                                | (val_scenarios_new & test_scenarios_new)
                            )
                            if overlaps_new:
                                print(
                                    f"❌ ERROR: Re-split still has overlapping scenarios: "
                                    f"{len(overlaps_new)} scenarios"
                                )
                                print(
                                    f"   This indicates a bug in split_dataset. Please report this issue."
                                )
                            else:
                                print(f"\n✓ Split verification: No overlapping scenarios (correct)")

                            processed_dir = _get_global_processed_dir(experiment_id)

                            train_filename = generate_timestamped_filename("train_data", "csv")
                            val_filename = generate_timestamped_filename("val_data", "csv")
                            test_filename = generate_timestamped_filename("test_data", "csv")

                            train_path = processed_dir / train_filename
                            val_path = processed_dir / val_filename
                            test_path = processed_dir / test_filename

                            train_data_filtered.to_csv(train_path, index=False)
                            val_data_filtered.to_csv(val_path, index=False)
                            test_data_filtered.to_csv(test_path, index=False)

                            print(f"\n[OK] Re-preprocessed data with angle filtering:")
                            print(
                                f"Train: {train_path.name} ({len(train_data_filtered):,} rows,"
                                f"{len(train_data_filtered['scenario_id'].unique())} scenarios)"
                            )
                            print(
                                f"Val: {val_path.name} ({len(val_data_filtered):,} rows,"
                                f"{len(val_data_filtered['scenario_id'].unique())} scenarios)"
                            )
                            print(
                                f"Test: {test_path.name} ({len(test_data_filtered):,} rows,"
                                f"{len(test_data_filtered['scenario_id'].unique())} scenarios)"
                            )

                return train_path, val_path, test_path

            # Look for trajectory data
            traj_files = list(data_dir.glob("trajectory_data_*.csv")) + list(
                data_dir.glob("parameter_sweep_data_*.csv")
            )
            if traj_files:
                data_path = sorted(traj_files)[-1]
                print(f"[OK] Using existing raw data: {data_path}")

                # FIX: Check if preprocessing with angle filtering is needed
                preprocess_config = config.get("data", {}).get("preprocessing", {})
                filter_angles = preprocess_config.get("filter_angles", False)

                if filter_angles:
                    # Apply preprocessing with angle filtering
                    print(f"\n📊 Preprocessing raw data with angle filtering...")
                    from data_generation.preprocessing import preprocess_data

                    processed_dir = _get_global_processed_dir(experiment_id)

                    train_ratio = preprocess_config.get("train_ratio", 0.7)
                    val_ratio = preprocess_config.get("val_ratio", 0.15)
                    test_ratio = preprocess_config.get("test_ratio", 0.15)
                    random_state = config.get("reproducibility", {}).get("random_seed", 42)
                    stratify_by = preprocess_config.get("stratify_by", "scenario_id")
                    max_angle_deg = preprocess_config.get("max_angle_deg", 360.0)
                    stability_threshold_deg = preprocess_config.get(
                        "stability_threshold_deg", 180.0
                    )

                    data = pd.read_csv(data_path)
                    preprocess_result = preprocess_data(
                        data=data,
                        normalize=False,
                        apply_feature_engineering=False,
                        split=True,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                        stratify_by=stratify_by,
                        random_state=random_state,
                        filter_angles=True,
                        max_angle_deg=max_angle_deg,
                        stability_threshold_deg=stability_threshold_deg,
                    )
                    train_data = preprocess_result["train_data"]
                    val_data = preprocess_result["val_data"]
                    test_data = preprocess_result["test_data"]
                    filter_stats = preprocess_result.get("filter_stats")

                    if filter_stats:
                        print(f"\n[INFO] Angle filtering statistics:")
                        print(
                            f"Points: {filter_stats.get('original_points', 0):,} →"
                            f"{filter_stats.get('filtered_points', 0):,}"
                        )
                        print(
                            f"Scenarios: {filter_stats.get('original_scenarios', 0)} →"
                            f"{filter_stats.get('filtered_scenarios', 0)}"
                        )
                        print(
                            f"Max angle: {filter_stats.get('max_angle_before', 0):.1f}° →"
                            f"{filter_stats.get('max_angle_after', 0):.1f}°"
                        )

                    # Verify the split is correct (no overlaps)
                    train_scenarios = set(train_data["scenario_id"].unique())
                    val_scenarios = set(val_data["scenario_id"].unique())
                    test_scenarios = set(test_data["scenario_id"].unique())

                    overlaps = (
                        (train_scenarios & val_scenarios)
                        | (train_scenarios & test_scenarios)
                        | (val_scenarios & test_scenarios)
                    )
                    if overlaps:
                        print(
                            f"❌ ERROR: Preprocessing created overlapping scenarios: "
                            f"{len(overlaps)} scenarios"
                        )
                        print(
                            f"This indicates a bug in preprocess_data or split_dataset. Please"
                            f"report this issue."
                        )
                        print(
                            f"   Overlapping scenario IDs: {sorted(list(overlaps))[:10]}..."
                            if len(overlaps) > 10
                            else f"   Overlapping scenario IDs: {sorted(list(overlaps))}"
                        )
                    else:
                        print(f"\n✓ Split verification: No overlapping scenarios (correct)")

                    # Save preprocessed splits
                    train_filename = generate_timestamped_filename("train_data", "csv")
                    val_filename = generate_timestamped_filename("val_data", "csv")
                    test_filename = generate_timestamped_filename("test_data", "csv")

                    train_path = processed_dir / train_filename
                    val_path = processed_dir / val_filename
                    test_path = processed_dir / test_filename

                    train_data.to_csv(train_path, index=False)
                    val_data.to_csv(val_path, index=False)
                    test_data.to_csv(test_path, index=False)

                    print(f"\n[OK] Data preprocessed and split (saved to global location)")
                    print(f"     Location: {processed_dir}")
                    print(
                        f"Train: {train_path.name} ({len(train_data):,} rows,"
                        f"{len(train_data['scenario_id'].unique())} scenarios)"
                    )
                    print(
                        f"Val: {val_path.name} ({len(val_data):,} rows,"
                        f"{len(val_data['scenario_id'].unique())} scenarios)"
                    )
                    print(
                        f"Test: {test_path.name} ({len(test_data):,} rows,"
                        f"{len(test_data['scenario_id'].unique())} scenarios)"
                    )

                    return train_path, val_path, test_path
                else:
                    # No filtering needed, return raw data path
                    return data_path, None, None

            raise FileNotFoundError(f"No data files found in: {data_dir}")

        raise ValueError("--skip-data-generation requires --data-path or --data-dir")

    # Generate new data
    print("\n" + "=" * 70)
    print("STEP 1: DATA GENERATION")
    print("=" * 70)

    use_common_repository = config.get("use_common_repository", True)
    # NOTE: data_dir is not used when use_common_repository=True (default)
    # Data is saved only to common repository (data/common/ or data/processed/)
    # This prevents large data files from being duplicated in experiment directories
    data_dir = dirs.get("data", Path("data/temp"))  # Fallback for legacy compatibility

    data_path, validation_results = generate_training_data(
        config=config,
        output_dir=data_dir,
        validate_physics=True,
        skip_if_exists=False,
        use_common_repository=use_common_repository,  # Always True by default - saves to common repository
        force_regenerate=args.force_regenerate,
    )

    print(f"[OK] Data generated: {data_path.name}")

    # Check if preprocessing is needed
    # If data_path contains "trajectory_data" or "parameter_sweep_data", it needs preprocessing
    # Skip if --skip-preprocessing flag is set
    if not args.skip_preprocessing and (
        "trajectory_data" in str(data_path.name) or "parameter_sweep_data" in str(data_path.name)
    ):
        # Data needs to be preprocessed/split
        # Training functions will handle this, but we can preprocess here for consistency
        from data_generation.preprocessing import preprocess_data

        processed_dir = _get_global_processed_dir(experiment_id)

        # Get preprocessing config
        preprocess_config = config.get("data", {}).get("preprocessing", {})
        train_ratio = preprocess_config.get("train_ratio", 0.7)
        val_ratio = preprocess_config.get("val_ratio", 0.15)
        test_ratio = preprocess_config.get("test_ratio", 0.15)
        random_state = config.get("reproducibility", {}).get("random_seed", 42)

        # FIX: Use stratify_by from config (not hardcoded)
        stratify_by = preprocess_config.get("stratify_by", "scenario_id")

        # FIX: Get angle filtering settings from config
        filter_angles = preprocess_config.get("filter_angles", False)
        max_angle_deg = preprocess_config.get("max_angle_deg", 360.0)
        stability_threshold_deg = preprocess_config.get("stability_threshold_deg", 180.0)

        # Load data
        data = pd.read_csv(data_path)

        # FIX: Apply angle filtering if enabled, then split dataset
        if filter_angles:
            print(
                f"📊 Applying angle filtering (max: {max_angle_deg}°, threshold: "
                f"{stability_threshold_deg}°)"
            )
            print(f"   Stratification: {stratify_by}")
            preprocess_result = preprocess_data(
                data=data,
                normalize=False,  # Normalization happens in training
                apply_feature_engineering=False,  # Feature engineering happens in training
                split=True,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                stratify_by=stratify_by,  # FIX: Use from config
                random_state=random_state,
                filter_angles=True,  # FIX: Apply angle filtering
                max_angle_deg=max_angle_deg,
                stability_threshold_deg=stability_threshold_deg,
            )
            train_data = preprocess_result["train_data"]
            val_data = preprocess_result["val_data"]
            test_data = preprocess_result["test_data"]
            filter_stats = preprocess_result.get("filter_stats")
            if filter_stats:
                print(f"\n[INFO] Angle filtering statistics:")
                print(
                    f"Points: {filter_stats.get('original_points', 0):,} →"
                    f"{filter_stats.get('filtered_points', 0):,}"
                )
                print(
                    f"Scenarios: {filter_stats.get('original_scenarios', 0)} →"
                    f"{filter_stats.get('filtered_scenarios', 0)}"
                )
                print(
                    f"Max angle: {filter_stats.get('max_angle_before', 0):.1f}° →"
                    f"{filter_stats.get('max_angle_after', 0):.1f}°"
                )
        else:
            # No angle filtering, just split
            print(f"\nSplitting dataset (stratify_by: {stratify_by})...")
            train_data, val_data, test_data = split_dataset(
                data,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                stratify_by=stratify_by,  # FIX: Use from config
                random_state=random_state,
            )

        # Verify the split is correct (no overlaps)
        train_scenarios = set(train_data["scenario_id"].unique())
        val_scenarios = set(val_data["scenario_id"].unique())
        test_scenarios = set(test_data["scenario_id"].unique())

        # Debug: Print scenario counts
        print(f"\n[DEBUG] Split verification:")
        print(f"  Train scenarios: {len(train_scenarios)}")
        print(f"  Val scenarios: {len(val_scenarios)}")
        print(f"  Test scenarios: {len(test_scenarios)}")
        print(f"  Total unique across all: {len(train_scenarios | val_scenarios | test_scenarios)}")

        overlaps = (
            (train_scenarios & val_scenarios)
            | (train_scenarios & test_scenarios)
            | (val_scenarios & test_scenarios)
        )
        if overlaps:
            print(f"\n❌ ERROR: Split created overlapping scenarios: {len(overlaps)} scenarios")
            print(f"   This indicates a bug in split_dataset. Please report this issue.")
            print(
                f"   Overlapping scenario IDs: {sorted(list(overlaps))[:10]}..."
                if len(overlaps) > 10
                else f"   Overlapping scenario IDs: {sorted(list(overlaps))}"
            )
            print(f"\n[DEBUG] Detailed overlap analysis:")
            train_val_overlap = train_scenarios & val_scenarios
            train_test_overlap = train_scenarios & test_scenarios
            val_test_overlap = val_scenarios & test_scenarios
            if train_val_overlap:
                print(f"  Train/Val overlap: {len(train_val_overlap)} scenarios")
            if train_test_overlap:
                print(f"  Train/Test overlap: {len(train_test_overlap)} scenarios")
            if val_test_overlap:
                print(f"  Val/Test overlap: {len(val_test_overlap)} scenarios")
            # Don't raise error here - let it continue so we can see the debug output
        else:
            print(f"\n✓ Split verification: No overlapping scenarios (correct)")

        # Save splits
        train_filename = generate_timestamped_filename("train_data", "csv")
        val_filename = generate_timestamped_filename("val_data", "csv")
        test_filename = generate_timestamped_filename("test_data", "csv")

        train_path = processed_dir / train_filename
        val_path = processed_dir / val_filename
        test_path = processed_dir / test_filename

        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
        test_data.to_csv(test_path, index=False)

        print(f"[OK] Data preprocessed and split")
        print(
            f"     Train: {len(train_data):,} rows ({len(train_data['scenario_id'].unique())} scenarios)"
        )
        print(
            f"     Val: {len(val_data):,} rows ({len(val_data['scenario_id'].unique())} scenarios)"
        )
        print(
            f"     Test: {len(test_data):,} rows ({len(test_data['scenario_id'].unique())} scenarios)"
        )

        return train_path, val_path, test_path

    return data_path, None, None


def _handle_data_analysis(data_path: Path, analysis_dir: Path) -> None:
    """Handle ANDES data analysis."""
    print("\n" + "=" * 70)
    print("STEP 2: DATA ANALYSIS (ANDES)")
    print("=" * 70)

    # Set matplotlib backend
    import matplotlib

    matplotlib.use("Agg")

    # Import analysis functions
    from scripts.analyze_data import (
        compute_cct_correlations,
        compute_dataset_statistics,
        compute_trajectory_statistics,
        generate_cct_figures,
        generate_parameter_space_figures,
        generate_summary_report,
        generate_trajectory_figures,
        load_data,
    )

    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load and analyze data
    df = load_data(data_path)
    stats_dict = compute_dataset_statistics(df)
    traj_stats_df = compute_trajectory_statistics(df)
    cct_correlations = compute_cct_correlations(df)

    # Generate figures
    generate_parameter_space_figures(df, figures_dir, ["png"])
    generate_trajectory_figures(df, figures_dir, ["png"])
    generate_cct_figures(df, figures_dir, ["png"])

    # Generate report
    generate_summary_report(stats_dict, traj_stats_df, cct_correlations, analysis_dir)

    # Save trajectory statistics
    if len(traj_stats_df) > 0:
        traj_stats_filename = generate_timestamped_filename("trajectory_statistics", "csv")
        traj_stats_path = analysis_dir / traj_stats_filename
        traj_stats_df.to_csv(traj_stats_path, index=False)

    print("[OK] Data analysis complete")


def _train_pinn_model(
    config: Dict,
    train_data_path: Path,
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    experiment_id: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[Dict]]:
    """Train PINN model."""
    if args.skip_pinn_training:
        if args.pinn_model_path:
            model_path = _expand_wildcard_path(args.pinn_model_path)
            if model_path and model_path.exists():
                print(f"[OK] Using existing PINN model: {model_path}")
                return model_path, None
            else:
                raise FileNotFoundError(f"PINN model not found: {args.pinn_model_path}")
        else:
            print("[SKIP] PINN training skipped (no model path provided)")
            return None, None

    print("\n" + "=" * 70)
    print("STEP 3: PINN TRAINING")
    print("=" * 70)

    # Override epochs if specified
    if args.pinn_epochs:
        config["training"]["epochs"] = args.pinn_epochs
    elif args.epochs:
        config["training"]["epochs"] = args.epochs

    # Override regularization if specified (for fair comparison with ML baseline)
    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout
        print(f"  ✓ Overriding dropout: {args.dropout} (applied to both PINN and ML baseline)")
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay
        print(
            f"  ✓ Overriding weight_decay: {args.weight_decay} (applied to both PINN and ML baseline)"
        )
    if args.early_stopping_patience is not None:
        config["training"]["early_stopping_patience"] = args.early_stopping_patience
        print(
            f"✓ Overriding early_stopping_patience: {args.early_stopping_patience} (applied to both"
            f"PINN and ML baseline)"
        )

    # Train model
    start_time = time.time()
    model_path, training_history = train_model(
        config=config,
        data_path=train_data_path,
        output_dir=dirs["pinn"],
        seed=config.get("reproducibility", {}).get("random_seed", 42),
        use_common_repository=False,  # Save in experiment directory
        force_retrain=False,
    )
    training_time = time.time() - start_time

    # Move model to pinn/model subdirectory
    pinn_model_dir = dirs["pinn_model"]
    pinn_model_dir.mkdir(parents=True, exist_ok=True)

    # Copy model to unified location
    if model_path.parent != pinn_model_dir:
        new_model_path = pinn_model_dir / model_path.name
        shutil.copy2(model_path, new_model_path)
        model_path = new_model_path

    # Also save to global models directory for reuse
    global_models_dir = _get_global_models_dir(experiment_id)
    pinn_global_dir = global_models_dir / "pinn"
    pinn_global_dir.mkdir(parents=True, exist_ok=True)

    # Copy model to global location
    global_model_path = pinn_global_dir / model_path.name
    shutil.copy2(model_path, global_model_path)

    # Copy training history to global location
    if training_history:
        history_filename = model_path.name.replace("best_model_", "training_history_").replace(
            ".pth", ".json"
        )
        history_path = pinn_model_dir / history_filename
        if history_path.exists():
            global_history_path = pinn_global_dir / history_filename
            shutil.copy2(history_path, global_history_path)

    # Create model metadata file with experiment information
    model_metadata = {
        "experiment_id": experiment_id,
        "model_type": "pinn",
        "model_path": str(global_model_path),
        "experiment_path": str(model_path),
        "training_time_seconds": training_time,
        "config": {
            "model": config.get("model", {}),
            "training": config.get("training", {}),
            "loss": config.get("loss", {}),
        },
        "data": {
            "train_path": str(train_data_path) if train_data_path else None,
        },
        "reproducibility": {
            "random_seed": config.get("reproducibility", {}).get("random_seed", 42),
            "git_commit": get_git_commit(),
            "git_branch": get_git_branch(),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    metadata_path = pinn_global_dir / f"model_metadata_{experiment_id}.json"
    save_json(model_metadata, metadata_path)

    print(f"[OK] PINN training complete: {model_path.name}")
    print(f"     Training time: {training_time:.1f} seconds")
    print(f"     Also saved to global location: {global_model_path}")
    print(f"     Metadata: {metadata_path.name}")

    return model_path, {
        "training_history": training_history,
        "training_time": training_time,
    }


def _train_ml_baseline(
    train_data_path: Path,
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    config: Dict,
    experiment_id: Optional[str] = None,
) -> Dict[str, Dict]:
    """Train ML baseline models."""
    if args.skip_ml_baseline_training:
        results = {}
        if args.ml_baseline_model_path:
            model_path = Path(args.ml_baseline_model_path)
            if model_path.exists():
                print(f"[OK] Using existing ML baseline model: {model_path}")
                # Infer model type from path
                model_type = "standard_nn"  # Default
                if "lstm" in str(model_path).lower():
                    model_type = "lstm"
                results[model_type] = {"model_path": str(model_path)}
                return results
            else:
                raise FileNotFoundError(f"ML baseline model not found: {model_path}")
        else:
            print("[SKIP] ML baseline training skipped (no model path provided)")
            return {}

    print("\n" + "=" * 70)
    print("STEP 4: ML BASELINE TRAINING")
    print("=" * 70)

    # Determine which models to train
    models_to_train = args.ml_baseline_models
    if isinstance(models_to_train, str):
        models_to_train = [m.strip() for m in models_to_train.split(",")]

    # Override epochs if specified
    epochs = args.ml_baseline_epochs or args.epochs or config.get("training", {}).get("epochs", 400)
    # Default to pe_direct (9 dimensions: Pe + tf, tc for steady-state context)
    # reactance (11 dimensions) is optional; pe_direct_7 (7-D) for ML-only best-baseline replication
    input_method = config.get("model", {}).get("input_method", "pe_direct")
    ml_baseline_config = config.get("ml_baseline", {})
    # ML baseline can use a different input method (e.g. pe_direct_7 for best R² replication)
    ml_input_method = ml_baseline_config.get("input_method", input_method)

    results = {}

    for model_type in models_to_train:
        print("\n" + "=" * 70)
        print(f"TRAINING {model_type.upper()}")
        print("=" * 70)

        # Create model-specific directory structure (similar to PINN)
        # Structure: ml_baseline/{model_type}/model/ (like pinn/model/)
        model_type_dir = dirs["ml_baseline"] / model_type
        model_dir = model_type_dir / "model"  # Separate model subdirectory like PINN
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create trainer: ML can override hidden_dims/dropout for best-baseline replication
        model_config_dict = config.get("model", {})
        pinn_dropout = float(model_config_dict.get("dropout", 0.0))

        # Priority: CLI > ml_baseline config > PINN config
        if args.dropout is not None:
            dropout = args.dropout
        elif args.ml_baseline_dropout is not None:
            dropout = args.ml_baseline_dropout
        elif ml_baseline_config.get("dropout") is not None:
            dropout = float(ml_baseline_config["dropout"])
        else:
            dropout = pinn_dropout

        hidden_dims = (
            model_config_dict.get("hidden_dims", [256, 256, 128, 128])
            if model_type == "standard_nn"
            else None
        )
        if model_type == "standard_nn" and ml_baseline_config.get("hidden_dims") is not None:
            hidden_dims = list(ml_baseline_config["hidden_dims"])

        trainer = MLBaselineTrainer(
            model_type=model_type,
            model_config={
                "hidden_dims": hidden_dims,
                "hidden_size": 128 if model_type == "lstm" else None,
                "num_layers": 2 if model_type == "lstm" else None,
                "activation": model_config_dict.get("activation", "tanh"),
                "dropout": dropout,
            },
        )

        # Prepare data - use preprocessed val file if available (for fair comparison with PINN)
        print(f"\nPreparing data...")
        print(f"  Data path: {train_data_path}")
        print(f"  Input method (ML): {ml_input_method}")

        # Look for corresponding val file (same as PINN does)
        val_data_path = None
        if "train_data_" in train_data_path.name:
            val_data_path = train_data_path.parent / train_data_path.name.replace(
                "train_data_", "val_data_"
            )
            if not val_data_path.exists():
                val_data_path = None

        loss_config = config.get("loss", {})
        scale_to_norm = loss_config.get("scale_to_norm", [1.0, 100.0])
        # ML-only override for scale_to_norm (e.g. [1, 100] for best-baseline replication)
        if ml_baseline_config.get("scale_to_norm") is not None:
            scale_to_norm = list(ml_baseline_config["scale_to_norm"])
        unstable_weight = float(ml_baseline_config.get("unstable_weight", 1.0))
        use_fixed_target_scale = bool(ml_baseline_config.get("use_fixed_target_scale", False))

        train_loader, val_loader, scalers = trainer.prepare_data(
            data_path=train_data_path,
            input_method=ml_input_method,
            val_data_path=(val_data_path if val_data_path and val_data_path.exists() else None),
            scale_to_norm=scale_to_norm,
            unstable_weight=unstable_weight,
            use_fixed_target_scale=use_fixed_target_scale,
        )

        if val_data_path and val_data_path.exists():
            print(f"  ✓ Using preprocessed validation file: {val_data_path.name}")
        else:
            print(f"  ⚠️  Warning: Preprocessed val file not found, will split on-the-fly")

        # Train model
        start_time = time.time()
        # Use same hyperparameters as PINN for fair comparison
        # Get from config if available, otherwise use defaults
        train_config = config.get("training", {})
        learning_rate = float(train_config.get("learning_rate", 1e-3))  # Match PINN default

        # Weight decay: CLI > ml_baseline config > PINN config
        pinn_weight_decay = float(train_config.get("weight_decay", 1e-5))
        if args.weight_decay is not None:
            weight_decay = args.weight_decay
        elif args.ml_baseline_weight_decay is not None:
            weight_decay = args.ml_baseline_weight_decay
        elif ml_baseline_config.get("weight_decay") is not None:
            weight_decay = float(ml_baseline_config["weight_decay"])
        else:
            weight_decay = pinn_weight_decay

        # Early stopping patience: CLI > ml_baseline config > training.ml_baseline_early_stopping_patience > PINN
        pinn_patience = train_config.get("early_stopping_patience", None)
        ml_patience_config = train_config.get("ml_baseline_early_stopping_patience", None)
        if args.early_stopping_patience is not None:
            early_stopping_patience = args.early_stopping_patience
        elif (
            args.ml_baseline_early_stopping_patience is not None
            and args.ml_baseline_early_stopping_patience > 0
        ):
            early_stopping_patience = args.ml_baseline_early_stopping_patience
        elif ml_baseline_config.get("early_stopping_patience") is not None:
            early_stopping_patience = int(ml_baseline_config["early_stopping_patience"])
        elif ml_patience_config is not None and ml_patience_config > 0:
            early_stopping_patience = int(ml_patience_config)
        elif pinn_patience is not None:
            early_stopping_patience = pinn_patience
        else:
            early_stopping_patience = None

        # Use same random seed as PINN for reproducibility
        random_seed = config.get("reproducibility", {}).get("random_seed", 42)

        # Read loss configuration from config (for fair comparison with PINN)
        lambda_ic = float(loss_config.get("lambda_ic", 10.0))  # Match PINN default

        # ML baseline fair-comparison options (optional; defaults = no weighting, current IC)
        pre_fault_weight = float(ml_baseline_config.get("pre_fault_weight", 1.0))
        lambda_steady_state = float(ml_baseline_config.get("lambda_steady_state", 0.0))
        use_ic_over_prefault = bool(ml_baseline_config.get("use_ic_over_prefault", False))
        two_phase_training = bool(ml_baseline_config.get("two_phase_training", False))
        phase1_epochs = int(ml_baseline_config.get("phase1_epochs", 0))
        phase2_pre_fault_weight = ml_baseline_config.get("phase2_pre_fault_weight")
        if phase2_pre_fault_weight is not None:
            phase2_pre_fault_weight = float(phase2_pre_fault_weight)

        # Validate early stopping patience for ML baseline (optional warning)
        if early_stopping_patience is not None and early_stopping_patience > 50:
            print(
                f"⚠️  Warning: Early stopping patience ({early_stopping_patience}) is high for ML baseline. "
                f"Recommended: 10-20 epochs. PINN typically uses 50-200, but ML baselines converge faster."
            )

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            lambda_ic=lambda_ic,
            scale_to_norm=scale_to_norm,
            pre_fault_weight=pre_fault_weight,
            lambda_steady_state=lambda_steady_state,
            use_ic_over_prefault=use_ic_over_prefault,
            two_phase_training=two_phase_training,
            phase1_epochs=phase1_epochs,
            phase2_pre_fault_weight=phase2_pre_fault_weight,
        )
        training_time = time.time() - start_time

        # Evaluate on validation set
        metrics = trainer.evaluate(val_loader)

        # Save model (best model is already loaded in trainer.model at this point)
        # The MLBaselineTrainer.train() method automatically loads the best model state
        # (based on best validation loss) into trainer.model before returning
        # Use timestamped filename like PINN for consistency
        from scripts.core.utils import generate_timestamped_filename

        model_filename = generate_timestamped_filename("best_model", "pth")
        model_path = model_dir / model_filename

        # Verify: Extract best model info from training history
        if history and "val_losses" in history and history["val_losses"]:
            best_val_loss = min(history["val_losses"])
            best_epoch = history["val_losses"].index(best_val_loss)
            print(f"\n[VERIFY] Saving best ML baseline model:")
            print(f"  Best validation loss: {best_val_loss:.6f}")
            print(f"  Best epoch: {best_epoch + 1}")
            print(f"  Model path: {model_path.name}")
            print(f"  ✅ Confirmed: Best model state is loaded in trainer.model")

        # Also save as model.pth for backward compatibility
        model_path_compat = model_dir / "model.pth"

        checkpoint_data = {
            "model_state_dict": trainer.model.state_dict(),
            "model_type": model_type,
            "model_config": trainer.model_config,
            "scalers": scalers,
            "input_method": ml_input_method,
            "training_history": history,
        }

        torch.save(checkpoint_data, model_path)
        torch.save(checkpoint_data, model_path_compat)  # Also save as model.pth

        # Also save to global models directory for reuse
        global_models_dir = _get_global_models_dir(experiment_id)
        ml_global_dir = global_models_dir / "ml_baseline" / model_type
        ml_global_dir.mkdir(parents=True, exist_ok=True)

        # Copy model to global location
        global_model_path = ml_global_dir / model_path.name
        shutil.copy2(model_path, global_model_path)

        # Copy training history to global location
        history_filename = generate_timestamped_filename("training_history", "json")
        history_path = model_dir / history_filename
        save_json(history, history_path)
        global_history_path = ml_global_dir / history_filename
        shutil.copy2(history_path, global_history_path)

        # Create model metadata file with experiment information
        model_metadata = {
            "experiment_id": experiment_id,
            "model_type": model_type,
            "model_path": str(global_model_path),
            "experiment_path": str(model_path),
            "training_time_seconds": training_time,
            "model_config": trainer.model_config,
            "input_method": ml_input_method,
            "training": {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "early_stopping_patience": early_stopping_patience,
            },
            "data": {
                "train_path": str(train_data_path) if train_data_path else None,
                "val_path": str(val_data_path) if val_data_path else None,
            },
            "reproducibility": {
                "random_seed": random_seed,
                "git_commit": get_git_commit(),
                "git_branch": get_git_branch(),
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        metadata_path = ml_global_dir / f"model_metadata_{experiment_id}.json"
        save_json(model_metadata, metadata_path)

        print(f"[OK] {model_type} training complete: {model_path.name}")
        print(f"     Also saved as: {model_path_compat.name} (for compatibility)")
        print(f"     Training time: {training_time:.1f} seconds")
        print(f"     Also saved to global location: {global_model_path}")
        print(f"     Metadata: {metadata_path.name}")

        # Save results
        results[model_type] = {
            "model_path": str(model_path),  # Use timestamped path (best model)
            "metrics": metrics,
            "training_history": history,
            "training_time": training_time,
        }

        # Save metrics (save to model_type_dir for consistency with other files)
        metrics_file = model_type_dir / "metrics.json"
        save_json(metrics, metrics_file)
        print(f"     Metrics: {metrics_file.name}")

        # Save ML baseline configuration (similar to PINN's config.yaml)
        # ML baseline loss weights: lambda_data=1.0 (implicit, MSE loss), lambda_ic from config (matches PINN)
        # Note: ML baseline doesn't have physics loss (no lambda_physics)
        # Values read from config for fair comparison with PINN
        ml_config = {
            "model_type": model_type,
            "model_config": trainer.model_config,
            "input_method": ml_input_method,
            "training": {
                "epochs": epochs,
                "learning_rate": learning_rate,  # From config (matches PINN)
                "weight_decay": weight_decay,  # From config (matches PINN)
            },
            "loss": {
                "lambda_data": 1.0,  # Implicit weight for data loss (MSE)
                "lambda_ic": lambda_ic,  # From config (matches PINN for fair comparison)
                "lambda_physics": None,  # ML baseline doesn't use physics loss
                "scale_to_norm": scale_to_norm,  # From config (matches PINN)
                "loss_function": "MSE",
                "total_loss": "data_loss + lambda_ic * ic_loss",
            },
            "ml_baseline_fair_comparison": {
                "pre_fault_weight": pre_fault_weight,
                "phase2_pre_fault_weight": phase2_pre_fault_weight,
                "lambda_steady_state": lambda_steady_state,
                "use_ic_over_prefault": use_ic_over_prefault,
            },
            "data": {
                "train_path": str(train_data_path) if train_data_path else None,
                "val_path": str(val_data_path) if val_data_path else None,
            },
            "reproducibility": {
                "random_seed": random_seed,  # Same seed as PINN for fair comparison
            },
            "experiment_id": dirs["root"].name,
            "training_time_seconds": training_time,
        }
        config_file = model_type_dir / "config.yaml"
        save_config(ml_config, config_file)
        print(f"     Config: {config_file.name}")

        # Note: Training history is saved in model_dir (same location as model checkpoints)
        # This matches PINN structure where training_history is in pinn/ directory
        # No need to duplicate it in model_type_dir

    return results


def _evaluate_pinn(
    config: Dict,
    model_path: Path,
    test_data_path: Path,
    dirs: Dict[str, Path],
    train_data_path: Optional[Path] = None,
) -> Dict:
    """Evaluate PINN model."""
    print("\n" + "=" * 70)
    print("STEP 5: PINN EVALUATION")
    print("=" * 70)

    # Create results directory structure similar to run_experiment.py
    # This ensures figures are generated in the expected location
    pinn_results_dir = dirs["pinn"] / "results"
    pinn_results_dir.mkdir(parents=True, exist_ok=True)

    # Ensure original dataset is accessible for figure generation
    # The evaluation function looks for data in output_dir.parent / "data"
    # Create a symlink or copy the original data file to pinn/data/ if needed
    # NOTE: Removed code that copies/links data to pinn/data/ directory
    # Data is stored only in common repository (data/processed/) to avoid duplication
    # The train_data_path already points to the common repository location
    # No need to create additional copies in experiment directories
    # This prevents large data files from being duplicated in each experiment directory

    results = evaluate_model(
        config=config,
        model_path=model_path,
        test_data_path=test_data_path,
        output_dir=pinn_results_dir,  # Use results/ structure like run_experiment.py
        device=config.get("training", {}).get("device", "auto"),
    )

    print("[OK] PINN evaluation complete")
    print(f"     Figures saved to: {pinn_results_dir / 'figures'}")
    return results


def _evaluate_ml_baseline(
    model_path: Path,
    test_data_path: Path,
    model_type: str,
    dirs: Dict[str, Path],
) -> Dict:
    """Evaluate ML baseline model."""
    print(f"\nEvaluating {model_type}...")

    # Use evaluate_ml_baseline.py as subprocess to reuse existing logic
    import subprocess

    eval_dir = dirs["ml_baseline"] / model_type / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    # FIX: test_data_path is already a pre-split test file, so pass it as --test-split-path
    # Also need --test-data for fallback (use the same file or find the full dataset)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_ml_baseline.py"),
        "--model-path",
        str(model_path),
        "--test-data",
        str(
            test_data_path
        ),  # Required parameter, but will be ignored if --test-split-path is provided
        "--test-split-path",
        str(test_data_path),  # FIX: Pass as test-split-path so it's used directly
        "--output-dir",
        str(eval_dir),
    ]

    # Run evaluation - show output in real-time for better visibility
    print(f"\n{'=' * 70}")
    print(f"STEP 6: ML BASELINE EVALUATION ({model_type.upper()})")
    print("=" * 70)
    result = subprocess.run(cmd, text=True)  # Remove capture_output to show output in real-time

    if result.returncode != 0:
        print(f"\n[WARNING] ML baseline evaluation returned error code {result.returncode}")
        return {}

    # Load evaluation results if available
    metrics_file = eval_dir / "metrics.json"
    if metrics_file.exists():
        results = load_json(metrics_file)
        # Print summary metrics if available
        if results:
            print(f"\n{'=' * 70}")
            print(f"ML BASELINE ({model_type}) EVALUATION SUMMARY")
            print("=" * 70)
            if "delta_rmse" in results:
                print(f"  RMSE Delta: {results.get('delta_rmse', 'N/A'):.6f} rad")
            if "omega_rmse" in results:
                print(f"  RMSE Omega: {results.get('omega_rmse', 'N/A'):.6f} pu")
            if "delta_mae" in results:
                print(f"  MAE Delta: {results.get('delta_mae', 'N/A'):.6f} rad")
            if "omega_mae" in results:
                print(f"  MAE Omega: {results.get('omega_mae', 'N/A'):.6f} pu")
            if "delta_r2" in results:
                print(f"  R² Delta: {results.get('delta_r2', 'N/A'):.6f}")
            if "omega_r2" in results:
                print(f"  R² Omega: {results.get('omega_r2', 'N/A'):.6f}")
            if "stability_classification_accuracy" in results:
                print(
                    f"  Stability Classification: {results.get('stability_classification_accuracy', 0) * 100:.1f}%"
                )
            print(f"  Results saved to: {metrics_file}")
            print("=" * 70)
    else:
        results = {}

    print(f"\n[OK] {model_type} evaluation complete")
    return results


def _compare_models(
    pinn_model_path: Path,
    ml_baseline_model_path: Path,
    test_data_path: Path,
    dirs: Dict[str, Path],
    config: Dict,
    *,
    delta_split_by_stability: bool = True,
    overlaid_plots: bool = False,
) -> Dict:
    """Compare PINN and ML baseline models."""
    print("\n" + "=" * 70)
    print("STEP 6: MODEL COMPARISON")
    print("=" * 70)

    # Use compare_models.py as subprocess to reuse existing logic
    import subprocess

    # Find PINN config if available
    pinn_config_path = dirs["root"] / "config.yaml"
    if not pinn_config_path.exists():
        pinn_config_path = None

    # Build command
    # FIX: test_data_path is already a pre-split test file, so pass it as --test-split-path
    # Also pass --full-trajectory-data (same path) so compare_models argparse is satisfied;
    # load_test_data prefers --test-split-path when it exists.
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "compare_models.py"),
        "--ml-baseline-model",
        str(ml_baseline_model_path),
        "--pinn-model",
        str(pinn_model_path),
        "--full-trajectory-data",
        str(test_data_path),
        "--test-split-path",
        str(test_data_path),
        "--output-dir",
        str(dirs["comparison"]),
    ]

    if pinn_config_path and pinn_config_path.exists():
        cmd.extend(["--pinn-config", str(pinn_config_path)])

    if not delta_split_by_stability:
        cmd.append("--combine-delta-only-figure")
    if overlaid_plots:
        cmd.append("--overlaid-plots")

    # Run comparison (UTF-8 encoding to avoid Windows cp949 decode errors on Unicode output)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # Always show comparison script output so figure-generation failures are visible
    if result.stdout is not None and result.stdout.strip():
        print(result.stdout)
    if result.stderr is not None and result.stderr.strip():
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"[WARNING] Comparison script returned error code {result.returncode}")
        return {}

    # Warn if no comparison figures were produced
    figures_dir = dirs["comparison_figures"]
    if figures_dir.exists():
        pngs = list(figures_dir.glob("*.png"))
        if not pngs:
            print(
                "[WARNING] Model comparison completed but no figures were saved to "
                f"{figures_dir}. Check comparison script output above for errors."
            )
    else:
        print(
            "[WARNING] Comparison figures directory was not created: "
            f"{figures_dir}. Overlaid and delta-only plots may have failed."
        )

    # Load comparison results and verify expected content
    comparison_results_file = dirs["comparison"] / "comparison_results.json"
    if comparison_results_file.exists():
        comparison_results = load_json(comparison_results_file)
        from scripts.core.experiment_ui import DELTA_ONLY_EXPERIMENT_UI

        expected_keys = (
            ["delta_comparison"]
            if DELTA_ONLY_EXPERIMENT_UI
            else ["delta_comparison", "omega_comparison"]
        )
        missing = [k for k in expected_keys if k not in (comparison_results or {})]
        if missing:
            print(
                "[WARNING] Comparison results file exists but is missing expected keys: "
                f"{missing}. Comparison may have failed or produced partial output."
            )
    else:
        comparison_results = {}
        print(
            "[WARNING] Comparison results file not found: "
            f"{comparison_results_file}. Comparison script may have failed to write results."
        )

    print("[OK] Model comparison complete")
    return comparison_results


def _generate_experiment_summary(
    experiment_id: str,
    config: Dict,
    dirs: Dict[str, Path],
    data_paths: Tuple[Optional[Path], Optional[Path], Optional[Path]],
    pinn_results: Optional[Dict],
    ml_baseline_results: Dict,
    comparison_results: Optional[Dict],
    invocation: Optional[Dict] = None,
) -> Dict:
    """Generate comprehensive experiment summary."""
    train_path, val_path, test_path = data_paths

    summary = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_path": str(config.get("_config_path", "")),
        "reproducibility": {
            "git_commit": get_git_commit(),
            "git_branch": get_git_branch(),
            "package_versions": get_package_versions(),
            "python_version": sys.version,
            "random_seed": config.get("reproducibility", {}).get("random_seed", 42),
        },
        "data": {
            "source": (
                "generated" if train_path and "trajectory_data" in str(train_path) else "reused"
            ),
            "train_path": str(train_path) if train_path else None,
            "val_path": str(val_path) if val_path else None,
            "test_path": str(test_path) if test_path else None,
        },
        "pinn": pinn_results or {},
        "ml_baseline": ml_baseline_results,
        "comparison": comparison_results or {},
    }

    if invocation:
        summary["invocation_latest"] = {
            "timestamp": invocation.get("timestamp"),
            "cwd": invocation.get("cwd"),
            "project_root": invocation.get("project_root"),
            "argv": invocation.get("argv"),
            "command_posix": invocation.get("command_posix"),
            "command_windows_cmd": invocation.get("command_windows_cmd"),
            "rerun_md": str((dirs["root"] / "RERUN.md").resolve()),
            "invocations_log": str((dirs["root"] / "run_invocations.jsonl").resolve()),
        }

    return summary


def main():
    """Main workflow orchestration."""
    parser = argparse.ArgumentParser(description="Complete experiment workflow")

    # Required arguments
    parser.add_argument("--config", type=str, required=True, help="YAML config file path")

    # Data options
    parser.add_argument("--skip-data-generation", action="store_true", help="Reuse existing data")
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip data preprocessing/splitting (keep raw data only)",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of data even if it exists in common repository",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Path to existing data directory"
    )
    parser.add_argument("--data-path", type=str, default=None, help="Direct path to data file")
    parser.add_argument(
        "--skip-data-analysis", action="store_true", help="Skip ANDES data analysis"
    )

    # Training options
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for both models")
    parser.add_argument(
        "--pinn-epochs", type=int, default=None, help="Override epochs for PINN only"
    )
    parser.add_argument(
        "--ml-baseline-epochs",
        type=int,
        default=None,
        help="Override epochs for ML baseline only",
    )

    # Regularization options (applied to both models for fair comparison)
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate for both PINN and ML baseline (default: from config, or 0.0 if not in config)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay for both PINN and ML baseline (default: from config, or 1e-5 if not in config)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Early stopping patience for both models (default: from config)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help=(
            "Override reproducibility.random_seed in config (PyTorch/NumPy/Python; "
            "used for PINN and ML baseline training in this run)."
        ),
    )

    # PINN options
    parser.add_argument("--skip-pinn-training", action="store_true", help="Skip PINN training")
    parser.add_argument(
        "--pinn-model-path", type=str, default=None, help="Path to existing PINN model"
    )
    parser.add_argument(
        "--lambda-physics",
        type=float,
        default=None,
        help="Override loss.lambda_physics for PINN training only (ML has no physics loss)",
    )
    parser.add_argument("--skip-pinn-evaluation", action="store_true", help="Skip PINN evaluation")

    # ML baseline options
    parser.add_argument(
        "--skip-ml-baseline-training",
        action="store_true",
        help="Skip ML baseline training",
    )
    parser.add_argument(
        "--ml-baseline-model-path",
        type=str,
        default=None,
        help="Path to existing ML baseline model",
    )
    parser.add_argument(
        "--ml-baseline-models",
        type=str,
        default="standard_nn",
        help="ML baseline models to train (comma-separated)",
    )
    parser.add_argument(
        "--ml-baseline-dropout",
        type=float,
        default=None,
        help="Dropout rate for ML baseline only (overridden by --dropout if provided)",
    )
    parser.add_argument(
        "--ml-baseline-weight-decay",
        type=float,
        default=None,
        help="Weight decay for ML baseline only (overridden by --weight-decay if provided)",
    )
    parser.add_argument(
        "--ml-baseline-early-stopping-patience",
        type=int,
        default=None,
        help="Early stopping patience for ML baseline only (overridden by --early-stopping-patience if provided)",
    )
    parser.add_argument(
        "--ml-baseline-unstable-weight",
        type=float,
        default=None,
        help="Override ml_baseline.unstable_weight (unstable-scenario row reweighting)",
    )
    parser.add_argument(
        "--ml-baseline-omega-weight",
        type=float,
        default=None,
        help="Override ml_baseline.scale_to_norm[1] (sets scale_to_norm to [1.0, value] for ML MSE)",
    )
    parser.add_argument(
        "--skip-ml-baseline-evaluation",
        action="store_true",
        help="Skip ML baseline evaluation",
    )

    # Comparison options
    parser.add_argument("--skip-comparison", action="store_true", help="Skip model comparison")
    parser.add_argument(
        "--combine-delta-only-comparison-figure",
        action="store_true",
        help=(
            "Emit one combined delta-only comparison PNG instead of separate stable/unstable figures "
            "(forwards to compare_models.py --combine-delta-only-figure). "
            "YAML: set evaluation.comparison.delta_split_by_stability: false to disable split."
        ),
    )
    parser.add_argument(
        "--overlaid-comparison-plots",
        action="store_true",
        help=(
            "Also emit model_comparison_overlaid_*.png (forwards to compare_models.py --overlaid-plots). "
            "YAML: evaluation.comparison.overlaid_plots: true"
        ),
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/complete_experiments",
        help="Base output directory",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help=(
            "Reuse this experiment folder under --output-dir (e.g. exp_20260404_120208). "
            "Skips pre-run cleanup so existing PINN artifacts are preserved. "
            "Default: new timestamped exp_* id."
        ),
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip cleanup of duplicate files at the end",
    )
    parser.add_argument(
        "--no-prepare-clean",
        action="store_true",
        help="Skip cleaning existing files when starting experiment",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    config["_config_path"] = str(config_path)

    if args.random_seed is not None:
        config.setdefault("reproducibility", {})
        config["reproducibility"]["random_seed"] = int(args.random_seed)
        print(f"[CLI] Overriding reproducibility.random_seed = {args.random_seed}")

    if args.lambda_physics is not None:
        config.setdefault("loss", {})
        config["loss"]["lambda_physics"] = float(args.lambda_physics)
        print(f"[CLI] Overriding loss.lambda_physics = {args.lambda_physics} (PINN only)")

    if args.ml_baseline_unstable_weight is not None:
        config.setdefault("ml_baseline", {})
        config["ml_baseline"]["unstable_weight"] = float(args.ml_baseline_unstable_weight)
        print(f"[CLI] Overriding ml_baseline.unstable_weight = {args.ml_baseline_unstable_weight}")
    if args.ml_baseline_omega_weight is not None:
        config.setdefault("ml_baseline", {})
        ow = float(args.ml_baseline_omega_weight)
        config["ml_baseline"]["scale_to_norm"] = [1.0, ow]
        print(f"[CLI] Overriding ml_baseline.scale_to_norm = [1.0, {ow}] (omega MSE weight)")

    # Create experiment directory
    base_output_dir = Path(args.output_dir)
    if args.experiment_id:
        experiment_id = args.experiment_id.strip()
        if not experiment_id:
            print("[ERROR] --experiment-id must be non-empty")
            sys.exit(1)
        resume_same_folder = True
    else:
        experiment_id = generate_experiment_id()
        resume_same_folder = False

    experiment_dir = base_output_dir / experiment_id

    # Prepare clean experiment directory (remove duplicates if re-running)
    # Never run destructive cleanup when reusing an existing experiment folder.
    if not args.no_prepare_clean and not resume_same_folder:
        prepare_clean_experiment_directory(experiment_dir, cleanup_existing=True)
    elif resume_same_folder:
        print(f"[INFO] Reusing experiment directory (no pre-run cleanup): {experiment_dir}")

    dirs = _create_unified_directory(base_output_dir, experiment_id)

    # Save config to experiment directory
    save_config(config, dirs["root"] / "config.yaml")

    invocation_meta = _collect_invocation_metadata()
    _save_experiment_invocation_artifacts(dirs["root"], invocation_meta)
    print(f"[INFO] Saved rerun hints: {dirs['root'] / 'RERUN.md'}")

    print("=" * 70)
    print("COMPLETE EXPERIMENT WORKFLOW")
    print("=" * 70)
    print(f"Experiment ID: {experiment_id}")
    print(f"Output directory: {dirs['root']}")
    print(f"Config: {config_path}")

    try:
        # Step 1: Data generation/reuse
        train_path, val_path, test_path = _handle_data_generation(config, dirs, args)

        # Step 2: Data analysis (if not skipped)
        if not args.skip_data_analysis and train_path:
            _handle_data_analysis(train_path, dirs["analysis"])

        # Step 3: PINN training
        pinn_model_path, pinn_training_info = _train_pinn_model(
            config, train_path, dirs, args, experiment_id
        )

        # Step 4: ML baseline training
        ml_baseline_results = _train_ml_baseline(train_path, dirs, args, config, experiment_id)

        # Step 5: PINN evaluation
        pinn_eval_results = None
        if not args.skip_pinn_evaluation and pinn_model_path and test_path:
            pinn_eval_results = _evaluate_pinn(config, pinn_model_path, test_path, dirs, train_path)

        # Step 6: ML baseline evaluation
        if not args.skip_ml_baseline_evaluation and test_path:
            for model_type, model_info in ml_baseline_results.items():
                if "model_path" in model_info:
                    ml_model_path = Path(model_info["model_path"])
                    # Verify: Confirm we're using the best ML baseline model
                    print(f"\n[VERIFY] Using best ML baseline model ({model_type}) for evaluation:")
                    print(f"  Model path: {ml_model_path}")
                    print(f"  Model name: {ml_model_path.name}")
                    if (
                        "best_model" not in ml_model_path.name
                        and "model.pth" not in ml_model_path.name
                    ):
                        print(
                            f"  ⚠️  WARNING: Model filename doesn't contain 'best_model' or 'model.pth'"
                        )
                    eval_results = _evaluate_ml_baseline(
                        ml_model_path,
                        test_path,
                        model_type,
                        dirs,
                    )
                    ml_baseline_results[model_type]["evaluation"] = eval_results

        # Step 7: Model comparison
        comparison_results = None
        if not args.skip_comparison and pinn_model_path and test_path:
            # Get first ML baseline model for comparison
            ml_model_path = None
            for model_type, model_info in ml_baseline_results.items():
                if "model_path" in model_info:
                    ml_model_path = Path(model_info["model_path"])
                    break

            if ml_model_path:
                cfg_split = (
                    config.get("evaluation", {})
                    .get("comparison", {})
                    .get("delta_split_by_stability", True)
                )
                if isinstance(cfg_split, str):
                    cfg_split = str(cfg_split).strip().lower() in ("1", "true", "yes")
                split_delta = bool(cfg_split) and not args.combine_delta_only_comparison_figure
                cfg_overlaid = (
                    config.get("evaluation", {}).get("comparison", {}).get("overlaid_plots", False)
                )
                if isinstance(cfg_overlaid, str):
                    cfg_overlaid = str(cfg_overlaid).strip().lower() in ("1", "true", "yes")
                want_overlaid = bool(cfg_overlaid) or args.overlaid_comparison_plots
                comparison_results = _compare_models(
                    pinn_model_path,
                    ml_model_path,
                    test_path,
                    dirs,
                    config,
                    delta_split_by_stability=split_delta,
                    overlaid_plots=want_overlaid,
                )

        # Generate experiment summary
        pinn_results = {
            "model_path": str(pinn_model_path) if pinn_model_path else None,
            "training": pinn_training_info,
            "evaluation": pinn_eval_results,
        }

        summary = _generate_experiment_summary(
            experiment_id,
            config,
            dirs,
            (train_path, val_path, test_path),
            pinn_results,
            ml_baseline_results,
            comparison_results,
            invocation=invocation_meta,
        )

        # Save summary
        summary_file = dirs["root"] / "experiment_summary.json"
        save_json(summary, summary_file)

        record_complete_experiment(summary, dirs["root"], PROJECT_ROOT)

        # Clean up duplicate files before generating final outputs
        if not args.no_cleanup:
            print("\n" + "=" * 70)
            print("CLEANING UP DUPLICATE FILES")
            print("=" * 70)
            cleanup_stats = cleanup_experiment_directory(
                dirs["root"],
                cleanup_figures=True,
                cleanup_old_checkpoints=True,
                cleanup_duplicates=True,
                keep_latest=True,
            )

            if cleanup_stats["total_removed"] > 0:
                print(f"Removed {cleanup_stats['total_removed']} duplicate/old files:")
                print(f"  - Figures: {cleanup_stats['figures_removed']}")
                print(f"  - Checkpoints: {cleanup_stats['checkpoints_removed']}")
                print(f"  - Other duplicates: {cleanup_stats['duplicates_removed']}")
            else:
                print("No duplicate files found - directory is clean")

        # Generate experiment remarks markdown file
        analysis_report_path = None
        if dirs["analysis"].exists():
            # Find the most recent analysis report
            analysis_reports = list(dirs["analysis"].glob("analysis_summary_report_*.txt"))
            if analysis_reports:
                analysis_report_path = max(analysis_reports, key=lambda p: p.stat().st_mtime)

        comparison_results_path = None
        if (dirs["root"] / "comparison" / "comparison_results.json").exists():
            comparison_results_path = dirs["root"] / "comparison" / "comparison_results.json"

        try:
            remarks_file = generate_experiment_remarks(
                experiment_dir=dirs["root"],
                experiment_id=experiment_id,
                config=config,
                summary=summary,
                analysis_report_path=analysis_report_path,
                comparison_results_path=comparison_results_path,
            )
            print(f"Experiment remarks saved to: {remarks_file}")
        except Exception as e:
            print(f"[WARNING] Failed to generate experiment remarks: {e}")
            import traceback

            traceback.print_exc()

        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        print(f"Summary saved to: {summary_file}")
        print(f"All results in: {dirs['root']}")

    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
