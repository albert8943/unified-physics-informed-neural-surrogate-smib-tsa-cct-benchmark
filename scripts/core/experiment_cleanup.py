"""
Experiment Directory Cleanup Utilities.

This module provides functions to clean up experiment directories,
removing duplicate files and ensuring clean, unique outputs.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict


def cleanup_experiment_directory(
    experiment_dir: Path,
    cleanup_figures: bool = True,
    cleanup_old_checkpoints: bool = True,
    cleanup_duplicates: bool = True,
    keep_latest: bool = True,
) -> Dict[str, int]:
    """
    Clean up experiment directory by removing duplicates and old files.

    Parameters
    ----------
    experiment_dir : Path
        Experiment directory to clean
    cleanup_figures : bool
        If True, remove duplicate figure files (keep only latest)
    cleanup_old_checkpoints : bool
        If True, remove old checkpoint files (keep only best model)
    cleanup_duplicates : bool
        If True, remove duplicate files based on base name
    keep_latest : bool
        If True, keep the latest version of each file type

    Returns
    -------
    Dict[str, int]
        Dictionary with cleanup statistics:
        - 'figures_removed': Number of duplicate figures removed
        - 'checkpoints_removed': Number of old checkpoints removed
        - 'duplicates_removed': Number of duplicate files removed
        - 'total_removed': Total files removed
    """
    stats = {
        "figures_removed": 0,
        "checkpoints_removed": 0,
        "duplicates_removed": 0,
        "total_removed": 0,
    }

    if not experiment_dir.exists():
        return stats

    # Clean up figures
    if cleanup_figures:
        figures_removed = _cleanup_figure_duplicates(experiment_dir, keep_latest=keep_latest)
        stats["figures_removed"] = figures_removed

    # Clean up old checkpoints
    if cleanup_old_checkpoints:
        checkpoints_removed = _cleanup_old_checkpoints(experiment_dir)
        stats["checkpoints_removed"] = checkpoints_removed

    # Clean up general duplicates
    if cleanup_duplicates:
        duplicates_removed = _cleanup_duplicate_files(experiment_dir, keep_latest=keep_latest)
        stats["duplicates_removed"] = duplicates_removed

    stats["total_removed"] = (
        stats["figures_removed"] + stats["checkpoints_removed"] + stats["duplicates_removed"]
    )

    return stats


def _cleanup_figure_duplicates(experiment_dir: Path, keep_latest: bool = True) -> int:
    """Remove duplicate figure files, keeping only the latest version."""
    removed = 0

    # Find all figure directories
    figure_dirs = [
        experiment_dir / "analysis" / "figures",
        experiment_dir / "pinn" / "results" / "figures",
        experiment_dir / "ml_baseline" / "standard_nn" / "evaluation" / "figures",
        experiment_dir / "comparison" / "figures",
        experiment_dir / "figures",  # Common figures directory
    ]

    for fig_dir in figure_dirs:
        if not fig_dir.exists():
            continue

        # Group files by base name (without timestamp)
        file_groups = defaultdict(list)

        for file_path in fig_dir.glob("*.png"):
            base_name = _extract_base_name(file_path.stem)
            file_groups[base_name].append(file_path)

        # For each group, keep only the latest file
        for base_name, files in file_groups.items():
            if len(files) > 1:
                # Sort by modification time (newest first)
                files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

                if keep_latest:
                    # Keep the latest, remove others
                    for file_to_remove in files_sorted[1:]:
                        try:
                            file_to_remove.unlink()
                            removed += 1
                        except Exception as e:
                            print(f"  ⚠️  Could not remove {file_to_remove.name}: {e}")

    return removed


def _cleanup_old_checkpoints(experiment_dir: Path) -> int:
    """Remove old checkpoint files, keeping only the best model."""
    removed = 0

    # Find checkpoint directories (check both direct and subdirectory locations)
    checkpoint_dirs = []

    # PINN directories
    pinn_dir = experiment_dir / "pinn"
    if pinn_dir.exists():
        checkpoint_dirs.append(pinn_dir)  # Direct location (older experiments)
        pinn_model_dir = pinn_dir / "model"
        if pinn_model_dir.exists():
            checkpoint_dirs.append(pinn_model_dir)  # Subdirectory (newer experiments)

    # ML baseline directories
    ml_baseline_dir = experiment_dir / "ml_baseline"
    if ml_baseline_dir.exists():
        # Check all model types (standard_nn, lstm, etc.)
        for model_type_dir in ml_baseline_dir.iterdir():
            if model_type_dir.is_dir():
                checkpoint_dirs.append(model_type_dir)  # Direct location (older)
                model_subdir = model_type_dir / "model"
                if model_subdir.exists():
                    checkpoint_dirs.append(model_subdir)  # Subdirectory (newer)

    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            continue

        # Find all checkpoint files (recursive search in this directory only)
        checkpoint_files = list(checkpoint_dir.glob("epoch_*_checkpoint_*.pth"))
        best_model_files = list(checkpoint_dir.glob("best_model_*.pth"))

        # Keep only the latest best model
        if len(best_model_files) > 1:
            best_model_files_sorted = sorted(
                best_model_files, key=lambda p: p.stat().st_mtime, reverse=True
            )
            for file_to_remove in best_model_files_sorted[1:]:
                try:
                    file_to_remove.unlink()
                    removed += 1
                except Exception as e:
                    print(f"  ⚠️  Could not remove {file_to_remove.name}: {e}")

        # Remove all intermediate checkpoints (keep only epoch_0000 and final)
        if checkpoint_files:
            checkpoint_files_sorted = sorted(checkpoint_files, key=lambda p: p.stat().st_mtime)
            # Keep first (epoch_0000) and last (final epoch)
            files_to_remove = (
                checkpoint_files_sorted[1:-1] if len(checkpoint_files_sorted) > 2 else []
            )
            for file_to_remove in files_to_remove:
                try:
                    file_to_remove.unlink()
                    removed += 1
                except Exception as e:
                    print(f"  ⚠️  Could not remove {file_to_remove.name}: {e}")

    return removed


def _cleanup_duplicate_files(experiment_dir: Path, keep_latest: bool = True) -> int:
    """Remove duplicate files based on base name pattern."""
    removed = 0

    # Common file patterns that might have duplicates
    patterns = ["*.png", "*.jpg", "*.pdf", "*.json", "*.csv"]

    for pattern in patterns:
        # Find all files matching pattern
        all_files = list(experiment_dir.rglob(pattern))

        # Group by base name (without timestamp)
        file_groups = defaultdict(list)

        for file_path in all_files:
            # Skip if in subdirectories we've already processed
            if any(
                part in str(file_path)
                for part in ["/model/", "/checkpoint/", "/archive/", "/backup/"]
            ):
                continue

            base_name = _extract_base_name(file_path.stem)
            file_groups[base_name].append(file_path)

        # For each group with duplicates, keep only the latest
        for base_name, files in file_groups.items():
            if len(files) > 1:
                # Sort by modification time (newest first)
                files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

                if keep_latest:
                    # Keep the latest, remove others
                    for file_to_remove in files_sorted[1:]:
                        try:
                            file_to_remove.unlink()
                            removed += 1
                        except Exception as e:
                            print(f"  ⚠️  Could not remove {file_to_remove.name}: {e}")

    return removed


def _extract_base_name(filename: str) -> str:
    """
    Extract base name from timestamped filename.

    Examples:
    ---------
    >>> _extract_base_name("loss_curves_20260105_185706")
    'loss_curves'
    >>> _extract_base_name("test_scenarios_predictions_20260105_185709")
    'test_scenarios_predictions'
    >>> _extract_base_name("parameter_space_coverage_3d_20260105_183420")
    'parameter_space_coverage_3d'
    """
    # Remove timestamp pattern: _YYYYMMDD_HHMMSS
    import re

    # Pattern: _ followed by 8 digits, underscore, 6 digits at the end
    pattern = r"_\d{8}_\d{6}$"
    base_name = re.sub(pattern, "", filename)

    return base_name


def prepare_clean_experiment_directory(
    experiment_dir: Path,
    cleanup_existing: bool = True,
) -> None:
    """
    Prepare a clean experiment directory before starting a new experiment.

    This function:
    1. Creates the directory structure
    2. Optionally cleans up any existing files

    Parameters
    ----------
    experiment_dir : Path
        Experiment directory to prepare
    cleanup_existing : bool
        If True, remove existing files before starting
    """
    if cleanup_existing and experiment_dir.exists():
        # Clean up existing files
        print(f"Cleaning up existing experiment directory: {experiment_dir}")
        stats = cleanup_experiment_directory(
            experiment_dir,
            cleanup_figures=True,
            cleanup_old_checkpoints=True,
            cleanup_duplicates=True,
            keep_latest=True,
        )

        if stats["total_removed"] > 0:
            print(f"  Removed {stats['total_removed']} duplicate/old files:")
            print(f"    - Figures: {stats['figures_removed']}")
            print(f"    - Checkpoints: {stats['checkpoints_removed']}")
            print(f"    - Duplicates: {stats['duplicates_removed']}")
        else:
            print("  No duplicate files found")


def get_duplicate_files(experiment_dir: Path) -> Dict[str, List[Path]]:
    """
    Identify duplicate files in experiment directory.

    Parameters
    ----------
    experiment_dir : Path
        Experiment directory to scan

    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping base names to lists of duplicate file paths
    """
    duplicates = defaultdict(list)

    # Scan figure directories
    figure_dirs = [
        experiment_dir / "analysis" / "figures",
        experiment_dir / "pinn" / "results" / "figures",
        experiment_dir / "ml_baseline" / "standard_nn" / "evaluation" / "figures",
        experiment_dir / "comparison" / "figures",
        experiment_dir / "figures",
    ]

    for fig_dir in figure_dirs:
        if not fig_dir.exists():
            continue

        file_groups = defaultdict(list)

        for file_path in fig_dir.glob("*.png"):
            base_name = _extract_base_name(file_path.stem)
            file_groups[base_name].append(file_path)

        # Add groups with duplicates
        for base_name, files in file_groups.items():
            if len(files) > 1:
                duplicates[base_name].extend(files)

    return dict(duplicates)
