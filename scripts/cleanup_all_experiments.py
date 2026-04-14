#!/usr/bin/env python
"""
Batch cleanup script to remove old accumulated best_model files from all experiments.

This script:
1. Finds all experiment directories in outputs/complete_experiments/
2. Removes duplicate best_model_*.pth files (keeps only latest)
3. Removes old intermediate checkpoints
4. Reports total space saved

Usage:
    python scripts/cleanup_all_experiments.py [--dry-run] [--experiments-dir PATH]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.experiment_cleanup import cleanup_experiment_directory


def get_experiment_directories(experiments_dir: Path) -> list[Path]:
    """Find all experiment directories."""
    if not experiments_dir.exists():
        print(f"[ERROR] Experiments directory not found: {experiments_dir}")
        return []

    experiment_dirs = [
        d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")
    ]

    return sorted(experiment_dirs)


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def calculate_directory_size(directory: Path) -> int:
    """Calculate total size of all files in directory."""
    total_size = 0
    try:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    except Exception:
        pass
    return total_size


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old model files from all experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually deleting files",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "complete_experiments",
        help="Path to experiments directory (default: outputs/complete_experiments)",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure cleanup (only clean checkpoints)",
    )
    parser.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="Skip duplicate file cleanup (only clean checkpoints)",
    )

    args = parser.parse_args()

    experiments_dir = args.experiments_dir.resolve()

    print("=" * 70)
    print("BATCH EXPERIMENT CLEANUP")
    print("=" * 70)
    print(f"Experiments directory: {experiments_dir}")
    print(
        f"Mode: {'DRY RUN (no files will be deleted)' if args.dry_run else 'LIVE (files will be deleted)'}"
    )
    print()

    # Find all experiment directories
    experiment_dirs = get_experiment_directories(experiments_dir)

    if not experiment_dirs:
        print("[ERROR] No experiment directories found")
        return

    print(f"Found {len(experiment_dirs)} experiment directories")
    print()

    # Track statistics
    total_stats: Dict[str, int] = {
        "figures_removed": 0,
        "checkpoints_removed": 0,
        "duplicates_removed": 0,
        "total_removed": 0,
    }

    experiments_cleaned = 0
    experiments_skipped = 0

    # Process each experiment
    for i, exp_dir in enumerate(experiment_dirs, 1):
        print(f"[{i}/{len(experiment_dirs)}] Processing {exp_dir.name}...", end=" ")

        if args.dry_run:
            # In dry-run mode, just check what would be removed
            from scripts.core.experiment_cleanup import (
                get_duplicate_files,
                _cleanup_old_checkpoints,
            )

            # Count best_model files
            pinn_dir = exp_dir / "pinn"
            pinn_model_dir = pinn_dir / "model" if pinn_dir.exists() else None

            best_model_count = 0
            if pinn_dir.exists():
                best_model_count += len(list(pinn_dir.glob("best_model_*.pth")))
            if pinn_model_dir and pinn_model_dir.exists():
                best_model_count += len(list(pinn_model_dir.glob("best_model_*.pth")))

            if best_model_count > 1:
                print(f"[OK] Would remove {best_model_count - 1} old best_model files")
                total_stats["checkpoints_removed"] += best_model_count - 1
                experiments_cleaned += 1
            else:
                print("[OK] No cleanup needed")
                experiments_skipped += 1
        else:
            # Actually run cleanup
            try:
                stats = cleanup_experiment_directory(
                    exp_dir,
                    cleanup_figures=not args.skip_figures,
                    cleanup_old_checkpoints=True,
                    cleanup_duplicates=not args.skip_duplicates,
                    keep_latest=True,
                )

                if stats["total_removed"] > 0:
                    print(f"[OK] Removed {stats['total_removed']} files")
                    print(f"   - Checkpoints: {stats['checkpoints_removed']}")
                    if not args.skip_figures:
                        print(f"   - Figures: {stats['figures_removed']}")
                    if not args.skip_duplicates:
                        print(f"   - Duplicates: {stats['duplicates_removed']}")

                    total_stats["figures_removed"] += stats["figures_removed"]
                    total_stats["checkpoints_removed"] += stats["checkpoints_removed"]
                    total_stats["duplicates_removed"] += stats["duplicates_removed"]
                    total_stats["total_removed"] += stats["total_removed"]
                    experiments_cleaned += 1
                else:
                    print("[OK] No cleanup needed")
                    experiments_skipped += 1
            except Exception as e:
                print(f"[ERROR] Error: {e}")

    # Print summary
    print()
    print("=" * 70)
    print("CLEANUP SUMMARY")
    print("=" * 70)
    print(f"Experiments processed: {len(experiment_dirs)}")
    print(f"Experiments cleaned: {experiments_cleaned}")
    print(f"Experiments skipped: {experiments_skipped}")
    print()

    if total_stats["total_removed"] > 0:
        print("Files removed:")
        print(f"  - Checkpoints: {total_stats['checkpoints_removed']}")
        if not args.skip_figures:
            print(f"  - Figures: {total_stats['figures_removed']}")
        if not args.skip_duplicates:
            print(f"  - Duplicates: {total_stats['duplicates_removed']}")
        print(f"  - Total: {total_stats['total_removed']}")
    else:
        print("No files needed cleanup - all experiments are clean!")

    if args.dry_run:
        print()
        print("[INFO] This was a dry run. Run without --dry-run to actually delete files.")


if __name__ == "__main__":
    main()
