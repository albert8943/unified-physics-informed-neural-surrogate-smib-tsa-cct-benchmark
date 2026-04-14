#!/usr/bin/env python
"""
Migrate trained models from experiment directories to common repository.

This script:
1. Finds all model files in outputs/experiments/exp_*/model/best_model_*.pth
2. Extracts configuration from experiment_summary.json
3. Computes model config hash
4. Generates proper filenames
5. Copies models to outputs/models/common/{task}/
6. Creates metadata files
7. Updates registry
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import only what we need to avoid dependency issues
import hashlib
import json
from datetime import datetime


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, path: Path):
    """Save JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def generate_timestamp() -> str:
    """Generate timestamp string in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def compute_model_config_hash(config: Dict) -> str:
    """
    Compute deterministic hash of model configuration.

    Parameters:
    -----------
    config : dict
        Configuration dictionary

    Returns:
    --------
    hash_str : str
        SHA256 hash as hex string
    """
    # Extract relevant config sections
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    loss_config = config.get("loss", {})

    # Normalize config for hashing
    hash_dict = {
        "model": {
            "input_method": model_config.get("input_method"),
            "input_dim": model_config.get("input_dim"),
            "hidden_dims": model_config.get("hidden_dims"),
            "output_dim": model_config.get("output_dim"),
            "activation": model_config.get("activation"),
            "use_residual": model_config.get("use_residual"),
            "dropout": model_config.get("dropout"),
        },
        "training": {
            "learning_rate": training_config.get("learning_rate"),
            "weight_decay": training_config.get("weight_decay"),
            "batch_size": training_config.get("batch_size"),
            "max_training_angle_degrees": training_config.get("max_training_angle_degrees"),
            "lambda_angle": training_config.get("lambda_angle"),
        },
        "loss": {
            "lambda_data": loss_config.get("lambda_data"),
            "lambda_physics": loss_config.get("lambda_physics"),
            "lambda_ic": loss_config.get("lambda_ic"),
            "use_fixed_lambda": loss_config.get("use_fixed_lambda"),
            "scale_to_norm": loss_config.get("scale_to_norm"),
        },
    }

    # Convert to JSON string and hash
    import json

    config_str = json.dumps(hash_dict, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()


# Define paths directly to avoid heavy imports
COMMON_MODELS_DIR = PROJECT_ROOT / "outputs" / "models" / "common"
MODEL_REGISTRY_PATH = COMMON_MODELS_DIR / "registry.json"


def _ensure_common_directories():
    """Ensure common repository directories exist."""
    COMMON_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (COMMON_MODELS_DIR / "trajectory").mkdir(parents=True, exist_ok=True)
    (COMMON_MODELS_DIR / "parameter_estimation").mkdir(parents=True, exist_ok=True)


def _load_registry(registry_path: Path) -> Dict:
    """Load registry JSON file."""
    if not registry_path.exists():
        return {}
    try:
        return load_json(registry_path)
    except Exception:
        return {}


def _save_registry(registry: Dict, registry_path: Path):
    """Save registry JSON file."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(registry, registry_path)


def get_common_model_path(task: str, config_hash: str, timestamp: Optional[str] = None) -> Path:
    """Get path to common model file."""
    _ensure_common_directories()

    if timestamp is None:
        timestamp = generate_timestamp()

    # Short hash for filename (first 8 chars)
    hash_short = config_hash[:8]

    filename = f"model_{hash_short}_{timestamp}.pth"
    return COMMON_MODELS_DIR / task / filename


def find_existing_model_files(search_dirs: List[Path]) -> List[Path]:
    """
    Find all existing model files in old locations.

    Parameters:
    -----------
    search_dirs : list of Path
        Directories to search

    Returns:
    --------
    model_files : list of Path
        List of model file paths
    """
    model_files = []

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Search for best_model files only (skip checkpoints)
        pattern = "best_model_*.pth"

        # Search in model subdirectories
        for model_file in search_dir.rglob(pattern):
            # Only include files in model/ subdirectories
            if "model" in model_file.parts:
                model_files.append(model_file)

    # Remove duplicates and sort
    model_files = sorted(set(model_files))
    return model_files


def extract_experiment_config(exp_dir: Path) -> Optional[Dict]:
    """
    Extract configuration from experiment directory.

    Parameters:
    -----------
    exp_dir : Path
        Experiment directory path

    Returns:
    --------
    config : dict or None
        Configuration dictionary
    """
    # Try experiment_summary.json first
    summary_path = exp_dir / "experiment_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        return summary.get("config", summary)

    # Try reproducibility.json as fallback
    repro_path = exp_dir / "reproducibility.json"
    if repro_path.exists():
        repro = load_json(repro_path)
        return repro.get("config", repro)

    return None


def extract_training_metrics(exp_dir: Path) -> Dict:
    """
    Extract training metrics from experiment directory.

    Parameters:
    -----------
    exp_dir : Path
        Experiment directory path

    Returns:
    --------
    metrics : dict
        Training metrics
    """
    metrics = {}

    # Try to find training history
    model_dir = exp_dir / "model"
    if model_dir.exists():
        for history_file in model_dir.glob("training_history_*.json"):
            history = load_json(history_file)
            # Extract final metrics
            if "final_metrics" in history:
                metrics.update(history["final_metrics"])
            elif "metrics" in history:
                metrics.update(history["metrics"])
            # Extract best validation loss
            if "val_losses" in history and history["val_losses"]:
                metrics["best_val_loss"] = min(history["val_losses"])
            if "train_losses" in history and history["train_losses"]:
                metrics["final_train_loss"] = history["train_losses"][-1]
            break

    # Try experiment_summary.json
    summary_path = exp_dir / "experiment_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        if "results" in summary and "metrics" in summary["results"]:
            metrics.update(summary["results"]["metrics"])

    return metrics


def find_data_fingerprint_from_experiment(exp_dir: Path, config: Dict) -> Optional[str]:
    """
    Try to find data fingerprint from experiment.

    Parameters:
    -----------
    exp_dir : Path
        Experiment directory
    config : dict
        Configuration dictionary

    Returns:
    --------
    fingerprint : str or None
        Data fingerprint if found
    """
    # Try to find data file in experiment directory
    data_dir = exp_dir / "data"
    if data_dir.exists():
        # Look for trajectory data files
        for data_file in data_dir.rglob("*.csv"):
            # Check if there's metadata
            metadata_path = data_file.with_suffix(".json").with_name(
                data_file.stem + "_metadata.json"
            )
            if metadata_path.exists():
                try:
                    metadata = load_json(metadata_path)
                    fingerprint = metadata.get("data_fingerprint")
                    if fingerprint:
                        return fingerprint
                except Exception:
                    pass

    # Try to find in common repository using config
    # Generate fingerprint from config (simplified version)
    try:
        gen_config = config.get("data", {}).get("generation", {})
        if gen_config:
            # Simple hash of key parameters
            key_params = {
                "n_samples": gen_config.get("n_samples"),
                "sampling_strategy": gen_config.get("sampling_strategy"),
                "parameter_ranges": gen_config.get("parameter_ranges"),
            }
            config_str = json.dumps(key_params, sort_keys=True, default=str)
            data_fingerprint = hashlib.sha256(config_str.encode()).hexdigest()

            # Try to find matching data in common repository
            common_data_dir = PROJECT_ROOT / "data" / "common"
            if common_data_dir.exists():
                for data_file in common_data_dir.glob("*.csv"):
                    metadata_path = data_file.with_suffix(".json").with_name(
                        data_file.stem + "_metadata.json"
                    )
                    if metadata_path.exists():
                        try:
                            metadata = load_json(metadata_path)
                            if metadata.get("data_fingerprint") == data_fingerprint:
                                return data_fingerprint
                        except Exception:
                            pass
    except Exception:
        pass

    return None


def migrate_model_file(
    model_path: Path, dry_run: bool = False, overwrite: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Migrate a single model file to common repository.

    Parameters:
    -----------
    model_path : Path
        Path to model file
    dry_run : bool
        If True, don't actually migrate
    overwrite : bool
        If True, overwrite existing files

    Returns:
    --------
    success : bool
        True if migration successful
    error : str or None
        Error message if failed
    """
    try:
        # Find experiment directory
        exp_dir = None
        for parent in model_path.parents:
            if parent.name.startswith("exp_") and len(parent.name) > 4:
                exp_dir = parent
                break

        if not exp_dir:
            return False, f"Could not find experiment directory for {model_path}"

        # Extract configuration
        config = extract_experiment_config(exp_dir)
        if not config:
            return False, f"Could not find experiment config for {exp_dir}"

        # Determine task
        task = config.get("data", {}).get("task", "trajectory")

        # Compute config hash
        config_hash = compute_model_config_hash(config)

        # Check if model already exists
        existing_path = get_common_model_path(task, config_hash)
        if existing_path.exists() and not overwrite:
            return True, f"Model already exists: {existing_path.name} (skipped)"

        # Extract metrics
        metrics = extract_training_metrics(exp_dir)

        # Find data fingerprint
        data_fingerprint = find_data_fingerprint_from_experiment(exp_dir, config)

        # Generate timestamp
        timestamp = generate_timestamp()

        # Generate destination path
        dest_path = get_common_model_path(task, config_hash, timestamp)

        if dry_run:
            print(f"[DRY RUN] Would migrate: {model_path.name}")
            print(f"  -> {dest_path.name}")
            print(f"  Config hash: {config_hash[:16]}...")
            print(f"  Task: {task}")
            if data_fingerprint:
                print(f"  Data fingerprint: {data_fingerprint[:16]}...")
            return True, None

        # Ensure directories exist
        _ensure_common_directories()

        # Copy model file (no need to load torch for this)
        shutil.copy2(model_path, dest_path)
        print(f"✓ Migrated: {model_path.name} -> {dest_path.name}")

        # Build metadata
        metadata = {
            "model_config_hash": config_hash,
            "task": task,
            "filename": dest_path.name,
            "original_path": str(model_path),
            "migration_timestamp": datetime.now().isoformat(),
            "training_config": {
                "model": config.get("model", {}),
                "training": config.get("training", {}),
                "loss": config.get("loss", {}),
            },
            "metrics": metrics,
            "data_fingerprint": data_fingerprint,
            "reproducibility": {
                "git_commit": config.get("reproducibility", {}).get("git_commit"),
                "git_branch": config.get("reproducibility", {}).get("git_branch"),
                "package_versions": config.get("reproducibility", {}).get("package_versions"),
                "python_version": config.get("reproducibility", {}).get("python_version"),
                "random_seed": config.get("reproducibility", {}).get("random_seed"),
                "timestamp": timestamp,
                "note": "Migrated from legacy location - some reproducibility info may be incomplete",
            },
        }

        # Save metadata
        metadata_path = dest_path.with_suffix(".json").with_name(dest_path.stem + "_metadata.json")
        save_json(metadata, metadata_path)

        # Update registry
        registry = _load_registry(MODEL_REGISTRY_PATH)
        if "config_hashes" not in registry:
            registry["config_hashes"] = {}

        rel_path = dest_path.relative_to(COMMON_MODELS_DIR)
        registry["config_hashes"][config_hash] = {
            "path": str(rel_path),
            "task": task,
            "data_fingerprint": data_fingerprint,
            "timestamp": timestamp,
            "metrics": metrics,
        }
        _save_registry(registry, MODEL_REGISTRY_PATH)

        return True, None

    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Migrate trained models from experiment directories to common repository"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually migrating",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing models in common repository",
    )
    parser.add_argument(
        "--search-dirs",
        nargs="+",
        default=[PROJECT_ROOT / "outputs" / "experiments"],
        help="Directories to search for model files",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MIGRATE MODELS TO COMMON REPOSITORY")
    print("=" * 70)
    print(f"Search directories: {args.search_dirs}")
    print(f"Dry run: {args.dry_run}")
    print(f"Overwrite: {args.overwrite}")
    print()

    # Find all model files
    search_dirs = [Path(d) for d in args.search_dirs]
    model_files = find_existing_model_files(search_dirs)

    if not model_files:
        print("No model files found.")
        return

    print(f"Found {len(model_files)} model file(s)")
    print()

    # Migrate each file
    migrated_count = 0
    skipped_count = 0
    error_count = 0
    errors = []

    for model_path in model_files:
        success, error = migrate_model_file(model_path, args.dry_run, args.overwrite)

        if success:
            if error and "already exists" in error:
                skipped_count += 1
                print(f"[SKIP] {error}")
            else:
                migrated_count += 1
        else:
            error_count += 1
            errors.append((model_path.name, error))
            print(f"[ERROR] {model_path.name}: {error}")

    print()
    print("=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)
    print(f"Total files found: {len(model_files)}")
    print(f"Successfully migrated: {migrated_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Errors: {error_count}")

    if errors:
        print()
        print("Errors:")
        for filename, error in errors:
            print(f"  {filename}: {error}")

    if args.dry_run:
        print()
        print("[NOTE] This was a dry run. Use without --dry-run to actually migrate models.")


if __name__ == "__main__":
    main()
