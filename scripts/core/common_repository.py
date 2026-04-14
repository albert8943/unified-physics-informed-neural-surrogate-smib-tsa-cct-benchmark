"""
Common Repository Utilities.

This module provides functions for managing centralized data and model repositories
with fingerprinting, registry management, and provenance tracking.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.fingerprinting import (
    compute_data_fingerprint,
    compute_file_checksum,
    compute_model_config_hash,
    verify_file_checksum,
)
from scripts.core.utils import (
    generate_timestamp,
    get_git_branch,
    get_git_commit,
    get_package_versions,
    load_json,
    save_json,
)

# Common repository paths
COMMON_DATA_DIR = PROJECT_ROOT / "data" / "common"
COMMON_MODELS_DIR = PROJECT_ROOT / "outputs" / "models" / "common"
DATA_REGISTRY_PATH = COMMON_DATA_DIR / "registry.json"
MODEL_REGISTRY_PATH = COMMON_MODELS_DIR / "registry.json"


def _ensure_common_directories():
    """Ensure common repository directories exist."""
    COMMON_DATA_DIR.mkdir(parents=True, exist_ok=True)
    COMMON_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Create task subdirectories for models
    (COMMON_MODELS_DIR / "trajectory").mkdir(parents=True, exist_ok=True)
    (COMMON_MODELS_DIR / "parameter_estimation").mkdir(parents=True, exist_ok=True)


def _load_registry(registry_path: Path) -> Dict[str, Any]:
    """Load registry JSON file."""
    if not registry_path.exists():
        return {}
    try:
        return load_json(registry_path)
    except Exception:
        return {}


def _save_registry(registry: Dict[str, Any], registry_path: Path):
    """Save registry JSON file."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(registry, registry_path)


def _extract_key_params_for_filename(config: Dict[str, Any], task: str) -> str:
    """
    Extract key parameters from config for human-readable filename.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    task : str
        Task type: 'trajectory' or 'parameter_estimation'

    Returns:
    --------
    params_str : str
        String representation of key parameters (formatted to 2 decimal places)
    """
    gen_config = config.get("data", {}).get("generation", {})
    param_ranges = gen_config.get("parameter_ranges", {})

    params = []

    def format_param(name: str, min_val: float, max_val: float) -> str:
        """Format parameter range with reasonable precision."""
        # Round to 2 decimal places for readability, then clean up
        min_str = f"{min_val:.2f}".rstrip("0").rstrip(".")
        max_str = f"{max_val:.2f}".rstrip("0").rstrip(".")
        # Ensure we don't have empty strings (use "0" if all digits removed)
        if not min_str or min_str == ".":
            min_str = "0"
        if not max_str or max_str == ".":
            max_str = "0"
        # Remove any leading/trailing dots that might cause issues
        min_str = min_str.lstrip(".").rstrip(".")
        max_str = max_str.lstrip(".").rstrip(".")
        return f"{name}{min_str}-{max_str}"

    if task == "trajectory":
        # Extract H and D ranges
        H_val = param_ranges.get("H", (2.0, 10.0, 5))
        if isinstance(H_val, (list, tuple)) and len(H_val) >= 2:
            H_min, H_max = float(H_val[0]), float(H_val[1])
            params.append(format_param("H", H_min, H_max))

        D_val = param_ranges.get("D", (0.5, 3.0, 5))
        if isinstance(D_val, (list, tuple)) and len(D_val) >= 2:
            D_min, D_max = float(D_val[0]), float(D_val[1])
            params.append(format_param("D", D_min, D_max))

        # Check for load range (preferred) or Pm range (backward compatibility)
        load_val = param_ranges.get("load", None)
        if load_val is not None and isinstance(load_val, (list, tuple)) and len(load_val) >= 2:
            load_min, load_max = float(load_val[0]), float(load_val[1])
            params.append(format_param("Pload", load_min, load_max))
        else:
            # Fallback to Pm for backward compatibility
            Pm_val = param_ranges.get("Pm", None)
            if Pm_val is not None and isinstance(Pm_val, (list, tuple)) and len(Pm_val) >= 2:
                Pm_min, Pm_max = float(Pm_val[0]), float(Pm_val[1])
                params.append(format_param("Pm", Pm_min, Pm_max))

    elif task == "parameter_estimation":
        # Check for load range (preferred) or Pm range (backward compatibility)
        load_val = param_ranges.get("load", None)
        if load_val is not None:
            if isinstance(load_val, (list, tuple)) and len(load_val) >= 2:
                load_min, load_max = float(load_val[0]), float(load_val[1])
                params.append(format_param("Pload", load_min, load_max))
            elif isinstance(load_val, list):
                # List of values
                load_min, load_max = float(min(load_val)), float(max(load_val))
                params.append(format_param("Pload", load_min, load_max))
        else:
            # Fallback to Pm for backward compatibility
            Pm_val = param_ranges.get("Pm", None)
            if Pm_val is not None:
                if isinstance(Pm_val, (list, tuple)) and len(Pm_val) >= 2:
                    Pm_min, Pm_max = float(Pm_val[0]), float(Pm_val[1])
                    params.append(format_param("Pm", Pm_min, Pm_max))
                elif isinstance(Pm_val, list):
                    # List of values
                    Pm_min, Pm_max = float(min(Pm_val)), float(max(Pm_val))
                    params.append(format_param("Pm", Pm_min, Pm_max))

        # Also include H and D if specified
        H_val = param_ranges.get("H", None)
        if H_val is not None and isinstance(H_val, (list, tuple)) and len(H_val) >= 2:
            H_min, H_max = float(H_val[0]), float(H_val[1])
            params.append(format_param("H", H_min, H_max))

        D_val = param_ranges.get("D", None)
        if D_val is not None and isinstance(D_val, (list, tuple)) and len(D_val) >= 2:
            D_min, D_max = float(D_val[0]), float(D_val[1])
            params.append(format_param("D", D_min, D_max))

    return "_".join(params) if params else "default"


def generate_data_filename(
    task: str,
    config: Dict[str, Any],
    fingerprint: str,
    n_samples: Optional[int] = None,
    timestamp: Optional[str] = None,
    data_type: str = "full",
) -> str:
    """
    Generate descriptive filename for data file.

    Format: {data_type}_{task}_data_{n_samples}_{key_params}_{fingerprint[:8]}_{timestamp}.csv

    Parameters:
    -----------
    task : str
        Task type: 'trajectory' or 'parameter_estimation'
    config : dict
        Configuration dictionary
    fingerprint : str
        Data fingerprint (full hash)
    n_samples : int, optional
        Number of samples (will be extracted from data if not provided)
    timestamp : str, optional
        Timestamp string (YYYYMMDD_HHMMSS). If None, uses current time.
    data_type : str, optional
        Data type prefix: 'full' for full trajectory data, 'summary' for summary statistics.
        Default is 'full'. Summary statistics should not be saved to common repository.

    Returns:
    --------
    filename : str
        Generated filename
    """
    if timestamp is None:
        timestamp = generate_timestamp()

    # Extract key parameters for filename
    key_params = _extract_key_params_for_filename(config, task)

    # Get n_samples if not provided (try to estimate from config)
    if n_samples is None:
        gen_config = config.get("data", {}).get("generation", {})
        param_ranges = gen_config.get("parameter_ranges", {})
        sampling_strategy = gen_config.get("sampling_strategy", "full_factorial")

        if sampling_strategy == "full_factorial":
            # Estimate from parameter ranges
            n_samples = 1
            for key, val in param_ranges.items():
                if isinstance(val, (list, tuple)) and len(val) >= 3:
                    n_samples *= val[2]  # num_points
                elif isinstance(val, list):
                    n_samples *= len(val)
        else:
            n_samples = gen_config.get("n_samples", 0)

    # Short fingerprint for filename (first 8 chars)
    fp_short = fingerprint[:8]

    # Build filename with data type prefix
    parts = [data_type, task, "data", str(n_samples)]
    if key_params:
        parts.append(key_params)
    parts.extend([fp_short, timestamp])

    return "_".join(parts) + ".csv"


def get_common_data_path(
    task: str,
    config: Dict[str, Any],
    fingerprint: Optional[str] = None,
    timestamp: Optional[str] = None,
    n_samples: Optional[int] = None,
) -> Path:
    """
    Get path to common data file.

    Parameters:
    -----------
    task : str
        Task type
    config : dict
        Configuration dictionary
    fingerprint : str, optional
        Data fingerprint. If None, will be computed.
    timestamp : str, optional
        Timestamp. If None, uses current time.
    n_samples : int, optional
        Number of samples for filename

    Returns:
    --------
    data_path : Path
        Path to data file
    """
    _ensure_common_directories()

    if fingerprint is None:
        fingerprint = compute_data_fingerprint(config)

    filename = generate_data_filename(task, config, fingerprint, n_samples, timestamp)
    return COMMON_DATA_DIR / filename


def find_data_by_fingerprint(fingerprint: str, task: Optional[str] = None) -> Optional[Path]:
    """
    Find data file by fingerprint.

    Parameters:
    -----------
    fingerprint : str
        Data fingerprint
    task : str, optional
        Task type filter

    Returns:
    --------
    data_path : Path or None
        Path to data file if found
    """
    registry = _load_registry(DATA_REGISTRY_PATH)
    fingerprints = registry.get("fingerprints", {})

    if fingerprint in fingerprints:
        entry = fingerprints[fingerprint]
        if task is None or entry.get("task") == task:
            filename = entry.get("filename")
            if filename:
                data_path = COMMON_DATA_DIR / filename
                if data_path.exists():
                    return data_path

    return None


def find_data_by_params(task: str, n_samples: Optional[int] = None, **params) -> List[Path]:
    """
    Find data files by parameter matching.

    Parameters:
    -----------
    task : str
        Task type
    n_samples : int, optional
        Number of samples filter
    **params : dict
        Additional parameter filters (e.g., H_range, D_range)

    Returns:
    --------
    data_paths : list
        List of matching data file paths
    """
    registry = _load_registry(DATA_REGISTRY_PATH)
    fingerprints = registry.get("fingerprints", {})

    matches = []
    for fingerprint, entry in fingerprints.items():
        if entry.get("task") != task:
            continue

        if n_samples is not None and entry.get("n_samples") != n_samples:
            continue

        # Check key_params if provided
        key_params = entry.get("key_params", {})
        match = True
        for key, value in params.items():
            if key in key_params:
                if key_params[key] != value:
                    match = False
                    break

        if match:
            filename = entry.get("filename")
            if filename:
                data_path = COMMON_DATA_DIR / filename
                if data_path.exists():
                    matches.append(data_path)

    return matches


def find_latest_data(task: str, **filters) -> Optional[Path]:
    """
    Find latest data file by timestamp with optional filters.

    Parameters:
    -----------
    task : str
        Task type
    **filters : dict
        Optional filters (n_samples, key_params, etc.)

    Returns:
    --------
    data_path : Path or None
        Latest matching data file path
    """
    matches = find_data_by_params(task, **filters)

    if not matches:
        return None

    # Sort by modification time (newest first)
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def save_data_to_common(
    data: pd.DataFrame,
    task: str,
    config: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    force_regenerate: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Save data to common repository with fingerprinting and metadata.

    Parameters:
    -----------
    data : pd.DataFrame
        Data to save
    task : str
        Task type
    config : dict
        Configuration dictionary
    metadata : dict, optional
        Additional metadata to include
    force_regenerate : bool, optional
        If True, overwrite existing data even if fingerprint matches (default: False)

    Returns:
    --------
    data_path : Path
        Path to saved data file
    full_metadata : dict
        Complete metadata dictionary
    """
    _ensure_common_directories()

    # Compute fingerprint
    fingerprint = compute_data_fingerprint(config)

    # Check if data already exists
    existing_path = find_data_by_fingerprint(fingerprint, task)
    if existing_path and existing_path.exists():
        if force_regenerate:
            print(f"⚠️  Data with same fingerprint exists: {existing_path.name}")
            print("  Force regenerate enabled - overwriting existing data...")
            # Delete existing data file and metadata
            try:
                if existing_path.exists():
                    existing_path.unlink()
                    print(f"  [OK] Deleted old data file: {existing_path.name}")
                metadata_path = existing_path.with_suffix(".json").with_name(
                    existing_path.stem + "_metadata.json"
                )
                if metadata_path.exists():
                    metadata_path.unlink()
                    print(f"  [OK] Deleted old metadata file")
                validation_path = existing_path.parent / f"{existing_path.stem}_validation.json"
                if validation_path.exists():
                    validation_path.unlink()
                    print(f"  [OK] Deleted old validation file")
            except Exception as e:
                print(f"  [WARNING] Error deleting old files: {e}")
                print("  Continuing with regeneration (may overwrite existing files)...")
        else:
            print(f"✓ Data with same fingerprint already exists: {existing_path.name}")
            print("  Reusing existing data (use --force-regenerate to override)")
            return existing_path, _load_metadata(existing_path)

    # Validate data type - reject summary statistics
    is_summary_stats = (
        all(c in data.columns for c in ["H", "D", "Pm", "CCT"])
        and "time" not in data.columns
        and "param_H" not in data.columns
    )

    if is_summary_stats:
        raise ValueError(
            "Summary statistics files are not allowed in common repository. "
            "Only full trajectory data (with 'time' and 'param_*' columns) can be saved. "
            "Summary statistics should be stored separately or excluded."
        )

    # Generate filename - n_samples should be the number of unique parameter
    # combinations (H, D, Pload/Pm)
    # NOT the number of scenarios/trajectories (which is n_samples × n_samples_per_combination)
    # Full trajectory data format: count unique parameter combinations
    # Check for param_load first (preferred), then fallback to param_Pm (backward compatibility)
    if all(c in data.columns for c in ["param_H", "param_D", "param_load"]):
        param_combos = data[["param_H", "param_D", "param_load"]].drop_duplicates()
        n_samples = len(param_combos)
    elif all(c in data.columns for c in ["param_H", "param_D", "param_Pm"]):
        param_combos = data[["param_H", "param_D", "param_Pm"]].drop_duplicates()
        n_samples = len(param_combos)
    elif "scenario_id" in data.columns:
        # Fallback: if param columns not available, try to infer from scenarios
        # This is less accurate but better than nothing
        n_scenarios = data["scenario_id"].nunique()
        # Assume typical n_samples_per_combination = 5, but this may not be accurate
        # Round to nearest integer
        n_samples = round(n_scenarios / 5)
        if n_samples == 0:
            n_samples = 1  # At least 1
    else:
        # Last resort: use total rows (for non-trajectory data)
        n_samples = len(data)

    timestamp = generate_timestamp()
    filename = generate_data_filename(
        task, config, fingerprint, n_samples, timestamp, data_type="full"
    )
    data_path = COMMON_DATA_DIR / filename

    # Save data
    data.to_csv(data_path, index=False)
    print(f"✓ Data saved to: {data_path.name}")

    # Compute file checksum
    file_checksum = compute_file_checksum(data_path)

    # Build metadata
    from scripts.core.utils import get_git_branch, get_git_commit, get_package_versions

    full_metadata = {
        "data_fingerprint": fingerprint,
        "file_checksum": file_checksum,
        "task": task,
        "n_samples": n_samples,
        "filename": filename,
        "generation_config": config.get("data", {}).get("generation", {}),
        "statistics": {
            "total_rows": len(data),
            "unique_scenarios": (
                data["scenario_id"].nunique() if "scenario_id" in data.columns else None
            ),
        },
        "reproducibility": {
            "git_commit": get_git_commit(),
            "git_branch": get_git_branch(),
            "package_versions": get_package_versions(),
            "python_version": sys.version,
            "random_seed": config.get("reproducibility", {}).get("random_seed"),
            "timestamp": datetime.now().isoformat(),
        },
        "models_trained": [],  # Will be updated when models are trained on this data
    }

    # Add any additional metadata
    if metadata:
        full_metadata.update(metadata)

    # Save metadata
    metadata_path = data_path.with_suffix(".json").with_name(data_path.stem + "_metadata.json")
    save_json(full_metadata, metadata_path)

    # Update registry
    _update_data_registry(fingerprint, filename, task, n_samples, config, timestamp)

    return data_path, full_metadata


def _update_data_registry(
    fingerprint: str, filename: str, task: str, n_samples: int, config: Dict, timestamp: str
):
    """Update data registry with new entry."""
    registry = _load_registry(DATA_REGISTRY_PATH)

    if "fingerprints" not in registry:
        registry["fingerprints"] = {}

    # Extract key parameters (rounded for cleaner registry entries)
    key_params = {}
    param_ranges = config.get("data", {}).get("generation", {}).get("parameter_ranges", {})
    # Process H and D
    for key in ["H", "D"]:
        if key in param_ranges:
            val = param_ranges[key]
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                # Round to 2 decimal places for cleaner registry
                key_params[f"{key}_range"] = [
                    round(float(val[0]), 2),
                    round(float(val[1]), 2),
                ]
    # Check for load first (preferred), then Pm (backward compatibility)
    if "load" in param_ranges:
        val = param_ranges["load"]
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            key_params["Pload_range"] = [
                round(float(val[0]), 2),
                round(float(val[1]), 2),
            ]
    elif "Pm" in param_ranges:
        val = param_ranges["Pm"]
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            key_params["Pm_range"] = [
                round(float(val[0]), 2),
                round(float(val[1]), 2),
            ]

    registry["fingerprints"][fingerprint] = {
        "filename": filename,
        "task": task,
        "n_samples": n_samples,
        "key_params": key_params,
        "timestamp": timestamp,
        "config_summary": {
            "case_file": config.get("data", {}).get("generation", {}).get("case_file"),
            "sampling_strategy": config.get("data", {})
            .get("generation", {})
            .get("sampling_strategy"),
        },
    }

    _save_registry(registry, DATA_REGISTRY_PATH)


def _load_metadata(data_path: Path) -> Dict[str, Any]:
    """Load metadata for a data file."""
    metadata_path = data_path.with_suffix(".json").with_name(data_path.stem + "_metadata.json")
    if metadata_path.exists():
        return load_json(metadata_path)
    return {}


def get_common_model_path(task: str, config_hash: str, timestamp: Optional[str] = None) -> Path:
    """
    Get path to common model file.

    Parameters:
    -----------
    task : str
        Task type: 'trajectory' or 'parameter_estimation'
    config_hash : str
        Model configuration hash
    timestamp : str, optional
        Timestamp. If None, uses current time.

    Returns:
    --------
    model_path : Path
        Path to model file
    """
    _ensure_common_directories()

    if timestamp is None:
        timestamp = generate_timestamp()

    # Short hash for filename (first 8 chars)
    hash_short = config_hash[:8]

    filename = f"model_{hash_short}_{timestamp}.pth"
    return COMMON_MODELS_DIR / task / filename


def find_model_by_config_hash(config_hash: str, task: Optional[str] = None) -> Optional[Path]:
    """
    Find model file by configuration hash.

    Parameters:
    -----------
    config_hash : str
        Model configuration hash
    task : str, optional
        Task type filter

    Returns:
    --------
    model_path : Path or None
        Path to model file if found
    """
    registry = _load_registry(MODEL_REGISTRY_PATH)
    config_hashes = registry.get("config_hashes", {})

    if config_hash in config_hashes:
        entry = config_hashes[config_hash]
        if task is None or entry.get("task") == task:
            path_str = entry.get("path")
            if path_str:
                model_path = COMMON_MODELS_DIR / path_str
                if model_path.exists():
                    return model_path

    return None


def find_models_trained_on_data(data_fingerprint: str) -> List[Path]:
    """
    Find models trained on a specific dataset (provenance query).

    Parameters:
    -----------
    data_fingerprint : str
        Data fingerprint

    Returns:
    --------
    model_paths : list
        List of model file paths
    """
    registry = _load_registry(MODEL_REGISTRY_PATH)
    config_hashes = registry.get("config_hashes", {})

    matches = []
    for config_hash, entry in config_hashes.items():
        if entry.get("data_fingerprint") == data_fingerprint:
            path_str = entry.get("path")
            if path_str:
                model_path = COMMON_MODELS_DIR / path_str
                if model_path.exists():
                    matches.append(model_path)

    return matches


def save_model_to_common(
    model_state: Dict[str, Any],
    task: str,
    config: Dict[str, Any],
    data_fingerprint: str,
    data_path: Path,
    metrics: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Save model to common repository with provenance tracking.

    Parameters:
    -----------
    model_state : dict
        Model state dict (from model.state_dict())
    task : str
        Task type
    config : dict
        Configuration dictionary
    data_fingerprint : str
        Fingerprint of training data
    data_path : Path
        Path to training data file
    metrics : dict
        Training/evaluation metrics
    metadata : dict, optional
        Additional metadata

    Returns:
    --------
    model_path : Path
        Path to saved model file
    full_metadata : dict
        Complete metadata dictionary
    """
    _ensure_common_directories()

    # Compute config hash
    config_hash = compute_model_config_hash(config)

    # Check if model already exists
    existing_path = find_model_by_config_hash(config_hash, task)
    if existing_path and existing_path.exists():
        print(f"✓ Model with same config hash already exists: {existing_path.name}")
        print("  Reusing existing model (use --force-retrain to override)")
        return existing_path, _load_model_metadata(existing_path)

    # Generate path
    timestamp = generate_timestamp()
    model_path = get_common_model_path(task, config_hash, timestamp)

    # Save model
    import torch

    torch.save(model_state, model_path)
    print(f"✓ Model saved to: {model_path.name}")

    # Build metadata
    full_metadata = {
        "model_config_hash": config_hash,
        "data_fingerprint": data_fingerprint,
        "data_path": str(data_path),
        "task": task,
        "training_config": {
            "model": config.get("model", {}),
            "training": config.get("training", {}),
            "loss": config.get("loss", {}),
        },
        "metrics": metrics,
        "reproducibility": {
            "git_commit": get_git_commit(),
            "git_branch": get_git_branch(),
            "package_versions": get_package_versions(),
            "python_version": sys.version,
            "random_seeds": {
                "data_gen": config.get("reproducibility", {}).get("random_seed"),
                "split": config.get("reproducibility", {}).get("random_seed"),
                "training": config.get("reproducibility", {}).get("random_seed"),
            },
            "device": config.get("training", {}).get("device", "auto"),
            "timestamp": datetime.now().isoformat(),
        },
    }

    # Add any additional metadata
    if metadata:
        full_metadata.update(metadata)

    # Save metadata
    metadata_path = model_path.with_suffix(".json").with_name(model_path.stem + "_metadata.json")
    save_json(full_metadata, metadata_path)

    # Update registry
    _update_model_registry(config_hash, task, data_fingerprint, model_path, metrics, timestamp)

    # Update data metadata with model fingerprint (provenance link)
    _update_data_metadata_with_model(data_path, config_hash)

    return model_path, full_metadata


def _update_model_registry(
    config_hash: str,
    task: str,
    data_fingerprint: str,
    model_path: Path,
    metrics: Dict,
    timestamp: str,
):
    """Update model registry with new entry."""
    registry = _load_registry(MODEL_REGISTRY_PATH)

    if "config_hashes" not in registry:
        registry["config_hashes"] = {}

    # Relative path from common models dir
    rel_path = model_path.relative_to(COMMON_MODELS_DIR)

    registry["config_hashes"][config_hash] = {
        "path": str(rel_path),
        "task": task,
        "data_fingerprint": data_fingerprint,
        "timestamp": timestamp,
        "metrics": metrics,
    }

    _save_registry(registry, MODEL_REGISTRY_PATH)


def _update_data_metadata_with_model(data_path: Path, model_config_hash: str):
    """Update data metadata to include model fingerprint (provenance link)."""
    metadata_path = data_path.with_suffix(".json").with_name(data_path.stem + "_metadata.json")
    if not metadata_path.exists():
        return

    metadata = load_json(metadata_path)
    if "models_trained" not in metadata:
        metadata["models_trained"] = []

    if model_config_hash not in metadata["models_trained"]:
        metadata["models_trained"].append(model_config_hash)
        save_json(metadata, metadata_path)


def _load_model_metadata(model_path: Path) -> Dict[str, Any]:
    """Load metadata for a model file."""
    metadata_path = model_path.with_suffix(".json").with_name(model_path.stem + "_metadata.json")
    if metadata_path.exists():
        return load_json(metadata_path)
    return {}


def find_models_by_config(config: Dict[str, Any], task: Optional[str] = None) -> List[Path]:
    """
    Find models matching a configuration.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    task : str, optional
        Task type filter

    Returns:
    --------
    model_paths : list
        List of matching model paths
    """
    config_hash = compute_model_config_hash(config)
    model_path = find_model_by_config_hash(config_hash, task)
    if model_path:
        return [model_path]
    return []


def find_best_model(task: str, metric: str = "val_loss", minimize: bool = True) -> Optional[Path]:
    """
    Find best model by metric.

    Parameters:
    -----------
    task : str
        Task type
    metric : str
        Metric name (default: 'val_loss')
    minimize : bool
        True to minimize metric, False to maximize

    Returns:
    --------
    model_path : Path or None
        Path to best model
    """
    registry = _load_registry(MODEL_REGISTRY_PATH)
    config_hashes = registry.get("config_hashes", {})

    best_path = None
    best_value = None

    for config_hash, entry in config_hashes.items():
        if entry.get("task") != task:
            continue

        metrics = entry.get("metrics", {})
        if metric not in metrics:
            continue

        value = metrics[metric]
        if best_value is None:
            best_value = value
            path_str = entry.get("path")
            if path_str:
                best_path = COMMON_MODELS_DIR / path_str
        else:
            if minimize and value < best_value:
                best_value = value
                path_str = entry.get("path")
                if path_str:
                    best_path = COMMON_MODELS_DIR / path_str
            elif not minimize and value > best_value:
                best_value = value
                path_str = entry.get("path")
                if path_str:
                    best_path = COMMON_MODELS_DIR / path_str

    return best_path if best_path and best_path.exists() else None


def validate_data_integrity(
    data_path: Path, verify_checksum: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validate data file integrity.

    Parameters:
    -----------
    data_path : Path
        Path to data file
    verify_checksum : bool
        Whether to verify checksum

    Returns:
    --------
    is_valid : bool
        True if data is valid
    error_msg : str or None
        Error message if validation failed
    """
    if not data_path.exists():
        return False, "Data file does not exist"

    # Load metadata
    metadata = _load_metadata(data_path)
    if not metadata:
        return True, None  # No metadata to validate against

    # Verify checksum if requested
    if verify_checksum and "file_checksum" in metadata:
        expected_checksum = metadata["file_checksum"]
        if not verify_file_checksum(data_path, expected_checksum):
            return False, "File checksum mismatch - file may be corrupted or modified"

    return True, None
