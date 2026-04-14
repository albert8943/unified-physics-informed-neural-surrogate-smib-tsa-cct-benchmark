"""
Fingerprinting and Hashing Utilities.

This module provides functions for computing deterministic fingerprints/hashes
from configurations for data and model versioning and matching.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def normalize_config_for_hashing(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize configuration by removing non-deterministic fields.

    Parameters:
    -----------
    config : dict
        Configuration dictionary

    Returns:
    --------
    normalized_config : dict
        Normalized configuration with non-deterministic fields removed
    """
    # Create a deep copy to avoid modifying original
    normalized = json.loads(json.dumps(config))

    # Fields to exclude (non-deterministic or not relevant for matching)
    exclude_fields = [
        "output_dir",
        "timestamp",
        "experiment_id",
        "device",  # Device doesn't affect data/model identity
        "notes",
        "description",
    ]

    def remove_fields(obj, path=""):
        """Recursively remove excluded fields."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                full_path = f"{path}.{key}" if path else key
                if key not in exclude_fields:
                    result[key] = remove_fields(value, full_path)
            return result
        elif isinstance(obj, list):
            return [remove_fields(item, path) for item in obj]
        else:
            return obj

    return remove_fields(normalized)


def compute_data_fingerprint(config: Dict[str, Any]) -> str:
    """
    Compute deterministic hash from data generation parameters.

    Includes: parameter_ranges, case_file, simulation_time, time_step,
             fault parameters, sampling_strategy, random_seed
    Excludes: output_dir, timestamps (non-deterministic)

    Parameters:
    -----------
    config : dict
        Configuration dictionary with data generation parameters

    Returns:
    --------
    fingerprint : str
        SHA256 hash of normalized config (hex string)
    """
    # Extract data generation config
    data_config = config.get("data", {})
    gen_config = data_config.get("generation", {})

    # Build normalized config for hashing
    hash_config = {
        "task": data_config.get("task", "trajectory"),
        "generation": gen_config,
        "reproducibility": {
            "random_seed": config.get("reproducibility", {}).get("random_seed"),
        },
    }

    # Normalize to remove non-deterministic fields
    normalized = normalize_config_for_hashing(hash_config)

    # Convert to JSON string with sorted keys for consistency
    config_str = json.dumps(normalized, sort_keys=True, default=str)

    # Compute SHA256 hash
    hash_obj = hashlib.sha256(config_str.encode("utf-8"))
    return hash_obj.hexdigest()


def compute_model_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute deterministic hash from training configuration.

    Includes: model architecture, training hyperparameters, loss config,
             optimizer settings, learning rate schedule, etc.

    Parameters:
    -----------
    config : dict
        Configuration dictionary with training parameters

    Returns:
    --------
    config_hash : str
        SHA256 hash of normalized config (hex string)
    """
    # Extract relevant config sections
    hash_config = {
        "model": config.get("model", {}),
        "training": config.get("training", {}),
        "loss": config.get("loss", {}),
        "reproducibility": {
            "random_seed": config.get("reproducibility", {}).get("random_seed"),
        },
    }

    # Normalize to remove non-deterministic fields
    normalized = normalize_config_for_hashing(hash_config)

    # Convert to JSON string with sorted keys for consistency
    config_str = json.dumps(normalized, sort_keys=True, default=str)

    # Compute SHA256 hash
    hash_obj = hashlib.sha256(config_str.encode("utf-8"))
    return hash_obj.hexdigest()


def compute_file_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute file checksum for integrity verification.

    Parameters:
    -----------
    file_path : Path
        Path to file
    algorithm : str
        Hash algorithm to use: 'md5' or 'sha256' (default: 'sha256')

    Returns:
    --------
    checksum : str
        Hex string of file checksum
    """
    if algorithm == "sha256":
        hash_obj = hashlib.sha256()
    elif algorithm == "md5":
        hash_obj = hashlib.md5()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'md5' or 'sha256'")

    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def verify_file_checksum(
    file_path: Path, expected_checksum: str, algorithm: str = "sha256"
) -> bool:
    """
    Verify file checksum matches expected value.

    Parameters:
    -----------
    file_path : Path
        Path to file
    expected_checksum : str
        Expected checksum value
    algorithm : str
        Hash algorithm to use: 'md5' or 'sha256' (default: 'sha256')

    Returns:
    --------
    is_valid : bool
        True if checksum matches
    """
    if not file_path.exists():
        return False

    actual_checksum = compute_file_checksum(file_path, algorithm)
    return actual_checksum.lower() == expected_checksum.lower()
