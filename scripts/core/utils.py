"""
Core Utilities for Experiment Management.

This module provides utility functions for:
- Experiment ID generation
- Directory structure creation
- Config loading and validation
- Result saving
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Optional timezone by name (e.g. America/Los_Angeles). Python 3.9+ has zoneinfo.
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[misc, assignment]


def _now_for_experiment() -> datetime:
    """
    Current time used for experiment IDs and timestamps.

    Uses the operating system's local timezone by default so folder names match local wall clock.
    Set env PINN_TZ to an IANA timezone name (e.g. America/Los_Angeles) to force a specific zone
    (e.g. Colab VM in UTC but you want home-region names, or reproducible UTC via PINN_TZ=UTC).
    """
    tz_str = os.environ.get("PINN_TZ", "").strip()
    if tz_str and ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(tz_str))
        except Exception:
            pass
    return datetime.now().astimezone()


def generate_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Generate a timestamp string in format YYYYMMDD_HHMMSS.

    Uses local system time by default (or PINN_TZ if set).

    Parameters:
    -----------
    timestamp : datetime, optional
        Timestamp to use. If None, uses current time (local or PINN_TZ).

    Returns:
    --------
    timestamp_str : str
        Timestamp in format: YYYYMMDD_HHMMSS
    """
    if timestamp is None:
        timestamp = _now_for_experiment()

    return timestamp.strftime("%Y%m%d_%H%M%S")


def generate_experiment_id(timestamp: Optional[datetime] = None) -> str:
    """
    Generate a unique experiment ID based on timestamp.

    Uses local system time by default. Set PINN_TZ (e.g. UTC, America/Los_Angeles) to override.

    Parameters:
    -----------
    timestamp : datetime, optional
        Timestamp to use. If None, uses current time (local or PINN_TZ).

    Returns:
    --------
    experiment_id : str
        Experiment ID in format: exp_YYYYMMDD_HHMMSS
    """
    if timestamp is None:
        timestamp = _now_for_experiment()

    return timestamp.strftime("exp_%Y%m%d_%H%M%S")


def generate_timestamped_filename(
    base_name: str,
    extension: str,
    timestamp: Optional[datetime] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """
    Generate a timestamped filename following the Colab pattern.

    Format: [prefix_]base_name[_suffix]_YYYYMMDD_HHMMSS.extension

    Parameters:
    -----------
    base_name : str
        Base name for the file (e.g., "parameter_sweep_data", "best_model")
    extension : str
        File extension (e.g., "csv", "pth", "png", "json")
    timestamp : datetime, optional
        Timestamp to use. If None, uses current time.
    prefix : str, optional
        Optional prefix (e.g., "checkpoint_epoch_001")
    suffix : str, optional
        Optional suffix (e.g., "final")

    Returns:
    --------
    filename : str
        Timestamped filename

    Examples:
    ---------
    >>> generate_timestamped_filename("parameter_sweep_data", "csv")
    'parameter_sweep_data_20241205_143022.csv'

    >>> generate_timestamped_filename("best_model", "pth", prefix="checkpoint_epoch_001")
    'checkpoint_epoch_001_best_model_20241205_143022.pth'

    >>> generate_timestamped_filename("loss_curves", "png", suffix="final")
    'loss_curves_final_20241205_143022.png'
    """
    ts = generate_timestamp(timestamp)

    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(base_name)
    if suffix:
        parts.append(suffix)
    parts.append(ts)

    filename = "_".join(parts) + f".{extension}"
    return filename


def create_experiment_directory(base_dir: Path, experiment_id: str) -> Dict[str, Path]:
    """
    Create directory structure for an experiment.

    Parameters:
    -----------
    base_dir : Path
        Base directory for experiments (e.g., outputs/experiments)
    experiment_id : str
        Unique experiment ID

    Returns:
    --------
    dirs : dict
        Dictionary with paths to created directories:
        - 'root': Experiment root directory
        - 'model': Model checkpoints directory
        - 'results': Results directory (figures are saved in results/figures/)
        - 'logs': Logs directory
    """
    exp_dir = base_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    dirs = {
        "root": exp_dir,
        "model": exp_dir / "model",
        "results": exp_dir / "results",
        "logs": exp_dir / "logs",
    }

    # Create all subdirectories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Parameters:
    -----------
    config_path : Path
        Path to YAML configuration file

    Returns:
    --------
    config : dict
        Configuration dictionary

    Raises:
    -------
    FileNotFoundError
        If config file doesn't exist
    yaml.YAMLError
        If config file is invalid YAML
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def validate_config(config: Dict[str, Any], required_keys: Optional[list] = None) -> bool:
    """
    Validate configuration has required keys.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    required_keys : list, optional
        List of required keys. If None, uses default required keys.

    Returns:
    --------
    is_valid : bool
        True if config is valid

    Raises:
    -------
    ValueError
        If required keys are missing
    """
    if required_keys is None:
        required_keys = ["data", "model", "training"]

    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        raise ValueError(f"Config missing required keys: {missing_keys}")

    return True


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """
    Save configuration to YAML file.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    output_path : Path
        Path to save configuration
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_json(data: Dict[str, Any], output_path: Path, indent: int = 2) -> None:
    """
    Save data to JSON file.

    Parameters:
    -----------
    data : dict
        Data to save
    output_path : Path
        Path to save JSON file
    indent : int
        JSON indentation (default: 2)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        """Convert numpy types to Python types."""
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        return obj

    serializable_data = convert_to_serializable(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=indent, default=str)


def load_json(input_path: Path) -> Dict[str, Any]:
    """
    Load data from JSON file.

    Parameters:
    -----------
    input_path : Path
        Path to JSON file

    Returns:
    --------
    data : dict
        Loaded data
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_git_commit() -> Optional[str]:
    """
    Get current git commit hash.

    Returns:
    --------
    commit_hash : str or None
        Git commit hash, or None if not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_branch() -> Optional[str]:
    """
    Get current git branch name.

    Returns:
    --------
    branch : str or None
        Git branch name, or None if not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_package_versions() -> Dict[str, str]:
    """
    Get versions of key packages.

    Returns:
    --------
    versions : dict
        Dictionary of package names and versions
    """
    versions = {}

    packages = ["torch", "numpy", "pandas", "scipy", "sklearn", "matplotlib", "yaml"]

    for package in packages:
        try:
            if package == "yaml":
                import yaml as yaml_module

                versions[package] = getattr(yaml_module, "__version__", "unknown")
            elif package == "sklearn":
                import sklearn

                versions[package] = sklearn.__version__
            else:
                mod = __import__(package)
                versions[package] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[package] = "not installed"

    return versions
