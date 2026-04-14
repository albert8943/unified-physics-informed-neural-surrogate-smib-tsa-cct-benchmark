#!/usr/bin/env python3
"""
Data exporter module.

Functions for exporting data in PINN-friendly formats and managing checkpoints.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False

# Import schema validation
try:
    from utils.data_schema import SCHEMA, validate_data_schema
except ImportError:
    try:
        from examples.scripts.utils.data_schema import SCHEMA, validate_data_schema
    except ImportError:
        # Minimal schema if not available
        SCHEMA = {
            "scalar_features": [
                "Pm",
                "M",
                "D",
                "Xprefault",
                "Xfault",
                "Xpostfault",
                "fault_bus",
                "fault_start",
                "clearing_time",
                "fault_reactance",
                "delta0",
                "omega0",
                "V0",
                "Pe0",
            ],
            "time_series": ["time", "delta", "omega", "voltage", "Pe"],
            "derived_features": ["delta_dev", "omega_dev", "normalized_time"],
            "labels": ["is_stable", "max_angle", "max_freq_dev", "cct"],
        }

        def validate_data_schema(data_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
            return True, []


def save_checkpoint(state: Dict[str, Any], checkpoint_file: Path) -> Path:
    """
    Save generation state to checkpoint file.

    Parameters:
    -----------
    state : dict
        Generation state dictionary
    checkpoint_file : Path
        Path to checkpoint file

    Returns:
    --------
    checkpoint_path : Path
        Path to saved checkpoint file
    """
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    state_serializable = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            state_serializable[key] = value.tolist()
        elif isinstance(value, np.generic):
            state_serializable[key] = float(value)
        else:
            state_serializable[key] = value

    with open(checkpoint_file, "w") as f:
        json.dump(state_serializable, f, indent=2, default=str)

    return checkpoint_file


def load_checkpoint(checkpoint_file: Path) -> Dict[str, Any]:
    """
    Load generation state from checkpoint file.

    Parameters:
    -----------
    checkpoint_file : Path
        Path to checkpoint file

    Returns:
    --------
    state : dict
        Generation state dictionary
    """
    with open(checkpoint_file, "r") as f:
        state = json.load(f)

    return state


def export_pinn_data(
    data_list: List[Dict[str, Any]],
    output_dir: Path,
    format: str = "hdf5",
    data_version: str = "1.0.0",
    validate_schema: bool = True,
) -> Dict[str, Path]:
    """
    Export data in PINN-friendly format.

    Parameters:
    -----------
    data_list : list
        List of data dictionaries (one per sample)
    output_dir : Path
        Output directory
    format : str
        Export format: 'hdf5', 'csv', or 'both'
    data_version : str
        Data version string
    validate_schema : bool
        Whether to validate schema before export

    Returns:
    --------
    file_paths : dict
        Dictionary with paths to exported files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_paths = {}

    # Validate schema if requested
    if validate_schema:
        for i, data in enumerate(data_list):
            is_valid, errors = validate_data_schema(data)
            if not is_valid:
                warnings.warn("Schema validation failed for sample {i}: {errors}")

    # Prepare data for export
    # Separate scalar features from time series
    scalar_data = []
    trajectory_data = []

    for data in data_list:
        scalar_row = {}
        traj_dict = {}

        for key in SCHEMA["scalar_features"]:
            if key in data:
                scalar_row[key] = data[key]

        for key in SCHEMA["time_series"]:
            if key in data:
                traj_dict[key] = data[key]

        for key in SCHEMA["derived_features"]:
            if key in data:
                traj_dict[key] = data[key]

        for key in SCHEMA["labels"]:
            if key in data:
                scalar_row[key] = data[key]

        scalar_data.append(scalar_row)
        trajectory_data.append(traj_dict)

    # Export CSV
    if format in ["csv", "both"]:
        # Create DataFrame: one row per time point
        csv_rows = []
        for i, (scalar, traj) in enumerate(zip(scalar_data, trajectory_data)):
            time = traj.get("time", np.array([]))
            if len(time) > 0:
                for j in range(len(time)):
                    row = scalar.copy()
                    for key, value in traj.items():
                        if isinstance(value, np.ndarray) and len(value) > j:
                            row[key] = value[j]
                        elif not isinstance(value, np.ndarray):
                            row[key] = value
                    row["sample_id"] = i
                    row["time_index"] = j
                    csv_rows.append(row)

        if csv_rows:
            df = pd.DataFrame(csv_rows)
            csv_file = output_dir / f"pinn_training_data_v{data_version}.csv"
            df.to_csv(csv_file, index=False)
            file_paths["csv"] = csv_file

    # Export HDF5
    if format in ["hdf5", "both"] and H5PY_AVAILABLE:
        hdf5_file = output_dir / f"pinn_training_data_v{data_version}.h5"

        with h5py.File(hdf5_file, "w") as f:
            # Parameters group
            params_group = f.create_group("parameters")
            for key in SCHEMA["scalar_features"]:
                values = [d.get(key, 0) for d in scalar_data]
                params_group.create_dataset(key, data=np.array(values))

            # Trajectories group
            traj_group = f.create_group("trajectories")
            for key in SCHEMA["time_series"]:
                # Stack all trajectories (samples × time_points)
                traj_list = []
                max_len = 0
                for traj in trajectory_data:
                    if key in traj:
                        arr = np.array(traj[key])
                        traj_list.append(arr)
                        max_len = max(max_len, len(arr))
                    else:
                        traj_list.append(np.array([]))

                # Pad to same length
                padded = []
                for arr in traj_list:
                    if len(arr) < max_len:
                        padded_arr = np.pad(
                            arr, (0, max_len - len(arr)), mode="constant", constant_values=np.nan
                        )
                    else:
                        padded_arr = arr
                    padded.append(padded_arr)

                if padded:
                    traj_group.create_dataset(key, data=np.array(padded))

            # Labels group
            labels_group = f.create_group("labels")
            for key in SCHEMA["labels"]:
                values = [d.get(key, 0) for d in scalar_data]
                labels_group.create_dataset(key, data=np.array(values))

            # Metadata
            f.attrs["data_version"] = data_version
            f.attrs["n_samples"] = len(data_list)
            f.attrs["schema"] = json.dumps(SCHEMA)

        file_paths["hdf5"] = hdf5_file

    return file_paths


def save_metadata(
    output_dir: Path,
    param_ranges: Dict[str, List[float]],
    statistics: Dict[str, Any],
    andes_version: Optional[str] = None,
    case_file_hash: Optional[str] = None,
    data_version: str = "1.0.0",
) -> Path:
    """
    Save generation metadata.

    Parameters:
    -----------
    output_dir : Path
        Output directory
    param_ranges : dict
        Parameter ranges used
    statistics : dict
        Generation statistics
    andes_version : str, optional
        ANDES version
    case_file_hash : str, optional
        Case file hash
    data_version : str
        Data version

    Returns:
    --------
    metadata_path : Path
        Path to metadata file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "data_version": data_version,
        "generation_timestamp": datetime.now().isoformat(),
        "andes_version": andes_version or (andes.__version__ if ANDES_AVAILABLE else "unknown"),
        "case_file_hash": case_file_hash,
        "parameter_ranges": param_ranges,
        "statistics": statistics,
        "schema": SCHEMA,
    }

    metadata_path = output_dir / "metadata_v{data_version}.json"

    # Convert numpy types for JSON
    def convert_numpy(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    metadata_serializable = convert_numpy(metadata)

    with open(metadata_path, "w") as f:
        json.dump(metadata_serializable, f, indent=2, default=str)

    return metadata_path
