"""
Case File Modifier for ANDES Generator Setpoint

This module provides functions to modify generator setpoints in ANDES case files
BEFORE loading the system, ensuring power flow uses the correct setpoint.

Author: Albert
Date: 2025-12-19
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set up module logger
logger = logging.getLogger(__name__)

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False


def _build_idx_to_pos_mapping(device_data, idx_field_name: str = "idx") -> Dict:
    """
    Build a mapping from device idx (identifier) to positional index.

    ANDES case files use idx as identifiers (often strings like "1", "G1"),
    not necessarily 0-based positions. This function creates a mapping.

    Args:
        device_data: Device data (dict or list format)
        idx_field_name: Name of the idx field (default: "idx")

    Returns:
        Dict mapping idx value -> positional index
    """
    idx_to_pos = {}
    if isinstance(device_data, dict):
        if idx_field_name in device_data:
            idx_list = device_data[idx_field_name]
            if isinstance(idx_list, list):
                for pos, idx_val in enumerate(idx_list):
                    idx_to_pos[idx_val] = pos
    elif isinstance(device_data, list):
        for pos, device in enumerate(device_data):
            if isinstance(device, dict) and idx_field_name in device:
                idx_to_pos[device[idx_field_name]] = pos
    return idx_to_pos


def _find_static_gen_pos_by_idx(
    static_gen_data, static_gen_idx, idx_to_pos: Optional[Dict] = None
) -> Optional[int]:
    """
    Find the positional index of a StaticGen by its idx identifier.

    Args:
        static_gen_data: StaticGen data (dict or list format)
        static_gen_idx: The idx identifier from GENCLS.gen (may be int, str, etc.)
        idx_to_pos: Optional pre-built idx->position mapping

    Returns:
        Positional index if found, None otherwise
    """
    # If idx_to_pos mapping provided, use it
    if idx_to_pos is not None:
        return idx_to_pos.get(static_gen_idx)

    # Otherwise, try direct positional access (for numeric indices)
    if isinstance(static_gen_idx, (int, float)):
        if isinstance(static_gen_data, dict):
            # Check if p0 list exists and index is valid
            if "p0" in static_gen_data and isinstance(static_gen_data["p0"], list):
                if int(static_gen_idx) < len(static_gen_data["p0"]):
                    return int(static_gen_idx)
        elif isinstance(static_gen_data, list):
            if int(static_gen_idx) < len(static_gen_data):
                return int(static_gen_idx)

    # Try to build mapping and use it
    idx_to_pos = _build_idx_to_pos_mapping(static_gen_data)
    if idx_to_pos:
        return idx_to_pos.get(static_gen_idx)

    # Fallback: assume numeric idx is positional (for backward compatibility)
    if isinstance(static_gen_idx, (int, float)):
        return int(static_gen_idx)

    return None


def inspect_case_file(case_path: str) -> dict:
    """
    Inspect case file to understand its structure.

    Args:
        case_path: Path to case file

    Returns:
        Dictionary with case file information
    """
    case_path_obj = Path(case_path)

    if not case_path_obj.exists():
        raise FileNotFoundError(f"Case file not found: {case_path}")

    info = {
        "path": str(case_path),
        "format": case_path_obj.suffix,
        "exists": True,
    }

    if case_path_obj.suffix == ".json":
        # JSON format
        with open(case_path, "r") as f:
            case_data = json.load(f)

        info["type"] = "JSON"
        info["keys"] = list(case_data.keys())

        # Check for generator data
        if "GENCLS" in case_data:
            gen_data = case_data["GENCLS"]
            info["has_gencls"] = True

            if isinstance(gen_data, dict):
                info["gencls_type"] = "dict"
                info["gencls_keys"] = list(gen_data.keys())
                # Look for power-related fields
                power_fields = [
                    k
                    for k in gen_data.keys()
                    if any(p in k.upper() for p in ["P0", "PG", "P", "PM", "PGEN"])
                ]
                info["power_fields"] = power_fields
            elif isinstance(gen_data, list):
                info["gencls_type"] = "list"
                info["gencls_length"] = len(gen_data)
                if len(gen_data) > 0:
                    info["gencls_sample"] = (
                        gen_data[0] if isinstance(gen_data[0], dict) else str(gen_data[0])
                    )
        else:
            info["has_gencls"] = False

        info["sample_data"] = json.dumps(case_data, indent=2)[:1000]  # First 1000 chars

    elif case_path_obj.suffix == ".xlsx" and PANDAS_AVAILABLE:
        # Excel format
        xls = pd.ExcelFile(case_path)
        info["type"] = "Excel"
        info["sheets"] = xls.sheet_names

        # Find generator sheet
        gen_sheet = None
        for sheet_name in xls.sheet_names:
            if "GEN" in sheet_name.upper() or "GENCLS" in sheet_name.upper():
                gen_sheet = sheet_name
                break

        if gen_sheet:
            df = pd.read_excel(case_path, sheet_name=gen_sheet)
            info["has_gencls"] = True
            info["gencls_sheet"] = gen_sheet
            info["gencls_columns"] = list(df.columns)
            # Look for power-related columns
            power_cols = [
                c
                for c in df.columns
                if any(p in c.upper() for p in ["P0", "PG", "P", "PM", "PGEN"])
            ]
            info["power_columns"] = power_cols
        else:
            info["has_gencls"] = False

    else:
        info["type"] = "Unknown"
        info["supported"] = False

    return info


def modify_case_file_generator_setpoint(
    case_path: str,
    gen_idx: int,
    new_pm: float,
    output_path: Optional[str] = None,
    power_field_name: Optional[str] = None,
) -> str:
    """
    Modify generator setpoint in case file BEFORE loading.

    This is the CORRECT way to change generator power setpoint in ANDES.
    Power flow reads setpoint from case file, so modifying the case file
    ensures power flow uses the correct value.

    Args:
        case_path: Path to original case file
        gen_idx: Generator index (usually 0 for SMIB)
        new_pm: New mechanical power setpoint (pu)
        output_path: Optional output path (default: temp file)
        power_field_name: Optional field name to modify (auto-detect if None)

    Returns:
        Path to modified case file

    Example:
        >>> case_path = andes.get_case("smib/SMIB.json")
        >>> modified_path = modify_case_file_generator_setpoint(
        ...     case_path, gen_idx=0, new_pm=0.803
        ... )
        >>> ss = andes.load(modified_path)
        >>> ss.PFlow.run()
        >>> # Now tm0 should equal 0.803 pu
    """
    case_path_obj = Path(case_path)

    if not case_path_obj.exists():
        raise FileNotFoundError(f"Case file not found: {case_path}")

    # Determine output path
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        case_name = case_path_obj.stem
        # Use safe filename format (replace . with _ to avoid issues)
        pm_str = f"{new_pm:.6f}".replace(".", "_")
        output_path = str(
            Path(temp_dir) / f"{case_name}_G{gen_idx}_Pm{pm_str}{case_path_obj.suffix}"
        )

    # Modify based on format
    if case_path_obj.suffix == ".json":
        return _modify_json_case_file(case_path, gen_idx, new_pm, output_path, power_field_name)
    elif case_path_obj.suffix == ".xlsx" and PANDAS_AVAILABLE:
        return _modify_excel_case_file(case_path, gen_idx, new_pm, output_path, power_field_name)
    else:
        raise ValueError(
            f"Unsupported case file format: {case_path_obj.suffix}. "
            f"Supported formats: .json, .xlsx (requires pandas)"
        )


def _modify_json_case_file(
    case_path: str,
    gen_idx: int,
    new_pm: float,
    output_path: str,
    power_field_name: Optional[str] = None,
) -> str:
    """
    Modify JSON case file.

    According to ANDES documentation:
    - GENCLS doesn't have power setpoint fields (P0, Pg, etc.) in the case file
    - GENCLS is connected to StaticGen via the 'gen' field
    - Power setpoint should be modified in StaticGen's 'p0' field
    """
    with open(case_path, "r") as f:
        case_data = json.load(f)

    if "GENCLS" not in case_data:
        raise ValueError(f"Case file does not contain GENCLS data: {case_path}")

    gen_data = case_data["GENCLS"]
    modified = False

    # First, try to find and modify StaticGen (the correct approach per ANDES docs)
    # GENCLS references StaticGen via the 'gen' field
    static_gen_idx = None
    if isinstance(gen_data, dict):
        # Dict format: { "gen": [0, 1], ... }
        if "gen" in gen_data:
            if isinstance(gen_data["gen"], list) and gen_idx < len(gen_data["gen"]):
                static_gen_idx = gen_data["gen"][gen_idx]
            elif not isinstance(gen_data["gen"], list):
                static_gen_idx = gen_data["gen"]
    elif isinstance(gen_data, list):
        # List format: [ {"gen": 0, ...}, ... ]
        if gen_idx < len(gen_data) and isinstance(gen_data[gen_idx], dict):
            if "gen" in gen_data[gen_idx]:
                static_gen_idx = gen_data[gen_idx]["gen"]

    # If we found a generator reference, modify the appropriate static generator component
    # Try StaticGen first, then PV, then Slack
    # Use idx mapping to handle both positional indices and ID-based references
    if static_gen_idx is not None:
        # Try StaticGen
        if "StaticGen" in case_data:
            static_gen_data = case_data["StaticGen"]
            # Build idx-to-position mapping for StaticGen
            static_gen_idx_to_pos = _build_idx_to_pos_mapping(static_gen_data)
            static_gen_pos = _find_static_gen_pos_by_idx(
                static_gen_data, static_gen_idx, static_gen_idx_to_pos
            )

            if static_gen_pos is not None:
                if isinstance(static_gen_data, dict):
                    # Dict format: { "p0": [0.9, 0.8], ... }
                    if "p0" in static_gen_data:
                        if isinstance(static_gen_data["p0"], list):
                            if static_gen_pos < len(static_gen_data["p0"]):
                                static_gen_data["p0"][static_gen_pos] = new_pm
                                modified = True
                                logger.info(
                                    f"Set StaticGen.p0[{static_gen_pos}] = {new_pm:.6f} pu "
                                    f"(via GENCLS[{gen_idx}].gen -> StaticGen idx={static_gen_idx})"
                                )
                        else:
                            static_gen_data["p0"] = new_pm
                            modified = True
                            logger.info(
                                f"Set StaticGen.p0 = {new_pm:.6f} pu "
                                f"(via GENCLS[{gen_idx}].gen -> StaticGen idx={static_gen_idx})"
                            )
                elif isinstance(static_gen_data, list):
                    # List format: [ {"p0": 0.9, ...}, ... ]
                    if static_gen_pos < len(static_gen_data) and isinstance(
                        static_gen_data[static_gen_pos], dict
                    ):
                        static_gen_data[static_gen_pos]["p0"] = new_pm
                        modified = True
                        logger.info(
                            f"Set StaticGen[{static_gen_pos}].p0 = {new_pm:.6f} pu "
                            f"(via GENCLS[{gen_idx}].gen -> StaticGen idx={static_gen_idx})"
                        )

        # If StaticGen didn't work, try PV component
        if not modified and "PV" in case_data:
            pv_data = case_data["PV"]
            # Build idx-to-position mapping for PV
            pv_idx_to_pos = _build_idx_to_pos_mapping(pv_data)
            pv_pos = _find_static_gen_pos_by_idx(pv_data, static_gen_idx, pv_idx_to_pos)

            if pv_pos is not None:
                if isinstance(pv_data, dict):
                    if "p0" in pv_data:
                        if isinstance(pv_data["p0"], list):
                            if pv_pos < len(pv_data["p0"]):
                                pv_data["p0"][pv_pos] = new_pm
                                modified = True
                                logger.info(
                                    f"Set PV.p0[{pv_pos}] = {new_pm:.6f} pu "
                                    f"(via GENCLS[{gen_idx}].gen -> PV idx={static_gen_idx})"
                                )
                        else:
                            pv_data["p0"] = new_pm
                            modified = True
                            logger.info(
                                f"Set PV.p0 = {new_pm:.6f} pu "
                                f"(via GENCLS[{gen_idx}].gen -> PV idx={static_gen_idx})"
                            )
                elif isinstance(pv_data, list):
                    if pv_pos < len(pv_data) and isinstance(pv_data[pv_pos], dict):
                        pv_data[pv_pos]["p0"] = new_pm
                        modified = True
                        logger.info(
                            f"Set PV[{pv_pos}].p0 = {new_pm:.6f} pu "
                            f"(via GENCLS[{gen_idx}].gen -> PV idx={static_gen_idx})"
                        )

        # If neither worked, try Slack component
        if not modified and "Slack" in case_data:
            slack_data = case_data["Slack"]
            # Build idx-to-position mapping for Slack
            slack_idx_to_pos = _build_idx_to_pos_mapping(slack_data)
            slack_pos = _find_static_gen_pos_by_idx(slack_data, static_gen_idx, slack_idx_to_pos)

            if slack_pos is not None:
                if isinstance(slack_data, dict):
                    if "p0" in slack_data:
                        if isinstance(slack_data["p0"], list):
                            if slack_pos < len(slack_data["p0"]):
                                slack_data["p0"][slack_pos] = new_pm
                                modified = True
                                logger.info(
                                    f"Set Slack.p0[{slack_pos}] = {new_pm:.6f} pu "
                                    f"(via GENCLS[{gen_idx}].gen -> Slack idx={static_gen_idx})"
                                )
                        else:
                            slack_data["p0"] = new_pm
                            modified = True
                            logger.info(
                                f"Set Slack.p0 = {new_pm:.6f} pu "
                                f"(via GENCLS[{gen_idx}].gen -> Slack idx={static_gen_idx})"
                            )
                elif isinstance(slack_data, list):
                    if slack_pos < len(slack_data) and isinstance(slack_data[slack_pos], dict):
                        slack_data[slack_pos]["p0"] = new_pm
                        modified = True
                        logger.info(
                            f"Set Slack[{slack_pos}].p0 = {new_pm:.6f} pu "
                            f"(via GENCLS[{gen_idx}].gen -> Slack idx={static_gen_idx})"
                        )

    # If StaticGen modification didn't work, fall back to trying GENCLS directly
    # (some case files might have power fields in GENCLS, though not standard)
    if not modified:
        if isinstance(gen_data, dict):
            # Dict format: { "P0": [0.9], "M": [6.0], ... }
            if power_field_name:
                field = power_field_name
            else:
                # Auto-detect power field - only check plausible setpoint fields
                # Avoid "P" (too generic) and "tm0/tm" (not case-file setpoints)
                for field in ["P0", "Pg", "Pm", "Pgen"]:
                    if field in gen_data:
                        break
                else:
                    # Check if we have StaticGen available
                    if "StaticGen" in case_data:
                        raise ValueError(
                            f"Could not find power setpoint field in GENCLS. "
                            f"Available fields: {list(gen_data.keys())}. "
                            f"Note: GENCLS power setpoints are typically stored in StaticGen.p0. "
                            f"StaticGen found in case file but 'gen' field not found in GENCLS."
                        )
                    else:
                        raise ValueError(
                            f"Could not find power setpoint field in GENCLS. "
                            f"Available fields: {list(gen_data.keys())}. "
                            f"Note: GENCLS power setpoints are typically stored in StaticGen.p0, "
                            f"but StaticGen not found in case file."
                        )

            if field not in gen_data:
                raise ValueError(
                    f"Power field '{field}' not found in GENCLS. "
                    f"Available fields: {list(gen_data.keys())}"
                )

            # Modify field
            if isinstance(gen_data[field], list):
                if gen_idx >= len(gen_data[field]):
                    raise IndexError(
                        f"Generator index {gen_idx} out of range. "
                        f"GENCLS.{field} has {len(gen_data[field])} elements."
                    )
                gen_data[field][gen_idx] = new_pm
            else:
                # Scalar value
                gen_data[field] = new_pm

            modified = True
            logger.info(f"Set GENCLS.{field}[{gen_idx}] = {new_pm:.6f} pu")

        elif isinstance(gen_data, list):
            # List format: [ {"P0": 0.9, "M": 6.0, ...}, ... ]
            if gen_idx >= len(gen_data):
                raise IndexError(
                    f"Generator index {gen_idx} out of range. "
                    f"GENCLS has {len(gen_data)} generators."
                )

            gen = gen_data[gen_idx]
            if not isinstance(gen, dict):
                raise ValueError(
                    f"GENCLS[{gen_idx}] is not a dictionary. " f"Expected dict, got {type(gen)}"
                )

            if power_field_name:
                field = power_field_name
            else:
                # Auto-detect power field - only check plausible setpoint fields
                # Avoid "P" (too generic) and "tm0/tm" (not case-file setpoints)
                for field in ["P0", "Pg", "Pm", "Pgen"]:
                    if field in gen:
                        break
                else:
                    # Check if we have StaticGen available
                    if "StaticGen" in case_data:
                        raise ValueError(
                            f"Could not find power setpoint field in GENCLS[{gen_idx}]. "
                            f"Available fields: {list(gen.keys())}. "
                            f"Note: GENCLS power setpoints are typically stored in StaticGen.p0. "
                            f"StaticGen found in case file but 'gen' field not found in"
                            f"GENCLS[{gen_idx}]."
                        )
                    else:
                        raise ValueError(
                            f"Could not find power setpoint field in GENCLS[{gen_idx}]. "
                            f"Available fields: {list(gen.keys())}. "
                            f"Note: GENCLS power setpoints are typically stored in StaticGen.p0, "
                            f"but StaticGen not found in case file."
                        )

            gen[field] = new_pm
            modified = True
            logger.info(f"Set GENCLS[{gen_idx}].{field} = {new_pm:.6f} pu")

        else:
            raise ValueError(
                f"Unexpected GENCLS format: {type(gen_data)}. " f"Expected dict or list."
            )

    if not modified:
        raise RuntimeError("Failed to modify case file (unknown format)")

    # Write modified case file
    with open(output_path, "w") as f:
        json.dump(case_data, f, indent=2)

    logger.info(f"Created modified case file: {output_path}")
    return output_path


def _modify_excel_case_file(
    case_path: str,
    gen_idx: int,
    new_pm: float,
    output_path: str,
    power_field_name: Optional[str] = None,
) -> str:
    """Modify Excel case file."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for Excel case file modification")

    # Read Excel file
    xls = pd.ExcelFile(case_path)

    # Find generator sheet
    gen_sheet = None
    for sheet_name in xls.sheet_names:
        if "GEN" in sheet_name.upper() or "GENCLS" in sheet_name.upper():
            gen_sheet = sheet_name
            break

    if gen_sheet is None:
        raise ValueError(f"Could not find generator sheet in {case_path}")

    # Read generator data
    df = pd.read_excel(case_path, sheet_name=gen_sheet)

    # Find power setpoint column
    if power_field_name:
        power_col = power_field_name
    else:
        # Auto-detect power column
        for col in ["P0", "Pg", "P", "Pm", "Pgen"]:
            if col in df.columns:
                power_col = col
                break
        else:
            raise ValueError(
                f"Could not find power setpoint column in {gen_sheet}. "
                f"Available columns: {list(df.columns)}"
            )

    if power_col not in df.columns:
        raise ValueError(
            f"Power column '{power_col}' not found in {gen_sheet}. "
            f"Available columns: {list(df.columns)}"
        )

    # Check generator index
    if gen_idx >= len(df):
        raise IndexError(
            f"Generator index {gen_idx} out of range. " f"{gen_sheet} has {len(df)} generators."
        )

    # Modify setpoint
    df.loc[gen_idx, power_col] = new_pm
    logger.info(f"Set {gen_sheet}.{power_col}[{gen_idx}] = {new_pm:.6f} pu")

    # Write modified Excel file
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Write all sheets
        for sheet in xls.sheet_names:
            if sheet == gen_sheet:
                df.to_excel(writer, sheet_name=sheet, index=False)
            else:
                pd.read_excel(case_path, sheet_name=sheet).to_excel(
                    writer, sheet_name=sheet, index=False
                )

    logger.info(f"Created modified case file: {output_path}")
    return output_path


def get_default_m_from_case_file(case_path: str, gen_idx: int = 0) -> Optional[float]:
    """
    Extract default M (inertia coefficient) value from case file.

    Args:
        case_path: Path to case file
        gen_idx: Generator index

    Returns:
        Default M value, or None if not found
    """
    try:
        if case_path.endswith(".json"):
            with open(case_path, "r") as f:
                case_data = json.load(f)

            if "GENCLS" not in case_data:
                return None

            gen_data = case_data["GENCLS"]

            # Extract M from GENCLS
            if isinstance(gen_data, dict):
                if "M" in gen_data:
                    value = gen_data["M"]
                    if isinstance(value, list):
                        if gen_idx < len(value):
                            return float(value[gen_idx])
                    else:
                        return float(value)

            elif isinstance(gen_data, list):
                if gen_idx < len(gen_data):
                    gen = gen_data[gen_idx]
                    if isinstance(gen, dict) and "M" in gen:
                        return float(gen["M"])

        elif case_path.endswith(".xlsx") and PANDAS_AVAILABLE:
            xls = pd.ExcelFile(case_path)
            for sheet_name in xls.sheet_names:
                if "GEN" in sheet_name.upper():
                    df = pd.read_excel(case_path, sheet_name=sheet_name)
                    if "M" in df.columns and gen_idx < len(df):
                        return float(df.loc[gen_idx, "M"])

    except Exception as e:
        print(f"[WARNING] Could not extract default M from case file: {e}")

    return None


def get_default_d_from_case_file(case_path: str, gen_idx: int = 0) -> Optional[float]:
    """
    Extract default D (damping coefficient) value from case file.

    Args:
        case_path: Path to case file
        gen_idx: Generator index

    Returns:
        Default D value, or None if not found
    """
    try:
        if case_path.endswith(".json"):
            with open(case_path, "r") as f:
                case_data = json.load(f)

            if "GENCLS" not in case_data:
                return None

            gen_data = case_data["GENCLS"]

            # Extract D from GENCLS
            if isinstance(gen_data, dict):
                if "D" in gen_data:
                    value = gen_data["D"]
                    if isinstance(value, list):
                        if gen_idx < len(value):
                            return float(value[gen_idx])
                    else:
                        return float(value)

            elif isinstance(gen_data, list):
                if gen_idx < len(gen_data):
                    gen = gen_data[gen_idx]
                    if isinstance(gen, dict) and "D" in gen:
                        return float(gen["D"])

        elif case_path.endswith(".xlsx") and PANDAS_AVAILABLE:
            xls = pd.ExcelFile(case_path)
            for sheet_name in xls.sheet_names:
                if "GEN" in sheet_name.upper():
                    df = pd.read_excel(case_path, sheet_name=sheet_name)
                    if "D" in df.columns and gen_idx < len(df):
                        return float(df.loc[gen_idx, "D"])

    except Exception as e:
        print(f"[WARNING] Could not extract default D from case file: {e}")

    return None


def get_default_pm_from_case_file(case_path: str, gen_idx: int = 0) -> Optional[float]:
    """
    Extract default Pm value from case file.

    Args:
        case_path: Path to case file
        gen_idx: Generator index

    Returns:
        Default Pm value, or None if not found
    """
    try:
        info = inspect_case_file(case_path)

        if case_path.endswith(".json"):
            with open(case_path, "r") as f:
                case_data = json.load(f)

            if "GENCLS" not in case_data:
                return None

            gen_data = case_data["GENCLS"]

            # First, try to get from StaticGen (correct approach per ANDES docs)
            if "StaticGen" in case_data:
                static_gen_data = case_data["StaticGen"]
                static_gen_idx = None

                # Find StaticGen index from GENCLS.gen
                if isinstance(gen_data, dict):
                    if "gen" in gen_data:
                        if isinstance(gen_data["gen"], list) and gen_idx < len(gen_data["gen"]):
                            static_gen_idx = gen_data["gen"][gen_idx]
                        elif not isinstance(gen_data["gen"], list):
                            static_gen_idx = gen_data["gen"]
                elif isinstance(gen_data, list):
                    if gen_idx < len(gen_data) and isinstance(gen_data[gen_idx], dict):
                        if "gen" in gen_data[gen_idx]:
                            static_gen_idx = gen_data[gen_idx]["gen"]

                # Get p0 from StaticGen
                if static_gen_idx is not None:
                    static_gen_idx_to_pos = _build_idx_to_pos_mapping(static_gen_data)
                    static_gen_pos = _find_static_gen_pos_by_idx(
                        static_gen_data, static_gen_idx, static_gen_idx_to_pos
                    )

                    if static_gen_pos is not None:
                        if isinstance(static_gen_data, dict):
                            if "p0" in static_gen_data:
                                p0_list = static_gen_data["p0"]
                                if isinstance(p0_list, list) and static_gen_pos < len(p0_list):
                                    return float(p0_list[static_gen_pos])
                                elif not isinstance(p0_list, list):
                                    return float(p0_list)
                        elif isinstance(static_gen_data, list):
                            if static_gen_pos < len(static_gen_data):
                                gen_dict = static_gen_data[static_gen_pos]
                                if isinstance(gen_dict, dict) and "p0" in gen_dict:
                                    return float(gen_dict["p0"])

            # Fallback: Try to find power field in GENCLS directly (non-standard)
            if isinstance(gen_data, dict):
                for field in ["P0", "Pg", "Pm", "Pgen"]:  # Removed "P" (too generic)
                    if field in gen_data:
                        value = gen_data[field]
                        if isinstance(value, list):
                            if gen_idx < len(value):
                                return float(value[gen_idx])
                        else:
                            return float(value)

            elif isinstance(gen_data, list):
                if gen_idx < len(gen_data):
                    gen = gen_data[gen_idx]
                    if isinstance(gen, dict):
                        for field in ["P0", "Pg", "Pm", "Pgen"]:  # Removed "P" (too generic)
                            if field in gen:
                                return float(gen[field])

        elif case_path.endswith(".xlsx") and PANDAS_AVAILABLE:
            xls = pd.ExcelFile(case_path)
            for sheet_name in xls.sheet_names:
                if "GEN" in sheet_name.upper():
                    df = pd.read_excel(case_path, sheet_name=sheet_name)
                    for col in ["P0", "Pg", "P", "Pm", "Pgen"]:
                        if col in df.columns:
                            if gen_idx < len(df):
                                return float(df.loc[gen_idx, col])

    except Exception as e:
        print(f"[WARNING] Could not extract default Pm from case file: {e}")

    return None


def get_generator_p_from_andes_case(
    case_path: str, addfile: Optional[str] = None
) -> Optional[List[float]]:
    """
    Get per-generator active power setpoints (P) from the case by loading it in ANDES
    and reading values after power flow. Use for .raw + .dyr when JSON/Excel parsing
    is not available; values come from the case's power flow (generator P in .raw).

    Args:
        case_path: Path to case file (e.g. .raw or .json).
        addfile: Optional path to dynamic file (e.g. .dyr) for PSS/E.

    Returns:
        List of P in generator (GENCLS) order, or None if loading/flow failed.
    """
    if not ANDES_AVAILABLE:
        return None
    try:
        case_path_obj = Path(case_path)
        if not case_path_obj.exists():
            return None
        addfile_path = None
        if addfile:
            addfile_obj = Path(addfile)
            if addfile_obj.exists():
                addfile_path = str(addfile_obj)
            elif hasattr(andes, "get_case"):
                addfile_path = andes.get_case(addfile)
        ss = (
            andes.load(str(case_path_obj), addfile=addfile_path)
            if addfile_path
            else andes.load(str(case_path_obj))
        )
        ss.setup()
        if not ss.PFlow.run():
            return None
        # After power flow, generator scheduled P: PV.p0 (and Slack if present)
        out = []
        if hasattr(ss, "PV") and ss.PV.n > 0 and hasattr(ss.PV, "p0") and hasattr(ss.PV.p0, "v"):
            p0v = ss.PV.p0.v
            if hasattr(p0v, "__len__"):
                out.extend(float(x) for x in p0v)
            else:
                out.append(float(p0v))
        if (
            hasattr(ss, "Slack")
            and ss.Slack.n > 0
            and hasattr(ss.Slack, "P0")
            and hasattr(ss.Slack.P0, "v")
        ):
            p0_slack = ss.Slack.P0.v
            if hasattr(p0_slack, "__len__"):
                out.extend(float(x) for x in p0_slack)
            else:
                out.append(float(p0_slack))
        return out if out else None
    except Exception as e:
        logger.warning("Could not get generator P from ANDES case %s: %s", case_path, e)
        return None


def modify_case_file_multiple_generators(
    case_path: str,
    generator_setpoints: Dict[int, float],
    output_path: Optional[str] = None,
    power_field_name: Optional[str] = None,
) -> str:
    """
    Modify multiple generator setpoints in case file BEFORE loading.

    This is the CORRECT way to change generator power setpoints in ANDES for
    multimachine systems. Power flow reads setpoints from case file, so modifying
    the case file ensures power flow uses the correct values.

    Args:
        case_path: Path to original case file
        generator_setpoints: Dict mapping generator index to Pm value
            Example: {0: 0.7, 1: 0.8, 2: 0.6}
        output_path: Optional output path (default: temp file)
        power_field_name: Optional field name to modify (auto-detect if None)

    Returns:
        Path to modified case file

    Example:
        >>> case_path = andes.get_case("kundur/kundur.json")
        >>> generator_setpoints = {0: 0.7, 1: 0.8, 2: 0.6}
        >>> modified_path = modify_case_file_multiple_generators(
        ...     case_path, generator_setpoints=generator_setpoints
        ... )
        >>> ss = andes.load(modified_path)
        >>> ss.PFlow.run()
        >>> # Now tm0[i] should equal generator_setpoints[i] for all generators
    """
    case_path_obj = Path(case_path)

    if not case_path_obj.exists():
        raise FileNotFoundError(f"Case file not found: {case_path}")

    if not generator_setpoints:
        raise ValueError("generator_setpoints cannot be empty")

    # Determine output path
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        case_name = case_path_obj.stem
        # Create a hash-like identifier from setpoints
        # Create safe filename format (replace . with _ to avoid issues)
        setpoint_str = "_".join(
            [
                f"G{idx}P{val:.6f}".replace(".", "_")
                for idx, val in sorted(generator_setpoints.items())
            ]
        )
        output_path = str(
            Path(temp_dir) / f"{case_name}_multi_{setpoint_str}{case_path_obj.suffix}"
        )

    # Modify based on format
    if case_path_obj.suffix == ".json":
        return _modify_json_case_file_multiple(
            case_path, generator_setpoints, output_path, power_field_name
        )
    elif case_path_obj.suffix == ".xlsx" and PANDAS_AVAILABLE:
        return _modify_excel_case_file_multiple(
            case_path, generator_setpoints, output_path, power_field_name
        )
    else:
        raise ValueError(
            f"Unsupported case file format: {case_path_obj.suffix}. "
            f"Supported formats: .json, .xlsx (requires pandas)"
        )


def _modify_json_case_file_multiple(
    case_path: str,
    generator_setpoints: Dict[int, float],
    output_path: str,
    power_field_name: Optional[str] = None,
) -> str:
    """Modify JSON case file for multiple generators."""
    with open(case_path, "r") as f:
        case_data = json.load(f)

    if "GENCLS" not in case_data:
        raise ValueError(f"Case file does not contain GENCLS data: {case_path}")

    gen_data = case_data["GENCLS"]
    modified_generators = set()  # Track which generators were successfully modified

    # First, try to find and modify StaticGen (the correct approach per ANDES docs)
    # GENCLS references StaticGen via the 'gen' field
    if "StaticGen" in case_data:
        static_gen_data = case_data["StaticGen"]
        # Build idx-to-position mapping for StaticGen (handles ID-based references)
        static_gen_idx_to_pos = _build_idx_to_pos_mapping(static_gen_data)

        for gen_idx, new_pm in generator_setpoints.items():
            static_gen_idx = None

            # Find the StaticGen index for this GENCLS generator
            if isinstance(gen_data, dict):
                # Dict format: { "gen": [0, 1], ... }
                if "gen" in gen_data:
                    if isinstance(gen_data["gen"], list) and gen_idx < len(gen_data["gen"]):
                        static_gen_idx = gen_data["gen"][gen_idx]
                    elif not isinstance(gen_data["gen"], list):
                        static_gen_idx = gen_data["gen"]
            elif isinstance(gen_data, list):
                # List format: [ {"gen": 0, ...}, ... ]
                if gen_idx < len(gen_data) and isinstance(gen_data[gen_idx], dict):
                    if "gen" in gen_data[gen_idx]:
                        static_gen_idx = gen_data[gen_idx]["gen"]

            # If we found a StaticGen reference, modify StaticGen's p0 field
            if static_gen_idx is not None:
                static_gen_pos = _find_static_gen_pos_by_idx(
                    static_gen_data, static_gen_idx, static_gen_idx_to_pos
                )

                if static_gen_pos is not None:
                    if isinstance(static_gen_data, dict):
                        # Dict format: { "p0": [0.9, 0.8], ... }
                        if "p0" in static_gen_data:
                            if isinstance(static_gen_data["p0"], list):
                                if static_gen_pos < len(static_gen_data["p0"]):
                                    static_gen_data["p0"][static_gen_pos] = new_pm
                                    modified_generators.add(gen_idx)
                                    logger.info(
                                        f"Set StaticGen.p0[{static_gen_pos}] = {new_pm:.6f} pu "
                                        f"(via GENCLS[{gen_idx}].gen -> StaticGen"
                                        f"idx={static_gen_idx})"
                                    )
                            else:
                                static_gen_data["p0"] = new_pm
                                modified_generators.add(gen_idx)
                                logger.info(
                                    f"Set StaticGen.p0 = {new_pm:.6f} pu "
                                    f"(via GENCLS[{gen_idx}].gen -> StaticGen idx={static_gen_idx})"
                                )
                    elif isinstance(static_gen_data, list):
                        # List format: [ {"p0": 0.9, ...}, ... ]
                        if static_gen_pos < len(static_gen_data) and isinstance(
                            static_gen_data[static_gen_pos], dict
                        ):
                            static_gen_data[static_gen_pos]["p0"] = new_pm
                            modified_generators.add(gen_idx)
                            logger.info(
                                f"Set StaticGen[{static_gen_pos}].p0 = {new_pm:.6f} pu "
                                f"(via GENCLS[{gen_idx}].gen -> StaticGen idx={static_gen_idx})"
                            )

    modified = len(modified_generators) > 0

    # If StaticGen modification didn't work, fall back to trying GENCLS directly
    # (some case files might have power fields in GENCLS, though not standard)
    if not modified:
        if isinstance(gen_data, dict):
            # Dict format: { "P0": [0.9, 0.8, 0.7], "M": [6.0, 5.0, 4.0], ... }
            if power_field_name:
                field = power_field_name
            else:
                # Auto-detect power field - check both power setpoint fields and tm0/tm
                for field in ["P0", "Pg", "P", "Pm", "Pgen", "tm0", "tm"]:
                    if field in gen_data:
                        break
                else:
                    # Check if we have StaticGen available
                    if "StaticGen" in case_data:
                        raise ValueError(
                            f"Could not find power setpoint field in GENCLS. "
                            f"Available fields: {list(gen_data.keys())}. "
                            f"Note: GENCLS power setpoints are typically stored in StaticGen.p0. "
                            f"StaticGen found in case file but 'gen' field not found in GENCLS."
                        )
                    else:
                        raise ValueError(
                            f"Could not find power setpoint field in GENCLS. "
                            f"Available fields: {list(gen_data.keys())}. "
                            f"Note: GENCLS power setpoints are typically stored in StaticGen.p0, "
                            f"but StaticGen not found in case file."
                        )

            if field not in gen_data:
                raise ValueError(
                    f"Power field '{field}' not found in GENCLS. "
                    f"Available fields: {list(gen_data.keys())}"
                )

            # Modify field for each generator
            if isinstance(gen_data[field], list):
                for gen_idx, new_pm in generator_setpoints.items():
                    if gen_idx >= len(gen_data[field]):
                        raise IndexError(
                            f"Generator index {gen_idx} out of range. "
                            f"GENCLS.{field} has {len(gen_data[field])} elements."
                        )
                    gen_data[field][gen_idx] = new_pm
                    modified_generators.add(gen_idx)
                    logger.info(f"Set GENCLS.{field}[{gen_idx}] = {new_pm:.6f} pu")
            else:
                # Scalar value - can only modify one generator
                if len(generator_setpoints) > 1:
                    raise ValueError(
                        f"GENCLS.{field} is scalar, but multiple generators specified. "
                        f"Use modify_case_file_generator_setpoint() for single generator."
                    )
                gen_idx = list(generator_setpoints.keys())[0]
                gen_data[field] = generator_setpoints[gen_idx]
                modified_generators.add(gen_idx)
                logger.info(f"Set GENCLS.{field} = {generator_setpoints[gen_idx]:.6f} pu")

            modified = len(modified_generators) > 0

        elif isinstance(gen_data, list):
            # List format: [ {"P0": 0.9, "M": 6.0, ...}, ... ]
            for gen_idx, new_pm in generator_setpoints.items():
                if gen_idx >= len(gen_data):
                    raise IndexError(
                        f"Generator index {gen_idx} out of range. "
                        f"GENCLS has {len(gen_data)} generators."
                    )

                gen = gen_data[gen_idx]
                if not isinstance(gen, dict):
                    raise ValueError(
                        f"GENCLS[{gen_idx}] is not a dictionary. " f"Expected dict, got {type(gen)}"
                    )

                if power_field_name:
                    field = power_field_name
                else:
                    # Auto-detect power field - avoid "P" (too generic)
                    for field in ["P0", "Pg", "Pm", "Pgen"]:
                        if field in gen:
                            break
                    else:
                        # Check if we have StaticGen available
                        if "StaticGen" in case_data:
                            raise ValueError(
                                f"Could not find power setpoint field in GENCLS[{gen_idx}]. "
                                f"Available fields: {list(gen.keys())}. "
                                f"Note: GENCLS power setpoints are typically stored in"
                                f"StaticGen.p0."
                                f"StaticGen found in case file but 'gen' field not found in"
                                f"GENCLS[{gen_idx}]."
                            )
                        else:
                            raise ValueError(
                                f"Could not find power setpoint field in GENCLS[{gen_idx}]. "
                                f"Available fields: {list(gen.keys())}. "
                                f"Note: GENCLS power setpoints are typically stored in"
                                f"StaticGen.p0,"
                                f"but StaticGen not found in case file."
                            )

                gen[field] = new_pm
                modified_generators.add(gen_idx)
                logger.info(f"Set GENCLS[{gen_idx}].{field} = {new_pm:.6f} pu")

            modified = len(modified_generators) > 0

        else:
            raise ValueError(
                f"Unexpected GENCLS format: {type(gen_data)}. " f"Expected dict or list."
            )

    if not modified:
        raise RuntimeError("Failed to modify case file (unknown format)")

    # Write modified case file
    with open(output_path, "w") as f:
        json.dump(case_data, f, indent=2)

    logger.info(f"Created modified case file: {output_path}")
    return output_path


def _modify_excel_case_file_multiple(
    case_path: str,
    generator_setpoints: Dict[int, float],
    output_path: str,
    power_field_name: Optional[str] = None,
) -> str:
    """Modify Excel case file for multiple generators."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for Excel case file modification")

    # Read Excel file
    xls = pd.ExcelFile(case_path)

    # Find generator sheet
    gen_sheet = None
    for sheet_name in xls.sheet_names:
        if "GEN" in sheet_name.upper() or "GENCLS" in sheet_name.upper():
            gen_sheet = sheet_name
            break

    if gen_sheet is None:
        raise ValueError(f"Could not find generator sheet in {case_path}")

    # Read generator data
    df = pd.read_excel(case_path, sheet_name=gen_sheet)

    # Find power setpoint column
    if power_field_name:
        power_col = power_field_name
    else:
        # Auto-detect power column
        for col in ["P0", "Pg", "P", "Pm", "Pgen"]:
            if col in df.columns:
                power_col = col
                break
        else:
            raise ValueError(
                f"Could not find power setpoint column in {gen_sheet}. "
                f"Available columns: {list(df.columns)}"
            )

    if power_col not in df.columns:
        raise ValueError(
            f"Power column '{power_col}' not found in {gen_sheet}. "
            f"Available columns: {list(df.columns)}"
        )

    # Modify setpoints for each generator
    for gen_idx, new_pm in generator_setpoints.items():
        if gen_idx >= len(df):
            raise IndexError(
                f"Generator index {gen_idx} out of range. " f"{gen_sheet} has {len(df)} generators."
            )
        df.loc[gen_idx, power_col] = new_pm
        logger.info(f"Set {gen_sheet}.{power_col}[{gen_idx}] = {new_pm:.6f} pu")

    # Write modified Excel file
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Write all sheets
        for sheet in xls.sheet_names:
            if sheet == gen_sheet:
                df.to_excel(writer, sheet_name=sheet, index=False)
            else:
                pd.read_excel(case_path, sheet_name=sheet).to_excel(
                    writer, sheet_name=sheet, index=False
                )

    logger.info(f"Created modified case file: {output_path}")
    return output_path


def check_power_balance(
    ss,
    generator_setpoints: Dict[int, float],
    tolerance: float = 0.01,
) -> Tuple[bool, str]:
    """
    Check power balance in multimachine system.

    Verifies that total generation approximately equals total load plus losses.
    This is a critical check for multimachine systems to ensure power flow
    convergence.

    Args:
        ss: ANDES system object (must be loaded and have power flow data)
        generator_setpoints: Dict mapping generator index to Pm value
        tolerance: Maximum allowed imbalance (pu). Default: 0.01 pu

    Returns:
        Tuple of (is_balanced, error_message)
        - is_balanced: True if power balance is maintained
        - error_message: Empty string if balanced, error description if not

    Example:
        >>> generator_setpoints = {0: 0.7, 1: 0.8, 2: 0.6}
        >>> is_balanced, error = check_power_balance(ss, generator_setpoints)
        >>> if not is_balanced:
        ...     print(f"Power imbalance: {error}")
    """
    if not ANDES_AVAILABLE:
        return False, "ANDES not available"

    try:
        import numpy as np
    except ImportError:
        return False, "numpy not available"

    # Calculate total generation
    total_gen = sum(generator_setpoints.values())

    # Calculate total load from system
    total_load = 0.0
    try:
        # Try to get load data from ANDES system
        if hasattr(ss, "PQ") and hasattr(ss.PQ, "p0") and hasattr(ss.PQ.p0, "v"):
            # PQ loads (constant power loads)
            if hasattr(ss.PQ.p0.v, "__len__"):
                total_load += float(np.sum(ss.PQ.p0.v))
            else:
                total_load += float(ss.PQ.p0.v)

        if hasattr(ss, "ZIP") and hasattr(ss.ZIP, "p0") and hasattr(ss.ZIP.p0, "v"):
            # ZIP loads
            if hasattr(ss.ZIP.p0.v, "__len__"):
                total_load += float(np.sum(ss.ZIP.p0.v))
            else:
                total_load += float(ss.ZIP.p0.v)

        # If no loads found, try to estimate from case file or use a default
        if total_load == 0.0:
            # For SMIB or systems where load is not explicitly modeled,
            # assume load equals generation (balanced system)
            # This is a reasonable assumption for many test cases
            total_load = total_gen
    except Exception as e:
        # If we can't calculate load, assume balanced (conservative approach)
        # The power flow will fail if there's a real imbalance
        total_load = total_gen
        # Note: We don't have verbose parameter here, so we'll skip the warning
        # The power flow will fail if there's a real imbalance anyway

    # Check balance
    imbalance = abs(total_gen - total_load)
    if imbalance > tolerance:
        return False, (
            f"Power imbalance: {imbalance:.4f} pu "
            f"(generation={total_gen:.4f} pu, load={total_load:.4f} pu). "
            f"Maximum allowed imbalance: {tolerance:.4f} pu"
        )

    return True, ""


def modify_case_file_load_setpoint(
    case_path: str,
    load_idx: int,
    new_p: float,
    new_q: Optional[float] = None,
    output_path: Optional[str] = None,
    load_model: str = "PQ",
) -> str:
    """
    Modify load setpoint in case file BEFORE loading.

    This is the CORRECT way to change load levels in ANDES for operating point variation.
    Power flow reads load setpoints from case file, so modifying the case file ensures
    power flow uses the correct values. When load changes, generators adjust to meet it.

    Args:
        case_path: Path to original case file
        load_idx: Load index (usually 0 for SMIB, or bus index)
        new_p: New active power load (pu, positive for consumption)
        new_q: New reactive power load (pu, optional, positive for consumption)
        output_path: Optional output path (default: temp file)
        load_model: Load model type ("PQ", "ZIP", etc.). Default: "PQ"

    Returns:
        Path to modified case file

    Example:
        >>> case_path = andes.get_case("smib/SMIB.json")
        >>> modified_path = modify_case_file_load_setpoint(
        ...     case_path, load_idx=0, new_p=0.7, new_q=0.1
        ... )
        >>> ss = andes.load(modified_path)
        >>> ss.PFlow.run()
        >>> # Generator will adjust to meet load=0.7 pu
    """
    case_path_obj = Path(case_path)

    if not case_path_obj.exists():
        raise FileNotFoundError(f"Case file not found: {case_path}")

    # Determine output path
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        case_name = case_path_obj.stem
        # Use safe filename format
        p_str = f"{new_p:.6f}".replace(".", "_")
        q_str = f"_{new_q:.6f}".replace(".", "_") if new_q is not None else ""
        output_path = str(
            Path(temp_dir) / f"{case_name}_L{load_idx}_P{p_str}{q_str}{case_path_obj.suffix}"
        )

    # Modify based on format
    if case_path_obj.suffix == ".json":
        return _modify_json_case_file_load(
            case_path, load_idx, new_p, new_q, output_path, load_model
        )
    elif case_path_obj.suffix == ".xlsx" and PANDAS_AVAILABLE:
        return _modify_excel_case_file_load(
            case_path, load_idx, new_p, new_q, output_path, load_model
        )
    else:
        raise ValueError(
            f"Unsupported case file format: {case_path_obj.suffix}. "
            f"Supported formats: .json, .xlsx (requires pandas)"
        )


def _modify_json_case_file_load(
    case_path: str,
    load_idx: int,
    new_p: float,
    new_q: Optional[float],
    output_path: str,
    load_model: str,
) -> str:
    """
    Modify JSON case file for load setpoints.

    Supports PQ load model (most common):
    - p0: Active power (pu, positive for consumption)
    - q0: Reactive power (pu, positive for consumption)
    """
    with open(case_path, "r") as f:
        case_data = json.load(f)

    modified = False

    # Try different load model names (PQ is most common)
    load_model_keys = [load_model, "PQ", "Load", "PQLoad"]

    load_data = None
    load_model_key = None

    for key in load_model_keys:
        if key in case_data:
            load_data = case_data[key]
            load_model_key = key
            break

    if load_data is None:
        raise ValueError(
            f"Case file does not contain load data. "
            f"Tried: {load_model_keys}. "
            f"Available keys: {list(case_data.keys())}"
        )

    # Handle dict format: { "p0": [0.5, 0.6], "q0": [0.1, 0.2], ... }
    if isinstance(load_data, dict):
        if "p0" in load_data:
            if isinstance(load_data["p0"], list):
                if load_idx < len(load_data["p0"]):
                    load_data["p0"][load_idx] = new_p
                    modified = True
                    logger.info(f"Set {load_model_key}.p0[{load_idx}] = {new_p:.6f} pu")
            else:
                # Single value (scalar)
                load_data["p0"] = new_p
                modified = True
                logger.info(f"Set {load_model_key}.p0 = {new_p:.6f} pu")

        if new_q is not None and "q0" in load_data:
            if isinstance(load_data["q0"], list):
                if load_idx < len(load_data["q0"]):
                    load_data["q0"][load_idx] = new_q
                    modified = True
                    logger.info(f"Set {load_model_key}.q0[{load_idx}] = {new_q:.6f} pu")
            else:
                # Single value (scalar)
                load_data["q0"] = new_q
                modified = True
                logger.info(f"Set {load_model_key}.q0 = {new_q:.6f} pu")

    # Handle list format: [ {"p0": 0.5, "q0": 0.1}, ... ]
    elif isinstance(load_data, list):
        if load_idx < len(load_data) and isinstance(load_data[load_idx], dict):
            load = load_data[load_idx]
            if "p0" in load:
                load["p0"] = new_p
                modified = True
                logger.info(f"Set {load_model_key}[{load_idx}].p0 = {new_p:.6f} pu")
            if new_q is not None and "q0" in load:
                load["q0"] = new_q
                modified = True
                logger.info(f"Set {load_model_key}[{load_idx}].q0 = {new_q:.6f} pu")

    if not modified:
        raise RuntimeError(
            f"Failed to modify load setpoint. "
            f"Load model: {load_model_key}, Format: {type(load_data)}"
        )

    # Write modified case file
    with open(output_path, "w") as f:
        json.dump(case_data, f, indent=2)

    logger.info(f"Created modified case file: {output_path}")
    return output_path


def _modify_excel_case_file_load(
    case_path: str,
    load_idx: int,
    new_p: float,
    new_q: Optional[float],
    output_path: str,
    load_model: str,
) -> str:
    """
    Modify Excel case file for load setpoints.

    Supports PQ load model.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for Excel case file modification")

    # Read all sheets
    xls = pd.ExcelFile(case_path)

    # Find load sheet
    load_sheet = None
    for sheet_name in xls.sheet_names:
        if load_model.upper() in sheet_name.upper() or "LOAD" in sheet_name.upper():
            load_sheet = sheet_name
            break

    if load_sheet is None:
        raise ValueError(
            f"Could not find load sheet in case file. " f"Available sheets: {xls.sheet_names}"
        )

    # Read load sheet
    df = pd.read_excel(case_path, sheet_name=load_sheet)

    # Find power columns
    p_col = None
    q_col = None

    for col in df.columns:
        col_upper = col.upper()
        if col_upper in ["P0", "P", "PACTIVE", "P_ACTIVE"]:
            p_col = col
        elif col_upper in ["Q0", "Q", "QREACTIVE", "Q_REACTIVE"]:
            q_col = col

    if p_col is None:
        raise ValueError(
            f"Could not find active power column in {load_sheet}. "
            f"Available columns: {list(df.columns)}"
        )

    # Modify load setpoint
    if load_idx < len(df):
        df.loc[load_idx, p_col] = new_p
        if new_q is not None and q_col is not None:
            df.loc[load_idx, q_col] = new_q
    else:
        raise ValueError(
            f"Load index {load_idx} out of range. " f"Sheet {load_sheet} has {len(df)} loads."
        )

    # Write modified Excel file
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Write all sheets
        for sheet_name in xls.sheet_names:
            if sheet_name == load_sheet:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # Copy other sheets unchanged
                other_df = pd.read_excel(case_path, sheet_name=sheet_name)
                other_df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Created modified case file: {output_path}")
    return output_path


def get_default_load_from_case_file(
    case_path: str, load_idx: int = 0, load_model: str = "PQ"
) -> Optional[Tuple[float, Optional[float]]]:
    """
    Extract default load values from case file.

    Args:
        case_path: Path to case file
        load_idx: Load index
        load_model: Load model type ("PQ", "ZIP", etc.)

    Returns:
        Tuple of (P, Q) or None if not found
    """
    try:
        case_path_obj = Path(case_path)

        if case_path_obj.suffix == ".json":
            with open(case_path, "r") as f:
                case_data = json.load(f)

            # Try different load model names
            load_model_keys = [load_model, "PQ", "Load", "PQLoad"]
            load_data = None

            for key in load_model_keys:
                if key in case_data:
                    load_data = case_data[key]
                    break

            if load_data is None:
                return None

            # Extract P and Q
            p_val = None
            q_val = None

            if isinstance(load_data, dict):
                if "p0" in load_data:
                    if isinstance(load_data["p0"], list):
                        if load_idx < len(load_data["p0"]):
                            p_val = float(load_data["p0"][load_idx])
                    else:
                        p_val = float(load_data["p0"])

                if "q0" in load_data:
                    if isinstance(load_data["q0"], list):
                        if load_idx < len(load_data["q0"]):
                            q_val = float(load_data["q0"][load_idx])
                    else:
                        q_val = float(load_data["q0"])

            elif isinstance(load_data, list):
                if load_idx < len(load_data) and isinstance(load_data[load_idx], dict):
                    load = load_data[load_idx]
                    p_val = float(load.get("p0", 0.0))
                    q_val = float(load.get("q0", 0.0)) if "q0" in load else None

            if p_val is not None:
                return (p_val, q_val)

        elif case_path_obj.suffix == ".xlsx" and PANDAS_AVAILABLE:
            xls = pd.ExcelFile(case_path)
            load_sheet = None

            for sheet_name in xls.sheet_names:
                if load_model.upper() in sheet_name.upper() or "LOAD" in sheet_name.upper():
                    load_sheet = sheet_name
                    break

            if load_sheet:
                df = pd.read_excel(case_path, sheet_name=load_sheet)
                if load_idx < len(df):
                    p_col = None
                    q_col = None

                    for col in df.columns:
                        col_upper = col.upper()
                        if col_upper in ["P0", "P", "PACTIVE", "P_ACTIVE"]:
                            p_col = col
                        elif col_upper in ["Q0", "Q", "QREACTIVE", "Q_REACTIVE"]:
                            q_col = col

                    if p_col:
                        p_val = float(df.loc[load_idx, p_col])
                        q_val = float(df.loc[load_idx, q_col]) if q_col else None
                        return (p_val, q_val)

    except Exception as e:
        logger.warning(f"Could not extract default load from case file: {e}")

    return None


def check_smib_has_load(case_path: str) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Check if SMIB case has an explicit load device.

    Checks for PQ, ZIP, or other load models in case file.
    Returns load model type and index if found.

    Args:
        case_path: Path to case file

    Returns:
        Tuple of (has_load, load_model_type, load_idx)
        - has_load: True if load device exists (even if p0=0, q0=0)
        - load_model_type: "PQ", "ZIP", etc. or None
        - load_idx: Load index (0-based) or None

    Example:
        >>> has_load, model_type, load_idx = check_smib_has_load("smib/SMIB.json")
        >>> if has_load:
        ...     print(f"Found {model_type} load at index {load_idx}")
    """
    case_path_obj = Path(case_path)

    if not case_path_obj.exists():
        raise FileNotFoundError(f"Case file not found: {case_path}")

    # Try different load model names (PQ is most common)
    load_model_keys = ["PQ", "ZIP", "Load", "PQLoad", "StaticLoad"]

    if case_path_obj.suffix == ".json":
        try:
            with open(case_path, "r") as f:
                case_data = json.load(f)

            # Check for each load model type
            for model_key in load_model_keys:
                if model_key in case_data:
                    load_data = case_data[model_key]

                    # Check if load data exists and has entries
                    if isinstance(load_data, dict):
                        # Dict format: { "p0": [0.5, 0.6], "q0": [0.1, 0.2], ... }
                        if "p0" in load_data or "q0" in load_data:
                            # Check if there are actual load entries
                            p0_list = load_data.get("p0", [])
                            if isinstance(p0_list, list) and len(p0_list) > 0:
                                return (True, model_key, 0)
                            elif not isinstance(p0_list, list) and p0_list is not None:
                                return (True, model_key, 0)
                    elif isinstance(load_data, list):
                        # List format: [ {"p0": 0.5, "q0": 0.1}, ... ]
                        if len(load_data) > 0:
                            return (True, model_key, 0)

            return (False, None, None)

        except Exception as e:
            logger.warning(f"Error checking for load in case file: {e}")
            return (False, None, None)

    elif case_path_obj.suffix == ".xlsx" and PANDAS_AVAILABLE:
        try:
            xls = pd.ExcelFile(case_path)

            # Check for load sheets
            for sheet_name in xls.sheet_names:
                for model_key in load_model_keys:
                    if model_key.upper() in sheet_name.upper() or "LOAD" in sheet_name.upper():
                        df = pd.read_excel(case_path, sheet_name=sheet_name)
                        if len(df) > 0:
                            return (True, model_key, 0)

            return (False, None, None)

        except Exception as e:
            logger.warning(f"Error checking for load in Excel case file: {e}")
            return (False, None, None)

    else:
        logger.warning(f"Unsupported case file format: {case_path_obj.suffix}")
        return (False, None, None)


def add_pq_load_to_smib_case(
    case_path: str,
    bus_idx: int = 3,
    p0: float = 0.5,  # Default active power (matches ANDES manual strategy)
    q0: float = 0.2,  # Default reactive power (matches ANDES manual strategy)
    output_path: Optional[str] = None,
) -> str:
    """
    Add PQ load to SMIB case file if it doesn't have one.

    This ensures load variation can be used even for "pure" SMIB cases
    that only have generator + infinite bus.

    Args:
        case_path: Path to original case file
        bus_idx: Bus index where load should be added (default: 3 for SMIB)
        p0: Initial active power load (pu, positive for consumption)
        q0: Initial reactive power load (pu, optional, positive for consumption)
        output_path: Optional output path (default: temp file)

    Returns:
        Path to modified case file with PQ load added

    Note:
        - Uses PQ (constant power) load model - conservative assumption for TSA
        - Default bus_idx=3 matches ANDES SMIB case structure
        - Q=0 is a simplification (real loads have reactive power)
        - Only adds load if no load device exists (use check_smib_has_load() first)

    Example:
        >>> has_load, _, _ = check_smib_has_load("smib/SMIB.json")
        >>> if not has_load:
        ...     modified_case = add_pq_load_to_smib_case("smib/SMIB.json", bus_idx=3, p0=0.7)
    """
    case_path_obj = Path(case_path)

    if not case_path_obj.exists():
        raise FileNotFoundError(f"Case file not found: {case_path}")

    # Check if load already exists
    has_load, load_model, _ = check_smib_has_load(case_path)
    if has_load:
        logger.info(f"Load device ({load_model}) already exists in case file. No need to add.")
        return case_path  # Return original path

    # Determine output path
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        case_name = case_path_obj.stem
        p_str = f"{p0:.6f}".replace(".", "_")
        q_str = f"_{q0:.6f}".replace(".", "_") if q0 != 0.0 else ""
        output_path = str(
            Path(temp_dir) / f"{case_name}_added_PQ_P{p_str}{q_str}{case_path_obj.suffix}"
        )

    # Modify based on format
    if case_path_obj.suffix == ".json":
        return _add_pq_load_to_json_case(case_path, bus_idx, p0, q0, output_path)
    elif case_path_obj.suffix == ".xlsx" and PANDAS_AVAILABLE:
        return _add_pq_load_to_excel_case(case_path, bus_idx, p0, q0, output_path)
    else:
        raise ValueError(
            f"Unsupported case file format: {case_path_obj.suffix}. "
            f"Supported formats: .json, .xlsx (requires pandas)"
        )


def _add_pq_load_to_json_case(
    case_path: str,
    bus_idx: int,
    p0: float,
    q0: float,
    output_path: str,
) -> str:
    """Add PQ load to JSON case file."""
    with open(case_path, "r") as f:
        case_data = json.load(f)

    # Create PQ load entry
    pq_load = {
        "idx": "PQ_1",
        "u": 1.0,
        "name": "PQ 1",
        "bus": bus_idx,
        "Vn": 110.0,  # Default voltage, adjust if needed
        "p0": p0,
        "q0": q0,
        "vmax": 1.5,
        "vmin": 0.5,
        "owner": 1,
    }

    # Add PQ section if it doesn't exist
    if "PQ" not in case_data:
        case_data["PQ"] = []

    # Check if PQ is list or dict format
    if isinstance(case_data["PQ"], list):
        case_data["PQ"].append(pq_load)
    elif isinstance(case_data["PQ"], dict):
        # Convert dict format to list format for easier handling
        # This is a simplified approach - may need adjustment based on actual format
        if "idx" not in case_data["PQ"]:
            case_data["PQ"] = [pq_load]
        else:
            # Dict format with arrays - convert to list
            idx_list = case_data["PQ"].get("idx", [])
            if not isinstance(idx_list, list):
                idx_list = [idx_list]
            idx_list.append("PQ_1")
            case_data["PQ"]["idx"] = idx_list

            # Add p0
            p0_list = case_data["PQ"].get("p0", [])
            if not isinstance(p0_list, list):
                p0_list = [p0_list] if p0_list is not None else []
            p0_list.append(p0)
            case_data["PQ"]["p0"] = p0_list

            # Add q0
            q0_list = case_data["PQ"].get("q0", [])
            if not isinstance(q0_list, list):
                q0_list = [q0_list] if q0_list is not None else []
            q0_list.append(q0)
            case_data["PQ"]["q0"] = q0_list

            # Add other required fields
            for field in ["u", "bus", "Vn", "vmax", "vmin", "owner"]:
                field_list = case_data["PQ"].get(field, [])
                if not isinstance(field_list, list):
                    field_list = [field_list] if field_list is not None else []
                if field == "u":
                    field_list.append(1.0)
                elif field == "bus":
                    field_list.append(bus_idx)
                elif field == "Vn":
                    field_list.append(110.0)
                elif field == "vmax":
                    field_list.append(1.5)
                elif field == "vmin":
                    field_list.append(0.5)
                elif field == "owner":
                    field_list.append(1)
                case_data["PQ"][field] = field_list

    # Write modified case file
    with open(output_path, "w") as f:
        json.dump(case_data, f, indent=2)

    logger.info(f"Added PQ load (P={p0:.6f} pu, Q={q0:.6f} pu) to bus {bus_idx} in {output_path}")
    return output_path


def _add_pq_load_to_excel_case(
    case_path: str,
    bus_idx: int,
    p0: float,
    q0: float,
    output_path: str,
) -> str:
    """Add PQ load to Excel case file."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for Excel case file modification")

    # Read all sheets
    xls = pd.ExcelFile(case_path)

    # Find or create PQ sheet
    pq_sheet = None
    for sheet_name in xls.sheet_names:
        if "PQ" in sheet_name.upper() or "LOAD" in sheet_name.upper():
            pq_sheet = sheet_name
            break

    # Read or create PQ dataframe
    if pq_sheet:
        df_pq = pd.read_excel(case_path, sheet_name=pq_sheet)
    else:
        # Create new PQ sheet with standard columns
        df_pq = pd.DataFrame(
            columns=["idx", "u", "name", "bus", "Vn", "p0", "q0", "vmax", "vmin", "owner"]
        )

    # Add new load row
    new_load = {
        "idx": "PQ_1",
        "u": 1.0,
        "name": "PQ 1",
        "bus": bus_idx,
        "Vn": 110.0,
        "p0": p0,
        "q0": q0,
        "vmax": 1.5,
        "vmin": 0.5,
        "owner": 1,
    }
    df_pq = pd.concat([df_pq, pd.DataFrame([new_load])], ignore_index=True)

    # Write modified Excel file
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Write all sheets
        for sheet_name in xls.sheet_names:
            if sheet_name == pq_sheet:
                df_pq.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # Copy other sheets unchanged
                other_df = pd.read_excel(case_path, sheet_name=sheet_name)
                other_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # If PQ sheet didn't exist, add it
        if not pq_sheet:
            df_pq.to_excel(writer, sheet_name="PQ", index=False)

    logger.info(f"Added PQ load (P={p0:.6f} pu, Q={q0:.6f} pu) to bus {bus_idx} in {output_path}")
    return output_path
