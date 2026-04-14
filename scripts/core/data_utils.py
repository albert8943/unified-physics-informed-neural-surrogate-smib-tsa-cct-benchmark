"""
Data Selection Utilities

Helper functions for finding and selecting data files across the workflow.
Supports:
- Finding latest files in standard directories
- Selecting specific files by path or timestamp
- Using data from different sources (local, Colab, etc.)
- Finding data in common repository by fingerprint or parameters
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.common_repository import (
    COMMON_DATA_DIR,
    find_data_by_fingerprint,
    find_data_by_params,
    find_latest_data,
)


def find_latest_data_file(
    data_dir: Optional[Path] = None,
    pattern: str = "parameter_sweep_data_*.csv",
    level: Optional[str] = None,
    use_common_repository: bool = True,
) -> Path:
    """
    Find the latest data file in a directory.

    Parameters
    ----------
    data_dir : Path, optional
        Directory to search. If None, searches standard locations or common repository.
    pattern : str
        File pattern to match (default: "parameter_sweep_data_*.csv", for legacy directories)
    level : str, optional
        Data level: "quick", "moderate", "comprehensive". Used if data_dir is None (legacy).
    use_common_repository : bool
        If True, check common repository first (default: True)

    Returns
    -------
    Path
        Path to latest data file

    Raises
    ------
    FileNotFoundError
        If no matching files found
    """
    # Check common repository first if enabled and data_dir not specified
    if use_common_repository and data_dir is None and COMMON_DATA_DIR.exists():
        # Try to find latest data in common repository
        for task in ["trajectory", "parameter_estimation"]:
            latest = find_latest_data(task)
            if latest:
                return latest

    # Legacy: use old directory structure
    if data_dir is None:
        # Default to quick_test if level not specified
        if level is None:
            level = "quick"
        data_dir = PROJECT_ROOT / "data" / "generated" / f"{level}_test"
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = list(data_dir.glob(pattern))
    if not csv_files:
        # Try searching in parent directory
        parent_csv_files = list(data_dir.parent.glob(pattern))
        if parent_csv_files:
            csv_files = parent_csv_files
        else:
            raise FileNotFoundError(
                f"No data files found matching '{pattern}' in {data_dir} or {data_dir.parent}"
            )

    # Sort by modification time (most recent first)
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    return latest_file


def find_data_file(
    data_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    level: Optional[str] = None,
    pattern: str = "parameter_sweep_data_*.csv",
    task: Optional[str] = None,
    use_common_repository: bool = True,
) -> Path:
    """
    Find data file using flexible selection logic.

    Priority:
    1. Explicit data_path if provided
    2. Latest file in data_dir if provided
    3. Common repository search (if use_common_repository=True)
    4. Latest file in standard location based on level
    5. Latest file in quick_test (default)

    Parameters
    ----------
    data_path : str, optional
        Explicit path to data file
    data_dir : str, optional
        Directory to search for latest file
    level : str, optional
        Data level: "quick", "moderate", "comprehensive" (legacy, not used in common repo)
    pattern : str
        File pattern to match (for legacy directories)
    task : str, optional
        Task type for common repository search: "trajectory" or "parameter_estimation"
    use_common_repository : bool
        If True, search common repository first (default: True)

    Returns
    -------
    Path
        Path to data file

    Raises
    ------
    FileNotFoundError
        If file not found
    """
    # Priority 1: Explicit path
    if data_path:
        path = Path(data_path)
        if path.exists():
            return path
        # Try expanding glob pattern
        if "*" in str(path):
            matches = list(path.parent.glob(path.name))
            if matches:
                # Use latest if multiple matches
                return max(matches, key=lambda p: p.stat().st_mtime)
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Priority 2: Directory provided
    if data_dir:
        return find_latest_data_file(data_dir=Path(data_dir), pattern=pattern)

    # Priority 3: Common repository search
    if use_common_repository and COMMON_DATA_DIR.exists():
        if task:
            # Search by task
            latest = find_latest_data(task)
            if latest:
                return latest
        else:
            # Try both tasks
            for t in ["trajectory", "parameter_estimation"]:
                latest = find_latest_data(t)
                if latest:
                    return latest

    # Priority 4: Level-based search (legacy)
    if level:
        return find_latest_data_file(level=level, pattern=pattern)

    # Priority 5: Default to quick_test (legacy)
    return find_latest_data_file(level="quick", pattern=pattern)


def list_available_data_files(
    data_dir: Optional[Path] = None,
    level: Optional[str] = None,
    pattern: str = "parameter_sweep_data_*.csv",
) -> list[Path]:
    """
    List all available data files.

    Parameters
    ----------
    data_dir : Path, optional
        Directory to search
    level : str, optional
        Data level: "quick", "moderate", "comprehensive"
    pattern : str
        File pattern to match

    Returns
    -------
    list[Path]
        List of data files, sorted by modification time (newest first)
    """
    if data_dir is None:
        if level is None:
            level = "quick"
        data_dir = PROJECT_ROOT / "data" / "generated" / f"{level}_test"
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        return []

    csv_files = list(data_dir.glob(pattern))
    # Sort by modification time (newest first)
    return sorted(csv_files, key=lambda p: p.stat().st_mtime, reverse=True)


def get_data_file_info(data_path: Path) -> dict:
    """
    Get information about a data file.

    Parameters
    ----------
    data_path : Path
        Path to data file

    Returns
    -------
    dict
        Dictionary with file information:
        - path: Full path
        - name: Filename
        - size_mb: Size in MB
        - modified: Modification time
        - timestamp: Extracted timestamp from filename (if available)
    """
    import pandas as pd
    from datetime import datetime

    stat = data_path.stat()
    info = {
        "path": str(data_path),
        "name": data_path.name,
        "size_mb": stat.st_size / (1024 * 1024),
        "modified": datetime.fromtimestamp(stat.st_mtime),
    }

    # Try to extract timestamp from filename
    # Format: parameter_sweep_data_YYYYMMDD_HHMMSS.csv
    if "_" in data_path.stem:
        parts = data_path.stem.split("_")
        if len(parts) >= 4:
            try:
                date_str = parts[-2]  # YYYYMMDD
                time_str = parts[-1]  # HHMMSS
                timestamp_str = f"{date_str}_{time_str}"
                info["timestamp"] = timestamp_str
            except (IndexError, ValueError):
                info["timestamp"] = None
        else:
            info["timestamp"] = None
    else:
        info["timestamp"] = None

    # Try to get row count (quick check)
    try:
        df = pd.read_csv(data_path, nrows=0)  # Just read header
        info["columns"] = len(df.columns)
    except Exception:
        info["columns"] = None

    return info
