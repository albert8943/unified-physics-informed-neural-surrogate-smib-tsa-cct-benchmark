#!/usr/bin/env python
"""
Cross-check all data/common files with configs/experiments .yaml files to verify consistency.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
COMMON_DATA_DIR = PROJECT_ROOT / "data" / "common"
EXPERIMENT_CONFIGS_DIR = PROJECT_ROOT / "configs" / "experiments"
DATA_GEN_CONFIGS_DIR = PROJECT_ROOT / "configs" / "data_generation"


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict:
    """Load YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_param_range(param_range: List) -> Tuple[float, float]:
    """Normalize parameter range to (min, max)."""
    if isinstance(param_range, list) and len(param_range) >= 2:
        return (float(param_range[0]), float(param_range[1]))
    return (0.0, 0.0)


def compare_param_ranges(
    metadata_ranges: Dict, config_ranges: Dict, tolerance: float = 0.1
) -> Tuple[bool, List[str]]:
    """Compare parameter ranges between metadata and config.

    Uses a more lenient tolerance (0.1) since actual sampled values may differ
    from config ranges due to sampling strategies.
    """
    issues = []
    all_match = True

    for param in ["H", "D", "Pm"]:
        meta_key = param
        config_key = param

        if meta_key not in metadata_ranges:
            issues.append(f"Missing {param} in metadata")
            all_match = False
            continue

        if config_key not in config_ranges:
            issues.append(f"Missing {param} in config")
            all_match = False
            continue

        meta_range = normalize_param_range(metadata_ranges[meta_key])
        config_range = normalize_param_range(config_ranges[config_key])

        # Check if ranges match (within tolerance)
        # For min: metadata should be >= config - tolerance (allows slight expansion)
        # For max: metadata should be <= config + tolerance (allows slight expansion)
        min_diff = meta_range[0] - config_range[0]
        max_diff = meta_range[1] - config_range[1]

        if abs(min_diff) > tolerance or abs(max_diff) > tolerance:
            # Check if metadata range is within config range (more lenient)
            if (
                meta_range[0] < config_range[0] - tolerance
                or meta_range[1] > config_range[1] + tolerance
            ):
                issues.append(
                    f"{param} range mismatch: metadata {meta_range} vs config {config_range} (diff:"
                    f"min={min_diff:.3f}, max={max_diff:.3f})"
                )
                all_match = False
            # If within tolerance, it's just a warning
            elif abs(min_diff) > 0.05 or abs(max_diff) > 0.05:
                issues.append(
                    f"{param} range slight difference: metadata {meta_range} vs config"
                    f"{config_range} (diff: min={min_diff:.3f}, max={max_diff:.3f})"
                )

    return all_match, issues


def extract_config_data_generation(config: dict) -> Optional[dict]:
    """Extract data generation config from experiment config."""
    # Try different possible paths
    if "data" in config and "generation" in config["data"]:
        return config["data"]["generation"]
    elif "data_generation" in config:
        return config["data_generation"]
    elif "generation" in config:
        return config["generation"]
    return None


def find_matching_config(metadata: dict) -> Optional[Tuple[Path, dict, str]]:
    """Find matching config file for metadata.

    Returns: (config_path, config_dict, config_type) where config_type is 'experiment' or 'data_generation'
    """
    gen_config = metadata.get("generation_config", {})
    meta_ranges = gen_config.get("parameter_ranges", {})

    if not meta_ranges:
        return None

    # Try both experiment and data_generation configs
    config_dirs = [
        (EXPERIMENT_CONFIGS_DIR, "experiment"),
        (DATA_GEN_CONFIGS_DIR, "data_generation"),
    ]

    best_match = None
    best_score = 0

    for configs_dir, config_type in config_dirs:
        if not configs_dir.exists():
            continue

        config_files = list(configs_dir.glob("*.yaml"))
        for config_path in config_files:
            try:
                config = load_yaml(config_path)
                config_gen = extract_config_data_generation(config)

                if not config_gen:
                    continue

                config_ranges = config_gen.get("parameter_ranges", {})
                if not config_ranges:
                    continue

                # Score based on matching ranges (more lenient)
                score = 0
                for param in ["H", "D", "Pm"]:
                    if param in meta_ranges and param in config_ranges:
                        meta_range = normalize_param_range(meta_ranges[param])
                        config_range = normalize_param_range(config_ranges[param])
                        # More lenient matching (within 0.1)
                        if (
                            abs(meta_range[0] - config_range[0]) < 0.1
                            and abs(meta_range[1] - config_range[1]) < 0.1
                        ):
                            score += 1

                # Bonus points for matching n_samples
                meta_n_samples = metadata.get("n_samples")
                config_n_samples = config_gen.get("n_samples")
                if meta_n_samples and config_n_samples and meta_n_samples == config_n_samples:
                    score += 2  # Strong indicator of match

                if score > best_score:
                    best_score = score
                    best_match = (config_path, config, config_type)

            except Exception as e:
                # Silently continue on errors
                continue

    return best_match if best_score > 0 else None


def verify_data_file(metadata_path: Path) -> Dict:
    """Verify a single data file against configs."""
    result = {
        "file": metadata_path.name,
        "matched_config": None,
        "issues": [],
        "warnings": [],
        "verified": False,
    }

    try:
        metadata = load_json(metadata_path)
    except Exception as e:
        result["issues"].append(f"Error loading metadata: {e}")
        return result

    # Extract key information
    gen_config = metadata.get("generation_config", {})
    meta_ranges = gen_config.get("parameter_ranges", {})
    n_samples_meta = metadata.get("n_samples")
    filename = metadata.get("filename", "")

    # Try to find matching config
    match = find_matching_config(metadata)
    if match:
        config_path, config, config_type = match
        result["matched_config"] = f"{config_path.name} ({config_type})"

        config_gen = extract_config_data_generation(config)
        if config_gen:
            config_ranges = config_gen.get("parameter_ranges", {})

            # Compare parameter ranges
            ranges_match, range_issues = compare_param_ranges(meta_ranges, config_ranges)
            result["issues"].extend(range_issues)

            # Check n_samples if available in config
            config_n_samples = config_gen.get("n_samples")
            if config_n_samples and n_samples_meta:
                if config_n_samples != n_samples_meta:
                    result["warnings"].append(
                        f"n_samples mismatch: metadata={n_samples_meta}, config={config_n_samples}"
                    )

            # Check sampling strategy
            meta_strategy = gen_config.get("sampling_strategy")
            config_strategy = config_gen.get("sampling_strategy")
            if meta_strategy and config_strategy and meta_strategy != config_strategy:
                result["warnings"].append(
                    f"Sampling strategy mismatch: metadata={meta_strategy},"
                    f"config={config_strategy}"
                )

            # Check simulation parameters
            for param in ["simulation_time", "time_step"]:
                meta_val = gen_config.get(param)
                config_val = config_gen.get(param)
                if meta_val and config_val and abs(meta_val - config_val) > 0.0001:
                    result["warnings"].append(
                        f"{param} mismatch: metadata={meta_val}, config={config_val}"
                    )

            result["verified"] = len(result["issues"]) == 0
        else:
            result["issues"].append("Could not extract generation config from matched config file")
    else:
        result["warnings"].append("No matching config file found")

    return result


def main():
    """Main function."""
    print("=" * 70)
    print("CROSS-CHECK DATA FILES WITH CONFIG FILES")
    print("=" * 70)
    print(f"Data directory: {COMMON_DATA_DIR}")
    print(f"Experiment configs: {EXPERIMENT_CONFIGS_DIR}")
    print(f"Data generation configs: {DATA_GEN_CONFIGS_DIR}")
    print()

    if not COMMON_DATA_DIR.exists():
        print(f"[ERROR] Data directory not found: {COMMON_DATA_DIR}")
        return

    if not EXPERIMENT_CONFIGS_DIR.exists() and not DATA_GEN_CONFIGS_DIR.exists():
        print(f"[ERROR] No config directories found")
        return

    # Find all metadata files
    metadata_files = sorted(COMMON_DATA_DIR.glob("*_metadata.json"))
    print(f"Found {len(metadata_files)} data files to verify")
    print()

    # List available config files
    exp_configs = (
        sorted(EXPERIMENT_CONFIGS_DIR.glob("*.yaml")) if EXPERIMENT_CONFIGS_DIR.exists() else []
    )
    data_gen_configs = (
        sorted(DATA_GEN_CONFIGS_DIR.glob("*.yaml")) if DATA_GEN_CONFIGS_DIR.exists() else []
    )
    print(f"Available experiment config files ({len(exp_configs)}):")
    for cf in exp_configs:
        print(f"  - {cf.name}")
    print(f"Available data generation config files ({len(data_gen_configs)}):")
    for cf in data_gen_configs:
        print(f"  - {cf.name}")
    print()

    # Verify each file
    results = []
    for metadata_path in metadata_files:
        result = verify_data_file(metadata_path)
        results.append(result)

    # Print results
    print("=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    print()

    verified_count = 0
    warning_count = 0
    error_count = 0

    for result in results:
        status = "[OK]" if result["verified"] else "[ISSUES]"
        if result["issues"]:
            status = "[ERROR]"
        elif result["warnings"]:
            status = "[WARNING]"

        print(f"{status} {result['file']}")
        if result["matched_config"]:
            print(f"   Matched config: {result['matched_config']}")
        else:
            print(f"   Matched config: None")

        if result["issues"]:
            for issue in result["issues"]:
                print(f"   [ERROR] {issue}")
            error_count += 1
        elif result["warnings"]:
            for warning in result["warnings"]:
                print(f"   [WARNING] {warning}")
            warning_count += 1
        else:
            verified_count += 1

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(results)}")
    print(f"[OK] Verified: {verified_count}")
    print(f"[WARNING] Warnings: {warning_count}")
    print(f"[ERROR] Errors: {error_count}")
    print()

    if error_count > 0:
        print("[NOTE] Files with errors may have:")
        print("  - Parameter range mismatches")
        print("  - Missing or incomplete metadata")
        print("  - No matching config file found")
        print()
        print("  These may be from older experiments or custom configurations.")
        sys.exit(1)


if __name__ == "__main__":
    main()
