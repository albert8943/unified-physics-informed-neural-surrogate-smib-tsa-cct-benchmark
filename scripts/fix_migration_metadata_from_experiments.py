#!/usr/bin/env python
"""
Comprehensive fix for migrated metadata by cross-checking with original experiment configs.

This script:
1. Finds original experiment summary/config files
2. Extracts accurate generation parameters
3. Updates metadata files with correct information
4. Handles: sampling_strategy, time_step, n_samples_per_combination, etc.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import only what we need to avoid dependency issues
COMMON_DATA_DIR = PROJECT_ROOT / "data" / "common"


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    """Save JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _load_registry(registry_path: Path) -> dict:
    """Load registry JSON file."""
    if not registry_path.exists():
        return {}
    try:
        return load_json(registry_path)
    except Exception:
        return {}


def _save_registry(registry: dict, registry_path: Path):
    """Save registry JSON file."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(registry, registry_path)


def find_experiment_config(original_path: str) -> Optional[Dict]:
    """
    Find experiment summary/config file from original path.

    Parameters:
    -----------
    original_path : str
        Original path from metadata

    Returns:
    --------
    config : dict or None
        Experiment config if found
    """
    try:
        # Extract experiment ID from path
        path_parts = Path(original_path).parts
        exp_dir = None
        for i, part in enumerate(path_parts):
            if part.startswith("exp_") and len(part) > 4:
                exp_dir = PROJECT_ROOT / Path(*path_parts[: i + 1])
                break

        if exp_dir and exp_dir.exists():
            # Try experiment_summary.json first (most complete)
            summary_path = exp_dir / "experiment_summary.json"
            if summary_path.exists():
                summary = load_json(summary_path)
                return summary.get("config", summary)

            # Try reproducibility.json as fallback
            repro_path = exp_dir / "reproducibility.json"
            if repro_path.exists():
                repro = load_json(repro_path)
                return repro.get("config", repro)

    except Exception as e:
        print(f"[DEBUG] Error finding experiment config: {e}")

    return None


def extract_generation_config(exp_config: Dict) -> Optional[Dict]:
    """Extract generation config from experiment config."""
    # Try different possible paths
    if "data" in exp_config and "generation" in exp_config["data"]:
        return exp_config["data"]["generation"]
    elif "generation" in exp_config:
        return exp_config["generation"]
    return None


def fix_metadata_from_experiment(metadata_path: Path, dry_run: bool = False) -> Dict:
    """
    Fix metadata file by checking original experiment config.

    Returns:
    --------
    result : dict
        Dictionary with fix results
    """
    result = {
        "file": metadata_path.name,
        "updated": False,
        "changes": [],
        "errors": [],
    }

    try:
        metadata = load_json(metadata_path)

        # Try to find original experiment config
        original_path = metadata.get("original_path", "")
        if not original_path:
            result["errors"].append("No original_path in metadata")
            return result

        exp_config = find_experiment_config(original_path)
        if not exp_config:
            result["errors"].append("Could not find experiment config")
            return result

        # Extract generation config
        gen_config_exp = extract_generation_config(exp_config)
        if not gen_config_exp:
            result["errors"].append("Could not extract generation config from experiment")
            return result

        # Get current generation config from metadata
        gen_config_meta = metadata.get("generation_config", {})
        if not gen_config_meta:
            gen_config_meta = {}
            metadata["generation_config"] = gen_config_meta

        # Check and update each field
        updates = {}

        # 1. Sampling strategy
        exp_strategy = gen_config_exp.get("sampling_strategy")
        meta_strategy = gen_config_meta.get("sampling_strategy")
        if exp_strategy and exp_strategy != meta_strategy:
            updates["sampling_strategy"] = (meta_strategy, exp_strategy)

        # 2. Time step
        exp_time_step = gen_config_exp.get("time_step")
        meta_time_step = gen_config_meta.get("time_step")
        if exp_time_step is not None and exp_time_step != meta_time_step:
            updates["time_step"] = (meta_time_step, exp_time_step)

        # 3. Simulation time
        exp_sim_time = gen_config_exp.get("simulation_time")
        meta_sim_time = gen_config_meta.get("simulation_time")
        if exp_sim_time is not None and exp_sim_time != meta_sim_time:
            updates["simulation_time"] = (meta_sim_time, exp_sim_time)

        # 4. n_samples_per_combination
        exp_n_per_combo = gen_config_exp.get("n_samples_per_combination")
        meta_n_per_combo = gen_config_meta.get("n_samples_per_combination")
        if exp_n_per_combo is not None and exp_n_per_combo != meta_n_per_combo:
            updates["n_samples_per_combination"] = (meta_n_per_combo, exp_n_per_combo)

        # 5. use_cct_based_sampling
        exp_use_cct = gen_config_exp.get("use_cct_based_sampling")
        meta_use_cct = gen_config_meta.get("use_cct_based_sampling")
        if exp_use_cct is not None and exp_use_cct != meta_use_cct:
            updates["use_cct_based_sampling"] = (meta_use_cct, exp_use_cct)

        # 6. additional_clearing_time_offsets
        exp_offsets = gen_config_exp.get("additional_clearing_time_offsets")
        meta_offsets = gen_config_meta.get("additional_clearing_time_offsets")
        if exp_offsets is not None and exp_offsets != meta_offsets:
            updates["additional_clearing_time_offsets"] = (meta_offsets, exp_offsets)

        # 7. Parameter ranges - keep actual sampled ranges in metadata
        # But add intended ranges from config as additional info if missing
        # Note: Metadata should keep actual sampled ranges (what was generated)
        # but we can add intended_ranges from config for reference
        exp_param_ranges = gen_config_exp.get("parameter_ranges", {})
        meta_param_ranges = gen_config_meta.get("parameter_ranges", {})

        # Add intended parameter ranges from config if not present
        if "intended_parameter_ranges" not in gen_config_meta and exp_param_ranges:
            # Store intended ranges separately (for reference)
            updates["intended_parameter_ranges"] = (None, exp_param_ranges)

        # 8. Case file
        exp_case_file = gen_config_exp.get("case_file")
        meta_case_file = gen_config_meta.get("case_file")
        if exp_case_file and exp_case_file != meta_case_file:
            updates["case_file"] = (meta_case_file, exp_case_file)

        # 9. Fault parameters
        exp_fault = gen_config_exp.get("fault", {})
        meta_fault = gen_config_meta.get("fault", {})
        if exp_fault:
            fault_updates = {}
            for key in ["start_time", "bus", "reactance"]:
                if key in exp_fault and exp_fault[key] != meta_fault.get(key):
                    fault_updates[key] = (meta_fault.get(key), exp_fault[key])
            if fault_updates:
                updates["fault"] = fault_updates

        # Apply updates
        if not updates:
            return result  # No changes needed

        if dry_run:
            # Format changes for display
            for key, value in updates.items():
                if key == "parameter_ranges":
                    for param, (old_range, new_range) in value.items():
                        result["changes"].append(
                            f"{param} range: {old_range[:2]} -> {new_range[:2]}"
                        )
                elif key == "fault":
                    for fault_key, (old_fault_val, new_fault_val) in value.items():
                        result["changes"].append(
                            f"fault.{fault_key}: {old_fault_val} -> {new_fault_val}"
                        )
                else:
                    old_val, new_val = value
                    result["changes"].append(f"{key}: {old_val} -> {new_val}")
            return result

        # Apply updates to metadata
        for key, value in updates.items():
            if key == "parameter_ranges":
                # Update parameter ranges
                if "parameter_ranges" not in gen_config_meta:
                    gen_config_meta["parameter_ranges"] = {}
                for param, (old_range, new_range) in value.items():
                    gen_config_meta["parameter_ranges"][param] = new_range
                    result["changes"].append(f"{param} range: {old_range[:2]} -> {new_range[:2]}")
            elif key == "fault":
                # Update fault parameters
                if "fault" not in gen_config_meta:
                    gen_config_meta["fault"] = {}
                for fault_key, (old_fault_val, new_fault_val) in value.items():
                    gen_config_meta["fault"][fault_key] = new_fault_val
                    result["changes"].append(
                        f"fault.{fault_key}: {old_fault_val} -> {new_fault_val}"
                    )
            else:
                old_val, new_val = value
                gen_config_meta[key] = new_val
                result["changes"].append(f"{key}: {old_val} -> {new_val}")

        # Save updated metadata
        save_json(metadata, metadata_path)
        result["updated"] = True

        # Update registry if needed
        registry_path = COMMON_DATA_DIR / "registry.json"
        if registry_path.exists():
            registry = _load_registry(registry_path)
            fingerprint = metadata.get("data_fingerprint")
            if (
                fingerprint
                and "fingerprints" in registry
                and fingerprint in registry["fingerprints"]
            ):
                # Update registry entry with corrected values
                for key, value in updates.items():
                    if key == "parameter_ranges" or key == "fault":
                        continue  # Skip complex nested updates
                    old_val, new_val = value
                    if key not in registry["fingerprints"][fingerprint]:
                        registry["fingerprints"][fingerprint][key] = {}
                    registry["fingerprints"][fingerprint][key] = new_val
                _save_registry(registry, registry_path)

    except Exception as e:
        result["errors"].append(f"Error: {str(e)}")

    return result


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Fix migrated metadata by cross-checking with original experiment configs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be fixed without updating"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FIX MIGRATED METADATA FROM EXPERIMENT CONFIGS")
    print("=" * 70)
    print(f"Directory: {COMMON_DATA_DIR}")
    print(f"Dry run: {args.dry_run}")
    print()

    if not COMMON_DATA_DIR.exists():
        print(f"[ERROR] Common data directory not found: {COMMON_DATA_DIR}")
        return

    # Find all metadata files
    metadata_files = sorted(COMMON_DATA_DIR.glob("*_metadata.json"))
    print(f"Found {len(metadata_files)} metadata file(s)")
    print()

    if not metadata_files:
        print("No metadata files found.")
        return

    # Fix each file
    updated_count = 0
    skipped_count = 0
    error_count = 0
    total_changes = 0

    for metadata_path in metadata_files:
        result = fix_metadata_from_experiment(metadata_path, args.dry_run)

        if result["errors"]:
            print(f"[ERROR] {result['file']}")
            for error in result["errors"]:
                print(f"   {error}")
            error_count += 1
        elif result["changes"]:
            status = "[DRY RUN]" if args.dry_run else "[OK]"
            print(f"{status} {result['file']}")
            for change in result["changes"]:
                print(f"   {change}")
            updated_count += 1
            total_changes += len(result["changes"])
        else:
            skipped_count += 1

    print()
    print("=" * 70)
    print("FIX SUMMARY")
    print("=" * 70)
    print(f"Files updated: {updated_count}")
    print(f"Files skipped (no changes needed): {skipped_count}")
    print(f"Files with errors: {error_count}")
    print(f"Total changes: {total_changes}")
    print(f"Total files processed: {len(metadata_files)}")

    if args.dry_run:
        print()
        print("[NOTE] This was a dry run. Use without --dry-run to actually update metadata.")


if __name__ == "__main__":
    main()
