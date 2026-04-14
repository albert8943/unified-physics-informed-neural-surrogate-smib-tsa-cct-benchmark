#!/usr/bin/env python
"""
Stage 1: Kundur two-area ANDES data generation only.

Generates trajectory data for the Kundur system and saves to data/multimachine/kundur/.
Use this before any ML/PINN training. See docs/multimachine_case_studies/KUNDUR_EXPERIMENT_PLAN.md.

Usage:
    python scripts/run_kundur_stage1_data_generation.py
    python scripts/run_kundur_stage1_data_generation.py --config configs/publication/kundur_2area.yaml --output data/multimachine/kundur --force
    python scripts/run_kundur_stage1_data_generation.py --validate-only  # Only run quality check on existing data
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run_quality_check(data_dir: Path) -> bool:
    """Quick quality check: load latest CSV, report rows, scenarios, NaNs, basic stats."""
    csv_files = sorted(
        data_dir.glob("parameter_sweep_data_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not csv_files:
        print(f"No parameter_sweep_data_*.csv found in {data_dir}")
        return False
    import pandas as pd

    path = csv_files[0]
    df = pd.read_csv(path, nrows=100000)
    n_rows = len(df)
    n_nan = df.isna().sum().sum()
    scenario_col = "scenario_id" if "scenario_id" in df.columns else None
    n_scenarios = df[scenario_col].nunique() if scenario_col else None
    print(f"File: {path.name}")
    print(f"  Rows: {n_rows:,}")
    print(f"  Scenarios: {n_scenarios:,}" if n_scenarios is not None else "  (no scenario_id)")
    print(f"  NaNs: {n_nan}")
    if n_nan > 0:
        print("  WARNING: Data contains NaNs.")
    return n_nan == 0


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Kundur two-area ANDES data generation (and optional quality check)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/publication/kundur_2area.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/multimachine/kundur",
        help="Output directory for generated CSV",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing data")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run quality check on existing data in --output",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        print("Quality check (validate-only) on:", output_path)
        ok = _run_quality_check(output_path)
        sys.exit(0 if ok else 1)

    # Delegate to existing multimachine data generation script (it resolves relative paths)
    import scripts.generate_multimachine_data as gen_mod

    argv_orig = sys.argv
    sys.argv = [
        "generate_multimachine_data.py",
        "--config",
        args.config,
        "--output",
        str(output_path),
    ]
    if args.force:
        sys.argv.append("--force")
    try:
        gen_mod.main()
    finally:
        sys.argv = argv_orig

    print("\nStage 1 complete. Running quick quality check...")
    if _run_quality_check(output_path):
        print("Quality check passed.")
    else:
        print("Quality check reported issues; review data before Stage 2.")


if __name__ == "__main__":
    main()
