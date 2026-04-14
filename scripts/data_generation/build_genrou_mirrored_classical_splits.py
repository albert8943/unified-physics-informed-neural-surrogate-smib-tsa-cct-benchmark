#!/usr/bin/env python3
"""
Build a GENROU trajectory dataset whose train/val/test **scenario_id** sets match a classical
processed dataset, then apply the same preprocessing steps as ``scripts/preprocess_data.py``
(angle filter, optional feature engineering, optional normalization) **before** assigning rows
to splits (so the pipeline order matches the classical script: filter → engineer → normalize →
split). Here the split is **deterministic**: rows go to train/val/test by membership in the
classical CSVs, not by ``train_test_split``.

Workflow
--------
1. Load classical ``train_data_*.csv``, ``val_data_*.csv``, ``test_data_*.csv`` (or pass paths).
2. Build one reference row per ``scenario_id`` (union of splits) for H, D, Pm, tf, tc.
3. Run GENROU for each scenario (same as ``generate_genrou_trajectory_dataset.py``).
4. Preprocess the combined GENROU frame like ``preprocess_data.py`` (no random split).
5. Write ``train_data_*.csv``, ``val_data_*.csv``, ``test_data_*.csv`` under
   ``<output_base_dir>/exp_<timestamp>/``.

Example (repo root, Windows)::

    python scripts/data_generation/build_genrou_mirrored_classical_splits.py ^
      --classical-processed-dir data/processed/exp_20260211_190612 ^
      --filter-angles --max-angle-deg 360 --stability-threshold-deg 180

Reuse an existing combined GENROU raw CSV (skip ANDES)::

    python scripts/data_generation/build_genrou_mirrored_classical_splits.py ^
      --classical-processed-dir data/processed/exp_20260211_190612 ^
      --raw-genrou-csv data/raw/trajectory_data_genrou_20260408.csv ^
      --filter-angles --max-angle-deg 360 --stability-threshold-deg 180
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.preprocessing import (  # noqa: E402
    engineer_features,
    map_multimachine_to_smib_columns,
    normalize_data,
)
from scripts.core.utils import (  # noqa: E402
    generate_experiment_id,
    generate_timestamped_filename,
    save_json,
)
from scripts.data_generation.generate_genrou_trajectory_dataset import (  # noqa: E402
    generate_genrou_from_scenario_table,
)

try:
    from utils.angle_filter import filter_trajectory_by_angle

    ANGLE_FILTER_AVAILABLE = True
except ImportError:
    ANGLE_FILTER_AVAILABLE = False
    filter_trajectory_by_angle = None


def _latest_csv(directory: Path, glob_pattern: str) -> Path:
    matches = list(directory.glob(glob_pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {glob_pattern!r} under {directory}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def classical_reference_and_split_ids(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
) -> Tuple[pd.DataFrame, Set[Any], Set[Any], Set[Any]]:
    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    test = pd.read_csv(test_csv)
    for label, df in ("train", train), ("val", val), ("test", test):
        if "scenario_id" not in df.columns:
            raise ValueError(f"{label} CSV missing column 'scenario_id': {df.columns.tolist()}")

    train_ids = set(train["scenario_id"].unique())
    val_ids = set(val["scenario_id"].unique())
    test_ids = set(test["scenario_id"].unique())

    for a, b, la, lb in (
        (train_ids, val_ids, "train", "val"),
        (train_ids, test_ids, "train", "test"),
        (val_ids, test_ids, "val", "test"),
    ):
        inter = a & b
        if inter:
            raise ValueError(f"Overlapping scenario_id between {la} and {lb}: {sorted(inter)[:20]}")

    combined = pd.concat([train, val, test], ignore_index=True)
    ref = combined.sort_values("time").groupby("scenario_id", sort=False).first().reset_index()
    ref = ref.sort_values("scenario_id").reset_index(drop=True)
    union = train_ids | val_ids | test_ids
    if set(ref["scenario_id"]) != union:
        missing = union - set(ref["scenario_id"])
        extra = set(ref["scenario_id"]) - union
        raise ValueError(f"Reference table mismatch: missing={missing}, extra={extra}")

    return ref, train_ids, val_ids, test_ids


def preprocess_like_preprocess_script(
    df: pd.DataFrame,
    *,
    filter_angles: bool,
    max_angle_deg: float,
    stability_threshold_deg: float,
    engineer_features_flag: bool,
    normalize: bool,
    normalization_method: str,
) -> Tuple[pd.DataFrame, Optional[Dict], Optional[dict]]:
    """Same order as ``scripts/preprocess_data.py``: map → filter → engineer → normalize."""
    out = map_multimachine_to_smib_columns(df.copy())
    filter_stats = None
    if filter_angles:
        if not ANGLE_FILTER_AVAILABLE:
            print("[WARNING] angle_filter not available; skipping --filter-angles.")
        elif "delta" not in out.columns:
            print("[WARNING] no 'delta' column; skipping --filter-angles.")
        else:
            out, filter_stats = filter_trajectory_by_angle(
                data=out,
                max_angle_deg=max_angle_deg,
                stability_threshold_deg=stability_threshold_deg,
            )
    if engineer_features_flag:
        out = engineer_features(out)
    scaler_dict = None
    if normalize:
        out, scaler_dict = normalize_data(out, method=normalization_method)
    return out, scaler_dict, filter_stats


def main() -> None:
    p = argparse.ArgumentParser(
        description="GENROU trajectories + classical-mirrored splits + preprocess_data-equivalent steps"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--classical-processed-dir",
        type=Path,
        help="Directory containing train_data_*.csv, val_data_*.csv, test_data_*.csv (latest match each)",
    )
    g.add_argument(
        "--classical-csv-triple",
        nargs=3,
        metavar=("TRAIN_CSV", "VAL_CSV", "TEST_CSV"),
        help="Explicit paths to classical train, val, test CSVs",
    )
    p.add_argument(
        "--output-base-dir",
        type=Path,
        default=None,
        help="Parent folder for exp_<timestamp> output (default: <classical_dir>_genrou next to classical)",
    )
    p.add_argument(
        "--genrou-case",
        type=str,
        default="test_cases/SMIB_genrou.json",
        help="ANDES GENROU case (path or andes.get_case name)",
    )
    p.add_argument(
        "--raw-genrou-csv",
        type=Path,
        default=None,
        help="If set, load this combined GENROU CSV instead of running simulations",
    )
    p.add_argument(
        "--save-raw-genrou-csv",
        type=Path,
        default=None,
        help="After simulation, also write combined raw GENROU CSV to this path",
    )
    p.add_argument("--max-scenarios", type=int, default=None, help="Limit scenarios (smoke test)")
    p.add_argument(
        "--no-fold-delta",
        action="store_true",
        help="Pass through to GENROU generation (stored delta not principal-valued).",
    )
    p.add_argument("--filter-angles", action="store_true")
    p.add_argument("--max-angle-deg", type=float, default=360.0)
    p.add_argument("--stability-threshold-deg", type=float, default=180.0)
    p.add_argument("--engineer-features", action="store_true")
    p.add_argument("--normalize", action="store_true")
    p.add_argument(
        "--normalization-method",
        type=str,
        choices=["standard", "minmax"],
        default="standard",
    )
    args = p.parse_args()

    if args.classical_csv_triple is not None:
        train_csv, val_csv, test_csv = (Path(x).resolve() for x in args.classical_csv_triple)
        classical_dir = train_csv.parent
    else:
        classical_dir = args.classical_processed_dir.resolve()
        train_csv = _latest_csv(classical_dir, "train_data_*.csv")
        val_csv = _latest_csv(classical_dir, "val_data_*.csv")
        test_csv = _latest_csv(classical_dir, "test_data_*.csv")

    for path, label in (train_csv, "train"), (val_csv, "val"), (test_csv, "test"):
        if not path.is_file():
            print(f"Error: {label} CSV not found: {path}", file=sys.stderr)
            sys.exit(1)

    ref, train_ids, val_ids, test_ids = classical_reference_and_split_ids(
        train_csv, val_csv, test_csv
    )
    train_ids_full = set(train_ids)
    val_ids_full = set(val_ids)
    test_ids_full = set(test_ids)
    classical_union = train_ids_full | val_ids_full | test_ids_full

    if args.output_base_dir is not None:
        output_base = Path(args.output_base_dir).resolve()
    else:
        output_base = classical_dir.parent / f"{classical_dir.name}_genrou"
    experiment_id = generate_experiment_id()
    output_dir = output_base / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENROU + classical-mirrored splits")
    print("=" * 70)
    print(f"Classical dir: {classical_dir}")
    print(f"Train CSV: {train_csv.name} ({len(train_ids_full)} scenarios)")
    print(f"Val CSV:   {val_csv.name} ({len(val_ids_full)} scenarios)")
    print(f"Test CSV:  {test_csv.name} ({len(test_ids_full)} scenarios)")
    print(f"Output:    {output_dir}")
    print()

    failed: list = []
    if args.raw_genrou_csv is not None:
        raw_path = args.raw_genrou_csv.resolve()
        if not raw_path.is_file():
            print(f"Error: --raw-genrou-csv not found: {raw_path}", file=sys.stderr)
            sys.exit(1)
        combined = pd.read_csv(raw_path)
        print(f"Loaded raw GENROU: {raw_path} ({len(combined):,} rows)")
    else:
        scenarios = ref.copy()
        if args.max_scenarios is not None:
            scenarios = scenarios.iloc[: int(args.max_scenarios)].copy()
        fold_delta = not args.no_fold_delta
        print(f"Simulating GENROU for {len(scenarios)} scenarios (fold_delta={fold_delta})...")
        combined, failed = generate_genrou_from_scenario_table(
            scenarios,
            args.genrou_case,
            fold_delta=fold_delta,
            max_scenarios=None,
        )
        if combined.empty:
            print("Error: GENROU generation produced no rows.", file=sys.stderr)
            sys.exit(2)
        if failed:
            print(f"Warning: {len(failed)} scenario(s) failed during GENROU: {failed[:15]}...")
        if args.save_raw_genrou_csv is not None:
            raw_out = Path(args.save_raw_genrou_csv).resolve()
            raw_out.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(raw_out, index=False)
            print(f"Wrote combined raw GENROU -> {raw_out}")

    if "scenario_id" not in combined.columns:
        print("Error: GENROU data must contain 'scenario_id'.", file=sys.stderr)
        sys.exit(2)

    sim_ids = set(combined["scenario_id"].unique())
    partial = args.max_scenarios is not None

    if partial:
        print(
            f"Partial run (--max-scenarios={args.max_scenarios}): "
            "restricting outputs to simulated scenarios that appear in classical splits."
        )
        train_ids = train_ids_full & sim_ids
        val_ids = val_ids_full & sim_ids
        test_ids = test_ids_full & sim_ids
        planned = set(ref["scenario_id"].iloc[: int(args.max_scenarios)].unique())  # type: ignore[index]
        missing_runs = planned - sim_ids
        if missing_runs:
            print(
                f"Error: GENROU failed for scenario_id(s): {sorted(missing_runs)[:40]}",
                file=sys.stderr,
            )
            sys.exit(3)
        extra = sim_ids - classical_union
        if extra:
            print(
                f"Note: trimming {len(extra)} scenario(s) not in classical union: {sorted(extra)[:10]}..."
            )
            combined = combined[combined["scenario_id"].isin(classical_union)].copy()
            sim_ids = set(combined["scenario_id"].unique())
    else:
        missing = classical_union - sim_ids
        if missing:
            print(
                f"Error: GENROU data missing scenario_id values: {sorted(missing)[:30]}...",
                file=sys.stderr,
            )
            sys.exit(3)
        if failed:
            print(
                f"Error: {len(failed)} scenario(s) failed during GENROU (full run requires all): "
                f"{failed[:30]}",
                file=sys.stderr,
            )
            sys.exit(3)
        extra = sim_ids - classical_union
        if extra:
            print(
                f"Note: trimming {len(extra)} scenario(s) not in classical union: {sorted(extra)[:10]}..."
            )
            combined = combined[combined["scenario_id"].isin(classical_union)].copy()

    print("Preprocessing (preprocess_data.py order, mirrored splits)...")
    processed, scaler_dict, filter_stats = preprocess_like_preprocess_script(
        combined,
        filter_angles=args.filter_angles,
        max_angle_deg=args.max_angle_deg,
        stability_threshold_deg=args.stability_threshold_deg,
        engineer_features_flag=args.engineer_features,
        normalize=args.normalize,
        normalization_method=args.normalization_method,
    )

    train_data = processed[processed["scenario_id"].isin(train_ids)].copy()
    val_data = processed[processed["scenario_id"].isin(val_ids)].copy()
    test_data = processed[processed["scenario_id"].isin(test_ids)].copy()

    def _nuniq(d: pd.DataFrame) -> int:
        return d["scenario_id"].nunique() if len(d) else 0

    if (
        _nuniq(train_data) != len(train_ids)
        or _nuniq(val_data) != len(val_ids)
        or _nuniq(test_data) != len(test_ids)
    ):
        print(
            "Error: after preprocessing, scenario counts do not match classical splits.\n"
            f"  train scenarios: {_nuniq(train_data)} vs {len(train_ids)}\n"
            f"  val scenarios:   {_nuniq(val_data)} vs {len(val_ids)}\n"
            f"  test scenarios:  {_nuniq(test_data)} vs {len(test_ids)}",
            file=sys.stderr,
        )
        sys.exit(4)

    ts_token = generate_timestamped_filename("", "").replace("_", "").replace(".", "")
    train_path = output_dir / f"train_data_{ts_token}.csv"
    val_path = output_dir / f"val_data_{ts_token}.csv"
    test_path = output_dir / f"test_data_{ts_token}.csv"
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)

    metadata: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "classical_processed_dir": str(classical_dir),
        "classical_train_csv": str(train_csv),
        "classical_val_csv": str(val_csv),
        "classical_test_csv": str(test_csv),
        "genrou_case": args.genrou_case,
        "raw_genrou_csv": str(args.raw_genrou_csv) if args.raw_genrou_csv else None,
        "preprocessing": {
            "filter_angles": args.filter_angles,
            "max_angle_deg": args.max_angle_deg,
            "stability_threshold_deg": args.stability_threshold_deg,
            "engineer_features": args.engineer_features,
            "normalize": args.normalize,
            "normalization_method": args.normalization_method if args.normalize else None,
            "filter_stats": filter_stats,
        },
        "split_mirror": {
            "method": "scenario_id membership from classical train/val/test CSVs",
            "partial_max_scenarios": int(args.max_scenarios) if partial else None,
            "classical_train_ids_sorted": sorted(
                train_ids_full, key=lambda x: (str(type(x)), str(x))
            ),
            "classical_val_ids_sorted": sorted(val_ids_full, key=lambda x: (str(type(x)), str(x))),
            "classical_test_ids_sorted": sorted(
                test_ids_full, key=lambda x: (str(type(x)), str(x))
            ),
            "output_train_ids_sorted": sorted(train_ids, key=lambda x: (str(type(x)), str(x))),
            "output_val_ids_sorted": sorted(val_ids, key=lambda x: (str(type(x)), str(x))),
            "output_test_ids_sorted": sorted(test_ids, key=lambda x: (str(type(x)), str(x))),
        },
        "statistics": {
            "train_rows": len(train_data),
            "val_rows": len(val_data),
            "test_rows": len(test_data),
            "train_scenarios": _nuniq(train_data),
            "val_scenarios": _nuniq(val_data),
            "test_scenarios": _nuniq(test_data),
        },
        "output_files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
        "run_complete_experiment_data_dir": str(output_dir),
        "yaml_note": (
            "If you used --filter-angles or --normalize here, disable the same steps in "
            "data.preprocessing for run_complete_experiment (especially filter_angles), "
            "to avoid re-splitting scenarios."
        ),
    }
    meta_path = output_dir / f"genrou_mirror_metadata_{ts_token}.json"
    save_json(metadata, meta_path)
    if scaler_dict:
        with open(output_dir / f"scalers_{ts_token}.pkl", "wb") as f:
            pickle.dump(scaler_dict, f)

    print()
    print(f"Wrote train -> {train_path}")
    print(f"Wrote val   -> {val_path}")
    print(f"Wrote test  -> {test_path}")
    print(f"Metadata    -> {meta_path}")
    print()
    print(
        "Use --data-dir pointing at this experiment folder (the directory that contains train_data_*.csv), e.g.:"
    )
    print(f'  --data-dir "{output_dir}"')
    if args.filter_angles:
        print()
        print(
            "IMPORTANT: These CSVs are already angle-filtered if you used --filter-angles. "
            "Set data.preprocessing.filter_angles: false in your YAML when using "
            "run_complete_experiment.py, or the driver may re-filter and re-split scenarios."
        )


if __name__ == "__main__":
    main()
