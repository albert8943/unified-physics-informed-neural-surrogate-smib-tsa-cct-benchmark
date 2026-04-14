#!/usr/bin/env python
"""
Remap full-angle trajectory rows onto frozen train/val/test scenario_id sets.

Use case: keep the same scenario membership as ``data/processed/exp_20260211_190612/``
(paper / val-gate test IDs) while using long unwrapped δ from
``data/preprocessed/publication/exp_20260116_163828/`` where available.

Some scenario_ids in the reference split may be missing from the full-angle pool
(current tree: 34, 134, 152 for 60211 vs 60116). Use
``--fallback-missing-from`` with ``all_splits_*.csv`` from the reference run to
fill those scenarios with truncated reference rows, or omit it and fix with
``--strict`` after regenerating simulations.

Example (Windows, repo root):

  python scripts/build_full_angle_splits_from_reference.py ^
    --reference-dir data/processed/exp_20260211_190612 ^
    --full-angle-dir data/preprocessed/publication/exp_20260116_163828 ^
    --output-dir data/processed/exp_fullangle_paper_split_20260408 ^
    --fallback-missing-from data/processed/exp_20260211_190612/all_splits_20260211_190612.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _latest_csv(parent: Path, prefix: str) -> Path:
    files = sorted(parent.glob(f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No {prefix}_*.csv under {parent}")
    return files[-1]


def _split_suffix(path: Path) -> str:
    m = re.search(r"_(20\d{6}_\d+)\.csv$", path.name)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse timestamp suffix from {path.name}")


def _scenario_sets_from_reference(ref_dir: Path) -> Tuple[str, Dict[str, Set[int]]]:
    train = _latest_csv(ref_dir, "train_data")
    val = _latest_csv(ref_dir, "val_data")
    test = _latest_csv(ref_dir, "test_data")
    suffix = _split_suffix(train)
    for p in (val, test):
        if _split_suffix(p) != suffix:
            raise ValueError(
                f"Reference split timestamps differ: {train.name} vs {val.name} / {test.name}"
            )

    def ids(p: Path) -> Set[int]:
        return set(pd.read_csv(p, usecols=["scenario_id"])["scenario_id"].unique())

    return suffix, {
        "train": ids(train),
        "val": ids(val),
        "test": ids(test),
    }


def _concat_full_angle_pool(full_dir: Path) -> pd.DataFrame:
    parts = []
    for prefix in ("train_data", "val_data", "test_data"):
        p = _latest_csv(full_dir, prefix)
        parts.append(pd.read_csv(p))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/val/test CSVs: reference scenario_ids + full-angle rows."
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Folder with reference train_data_*.csv, val_data_*.csv, test_data_*.csv",
    )
    parser.add_argument(
        "--full-angle-dir",
        type=Path,
        required=True,
        help="Folder with full-angle publication splits (e.g. exp_20260116_163828)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory (created). Writes train/val/test with new timestamp suffix.",
    )
    parser.add_argument(
        "--fallback-missing-from",
        type=Path,
        default=None,
        help=(
            "Optional combined CSV (e.g. all_splits_*.csv) for scenario_ids "
            "absent from full-angle pool"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any reference scenario_id is missing from full-angle pool (ignores fallback)",
    )
    args = parser.parse_args()

    ref_suffix, sets = _scenario_sets_from_reference(args.reference_dir)
    print(f"Reference timestamp suffix: {ref_suffix}")
    print(
        "Scenario counts: train {}, val {}, test {}".format(
            len(sets["train"]), len(sets["val"]), len(sets["test"])
        )
    )

    pool = _concat_full_angle_pool(args.full_angle_dir)
    if "scenario_id" not in pool.columns:
        raise KeyError("full-angle pool missing column scenario_id")
    pool_ids = set(pool["scenario_id"].unique())

    all_ref = sets["train"] | sets["val"] | sets["test"]
    missing = sorted(all_ref - pool_ids)
    if missing:
        print(
            "WARNING: scenario_ids in reference but not in full-angle pool "
            f"({len(missing)}): {missing}"
        )
        if args.strict:
            sys.exit(1)
        if args.fallback_missing_from is None:
            print("ERROR: Provide --fallback-missing-from or regenerate full-angle data.")
            sys.exit(1)

    fallback: pd.DataFrame | None = None
    if args.fallback_missing_from is not None:
        fallback = pd.read_csv(args.fallback_missing_from)
        if "scenario_id" not in fallback.columns:
            raise KeyError("fallback CSV missing scenario_id")

    from scripts.core.utils import generate_timestamped_filename

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_suffix = _split_suffix(Path(generate_timestamped_filename("train_data", "csv")))
    print(f"Output timestamp suffix: {out_suffix}")

    for split_name, id_set in sets.items():
        sub = pool[pool["scenario_id"].isin(id_set)].copy()
        got = set(sub["scenario_id"].unique())
        still = sorted(id_set - got)
        if still:
            if fallback is None:
                print(f"ERROR: Split {split_name}: missing rows for scenarios {still}")
                sys.exit(1)
            fb = fallback[fallback["scenario_id"].isin(still)].copy()
            if fb["scenario_id"].nunique() != len(still):
                have = set(fb["scenario_id"].unique())
                print(
                    "ERROR: Fallback still missing scenarios {}".format(sorted(set(still) - have))
                )
                sys.exit(1)
            print(
                f"  {split_name}: filled {len(still)} scenario(s) from fallback "
                "(truncated reference rows)"
            )
            sub = pd.concat([sub, fb], ignore_index=True)

        out_name = f"{split_name}_data_{out_suffix}.csv"
        out_path = args.output_dir / out_name
        sub.sort_values(["scenario_id", "time"], inplace=True)
        sub.to_csv(out_path, index=False)
        print(
            f"  Wrote {out_path.name}: {len(sub):,} rows, "
            f"{sub['scenario_id'].nunique()} scenarios"
        )

    print("\n[OK] Done. Point run_complete_experiment.py --data-dir at:", args.output_dir)


if __name__ == "__main__":
    main()
