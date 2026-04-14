#!/usr/bin/env python
"""
Compare test scenario IDs across artifacts (e.g. Colab experiment_summary vs local repro).

Accepts:
  - experiment_summary.json (uses comparison.test_scenario_ids)
  - test_scenario_ids.json (flat list)
  - test_data_*.csv (unique scenario_id column)

Exit code 0 if sets match, 1 otherwise.

Examples:
  python scripts/verify_test_scenarios_match.py \\
    "colab/results from colab/smib_complete_experiments/exp_20260330_163851/experiment_summary.json" \\
    outputs/complete_experiments/exp_20260211_190610/experiment_summary.json

  python scripts/verify_test_scenarios_match.py reference_summary.json \\
    data/processed/exp_20260211_190612/test_data_20260211_190612.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Set, cast

import pandas as pd


def _ids_from_summary_dict(data: dict[str, Any]) -> Optional[list[Any]]:
    comp = data.get("comparison")
    if isinstance(comp, dict) and "test_scenario_ids" in comp:
        return cast(list[Any], comp["test_scenario_ids"])
    if "test_scenario_ids" in data:
        return cast(list[Any], data["test_scenario_ids"])
    return None


def load_scenario_ids(path: Path) -> Set[Any]:
    """Load a set of test scenario IDs from JSON or CSV."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        if "scenario_id" not in df.columns:
            raise ValueError(f"No scenario_id column in {path}")
        return set(df["scenario_id"].unique().tolist())

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict):
        ids = _ids_from_summary_dict(data)
        if ids is not None:
            return set(ids)
    raise ValueError(
        f"Unrecognized JSON shape in {path}: expected list or dict with "
        "comparison.test_scenario_ids or test_scenario_ids"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify two artifacts define the same test scenario_id set."
    )
    parser.add_argument(
        "path_a",
        type=Path,
        help="First file: experiment_summary.json, test_scenario_ids.json, or test CSV",
    )
    parser.add_argument(
        "path_b",
        type=Path,
        help="Second file (same formats)",
    )
    args = parser.parse_args()

    try:
        a_ids = load_scenario_ids(args.path_a)
        b_ids = load_scenario_ids(args.path_b)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"A: {args.path_a} ({len(a_ids)} ids)")
    print(f"B: {args.path_b} ({len(b_ids)} ids)")

    if a_ids == b_ids:
        print("OK: Test scenario ID sets match.")
        return 0

    only_a = sorted(a_ids - b_ids, key=lambda x: (str(type(x)), x))
    only_b = sorted(b_ids - a_ids, key=lambda x: (str(type(x)), x))
    print("MISMATCH:")
    if only_a:
        print(f"  Only in A ({len(only_a)}): {only_a[:50]}{' ...' if len(only_a) > 50 else ''}")
    if only_b:
        print(f"  Only in B ({len(only_b)}): {only_b[:50]}{' ...' if len(only_b) > 50 else ''}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
