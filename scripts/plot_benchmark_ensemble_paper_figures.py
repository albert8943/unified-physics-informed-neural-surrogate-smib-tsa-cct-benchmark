#!/usr/bin/env python
"""
Vector PDFs for IEEE Access: benchmark parameter-space and ensemble CCT figures.

Reuses plotting logic from ``scripts/analyze_data.py`` with stable filenames and
TrueType text in PDFs (``pdf.fonttype = 42``) for journal-friendly embedding.

Outputs (under ``--output-dir``)::

    parameter_distributions.pdf
    parameter_space_coverage.pdf
    parameter_space_coverage_3d.pdf
    cct_distribution.pdf
    cct_vs_parameters.pdf

Example (repository root)::

    python scripts/plot_benchmark_ensemble_paper_figures.py ^
      --data data/processed/exp_20260211_190612/all_splits_20260211_190612.csv

If ``--data`` is omitted, the script searches ``data/processed/exp_20260211_190612``
and ``data/common`` for ``all_splits*.csv`` or ``full_trajectory_data*.csv`` (newest).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Editable text in PDF/PS (Type 42 TrueType); preferred for IEEE figures.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from scripts.analyze_data import (  # noqa: E402
    generate_cct_figures,
    generate_parameter_space_figures,
    load_data,
)


def _default_data_csv(repo: Path) -> Path:
    search_roots: List[Path] = [
        repo / "data/processed/exp_20260211_190612",
        repo / "data/common",
    ]
    patterns = ("all_splits*.csv", "full_trajectory_data*.csv")
    candidates: List[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for pat in patterns:
            candidates.extend(root.glob(pat))
    if not candidates:
        raise FileNotFoundError(
            "No trajectory CSV found. Pass --data explicitly, e.g.\n"
            "  data/processed/exp_20260211_190612/all_splits_*.csv"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    repo = _REPO_ROOT
    # Avoid UnicodeEncodeError from analyze_data emoji logs on Windows cp949 consoles.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    default_out = repo / "paper_writing/IEEE Access Template/figures/benchmark_ensemble"

    parser = argparse.ArgumentParser(
        description="Regenerate benchmark ensemble figures as publication PDFs."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Trajectory CSV with param_H, param_D, param_Pm, scenario_id, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Destination directory (default: {default_out})",
    )
    args = parser.parse_args()

    data_path = args.data.resolve() if args.data else _default_data_csv(repo)
    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data: {data_path}")
    print(f"Output: {out_dir}")

    df = load_data(data_path)
    formats = ["pdf"]
    generate_parameter_space_figures(df, out_dir, formats, stable_paper_names=True)
    generate_cct_figures(df, out_dir, formats, stable_paper_names=True)

    print("Done.")


if __name__ == "__main__":
    main()
