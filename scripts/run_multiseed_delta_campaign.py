#!/usr/bin/env python
"""
Task 3 (parity doc): multi-seed trajectory robustness, **delta RMSE only**.

For each seed, runs ``run_complete_experiment.py`` with frozen PINN*/ML* hyperparameters
(``--lambda-physics 0.5``, ``--ml-baseline-unstable-weight 2``, ``--ml-baseline-omega-weight 50``)
on a fixed processed dataset, then reads ``comparison/delta_comparison`` means.

You may inject the already-frozen Phase~4 run for seed 42 via ``--inject-seed-json`` so you only
retrain seeds 142 and 242.

Usage (repo root, PowerShell example)::

    python scripts/run_multiseed_delta_campaign.py \\
      --data-dir data/processed/exp_20260211_190612 \\
      --output-dir outputs/campaign_multiseed_trajectory \\
      --inject-seed-json outputs/campaign_indep_final_test_20260406/comparison_results.json \\
      --seeds 142 242

Full three-seed from scratch (long)::

    python scripts/run_multiseed_delta_campaign.py \\
      --data-dir data/processed/exp_20260211_190612 \\
      --output-dir outputs/campaign_multiseed_trajectory \\
      --seeds 42 142 242
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/smib/smib_pinn_ml_matched_pe_direct_7_parity_dropout_wd.yaml"
)


def _load_delta_means(comparison_json: Path) -> Tuple[float, float]:
    data = json.loads(comparison_json.read_text(encoding="utf-8"))
    dc = data.get("delta_comparison") or {}
    ml_m = float((dc.get("ml_baseline") or {}).get("mean"))
    pinn_m = float((dc.get("pinn") or {}).get("mean"))
    return ml_m, pinn_m


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-seed delta RMSE campaign (Task 3)")
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Base YAML (parity dropout/WD recipe)",
    )
    p.add_argument(
        "--data-dir", type=Path, required=True, help="Processed folder with train/val/test CSVs"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Campaign root (each seed gets a subdirectory experiment id)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 142, 242],
        help="Random seeds (reproducibility.random_seed)",
    )
    p.add_argument(
        "--inject-seed-json",
        type=str,
        nargs=2,
        action="append",
        metavar=("SEED", "PATH"),
        default=[],
        help=(
            "Use existing comparison_results.json for this seed (skip training). "
            "Repeatable, e.g. --inject-seed-json 42 outputs/.../comparison_results.json"
        ),
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs for both models (optional)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write aggregate JSON (default: docs/publication/pre_specs/multiseed_delta_rmse_results.json)",
    )
    args = p.parse_args()

    inject: Dict[int, Path] = {}
    for pair in args.inject_seed_json:
        s, path_s = int(pair[0]), pair[1]
        inject[s] = Path(path_s)

    out_json = args.out_json or (
        PROJECT_ROOT / "docs/publication/pre_specs/multiseed_delta_rmse_results.json"
    )

    # Include injected seeds even when omitted from --seeds (recipe: inject 42, train 142/242 only).
    inject_only = sorted(set(inject.keys()) - set(args.seeds))
    seeds_to_run: List[int] = inject_only + list(args.seeds)

    per_seed: List[Dict[str, Any]] = []
    ml_vals: List[float] = []
    pinn_vals: List[float] = []

    for seed in seeds_to_run:
        exp_id = f"multiseed_lp05_uw2ow50_seed{seed}"
        cmp_path = args.output_dir / exp_id / "comparison" / "comparison_results.json"

        if seed in inject:
            src = inject[seed]
            if not src.is_file():
                raise SystemExit(f"Inject path missing for seed {seed}: {src}")
            ml_m, pinn_m = _load_delta_means(src)
            per_seed.append(
                {
                    "seed": seed,
                    "experiment_id": exp_id,
                    "source": "injected",
                    "comparison_results_json": str(src.resolve()),
                    "delta_rmse_mean_per_scenario": {"ml_baseline": ml_m, "pinn": pinn_m},
                }
            )
            ml_vals.append(ml_m)
            pinn_vals.append(pinn_m)
            continue

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_complete_experiment.py"),
            "--config",
            str(args.config),
            "--skip-data-generation",
            "--skip-data-analysis",
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(args.output_dir),
            "--experiment-id",
            exp_id,
            "--random-seed",
            str(seed),
            "--lambda-physics",
            "0.5",
            "--ml-baseline-unstable-weight",
            "2.0",
            "--ml-baseline-omega-weight",
            "50",
        ]
        if args.epochs is not None:
            cmd.extend(["--epochs", str(int(args.epochs))])

        print("\n" + "=" * 70)
        print(" ".join(cmd))
        print("=" * 70 + "\n")
        if args.dry_run:
            per_seed.append({"seed": seed, "experiment_id": exp_id, "dry_run_command": cmd})
            continue

        args.output_dir.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if r.returncode != 0:
            raise SystemExit(
                f"run_complete_experiment failed for seed {seed} (exit {r.returncode})"
            )

        if not cmp_path.is_file():
            raise SystemExit(f"Missing comparison results after run: {cmp_path}")

        ml_m, pinn_m = _load_delta_means(cmp_path)
        per_seed.append(
            {
                "seed": seed,
                "experiment_id": exp_id,
                "source": "trained",
                "comparison_results_json": str(cmp_path.resolve()),
                "delta_rmse_mean_per_scenario": {"ml_baseline": ml_m, "pinn": pinn_m},
            }
        )
        ml_vals.append(ml_m)
        pinn_vals.append(pinn_m)

    aggregate: Optional[Dict[str, Any]] = None
    if ml_vals and pinn_vals and not args.dry_run:
        import statistics

        aggregate = {
            "ml_baseline_mean_delta_rmse_across_seeds": {
                "mean": statistics.mean(ml_vals),
                "stdev": statistics.stdev(ml_vals) if len(ml_vals) > 1 else 0.0,
                "n_seeds": len(ml_vals),
            },
            "pinn_mean_delta_rmse_across_seeds": {
                "mean": statistics.mean(pinn_vals),
                "stdev": statistics.stdev(pinn_vals) if len(pinn_vals) > 1 else 0.0,
                "n_seeds": len(pinn_vals),
            },
        }

    report = {
        "protocol": "PARITY_TASK3_multiseed_delta_rmse",
        "driver": "scripts/run_multiseed_delta_campaign.py",
        "config": str(args.config.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seeds": seeds_to_run,
        "epochs_override": args.epochs,
        "metric": "delta_comparison.{ml_baseline,pinn}.mean (mean per-scenario δ RMSE, test split)",
        "per_seed": per_seed,
        "aggregate_across_seeds": aggregate,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote aggregate report: {out_json}")
    else:
        print("\nDry-run: not writing aggregate report (--out-json unchanged).")
    if aggregate:
        print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
