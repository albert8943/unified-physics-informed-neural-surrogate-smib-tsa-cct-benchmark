#!/usr/bin/env python3
"""
Verify two YAML configs match on PINN-affecting sections (model, loss, training).

Training key `ml_baseline_early_stopping_patience` is ignored because only the ML
baseline trainer consumes it; PINN early stopping uses `early_stopping_patience`.

Example:
  python scripts/verify_pinn_sections_match.py \\
    configs/publication/smib_delta20_omega40.yaml \\
    configs/experiments/smib/pinn_ml_fair_loss_tune.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


TRAINING_IGNORE = frozenset({"ml_baseline_early_stopping_patience"})


def _load(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required (pip install pyyaml)")
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root: {path}")
    return data


def _strip_training(tr: Any) -> Dict[str, Any]:
    if not isinstance(tr, dict):
        return {}
    return {k: v for k, v in tr.items() if k not in TRAINING_IGNORE}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_yaml", type=Path, help="Publication / baseline config")
    parser.add_argument("candidate_yaml", type=Path, help="Tuned config to verify")
    args = parser.parse_args()

    ref = _load(args.reference_yaml)
    cand = _load(args.candidate_yaml)

    errors = []
    for section in ("model", "loss"):
        if ref.get(section) != cand.get(section):
            errors.append(f"Mismatch in section {section!r}")

    tr_ref = _strip_training(ref.get("training"))
    tr_cand = _strip_training(cand.get("training"))
    if tr_ref != tr_cand:
        errors.append("Mismatch in training (after ignoring ml_baseline_early_stopping_patience)")

    if errors:
        print("Verification failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("OK: model, loss, and training (PINN-relevant) match between:")
    print(f"  {args.reference_yaml}")
    print(f"  {args.candidate_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
