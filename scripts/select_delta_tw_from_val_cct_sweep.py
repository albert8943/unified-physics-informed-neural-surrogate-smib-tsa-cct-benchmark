#!/usr/bin/env python
"""
Select persistence window Δt_w (Option A, CCT-focused) from a validation sweep.

**Prerequisite:** run ``run_pe_input_cct_test.py`` (or
``run_pe_input_cct_persistence_window_sensitivity.py`` with ``--test-csv`` pointing
to the **validation** CSV) for each candidate width so that each folder
``<sweep_dir>/pw_<tag>/pe_input_cct_test_results.json`` exists. JSONs produced after
``run_pe_input_cct_test`` records ``bracket_high_s`` / ``bracket_low_s`` per scenario
are required for bracket-saturation filtering.

**Objective (default):** minimize
    J = (MAE_CCT^PINN + MAE_CCT^StdNN) / 2
on validation, over admissible widths only.

**Admissible:** for each width, upper-bracket saturation fraction is
    (# scenarios with |cct_est - bracket_high_s| ≤ ε) / n
computed separately for PINN and ML; the width is admissible if
    max(frac_PINN, frac_ML) ≤ --max-upper-saturation-frac.

**Tie-break:** smallest Δt_w among widths with J within --j-tie-eps of the minimum.

Example (after sweep on validation)::

    python scripts/select_delta_tw_from_val_cct_sweep.py ^
      --sweep-dir outputs/pe_input_cct_val_pw_sweep ^
      --windows 0.05 0.15 0.25 0.5

See ``docs/publication/rerun_recipes/RERUN_delta_tw_validation_selection_option_A.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def persistence_window_dirname(seconds: float) -> str:
    w = round(float(seconds), 6)
    s = f"{w:.10f}".rstrip("0").rstrip(".")
    return "pw_" + s.replace(".", "p")


def _saturation_fractions(
    block: Dict[str, Any], *, eps: float
) -> Tuple[Optional[float], Optional[float], Optional[int], bool]:
    """
    Return (frac_upper, frac_lower, n, has_bracket_fields).
    frac_* is None if n==0 or bracket fields missing.
    """
    rows = [
        r
        for r in block.get("per_scenario") or []
        if "cct_est" in r and "bracket_high_s" in r and "bracket_low_s" in r
    ]
    n = len(rows)
    if n == 0:
        any_row = [r for r in block.get("per_scenario") or [] if "cct_est" in r]
        has_meta = any("bracket_high_s" in r and "bracket_low_s" in r for r in any_row)
        return None, None, None, has_meta
    hi = sum(1 for r in rows if abs(float(r["cct_est"]) - float(r["bracket_high_s"])) <= eps)
    lo = sum(1 for r in rows if abs(float(r["cct_est"]) - float(r["bracket_low_s"])) <= eps)
    return hi / float(n), lo / float(n), n, True


def _max_abs_error(block: Dict[str, Any]) -> Optional[float]:
    rows = [r for r in block.get("per_scenario") or [] if "abs_error_s" in r]
    if not rows:
        return None
    return float(max(float(r["abs_error_s"]) for r in rows))


@dataclass
class WindowEval:
    delta_tw_s: float
    json_path: Path
    pinn_mae_s: Optional[float]
    ml_mae_s: Optional[float]
    pinn_rmse_s: Optional[float]
    ml_rmse_s: Optional[float]
    j_mae: Optional[float]
    j_rmse: Optional[float]
    pinn_frac_upper: Optional[float]
    ml_frac_upper: Optional[float]
    pinn_frac_lower: Optional[float]
    ml_frac_lower: Optional[float]
    n_scenarios: Optional[int]
    admissible: bool
    admissible_reason: str
    pinn_max_abs_s: Optional[float]
    ml_max_abs_s: Optional[float]


def evaluate_window(
    sweep_dir: Path,
    delta_tw_s: float,
    *,
    eps: float,
    max_upper_frac: float,
    objective: str,
) -> WindowEval:
    sub = sweep_dir / persistence_window_dirname(delta_tw_s) / "pe_input_cct_test_results.json"
    if not sub.is_file():
        return WindowEval(
            delta_tw_s=delta_tw_s,
            json_path=sub,
            pinn_mae_s=None,
            ml_mae_s=None,
            pinn_rmse_s=None,
            ml_rmse_s=None,
            j_mae=None,
            j_rmse=None,
            pinn_frac_upper=None,
            ml_frac_upper=None,
            pinn_frac_lower=None,
            ml_frac_lower=None,
            n_scenarios=None,
            admissible=False,
            admissible_reason=f"missing JSON: {sub}",
            pinn_max_abs_s=None,
            ml_max_abs_s=None,
        )

    with open(sub, encoding="utf-8") as f:
        data = json.load(f)
    pinn_b = data.get("pinn") or {}
    ml_b = data.get("ml") or {}

    p_mae = pinn_b.get("mae_s")
    m_mae = ml_b.get("mae_s")
    p_rmse = pinn_b.get("rmse_s")
    m_rmse = ml_b.get("rmse_s")

    if objective == "pinn_mae_only":
        j_mae = float(p_mae) if p_mae is not None else None
        j_rmse = float(p_rmse) if p_rmse is not None else None
    else:
        if p_mae is not None and m_mae is not None:
            j_mae = 0.5 * (float(p_mae) + float(m_mae))
        else:
            j_mae = None
        if p_rmse is not None and m_rmse is not None:
            j_rmse = 0.5 * (float(p_rmse) + float(m_rmse))
        else:
            j_rmse = None

    fu_p, fl_p, n_p, meta_p = _saturation_fractions(pinn_b, eps=eps)
    fu_m, fl_m, n_m, meta_m = _saturation_fractions(ml_b, eps=eps)

    n_sc = n_p if n_p is not None else n_m
    reason = "ok"
    admissible = True

    if not meta_p and not meta_m and (pinn_b.get("per_scenario") or ml_b.get("per_scenario")):
        admissible = False
        reason = (
            "per_scenario rows lack bracket_high_s/bracket_low_s (re-run run_pe_input_cct_test.py)"
        )
    elif fu_p is None or fu_m is None:
        admissible = False
        reason = "no CCT rows with bracket metadata"
    else:
        sat_max = max(fu_p, fu_m)
        if sat_max > max_upper_frac + 1e-15:
            admissible = False
            reason = (
                f"upper-bracket saturation max(PINN,ML)={sat_max:.3f} "
                f"> {max_upper_frac:.3f} (ε={eps:g} s)"
            )

    return WindowEval(
        delta_tw_s=delta_tw_s,
        json_path=sub,
        pinn_mae_s=float(p_mae) if p_mae is not None else None,
        ml_mae_s=float(m_mae) if m_mae is not None else None,
        pinn_rmse_s=float(p_rmse) if p_rmse is not None else None,
        ml_rmse_s=float(m_rmse) if m_rmse is not None else None,
        j_mae=j_mae,
        j_rmse=j_rmse,
        pinn_frac_upper=fu_p,
        ml_frac_upper=fu_m,
        pinn_frac_lower=fl_p,
        ml_frac_lower=fl_m,
        n_scenarios=n_sc,
        admissible=admissible,
        admissible_reason=reason,
        pinn_max_abs_s=_max_abs_error(pinn_b),
        ml_max_abs_s=_max_abs_error(ml_b),
    )


def _pick_best(
    rows: Sequence[WindowEval],
    *,
    use_rmse: bool,
    j_tie_eps: float,
) -> Tuple[Optional[WindowEval], List[str]]:
    issues: List[str] = []

    def _j(r: WindowEval) -> Optional[float]:
        v = r.j_rmse if use_rmse else r.j_mae
        if v is None:
            return None
        x = float(v)
        return x if math.isfinite(x) else None

    candidates = [r for r in rows if r.admissible and _j(r) is not None]
    if not candidates:
        issues.append(
            "No admissible window with finite objective; relax --max-upper-saturation-frac or fix JSONs."
        )
        return None, issues

    best_j = min(float(_j(r)) for r in candidates)
    near = [r for r in candidates if abs(float(_j(r)) - best_j) <= j_tie_eps]
    near.sort(key=lambda r: r.delta_tw_s)
    chosen = near[0]
    if len(near) > 1:
        issues.append(
            f"Tie-break: {len(near)} widths within J tie tolerance; chose smallest Δt_w = {chosen.delta_tw_s:g} s."
        )
    return chosen, issues


def _serialize_row(r: WindowEval) -> Dict[str, Any]:
    return {
        "delta_tw_s": r.delta_tw_s,
        "json": str(r.json_path),
        "pinn_mae_s": r.pinn_mae_s,
        "ml_mae_s": r.ml_mae_s,
        "pinn_rmse_s": r.pinn_rmse_s,
        "ml_rmse_s": r.ml_rmse_s,
        # Values match the active objective (averaged over models or PINN-only).
        "j_mae": r.j_mae,
        "j_rmse": r.j_rmse,
        "pinn_frac_upper_bracket": r.pinn_frac_upper,
        "ml_frac_upper_bracket": r.ml_frac_upper,
        "pinn_frac_lower_bracket": r.pinn_frac_lower,
        "ml_frac_lower_bracket": r.ml_frac_lower,
        "n_scenarios": r.n_scenarios,
        "admissible": r.admissible,
        "admissible_reason": r.admissible_reason,
        "pinn_max_abs_error_s": r.pinn_max_abs_s,
        "ml_max_abs_error_s": r.ml_max_abs_s,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Select Δt_w from validation CCT sweep (Option A).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Directory containing pw_*/pe_input_cct_test_results.json",
    )
    p.add_argument(
        "--windows",
        type=float,
        nargs="+",
        default=[0.05, 0.15, 0.25, 0.5],
        help="Candidate Δt_w values (must match folder names)",
    )
    p.add_argument(
        "--objective",
        choices=("avg_mae", "avg_rmse", "pinn_mae_only"),
        default="avg_mae",
        help="avg_mae: mean of PINN and Std NN MAE; avg_rmse: same for RMSE; pinn_mae_only: PINN MAE only",
    )
    p.add_argument(
        "--use-rmse-objective",
        action="store_true",
        help=(
            "Minimize RMSE instead of MAE: averaged over models unless "
            "--objective pinn_mae_only (then PINN RMSE only). "
            "Redundant if --objective avg_rmse."
        ),
    )
    p.add_argument(
        "--upper-bracket-eps",
        type=float,
        default=0.002,
        help="Tolerance (s) for treating cct_est as sitting on upper bracket high",
    )
    p.add_argument(
        "--max-upper-saturation-frac",
        type=float,
        default=0.8,
        help="Reject width if max(PINN,ML) upper-saturation fraction exceeds this",
    )
    p.add_argument(
        "--j-tie-eps",
        type=float,
        default=1e-5,
        help="Objective values within this (s for MAE/RMSE) tie; prefer smaller Δt_w",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write machine-readable report (default: <sweep-dir>/delta_tw_selection_report.json)",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Write markdown report (default: <sweep-dir>/delta_tw_selection_report.md)",
    )
    args = p.parse_args()

    if args.upper_bracket_eps <= 0:
        raise SystemExit("--upper-bracket-eps must be positive")
    if not 0.0 <= args.max_upper_saturation_frac <= 1.0:
        raise SystemExit("--max-upper-saturation-frac must be in [0, 1]")
    if args.j_tie_eps < 0:
        raise SystemExit("--j-tie-eps must be non-negative")

    sweep_dir = args.sweep_dir
    use_rmse = args.use_rmse_objective or args.objective == "avg_rmse"
    # Label what is actually minimized (avoid "avg_rmse" when pinn-only).
    if args.objective == "pinn_mae_only":
        obj_tag = "pinn_rmse_only" if use_rmse else "pinn_mae_only"
    elif use_rmse:
        obj_tag = "avg_rmse"
    else:
        obj_tag = args.objective

    # Preserve order, drop duplicate widths (e.g. user pasted 0.25 twice).
    seen: set[float] = set()
    windows_unique: List[float] = []
    for w in args.windows:
        wf = float(w)
        if wf not in seen:
            seen.add(wf)
            windows_unique.append(wf)

    rows: List[WindowEval] = []
    for w in windows_unique:
        rows.append(
            evaluate_window(
                sweep_dir,
                w,
                eps=args.upper_bracket_eps,
                max_upper_frac=args.max_upper_saturation_frac,
                objective=("pinn_mae_only" if args.objective == "pinn_mae_only" else "avg_mae"),
            )
        )

    chosen, issues = _pick_best(rows, use_rmse=use_rmse, j_tie_eps=args.j_tie_eps)

    out_j = args.out_json or (sweep_dir / "delta_tw_selection_report.json")
    out_md = args.out_md or (sweep_dir / "delta_tw_selection_report.md")

    report = {
        "sweep_dir": str(sweep_dir.resolve()),
        "windows_requested": windows_unique,
        "objective": obj_tag,
        "minimize": (
            ("pinn_rmse" if args.objective == "pinn_mae_only" else "rmse_avg")
            if use_rmse
            else ("pinn_mae" if args.objective == "pinn_mae_only" else "mae_avg")
        ),
        "upper_bracket_eps_s": float(args.upper_bracket_eps),
        "max_upper_saturation_frac": float(args.max_upper_saturation_frac),
        "j_tie_eps": float(args.j_tie_eps),
        "per_window": [_serialize_row(r) for r in rows],
        "recommended_delta_tw_s": chosen.delta_tw_s if chosen else None,
        "recommended_json": str(chosen.json_path) if chosen else None,
        "notes": issues,
    }

    out_j.parent.mkdir(parents=True, exist_ok=True)
    with open(out_j, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_j}")

    lines: List[str] = [
        "# Δt_w selection report (validation, Option A)",
        "",
        f"- **Sweep directory:** `{sweep_dir}`",
        f"- **Objective:** `{report['minimize']}` (`{obj_tag}`)",
        f"- **Upper-bracket ε:** {args.upper_bracket_eps} s",
        f"- **Max allowed upper saturation:** {args.max_upper_saturation_frac}",
        "",
        "## Per-window summary",
        "",
        "| Δt_w (s) | Admissible | J (used) | PINN MAE (s) | ML MAE (s) | sat↑ PINN | sat↑ ML | max|PINN err| | max|ML err| |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in rows:
        j_used = r.j_rmse if use_rmse else r.j_mae
        j_str = f"{j_used:.6g}" if j_used is not None else "---"
        pmae = f"{r.pinn_mae_s:.6g}" if r.pinn_mae_s is not None else "---"
        mmae = f"{r.ml_mae_s:.6g}" if r.ml_mae_s is not None else "---"
        fup = f"{r.pinn_frac_upper:.3f}" if r.pinn_frac_upper is not None else "---"
        fum = f"{r.ml_frac_upper:.3f}" if r.ml_frac_upper is not None else "---"
        mxp = f"{r.pinn_max_abs_s:.6g}" if r.pinn_max_abs_s is not None else "---"
        mxm = f"{r.ml_max_abs_s:.6g}" if r.ml_max_abs_s is not None else "---"
        adm = "yes" if r.admissible else "no"
        lines.append(
            f"| {r.delta_tw_s:g} | {adm} | {j_str} | {pmae} | {mmae} | {fup} | {fum} | {mxp} | {mxm} |"
        )
    lines.append("")
    lines.append(
        f"_J (used)_ minimizes {'RMSE' if use_rmse else 'MAE'} per `{report['minimize']}`."
    )
    lines.extend(
        [
            "",
            f"- **Recommended Δt_w:** `{chosen.delta_tw_s if chosen else 'NONE'}` s",
            "",
            "## Selection notes",
            "",
        ]
    )
    for msg in issues:
        lines.append(f"- {msg}")
    if not issues:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## Next step",
            "",
            "Run **one** `run_pe_input_cct_test.py` on the **held-out test** CSV with `--persistence-window-seconds` set to the recommended value; regenerate paper figures/tables. Do **not** change Δt_w after inspecting test metrics.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")

    if chosen:
        print(f"\nRecommended Δt_w = {chosen.delta_tw_s:g} s " f"(from {chosen.json_path})")
    else:
        print("\nNo recommendation; see report for reasons.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
