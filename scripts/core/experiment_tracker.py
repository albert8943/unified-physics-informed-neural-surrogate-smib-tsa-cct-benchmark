"""
Append-only registry of completed experiments under outputs/.

Writes:
  - outputs/experiment_tracker.jsonl — one JSON object per completed run (richer detail)
  - outputs/experiment_tracker.csv — flat columns for spreadsheets

Other entry points can call record_complete_experiment() the same way.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

RECORD_VERSION = 1

TRACKER_JSONL = "experiment_tracker.jsonl"
TRACKER_CSV = "experiment_tracker.csv"

CSV_FIELDS: List[str] = [
    "completed_at",
    "experiment_id",
    "experiment_dir",
    "summary_json",
    "config_path",
    "git_commit",
    "git_branch",
    "random_seed",
    "data_source",
    "test_data_path",
    "pinn_model_path",
    "pinn_delta_rmse",
    "pinn_omega_rmse",
    "pinn_delta_r2",
    "pinn_omega_r2",
    "ml_baseline_type",
    "ml_model_path",
    "ml_delta_rmse",
    "ml_omega_rmse",
    "ml_delta_r2",
    "ml_omega_r2",
    "comparison_delta_pinn_rmse",
    "comparison_delta_ml_rmse",
    "comparison_omega_pinn_rmse",
    "comparison_omega_ml_rmse",
]


def _resolve_under_root(project_root: Path, path: Optional[str]) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(p)


def _pick_ml_baseline(summary: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    ml = summary.get("ml_baseline") or {}
    if not isinstance(ml, dict) or not ml:
        return "", {}
    if "standard_nn" in ml:
        return "standard_nn", ml["standard_nn"] if isinstance(ml["standard_nn"], dict) else {}
    key = next(iter(ml))
    block = ml[key]
    return key, block if isinstance(block, dict) else {}


def _eval_block_metrics(evaluation: Any) -> Dict[str, Any]:
    if not isinstance(evaluation, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in (
        "delta_rmse",
        "omega_rmse",
        "delta_r2",
        "omega_r2",
        "combined_rmse",
    ):
        if k in evaluation:
            out[k] = evaluation[k]
    return out


def _mean_from_comparison_block(block: Any) -> str:
    if isinstance(block, dict) and "mean" in block:
        return str(block["mean"])
    return ""


def _comparison_flat(comparison: Any) -> Dict[str, str]:
    out = {
        "comparison_delta_pinn_rmse": "",
        "comparison_delta_ml_rmse": "",
        "comparison_omega_pinn_rmse": "",
        "comparison_omega_ml_rmse": "",
    }
    if not isinstance(comparison, dict):
        return out

    def _delta_ml(block: Dict[str, Any]) -> str:
        return (
            _mean_from_comparison_block(block.get("ml_baseline"))
            or _mean_from_comparison_block(block.get("ml"))
            or _mean_from_comparison_block(block.get("baseline"))
            or (str(block["ml_rmse"]) if "ml_rmse" in block else "")
        )

    def _delta_pinn(block: Dict[str, Any]) -> str:
        v = _mean_from_comparison_block(block.get("pinn"))
        return v or (str(block["pinn_rmse"]) if "pinn_rmse" in block else "")

    dc = comparison.get("delta_comparison") or {}
    oc = comparison.get("omega_comparison") or {}
    if isinstance(dc, dict):
        out["comparison_delta_pinn_rmse"] = _delta_pinn(dc)
        out["comparison_delta_ml_rmse"] = _delta_ml(dc)
    if isinstance(oc, dict):
        out["comparison_omega_pinn_rmse"] = _delta_pinn(oc)
        out["comparison_omega_ml_rmse"] = _delta_ml(oc)
    return out


def build_tracker_entry(
    summary: Dict[str, Any],
    experiment_root: Path,
    project_root: Path,
) -> Dict[str, Any]:
    """Compact record for JSONL (paths relative to project_root where possible)."""
    exp_rel = _resolve_under_root(project_root, str(experiment_root))
    summary_path = (experiment_root / "experiment_summary.json").resolve()
    summary_rel = _resolve_under_root(project_root, str(summary_path))

    rep = summary.get("reproducibility") or {}
    data = summary.get("data") or {}
    pinn = summary.get("pinn") or {}
    pinn_eval = pinn.get("evaluation") if isinstance(pinn, dict) else None
    ml_type, ml_block = _pick_ml_baseline(summary)
    ml_eval = ml_block.get("evaluation") if isinstance(ml_block, dict) else None

    return {
        "record_version": RECORD_VERSION,
        "kind": "complete_experiment",
        "completed_at": summary.get("timestamp", ""),
        "experiment_id": summary.get("experiment_id", ""),
        "experiment_dir": exp_rel,
        "summary_path": summary_rel,
        "config_path": _resolve_under_root(project_root, summary.get("config_path")),
        "reproducibility": {
            "git_commit": rep.get("git_commit"),
            "git_branch": rep.get("git_branch"),
            "random_seed": rep.get("random_seed"),
        },
        "data": {
            "source": data.get("source"),
            "test_path": _resolve_under_root(project_root, data.get("test_path")),
        },
        "pinn": {
            "model_path": _resolve_under_root(project_root, pinn.get("model_path"))
            if isinstance(pinn, dict)
            else "",
            "evaluation_metrics": _eval_block_metrics(pinn_eval),
        },
        "ml_baseline": {
            "type": ml_type,
            "model_path": _resolve_under_root(project_root, ml_block.get("model_path"))
            if ml_block
            else "",
            "evaluation_metrics": _eval_block_metrics(ml_eval),
        },
        "comparison": summary.get("comparison")
        if isinstance(summary.get("comparison"), dict)
        else {},
    }


def build_csv_row(entry: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, str]:
    row: Dict[str, str] = {k: "" for k in CSV_FIELDS}
    row["completed_at"] = str(entry.get("completed_at", ""))
    row["experiment_id"] = str(entry.get("experiment_id", ""))
    row["experiment_dir"] = str(entry.get("experiment_dir", ""))
    row["summary_json"] = str(entry.get("summary_path", ""))
    row["config_path"] = str(entry.get("config_path", ""))

    rep = entry.get("reproducibility") or {}
    row["git_commit"] = str(rep.get("git_commit") or "")
    row["git_branch"] = str(rep.get("git_branch") or "")
    row["random_seed"] = str(rep.get("random_seed") if rep.get("random_seed") is not None else "")

    dat = entry.get("data") or {}
    row["data_source"] = str(dat.get("source") or "")
    row["test_data_path"] = str(dat.get("test_path") or "")

    pe = entry.get("pinn") or {}
    pm = pe.get("evaluation_metrics") or {}
    row["pinn_model_path"] = str(pe.get("model_path") or "")
    row["pinn_delta_rmse"] = str(pm.get("delta_rmse", ""))
    row["pinn_omega_rmse"] = str(pm.get("omega_rmse", ""))
    row["pinn_delta_r2"] = str(pm.get("delta_r2", ""))
    row["pinn_omega_r2"] = str(pm.get("omega_r2", ""))

    mb = entry.get("ml_baseline") or {}
    mm = mb.get("evaluation_metrics") or {}
    row["ml_baseline_type"] = str(mb.get("type") or "")
    row["ml_model_path"] = str(mb.get("model_path") or "")
    row["ml_delta_rmse"] = str(mm.get("delta_rmse", ""))
    row["ml_omega_rmse"] = str(mm.get("omega_rmse", ""))
    row["ml_delta_r2"] = str(mm.get("delta_r2", ""))
    row["ml_omega_r2"] = str(mm.get("omega_r2", ""))

    comp = _comparison_flat(summary.get("comparison"))
    row.update(comp)
    return row


def infer_project_root(experiment_root: Path) -> Path:
    """Walk parents to find repo root (directory containing scripts/run_complete_experiment.py)."""
    r = experiment_root.resolve()
    while True:
        if (r / "scripts" / "run_complete_experiment.py").exists():
            return r
        if r.parent == r:
            break
        r = r.parent
    # Typical layout: <project>/outputs/complete_experiments/<exp_id>
    return experiment_root.resolve().parent.parent.parent


def record_complete_experiment(
    summary: Dict[str, Any],
    experiment_root: Path,
    project_root: Optional[Path] = None,
) -> None:
    """
    Append this run to outputs/experiment_tracker.jsonl and outputs/experiment_tracker.csv.

    Never raises: failures are printed to stdout so the main experiment still succeeds.
    """
    try:
        pr = project_root if project_root is not None else infer_project_root(experiment_root)

        out_dir = pr / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        entry = build_tracker_entry(summary, experiment_root, pr)
        jsonl_path = out_dir / TRACKER_JSONL
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        csv_path = out_dir / TRACKER_CSV
        row = build_csv_row(entry, summary)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[WARNING] Experiment tracker update failed: {e}")
