#!/usr/bin/env python
"""
Colab/online-GPU runner for the multimachine complete experiment.

Sets default paths from environment variables so you can run the same pipeline
on Google Colab (or other cloud) without retyping long paths. For full options,
run the main script directly: scripts/run_multimachine_complete_experiment.py.

Environment variables (optional):
  PINN_DATA_PATH       Default for --data-path
  PINN_PROCESSED_DIR   Default for --processed-dir
  PINN_OUTPUT_BASE     Default for --output-base
  (e.g. /content/drive/MyDrive/pinn_data/... on Colab)

Usage (Colab):
  # After cloning repo and mounting Drive:
  !python scripts/run_multimachine_complete_experiment_colab.py \\
    --data-path "/content/drive/MyDrive/pinn_data/parameter_sweep_data_*.csv" \\
    --output-base /content/drive/MyDrive/pinn_results

  # Or set env vars in a cell and run with minimal args:
  import os
  os.environ["PINN_DATA_PATH"] = "/content/drive/MyDrive/pinn_data/parameter_sweep_data_*.csv"
  os.environ["PINN_OUTPUT_BASE"] = "/content/drive/MyDrive/pinn_results"
  !python scripts/run_multimachine_complete_experiment_colab.py --skip-ml-baseline-training
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _has_arg(args: list, name: str) -> bool:
    """Return True if args contains name (e.g. --data-path)."""
    return any(a == name or a.startswith(name + "=") for a in args)


def _inject_defaults(argv: list) -> list:
    """Inject --data-path, --processed-dir, --output-base from env if not in argv."""
    out = list(argv)
    env_data = os.environ.get("PINN_DATA_PATH")
    env_processed = os.environ.get("PINN_PROCESSED_DIR")
    env_output = os.environ.get("PINN_OUTPUT_BASE")
    if env_data and not _has_arg(out, "--data-path"):
        out.extend(["--data-path", env_data])
    if env_processed and not _has_arg(out, "--processed-dir"):
        out.extend(["--processed-dir", env_processed])
    if env_output and not _has_arg(out, "--output-base"):
        out.extend(["--output-base", env_output])
    return out


def _is_colab() -> bool:
    """Return True if running inside Google Colab."""
    return "google.colab" in sys.modules or os.environ.get("COLAB_GPU", "") != ""


def main():
    main_script = PROJECT_ROOT / "scripts" / "run_multimachine_complete_experiment.py"
    if not main_script.exists():
        print(f"[ERROR] Main script not found: {main_script}", file=sys.stderr)
        sys.exit(1)

    if _is_colab():
        print(
            "[Colab] Free: save --output-base to Drive, session ~12h. "
            "Pro: 24h, background run. See docs/guides/COLAB_MULTIMACHINE_WORKFLOW.md"
        )

    argv = sys.argv[1:]
    argv = _inject_defaults(argv)
    cmd = [sys.executable, str(main_script)] + argv

    env = os.environ.copy()
    project_root_str = str(PROJECT_ROOT)
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = project_root_str + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = project_root_str

    rc = subprocess.run(cmd, cwd=project_root_str, env=env)
    sys.exit(rc.returncode)


if __name__ == "__main__":
    main()
