"""
Regression checks for utils.cct_binary_search (run from repo root):
  python scripts/test_cct_binary_search_regression.py
"""

from __future__ import annotations

import io
import contextlib
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cct_binary_search import estimate_cct_binary_search  # noqa: E402


class MonotonicMockModel:
    """Stable iff candidate clearing time <= true_cct."""

    def __init__(self, true_cct: float = 0.35) -> None:
        self.true_cct = true_cct

    def predict(
        self,
        t,
        delta0,
        omega0,
        H,
        D,
        Pm,
        Xprefault,
        Xfault,
        Xpostfault,
        tf,
        tc,
        device="cpu",
    ):
        t = np.asarray(t, dtype=float).reshape(-1)
        if float(tc) <= self.true_cct:
            delta = np.full_like(t, 0.5)
        else:
            delta = np.full_like(t, 4.0)
        omega = np.ones_like(t)
        return delta, omega


class BrokenPredictModel:
    def predict(self, **kwargs):
        raise RuntimeError("intentional failure")


def main() -> None:
    t_eval = np.linspace(0, 2.0, 50)
    m = MonotonicMockModel(true_cct=0.355)
    _cct, info = estimate_cct_binary_search(
        m,
        0.1,
        1.0,
        5.0,
        1.0,
        0.8,
        0.5,
        1e-4,
        0.5,
        tf=0.1,
        t_eval=t_eval,
        low=0.11,
        high=2.0,
        tolerance=0.01,
        max_iterations=50,
        verbose=False,
    )
    assert info["converged"] is True, info
    assert info["bracket_width"] <= 0.01 + 1e-9, info

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        estimate_cct_binary_search(
            BrokenPredictModel(),
            0.1,
            1.0,
            5.0,
            1.0,
            0.8,
            0.5,
            1e-4,
            0.5,
            tf=0.1,
            t_eval=np.linspace(0, 1.0, 10),
            low=0.11,
            high=0.4,
            tolerance=0.02,
            max_iterations=5,
            verbose=True,
        )
    out = buf.getvalue()
    assert "no trajectory" in out or "UNSTABLE" in out, out

    print("test_cct_binary_search_regression: OK")


if __name__ == "__main__":
    main()
