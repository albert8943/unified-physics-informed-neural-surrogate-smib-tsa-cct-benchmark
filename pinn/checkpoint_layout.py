"""
Infer PINN backbone layout from a saved ``state_dict``.

Legacy checkpoints use a flat ``nn.Sequential`` (``network.0.weight``, ...).
Checkpoints trained with ``use_residual=True`` (after the backbone fix) use
``network.stem.*`` and ``network.blocks.*``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

__all__ = [
    "is_residual_backbone_state_dict",
    "infer_architecture_from_state_dict",
]


def is_residual_backbone_state_dict(state_dict: Dict[str, Any]) -> bool:
    return any(str(k).startswith("network.stem.") for k in state_dict)


def _infer_sequential(state_dict: Dict[str, Any]) -> Tuple[int, List[int]]:
    linear_keys = [k for k in state_dict if re.match(r"network\.\d+\.weight", k)]
    nums = (
        sorted(set(int(re.search(r"network\.(\d+)\.weight", k).group(1)) for k in linear_keys))
        if linear_keys
        else []
    )
    if not nums:
        input_dim = (
            int(state_dict["input_standardization.mean"].shape[0])
            if "input_standardization.mean" in state_dict
            else 11
        )
        return input_dim, []
    first_w = state_dict[f"network.{nums[0]}.weight"]
    input_dim = int(first_w.shape[1])
    hidden_dims: List[int] = []
    for n in nums:
        w = state_dict[f"network.{n}.weight"]
        if w.shape[0] == 2:
            break
        hidden_dims.append(int(w.shape[0]))
    return input_dim, hidden_dims


def _infer_residual(state_dict: Dict[str, Any]) -> Tuple[int, List[int]]:
    stem_w = state_dict["network.stem.0.weight"]
    input_dim = int(stem_w.shape[1])
    h0 = int(stem_w.shape[0])
    hidden_dims: List[int] = [h0]
    i = 0
    while True:
        lin = f"network.blocks.{i}.lin1.weight"
        main = f"network.blocks.{i}.f.0.weight"
        if lin in state_dict:
            d = int(state_dict[lin].shape[0])
            hidden_dims.append(d)
            i += 1
            continue
        if main in state_dict:
            d = int(state_dict[main].shape[0])
            hidden_dims.append(d)
            i += 1
            continue
        break
    return input_dim, hidden_dims


def infer_architecture_from_state_dict(
    state_dict: Dict[str, Any],
) -> Tuple[bool, int, List[int]]:
    """
    Returns
    -------
    use_residual
        True if weights use the stem/blocks layout.
    input_dim
        NN input width after standardization.
    hidden_dims
        Width at each hidden stage (same convention as config ``hidden_dims``).
    """
    if is_residual_backbone_state_dict(state_dict):
        inp, hid = _infer_residual(state_dict)
        return True, inp, hid
    inp, hid = _infer_sequential(state_dict)
    return False, inp, hid
