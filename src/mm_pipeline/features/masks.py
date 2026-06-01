"""Mask helpers for pairwise feature extraction"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


def require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Feature mask helpers require numpy.") from exc
    return np


def as_2d_label_image(label_img: Any, *, name: str = "label image"):
    np = require_numpy()
    arr = np.asarray(label_img)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D label image; got shape {arr.shape}.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} dtype must be integer; got {arr.dtype}.")
    return arr


def as_label_stack(labels: Any):
    np = require_numpy()
    arr = np.asarray(labels)
    if arr.ndim != 3:
        raise ValueError(f"labels must have shape (T,H,W); got {arr.shape}.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"labels dtype must be integer; got {arr.dtype}.")
    return arr


def get_label_mask(label_img: Any, label: int, cache: MutableMapping[int, Any]):
    if int(label) not in cache:
        cache[int(label)] = label_img == int(label)
    return cache[int(label)]


def shift_mask_y(mask: Any, shift_rows: int):
    np = require_numpy()
    if shift_rows == 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    h = mask.shape[0]
    if abs(shift_rows) >= h:
        return out
    if shift_rows > 0:
        out[shift_rows:, :] = mask[: h - shift_rows, :]
    else:
        s = -shift_rows
        out[: h - s, :] = mask[s:, :]
    return out


def iou(mask_a: Any, mask_b: Any) -> float:
    np = require_numpy()
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return float("nan")
    return float(inter / union)
