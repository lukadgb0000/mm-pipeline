"""Validation for segmentation label outputs"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from mm_pipeline.io.labels import collect_label_paths, load_label_stack


@dataclass(frozen=True)
class LabelStackValidation:
    is_valid: bool
    frame_count: int
    frame_shape: Optional[tuple[int, int]]
    dtype: str
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))


def validate_label_stack(
    labels: np.ndarray,
    *,
    raw_image_shapes: Optional[Iterable[tuple[int, ...]]] = None,
    allow_empty_frames: bool = False,
) -> LabelStackValidation:
    """Validate the label-stack contract required by tracking"""

    import numpy as np

    arr = np.asarray(labels)
    errors: list[str] = []
    warnings: list[str] = []

    if arr.ndim != 3:
        errors.append(f"label stack must have shape (T,H,W); got {arr.shape}")
        frame_count = int(arr.shape[0]) if arr.ndim else 0
        frame_shape = None
    else:
        frame_count = int(arr.shape[0])
        frame_shape = (int(arr.shape[1]), int(arr.shape[2]))

    if not np.issubdtype(arr.dtype, np.integer):
        errors.append(f"label stack dtype must be integer; got {arr.dtype}")

    if arr.size and np.nanmin(arr) < 0:
        errors.append("label stack contains negative labels")

    if arr.ndim == 3 and not allow_empty_frames:
        empty = [i for i, frame in enumerate(arr) if not np.any(frame > 0)]
        if empty:
            errors.append(f"empty label frames found: {empty}")

    if raw_image_shapes is not None and arr.ndim == 3:
        shapes = [tuple(shape) for shape in raw_image_shapes]
        if len(shapes) != arr.shape[0]:
            errors.append(f"raw image count {len(shapes)} != label frame count {arr.shape[0]}")
        for i, shape in enumerate(shapes[: arr.shape[0]]):
            if tuple(shape[:2]) != tuple(arr.shape[1:3]):
                errors.append(f"raw image frame {i} shape {shape[:2]} != label shape {arr.shape[1:3]}")
                break

    return LabelStackValidation(
        is_valid=not errors,
        frame_count=frame_count,
        frame_shape=frame_shape,
        dtype=str(arr.dtype),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def validate_label_directory(labels_dir: str | Path, *, allow_empty_frames: bool = False) -> LabelStackValidation:
    paths = collect_label_paths(labels_dir)
    if not paths:
        return LabelStackValidation(
            is_valid=False,
            frame_count=0,
            frame_shape=None,
            dtype="",
            errors=(f"No label TIFFs found in {labels_dir}",),
        )
    labels = load_label_stack(paths)
    return validate_label_stack(labels, allow_empty_frames=allow_empty_frames)
