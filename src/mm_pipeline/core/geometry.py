"""Geometry and frame-pair contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from .cells import Axis, OpenEnd


@dataclass(frozen=True)
class FramePair:
    """Context for one tracking problem between two frames"""

    dataset_id: str
    t: int
    k: int
    frame_shape: tuple[int, int]
    axis: Axis
    open_end: OpenEnd
    pair_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty.")
        if int(self.t) < 0:
            raise ValueError("t must be non-negative.")
        if int(self.k) <= int(self.t):
            raise ValueError("k must be greater than t.")
        if self.axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'.")
        if self.open_end not in ("low", "high"):
            raise ValueError("open_end must be 'low' or 'high'.")
        if len(self.frame_shape) != 2:
            raise ValueError("frame_shape must be a 2-tuple of (height, width).")

        h, w = (int(self.frame_shape[0]), int(self.frame_shape[1]))
        if h <= 0 or w <= 0:
            raise ValueError("frame_shape dimensions must be positive.")

        object.__setattr__(self, "t", int(self.t))
        object.__setattr__(self, "k", int(self.k))
        object.__setattr__(self, "frame_shape", (h, w))
        object.__setattr__(self, "pair_id", f"{self.dataset_id}:{self.t}->{self.k}")
