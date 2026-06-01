"""Cell-instance contracts and extraction helpers"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

Axis = Literal["x", "y"]
OpenEnd = Literal["low", "high"]


@dataclass(frozen=True)
class CellInstance:
    """Represents one labelled cell object in one frame
    """

    label: int
    x: float
    y: float
    area: float
    bbox_minr: int
    bbox_minc: int
    bbox_maxr: int
    bbox_maxc: int
    dataset_id: Optional[str] = None
    frame: Optional[int] = None

    @property
    def centroid_row(self) -> float:
        return self.y

    @property
    def centroid_col(self) -> float:
        return self.x


# Compatibility alias for old code - ignore
Obj = CellInstance


def _validate_axis(axis: str) -> None:
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'.")


def _validate_open_end(open_end: str) -> None:
    if open_end not in ("low", "high"):
        raise ValueError("open_end must be 'low' or 'high'.")


def extract_cell_instances(
    label_img,
    *,
    dataset_id: Optional[str] = None,
    frame: Optional[int] = None,
) -> list[CellInstance]:
    """Extract cell geometry from a single 2D label image. Background label 0 is ignored. Labels are returned in ascending label order,
    matching the natural ordering from regionprops for integer labels.
    """

    import numpy as np

    arr = np.asarray(label_img)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D label image, got shape {arr.shape}.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Label image dtype must be integer, got {arr.dtype}.")

    labels = [int(x) for x in np.unique(arr) if int(x) != 0]
    out: list[CellInstance] = []
    for label in labels:
        rows, cols = np.nonzero(arr == label)
        if rows.size == 0:
            continue
        out.append(
            CellInstance(
                label=label,
                x=float(cols.mean()),
                y=float(rows.mean()),
                area=float(rows.size),
                bbox_minr=int(rows.min()),
                bbox_minc=int(cols.min()),
                bbox_maxr=int(rows.max()) + 1,
                bbox_maxc=int(cols.max()) + 1,
                dataset_id=dataset_id,
                frame=frame,
            )
        )
    return out


def extract_objects_for_frame(label_img) -> list[CellInstance]:
    """Another compatibility alias - ignore"""

    return extract_cell_instances(label_img)


def cell_axis_len(cell: CellInstance, axis: str) -> float:
    _validate_axis(axis)
    if axis == "y":
        return float(cell.bbox_maxr - cell.bbox_minr)
    return float(cell.bbox_maxc - cell.bbox_minc)


def sort_cells_along_trench(
    cells: Iterable[CellInstance],
    axis: str,
    open_end: str,
) -> list[CellInstance]:
    """Sort bottom-to-top, where bottom is the open end."""

    _validate_axis(axis)
    _validate_open_end(open_end)
    key = (lambda cell: cell.y) if axis == "y" else (lambda cell: cell.x)
    reverse = open_end == "high"
    return sorted(cells, key=key, reverse=reverse)


def sort_objects(objs: Iterable[CellInstance], axis: str, open_end: str) -> list[CellInstance]:
    """Yet another compatibility alias - ignore. Sorry :)"""

    return sort_cells_along_trench(objs, axis, open_end)


def cells_by_label(cells: Iterable[CellInstance]) -> dict[int, CellInstance]:
    return {int(cell.label): cell for cell in cells}
