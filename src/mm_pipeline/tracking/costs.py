"""Old cost functions - need updating displacement features to ratios"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from mm_pipeline.config import TrackerParams
from mm_pipeline.core import (
    CellInstance,
    FramePair,
    TrackingOperation,
    cell_axis_len,
)
from mm_pipeline.core.operations import normalize_operation

EPS = 1e-9
INFEASIBLE_DIVISION_COST = 1e9


def validate_tracker_context(frame_pair: FramePair, params: TrackerParams) -> None:
    if params.axis != frame_pair.axis:
        raise ValueError(
            f"TrackerParams.axis ({params.axis!r}) must match FramePair.axis ({frame_pair.axis!r})."
        )


def position(cell: CellInstance, axis: str) -> float:
    if axis == "y":
        return float(cell.y)
    if axis == "x":
        return float(cell.x)
    raise ValueError("axis must be 'x' or 'y'.")


def touches_open(
    cell: CellInstance,
    frame_pair: FramePair,
    border_margin: int,
) -> bool:
    h, w = frame_pair.frame_shape
    if frame_pair.axis == "y":
        if frame_pair.open_end == "low":
            return cell.bbox_minr <= border_margin
        return cell.bbox_maxr >= (h - 1 - border_margin)
    if frame_pair.open_end == "low":
        return cell.bbox_minc <= border_margin
    return cell.bbox_maxc >= (w - 1 - border_margin)


def exit_increment(k_exit: int, params: TrackerParams) -> float:
    return params.exit_lin + params.exit_quad * (2.0 * float(k_exit) - 1.0)


def link_cost(
    source: CellInstance,
    dest: CellInstance,
    *,
    bottom_label_dest: int | None,
    frame_pair: FramePair,
    params: TrackerParams,
) -> float:
    dpos = abs(position(source, frame_pair.axis) - position(dest, frame_pair.axis))
    da = abs(math.log((source.area + EPS) / (dest.area + EPS)))
    ratio = (dest.area + EPS) / (source.area + EPS)
    shrink = max(0.0, (1.0 - ratio) - params.shrink_tol)
    use_border_weight = (
        bottom_label_dest is not None
        and dest.label == bottom_label_dest
        and touches_open(dest, frame_pair, params.border_margin)
    )
    shrink_weight = params.wshrink_border if use_border_weight else params.wshrink
    return params.wy * dpos + params.wa * da + shrink_weight * shrink


def divide_cost(
    source: CellInstance,
    dest1: CellInstance,
    dest2: CellInstance,
    *,
    frame_pair: FramePair,
    params: TrackerParams,
) -> float:
    source_area = float(source.area)
    dest_sum_area = float(dest1.area + dest2.area)
    dest_max_area = float(max(dest1.area, dest2.area))

    source_len = cell_axis_len(source, frame_pair.axis)
    dest_sum_len = cell_axis_len(dest1, frame_pair.axis) + cell_axis_len(dest2, frame_pair.axis)
    dest_max_len = max(cell_axis_len(dest1, frame_pair.axis), cell_axis_len(dest2, frame_pair.axis))

    if dest_sum_area > (1.0 + params.div_tol_sum_area) * source_area:
        return INFEASIBLE_DIVISION_COST
    if dest_max_area > (1.0 + params.div_tol_ind_area) * source_area:
        return INFEASIBLE_DIVISION_COST
    if dest_sum_len > (1.0 + params.div_tol_sum_len) * source_len:
        return INFEASIBLE_DIVISION_COST
    if dest_max_len > (1.0 + params.div_tol_ind_len) * source_len:
        return INFEASIBLE_DIVISION_COST

    dpos = abs(position(source, frame_pair.axis) - position(dest1, frame_pair.axis)) + abs(
        position(source, frame_pair.axis) - position(dest2, frame_pair.axis)
    )
    area_err = abs(source.area - (dest1.area + dest2.area)) / max(source.area, 1.0)
    sym = abs(dest1.area - dest2.area) / (dest1.area + dest2.area + EPS)
    ratio1 = (dest1.area + EPS) / (source.area + EPS)
    ratio2 = (dest2.area + EPS) / (source.area + EPS)
    shrink_div = max(0.0, (1.0 - ratio1) - params.shrink_tol) + max(
        0.0,
        (1.0 - ratio2) - params.shrink_tol,
    )
    return (
        params.c_div0
        + (params.wy * dpos)
        + (params.wa * area_err)
        + (params.wsym * sym)
        - (params.w_divshrink * shrink_div)
    )


def candidate_ops_cost(
    cells_t: Sequence[CellInstance],
    cells_k: Sequence[CellInstance],
    frame_pair: FramePair,
    params: TrackerParams,
    ops: Iterable[TrackingOperation | Sequence[object]],
) -> float:
    """Evaluate one complete operation sequence with the DP's exact costs.

    The cell sequences and operations use the tracker's internal open-end-first
    order. Validation is deliberately performed here so this function is safe for
    manual review input as well as generated candidates. Exit increments use the
    same 1-based exit count as the DP, and link costs receive the same first
    destination label for open-boundary shrink weighting.
    """

    from .validation import assert_ops_valid

    normalised = [normalize_operation(op) for op in ops]
    assert_ops_valid(cells_t, cells_k, normalised)
    validate_tracker_context(frame_pair, params)

    sources = {int(cell.label): cell for cell in cells_t}
    dests = {int(cell.label): cell for cell in cells_k}
    bottom_label_dest = cells_k[0].label if cells_k else None
    exit_count = 0
    total = 0.0

    for op in normalised:
        source = sources[int(op.src_label)]
        if op.kind == "exit":
            exit_count += 1
            total += exit_increment(exit_count, params)
        elif op.kind == "link":
            assert op.dst1_label is not None
            total += link_cost(
                source,
                dests[int(op.dst1_label)],
                bottom_label_dest=bottom_label_dest,
                frame_pair=frame_pair,
                params=params,
            )
        elif op.kind == "divide":
            assert op.dst1_label is not None and op.dst2_label is not None
            total += divide_cost(
                source,
                dests[int(op.dst1_label)],
                dests[int(op.dst2_label)],
                frame_pair=frame_pair,
                params=params,
            )
    return float(total)
