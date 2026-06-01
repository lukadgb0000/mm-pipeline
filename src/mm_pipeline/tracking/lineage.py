"""Lineage reconstruction from per-pair operations and QA decisions"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from mm_pipeline.core import (
    CellInstance,
    TrackingOperation,
    cell_axis_len,
    deserialize_ops_json,
    extract_cell_instances,
    sort_cells_along_trench,
)
from mm_pipeline.qa.decisions import Action, QADecision


def apply_ops_to_lineage(
    t: int,
    ops: Iterable[TrackingOperation],
    frame_a: dict[int, CellInstance],
    frame_b: dict[int, CellInstance],
    label_to_track: dict[int, int],
    next_track_id: int,
    tracks_rows: list[dict[str, Any]],
    events_rows: list[dict[str, Any]],
    division_rows: list[dict[str, Any]],
    *,
    axis: str = "y",
    dst_t: int | None = None,
) -> int:
    """Apply one frame-pair's operations to growing lineage tables.

    Migrated from ``17tracka_3.apply_ops_to_tracks``. ``dst_t`` overrides the
    destination frame index for bridge ops where the destination is not
    ``t + 1``; events get an extra ``dst_t`` column so
    can tell normal links from bridge links
    """

    destination_t = t + 1 if dst_t is None else int(dst_t)
    new_label_to_track: dict[int, int] = {}

    for op in ops:
        kind = op.kind
        src = op.src_label
        d1 = op.dst1_label
        d2 = op.dst2_label

        if kind == "link":
            if d1 is None:
                raise ValueError("link op missing dst1_label")
            tid = label_to_track.get(src)
            if tid is None:
                tid = next_track_id
                next_track_id += 1
            new_label_to_track[d1] = tid
            events_rows.append(
                {
                    "t": t,
                    "event": "link",
                    "src_label": src,
                    "dst1_label": d1,
                    "dst2_label": "",
                    "track_id": tid,
                    "cost": None,
                    "dst_t": destination_t,
                }
            )
        elif kind == "divide":
            if d1 is None or d2 is None:
                raise ValueError("divide op missing daughter labels")
            mother_tid = label_to_track.get(src)
            if mother_tid is None:
                mother_tid = next_track_id
                next_track_id += 1
            d1_tid = next_track_id
            next_track_id += 1
            d2_tid = next_track_id
            next_track_id += 1
            new_label_to_track[d1] = d1_tid
            new_label_to_track[d2] = d2_tid
            events_rows.append(
                {
                    "t": t,
                    "event": "divide",
                    "src_label": src,
                    "dst1_label": d1,
                    "dst2_label": d2,
                    "track_id": mother_tid,
                    "cost": None,
                    "dst_t": destination_t,
                }
            )
            division_rows.append(
                {
                    "t_div": t,
                    "mother_track_id": mother_tid,
                    "d1_track_id": d1_tid,
                    "d2_track_id": d2_tid,
                }
            )
        elif kind == "exit":
            events_rows.append(
                {
                    "t": t,
                    "event": "exit",
                    "src_label": src,
                    "dst1_label": "",
                    "dst2_label": "",
                    "track_id": label_to_track.get(src, -1),
                    "cost": None,
                    "dst_t": destination_t,
                }
            )
        else:
            raise ValueError(f"Unknown op kind: {kind}")

    for lab, tid in new_label_to_track.items():
        cell = frame_b[lab]
        tracks_rows.append(
            {
                "track_id": tid,
                "t": destination_t,
                "label": lab,
                "x": cell.x,
                "y": cell.y,
                "area": cell.area,
                "axis_len": cell_axis_len(cell, axis=axis),  # type: ignore[arg-type]
            }
        )

    label_to_track.clear()
    label_to_track.update(new_label_to_track)
    return next_track_id


def _frame_cell_map(label_img: Any, *, axis: str, open_end: str) -> dict[int, CellInstance]:
    cells = sort_cells_along_trench(
        extract_cell_instances(label_img),
        axis=axis,  # type: ignore[arg-type]
        open_end=open_end,  # type: ignore[arg-type]
    )
    return {int(cell.label): cell for cell in cells}


def _init_tracks_at_frame(
    labels: Any,
    t: int,
    label_to_track: dict[int, int],
    next_track_id: int,
    tracks_rows: list[dict[str, Any]],
    *,
    axis: str,
    open_end: str,
) -> int:
    label_to_track.clear()
    cells = sort_cells_along_trench(
        extract_cell_instances(labels[t]),
        axis=axis,  # type: ignore[arg-type]
        open_end=open_end,  # type: ignore[arg-type]
    )
    for cell in cells:
        tid = next_track_id
        next_track_id += 1
        label_to_track[int(cell.label)] = tid
        tracks_rows.append(
            {
                "track_id": tid,
                "t": t,
                "label": int(cell.label),
                "x": cell.x,
                "y": cell.y,
                "area": cell.area,
                "axis_len": cell_axis_len(cell, axis=axis),  # type: ignore[arg-type]
            }
        )
    return next_track_id


def reconstruct_from_qa_decisions(
    decisions: Sequence[QADecision],
    candidate_features: Any,  # kept in signature for symmetry / future use
    labels: Any,
    *,
    open_end: str = "high",
    axis: str = "y",
) -> tuple[Any, Any, Any]:
    """Build (tracks, events, divisions) DataFrames from QA decisions.

    Decisions are walked in order of happening. Action.KEEP applies the chosen
    candidate's ops to extend track IDs from t to t+1. Action.DROP
    breaks the lineage at t and the next non-dropped, non-bridged frame
    re-initialises track IDs. Action.BRIDGE with bridge_is_primary=True``
    applies bridge ops from t_a directly to t_b; intermediate frames
    are skipped entirely
    """

    try:
        import pandas as pd  # noqa: F401  (used by _frames)
    except ImportError as exc:
        raise RuntimeError("Lineage reconstruction requires pandas.") from exc

    if labels is None or int(getattr(labels, "shape", [0])[0]) == 0:
        return _empty_lineage()

    decisions_by_t: dict[int, QADecision] = {}
    bridge_primary_by_t_a: dict[int, QADecision] = {}
    bridge_skip: set[int] = set()
    for d in decisions:
        decisions_by_t[d.t] = d
        if d.action == Action.BRIDGE and d.bridge_span is not None:
            t_a, t_b = d.bridge_span
            if d.bridge_is_primary:
                bridge_primary_by_t_a[t_a] = d
            for ti in range(t_a + 1, t_b):
                bridge_skip.add(ti)

    T = int(labels.shape[0])

    tracks_rows: list[dict[str, Any]] = []
    events_rows: list[dict[str, Any]] = []
    division_rows: list[dict[str, Any]] = []
    label_to_track: dict[int, int] = {}
    next_track_id = 1

    def _ensure_init(t: int) -> int:
        nonlocal next_track_id
        if not label_to_track:
            next_track_id = _init_tracks_at_frame(
                labels, t, label_to_track, next_track_id, tracks_rows,
                axis=axis, open_end=open_end,
            )
        return next_track_id

    for t in range(T - 1):
        if t in bridge_skip:
            continue

        if t in bridge_primary_by_t_a:
            primary = bridge_primary_by_t_a[t]
            assert primary.bridge_span is not None
            assert primary.bridge_ops_json is not None
            t_a, t_b = primary.bridge_span
            ops = deserialize_ops_json(primary.bridge_ops_json)
            frame_a = _frame_cell_map(labels[t_a], axis=axis, open_end=open_end)
            frame_b = _frame_cell_map(labels[t_b], axis=axis, open_end=open_end)
            next_track_id = _ensure_init(t_a)
            next_track_id = apply_ops_to_lineage(
                t_a, ops, frame_a, frame_b, label_to_track, next_track_id,
                tracks_rows, events_rows, division_rows,
                axis=axis, dst_t=t_b,
            )
            continue

        d = decisions_by_t.get(t)
        if d is None or d.action == Action.DROP or d.action == Action.BRIDGE:
            label_to_track.clear()
            continue

        if d.chosen_ops_json is None:
            label_to_track.clear()
            continue

        ops = deserialize_ops_json(d.chosen_ops_json)
        frame_a = _frame_cell_map(labels[t], axis=axis, open_end=open_end)
        frame_b = _frame_cell_map(labels[t + 1], axis=axis, open_end=open_end)
        next_track_id = _ensure_init(t)
        next_track_id = apply_ops_to_lineage(
            t, ops, frame_a, frame_b, label_to_track, next_track_id,
            tracks_rows, events_rows, division_rows,
            axis=axis,
        )

    return _frames(tracks_rows, events_rows, division_rows)


def _empty_lineage() -> tuple[Any, Any, Any]:
    import pandas as pd
    return (
        pd.DataFrame(columns=["track_id", "t", "label", "x", "y", "area", "axis_len"]),
        pd.DataFrame(columns=["t", "event", "src_label", "dst1_label", "dst2_label", "track_id", "cost", "dst_t"]),
        pd.DataFrame(columns=["t_div", "mother_track_id", "d1_track_id", "d2_track_id"]),
    )


def _frames(tracks_rows, events_rows, division_rows) -> tuple[Any, Any, Any]:
    import pandas as pd
    tracks_df = pd.DataFrame(tracks_rows) if tracks_rows else _empty_lineage()[0]
    events_df = pd.DataFrame(events_rows) if events_rows else _empty_lineage()[1]
    divisions_df = pd.DataFrame(division_rows) if division_rows else _empty_lineage()[2]
    return tracks_df, events_df, divisions_df
