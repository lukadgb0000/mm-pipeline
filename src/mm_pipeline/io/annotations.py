"""Ground-truth annotation IO helpers"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mm_pipeline.core import CellInstance, TrackingOperation


@dataclass(frozen=True)
class Division:
    t_div: int
    mother: int
    d1: int
    d2: int


@dataclass(frozen=True)
class GTContext:
    frame_label_to_track: dict[int, dict[int, int]]
    frame_track_to_label: dict[int, dict[int, int]]
    division_by_mother: dict[int, Division]


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Reading ground-truth annotation CSVs requires pandas.") from exc
    return pd


def _resolve_col(df: Any, candidates: tuple[str, ...], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing {label} column. Tried {list(candidates)}; found {list(df.columns)}")


def load_gt_context(gt_tracks_csv: str | Path, gt_divisions_csv: str | Path) -> GTContext:
    """Load saved track/division CSVs into lookup maps for GT operation building"""

    pd = _require_pandas()
    tracks_path = Path(gt_tracks_csv)
    divisions_path = Path(gt_divisions_csv)
    tracks = pd.read_csv(tracks_path)
    divisions = pd.read_csv(divisions_path)

    tid_col = _resolve_col(tracks, ("track_id",), "track id")
    t_col = _resolve_col(tracks, ("t",), "time index")
    label_col = _resolve_col(tracks, ("label",), "label")

    tracks = tracks[[tid_col, t_col, label_col]].copy()
    tracks.columns = ["track_id", "t", "label"]
    tracks["track_id"] = tracks["track_id"].astype(int)
    tracks["t"] = tracks["t"].astype(int)
    tracks["label"] = tracks["label"].astype(int)

    if tracks.duplicated(subset=["t", "label"]).any():
        raise ValueError(f"Duplicate (t,label) rows in {tracks_path}")

    frame_label_to_track: dict[int, dict[int, int]] = {}
    frame_track_to_label: dict[int, dict[int, int]] = {}
    for row in tracks.itertuples(index=False):
        frame_label_to_track.setdefault(int(row.t), {})[int(row.label)] = int(row.track_id)
        frame_track_to_label.setdefault(int(row.t), {})[int(row.track_id)] = int(row.label)

    m_col = _resolve_col(divisions, ("mother_track_id",), "mother track id")
    td_col = _resolve_col(divisions, ("t_div",), "division time")
    d1_col = _resolve_col(divisions, ("d1_track_id", "daughter1_track_id"), "daughter 1 track id")
    d2_col = _resolve_col(divisions, ("d2_track_id", "daughter2_track_id"), "daughter 2 track id")

    divisions = divisions[[td_col, m_col, d1_col, d2_col]].copy()
    divisions.columns = ["t_div", "mother_track_id", "d1_track_id", "d2_track_id"]
    division_by_mother: dict[int, Division] = {}
    for row in divisions.itertuples(index=False):
        mother = int(row.mother_track_id)
        if mother in division_by_mother:
            raise ValueError(f"Mother track {mother} has multiple division rows in {divisions_path}")
        division_by_mother[mother] = Division(
            t_div=int(row.t_div),
            mother=mother,
            d1=int(row.d1_track_id),
            d2=int(row.d2_track_id),
        )

    return GTContext(
        frame_label_to_track=frame_label_to_track,
        frame_track_to_label=frame_track_to_label,
        division_by_mother=division_by_mother,
    )


def build_gt_ops_for_pair(
    t: int,
    cells_t: list[CellInstance] | tuple[CellInstance, ...],
    gt: GTContext,
) -> list[TrackingOperation]:
    """Build GT operations for one sorted source frame. Bridging/non-adjacent workflows need to generalise this around
    FramePair.k, remember that not always k = t + 1
    """

    k_frame = int(t) + 1
    map_t = gt.frame_label_to_track.get(int(t), {})
    map_k = gt.frame_track_to_label.get(k_frame, {})

    ops: list[TrackingOperation] = []
    for source in cells_t:
        src_label = int(source.label)
        src_track = map_t.get(src_label)
        if src_track is None:
            raise ValueError(f"Missing GT track_id for label {src_label} at t={t}.")

        div = gt.division_by_mother.get(int(src_track))
        if div is not None and int(div.t_div) == int(t):
            d1_label = map_k.get(int(div.d1))
            d2_label = map_k.get(int(div.d2))
            if d1_label is None or d2_label is None:
                raise ValueError(
                    f"Missing daughter label at t={k_frame} for mother track {src_track} at t_div={t}."
                )
            ops.append(TrackingOperation("divide", src_label, int(d1_label), int(d2_label)))
            continue

        dst_label = map_k.get(int(src_track))
        if dst_label is None:
            ops.append(TrackingOperation("exit", src_label, None, None))
        else:
            ops.append(TrackingOperation("link", src_label, int(dst_label), None))

    return ops
