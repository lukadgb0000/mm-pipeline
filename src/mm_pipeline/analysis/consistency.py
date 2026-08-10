"""Division + length-drop consistency check over a reconstructed :class:`Lineage`.

A post-tracking diagnostic. Same contract as the rest of ``mm_pipeline.analysis``: 
reads the reconstructed tables via ``Lineage`` and imports nothing from ``tracking``/``features``/``core``).

The physical model: a cell grows within its cycle and its length drops sharply
(~halving) at division. Two deviations from this model are flagged:

- a large negative length step with no recorded division
- a recorded division with no large length drop

In the reconstructed representation each ``track_id`` is one cell cycle, so the
division drop is a *track-boundary* transition (mother's last frame ``t`` ->
daughters' first frame ``t_next``). The two checks below
handle those two cases separately over the same length signal.

Each flag is reported as its underlying transition across a frame pair
``(t, t_next)``, naming the source cell and destination cell(s) by both
``track_id`` and segmentation mask ``label`` (the per-frame integer mask value —
what you search for in napari) so a finding can be located exactly:

- ``drop_without_division``: a link ``a (t) -> b (t_next)`` within one track:
  ``src_*`` is cell a, ``dst1_*`` is cell b (``dst2_*`` is empty).
- ``division_without_drop``: a division ``a (t) -> b, c (t_next)``:
  ``src_*`` is mother a, ``dst1_*``/``dst2_*`` are daughters b and c.

Cells exiting through the trench's open end shrink as they leave, which mimics a
length drop; those flags are not legitimate errors but rather
censoring by the open boundary of the trench. By default they are
excluded (``open_end_margin``), mirroring how the tracker excludes open-end-touching
masks from its DP shrink term.
"""

from __future__ import annotations

from typing import Any, Optional

FLAG_COLUMNS = [
    "dataset_id",
    "kind",
    "t",
    "t_next",
    "src_track_id",
    "src_label",
    "dst1_track_id",
    "dst1_label",
    "dst2_track_id",
    "dst2_label",
    "length_before",
    "length_after",
    "drop_frac",
]

DROP_WITHOUT_DIVISION = "drop_without_division"
DIVISION_WITHOUT_DROP = "division_without_drop"


def division_length_consistency(
    lin: Any,
    *,
    min_division_drop: float = 0.3,
    open_end_margin: Optional[int] = 0,
    prop: str = "axis_len",
    data: Any = None,
) -> Any:
    """Flag divisions and length steps that disagree with the growth model.

    Parameters
    lin : Lineage
        A reconstructed lineage (from ``track-select`` or ``modelvio``).
    min_division_drop : float, default 0.3
        The fractional length drop that "counts as" a division, ``tau``. A single
        threshold used both ways: a within-track step ``>= tau`` is division-sized
        (suspicious with no division), and a division whose boundary drop is
        ``< tau`` lacks the expected halving (suspicious division). Relative, so it
        is scale-free; ``tau`` around 0.3 sits between per-frame growth (a few %)
        and a real halving (~50%).
    open_end_margin : int or None, default 0
        Suppress a flag when a cell involved has a mask touching the OPEN-END
        boundary — a cell leaving the trench shrinks as it exits, which mimics a
        length drop (this is censoring, not an anomaly). ``0`` excludes cells whose
        mask reaches the boundary; ``d > 0`` excludes cells within ``d`` px of it;
        ``None`` disables the exclusion (and reads no label images). Mirrors the
        tracker's own ``touches_open`` handling of the DP shrink term
        (:func:`mm_pipeline.tracking.costs.touches_open`). Requires
        ``lin.labels_dir`` when not ``None``.
    prop : str, default "axis_len"
        Length column, read from ``lin.tracks_df`` by default. Mirrors
        :func:`mm_pipeline.analysis.plotting.plot_property_series`.
    data : DataFrame, optional
        A :func:`mm_pipeline.analysis.properties.cell_properties`-shaped frame
        (``track_id, t, label, <prop>``) to use a regionprops-fitted length such as
        ``major_axis_length_px`` instead of the tracker's ``axis_len``.

    Returns
    pandas.DataFrame
        One row per flag with columns :data:`FLAG_COLUMNS`, sorted by ``t`` then
        ``src_track_id``; empty (with those columns) when nothing is flagged.
        Each row describes a transition across the frame pair ``(t, t_next)``:
        ``src_track_id``/``src_label`` name the source cell in frame ``t`` and
        ``dst1_*``/``dst2_*`` the destination cell(s) in frame ``t_next``, where
        ``label`` is the segmentation mask integer id in that frame. ``drop_frac``
        is ``1 - length_after / length_before`` (for a division, against the
        largest — least-shrunk — daughter, which is what triggers the flag).
        Flags whose cells touch the open-end boundary are removed (see
        ``open_end_margin``).
    """

    import pandas as pd

    if open_end_margin is not None and getattr(lin, "labels_dir", None) is None:
        raise ValueError(
            "open_end_margin requires label images but lin.labels_dir is None. "
            "Build the Lineage from a spec with a labels directory, or pass "
            "open_end_margin=None to disable open-end exclusion."
        )

    tau = float(min_division_drop)
    series_by_track = _series_by_track(lin, prop=prop, data=data)
    records: list[dict[str, Any]] = []

    # 1. Within-track: a division-sized drop between consecutive frames of one track.
    # The link is a (frame t) -> b (frame t+1) of the SAME track_id.
    for tid, series in series_by_track.items():
        ts, vs, labs = series["t"], series["v"], series["lab"]
        for i in range(len(ts) - 1):
            if int(ts[i + 1]) - int(ts[i]) != 1:
                continue  # bridged gap: not a per-frame step, skip
            before, after = vs[i], vs[i + 1]
            drop = _drop_frac(before, after)
            if drop is None or drop < tau:
                continue
            records.append(
                {
                    "dataset_id": lin.dataset_id,
                    "kind": DROP_WITHOUT_DIVISION,
                    "t": int(ts[i]),
                    "t_next": int(ts[i + 1]),
                    "src_track_id": int(tid),
                    "src_label": _as_int(labs[i]),
                    "dst1_track_id": int(tid),
                    "dst1_label": _as_int(labs[i + 1]),
                    "dst2_track_id": pd.NA,
                    "dst2_label": pd.NA,
                    "length_before": float(before),
                    "length_after": float(after),
                    "drop_frac": float(drop),
                }
            )

    # 2. Division boundary: a recorded division whose largest daughter barely drops in length.
    # The transition is mother a (frame t_div) -> daughters b, c (frame t_next).
    for _, row in lin.divisions_df.iterrows():
        mother = int(row["mother_track_id"])
        m_series = series_by_track.get(mother)
        if m_series is None or not len(m_series["v"]):
            continue
        mother_last = m_series["v"][-1]
        if not _positive(mother_last):
            continue

        d1_id, d2_id = int(row["d1_track_id"]), int(row["d2_track_id"])
        d1_series = series_by_track.get(d1_id)
        d2_series = series_by_track.get(d2_id)

        daughter_lens = [
            ds["v"][0]
            for ds in (d1_series, d2_series)
            if ds is not None and len(ds["v"]) and _positive(ds["v"][0])
        ]
        if not daughter_lens:
            continue

        # Largest daughter = smallest drop = the one that looks like continuation.
        drop = _drop_frac(mother_last, max(daughter_lens))
        if drop is None or drop >= tau:
            continue

        records.append(
            {
                "dataset_id": lin.dataset_id,
                "kind": DIVISION_WITHOUT_DROP,
                "t": int(row["t_div"]),
                "t_next": _daughter_birth_t(d1_series, d2_series, int(row["t_div"]) + 1),
                "src_track_id": mother,
                "src_label": _as_int(m_series["lab"][-1]),
                "dst1_track_id": d1_id,
                "dst1_label": _first_label(d1_series),
                "dst2_track_id": d2_id,
                "dst2_label": _first_label(d2_series),
                "length_before": float(mother_last),
                "length_after": float(max(daughter_lens)),
                "drop_frac": float(drop),
            }
        )

    # Drop flags that are open-end censoring: a cell exiting the trench shrinks,
    # which mimics a length drop. Mirrors the tracker's touches_open handling.
    if open_end_margin is not None and records:
        touches = _make_open_end_tester(lin, int(open_end_margin))
        records = [r for r in records if not _record_touches_open(r, touches)]

    if not records:
        return pd.DataFrame(columns=FLAG_COLUMNS)
    return (
        pd.DataFrame.from_records(records, columns=FLAG_COLUMNS)
        .sort_values(["t", "src_track_id"])
        .reset_index(drop=True)
    )


def _series_by_track(lin: Any, *, prop: str, data: Any) -> dict[int, dict[str, Any]]:
    """``{track_id: {"t": [...], "v": [...], "lab": [...]}}`` sorted by ``t``.

    ``lab`` is the per-detection segmentation ``label`` (``None`` where the source
    frame carries no ``label`` column). Reads ``prop`` from ``data`` when given
    (grouped by ``track_id``), else from the lineage's per-track ``tracks_df`` slices.
    """

    def _labels(frame: Any, n: int) -> list[Any]:
        return frame["label"].to_list() if "label" in frame.columns else [None] * n

    out: dict[int, dict[str, Any]] = {}
    if data is not None:
        for tid, sub in data.groupby("track_id"):
            sub = sub.sort_values("t")
            out[int(tid)] = {
                "t": sub["t"].to_list(),
                "v": sub[prop].to_list(),
                "lab": _labels(sub, len(sub)),
            }
        return out

    for tid, frames in lin.frames_by_track.items():
        # frames_by_track slices are already time-sorted.
        out[int(tid)] = {
            "t": frames["t"].to_list(),
            "v": frames[prop].to_list(),
            "lab": _labels(frames, len(frames)),
        }
    return out


def _daughter_birth_t(d1_series: Any, d2_series: Any, fallback: int) -> int:
    """First observed frame of either daughter (``t_div + 1`` normally)."""

    for ds in (d1_series, d2_series):
        if ds is not None and len(ds["t"]):
            return int(ds["t"][0])
    return int(fallback)


def _first_label(series: Any) -> Any:
    import pandas as pd

    if series is None or not len(series["lab"]):
        return pd.NA
    return _as_int(series["lab"][0])


def _record_touches_open(rec: dict[str, Any], touches: Any) -> bool:
    """True if any cell in the flag's length comparison touches the open end.

    Mirrors the tracker's rule (``features.pairwise``): a transition is at the open
    end when *either* endpoint is. For a drop that is the source and destination;
    for a division the mother and both daughters.
    """

    cells = [(rec["t"], rec["src_label"]), (rec["t_next"], rec["dst1_label"])]
    if rec["kind"] == DIVISION_WITHOUT_DROP:
        cells.append((rec["t_next"], rec["dst2_label"]))
    return any(touches(frame_idx, label) for frame_idx, label in cells)


def _make_open_end_tester(lin: Any, margin: int) -> Any:
    """Return ``touches(frame_idx, label) -> bool`` for the open-end boundary.

    Recomputes the mask's open-end-facing bbox edge from the label images (the
    reconstructed ``tracks_df`` keeps no bbox), caching each frame it reads. Only
    the frames referenced by candidate flags get read.
    """

    import pandas as pd

    from mm_pipeline.io.labels import collect_label_paths, read_label

    paths = collect_label_paths(lin.labels_dir)
    axis, open_end = lin.axis, lin.open_end
    cache: dict[int, Any] = {}

    def touches(frame_idx: Any, label: Any) -> bool:
        if label is None or pd.isna(label):
            return False
        fi = int(frame_idx)
        if fi < 0 or fi >= len(paths):
            return False
        img = cache.get(fi)
        if img is None:
            img = read_label(paths[fi])
            cache[fi] = img
        return _touches_open_edge(img, int(label), axis, open_end, margin)

    return touches


def _touches_open_edge(img: Any, label: int, axis: str, open_end: str, margin: int) -> bool:
    """Whether ``label``'s mask edge is within ``margin`` px of the open end.

    Faithful to :func:`mm_pipeline.tracking.costs.touches_open`, including its
    exclusive bbox-max convention (``max()+1``), so a cell the tracker treats as
    open-end-touching is treated the same here.
    """

    import numpy as np

    rows, cols = np.nonzero(img == label)
    if rows.size == 0:
        return False
    h, w = int(img.shape[0]), int(img.shape[1])
    if axis == "y":
        if open_end == "low":
            return int(rows.min()) <= margin
        return int(rows.max()) + 1 >= (h - 1 - margin)
    if open_end == "low":
        return int(cols.min()) <= margin
    return int(cols.max()) + 1 >= (w - 1 - margin)


def _positive(value: Any) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return v == v and v > 0.0  # v == v rejects NaN


def _drop_frac(before: Any, after: Any) -> Optional[float]:
    """``1 - after/before`` (positive = a drop), or ``None`` if not computable."""

    if not _positive(before) or after is None:
        return None
    try:
        a = float(after)
    except (TypeError, ValueError):
        return None
    if a != a:  # NaN
        return None
    return 1.0 - a / float(before)


def _as_int(value: Any) -> Any:
    """Return ``int(value)`` or ``pandas.NA`` when it is missing/non-numeric."""

    import pandas as pd

    if value is None:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return pd.NA


__all__ = [
    "division_length_consistency",
    "FLAG_COLUMNS",
    "DROP_WITHOUT_DIVISION",
    "DIVISION_WITHOUT_DROP",
]
