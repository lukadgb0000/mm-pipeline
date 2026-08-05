"""Swimlane, dendrogram, and property-series plots over a :class:`Lineage`.

Every plotter takes a :class:`~mm_pipeline.analysis.tree.Lineage` (plus an
optional :class:`~mm_pipeline.analysis.selection.TrackSet` to highlight), accepts
``ax=None``, **returns the Axes**, and draws **one line per ``track_id`` tagged
with** ``line.set_gid(track_id)`` — the invariant the future picking layer attaches
to. Division connectors and legend proxies carry no gid.

matplotlib is imported lazily inside each function (mirroring ``tree.py``'s
lazy-pandas discipline) so importing ``mm_pipeline.analysis`` never requires it.
Position along the trench axis is read via ``lin.axis_col``, never a hardcoded
``"y"``, so ``axis="x"`` needs no special-casing. The x-axis is minutes when the
lineage carries a ``frame_interval_min`` and frames otherwise.
"""

from __future__ import annotations

from typing import Any, Optional

HIGHLIGHT_COLOR = "#d62728"
BASE_COLOR = "#666666"
CONNECTOR_COLOR = "#aaaaaa"
SERIES_COLOR = "#1f77b4"


def _new_ax(ax: Any, figsize: tuple[float, float]) -> Any:
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=figsize)
    return ax


def _time_axis(lin: Any) -> tuple[float, str]:
    """``(scale, xlabel)``. This is minutes when ``frame_interval_min`` is set, else frames"""
    if lin.frame_interval_min is not None:
        return float(lin.frame_interval_min), "time (min)"
    return 1.0, "frame t"


def _hot(track_id: int, highlight: Any) -> bool:
    return highlight is not None and int(track_id) in highlight


def plot_swimlane(lin: Any, highlight: Optional[Any] = None, ax: Any = None) -> Any:
    """Every track's spatial trajectory over time, with division connectors

    Each track is drawn as ``(t, axis_pos)``; the ``highlight`` TrackSet (e.g.
    ``mother_branch(lin)``) is drawn in the highlight colour. The axis is inverted
    when the closed end sits at the top (``open_end == "high"``)
    """
    ax = _new_ax(ax, (12, 5))
    col = lin.axis_col
    scale, xlabel = _time_axis(lin)

    for tid, frames in lin.frames_by_track.items():
        hot = _hot(tid, highlight)
        (line,) = ax.plot(
            frames["t"] * scale,
            frames[col],
            color=HIGHLIGHT_COLOR if hot else BASE_COLOR,
            lw=1.8 if hot else 0.6,
            alpha=0.95 if hot else 0.55,
        )
        line.set_gid(int(tid))

    for mother, (d1, d2) in lin.child_map.items():
        m_frames = lin.frames_by_track.get(mother)
        if m_frames is None:
            continue
        m_last = m_frames.iloc[-1]
        for d in (d1, d2):
            d_frames = lin.frames_by_track.get(d)
            if d_frames is None:
                continue
            d_first = d_frames.iloc[0]
            hot = _hot(mother, highlight) and _hot(d, highlight)
            ax.plot(
                [m_last["t"] * scale, d_first["t"] * scale],
                [m_last[col], d_first[col]],
                color=HIGHLIGHT_COLOR if hot else CONNECTOR_COLOR,
                lw=1.2 if hot else 0.5,
                ls="--",
                alpha=0.7 if hot else 0.4,
            )

    if lin.open_end == "high":
        ax.invert_yaxis()
        closed = "top"
    else:
        closed = "bottom"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{col} position (px) — closed end at {closed}")
    ax.set_title(
        f"{lin.dataset_id} — lineage swimlane "
        f"({len(lin.frames_by_track)} tracks, {len(lin.child_map)} divisions)"
    )
    if highlight is not None:
        ax.plot([], [], color=HIGHLIGHT_COLOR, lw=1.8, label=highlight.name or "highlight")
        ax.plot([], [], color=BASE_COLOR, lw=0.6, label="other tracks")
        ax.legend(loc="upper right")
    return ax


def plot_property_series(
    lin: Any, tracks: Any, prop: str = "axis_len", ax: Any = None, data: Any = None
) -> Any:
    """A per-cell property over a branch, marked at divisions and broken at gaps.

    ``tracks`` is a TrackSet — typically a branch (``mother_branch(lin)``,
    ``ancestors_of(...)``). Segments are concatenated in birth order; the line is
    broken wherever a track has a time gap (bridged frames) so those intervals are
    not drawn as real data.

    ``prop`` is read from ``tracks_df`` by default (``axis_len``, ``area``). To plot
    a regionprops property, pass ``data`` — a :func:`cell_properties`-shaped frame
    with ``track_id, t, <prop>`` — e.g.
    ``plot_property_series(lin, branch, prop="major_axis_length_px", data=cell_properties(lin, branch))``.
    """
    import numpy as np
    import pandas as pd

    ax = _new_ax(ax, (9, 4))
    scale, xlabel = _time_axis(lin)

    def _segment(tid: int) -> Any:
        if data is not None:
            return data[data["track_id"] == tid][["t", prop]].sort_values("t")
        return lin.frames_by_track[tid][["t", prop]]

    birth_t = dict(zip(lin.cycles["track_id"], lin.cycles["birth_t"]))
    ordered = sorted(
        (int(t) for t in tracks if int(t) in lin.frames_by_track),
        key=lambda t: birth_t.get(t, 0),
    )
    frames = pd.concat([_segment(t) for t in ordered], ignore_index=True)
    t = frames["t"].to_numpy(dtype=float)
    v = frames[prop].to_numpy(dtype=float)

    # Deals with a bridged track's internal gap by inserting NaN so the line is broken there
    gaps = np.nonzero(np.diff(t) > 1)[0] + 1
    t = np.insert(t, gaps, np.nan)
    v = np.insert(v, gaps, np.nan)
    ax.plot(t * scale, v, color=SERIES_COLOR, lw=1.0, marker="o", ms=4, alpha=0.9, label=prop)

    selected = set(ordered)
    div = lin.divisions_df
    div_ts = sorted(
        int(td) for m, td in zip(div["mother_track_id"], div["t_div"]) if int(m) in selected
    )
    for i, td in enumerate(div_ts):
        ax.axvline(
            td * scale, color=HIGHLIGHT_COLOR, ls="--", lw=1.0, alpha=0.4,
            label="division" if i == 0 else None,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{prop} (px)")
    ax.set_title(
        f"{lin.dataset_id} — {tracks.name or 'selection'}: {prop} "
        f"({len(ordered)} cell cycles)"
    )
    ax.legend(loc="upper right")
    return ax


def plot_dendrogram(lin: Any, highlight: Optional[Any] = None, ax: Any = None) -> Any:
    """Genealogy-first view: time on x, one layout band per subtree on y

    Each track is a horizontal segment from birth to end at its layout row;
    divisions are vertical connectors. Generation reads directly off the x-axis.
    Complements the swimlane (spatial truth) as the branch-selection view
    """
    import pandas as pd

    ax = _new_ax(ax, (12, 6))
    scale, xlabel = _time_axis(lin)
    child = lin.child_map
    cyc = lin.cycles.set_index("track_id")

    # leaves take sequential rows, parents centre on children
    y: dict[int, float] = {}
    next_row = [0]

    def place(tid: int) -> float:
        kids = [k for k in child.get(tid, ()) if k in cyc.index]
        if kids:
            y[tid] = sum(place(k) for k in kids) / len(kids)
        else:
            y[tid] = float(next_row[0])
            next_row[0] += 1
        return y[tid]

    roots = [int(t) for t in cyc.index if pd.isna(cyc.loc[t, "parent_id"])]
    for r in sorted(roots):
        place(r)

    for tid in (int(t) for t in cyc.index):
        row = cyc.loc[tid]
        hot = _hot(tid, highlight)
        (line,) = ax.plot(
            [row["birth_t"] * scale, row["end_t"] * scale],
            [y[tid], y[tid]],
            color=HIGHLIGHT_COLOR if hot else BASE_COLOR,
            lw=2.0 if hot else 1.0,
            solid_capstyle="round",
        )
        line.set_gid(tid)
        ax.annotate(str(tid), (row["end_t"] * scale, y[tid]), fontsize=6, va="center", ha="left")

    for _, row in lin.divisions_df.iterrows():
        m = int(row["mother_track_id"])
        for d in (int(row["d1_track_id"]), int(row["d2_track_id"])):
            if m in y and d in y:
                ax.plot([row["t_div"] * scale] * 2, [y[m], y[d]], color=CONNECTOR_COLOR, lw=0.6)

    gen_span = "" if cyc.empty else f", gen 0..{int(cyc['generation'].max())}"
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("lineage layout (one band per subtree)")
    ax.set_title(f"{lin.dataset_id} — lineage dendrogram ({len(y)} tracks{gen_span})")
    return ax
