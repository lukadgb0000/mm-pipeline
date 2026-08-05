"""Unit tests for mm_pipeline.analysis.plotting (swimlane, series, dendrogram)."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")  # headless; no conftest, matplotlib not in the `test` extra

import numpy as np

from mm_pipeline.analysis import (
    Lineage,
    mother_branch,
    plot_dendrogram,
    plot_property_series,
    plot_swimlane,
)
from mm_pipeline.analysis.plotting import HIGHLIGHT_COLOR
from mm_pipeline.analysis.selection import TrackSet

pd = pytest.importorskip("pandas")


# --- a dividing lineage with distinct birth positions and varying length ------
#
#   1 (y=5) --div@1--> 2 (y=2), 3 (y=8)
#   2       --div@3--> 4 (y=1), 5 (y=4)
# open_end="high" => closed end at low y => mother lineage = 1 -> 2 -> 4.


def _rows(track_id, samples):
    return [
        {"track_id": track_id, "t": t, "label": 1, "x": 2.0, "y": float(y), "area": 10.0, "axis_len": float(length)}
        for t, y, length in samples
    ]


def _dividing_lineage(axis="y", open_end="high", frame_interval_min=None):
    tracks = pd.DataFrame(
        _rows(1, [(0, 5, 3.0), (1, 5, 4.0)])
        + _rows(2, [(2, 2, 2.0), (3, 2, 3.0)])
        + _rows(3, [(2, 8, 2.5), (3, 8, 3.5), (4, 8, 4.5)])
        + _rows(4, [(4, 1, 1.0), (5, 1, 2.0)])
        + _rows(5, [(4, 4, 1.5), (5, 4, 2.5)])
    )
    divisions = pd.DataFrame(
        [
            {"t_div": 1, "mother_track_id": 1, "d1_track_id": 2, "d2_track_id": 3},
            {"t_div": 3, "mother_track_id": 2, "d1_track_id": 4, "d2_track_id": 5},
        ]
    )
    events = pd.DataFrame(columns=["t", "event", "src_label", "dst1_label", "dst2_label", "track_id", "cost", "dst_t"])
    return Lineage(
        dataset_id="d1", tracks_df=tracks, divisions_df=divisions, events_df=events,
        axis=axis, open_end=open_end, frame_interval_min=frame_interval_min,
    )


def _gapped_lineage():
    # A single bridged track missing t=2, no divisions.
    tracks = pd.DataFrame(_rows(1, [(0, 5, 1.0), (1, 5, 2.0), (3, 5, 4.0)]))
    divisions = pd.DataFrame(columns=["t_div", "mother_track_id", "d1_track_id", "d2_track_id"])
    events = pd.DataFrame(columns=["t", "event", "src_label", "dst1_label", "dst2_label", "track_id", "cost", "dst_t"])
    return Lineage(dataset_id="d1", tracks_df=tracks, divisions_df=divisions, events_df=events)


def _gids(ax):
    return {ln.get_gid() for ln in ax.get_lines() if ln.get_gid() is not None}


# --- swimlane -----------------------------------------------------------------


def test_swimlane_returns_axes_and_tags_each_track_once():
    from matplotlib.axes import Axes

    ax = plot_swimlane(_dividing_lineage())
    assert isinstance(ax, Axes)
    # One tagged line per track; connectors carry no gid.
    assert _gids(ax) == {1, 2, 3, 4, 5}


def test_swimlane_inverts_y_only_for_high_open_end():
    hi = plot_swimlane(_dividing_lineage(open_end="high"))
    lo = plot_swimlane(_dividing_lineage(open_end="low"))
    assert hi.get_ylim()[0] > hi.get_ylim()[1]  # inverted
    assert lo.get_ylim()[0] < lo.get_ylim()[1]  # normal


def test_swimlane_highlights_branch_in_highlight_colour():
    lin = _dividing_lineage()
    branch = mother_branch(lin)  # {1, 2, 4}
    ax = plot_swimlane(lin, highlight=branch)
    coloured = {ln.get_gid(): ln.get_color() for ln in ax.get_lines() if ln.get_gid() is not None}
    for tid in branch:
        assert coloured[tid] == HIGHLIGHT_COLOR
    assert coloured[3] != HIGHLIGHT_COLOR  # off-branch


# --- property series ----------------------------------------------------------


def test_property_series_marks_each_in_branch_division():
    lin = _dividing_lineage()
    ax = plot_property_series(lin, mother_branch(lin))  # mothers 1 and 2 divide
    verticals = [ln for ln in ax.get_lines() if len(set(ln.get_xdata())) == 1]
    assert len(verticals) == 2


def test_property_series_breaks_line_at_gaps():
    lin = _gapped_lineage()
    tracks = TrackSet("d1", frozenset({1}), "gapped")
    ax = plot_property_series(lin, tracks, prop="axis_len")
    ydata = ax.get_lines()[0].get_ydata()
    assert np.isnan(np.asarray(ydata, dtype=float)).any()


def test_property_series_x_in_minutes_when_interval_set():
    lin = _dividing_lineage(frame_interval_min=2.0)
    ax = plot_property_series(lin, mother_branch(lin), prop="axis_len")
    assert ax.get_xlabel() == "time (min)"
    xdata = np.concatenate([ln.get_xdata() for ln in ax.get_lines()])
    assert np.nanmax(xdata) == pytest.approx(5 * 2.0)  # last frame t=5 -> 10 min


def test_property_series_reproduces_branch_axis_len_in_time_order():
    lin = _dividing_lineage()
    ax = plot_property_series(lin, mother_branch(lin), prop="axis_len")
    plotted = np.asarray(ax.get_lines()[0].get_ydata(), dtype=float)
    plotted = plotted[~np.isnan(plotted)]
    # Branch 1 -> 2 -> 4, concatenated in time order.
    expected = [3.0, 4.0, 2.0, 3.0, 1.0, 2.0]
    assert plotted.tolist() == expected


# --- dendrogram ---------------------------------------------------------------


def test_dendrogram_returns_axes_and_tags_all_tracks():
    from matplotlib.axes import Axes

    ax = plot_dendrogram(_dividing_lineage())
    assert isinstance(ax, Axes)
    assert _gids(ax) == {1, 2, 3, 4, 5}


def test_dendrogram_places_leaves_on_distinct_rows():
    lin = _dividing_lineage()
    ax = plot_dendrogram(lin)
    leaf_rows = {ln.get_gid(): ln.get_ydata()[0] for ln in ax.get_lines() if ln.get_gid() in (3, 4, 5)}
    assert len(set(leaf_rows.values())) == 3  # 3, 4, 5 are the leaves
