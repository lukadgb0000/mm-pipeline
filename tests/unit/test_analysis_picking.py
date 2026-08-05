"""Unit tests for mm_pipeline.analysis.picking (two-click branch selection)."""

from __future__ import annotations

import types

import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")  # headless

from mm_pipeline.analysis import Lineage, path_between, pick_tracks, plot_swimlane
from mm_pipeline.analysis.plotting import BASE_COLOR, HIGHLIGHT_COLOR

pd = pytest.importorskip("pandas")


#   1 --div@1--> 2, 3 ;  2 --div@3--> 4, 5
#   => 1 -> 2 -> 4 is one lineage path; 3 and 5 are off it.


def _rows(track_id, ty_pairs):
    return [
        {"track_id": track_id, "t": t, "label": 1, "x": 2.0, "y": float(y), "area": 10.0, "axis_len": 5.0}
        for t, y in ty_pairs
    ]


def _dividing_lineage():
    tracks = pd.DataFrame(
        _rows(1, [(0, 5), (1, 5)])
        + _rows(2, [(2, 2), (3, 2)])
        + _rows(3, [(2, 8), (3, 8), (4, 8)])
        + _rows(4, [(4, 1), (5, 1)])
        + _rows(5, [(4, 4), (5, 4)])
    )
    divisions = pd.DataFrame(
        [
            {"t_div": 1, "mother_track_id": 1, "d1_track_id": 2, "d2_track_id": 3},
            {"t_div": 3, "mother_track_id": 2, "d1_track_id": 4, "d2_track_id": 5},
        ]
    )
    events = pd.DataFrame(columns=["t", "event", "src_label", "dst1_label", "dst2_label", "track_id", "cost", "dst_t"])
    return Lineage(dataset_id="d1", tracks_df=tracks, divisions_df=divisions, events_df=events)


def _line(ax, gid):
    return next(ln for ln in ax.get_lines() if ln.get_gid() == gid)


def _click(selector, ax, gid):
    selector._on_pick(types.SimpleNamespace(artist=_line(ax, gid)))


def _colors(ax):
    return {ln.get_gid(): ln.get_color() for ln in ax.get_lines() if ln.get_gid() is not None}


# --- wiring -------------------------------------------------------------------


def test_pick_tracks_makes_only_tagged_lines_pickable():
    lin = _dividing_lineage()
    ax = plot_swimlane(lin)
    pick_tracks(lin, ax)
    for line in ax.get_lines():
        if line.get_gid() is not None:
            assert line.get_picker()
        else:
            assert not line.get_picker()


# --- two-click branch selection ----------------------------------------------


def test_first_click_sets_pending_start_and_highlights_it():
    lin = _dividing_lineage()
    ax = plot_swimlane(lin)
    selector = pick_tracks(lin, ax, echo=False)

    _click(selector, ax, 1)
    assert selector.selection is None  # not complete until the second click
    assert selector._start == 1
    colors = _colors(ax)
    assert colors[1] == HIGHLIGHT_COLOR
    assert colors[2] == BASE_COLOR and colors[3] == BASE_COLOR


def test_two_clicks_select_the_branch_and_highlight_it():
    lin = _dividing_lineage()
    ax = plot_swimlane(lin)
    selector = pick_tracks(lin, ax, echo=False)

    _click(selector, ax, 1)
    _click(selector, ax, 4)
    assert set(selector.selection) == {1, 2, 4}       # the lineage path 1 -> 2 -> 4
    assert set(selector.selection) == set(path_between(lin, 1, 4))
    assert selector._start is None                    # ready for the next branch
    assert len(selector.history) == 1

    colors = _colors(ax)
    assert all(colors[t] == HIGHLIGHT_COLOR for t in (1, 2, 4))
    assert all(colors[t] == BASE_COLOR for t in (3, 5))


def test_off_lineage_second_click_restarts():
    lin = _dividing_lineage()
    ax = plot_swimlane(lin)
    selector = pick_tracks(lin, ax, echo=False)

    _click(selector, ax, 3)
    _click(selector, ax, 4)  # 3 and 4 are on different branches
    assert selector.selection is None
    assert selector._start == 4               # restarted at the second click
    assert _colors(ax)[4] == HIGHLIGHT_COLOR


def test_echo_prints_the_declarative_path(capsys):
    lin = _dividing_lineage()
    ax = plot_swimlane(lin)
    selector = pick_tracks(lin, ax)  # echo=True
    _click(selector, ax, 1)
    _click(selector, ax, 4)
    assert "path_between(lin, 1, 4)" in capsys.readouterr().out


def test_on_pick_ignores_untagged_artists():
    lin = _dividing_lineage()
    ax = plot_swimlane(lin)
    selector = pick_tracks(lin, ax, echo=False)
    selector._on_pick(types.SimpleNamespace(artist=types.SimpleNamespace(get_gid=lambda: None)))
    assert selector.selection is None and selector._start is None
