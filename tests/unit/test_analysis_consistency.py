"""Tests for the post-reconstruction division/length consistency diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from mm_pipeline.analysis import Lineage, division_length_consistency
from mm_pipeline.analysis.consistency import (
    DIVISION_WITHOUT_DROP,
    DROP_WITHOUT_DIVISION,
    _touches_open_edge,
)
from mm_pipeline.core import FramePair, extract_cell_instances
from mm_pipeline.io.labels import save_label_stack
from mm_pipeline.tracking.costs import touches_open


def _lineage(tracks, divisions=(), *, labels_dir=None, axis="y", open_end="high"):
    tracks_df = pd.DataFrame(
        tracks,
        columns=["track_id", "t", "label", "x", "y", "area", "axis_len"],
    )
    divisions_df = pd.DataFrame(
        divisions,
        columns=["t_div", "mother_track_id", "d1_track_id", "d2_track_id"],
    )
    events_df = pd.DataFrame(
        columns=["t", "event", "src_label", "dst1_label", "dst2_label", "track_id"]
    )
    return Lineage(
        dataset_id="d1",
        tracks_df=tracks_df,
        divisions_df=divisions_df,
        events_df=events_df,
        labels_dir=labels_dir,
        axis=axis,
        open_end=open_end,
    )


def _track_row(track_id, t, label, length):
    return (track_id, t, label, 3.0, 10.0, float(length * 4), float(length))


def test_flags_drop_without_division_and_division_without_drop():
    lin = _lineage(
        [
            _track_row(1, 0, 1, 10),
            _track_row(1, 1, 1, 5),
            _track_row(2, 0, 2, 10),
            _track_row(3, 1, 3, 9),
            _track_row(4, 1, 4, 4),
        ],
        [(0, 2, 3, 4)],
    )

    flags = division_length_consistency(lin, open_end_margin=None)

    assert flags["kind"].tolist() == [DROP_WITHOUT_DIVISION, DIVISION_WITHOUT_DROP]
    drop = flags.iloc[0]
    assert (drop["t"], drop["t_next"], drop["src_label"], drop["dst1_label"]) == (0, 1, 1, 1)
    assert drop["drop_frac"] == pytest.approx(0.5)
    division = flags.iloc[1]
    assert (division["src_label"], division["dst1_label"], division["dst2_label"]) == (2, 3, 4)
    assert division["drop_frac"] == pytest.approx(0.1)


def test_threshold_and_external_property_data():
    lin = _lineage([_track_row(1, 0, 1, 10), _track_row(1, 1, 1, 8)])
    assert division_length_consistency(
        lin, min_division_drop=0.3, open_end_margin=None
    ).empty

    data = pd.DataFrame(
        {"track_id": [1, 1], "t": [0, 1], "label": [1, 1], "fitted": [10.0, 5.0]}
    )
    flags = division_length_consistency(
        lin,
        min_division_drop=0.3,
        open_end_margin=None,
        prop="fitted",
        data=data,
    )
    assert flags["kind"].tolist() == [DROP_WITHOUT_DIVISION]


def test_bridged_gap_is_not_treated_as_one_frame_drop():
    lin = _lineage([_track_row(1, 0, 1, 10), _track_row(1, 2, 1, 4)])
    assert division_length_consistency(lin, open_end_margin=None).empty


def test_empty_lineage_returns_stable_empty_schema():
    flags = division_length_consistency(_lineage([]), open_end_margin=None)
    assert flags.empty
    assert {"kind", "t", "t_next", "src_label", "dst1_label", "drop_frac"} <= set(
        flags.columns
    )


def test_open_end_filter_removes_censored_drop(tmp_path):
    labels = np.zeros((2, 20, 8), dtype=np.uint32)
    labels[0, 14:20, 2:6] = 1
    labels[1, 17:20, 2:6] = 1
    labels_dir = tmp_path / "labels"
    save_label_stack(labels, ["f0.tif", "f1.tif"], labels_dir)
    lin = _lineage(
        [_track_row(1, 0, 1, 6), _track_row(1, 1, 1, 3)],
        labels_dir=labels_dir,
    )

    assert len(division_length_consistency(lin, open_end_margin=None)) == 1
    assert division_length_consistency(lin, open_end_margin=0).empty


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("open_end", ["low", "high"])
@pytest.mark.parametrize("margin", [0, 2])
def test_open_edge_predicate_matches_tracker(axis, open_end, margin):
    img = np.zeros((20, 12), dtype=np.uint32)
    if axis == "y":
        rows = slice(0, 5) if open_end == "low" else slice(15, 20)
        img[rows, 3:8] = 1
    else:
        cols = slice(0, 5) if open_end == "low" else slice(7, 12)
        img[4:12, cols] = 1
    cell = extract_cell_instances(img)[0]
    pair = FramePair("d1", 0, 1, img.shape, axis, open_end)

    assert _touches_open_edge(img, 1, axis, open_end, margin) == touches_open(
        cell, pair, margin
    )
