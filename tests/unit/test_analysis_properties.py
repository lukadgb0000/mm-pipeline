"""Unit tests for mm_pipeline.analysis.properties (regionprops join)."""

from __future__ import annotations

import pytest

pytest.importorskip("skimage")
pd = pytest.importorskip("pandas")

import numpy as np

from mm_pipeline.analysis import Lineage, cell_properties, roots
from mm_pipeline.analysis.selection import TrackSet
from mm_pipeline.config import TrackerParams
from mm_pipeline.tracking.lineage import reconstruct_lineage
from mm_pipeline.tracking.select import DPCostMin, select_pairs

EXPECTED_COLUMNS = [
    "dataset_id", "track_id", "t", "label",
    "major_axis_length_px", "orientation", "eccentricity", "solidity",
    "bbox_len", "area",
]


def px_count(regionmask):
    """An extra_property: pixel count (equals regionprops area)."""
    return int(np.sum(regionmask))


def _stable_label_stack() -> np.ndarray:
    # Two cells, constant geometry over 5 frames: a 6x5 (30 px) and an 8x5 (40 px).
    labels = np.zeros((5, 40, 8), dtype=np.int32)
    for t in range(5):
        labels[t, 2:8, 1:6] = 1
        labels[t, 14:22, 1:6] = 2
    return labels


def _labelled_lineage(tmp_path, frame_interval_min=None) -> Lineage:
    from mm_pipeline.features import build_feature_table_for_stack
    from mm_pipeline.io.labels import save_label_stack

    labels = _stable_label_stack()
    names = [f"f{t:03d}.tif" for t in range(labels.shape[0])]
    save_label_stack(labels, names, tmp_path, overwrite=True)

    features = build_feature_table_for_stack(
        labels, dataset_id="d1", axis="y", open_end="high",
        params=TrackerParams(), top_k=4, store_ops=True,
    )
    selections = select_pairs(features, DPCostMin())
    tracks, events, divisions = reconstruct_lineage(selections, labels, open_end="high", axis="y")
    return Lineage(
        dataset_id="d1", tracks_df=tracks, divisions_df=divisions, events_df=events,
        axis="y", open_end="high", frame_interval_min=frame_interval_min, labels_dir=tmp_path,
    )


def test_columns_and_row_grain(tmp_path):
    lin = _labelled_lineage(tmp_path)
    props = cell_properties(lin)
    assert list(props.columns) == EXPECTED_COLUMNS
    # Inner join keeps exactly the real tracks_df rows (2 tracks * 5 frames).
    assert len(props) == len(lin.tracks_df) == 10
    assert (props["dataset_id"] == "d1").all()


def test_area_matches_tracker_and_bbox_len_from_axis_len(tmp_path):
    lin = _labelled_lineage(tmp_path)
    props = cell_properties(lin)
    assert sorted(props["area"].unique()) == [30.0, 40.0]        # tracker pixel counts
    assert sorted(props["bbox_len"].unique()) == [6.0, 8.0]      # axis_len (bbox height)
    assert (props["major_axis_length_px"] > 0).all()
    assert np.isfinite(props["major_axis_length_px"]).all()


def test_trackset_scoping_limits_rows(tmp_path):
    lin = _labelled_lineage(tmp_path)
    one = TrackSet(lin.dataset_id, frozenset({next(iter(lin.frames_by_track))}))
    props = cell_properties(lin, one)
    assert set(props["track_id"]) == set(one.track_ids)
    assert len(props) == 5  # a single track over 5 frames


def test_extra_properties_adds_a_column(tmp_path):
    lin = _labelled_lineage(tmp_path)
    props = cell_properties(lin, roots(lin), extra_properties=(px_count,))
    assert "px_count" in props.columns
    assert (props["px_count"] == props["area"]).all()


def test_missing_labels_dir_raises():
    empty = pd.DataFrame(columns=["track_id", "t", "label", "x", "y", "area", "axis_len"])
    lin = Lineage("d1", empty, empty, empty, labels_dir=None)
    with pytest.raises(ValueError, match="requires labels"):
        cell_properties(lin)
