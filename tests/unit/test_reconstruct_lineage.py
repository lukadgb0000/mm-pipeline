"""Unit tests for tracking.lineage.reconstruct_lineage"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mm_pipeline.config import TrackerParams
from mm_pipeline.features import build_feature_table_for_stack
from mm_pipeline.tracking.lineage import reconstruct_lineage
from mm_pipeline.tracking.select import DPCostMin, select_pairs


def _pd():
    return pytest.importorskip("pandas")


def _stable_label_stack() -> np.ndarray:
    """Five frames with two stable, non-touching labels"""

    labels = np.zeros((5, 40, 8), dtype=np.int32)
    for t in range(5):
        labels[t, 2:8, 1:6] = 1
        labels[t, 14:22, 1:6] = 2
    return labels


def _selections(labels):
    features = build_feature_table_for_stack(
        labels,
        dataset_id="d1",
        axis="y",
        open_end="high",
        params=TrackerParams(),
        top_k=4,
        store_ops=True,
    )
    assert not features.empty
    return select_pairs(features, DPCostMin())


def test_reconstruct_lineage_two_stable_cells():
    labels = _stable_label_stack()
    tracks, events, divisions = reconstruct_lineage(
        _selections(labels), labels, open_end="high", axis="y"
    )
    # Two non-touching cells across five frames -> 10 rows, 2 track IDs
    assert len(tracks) == 10
    assert tracks["track_id"].nunique() == 2
    assert (events["event"] == "link").all()
    assert divisions.empty


def test_reconstruct_lineage_none_ops_breaks_lineage():
    labels = _stable_label_stack()
    selections = _selections(labels)
    # Break the lineage at t=2 by clearing that pair's chosen ops
    broken = [replace(s, chosen_ops_json=None) if s.t == 2 else s for s in selections]

    tracks, _events, _divisions = reconstruct_lineage(
        broken, labels, open_end="high", axis="y"
    )
    pre = set(tracks.loc[tracks["t"] <= 2, "track_id"].unique())
    post = set(tracks.loc[tracks["t"] > 2, "track_id"].unique())
    assert post, "expected re-initialised tracks after the break"
    assert post.isdisjoint(pre), "track IDs must be re-initialised after a break"


def test_reconstruct_lineage_empty_labels_returns_empty_frames():
    tracks, events, divisions = reconstruct_lineage([], np.zeros((0, 4, 4)), axis="y")
    assert tracks.empty and events.empty and divisions.empty
