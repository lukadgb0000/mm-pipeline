"""Tests for apply_ops_to_lineage and reconstruct_from_qa_decisions"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mm_pipeline.core import TrackingOperation
from mm_pipeline.qa import Action, DropReason, QADecision
from mm_pipeline.tracking.lineage import apply_ops_to_lineage, reconstruct_from_qa_decisions


def _pd():
    return pytest.importorskip("pandas")


def _three_frame_labels() -> np.ndarray:
    """3 frames × 30×6 image with two stable labels (1, 2) in each frame."""

    labels = np.zeros((3, 30, 6), dtype=np.int32)
    for t in range(3):
        labels[t, 5:10, 1:5] = 1
        labels[t, 15:25, 1:5] = 2
    return labels


def _decision(t: int, ops_json: str, **overrides) -> QADecision:
    defaults = dict(
        dataset_id="d1",
        pair_id=f"d1:p{t}",
        t=t,
        n_candidates=1,
        within_pair_scorer="dp_cost_min",
        chosen_candidate_idx=0,
        chosen_ops_json=ops_json,
        dp_top1_idx=0,
        classifier_top1_idx=None,
        classifier_disagrees_with_dp=False,
        dp_cost_gap=0.0,
        dp_cost_gap_normalised=0.0,
        classifier_score_gap=0.0,
        within_pair_max_score=float("nan"),
        within_pair_entropy=float("nan"),
        within_pair_margin_top1_top2=float("nan"),
        disagreement_score=float("nan"),
        anomaly_detector="never_anomalous",
        anomaly_score=float("nan"),
        anomaly_flag=False,
        action=Action.KEEP,
    )
    defaults.update(overrides)
    return QADecision(**defaults)


def test_apply_ops_to_lineage_link_and_divide():
    from mm_pipeline.core import extract_cell_instances, sort_cells_along_trench

    labels = _three_frame_labels()
    cells_a = {c.label: c for c in sort_cells_along_trench(extract_cell_instances(labels[0]), axis="y", open_end="high")}
    cells_b = {c.label: c for c in sort_cells_along_trench(extract_cell_instances(labels[1]), axis="y", open_end="high")}
    tracks_rows: list[dict] = []
    events_rows: list[dict] = []
    division_rows: list[dict] = []
    label_to_track = {1: 100, 2: 200}
    next_id = apply_ops_to_lineage(
        t=0,
        ops=[
            TrackingOperation("link", 1, 1),
            TrackingOperation("link", 2, 2),
        ],
        frame_a=cells_a,
        frame_b=cells_b,
        label_to_track=label_to_track,
        next_track_id=300,
        tracks_rows=tracks_rows,
        events_rows=events_rows,
        division_rows=division_rows,
        axis="y",
    )
    assert next_id == 300
    assert len(events_rows) == 2
    assert all(e["event"] == "link" and e["dst_t"] == 1 for e in events_rows)
    assert len(tracks_rows) == 2
    assert label_to_track == {1: 100, 2: 200}


def test_reconstruct_keep_only():
    labels = _three_frame_labels()
    decisions = [
        _decision(t=0, ops_json='[["link", 1, 1, null], ["link", 2, 2, null]]'),
        _decision(t=1, ops_json='[["link", 1, 1, null], ["link", 2, 2, null]]'),
    ]
    tracks, events, divs = reconstruct_from_qa_decisions(decisions, candidate_features=None, labels=labels)

    assert len(tracks) == 6  # 2 cells × 3 frames
    assert (tracks["track_id"].nunique() == 2)
    assert (events["event"] == "link").all()
    assert divs.empty


def test_reconstruct_drop_breaks_tracks():
    labels = _three_frame_labels()
    decisions = [
        _decision(t=0, ops_json='[["link", 1, 1, null], ["link", 2, 2, null]]'),
        _decision(
            t=1,
            ops_json=None,
            chosen_candidate_idx=None,
            action=Action.DROP,
            drop_reason=DropReason.ANOMALY,
        ),
    ]
    tracks, events, divs = reconstruct_from_qa_decisions(decisions, candidate_features=None, labels=labels)
    # Frames 0 and 1 produce tracks (frame 0 init, frame 1 from link ops). Frame
    # 2 is unreachable because of the drop.
    assert set(tracks["t"].unique()) == {0, 1}
    # Track IDs at frame 0 and 1 are shared (one track per cell, linked).
    assert tracks["track_id"].nunique() == 2


def test_reconstruct_bridge_skips_intermediate_frames():
    labels = _three_frame_labels()
    decisions = [
        _decision(
            t=0,
            ops_json=None,
            chosen_candidate_idx=None,
            action=Action.BRIDGE,
            bridge_span=(0, 2),
            bridge_ops_json='[["link", 1, 1, null], ["link", 2, 2, null]]',
            bridge_score=0.9,
            bridge_is_primary=True,
        ),
        _decision(
            t=1,
            ops_json=None,
            chosen_candidate_idx=None,
            action=Action.BRIDGE,
            bridge_span=(0, 2),
            bridge_is_primary=False,
        ),
    ]
    tracks, events, divs = reconstruct_from_qa_decisions(decisions, candidate_features=None, labels=labels)
    # Bridged frame 1 is skipped; tracks should exist at t=0 and t=2 only.
    assert set(tracks["t"].unique()) == {0, 2}
    # Events should carry dst_t == 2 to indicate the bridge span.
    assert (events["dst_t"] == 2).all()
