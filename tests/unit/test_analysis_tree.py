"""Unit tests for mm_pipeline.analysis.tree (Lineage + cycles)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mm_pipeline.analysis import Lineage
from mm_pipeline.config import TrackerParams
from mm_pipeline.tracking.lineage import reconstruct_lineage
from mm_pipeline.tracking.select import DPCostMin, select_pairs

pd = pytest.importorskip("pandas")


# --- hand-built lineage covering every cycles code path -----------------------
#
# Structure (last frame = t6):
#   1 --div@1--> 2, 3            2 exits @4            3 --div@5--> 6, 7
#   4 : birth-unobserved root, born @3 (a re-init / drop boundary)
#   5 : birth-unobserved root, born @2, missing t3 (a bridge gap)


def _track_rows(track_id, ts):
    return [
        {"track_id": track_id, "t": t, "label": 1, "x": 2.0, "y": float(t), "area": 10.0, "axis_len": 5.0}
        for t in ts
    ]


def _hand_built_lineage():
    tracks = pd.DataFrame(
        _track_rows(1, [0, 1])
        + _track_rows(2, [2, 3, 4])
        + _track_rows(3, [2, 3, 4, 5])
        + _track_rows(6, [6])
        + _track_rows(7, [6])
        + _track_rows(4, [3, 4, 5])
        + _track_rows(5, [2, 4])  # missing t3 -> one gap frame
    )
    divisions = pd.DataFrame(
        [
            {"t_div": 1, "mother_track_id": 1, "d1_track_id": 2, "d2_track_id": 3},
            {"t_div": 5, "mother_track_id": 3, "d1_track_id": 6, "d2_track_id": 7},
        ]
    )
    events = pd.DataFrame(
        [{"t": 4, "event": "exit", "src_label": 1, "dst1_label": "", "dst2_label": "", "track_id": 2, "cost": None, "dst_t": 5}]
    )
    return Lineage(dataset_id="d1", tracks_df=tracks, divisions_df=divisions, events_df=events)


def test_cycles_columns_and_grain():
    cyc = _hand_built_lineage().cycles
    assert list(cyc.columns) == [
        "dataset_id", "track_id", "parent_id", "generation", "birth_t", "end_t",
        "n_frames", "n_gap_frames", "birth_observed", "end_cause", "complete_cycle",
    ]
    assert cyc["track_id"].tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert (cyc["dataset_id"] == "d1").all()


def test_cycles_structure_generation_and_parent():
    cyc = _hand_built_lineage().cycles.set_index("track_id")
    assert cyc.loc[1, "generation"] == 0 and pd.isna(cyc.loc[1, "parent_id"])
    assert cyc.loc[2, "generation"] == 1 and cyc.loc[2, "parent_id"] == 1
    assert cyc.loc[3, "generation"] == 1 and cyc.loc[3, "parent_id"] == 1
    assert cyc.loc[6, "generation"] == 2 and cyc.loc[6, "parent_id"] == 3
    assert cyc.loc[7, "generation"] == 2 and cyc.loc[7, "parent_id"] == 3


def test_cycles_end_cause():
    cyc = _hand_built_lineage().cycles.set_index("track_id")
    assert cyc.loc[1, "end_cause"] == "divided"
    assert cyc.loc[2, "end_cause"] == "exited"
    assert cyc.loc[3, "end_cause"] == "divided"
    assert cyc.loc[6, "end_cause"] == "censored"  # alive at the last frame
    assert cyc.loc[4, "end_cause"] == "censored"


def test_cycles_birth_observed_marks_reinit_roots():
    cyc = _hand_built_lineage().cycles.set_index("track_id")
    # Division daughters have observed births.
    assert bool(cyc.loc[2, "birth_observed"]) and bool(cyc.loc[3, "birth_observed"])
    # Frame-0 origin and mid-movie re-init roots do not.
    assert not bool(cyc.loc[1, "birth_observed"])
    assert not bool(cyc.loc[4, "birth_observed"])  # born at t=3 with no parent
    assert cyc.loc[4, "birth_t"] == 3 and pd.isna(cyc.loc[4, "parent_id"])


def test_cycles_gap_accounting():
    cyc = _hand_built_lineage().cycles.set_index("track_id")
    assert cyc.loc[5, "n_frames"] == 2 and cyc.loc[5, "n_gap_frames"] == 1
    assert cyc.loc[3, "n_gap_frames"] == 0


def test_cycles_complete_cycle_requires_observed_birth_division_and_no_gap():
    cyc = _hand_built_lineage().cycles.set_index("track_id")
    # Only track 3: born via division, divides, contiguous.
    assert bool(cyc.loc[3, "complete_cycle"])
    for tid in (1, 2, 4, 5, 6, 7):
        assert not bool(cyc.loc[tid, "complete_cycle"])


def test_frames_by_track_is_time_sorted():
    lin = _hand_built_lineage()
    frames = lin.frames_by_track[5]
    assert frames["t"].tolist() == [2, 4]
    assert set(lin.frames_by_track) == {1, 2, 3, 4, 5, 6, 7}


def test_axis_col_normalisation():
    assert Lineage("d", None, None, None, axis="y").axis_col == "y"
    assert Lineage("d", None, None, None, axis="x").axis_col == "x"


def test_lineage_equality_is_identity_and_hashable():
    # The eq=False guard: comparing/hashing must not choke on DataFrame fields.
    lin = _hand_built_lineage()
    assert lin == lin
    assert lin != _hand_built_lineage()
    assert lin in {lin}


def test_empty_lineage_yields_empty_cycles():
    empty = pd.DataFrame(columns=["track_id", "t", "label", "x", "y", "area", "axis_len"])
    cyc = Lineage("d", empty, empty, empty).cycles
    assert cyc.empty and list(cyc.columns) == [
        "dataset_id", "track_id", "parent_id", "generation", "birth_t", "end_t",
        "n_frames", "n_gap_frames", "birth_observed", "end_cause", "complete_cycle",
    ]


# --- integration against real reconstruct_lineage output ----------------------


def _stable_label_stack() -> np.ndarray:
    labels = np.zeros((5, 40, 8), dtype=np.int32)
    for t in range(5):
        labels[t, 2:8, 1:6] = 1
        labels[t, 14:22, 1:6] = 2
    return labels


def _lineage_from_labels(labels, selections) -> Lineage:
    from mm_pipeline.features import build_feature_table_for_stack

    if selections is None:
        features = build_feature_table_for_stack(
            labels, dataset_id="d1", axis="y", open_end="high",
            params=TrackerParams(), top_k=4, store_ops=True,
        )
        selections = select_pairs(features, DPCostMin())
    tracks, events, divisions = reconstruct_lineage(selections, labels, open_end="high", axis="y")
    return Lineage(dataset_id="d1", tracks_df=tracks, divisions_df=divisions, events_df=events)


def test_integration_two_stable_cells_are_censored_roots():
    lin = _lineage_from_labels(_stable_label_stack(), None)
    cyc = lin.cycles
    assert len(cyc) == 2
    assert (cyc["generation"] == 0).all()
    assert (~cyc["birth_observed"]).all()          # both present from frame 0
    assert (cyc["end_cause"] == "censored").all()  # neither divides nor exits
    assert (~cyc["complete_cycle"]).all()


def test_integration_break_produces_birth_unobserved_reinit_roots():
    from mm_pipeline.features import build_feature_table_for_stack

    labels = _stable_label_stack()
    features = build_feature_table_for_stack(
        labels, dataset_id="d1", axis="y", open_end="high",
        params=TrackerParams(), top_k=4, store_ops=True,
    )
    selections = select_pairs(features, DPCostMin())
    broken = [replace(s, chosen_ops_json=None) if s.t == 2 else s for s in selections]

    lin = _lineage_from_labels(labels, broken)
    cyc = lin.cycles
    # Tracks re-initialised after the break are born at t>0 with no parent.
    reinit = cyc[(cyc["birth_t"] > 0) & (~cyc["birth_observed"])]
    assert not reinit.empty
    assert (reinit["generation"] == 0).all()
