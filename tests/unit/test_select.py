"""Unit tests for tracking.select: SelectionResult + select_pairs"""

from __future__ import annotations

import pytest

from mm_pipeline.tracking.select import ClassifierMax, DPCostMin, select_pairs


def _pd():
    return pytest.importorskip("pandas")


def _scored():
    pd = _pd()
    # Two pairs, two candidates each. Distinct dp_cost / raw_score so the DP and classifier scorers pick different rows
    return pd.DataFrame(
        {
            "pair_id": ["p0", "p0", "p1", "p1"],
            "t": [0, 0, 1, 1],
            "ops_json": ["opsA", "opsB", "opsC", "opsD"],
            "dp_cost": [2.0, 1.0, 5.0, 3.0],
            "raw_score": [0.1, 0.9, 0.7, 0.2],
            "dataset_id": ["d1", "d1", "d1", "d1"],
        }
    )


def test_select_pairs_dp_cost_min_picks_lowest_cost():
    selections = select_pairs(_scored(), DPCostMin())
    assert [s.pair_id for s in selections] == ["p0", "p1"]
    assert [s.t for s in selections] == [0, 1]
    # p0 min dp_cost=1.0 -> opsB; p1 min dp_cost=3.0 -> opsD.
    assert [s.chosen_ops_json for s in selections] == ["opsB", "opsD"]
    assert [s.chosen_score for s in selections] == [1.0, 3.0]
    assert all(s.dataset_id == "d1" for s in selections)


def test_select_pairs_classifier_picks_highest_score():
    selections = select_pairs(_scored(), ClassifierMax())
    # p0 max raw_score=0.9 -> opsB; p1 max raw_score=0.7 -> opsC
    assert [s.chosen_ops_json for s in selections] == ["opsB", "opsC"]


def test_select_pairs_preserves_group_encounter_order_not_sorted():
    pd = _pd()
    # pair_ids appear p1 before p0; sort=False must keep that order
    df = pd.DataFrame(
        {
            "pair_id": ["p1", "p1", "p0", "p0"],
            "t": [1, 1, 0, 0],
            "ops_json": ["a", "b", "c", "d"],
            "dp_cost": [1.0, 2.0, 1.0, 2.0],
        }
    )
    selections = select_pairs(df, DPCostMin())
    assert [s.pair_id for s in selections] == ["p1", "p0"]


def test_select_pairs_none_ops_json_when_column_absent():
    df = _scored().drop(columns=["ops_json"])
    selections = select_pairs(df, DPCostMin())
    assert all(s.chosen_ops_json is None for s in selections)


def test_select_pairs_requires_pair_id():
    with pytest.raises(KeyError, match="pair_id"):
        select_pairs(_scored().drop(columns=["pair_id"]), DPCostMin())
