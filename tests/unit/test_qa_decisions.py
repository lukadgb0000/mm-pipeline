"""Tests for QADecision invariants and row flattening"""

from __future__ import annotations

import math

import pytest

from mm_pipeline.qa import Action, DropReason, QADecision, validate_decision


def _base_decision(**overrides) -> QADecision:
    defaults = dict(
        dataset_id="d1",
        pair_id="d1:0",
        t=0,
        n_candidates=2,
        within_pair_scorer="dp_cost_min",
        chosen_candidate_idx=0,
        chosen_ops_json="[]",
        dp_top1_idx=0,
        classifier_top1_idx=0,
        classifier_disagrees_with_dp=False,
        dp_cost_gap=0.0,
        dp_cost_gap_normalised=0.0,
        classifier_score_gap=0.0,
        within_pair_max_score=1.0,
        within_pair_entropy=0.0,
        within_pair_margin_top1_top2=1.0,
        disagreement_score=0.0,
        anomaly_detector="never_anomalous",
        anomaly_score=float("nan"),
        anomaly_flag=False,
        action=Action.KEEP,
        drop_reason=None,
    )
    defaults.update(overrides)
    return QADecision(**defaults)


def test_keep_decision_passes_validation():
    validate_decision(_base_decision())


def test_keep_with_drop_reason_rejected():
    d = _base_decision(drop_reason=DropReason.ANOMALY)
    with pytest.raises(ValueError, match="must not have a drop_reason"):
        validate_decision(d)


def test_drop_requires_drop_reason():
    d = _base_decision(action=Action.DROP, chosen_candidate_idx=None, chosen_ops_json=None)
    with pytest.raises(ValueError, match="requires a drop_reason"):
        validate_decision(d)


def test_bridge_primary_requires_ops_and_score():
    d = _base_decision(
        action=Action.BRIDGE,
        chosen_candidate_idx=None,
        chosen_ops_json=None,
        bridge_span=(0, 2),
        bridge_is_primary=True,
    )
    with pytest.raises(ValueError, match="primary bridge requires"):
        validate_decision(d)


def test_bridge_non_primary_must_not_carry_ops():
    d = _base_decision(
        action=Action.BRIDGE,
        chosen_candidate_idx=None,
        chosen_ops_json=None,
        bridge_span=(0, 2),
        bridge_is_primary=False,
        bridge_ops_json='[["link",1,1,null]]',
    )
    with pytest.raises(ValueError, match="non-primary bridge must not carry"):
        validate_decision(d)


def test_to_row_flattens_span_and_enums():
    d = _base_decision(
        action=Action.BRIDGE,
        chosen_candidate_idx=None,
        chosen_ops_json=None,
        bridge_span=(3, 5),
        bridge_ops_json='[["link",1,1,null]]',
        bridge_is_primary=True,
        bridge_score=0.7,
    )
    row = d.to_row()
    assert row["action"] == "bridge"
    assert row["drop_reason"] is None
    assert row["bridge_t_a"] == 3
    assert row["bridge_t_b"] == 5
    assert math.isclose(row["bridge_score"], 0.7)
    assert "bridge_span" not in row
