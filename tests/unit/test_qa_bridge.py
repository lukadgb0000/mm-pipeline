"""Tests for bridge_drops using a stub bridge scorer and solver"""

from __future__ import annotations

import math
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from mm_pipeline.config import DEFAULT_TRACKER_PARAMS, TrackerParams
from mm_pipeline.qa import Action, DropReason, QADecision
from mm_pipeline.qa.bridge import BridgeAttempt, bridge_drops


def _pd():
    return pytest.importorskip("pandas")


def _keep_decision(t: int) -> QADecision:
    return QADecision(
        dataset_id="d1",
        pair_id=f"d1:p{t}",
        t=t,
        n_candidates=1,
        within_pair_scorer="dp_cost_min",
        chosen_candidate_idx=0,
        chosen_ops_json='[["link", 1, 1, null]]',
        dp_top1_idx=0,
        classifier_top1_idx=None,
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
    )


def _drop_decision(t: int, drop_reason: DropReason = DropReason.ANOMALY) -> QADecision:
    return QADecision(
        dataset_id="d1",
        pair_id=f"d1:p{t}",
        t=t,
        n_candidates=2,
        within_pair_scorer="dp_cost_min",
        chosen_candidate_idx=None,
        chosen_ops_json=None,
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
        anomaly_score=0.99,
        anomaly_flag=True,
        action=Action.DROP,
        drop_reason=drop_reason,
    )


class _StubScorer:
    name = "stub"

    def __init__(self, score: float = 0.9):
        self.score_value = score

    def score(self, candidates):
        return np.full(len(candidates), self.score_value, dtype=float)


def _stub_solver(label_t, label_k, *, t, k, open_end, params, top_k, store_ops):
    pd = _pd()
    return pd.DataFrame(
        {
            "ops_json": [f'[["link", 1, 1, null]]_t{t}_k{k}'],
        }
    )


def test_bridge_converts_anomaly_drop_to_bridge():
    labels = np.zeros((4, 5, 5), dtype=np.int32)
    decisions = [_keep_decision(0), _drop_decision(1), _keep_decision(2)]
    params = TrackerParams()

    with patch("mm_pipeline.features.pairwise.solve_and_featurize_pair", _stub_solver):
        out = bridge_drops(
            decisions,
            labels=labels,
            bridge_scorer=_StubScorer(score=0.9),
            tau_bridge=0.5,
            max_gap=3,
            tracker_params=params,
            top_k=4,
            open_end="high",
        )
    actions = {d.t: d for d in out}
    # The bridge spans (0, 2): t=0 is primary; t=1 is non-primary; t=2 unchanged.
    assert actions[0].action == Action.BRIDGE
    assert actions[0].bridge_is_primary is True
    assert actions[0].bridge_span == (0, 2)
    assert actions[1].action == Action.BRIDGE
    assert actions[1].bridge_is_primary is False
    assert actions[1].bridge_span == (0, 2)
    assert actions[2].action == Action.KEEP


def test_bridge_fail_keeps_drop_with_bridge_failed_reason():
    labels = np.zeros((4, 5, 5), dtype=np.int32)
    decisions = [_keep_decision(0), _drop_decision(1), _keep_decision(2)]
    params = TrackerParams()

    with patch("mm_pipeline.features.pairwise.solve_and_featurize_pair", _stub_solver):
        out = bridge_drops(
            decisions,
            labels=labels,
            bridge_scorer=_StubScorer(score=0.1),
            tau_bridge=0.5,
            max_gap=3,
            tracker_params=params,
            top_k=4,
            open_end="high",
        )
    actions = {d.t: d for d in out}
    assert actions[1].action == Action.DROP
    assert actions[1].drop_reason == DropReason.BRIDGE_FAILED


def test_non_eligible_drops_are_left_alone():
    labels = np.zeros((4, 5, 5), dtype=np.int32)
    decisions = [
        _keep_decision(0),
        _drop_decision(1, drop_reason=DropReason.DISAGREEMENT),
        _keep_decision(2),
    ]
    params = TrackerParams()

    with patch("mm_pipeline.features.pairwise.solve_and_featurize_pair", _stub_solver):
        out = bridge_drops(
            decisions,
            labels=labels,
            bridge_scorer=_StubScorer(score=0.9),
            tau_bridge=0.5,
            max_gap=3,
            tracker_params=params,
            top_k=4,
            open_end="high",
            bridge_drop_reasons=(DropReason.ANOMALY,),
        )
    # The disagreement drop is not eligible, so it stays untouched.
    actions = {d.t: d for d in out}
    assert actions[1].action == Action.DROP
    assert actions[1].drop_reason == DropReason.DISAGREEMENT
