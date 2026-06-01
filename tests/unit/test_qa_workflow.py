"""End-to-end tests for apply_qa_workflow without bridging"""

from __future__ import annotations

import warnings

import pytest

from mm_pipeline.config import QAConfig
from mm_pipeline.features import FEATURE_COLUMNS
from mm_pipeline.qa import Action, DropReason, apply_qa_workflow


def _pd():
    return pytest.importorskip("pandas")


def _scored_fixture():
    """Two pairs, two candidates each, with explicit DP costs and raw scores."""

    pd = _pd()
    rows = []
    for pair_idx in range(2):
        for cand_idx in range(2):
            row = {
                "dataset_id": "d1",
                "pair_id": f"d1:p{pair_idx}",
                "t": pair_idx,
                "dp_cost": 0.1 if cand_idx == 0 else 0.4,
                "dp_rank_global": cand_idx + 1,
                "is_dpt_best": cand_idx == 0,
                "raw_score": 0.8 if cand_idx == 0 else 0.1,
                "pair_probability": 0.7 if cand_idx == 0 else 0.3,
                "candidate_correctness_probability": 0.8 if cand_idx == 0 else 0.1,
                "ops_json": '[["link", 1, 1, null]]',
                "is_correct": cand_idx == 0,
            }
            for feature in FEATURE_COLUMNS:
                row[feature] = float(cand_idx + 1)
            rows.append(row)
    return pd.DataFrame(rows)


def test_default_keeps_dp_top1_for_every_pair():
    df = _scored_fixture()
    config = QAConfig()
    decisions = apply_qa_workflow(df, config=config)

    assert len(decisions) == 2
    for d in decisions:
        assert d.action == Action.KEEP
        assert d.within_pair_scorer == "dp_cost_min"
        assert d.anomaly_detector == "never_anomalous"
        assert d.anomaly_flag is False
        assert d.classifier_disagrees_with_dp is False  # raw_score agrees with dp_cost here
        assert d.chosen_ops_json is not None


def test_classifier_scorer_picks_classifier_top1():
    df = _scored_fixture()
    config = QAConfig(within_pair_scorer="classifier")
    decisions = apply_qa_workflow(df, config=config)
    for d in decisions:
        assert d.action == Action.KEEP
        assert d.within_pair_scorer == "classifier"


def test_workflow_preserves_non_integer_candidate_index_labels():
    df = _scored_fixture()
    df.index = [f"cand-{i}" for i in range(len(df))]
    decisions = apply_qa_workflow(df, config=QAConfig())
    assert [d.chosen_candidate_idx for d in decisions] == ["cand-0", "cand-2"]
    assert [d.dp_top1_idx for d in decisions] == ["cand-0", "cand-2"]
    assert all(d.chosen_is_correct is True for d in decisions)


def test_anomaly_drop_records_anomaly_reason():
    df = _scored_fixture()
    config = QAConfig()

    class AlwaysAnomalous:
        name = "always"

        def detect(self, per_pair_features):
            pd = _pd()
            return pd.DataFrame(
                {
                    "pair_id": per_pair_features["pair_id"].astype(str).to_list(),
                    "anomaly_score": [0.99] * len(per_pair_features),
                    "anomaly_flag": [True] * len(per_pair_features),
                }
            )

    decisions = apply_qa_workflow(df, config=config, anomaly_detector=AlwaysAnomalous())
    for d in decisions:
        assert d.action == Action.DROP
        assert d.drop_reason == DropReason.ANOMALY
        assert d.anomaly_detector == "always"
        assert d.chosen_candidate_idx is None
        assert d.chosen_ops_json is None


def test_hard_disagreement_drop_triggers_only_on_disagreement():
    pd = _pd()
    rows = []
    # Pair 0: top-1s agree (no disagreement). Pair 1: top-1s disagree.
    pair_specs = [
        (0, [(0.1, 0.9), (0.4, 0.2)]),
        (1, [(0.1, 0.1), (0.4, 0.9)]),
    ]
    for pair_idx, candidates in pair_specs:
        for cand_idx, (dp, raw) in enumerate(candidates):
            row = {
                "dataset_id": "d1",
                "pair_id": f"d1:p{pair_idx}",
                "t": pair_idx,
                "dp_cost": dp,
                "raw_score": raw,
                "ops_json": '[["link", 1, 1, null]]',
            }
            for feature in FEATURE_COLUMNS:
                row[feature] = 1.0
            rows.append(row)
    df = pd.DataFrame(rows)

    config = QAConfig(disagreement_drop="hard")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decisions = apply_qa_workflow(df, config=config)
    by_pair = {d.pair_id: d for d in decisions}
    assert by_pair["d1:p0"].action == Action.KEEP
    assert by_pair["d1:p1"].action == Action.DROP
    assert by_pair["d1:p1"].drop_reason == DropReason.DISAGREEMENT


def test_soft_disagreement_threshold():
    pd = _pd()
    rows = []
    
    for cand_idx, (dp, raw) in enumerate([(0.1, 0.1), (5.0, 0.9)]):
        row = {
            "dataset_id": "d1",
            "pair_id": "d1:p0",
            "t": 0,
            "dp_cost": dp,
            "raw_score": raw,
            "ops_json": '[["link", 1, 1, null]]',
        }
        for feature in FEATURE_COLUMNS:
            row[feature] = 1.0
        rows.append(row)
    df = pd.DataFrame(rows)

    config = QAConfig(disagreement_drop="soft", disagreement_soft_threshold=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decisions = apply_qa_workflow(df, config=config)
    assert decisions[0].action == Action.DROP
    assert decisions[0].drop_reason == DropReason.DISAGREEMENT


def test_disagreement_drop_without_bridge_warns():
    df = _scored_fixture()
    config = QAConfig(disagreement_drop="hard")
    with pytest.warns(UserWarning, match="disagreement_drop is set"):
        apply_qa_workflow(df, config=config)


def test_empty_input_returns_empty():
    pd = _pd()
    df = pd.DataFrame(columns=["pair_id"])
    decisions = apply_qa_workflow(df, config=QAConfig())
    assert decisions == []
