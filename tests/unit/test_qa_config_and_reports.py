"""QAConfig validation + CSV writer smoke tests"""

from __future__ import annotations

import pytest

from mm_pipeline.config import QAConfig
from mm_pipeline.qa import Action, DropReason, QADecision, decisions_to_dataframe, write_qa_decisions_csv


def test_qa_config_defaults_are_conservative():
    cfg = QAConfig()
    assert cfg.within_pair_scorer == "dp_cost_min"
    assert cfg.anomaly_detector == "never_anomalous"
    assert cfg.bridge_enabled is False
    assert cfg.disagreement_drop == "never"


def test_qa_config_rejects_invalid_scorer():
    with pytest.raises(ValueError, match="within_pair_scorer"):
        QAConfig(within_pair_scorer="???")


def test_qa_config_rejects_invalid_mode_and_policy():
    with pytest.raises(ValueError, match="ensemble_mode"):
        QAConfig(within_pair_ensemble_mode="???")
    with pytest.raises(ValueError, match="disagreement_drop"):
        QAConfig(disagreement_drop="???")


def _decision(action: Action, drop_reason=None, t: int = 0) -> QADecision:
    return QADecision(
        dataset_id="d1",
        pair_id=f"d1:p{t}",
        t=t,
        n_candidates=1,
        within_pair_scorer="dp_cost_min",
        chosen_candidate_idx=0 if action == Action.KEEP else None,
        chosen_ops_json='[]' if action == Action.KEEP else None,
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
        anomaly_flag=action == Action.DROP,
        action=action,
        drop_reason=drop_reason,
    )


def test_decisions_to_dataframe_includes_bridge_split_columns(tmp_path):
    pd = pytest.importorskip("pandas")
    decisions = [
        _decision(Action.KEEP, t=0),
        _decision(Action.DROP, drop_reason=DropReason.ANOMALY, t=1),
    ]
    df = decisions_to_dataframe(decisions)
    assert "bridge_t_a" in df.columns
    assert "bridge_t_b" in df.columns
    assert "bridge_span" not in df.columns
    assert df.loc[0, "action"] == "keep"
    assert df.loc[1, "action"] == "drop"
    assert df.loc[1, "drop_reason"] == "anomaly"


def test_write_qa_decisions_csv_roundtrips(tmp_path):
    pd = pytest.importorskip("pandas")
    decisions = [_decision(Action.KEEP, t=0)]
    path = write_qa_decisions_csv(decisions, tmp_path / "qa" / "qa_decisions.csv")
    assert path.exists()
    df = pd.read_csv(path)
    assert list(df.loc[0])[0] == "d1"
