"""End-to-end Phase 9 test"""

from __future__ import annotations

import numpy as np
import pytest

from mm_pipeline.config import QAConfig, TrackerParams
from mm_pipeline.features import FEATURE_COLUMNS, build_feature_table_for_stack
from mm_pipeline.qa import (
    Action,
    DropReason,
    apply_qa_workflow,
    write_lineage_outputs,
    write_qa_decisions_csv,
)
from mm_pipeline.tracking.lineage import reconstruct_from_qa_decisions


def _pd():
    return pytest.importorskip("pandas")


def _stable_label_stack() -> np.ndarray:
    """Five frames with two stable, non-touching labels."""

    labels = np.zeros((5, 40, 8), dtype=np.int32)
    for t in range(5):
        labels[t, 2:8, 1:6] = 1
        labels[t, 14:22, 1:6] = 2
    return labels


def _score_table_from_features(features_df):
    """Add the minimum columns Phase 9 expects on top of feature output."""

    pd = _pd()
    out = features_df.copy()
    # Without a fitted classifier, mirror the DP cost ordering: lowest DP cost
    # gets the highest classifier raw_score and pair_probability.
    if "dp_cost" in out.columns:
        # Rank within each pair.
        out["raw_score"] = -out["dp_cost"]
        # Softmax-like normalisation per pair for pair_probability.
        out["pair_probability"] = (
            out.groupby("pair_id")["raw_score"].transform(lambda s: _softmax(s.to_numpy()))
        )
        out["candidate_correctness_probability"] = out["pair_probability"]
    return out


def _softmax(values):
    import numpy as np
    arr = np.asarray(values, dtype=float)
    arr = arr - float(np.nanmax(arr))
    exp = np.exp(arr)
    return exp / float(exp.sum())


def test_end_to_end_default_policy_reproduces_dp_lineage(tmp_path):
    labels = _stable_label_stack()
    params = TrackerParams()

    features = build_feature_table_for_stack(
        labels,
        dataset_id="d1",
        axis="y",
        open_end="high",
        params=params,
        top_k=4,
        store_ops=True,
    )
    assert not features.empty
    scored = _score_table_from_features(features)

    decisions = apply_qa_workflow(scored, config=QAConfig())
    # Every pair should KEEP (no anomalies, default policy).
    assert all(d.action == Action.KEEP for d in decisions)
    assert all(d.classifier_disagrees_with_dp is False for d in decisions)

    tracks, events, divisions = reconstruct_from_qa_decisions(
        decisions, candidate_features=scored, labels=labels, open_end="high",
    )
    # Two non-touching cells, five frames → 10 track rows (no exits/divides).
    assert len(tracks) == 10
    assert tracks["track_id"].nunique() == 2
    assert (events["event"] == "link").all()
    assert divisions.empty

    # Reports write successfully.
    write_qa_decisions_csv(decisions, tmp_path / "qa_decisions.csv")
    out_paths = write_lineage_outputs(tracks, events, divisions, tmp_path / "lineage")
    assert all(p.exists() for p in out_paths.values())


def test_end_to_end_anomaly_detector_drops_break_lineage():
    labels = _stable_label_stack()
    params = TrackerParams()

    features = build_feature_table_for_stack(
        labels,
        dataset_id="d1",
        axis="y",
        open_end="high",
        params=params,
        top_k=4,
        store_ops=True,
    )
    scored = _score_table_from_features(features)

    class DropMiddlePair:
        name = "drop_middle"

        def detect(self, per_pair_features):
            pd = _pd()
            # Flag the t=2 pair (pair_id ``d1:2->3``).
            anomaly_flag = per_pair_features["pair_id"].astype(str).str.startswith("d1:2->")
            return pd.DataFrame(
                {
                    "pair_id": per_pair_features["pair_id"].astype(str).to_list(),
                    "anomaly_score": [0.99 if f else 0.01 for f in anomaly_flag],
                    "anomaly_flag": anomaly_flag.tolist(),
                }
            )

    decisions = apply_qa_workflow(scored, config=QAConfig(), anomaly_detector=DropMiddlePair())
    drops = [d for d in decisions if d.action == Action.DROP]
    assert len(drops) >= 1
    assert all(d.drop_reason == DropReason.ANOMALY for d in drops)

    tracks, events, divisions = reconstruct_from_qa_decisions(
        decisions, candidate_features=scored, labels=labels, open_end="high",
    )
    # With a drop, the lineage should still produce track rows but with at
    # least one re-initialised track ID set after the drop.
    drop_t = drops[0].t
    pre_drop_ids = set(tracks.loc[tracks["t"] <= drop_t, "track_id"].unique())
    post_drop_ids = set(tracks.loc[tracks["t"] > drop_t, "track_id"].unique())
    if post_drop_ids:
        assert post_drop_ids.isdisjoint(pre_drop_ids), (
            "Track IDs must be re-initialised after a drop."
        )
