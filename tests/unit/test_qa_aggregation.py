"""Tests for the per-pair feature aggregator"""

from __future__ import annotations

import math

import pytest

from mm_pipeline.features import FEATURE_COLUMNS
from mm_pipeline.qa.aggregation import build_per_pair_features, per_pair_feature_columns


def _pd():
    return pytest.importorskip("pandas")


def _fixture():
    pd = _pd()
    rows = []
    
    for pair_idx in range(2):
        for cand_idx in range(3):
            row = {
                "dataset_id": "d1",
                "pair_id": f"d1:p{pair_idx}",
                "t": pair_idx,
                "dp_cost": 0.1 + 0.2 * cand_idx,
                "raw_score": 0.5 - 0.2 * cand_idx,
                "pair_probability": [0.6, 0.3, 0.1][cand_idx],
            }
            for feature in FEATURE_COLUMNS:
                row[feature] = float(cand_idx + 1)
            rows.append(row)
    return pd.DataFrame(rows)


def test_per_pair_feature_columns_count():
    cols = per_pair_feature_columns()
    
    assert len(cols) == 4 + 14 * 5 + 7 + 4 + 1


def test_build_per_pair_features_row_count_and_shape():
    df = _fixture()
    agg = build_per_pair_features(df)
    assert len(agg) == 2
    assert list(agg.columns) == per_pair_feature_columns()


def test_aggregations_compute_correct_basic_stats():
    df = _fixture()
    agg = build_per_pair_features(df)
    # 
    row = agg.iloc[0]
    assert row["max_shrink_pct_max"] == pytest.approx(3.0)
    assert row["max_shrink_pct_min"] == pytest.approx(1.0)
    assert row["max_shrink_pct_mean"] == pytest.approx(2.0)


def test_best_row_uses_argmax_of_raw_score():
    df = _fixture()
    agg = build_per_pair_features(df)
    
    assert agg.iloc[0]["max_shrink_pct_best"] == pytest.approx(1.0)


def test_aggregation_handles_non_integer_index_labels():
    df = _fixture()
    df.index = [f"candidate-{i}" for i in range(len(df))]
    agg = build_per_pair_features(df)
    assert len(agg) == 2
    assert agg.iloc[0]["max_shrink_pct_best"] == pytest.approx(1.0)
    assert agg.iloc[0]["disagreement_score"] == pytest.approx(0.0)


def test_dp_summary_columns():
    df = _fixture()
    agg = build_per_pair_features(df)
    row = agg.iloc[0]
    assert row["dp_cost_min"] == pytest.approx(0.1)
    assert row["dp_cost_max"] == pytest.approx(0.5)
    assert row["dp_cost_range"] == pytest.approx(0.4)


def test_disagreement_score_is_zero_when_top1_agrees():
    pd = _pd()
    # 
    df = pd.DataFrame(
        {
            "dataset_id": ["d1"] * 2,
            "pair_id": ["d1:p0"] * 2,
            "t": [0, 0],
            "dp_cost": [0.1, 0.5],
            "raw_score": [0.9, 0.2],
        }
        | {feature: [1.0, 2.0] for feature in FEATURE_COLUMNS}
    )
    agg = build_per_pair_features(df)
    assert agg.iloc[0]["disagreement_score"] == pytest.approx(0.0)


def test_disagreement_score_positive_when_top1_disagrees():
    df = _fixture()
    agg = build_per_pair_features(df)
    
    pd = _pd()
    df2 = pd.DataFrame(
        {
            "dataset_id": ["d1"] * 2,
            "pair_id": ["d1:p0"] * 2,
            "t": [0, 0],
            "dp_cost": [0.1, 0.5],
            "raw_score": [0.1, 0.9],
        }
        | {feature: [1.0, 2.0] for feature in FEATURE_COLUMNS}
    )
    agg = build_per_pair_features(df2)
    assert agg.iloc[0]["disagreement_score"] > 0.0


def test_score_entropy_finite_when_probabilities_present():
    df = _fixture()
    agg = build_per_pair_features(df)
    assert math.isfinite(agg.iloc[0]["score_entropy"])


def test_empty_input_returns_empty_dataframe_with_schema():
    pd = _pd()
    df = pd.DataFrame(columns=["pair_id", "dataset_id", "t", "dp_cost", "raw_score"])
    agg = build_per_pair_features(df)
    assert agg.empty
    assert list(agg.columns) == per_pair_feature_columns()
