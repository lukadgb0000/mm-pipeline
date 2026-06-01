"""Tests for tracking.workflow.candidates_to_dataframe"""

from __future__ import annotations

import json

import pytest

from mm_pipeline.config import TrackerParams
from mm_pipeline.tracking import (
    candidates_to_dataframe,
    generate_tracking_candidates_for_stack,
)


def _np():
    return pytest.importorskip("numpy")


def _build_tiny_run(*, mode: str = "topk", top_k: int = 4):
    """Three-frame label stack with one cell shifting downward each frame."""

    np = _np()
    labels = np.zeros((3, 8, 8), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 2:4, 1:3] = 1
    labels[2, 3:5, 1:3] = 1

    return generate_tracking_candidates_for_stack(
        labels,
        dataset_id="trench_x",
        axis="y",
        open_end="high",
        params=TrackerParams(),
        mode=mode,  # type: ignore[arg-type]
        top_k=top_k,
    )


def test_candidates_to_dataframe_returns_one_row_per_candidate():
    pd = pytest.importorskip("pandas")
    run = _build_tiny_run(mode="best")
    df = candidates_to_dataframe(run)

    # 2 adjacent pairs, 1 candidate per pair in 'best' mode
    assert len(df) == 2
    assert isinstance(df, pd.DataFrame)


def test_candidates_to_dataframe_has_expected_columns_no_features():
    pd = pytest.importorskip("pandas")
    run = _build_tiny_run()
    df = candidates_to_dataframe(run)

    expected_cols = {
        "dataset_id", "labels_dir", "t", "pair_id", "delta_t", "sample_rank",
        "dp_rank_global", "dp_cost", "is_dpt_best", "candidate_source",
        "n_candidates_pair", "sample_class", "is_correct",
        "n_links", "n_exits", "n_divides", "ops_json",
    }
    assert set(df.columns) == expected_cols

    # No feature columns — those belong to featurise_candidate_run.
    feature_cols = {
        "max_shrink_pct", "total_area_ratio_exit_adjusted", "link_area_ratio_median",
    }
    assert feature_cols.isdisjoint(set(df.columns))


def test_candidates_to_dataframe_sample_class_unknown():
    run = _build_tiny_run()
    df = candidates_to_dataframe(run)

    assert (df["sample_class"] == "unknown").all()
    assert df["is_correct"].isna().all()


def test_candidates_to_dataframe_dp_costs_populated():
    run = _build_tiny_run()
    df = candidates_to_dataframe(run)

    dp_rows = df[df["candidate_source"].str.startswith("dp_")]
    assert len(dp_rows) > 0
    assert dp_rows["dp_cost"].notna().all()


def test_candidates_to_dataframe_marks_dp_best():
    run = _build_tiny_run()
    df = candidates_to_dataframe(run)

    # Per pair: exactly one is_dpt_best=True (the lowest-cost DP candidate).
    for pair_id, group in df.groupby("pair_id"):
        assert group["is_dpt_best"].sum() == 1


def test_candidates_to_dataframe_sample_rank_is_one_indexed_per_pair():
    run = _build_tiny_run()
    df = candidates_to_dataframe(run)

    for pair_id, group in df.groupby("pair_id"):
        assert group["sample_rank"].min() == 1
        assert list(group["sample_rank"]) == list(range(1, len(group) + 1))


def test_candidates_to_dataframe_ops_json_round_trips():
    run = _build_tiny_run()
    df = candidates_to_dataframe(run)

    for ops_json in df["ops_json"]:
        decoded = json.loads(ops_json)
        assert isinstance(decoded, list)
        for op in decoded:
            assert len(op) == 4
            assert op[0] in {"link", "divide", "exit"}


def test_candidates_to_dataframe_op_counts_match_ops():
    run = _build_tiny_run()
    df = candidates_to_dataframe(run)

    for _, row in df.iterrows():
        ops = json.loads(row["ops_json"])
        n_links = sum(1 for o in ops if o[0] == "link")
        n_exits = sum(1 for o in ops if o[0] == "exit")
        n_divides = sum(1 for o in ops if o[0] == "divide")
        assert int(row["n_links"]) == n_links
        assert int(row["n_exits"]) == n_exits
        assert int(row["n_divides"]) == n_divides


def test_candidates_to_dataframe_labels_dir_recorded():
    run = _build_tiny_run()
    df = candidates_to_dataframe(run, labels_dir="/some/path")
    assert (df["labels_dir"] == "/some/path").all()


def test_candidates_to_dataframe_store_ops_false_omits_column():
    run = _build_tiny_run()
    df = candidates_to_dataframe(run, store_ops=False)
    assert "ops_json" not in df.columns


def test_candidates_to_dataframe_empty_run():
    pd = pytest.importorskip("pandas")
    np = _np()

    # Single-frame stack: no adjacent pairs → empty pair_results.
    labels = np.zeros((1, 8, 8), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 1
    run = generate_tracking_candidates_for_stack(
        labels, dataset_id="empty", axis="y", open_end="high",
    )
    df = candidates_to_dataframe(run)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "ops_json" in df.columns
