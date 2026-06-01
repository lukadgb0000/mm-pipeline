"""Tests for features.pairwise.featurise_candidate_run"""

from __future__ import annotations

import pytest

from mm_pipeline.config import TrackerParams
from mm_pipeline.features import (
    FEATURE_COLUMNS,
    SAMPLE_META_COLUMNS,
    build_feature_table_for_stack,
    featurise_candidate_run,
)
from mm_pipeline.tracking import generate_tracking_candidates_for_stack


def _np():
    return pytest.importorskip("numpy")


def _build_labels():
    np = _np()
    labels = np.zeros((3, 8, 8), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 2:4, 1:3] = 1
    labels[2, 3:5, 1:3] = 1
    return labels


def _build_run(labels, *, mode: str = "topk", top_k: int = 4):
    return generate_tracking_candidates_for_stack(
        labels,
        dataset_id="trench_x",
        axis="y",
        open_end="high",
        params=TrackerParams(),
        mode=mode,  # type: ignore[arg-type]
        top_k=top_k,
    )


def test_featurise_candidate_run_returns_dataframe_with_features():
    pytest.importorskip("pandas")
    labels = _build_labels()
    run = _build_run(labels)

    df = featurise_candidate_run(run, labels=labels)

    expected = set(SAMPLE_META_COLUMNS) | set(FEATURE_COLUMNS) | {"ops_json"}
    assert set(df.columns) == expected


def test_featurise_candidate_run_matches_build_feature_table_for_stack():
    pd = pytest.importorskip("pandas")
    labels = _build_labels()

    via_helper = featurise_candidate_run(
        _build_run(labels),
        labels=labels,
        store_ops=True,
    )
    via_full = build_feature_table_for_stack(
        labels,
        dataset_id="trench_x",
        axis="y",
        open_end="high",
        params=TrackerParams(),
        top_k=4,
        store_ops=True,
    )

    # Same row count.
    assert len(via_helper) == len(via_full)

    # Identical SAMPLE_META columns row-by-row.
    common_cols = [c for c in SAMPLE_META_COLUMNS if c in via_helper.columns and c in via_full.columns]
    # Drop sample_class / is_correct since those are NA when no GT is provided
    # — they need NA-aware comparison.
    for col in common_cols:
        if col in {"sample_class", "is_correct"}:
            continue
        # Compare position-by-position; the order should match because both
        # walk pair_results in the same order.
        assert list(via_helper[col]) == list(via_full[col]), f"Mismatch on {col!r}"

    # Identical feature columns (NaN-aware comparison via pandas).
    for col in FEATURE_COLUMNS:
        a = via_helper[col].astype(float).reset_index(drop=True)
        b = via_full[col].astype(float).reset_index(drop=True)
        pd.testing.assert_series_equal(a, b, check_names=False)


def test_featurise_candidate_run_empty_pair_results_returns_empty_table():
    np = _np()
    labels = np.zeros((1, 8, 8), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 1
    run = generate_tracking_candidates_for_stack(
        labels, dataset_id="empty", axis="y", open_end="high",
    )

    df = featurise_candidate_run(run, labels=labels)
    expected = set(SAMPLE_META_COLUMNS) | set(FEATURE_COLUMNS) | {"ops_json"}
    assert set(df.columns) == expected
    assert len(df) == 0


def test_featurise_candidate_run_labels_dir_recorded():
    labels = _build_labels()
    run = _build_run(labels)

    df = featurise_candidate_run(run, labels=labels, labels_dir="/x/y/z")
    assert (df["labels_dir"] == "/x/y/z").all()


def test_featurise_candidate_run_store_ops_false_omits_column():
    labels = _build_labels()
    run = _build_run(labels)

    df = featurise_candidate_run(run, labels=labels, store_ops=False)
    assert "ops_json" not in df.columns


def test_featurise_candidate_run_axis_mismatch_raises():
    labels = _build_labels()
    run = _build_run(labels)

    with pytest.raises(ValueError, match="axis"):
        featurise_candidate_run(
            run,
            labels=labels,
            params=TrackerParams(axis="x"),
        )
