import math

import numpy as np
import pytest

from mm_pipeline.segmentation import (
    aggregate_instance_evaluations,
    evaluate_instance_labels,
)


def test_perfect_match_ignores_label_identity() -> None:
    truth = np.array([[0, 1, 1, 0], [0, 2, 2, 0]], dtype=np.uint32)
    pred = np.array([[0, 9, 9, 0], [0, 4, 4, 0]], dtype=np.uint32)

    result = evaluate_instance_labels(pred, truth)

    assert (result.tp, result.fp, result.fn) == (2, 0, 0)
    assert [(m.pred_label, m.truth_label, m.iou) for m in result.matches] == [
        (4, 2, 1.0),
        (9, 1, 1.0),
    ]
    assert result.unmatched_pred_labels == ()
    assert result.unmatched_truth_labels == ()
    assert result.pixel_intersection == result.pixel_union == 4


def test_false_positive_and_false_negative_are_reported() -> None:
    truth = np.array([[1, 1, 0, 2, 2, 0]], dtype=np.uint16)
    pred = np.array([[7, 7, 0, 0, 0, 9]], dtype=np.uint16)

    result = evaluate_instance_labels(pred, truth)

    assert (result.tp, result.fp, result.fn) == (1, 1, 1)
    assert result.unmatched_pred_labels == (9,)
    assert result.unmatched_truth_labels == (2,)


def test_threshold_is_inclusive() -> None:
    truth = np.array([[1, 1, 0]], dtype=np.uint16)
    pred = np.array([[9, 0, 0]], dtype=np.uint16)

    assert evaluate_instance_labels(pred, truth, iou_threshold=0.5).tp == 1
    assert evaluate_instance_labels(pred, truth, iou_threshold=0.500001).tp == 0


def test_zero_threshold_does_not_match_disjoint_labels() -> None:
    truth = np.array([[1, 0]], dtype=np.uint16)
    pred = np.array([[0, 9]], dtype=np.uint16)

    result = evaluate_instance_labels(pred, truth, iou_threshold=0.0)

    assert (result.tp, result.fp, result.fn) == (0, 1, 1)


def test_split_and_merge_topology_diagnostics() -> None:
    split_truth = np.array([[1, 1, 1, 1]], dtype=np.uint16)
    split_pred = np.array([[5, 5, 6, 6]], dtype=np.uint16)
    split = evaluate_instance_labels(split_pred, split_truth)
    assert split.split_count == 1
    assert split.merge_count == 0

    merge_truth = np.array([[1, 1, 2, 2]], dtype=np.uint16)
    merge_pred = np.array([[5, 5, 5, 5]], dtype=np.uint16)
    merge = evaluate_instance_labels(merge_pred, merge_truth)
    assert merge.split_count == 0
    assert merge.merge_count == 1


def test_topology_threshold_ignores_tiny_secondary_overlap() -> None:
    truth = np.array([[1] * 10], dtype=np.uint16)
    pred = np.array([[5] * 9 + [6]], dtype=np.uint16)

    assert evaluate_instance_labels(
        pred, truth, topology_overlap_threshold=0.1
    ).split_count == 1
    assert evaluate_instance_labels(
        pred, truth, topology_overlap_threshold=0.11
    ).split_count == 0


def test_empty_prediction_and_empty_truth() -> None:
    zeros = np.zeros((2, 2), dtype=np.uint16)
    truth = np.array([[1, 1], [0, 0]], dtype=np.uint16)

    missing = evaluate_instance_labels(zeros, truth)
    assert (missing.tp, missing.fp, missing.fn) == (0, 0, 1)

    empty = evaluate_instance_labels(zeros, zeros)
    summary = aggregate_instance_evaluations([empty])
    assert (summary.tp, summary.fp, summary.fn) == (0, 0, 0)
    assert math.isnan(summary.precision)
    assert math.isnan(summary.recall)
    assert math.isnan(summary.f1)
    assert math.isnan(summary.modsa)
    assert math.isnan(summary.mean_matched_iou)
    assert math.isnan(summary.pixel_iou)


def test_aggregation_and_modsa() -> None:
    perfect = evaluate_instance_labels(
        np.array([[9, 9, 0]], dtype=np.uint16),
        np.array([[1, 1, 0]], dtype=np.uint16),
    )
    one_fp = evaluate_instance_labels(
        np.array([[9, 9, 8]], dtype=np.uint16),
        np.array([[1, 1, 0]], dtype=np.uint16),
    )

    summary = aggregate_instance_evaluations([perfect, one_fp])

    assert summary.n_frames == 2
    assert (summary.n_pred, summary.n_truth) == (3, 2)
    assert (summary.tp, summary.fp, summary.fn) == (2, 1, 0)
    assert summary.precision == pytest.approx(2 / 3)
    assert summary.recall == 1.0
    assert summary.f1 == pytest.approx(0.8)
    assert summary.modsa == pytest.approx(0.5)
    assert summary.mean_matched_iou == 1.0
    assert summary.pixel_iou == pytest.approx(4 / 5)


@pytest.mark.parametrize(
    ("pred", "truth", "error"),
    [
        (np.zeros((2, 2), dtype=np.uint8), np.zeros((3, 2), dtype=np.uint8), ValueError),
        (np.array([[0.0, 1.0]]), np.array([[0, 1]], dtype=np.uint8), TypeError),
        (np.array([[0, -1]]), np.array([[0, 1]]), ValueError),
        (np.array([], dtype=np.uint8), np.array([], dtype=np.uint8), ValueError),
    ],
)
def test_invalid_labels_are_rejected(pred, truth, error) -> None:
    with pytest.raises(error):
        evaluate_instance_labels(pred, truth)


def test_invalid_thresholds_are_rejected() -> None:
    labels = np.zeros((1, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="iou_threshold"):
        evaluate_instance_labels(labels, labels, iou_threshold=1.1)
    with pytest.raises(ValueError, match="topology_overlap_threshold"):
        evaluate_instance_labels(labels, labels, topology_overlap_threshold=0.0)
