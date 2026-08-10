"""Instance-level evaluation for labelled segmentation masks.

The matching objective is lexicographic, ie maximise the number of one-to-one
matches at or above the IoU threshold, then maximise their total IoU.  The
implementation uses a small min-cost-flow solver so the segmentation package
does not acquire a SciPy dependency merely for assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import nan
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class InstanceMatch:
    """One accepted predicted-label to ground-truth-label match."""

    pred_label: int
    truth_label: int
    iou: float


@dataclass(frozen=True)
class InstanceSegmentationEvaluation:
    """Counts and label identities from evaluating one pair of masks."""

    n_pred: int
    n_truth: int
    tp: int
    fp: int
    fn: int
    matches: tuple[InstanceMatch, ...]
    unmatched_pred_labels: tuple[int, ...]
    unmatched_truth_labels: tuple[int, ...]
    pixel_intersection: int
    pixel_union: int
    split_count: int
    merge_count: int


@dataclass(frozen=True)
class InstanceSegmentationSummary:
    """Aggregate detection and boundary metrics over one or more frames."""

    n_frames: int
    n_pred: int
    n_truth: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    modsa: float
    mean_matched_iou: float
    pixel_iou: float
    split_count: int
    merge_count: int


@dataclass
class _Edge:
    to: int
    rev: int
    capacity: int
    cost: float


def _add_edge(graph: list[list[_Edge]], src: int, dst: int, cost: float) -> None:
    forward = _Edge(dst, len(graph[dst]), 1, cost)
    reverse = _Edge(src, len(graph[src]), 0, -cost)
    graph[src].append(forward)
    graph[dst].append(reverse)


def _maximum_cardinality_weight_matching(
    weights: np.ndarray,
    valid: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    """Return max-cardinality, max-weight row/column pairs.

    Unit-capacity min-cost flow first sends as much flow as possible.  Because
    each accepted edge has cost ``-weight``, the minimum-cost maximum flow also
    maximises total weight.  Bellman-Ford permits residual rerouting and makes
    the result exact rather than greedy.
    """

    n_rows, n_cols = weights.shape
    if n_rows == 0 or n_cols == 0 or not np.any(valid):
        return ()

    source = 0
    row_offset = 1
    col_offset = row_offset + n_rows
    sink = col_offset + n_cols
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]

    for row in range(n_rows):
        _add_edge(graph, source, row_offset + row, 0.0)
    for row in range(n_rows):
        for col in range(n_cols):
            if bool(valid[row, col]):
                _add_edge(
                    graph,
                    row_offset + row,
                    col_offset + col,
                    -float(weights[row, col]),
                )
    for col in range(n_cols):
        _add_edge(graph, col_offset + col, sink, 0.0)

    n_nodes = len(graph)
    eps = 1e-15
    while True:
        dist = [float("inf")] * n_nodes
        parent: list[tuple[int, int] | None] = [None] * n_nodes
        dist[source] = 0.0

        for _ in range(n_nodes - 1):
            changed = False
            for node, edges in enumerate(graph):
                if not np.isfinite(dist[node]):
                    continue
                for edge_idx, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    candidate = dist[node] + edge.cost
                    if candidate < dist[edge.to] - eps:
                        dist[edge.to] = candidate
                        parent[edge.to] = (node, edge_idx)
                        changed = True
            if not changed:
                break

        if parent[sink] is None:
            break

        node = sink
        while node != source:
            previous, edge_idx = parent[node]  # type: ignore[misc]
            edge = graph[previous][edge_idx]
            edge.capacity -= 1
            graph[node][edge.rev].capacity += 1
            node = previous

    pairs: list[tuple[int, int]] = []
    for row in range(n_rows):
        row_node = row_offset + row
        for edge in graph[row_node]:
            if col_offset <= edge.to < sink and edge.capacity == 0:
                pairs.append((row, edge.to - col_offset))
    pairs.sort()
    return tuple(pairs)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else nan


def _label_overlap_tables(
    pred: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_ids, pred_inverse = np.unique(pred[pred > 0], return_inverse=True)
    truth_ids, truth_inverse = np.unique(truth[truth > 0], return_inverse=True)
    pred_areas = np.bincount(pred_inverse, minlength=len(pred_ids)).astype(np.int64)
    truth_areas = np.bincount(truth_inverse, minlength=len(truth_ids)).astype(np.int64)

    intersections = np.zeros((len(pred_ids), len(truth_ids)), dtype=np.int64)
    overlap = (pred > 0) & (truth > 0)
    if np.any(overlap):
        pred_overlap = np.searchsorted(pred_ids, pred[overlap])
        truth_overlap = np.searchsorted(truth_ids, truth[overlap])
        flat = pred_overlap * len(truth_ids) + truth_overlap
        intersections = np.bincount(
            flat,
            minlength=len(pred_ids) * len(truth_ids),
        ).reshape(len(pred_ids), len(truth_ids))

    return pred_ids, truth_ids, pred_areas, truth_areas, intersections


def evaluate_instance_labels(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    iou_threshold: float = 0.5,
    topology_overlap_threshold: float = 0.1,
) -> InstanceSegmentationEvaluation:
    """Evaluate one predicted label image against ground truth.

    Labels must be non-negative integer arrays with identical, non-empty
    shapes.  Label zero is background; positive label values need not agree
    between the two images.
    """

    pred_arr = np.asarray(pred)
    truth_arr = np.asarray(truth)
    if pred_arr.shape != truth_arr.shape:
        raise ValueError(
            f"Prediction and truth shapes differ: {pred_arr.shape} != {truth_arr.shape}."
        )
    if pred_arr.size == 0:
        raise ValueError("Prediction and truth arrays must be non-empty.")
    if not np.issubdtype(pred_arr.dtype, np.integer):
        raise TypeError("Prediction labels must have an integer dtype.")
    if not np.issubdtype(truth_arr.dtype, np.integer):
        raise TypeError("Ground-truth labels must have an integer dtype.")
    if np.any(pred_arr < 0) or np.any(truth_arr < 0):
        raise ValueError("Prediction and truth labels must be non-negative.")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1].")
    if not 0.0 < topology_overlap_threshold <= 1.0:
        raise ValueError("topology_overlap_threshold must be in (0, 1].")

    pred_ids, truth_ids, pred_areas, truth_areas, intersections = _label_overlap_tables(
        pred_arr, truth_arr
    )
    unions = (
        pred_areas[:, None] + truth_areas[None, :] - intersections
        if len(pred_ids) and len(truth_ids)
        else np.zeros((len(pred_ids), len(truth_ids)), dtype=np.int64)
    )
    iou = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=float),
        where=unions > 0,
    )
    # Even at a caller-supplied threshold of zero, spatially disjoint labels
    # are not matches merely because their IoU equals the numeric threshold.
    valid_matches = (intersections > 0) & (iou >= iou_threshold)
    pairs = _maximum_cardinality_weight_matching(iou, valid_matches)

    matches = tuple(
        InstanceMatch(int(pred_ids[row]), int(truth_ids[col]), float(iou[row, col]))
        for row, col in pairs
    )
    matched_pred = {match.pred_label for match in matches}
    matched_truth = {match.truth_label for match in matches}
    unmatched_pred = tuple(int(x) for x in pred_ids if int(x) not in matched_pred)
    unmatched_truth = tuple(int(x) for x in truth_ids if int(x) not in matched_truth)

    if len(pred_ids) and len(truth_ids):
        truth_cover = intersections / truth_areas[None, :]
        pred_cover = intersections / pred_areas[:, None]
        split_count = int(
            np.sum(np.sum(truth_cover >= topology_overlap_threshold, axis=0) >= 2)
        )
        merge_count = int(
            np.sum(np.sum(pred_cover >= topology_overlap_threshold, axis=1) >= 2)
        )
    else:
        split_count = 0
        merge_count = 0

    pixel_intersection = int(np.count_nonzero((pred_arr > 0) & (truth_arr > 0)))
    pixel_union = int(np.count_nonzero((pred_arr > 0) | (truth_arr > 0)))
    tp = len(matches)
    return InstanceSegmentationEvaluation(
        n_pred=len(pred_ids),
        n_truth=len(truth_ids),
        tp=tp,
        fp=len(pred_ids) - tp,
        fn=len(truth_ids) - tp,
        matches=matches,
        unmatched_pred_labels=unmatched_pred,
        unmatched_truth_labels=unmatched_truth,
        pixel_intersection=pixel_intersection,
        pixel_union=pixel_union,
        split_count=split_count,
        merge_count=merge_count,
    )


def aggregate_instance_evaluations(
    evaluations: Iterable[InstanceSegmentationEvaluation],
) -> InstanceSegmentationSummary:
    """Aggregate frame evaluations by summing their sufficient statistics."""

    rows = tuple(evaluations)
    n_pred = sum(row.n_pred for row in rows)
    n_truth = sum(row.n_truth for row in rows)
    tp = sum(row.tp for row in rows)
    fp = sum(row.fp for row in rows)
    fn = sum(row.fn for row in rows)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    modsa = _safe_ratio(tp - fp, n_truth)
    matched_ious = [match.iou for row in rows for match in row.matches]
    mean_matched_iou = float(np.mean(matched_ious)) if matched_ious else nan
    pixel_intersection = sum(row.pixel_intersection for row in rows)
    pixel_union = sum(row.pixel_union for row in rows)
    return InstanceSegmentationSummary(
        n_frames=len(rows),
        n_pred=n_pred,
        n_truth=n_truth,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        modsa=modsa,
        mean_matched_iou=mean_matched_iou,
        pixel_iou=_safe_ratio(pixel_intersection, pixel_union),
        split_count=sum(row.split_count for row in rows),
        merge_count=sum(row.merge_count for row in rows),
    )


__all__ = [
    "InstanceMatch",
    "InstanceSegmentationEvaluation",
    "InstanceSegmentationSummary",
    "aggregate_instance_evaluations",
    "evaluate_instance_labels",
]
