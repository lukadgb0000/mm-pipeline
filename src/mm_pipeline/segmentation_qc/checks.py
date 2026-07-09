"""Automated segmentation QC checks"""

from __future__ import annotations

from typing import Iterable

from mm_pipeline.config import SegmentationQCConfig, SegmentationQCFinding


def _label_counts(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    ids, counts = np.unique(frame, return_counts=True)
    keep = ids != 0
    return ids[keep].astype(int), counts[keep].astype(int)


def find_small_labels(labels: np.ndarray, min_size: int) -> list[tuple[int, int, int]]:
    """Return (frame_idx, label_id, pixel_count) for labels below min_size."""

    import numpy as np
    if min_size <= 0:
        return []
    hits: list[tuple[int, int, int]] = []
    arr = np.asarray(labels)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    for frame_idx, frame in enumerate(arr):
        ids, counts = _label_counts(frame)
        for label_id, count in zip(ids, counts):
            if int(count) < min_size:
                hits.append((frame_idx, int(label_id), int(count)))
    return hits


def check_empty_frames(labels: np.ndarray, dataset_id: str) -> list[SegmentationQCFinding]:
    import numpy as np

    findings: list[SegmentationQCFinding] = []
    arr = np.asarray(labels)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    for frame_idx, frame in enumerate(arr):
        if not np.any(frame > 0):
            findings.append(
                SegmentationQCFinding(
                    dataset_id=dataset_id,
                    frame=frame_idx,
                    severity="error",
                    check_name="empty_frame",
                    message="Frame contains no positive labels.",
                    metric_name="n_labels",
                    metric_value=0.0,
                    threshold=1.0,
                )
            )
    return findings


def check_small_labels(labels: np.ndarray, dataset_id: str, min_size: int) -> list[SegmentationQCFinding]:
    return [
        SegmentationQCFinding(
            dataset_id=dataset_id,
            frame=frame_idx,
            severity="warning",
            check_name="small_label",
            message=f"Label {label_id} has {count} pixels, below minimum {min_size}.",
            label_id=label_id,
            metric_name="pixel_count",
            metric_value=float(count),
            threshold=float(min_size),
        )
        for frame_idx, label_id, count in find_small_labels(labels, min_size)
    ]


def check_cell_count_jumps(
    labels: np.ndarray,
    dataset_id: str,
    threshold: int,
) -> list[SegmentationQCFinding]:
    import numpy as np

    if threshold <= 0:
        return []
    arr = np.asarray(labels)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    counts = [int(len(_label_counts(frame)[0])) for frame in arr]
    findings: list[SegmentationQCFinding] = []
    for frame_idx in range(1, len(counts)):
        delta = counts[frame_idx] - counts[frame_idx - 1]
        if abs(delta) > threshold:
            findings.append(
                SegmentationQCFinding(
                    dataset_id=dataset_id,
                    frame=frame_idx,
                    severity="warning",
                    check_name="cell_count_jump",
                    message=f"Cell count changed by {delta} from previous frame.",
                    metric_name="cell_count_delta",
                    metric_value=float(delta),
                    threshold=float(threshold),
                    metrics={"prev_count": counts[frame_idx - 1], "count": counts[frame_idx]},
                )
            )
    return findings


def check_total_area_jumps(
    labels: np.ndarray,
    dataset_id: str,
    threshold_fraction: float,
) -> list[SegmentationQCFinding]:
    import numpy as np

    if threshold_fraction <= 0:
        return []
    arr = np.asarray(labels)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    areas = [int(np.count_nonzero(frame > 0)) for frame in arr]
    findings: list[SegmentationQCFinding] = []
    for frame_idx in range(1, len(areas)):
        prev = max(areas[frame_idx - 1], 1)
        frac = (areas[frame_idx] - areas[frame_idx - 1]) / prev
        if abs(frac) > threshold_fraction:
            findings.append(
                SegmentationQCFinding(
                    dataset_id=dataset_id,
                    frame=frame_idx,
                    severity="warning",
                    check_name="total_area_jump",
                    message=f"Total mask area changed by {frac:.3f} from previous frame.",
                    metric_name="total_area_delta_fraction",
                    metric_value=float(frac),
                    threshold=float(threshold_fraction),
                    metrics={"prev_area": areas[frame_idx - 1], "area": areas[frame_idx]},
                )
            )
    return findings


def check_area_outliers(
    labels: np.ndarray,
    dataset_id: str,
    *,
    small_quantile: float,
    large_quantile: float,
) -> list[SegmentationQCFinding]:
    import numpy as np

    arr = np.asarray(labels)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    records: list[tuple[int, int, int]] = []
    for frame_idx, frame in enumerate(arr):
        ids, counts = _label_counts(frame)
        records.extend((frame_idx, int(label_id), int(count)) for label_id, count in zip(ids, counts))
    if len(records) < 3:
        return []

    values = np.asarray([r[2] for r in records], dtype=float)
    low = float(np.quantile(values, small_quantile))
    high = float(np.quantile(values, large_quantile))
    findings: list[SegmentationQCFinding] = []
    for frame_idx, label_id, count in records:
        if count < low:
            findings.append(
                SegmentationQCFinding(
                    dataset_id=dataset_id,
                    frame=frame_idx,
                    severity="info",
                    check_name="small_area_outlier",
                    message=f"Label {label_id} area {count} is below q={small_quantile}.",
                    label_id=label_id,
                    metric_name="pixel_count",
                    metric_value=float(count),
                    threshold=low,
                )
            )
        elif count > high:
            findings.append(
                SegmentationQCFinding(
                    dataset_id=dataset_id,
                    frame=frame_idx,
                    severity="info",
                    check_name="large_area_outlier",
                    message=f"Label {label_id} area {count} is above q={large_quantile}.",
                    label_id=label_id,
                    metric_name="pixel_count",
                    metric_value=float(count),
                    threshold=high,
                )
            )
    return findings


def run_basic_checks(
    labels: np.ndarray,
    dataset_id: str,
    config: SegmentationQCConfig | None = None,
) -> list[SegmentationQCFinding]:
    cfg = config or SegmentationQCConfig()
    findings: list[SegmentationQCFinding] = []
    for group in [
        check_empty_frames(labels, dataset_id),
        check_small_labels(labels, dataset_id, cfg.min_label_size),
        check_cell_count_jumps(labels, dataset_id, cfg.cell_count_jump_threshold),
        check_total_area_jumps(labels, dataset_id, cfg.total_area_jump_fraction),
        check_area_outliers(
            labels,
            dataset_id,
            small_quantile=cfg.small_area_quantile,
            large_quantile=cfg.large_area_quantile,
        ),
    ]:
        findings.extend(group)
    return findings


def findings_to_rows(findings: Iterable[SegmentationQCFinding]) -> list[dict[str, object]]:
    return [finding.to_dict() for finding in findings]
    
