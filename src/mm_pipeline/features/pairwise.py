"""Pairwise candidate feature extraction"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from mm_pipeline.config import DatasetSpec, TrackerParams
from mm_pipeline.core import (
    CandidateSolution,
    CellInstance,
    FramePair,
    TrackingOperation,
    canonical_ops_key,
    extract_cell_instances,
    sort_cells_along_trench,
)
from mm_pipeline.io.annotations import GTContext, build_gt_ops_for_pair, load_gt_context
from mm_pipeline.io.labels import load_labels_from_folder
from mm_pipeline.tracking.costs import EPS, touches_open
from mm_pipeline.tracking.workflow import extract_sorted_cells_for_stack, generate_tracking_candidates_for_stack

from .masks import as_2d_label_image, as_label_stack, get_label_mask, iou, shift_mask_y

FEATURE_COLUMNS: tuple[str, ...] = (
    "max_shrink_pct",
    "total_area_ratio_exit_adjusted",
    "exit_open_end_dist_median_norm",
    "link_area_ratio_median",
    "link_area_ratio_max",
    "link_dy_median_norm",
    "link_dy_max_norm",
    "link_iou_shifted_median",
    "div_mother_sum_area_ratio_max",
    "div_mother_sum_area_ratio_mean",
    "div_daughter_area_ratio_max",
    "div_daughter_area_ratio_mean",
    "div_mother_daughter_dy_max_norm",
    "div_mother_daughter_dy_mean_norm",
)

COUNT_COLUMNS: tuple[str, ...] = ("n_links", "n_exits", "n_divides")

SAMPLE_META_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "labels_dir",
    "t",
    "pair_id",
    "delta_t",
    "sample_rank",
    "dp_rank_global",
    "dp_cost",
    "is_dpt_best",
    "candidate_source",
    "n_candidates_pair",
    "sample_class",
    "is_correct",
    *COUNT_COLUMNS,
)

FAILURE_COLUMNS: tuple[str, ...] = ("dataset_id", "labels_dir", "t", "pair_id", "stage", "reason")


@dataclass(frozen=True)
class FeatureContext:
    """Inputs needed to compute one candidate's pairwise features"""

    label_t: Any
    label_k: Any
    cells_t: Sequence[CellInstance]
    cells_k: Sequence[CellInstance]
    candidate: CandidateSolution
    frame_pair: FramePair
    params: TrackerParams
    mask_cache_t: MutableMapping[int, Any] = field(default_factory=dict)
    mask_cache_k: MutableMapping[int, Any] = field(default_factory=dict)


@dataclass
class _FeatureStats:
    area_t: float
    area_k: float
    area_exits: float = 0.0
    max_shrink: float = 0.0
    exit_dist_norm: list[float] = field(default_factory=list)
    link_area_ratios: list[float] = field(default_factory=list)
    link_dy_norm: list[float] = field(default_factory=list)
    link_iou_shifted: list[float] = field(default_factory=list)
    div_mother_sum_area_ratios: list[float] = field(default_factory=list)
    div_daughter_area_ratios: list[float] = field(default_factory=list)
    div_mother_daughter_dy_norm: list[float] = field(default_factory=list)
    n_links: int = 0
    n_exits: int = 0
    n_divides: int = 0


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Feature table builders require pandas.") from exc
    return pd


def _sym_ratio(a: float, b: float) -> float:
    lo = min(float(a), float(b))
    hi = max(float(a), float(b))
    return hi / max(lo, EPS)


def _valid(vals: Iterable[float]) -> list[float]:
    return [float(v) for v in vals if not math.isnan(float(v))]


def _nanmedian(vals: Iterable[float]) -> float:
    clean = sorted(_valid(vals))
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2:
        return float(clean[mid])
    return float((clean[mid - 1] + clean[mid]) / 2.0)


def _nanmean(vals: Iterable[float]) -> float:
    clean = _valid(vals)
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def _nanmax(vals: Iterable[float]) -> float:
    clean = _valid(vals)
    if not clean:
        return float("nan")
    return float(max(clean))


def _feature_max_shrink_pct(stats: _FeatureStats) -> float:
    return float(stats.max_shrink * 100.0)


def _feature_total_area_ratio_exit_adjusted(stats: _FeatureStats) -> float:
    return float(stats.area_k / max(stats.area_t - stats.area_exits, EPS))


_FEATURE_REGISTRY: dict[str, Callable[[_FeatureStats], float]] = {
    "max_shrink_pct": _feature_max_shrink_pct,
    "total_area_ratio_exit_adjusted": _feature_total_area_ratio_exit_adjusted,
    "exit_open_end_dist_median_norm": lambda stats: _nanmedian(stats.exit_dist_norm),
    "link_area_ratio_median": lambda stats: _nanmedian(stats.link_area_ratios),
    "link_area_ratio_max": lambda stats: _nanmax(stats.link_area_ratios),
    "link_dy_median_norm": lambda stats: _nanmedian(stats.link_dy_norm),
    "link_dy_max_norm": lambda stats: _nanmax(stats.link_dy_norm),
    "link_iou_shifted_median": lambda stats: _nanmedian(stats.link_iou_shifted),
    "div_mother_sum_area_ratio_max": lambda stats: _nanmax(stats.div_mother_sum_area_ratios),
    "div_mother_sum_area_ratio_mean": lambda stats: _nanmean(stats.div_mother_sum_area_ratios),
    "div_daughter_area_ratio_max": lambda stats: _nanmax(stats.div_daughter_area_ratios),
    "div_daughter_area_ratio_mean": lambda stats: _nanmean(stats.div_daughter_area_ratios),
    "div_mother_daughter_dy_max_norm": lambda stats: _nanmax(stats.div_mother_daughter_dy_norm),
    "div_mother_daughter_dy_mean_norm": lambda stats: _nanmean(stats.div_mother_daughter_dy_norm),
}


def _cell_map(cells: Sequence[CellInstance], frame_name: str) -> dict[int, CellInstance]:
    out: dict[int, CellInstance] = {}
    for cell in cells:
        label = int(cell.label)
        if label in out:
            raise ValueError(f"{frame_name} contains duplicate label {label}.")
        out[label] = cell
    return out


def compute_solution_features(context: FeatureContext) -> dict[str, float]:
    """Compute the good old pairwise features for one candidate solution"""

    label_t = as_2d_label_image(context.label_t, name="label_t")
    label_k = as_2d_label_image(context.label_k, name="label_k")
    if label_t.shape != label_k.shape:
        raise ValueError(f"label_t and label_k shapes must match; got {label_t.shape} and {label_k.shape}.")
    if tuple(context.frame_pair.frame_shape) != tuple(label_t.shape):
        raise ValueError(
            f"FramePair.frame_shape must match label image shape; got {context.frame_pair.frame_shape} "
            f"and {label_t.shape}."
        )
    if context.params.axis != context.frame_pair.axis:
        raise ValueError(
            f"TrackerParams.axis ({context.params.axis!r}) must match FramePair.axis "
            f"({context.frame_pair.axis!r})."
        )

    h = int(label_t.shape[0])
    h_norm = max(h, 1)
    cells_t = tuple(context.cells_t)
    cells_k = tuple(context.cells_k)
    cells_t_by_label = _cell_map(cells_t, "cells_t")
    cells_k_by_label = _cell_map(cells_k, "cells_k")

    stats = _FeatureStats(
        area_t=float(sum(cell.area for cell in cells_t)),
        area_k=float(sum(cell.area for cell in cells_k)),
    )

    for op in context.candidate.ops:
        src = int(op.src_label)
        source = cells_t_by_label.get(src)
        if source is None:
            raise ValueError(f"Op references missing source label {src} in frame t.")

        if op.kind == "exit":
            stats.n_exits += 1
            stats.area_exits += float(source.area)
            dist = float(source.y) if context.frame_pair.open_end == "low" else float((h - 1) - source.y)
            stats.exit_dist_norm.append(dist / h_norm)
            continue

        if op.kind == "link":
            if op.dst1_label is None:
                raise ValueError("Link op missing destination label.")
            dst = int(op.dst1_label)
            dest = cells_k_by_label.get(dst)
            if dest is None:
                raise ValueError(f"Link op references missing destination label {dst} in frame k.")

            stats.n_links += 1
            stats.link_area_ratios.append(_sym_ratio(float(source.area), float(dest.area)))
            stats.link_dy_norm.append(abs(float(dest.y) - float(source.y)) / h_norm)

            if not (
                touches_open(source, context.frame_pair, context.params.border_margin)
                or touches_open(dest, context.frame_pair, context.params.border_margin)
            ):
                shrink = max(0.0, 1.0 - (float(dest.area) + EPS) / (float(source.area) + EPS))
                stats.max_shrink = max(stats.max_shrink, shrink)

            mask_t = get_label_mask(label_t, src, context.mask_cache_t)
            mask_k = get_label_mask(label_k, dst, context.mask_cache_k)
            shift = int(round(float(source.y) - float(dest.y)))
            stats.link_iou_shifted.append(iou(mask_t, shift_mask_y(mask_k, shift)))
            continue

        if op.kind == "divide":
            if op.dst1_label is None or op.dst2_label is None:
                raise ValueError("Divide op missing daughter labels.")
            d1 = int(op.dst1_label)
            d2 = int(op.dst2_label)
            dest1 = cells_k_by_label.get(d1)
            dest2 = cells_k_by_label.get(d2)
            if dest1 is None or dest2 is None:
                raise ValueError(f"Divide op references missing daughter labels ({d1}, {d2}) in frame k.")

            stats.n_divides += 1
            daughter_area = float(dest1.area + dest2.area)
            stats.div_mother_sum_area_ratios.append(_sym_ratio(float(source.area), daughter_area))
            stats.div_daughter_area_ratios.append(_sym_ratio(float(dest1.area), float(dest2.area)))
            stats.div_mother_daughter_dy_norm.append(abs(float(dest1.y) - float(source.y)) / h_norm)
            stats.div_mother_daughter_dy_norm.append(abs(float(dest2.y) - float(source.y)) / h_norm)
            continue

        raise ValueError(f"Unknown op kind '{op.kind}'.")

    features = {name: _FEATURE_REGISTRY[name](stats) for name in FEATURE_COLUMNS}
    features.update(
        {
            "n_links": float(stats.n_links),
            "n_exits": float(stats.n_exits),
            "n_divides": float(stats.n_divides),
        }
    )
    return features


def _resolve_params(params: TrackerParams | None, axis: str) -> TrackerParams:
    if params is None:
        return TrackerParams(axis=axis)
    if params.axis != axis:
        raise ValueError(f"TrackerParams.axis ({params.axis!r}) must match axis ({axis!r}).")
    return params


def _empty_feature_table(store_ops: bool = False):
    pd = _require_pandas()
    cols = list(SAMPLE_META_COLUMNS + FEATURE_COLUMNS)
    if store_ops:
        cols.append("ops_json")
    return pd.DataFrame(columns=cols)


def _unknown_correct_value():
    pd = _require_pandas()
    return pd.NA


def _candidate_row(
    *,
    label_t: Any,
    label_k: Any,
    cells_t: Sequence[CellInstance],
    cells_k: Sequence[CellInstance],
    candidate: CandidateSolution,
    frame_pair: FramePair,
    params: TrackerParams,
    labels_dir: str,
    sample_rank: int,
    n_candidates_pair: int,
    is_dpt_best: bool,
    gt_key: tuple[tuple[str, int, int, int], ...] | None,
    store_ops: bool,
    mask_cache_t: MutableMapping[int, Any],
    mask_cache_k: MutableMapping[int, Any],
) -> dict[str, Any]:
    features = compute_solution_features(
        FeatureContext(
            label_t=label_t,
            label_k=label_k,
            cells_t=cells_t,
            cells_k=cells_k,
            candidate=candidate,
            frame_pair=frame_pair,
            params=params,
            mask_cache_t=mask_cache_t,
            mask_cache_k=mask_cache_k,
        )
    )

    is_correct: Any
    if gt_key is None:
        is_correct = _unknown_correct_value()
        sample_class = "unknown"
    else:
        is_correct = canonical_ops_key(candidate.ops) == gt_key
        sample_class = "correct" if is_correct else "incorrect"

    row: dict[str, Any] = {
        "dataset_id": frame_pair.dataset_id,
        "labels_dir": labels_dir,
        "t": int(frame_pair.t),
        "pair_id": frame_pair.pair_id,
        "delta_t": int(frame_pair.k) - int(frame_pair.t),
        "sample_rank": int(sample_rank),
        "dp_rank_global": candidate.rank if candidate.generator.startswith("dp_") else _unknown_correct_value(),
        "dp_cost": float(candidate.cost) if candidate.cost is not None else _unknown_correct_value(),
        "is_dpt_best": bool(is_dpt_best),
        "candidate_source": str(candidate.generator),
        "n_candidates_pair": int(n_candidates_pair),
        "sample_class": sample_class,
        "is_correct": is_correct,
        "n_links": int(features["n_links"]),
        "n_exits": int(features["n_exits"]),
        "n_divides": int(features["n_divides"]),
    }
    for name in FEATURE_COLUMNS:
        row[name] = float(features[name])
    if store_ops:
        row["ops_json"] = candidate.to_ops_json()
    return row


def _lookup_gt_ops(
    gt_ops_by_pair: Mapping[Any, Iterable[TrackingOperation | Sequence[object]]] | None,
    frame_pair: FramePair,
) -> Iterable[TrackingOperation | Sequence[object]] | None:
    if not gt_ops_by_pair:
        return None
    for key in (frame_pair.pair_id, (frame_pair.t, frame_pair.k), frame_pair.t):
        if key in gt_ops_by_pair:
            return gt_ops_by_pair[key]
    return None


def _append_gt_candidate_if_needed(
    candidates: list[CandidateSolution],
    frame_pair: FramePair,
    gt_ops: Iterable[TrackingOperation | Sequence[object]] | None,
    include_gt_if_missing: bool,
) -> tuple[list[CandidateSolution], tuple[tuple[str, int, int, int], ...] | None]:
    if gt_ops is None:
        return candidates, None
    gt_candidate = CandidateSolution.from_ops(
        pair_id=frame_pair.pair_id,
        ops=list(gt_ops),
        generator="gt_injected",
        rank=None,
        cost=None,
    )
    gt_key = canonical_ops_key(gt_candidate.ops)
    if include_gt_if_missing and all(canonical_ops_key(candidate.ops) != gt_key for candidate in candidates):
        candidates.append(gt_candidate)
    return candidates, gt_key


def _mark_best_dp(candidates: Sequence[CandidateSolution]) -> int | None:
    dp_indices = [
        idx
        for idx, candidate in enumerate(candidates)
        if candidate.generator.startswith("dp_") and candidate.cost is not None and math.isfinite(float(candidate.cost))
    ]
    if not dp_indices:
        return None
    return min(dp_indices, key=lambda idx: float(candidates[idx].cost or math.inf))


def _rows_for_pair(
    *,
    label_t: Any,
    label_k: Any,
    cells_t: Sequence[CellInstance],
    cells_k: Sequence[CellInstance],
    candidates: Sequence[CandidateSolution],
    frame_pair: FramePair,
    params: TrackerParams,
    labels_dir: str,
    store_ops: bool,
    gt_ops: Iterable[TrackingOperation | Sequence[object]] | None = None,
    include_gt_if_missing: bool = False,
) -> list[dict[str, Any]]:
    candidate_list, gt_key = _append_gt_candidate_if_needed(
        list(candidates),
        frame_pair,
        gt_ops,
        include_gt_if_missing,
    )
    best_dp_idx = _mark_best_dp(candidate_list)
    n_candidates_pair = len(candidate_list)
    mask_cache_t: dict[int, Any] = {}
    mask_cache_k: dict[int, Any] = {}
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidate_list):
        rows.append(
            _candidate_row(
                label_t=label_t,
                label_k=label_k,
                cells_t=cells_t,
                cells_k=cells_k,
                candidate=candidate,
                frame_pair=frame_pair,
                params=params,
                labels_dir=labels_dir,
                sample_rank=idx + 1,
                n_candidates_pair=n_candidates_pair,
                is_dpt_best=idx == best_dp_idx,
                gt_key=gt_key,
                store_ops=store_ops,
                mask_cache_t=mask_cache_t,
                mask_cache_k=mask_cache_k,
            )
        )
    return rows


def _dataframe_from_rows(rows: list[dict[str, Any]], *, store_ops: bool):
    pd = _require_pandas()
    if not rows:
        return _empty_feature_table(store_ops=store_ops)
    cols = list(SAMPLE_META_COLUMNS + FEATURE_COLUMNS)
    if store_ops:
        cols.append("ops_json")
    df = pd.DataFrame(rows)
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[cols].copy()


def solve_and_featurize_pair(
    label_t: Any,
    label_k: Any,
    *,
    dataset_id: str = "dataset",
    t: int = 0,
    k: int = 1,
    axis: str = "y",
    open_end: str = "high",
    params: TrackerParams | None = None,
    top_k: int = 16,
    store_ops: bool = True,
) -> Any:
    """Solve and featurize one arbitrary label-image pair"""

    label_t_arr = as_2d_label_image(label_t, name="label_t")
    label_k_arr = as_2d_label_image(label_k, name="label_k")
    if label_t_arr.shape != label_k_arr.shape:
        raise ValueError(f"label_t and label_k shapes must match; got {label_t_arr.shape} and {label_k_arr.shape}.")
    if top_k < 1:
        return _empty_feature_table(store_ops=store_ops)

    resolved_params = _resolve_params(params, axis)
    cells_t = tuple(
        sort_cells_along_trench(
            extract_cell_instances(label_t_arr, dataset_id=dataset_id, frame=int(t)),
            axis=axis,
            open_end=open_end,
        )
    )
    cells_k = tuple(
        sort_cells_along_trench(
            extract_cell_instances(label_k_arr, dataset_id=dataset_id, frame=int(k)),
            axis=axis,
            open_end=open_end,
        )
    )
    frame_pair = FramePair(
        dataset_id=dataset_id,
        t=int(t),
        k=int(k),
        frame_shape=(int(label_t_arr.shape[0]), int(label_t_arr.shape[1])),
        axis=axis,  # type: ignore[arg-type]
        open_end=open_end,  # type: ignore[arg-type]
    )
    from mm_pipeline.tracking import solve_pair_topk

    candidates = solve_pair_topk(cells_t, cells_k, frame_pair, resolved_params, top_k=top_k)
    rows = _rows_for_pair(
        label_t=label_t_arr,
        label_k=label_k_arr,
        cells_t=cells_t,
        cells_k=cells_k,
        candidates=candidates,
        frame_pair=frame_pair,
        params=resolved_params,
        labels_dir="",
        store_ops=store_ops,
    )
    return _dataframe_from_rows(rows, store_ops=store_ops)


def build_feature_table_for_stack(
    labels: Any,
    *,
    dataset_id: str,
    axis: str = "y",
    open_end: str = "high",
    params: TrackerParams | None = None,
    top_k: int = 16,
    store_ops: bool = False,
    gt_ops_by_pair: Mapping[Any, Iterable[TrackingOperation | Sequence[object]]] | None = None,
    include_gt_if_missing: bool = False,
    labels_dir: str = "",
) -> Any:
    """Generate candidates and feature rows for every adjacent pair in a label stack"""

    arr = as_label_stack(labels)
    if top_k < 1:
        return _empty_feature_table(store_ops=store_ops)

    resolved_params = _resolve_params(params, axis)
    run = generate_tracking_candidates_for_stack(
        arr,
        dataset_id=dataset_id,
        axis=axis,
        open_end=open_end,
        params=resolved_params,
        mode="topk",
        top_k=top_k,
    )

    rows: list[dict[str, Any]] = []
    for result in run.pair_results:
        gt_ops = _lookup_gt_ops(gt_ops_by_pair, result.frame_pair)
        rows.extend(
            _rows_for_pair(
                label_t=arr[result.frame_pair.t],
                label_k=arr[result.frame_pair.k],
                cells_t=result.cells_t,
                cells_k=result.cells_k,
                candidates=result.candidates,
                frame_pair=result.frame_pair,
                params=resolved_params,
                labels_dir=labels_dir,
                store_ops=store_ops,
                gt_ops=gt_ops,
                include_gt_if_missing=include_gt_if_missing,
            )
        )

    return _dataframe_from_rows(rows, store_ops=store_ops)


def featurise_candidate_run(
    run: Any,
    *,
    labels: Any,
    params: TrackerParams | None = None,
    store_ops: bool = True,
    labels_dir: str = "",
    gt_ops_by_pair: Mapping[Any, Iterable[TrackingOperation | Sequence[object]]] | None = None,
    include_gt_if_missing: bool = False,
) -> Any:
    """Compute the 14 pairwise features for an existing TrackingCandidateRun

    This returns a DataFrame with SAMPLE_META_COLUMNS + FEATURE_COLUMNS
    (plus ops_json when store_ops=True). Behaviour is identical to
    the previous build_feature_table_for_stack function except the candidate run is
    pre-computed instead of re-generated by DP. Just helps modularity.

    The labels stack needs to match the one used to build run! The code doesn't check this so be responsible :)
    """

    arr = as_label_stack(labels)
    resolved_params = _resolve_params(params, run.axis)

    rows: list[dict[str, Any]] = []
    for result in run.pair_results:
        gt_ops = _lookup_gt_ops(gt_ops_by_pair, result.frame_pair)
        rows.extend(
            _rows_for_pair(
                label_t=arr[result.frame_pair.t],
                label_k=arr[result.frame_pair.k],
                cells_t=result.cells_t,
                cells_k=result.cells_k,
                candidates=result.candidates,
                frame_pair=result.frame_pair,
                params=resolved_params,
                labels_dir=labels_dir,
                store_ops=store_ops,
                gt_ops=gt_ops,
                include_gt_if_missing=include_gt_if_missing,
            )
        )

    return _dataframe_from_rows(rows, store_ops=store_ops)


def _normalize_specs(datasets: DatasetSpec | Sequence[DatasetSpec]) -> list[DatasetSpec]:
    if isinstance(datasets, DatasetSpec):
        return [datasets]
    specs = list(datasets)
    if not specs:
        raise ValueError("datasets must be non-empty.")
    return specs


def _empty_failures_table():
    pd = _require_pandas()
    return pd.DataFrame(columns=list(FAILURE_COLUMNS))


def _build_saved_gt_ops_by_pair(
    labels: Any,
    *,
    dataset_id: str,
    axis: str,
    open_end: str,
    gt_ctx: GTContext,
) -> dict[str, list[TrackingOperation]]:
    """Build adjacent-frame GT ops without running DP candidate generation"""

    arr = as_label_stack(labels)
    cells_by_frame = extract_sorted_cells_for_stack(
        arr,
        dataset_id=dataset_id,
        axis=axis,
        open_end=open_end,
    )
    frame_shape = (int(arr.shape[1]), int(arr.shape[2]))
    gt_ops_by_pair: dict[str, list[TrackingOperation]] = {}
    for t in range(int(arr.shape[0]) - 1):
        frame_pair = FramePair(
            dataset_id=dataset_id,
            t=t,
            k=t + 1,
            frame_shape=frame_shape,
            axis=axis,  # type: ignore[arg-type]
            open_end=open_end,  # type: ignore[arg-type]
        )
        gt_ops_by_pair[frame_pair.pair_id] = build_gt_ops_for_pair(
            frame_pair.t,
            cells_by_frame[t],
            gt_ctx,
        )
    return gt_ops_by_pair


def _failure_row(
    *,
    dataset_id: str,
    labels_dir: str,
    t: int | None,
    pair_id: str,
    stage: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "labels_dir": labels_dir,
        "t": _unknown_correct_value() if t is None else int(t),
        "pair_id": pair_id,
        "stage": stage,
        "reason": reason,
    }


def build_feature_dataframe(
    datasets: DatasetSpec | Sequence[DatasetSpec],
    *,
    gt_mode: Literal["none", "saved"] = "none",
    top_k_candidates: int = 16,
    include_gt_if_missing: bool = False,
    store_ops: bool = False,
    strict: bool = False,
) -> tuple[Any, Any]:
    """Build feature rows for one or more manifest dataset specs"""

    pd = _require_pandas()
    if gt_mode not in ("none", "saved"):
        raise ValueError("gt_mode must be one of {'none', 'saved'}.")
    if top_k_candidates < 1:
        raise ValueError("top_k_candidates must be >= 1.")

    sample_frames: list[Any] = []
    failure_rows: list[dict[str, Any]] = []

    for spec in _normalize_specs(datasets):
        dataset_id = spec.dataset_id
        labels_path = spec.effective_labels_dir
        labels_dir = "" if labels_path is None else str(Path(labels_path))
        if labels_path is None:
            err = "DatasetSpec requires approved_labels_dir or labels_dir for feature extraction."
            if strict:
                raise ValueError(err)
            failure_rows.append(
                _failure_row(
                    dataset_id=dataset_id,
                    labels_dir="",
                    t=None,
                    pair_id="",
                    stage="dataset",
                    reason=err,
                )
            )
            continue

        try:
            labels = load_labels_from_folder(labels_path)
            resolved_params = TrackerParams(axis=spec.axis)
            gt_ops_by_pair: dict[str, list[TrackingOperation]] | None = None

            if gt_mode == "saved":
                if spec.gt_tracks_csv is None or spec.gt_divisions_csv is None:
                    raise ValueError(
                        f"Dataset '{dataset_id}' requires gt_tracks_csv and gt_divisions_csv for gt_mode='saved'."
                    )
                gt_ctx = load_gt_context(spec.gt_tracks_csv, spec.gt_divisions_csv)
                gt_ops_by_pair = _build_saved_gt_ops_by_pair(
                    labels,
                    dataset_id=dataset_id,
                    axis=spec.axis,
                    open_end=spec.open_end,
                    gt_ctx=gt_ctx,
                )

            sample_frames.append(
                build_feature_table_for_stack(
                    labels,
                    dataset_id=dataset_id,
                    axis=spec.axis,
                    open_end=spec.open_end,
                    params=resolved_params,
                    top_k=top_k_candidates,
                    store_ops=store_ops,
                    gt_ops_by_pair=gt_ops_by_pair,
                    include_gt_if_missing=include_gt_if_missing,
                    labels_dir=labels_dir,
                )
            )
        except Exception as exc:
            if strict:
                raise
            failure_rows.append(
                _failure_row(
                    dataset_id=dataset_id,
                    labels_dir=labels_dir,
                    t=None,
                    pair_id="",
                    stage="dataset",
                    reason=str(exc),
                )
            )

    if sample_frames:
        samples = pd.concat(sample_frames, ignore_index=True)
        if samples.empty:
            samples = _empty_feature_table(store_ops=store_ops)
    else:
        samples = _empty_feature_table(store_ops=store_ops)

    failures = pd.DataFrame(failure_rows) if failure_rows else _empty_failures_table()
    for col in FAILURE_COLUMNS:
        if col not in failures.columns:
            failures[col] = pd.NA
    return samples, failures[list(FAILURE_COLUMNS)].copy()
