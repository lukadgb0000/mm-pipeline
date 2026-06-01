"""In-memory tracking candidate workflows
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from mm_pipeline.config import TrackerParams
from mm_pipeline.core import (
    CandidateSolution,
    CellInstance,
    FramePair,
    extract_cell_instances,
    sort_cells_along_trench,
)
from mm_pipeline.io.labels import load_labels_from_folder

from .dp import solve_pair_best
from .topk import solve_pair_topk

CandidateMode = Literal["best", "topk"]


@dataclass(frozen=True)
class PairCandidateResult:
    """Candidate-generation output for one adjacent frame pair"""

    frame_pair: FramePair
    cells_t: tuple[CellInstance, ...]
    cells_k: tuple[CellInstance, ...]
    candidates: tuple[CandidateSolution, ...]


@dataclass(frozen=True)
class TrackingCandidateRun:
    """In-memory candidate-generation output for a label stack"""

    dataset_id: str
    axis: str
    open_end: str
    frame_shape: tuple[int, int]
    cells_by_frame: tuple[tuple[CellInstance, ...], ...]
    pair_results: tuple[PairCandidateResult, ...]

    @property
    def candidates(self) -> tuple[CandidateSolution, ...]:
        """Return all candidates flattened across frame pairs."""

        return tuple(candidate for result in self.pair_results for candidate in result.candidates)


def _as_label_stack(labels) -> object:
    import numpy as np

    arr = np.asarray(labels)
    if arr.ndim != 3:
        raise ValueError(f"labels must have shape (T,H,W); got {arr.shape}.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"labels dtype must be integer, got {arr.dtype}.")
    return arr


def _resolve_params(params: TrackerParams | None, axis: str) -> TrackerParams:
    return TrackerParams(axis=axis) if params is None else params


def extract_sorted_cells_for_stack(
    labels,
    *,
    dataset_id: str,
    axis: str,
    open_end: str,
) -> tuple[tuple[CellInstance, ...], ...]:
    """Extract and sort cell instances for every frame in a label stack."""

    arr = _as_label_stack(labels)
    cells_by_frame: list[tuple[CellInstance, ...]] = []
    for frame_idx, label_img in enumerate(arr):
        cells = extract_cell_instances(label_img, dataset_id=dataset_id, frame=frame_idx)
        cells_by_frame.append(tuple(sort_cells_along_trench(cells, axis=axis, open_end=open_end)))
    return tuple(cells_by_frame)


def generate_pair_candidates(
    cells_t: Sequence[CellInstance],
    cells_k: Sequence[CellInstance],
    frame_pair: FramePair,
    *,
    params: TrackerParams | None = None,
    mode: CandidateMode = "topk",
    top_k: int = 16,
) -> PairCandidateResult:
    """Generate candidates for one already-sorted adjacent frame pair."""

    resolved_params = _resolve_params(params, frame_pair.axis)
    if mode == "best":
        candidates = (solve_pair_best(cells_t, cells_k, frame_pair, resolved_params),)
    elif mode == "topk":
        candidates = tuple(solve_pair_topk(cells_t, cells_k, frame_pair, resolved_params, top_k=top_k))
    else:
        raise ValueError("mode must be 'best' or 'topk'.")

    return PairCandidateResult(
        frame_pair=frame_pair,
        cells_t=tuple(cells_t),
        cells_k=tuple(cells_k),
        candidates=candidates,
    )


def generate_tracking_candidates_for_stack(
    labels,
    *,
    dataset_id: str,
    axis: str = "y",
    open_end: str = "high",
    params: TrackerParams | None = None,
    mode: CandidateMode = "topk",
    top_k: int = 16,
) -> TrackingCandidateRun:
    """Generate pairwise DP candidates for every adjacent pair in a label stack"""

    arr = _as_label_stack(labels)
    resolved_params = _resolve_params(params, axis)
    cells_by_frame = extract_sorted_cells_for_stack(
        arr,
        dataset_id=dataset_id,
        axis=axis,
        open_end=open_end,
    )
    frame_shape = (int(arr.shape[1]), int(arr.shape[2]))

    pair_results: list[PairCandidateResult] = []
    for t in range(int(arr.shape[0]) - 1):
        frame_pair = FramePair(
            dataset_id=dataset_id,
            t=t,
            k=t + 1,
            frame_shape=frame_shape,
            axis=axis,
            open_end=open_end,
        )
        pair_results.append(
            generate_pair_candidates(
                cells_by_frame[t],
                cells_by_frame[t + 1],
                frame_pair,
                params=resolved_params,
                mode=mode,
                top_k=top_k,
            )
        )

    return TrackingCandidateRun(
        dataset_id=dataset_id,
        axis=axis,
        open_end=open_end,
        frame_shape=frame_shape,
        cells_by_frame=cells_by_frame,
        pair_results=tuple(pair_results),
    )


def generate_tracking_candidates_from_labels_dir(
    labels_dir,
    *,
    dataset_id: str,
    axis: str = "y",
    open_end: str = "high",
    params: TrackerParams | None = None,
    mode: CandidateMode = "topk",
    top_k: int = 16,
) -> TrackingCandidateRun:
    """Load a label TIFF directory and generate in-memory pair candidates"""

    labels = load_labels_from_folder(labels_dir)
    return generate_tracking_candidates_for_stack(
        labels,
        dataset_id=dataset_id,
        axis=axis,
        open_end=open_end,
        params=params,
        mode=mode,
        top_k=top_k,
    )


run_tracking_on_labels = generate_tracking_candidates_for_stack
run_tracking_from_labels_dir = generate_tracking_candidates_from_labels_dir


_META_COLUMNS_NO_GT: tuple[str, ...] = (
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
    "n_links",
    "n_exits",
    "n_divides",
)


def _best_dp_index(candidates: Sequence[CandidateSolution]) -> int | None:
    """Return the index of the lowest-cost DP candidate, or ``None`` if absent.

    Local copy of ``features.pairwise._mark_best_dp`` to avoid a cross-module
    private import.
    """

    import math

    dp_indices = [
        idx
        for idx, candidate in enumerate(candidates)
        if candidate.generator.startswith("dp_")
        and candidate.cost is not None
        and math.isfinite(float(candidate.cost))
    ]
    if not dp_indices:
        return None
    return min(dp_indices, key=lambda idx: float(candidates[idx].cost or math.inf))


def _count_ops(candidate: CandidateSolution) -> tuple[int, int, int]:
    """Return ``(n_links, n_exits, n_divides)`` for one candidate."""

    n_links = sum(1 for op in candidate.ops if op.kind == "link")
    n_exits = sum(1 for op in candidate.ops if op.kind == "exit")
    n_divides = sum(1 for op in candidate.ops if op.kind == "divide")
    return n_links, n_exits, n_divides


def candidates_to_dataframe(
    run: TrackingCandidateRun,
    *,
    labels_dir: str = "",
    store_ops: bool = True,
) -> Any:
    """Convert a ``TrackingCandidateRun`` to a meta-only candidates DataFrame.

    Returns one row per candidate with columns matching the SAMPLE_META block
    of [features.pairwise.SAMPLE_META_COLUMNS](../features/pairwise.py) plus
    ``ops_json`` when ``store_ops`` is true. No feature columns are computed, that's kept as a separate step

    Columns:
        dataset_id, labels_dir, t, pair_id, delta_t, sample_rank,
        dp_rank_global, dp_cost, is_dpt_best, candidate_source,
        n_candidates_pair, sample_class ("unknown"), is_correct (NA),
        n_links, n_exits, n_divides, [ops_json]
    """

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("candidates_to_dataframe requires pandas.") from exc

    cols = list(_META_COLUMNS_NO_GT)
    if store_ops:
        cols.append("ops_json")

    rows: list[dict[str, Any]] = []
    for pair_result in run.pair_results:
        frame_pair = pair_result.frame_pair
        candidate_list = list(pair_result.candidates)
        best_dp_idx = _best_dp_index(candidate_list)
        n_candidates_pair = len(candidate_list)
        for idx, candidate in enumerate(candidate_list):
            n_links, n_exits, n_divides = _count_ops(candidate)
            row: dict[str, Any] = {
                "dataset_id": frame_pair.dataset_id,
                "labels_dir": labels_dir,
                "t": int(frame_pair.t),
                "pair_id": frame_pair.pair_id,
                "delta_t": int(frame_pair.k) - int(frame_pair.t),
                "sample_rank": idx + 1,
                "dp_rank_global": candidate.rank if candidate.generator.startswith("dp_") else pd.NA,
                "dp_cost": float(candidate.cost) if candidate.cost is not None else pd.NA,
                "is_dpt_best": bool(idx == best_dp_idx),
                "candidate_source": str(candidate.generator),
                "n_candidates_pair": int(n_candidates_pair),
                "sample_class": "unknown",
                "is_correct": pd.NA,
                "n_links": int(n_links),
                "n_exits": int(n_exits),
                "n_divides": int(n_divides),
            }
            if store_ops:
                row["ops_json"] = candidate.to_ops_json()
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[cols].copy()
