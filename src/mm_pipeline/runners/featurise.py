"""Runner for mm-pipeline featurise.

Consumes a candidates parquet from commit 5 plus the original labels and
emits a features parquet (SAMPLE_META + 14 FEATURE_COLUMNS + ops_json).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from mm_pipeline.config import DatasetSpec, TrackerParams
from mm_pipeline.core import (
    CandidateSolution,
    FramePair,
)
from mm_pipeline.features.pairwise import featurise_candidate_run
from mm_pipeline.io.labels import load_labels_from_folder
from mm_pipeline.io.manifests import load_dataset_manifest
from mm_pipeline.tracking.workflow import (
    PairCandidateResult,
    TrackingCandidateRun,
    extract_sorted_cells_for_stack,
)

from ._outputs import make_run_metadata, write_single_artefact_outputs


@dataclass(frozen=True)
class FeaturiseResult:
    """In-memory result of a ``run_featurise`` invocation."""

    features_df: Any
    resolved_config: dict[str, Any] = field(default_factory=dict)
    output_path: Path | None = None


def _normalise_specs(
    datasets: DatasetSpec | Sequence[DatasetSpec] | str | Path,
) -> tuple[list[DatasetSpec], str | None]:
    if isinstance(datasets, (str, Path)):
        return load_dataset_manifest(datasets), str(datasets)
    if isinstance(datasets, DatasetSpec):
        return [datasets], None
    specs = list(datasets)
    if not specs:
        raise ValueError("datasets must be non-empty.")
    for spec in specs:
        if not isinstance(spec, DatasetSpec):
            raise TypeError(
                f"datasets entries must be DatasetSpec; got {type(spec).__name__}."
            )
    return specs, None


def _candidate_run_from_dataframe(
    df: Any,
    *,
    dataset_spec: DatasetSpec,
    labels: Any,
) -> TrackingCandidateRun:
    """Reconstruct a TrackingCandidateRun from a candidates DataFrame slice.

    The slice must already be limited to one dataset_id. We rebuild
    cells_by_frame from the labels stack, then for each (t, k) group build
    the FramePair and CandidateSolution list from the parquet rows.
    """

    cells_by_frame = extract_sorted_cells_for_stack(
        labels,
        dataset_id=dataset_spec.dataset_id,
        axis=dataset_spec.axis,
        open_end=dataset_spec.open_end,
    )
    frame_shape = (int(labels.shape[1]), int(labels.shape[2]))

    pair_results: list[PairCandidateResult] = []
    # Sort by (t, sample_rank) so candidate order matches the original run.
    sort_cols = ["t", "sample_rank"]
    df_sorted = df.sort_values(sort_cols)
    for t_value, group in df_sorted.groupby("t", sort=True):
        t = int(t_value)
        delta_t = int(group["delta_t"].iloc[0])
        k = t + delta_t
        frame_pair = FramePair(
            dataset_id=dataset_spec.dataset_id,
            t=t,
            k=k,
            frame_shape=frame_shape,
            axis=dataset_spec.axis,  # type: ignore[arg-type]
            open_end=dataset_spec.open_end,  # type: ignore[arg-type]
        )
        candidates: list[CandidateSolution] = []
        for row in group.itertuples(index=False):
            row_dict = row._asdict()
            ops_json = row_dict.get("ops_json")
            if ops_json is None:
                raise ValueError(
                    "candidates DataFrame is missing 'ops_json'; "
                    "regenerate with store_ops=True (the default)."
                )
            rank_value = row_dict.get("dp_rank_global")
            cost_value = row_dict.get("dp_cost")
            try:
                rank = int(rank_value) if rank_value is not None and not _is_na(rank_value) else None
            except (TypeError, ValueError):
                rank = None
            try:
                cost = float(cost_value) if cost_value is not None and not _is_na(cost_value) else None
            except (TypeError, ValueError):
                cost = None
            candidates.append(
                CandidateSolution.from_ops_json(
                    pair_id=frame_pair.pair_id,
                    ops_json=str(ops_json),
                    generator=str(row_dict.get("candidate_source") or "dp_topk"),
                    rank=rank,
                    cost=cost,
                )
            )
        pair_results.append(
            PairCandidateResult(
                frame_pair=frame_pair,
                cells_t=cells_by_frame[t],
                cells_k=cells_by_frame[k],
                candidates=tuple(candidates),
            )
        )

    return TrackingCandidateRun(
        dataset_id=dataset_spec.dataset_id,
        axis=dataset_spec.axis,
        open_end=dataset_spec.open_end,
        frame_shape=frame_shape,
        cells_by_frame=cells_by_frame,
        pair_results=tuple(pair_results),
    )


def _is_na(value: Any) -> bool:
    try:
        import pandas as pd
        return bool(pd.isna(value))
    except Exception:
        return False


def _load_candidates(candidates: Any) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("run_featurise requires pandas.") from exc
    if isinstance(candidates, (str, Path)):
        return pd.read_parquet(candidates)
    if isinstance(candidates, pd.DataFrame):
        return candidates
    raise TypeError(
        f"candidates must be a DataFrame or path to a parquet; "
        f"got {type(candidates).__name__}."
    )


def run_featurise(
    datasets: DatasetSpec | Sequence[DatasetSpec] | str | Path,
    *,
    candidates: Any,
    tracker_params: TrackerParams | None = None,
    out_path: str | Path | None = None,
    overwrite: bool = False,
) -> FeaturiseResult:
    """Compute the 14 pairwise features for an existing candidates table.

    Parameters
    
    datasets : a DatasetSpec, a list of them, or a path to a CSV manifest.
        Used to resolve labels per dataset.
    candidates : either a candidates DataFrame (output of ``run_track_generate``)
        or a path to a parquet file containing one.
    tracker_params : optional TrackerParams. Defaults to package defaults.
    out_path : optional features-parquet path. When None, no files are
        written and ``result.output_path`` is None (notebook mode).
    overwrite : if True, allows clobbering an existing artefact path.
    """

    specs, manifest_path = _normalise_specs(datasets)
    resolved_params = tracker_params or TrackerParams()

    candidates_df = _load_candidates(candidates)
    if "dataset_id" not in candidates_df.columns:
        raise ValueError("candidates DataFrame is missing 'dataset_id' column.")

    import pandas as pd

    per_dataset_frames: list[Any] = []
    spec_by_id = {spec.dataset_id: spec for spec in specs}
    candidate_ids = set(candidates_df["dataset_id"].unique())
    missing_specs = candidate_ids - set(spec_by_id)
    if missing_specs:
        raise ValueError(
            f"candidates contains dataset_id(s) not in manifest: {sorted(missing_specs)}"
        )

    for spec in specs:
        slice_df = candidates_df[candidates_df["dataset_id"] == spec.dataset_id]
        if slice_df.empty:
            continue
        labels_dir = spec.effective_labels_dir
        if labels_dir is None:
            raise ValueError(
                f"Dataset {spec.dataset_id!r} has no labels directory. "
                "featurise requires approved_labels_dir or labels_dir."
            )
        labels = load_labels_from_folder(labels_dir)
        run = _candidate_run_from_dataframe(slice_df, dataset_spec=spec, labels=labels)
        features = featurise_candidate_run(
            run,
            labels=labels,
            params=resolved_params,
            labels_dir=str(labels_dir),
            store_ops=True,
        )
        per_dataset_frames.append(features)

    if per_dataset_frames:
        features_df = pd.concat(per_dataset_frames, ignore_index=True)
    else:
        # Same columns as featurise_candidate_run on an empty run
        from mm_pipeline.features import FEATURE_COLUMNS, SAMPLE_META_COLUMNS
        cols = list(SAMPLE_META_COLUMNS) + list(FEATURE_COLUMNS) + ["ops_json"]
        features_df = pd.DataFrame(columns=cols)

    resolved_config = {
        "tracker": resolved_params.to_dict(),
    }

    written_path: Path | None = None
    if out_path is not None:
        metadata = make_run_metadata(
            command="featurise",
            manifest_path=manifest_path,
            resolved_config=resolved_config,
            dataset_ids=[spec.dataset_id for spec in specs],
        )
        summary = {
            "n_datasets": len(specs),
            "n_rows": int(len(features_df)),
            "n_pairs": int(features_df["pair_id"].nunique()) if "pair_id" in features_df.columns else 0,
            "candidates_source": str(candidates) if isinstance(candidates, (str, Path)) else None,
        }
        paths = write_single_artefact_outputs(
            out_path=out_path,
            artefact=features_df,
            metadata=metadata,
            summary=summary,
            overwrite=overwrite,
        )
        written_path = paths["artefact"]

    return FeaturiseResult(
        features_df=features_df,
        resolved_config=resolved_config,
        output_path=written_path,
    )


__all__ = ["FeaturiseResult", "run_featurise"]
