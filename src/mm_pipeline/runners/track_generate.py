"""Runner for mm-pipeline candidates.

Generates candidate (t, t+1) mappings for every adjacent pair of approved
labels using the configured sampler

Output: a candidates parquet with the SAMPLE_META columns + ops_js``,
plus a sibling <out>.run.json carrying run metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from mm_pipeline.config import DatasetSpec, HypothesisModel, TrackerParams
from mm_pipeline.io.manifests import load_dataset_manifest
from mm_pipeline.tracking.workflow import (
    TrackingCandidateRun,
    candidates_to_dataframe,
    generate_tracking_candidates_from_labels_dir,
)

from ._outputs import make_run_metadata, write_single_artefact_outputs

Sampler = Literal["dp", "brute_force"]


@dataclass(frozen=True)
class CandidatesResult:
    """In-memory result of a ``run_candidates`` invocation"""

    candidates_df: Any
    runs_by_dataset: dict[str, TrackingCandidateRun] = field(default_factory=dict)
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


def _validate_sampler_hypothesis(sampler: Sampler, hm: HypothesisModel) -> None:
    if sampler == "brute_force":
        raise NotImplementedError(
            "brute_force sampler is planned but not yet implemented. "
            "Use --sampler dp."
        )
    if sampler != "dp":
        raise ValueError(f"Unknown sampler: {sampler!r}. Supported: 'dp', 'brute_force'.")
    if hm.name != "default":
        raise ValueError(
            f"Sampler 'dp' supports only HypothesisModel(name='default'); "
            f"got {hm.name!r}."
        )


def run_candidates(
    datasets: DatasetSpec | Sequence[DatasetSpec] | str | Path,
    *,
    tracker_params: TrackerParams | None = None,
    hypothesis_model: HypothesisModel | None = None,
    sampler: Sampler = "dp",
    top_k: int = 16,
    out_path: str | Path | None = None,
    overwrite: bool = False,
) -> CandidatesResult:
    """Generate candidate (t, t+1) mappings for every adjacent pair.

    Parameters
    ----------
    datasets : a DatasetSpec, a list of them, or a path to a CSV manifest
    tracker_params : optional TrackerParams 
    hypothesis_model : optional HypothesisModel (default: 'default')
    sampler : 'dp' (default) or 'brute_force' (raises NotImplementedError currently)
    top_k : number of candidates per pair (default: 16)
    out_path : optional output parquet path. When None, no files are
        written and result.output_path is None (notebook mode)
    
    """

    specs, manifest_path = _normalise_specs(datasets)
    resolved_params = tracker_params or TrackerParams()
    resolved_hm = hypothesis_model or HypothesisModel()
    _validate_sampler_hypothesis(sampler, resolved_hm)

    runs_by_dataset: dict[str, TrackingCandidateRun] = {}
    per_dataset_frames: list[Any] = []

    for spec in specs:
        labels_dir = spec.effective_labels_dir
        if labels_dir is None:
            raise ValueError(
                f"Dataset {spec.dataset_id!r} has no labels directory. "
                "candidates requires approved_labels_dir or labels_dir; "
                "run 'mm-pipeline segment' first."
            )
        run = generate_tracking_candidates_from_labels_dir(
            labels_dir,
            dataset_id=spec.dataset_id,
            axis=spec.axis,
            open_end=spec.open_end,
            params=resolved_params,
            mode="topk",
            top_k=top_k,
        )
        runs_by_dataset[spec.dataset_id] = run
        per_dataset_frames.append(
            candidates_to_dataframe(run, labels_dir=str(labels_dir), store_ops=True)
        )

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("run_candidates requires pandas.") from exc

    if per_dataset_frames:
        candidates_df = pd.concat(per_dataset_frames, ignore_index=True)
    else:
        candidates_df = candidates_to_dataframe(
            TrackingCandidateRun(
                dataset_id="",
                axis="y",
                open_end="high",
                frame_shape=(0, 0),
                cells_by_frame=tuple(),
                pair_results=tuple(),
            ),
            store_ops=True,
        )

    resolved_config = {
        "tracker": resolved_params.to_dict(),
        "candidates": {
            "top_k": int(top_k),
            "sampler": str(sampler),
            "hypothesis_model": resolved_hm.to_dict(),
        },
    }

    written_path: Path | None = None
    if out_path is not None:
        metadata = make_run_metadata(
            command="candidates",
            manifest_path=manifest_path,
            resolved_config=resolved_config,
            dataset_ids=[spec.dataset_id for spec in specs],
        )
        summary = {
            "n_datasets": len(specs),
            "n_pairs_total": int(sum(
                len(run.pair_results) for run in runs_by_dataset.values()
            )),
            "n_candidates_total": int(len(candidates_df)),
        }
        paths = write_single_artefact_outputs(
            out_path=out_path,
            artefact=candidates_df,
            metadata=metadata,
            summary=summary,
            overwrite=overwrite,
        )
        written_path = paths["artefact"]

    return CandidatesResult(
        candidates_df=candidates_df,
        runs_by_dataset=runs_by_dataset,
        resolved_config=resolved_config,
        output_path=written_path,
    )


__all__ = ["CandidatesResult", "run_candidates"]
