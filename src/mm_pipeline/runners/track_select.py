"""Runner for mm-pipeline track-select"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from mm_pipeline.config import DatasetSpec
from mm_pipeline.io.labels import load_labels_from_folder
from mm_pipeline.io.manifests import load_dataset_manifest
from mm_pipeline.io.tracks import write_lineage_outputs
from mm_pipeline.tracking.lineage import reconstruct_lineage
from mm_pipeline.tracking.select import SelectionResult, build_scorer, select_pairs

from ._outputs import make_run_metadata, resolve_run_tag, write_multifile_outputs


@dataclass(frozen=True)
class TrackSelectResult:
    """In-memory result of a run_track_select invocation"""

    tracks_by_dataset: dict[str, Any] = field(default_factory=dict)
    events_by_dataset: dict[str, Any] = field(default_factory=dict)
    divisions_by_dataset: dict[str, Any] = field(default_factory=dict)
    selections_by_dataset: dict[str, list[SelectionResult]] = field(default_factory=dict)
    resolved_config: dict[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None


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


def _load_table(value: Any) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("run_track_select requires pandas.") from exc
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return pd.read_parquet(value)
    if isinstance(value, pd.DataFrame):
        return value
    raise TypeError(
        f"Input must be a DataFrame or path to a parquet; got {type(value).__name__}."
    )


def run_track_select(
    datasets: DatasetSpec | Sequence[DatasetSpec] | str | Path,
    *,
    scored: Any = None,
    features: Any = None,
    candidates: Any = None,
    scorer: str = "dp_cost_min",
    ensemble_alpha: float = 0.5,
    ensemble_mode: str = "rank",
    out_dir: str | Path | None = None,
    run_tag: str | None = None,
    overwrite: bool = False,
) -> TrackSelectResult:
    """Pick the best candidate per frame-pair and reconstruct tracks

    Picks the input frame in order: scored > features > candidates
    (the chosen scorer decides which columns it needs — dp_cost_min
    reads dp_cost, classifier reads raw_score). Per dataset: load
    labels, select one candidate per pair, and reconstruct the lineage
    """

    specs, manifest_path = _normalise_specs(datasets)

    input_df = _load_table(scored)
    if input_df is None:
        input_df = _load_table(features)
    if input_df is None:
        input_df = _load_table(candidates)
    if input_df is None:
        raise ValueError("run_track_select requires one of: scored, features, candidates.")

    scorer_obj = build_scorer(
        scorer, ensemble_alpha=ensemble_alpha, ensemble_mode=ensemble_mode
    )
    resolved_config = {
        "scorer": scorer,
        "ensemble_alpha": ensemble_alpha,
        "ensemble_mode": ensemble_mode,
    }

    write_dir: Path | None = None
    if out_dir is not None:
        write_dir = Path(out_dir) / resolve_run_tag(run_tag)

    tracks_by_dataset: dict[str, Any] = {}
    events_by_dataset: dict[str, Any] = {}
    divisions_by_dataset: dict[str, Any] = {}
    selections_by_dataset: dict[str, list[SelectionResult]] = {}

    for spec in specs:
        slice_df = input_df[input_df["dataset_id"] == spec.dataset_id]
        if slice_df.empty:
            continue
        labels_dir = spec.effective_labels_dir
        if labels_dir is None:
            raise ValueError(
                f"Dataset {spec.dataset_id!r} has no labels directory. "
                "track-select requires approved_labels_dir or labels_dir; "
                "run 'mm-pipeline segment' first."
            )
        labels = load_labels_from_folder(labels_dir)

        selections = select_pairs(slice_df, scorer_obj)
        tracks_df, events_df, divisions_df = reconstruct_lineage(
            selections,
            labels,
            open_end=spec.open_end,
            axis=spec.axis,
        )

        tracks_by_dataset[spec.dataset_id] = tracks_df
        events_by_dataset[spec.dataset_id] = events_df
        divisions_by_dataset[spec.dataset_id] = divisions_df
        selections_by_dataset[spec.dataset_id] = selections

        if write_dir is not None:
            dataset_dir = write_dir / spec.dataset_id
            dataset_dir.mkdir(parents=True, exist_ok=True)
            write_lineage_outputs(tracks_df, events_df, divisions_df, dataset_dir)

    output_dir: Path | None = None
    if write_dir is not None:
        metadata = make_run_metadata(
            command="track-select",
            manifest_path=manifest_path,
            resolved_config=resolved_config,
            dataset_ids=[spec.dataset_id for spec in specs],
        )
        summary = {
            "n_datasets": len(specs),
            "n_selections_total": sum(len(v) for v in selections_by_dataset.values()),
            "n_selections_per_dataset": {
                k: len(v) for k, v in selections_by_dataset.items()
            },
        }
        write_multifile_outputs(
            out_dir=write_dir,
            summary=summary,
            metadata=metadata,
            title="track-select run",
            overwrite=overwrite,
        )
        output_dir = write_dir

    return TrackSelectResult(
        tracks_by_dataset=tracks_by_dataset,
        events_by_dataset=events_by_dataset,
        divisions_by_dataset=divisions_by_dataset,
        selections_by_dataset=selections_by_dataset,
        resolved_config=resolved_config,
        output_dir=output_dir,
    )


__all__ = ["TrackSelectResult", "run_track_select"]
