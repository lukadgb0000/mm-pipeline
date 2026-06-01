"""Runner for mm-pipeline qa

Wraps within-pair picking + per-pair anomaly detection + drop / bridge +
lineage reconstruction in one orchestrator. Not sure I like this will come back
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from mm_pipeline.config import DatasetSpec, QAConfig, TrackerParams
from mm_pipeline.features import FEATURE_COLUMNS
from mm_pipeline.io.labels import load_labels_from_folder
from mm_pipeline.io.manifests import load_dataset_manifest
from mm_pipeline.qa import (
    QADecision,
    apply_qa_workflow,
    build_detector,
    build_per_pair_features,
    build_scorer,
    decisions_to_dataframe,
    write_lineage_outputs,
    write_qa_decisions_csv,
)
from mm_pipeline.qa.bridge import ReuseCandidateClassifier
from mm_pipeline.scoring import FittedScorer, load_scorer
from mm_pipeline.tracking.lineage import reconstruct_from_qa_decisions

from ._outputs import (
    make_run_metadata,
    resolve_run_tag,
    write_multifile_outputs,
)


@dataclass(frozen=True)
class QAResult:
    """In-memory result of a ``run_qa`` invocation"""

    tracks_by_dataset: dict[str, Any] = field(default_factory=dict)
    events_by_dataset: dict[str, Any] = field(default_factory=dict)
    divisions_by_dataset: dict[str, Any] = field(default_factory=dict)
    decisions_by_dataset: dict[str, list[QADecision]] = field(default_factory=dict)
    per_pair_features_by_dataset: dict[str, Any] = field(default_factory=dict)
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
        raise RuntimeError("run_qa requires pandas.") from exc
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return pd.read_parquet(value)
    if isinstance(value, pd.DataFrame):
        return value
    raise TypeError(
        f"Input must be a DataFrame or path to a parquet; got {type(value).__name__}."
    )


def _load_model_optional(model: Any) -> FittedScorer | None:
    if model is None:
        return None
    if isinstance(model, FittedScorer):
        return model
    if isinstance(model, (str, Path)):
        return load_scorer(model)
    raise TypeError(
        f"model must be a FittedScorer or path to a joblib; "
        f"got {type(model).__name__}."
    )


def _validate_columns(input_df: Any, *, config: QAConfig, has_model: bool) -> None:
    """Validate that input columns satisfy the active QAConfig.

    Raises ValueError with a clear "run upstream command first" 
    """

    cols = set(input_df.columns)
    if "pair_id" not in cols:
        raise ValueError("input is missing required column 'pair_id'.")

    if config.within_pair_scorer == "dp_cost_min":
        if "dp_cost" not in cols:
            raise ValueError(
                "within_pair_scorer='dp_cost_min' requires 'dp_cost'. "
                "Run 'mm-pipeline candidates' or 'mm-pipeline featurise' first."
            )
    elif config.within_pair_scorer in {"classifier", "ensemble"}:
        if "raw_score" not in cols:
            raise ValueError(
                f"within_pair_scorer='{config.within_pair_scorer}' requires "
                "'raw_score'. Run 'mm-pipeline score' first."
            )

    if config.anomaly_detector != "never_anomalous":
        missing_features = [c for c in FEATURE_COLUMNS if c not in cols]
        if missing_features:
            raise ValueError(
                f"anomaly_detector='{config.anomaly_detector}' requires per-pair "
                f"feature columns. Missing: {missing_features[:3]}{'...' if len(missing_features) > 3 else ''}. "
                "Run 'mm-pipeline featurise' first."
            )

    if config.bridge_enabled:
        if "raw_score" not in cols:
            raise ValueError(
                "bridge_enabled=True requires 'raw_score'. "
                "Run 'mm-pipeline score' first."
            )
        if not has_model:
            raise ValueError(
                "bridge_enabled=True requires --model (the candidate scorer is "
                "reused for bridge scoring via ReuseCandidateClassifier)."
            )


def run_qa(
    datasets: DatasetSpec | Sequence[DatasetSpec] | str | Path,
    *,
    scored: Any = None,
    features: Any = None,
    candidates: Any = None,
    qa_config: QAConfig | None = None,
    model: Any = None,
    anomaly_model: str | Path | None = None,
    tracker_params: TrackerParams | None = None,
    out_dir: str | Path | None = None,
    run_tag: str | None = None,
    overwrite: bool = False,
) -> QAResult:
    """End-to-end QA workflow + lineage reconstruction

    Picks the input frame in order: ``scored`` > ``features`` > ``candidates``.
    Validates columns against the resolved ``QAConfig`` and fails with
    a clear message if a required column is missing

    Per dataset:
      Load labels via ``dataset.effective_labels_dir``.
      Slice the input frame to this ``dataset_id``.
      Build per-pair features (we do that always bc the cost is small and the anomaly detector may need them).
      Build helpers 
      Apply the QA workflow gives list[QADecision].
      Reconstruct lineage gives (tracks_df, events_df, divisions_df)
    """

    specs, manifest_path = _normalise_specs(datasets)
    resolved_config = qa_config or QAConfig()
    resolved_tracker = tracker_params or TrackerParams()

    fitted_scorer = _load_model_optional(model)
    has_model = fitted_scorer is not None

    # Pick input frame (prefer scored > features > candidates).
    input_df = _load_table(scored)
    if input_df is None:
        input_df = _load_table(features)
    if input_df is None:
        input_df = _load_table(candidates)
    if input_df is None:
        raise ValueError(
            "run_qa requires one of: scored, features, candidates."
        )

    _validate_columns(input_df, config=resolved_config, has_model=has_model)

    
    within_pair = build_scorer(
        resolved_config.within_pair_scorer,
        ensemble_alpha=resolved_config.within_pair_ensemble_alpha,
        ensemble_mode=resolved_config.within_pair_ensemble_mode,
    )
    detector_name = (
        str(anomaly_model) if anomaly_model is not None else resolved_config.anomaly_detector
    )
    detector = build_detector(detector_name)
    bridge_scorer = (
        ReuseCandidateClassifier(fitted_scorer)
        if (resolved_config.bridge_enabled and fitted_scorer is not None)
        else None
    )

    write_dir: Path | None = None
    if out_dir is not None:
        write_dir = Path(out_dir) / resolve_run_tag(run_tag)

    tracks_by_dataset: dict[str, Any] = {}
    events_by_dataset: dict[str, Any] = {}
    divisions_by_dataset: dict[str, Any] = {}
    decisions_by_dataset: dict[str, list[QADecision]] = {}
    per_pair_by_dataset: dict[str, Any] = {}

    for spec in specs:
        slice_df = input_df[input_df["dataset_id"] == spec.dataset_id]
        if slice_df.empty:
            continue
        labels_dir = spec.effective_labels_dir
        if labels_dir is None:
            raise ValueError(
                f"Dataset {spec.dataset_id!r} has no labels directory. "
                "qa requires approved_labels_dir or labels_dir; "
                "run 'mm-pipeline segment' first."
            )
        labels = load_labels_from_folder(labels_dir)

        # Always compute per-pair features when the input has the feature
        # columns (decided in phase11discussion §4.7). The anomaly detector
        # only consumes them when active.
        if all(c in slice_df.columns for c in FEATURE_COLUMNS):
            per_pair_features = build_per_pair_features(slice_df)
        else:
            per_pair_features = None

        decisions = apply_qa_workflow(
            slice_df,
            config=resolved_config,
            within_pair_scorer=within_pair,
            anomaly_detector=detector,
            bridge_scorer=bridge_scorer,
            labels=labels,
            tracker_params=resolved_tracker,
            open_end=spec.open_end,
        )
        tracks_df, events_df, divisions_df = reconstruct_from_qa_decisions(
            decisions,
            slice_df,
            labels,
            open_end=spec.open_end,
            axis=spec.axis,
        )

        tracks_by_dataset[spec.dataset_id] = tracks_df
        events_by_dataset[spec.dataset_id] = events_df
        divisions_by_dataset[spec.dataset_id] = divisions_df
        decisions_by_dataset[spec.dataset_id] = list(decisions)
        if per_pair_features is not None:
            per_pair_by_dataset[spec.dataset_id] = per_pair_features

        if write_dir is not None:
            dataset_dir = write_dir / spec.dataset_id
            dataset_dir.mkdir(parents=True, exist_ok=True)
            write_lineage_outputs(tracks_df, events_df, divisions_df, dataset_dir)
            write_qa_decisions_csv(decisions, dataset_dir / "qa_decisions.csv")
            if per_pair_features is not None:
                per_pair_features.to_parquet(dataset_dir / "per_pair_features.parquet", index=False)

    output_dir: Path | None = None
    if write_dir is not None:
        metadata = make_run_metadata(
            command="qa",
            manifest_path=manifest_path,
            resolved_config={
                "qa": resolved_config.to_dict(),
                "tracker": resolved_tracker.to_dict(),
                "model_path": str(model) if isinstance(model, (str, Path)) else None,
                "anomaly_model": str(anomaly_model) if anomaly_model is not None else None,
            },
            dataset_ids=[spec.dataset_id for spec in specs],
        )
        summary = {
            "n_datasets": len(specs),
            "n_decisions_total": sum(len(v) for v in decisions_by_dataset.values()),
            "n_decisions_per_dataset": {
                k: len(v) for k, v in decisions_by_dataset.items()
            },
        }
        write_multifile_outputs(
            out_dir=write_dir,
            summary=summary,
            metadata=metadata,
            title="QA run",
            overwrite=overwrite,
        )
        output_dir = write_dir

    return QAResult(
        tracks_by_dataset=tracks_by_dataset,
        events_by_dataset=events_by_dataset,
        divisions_by_dataset=divisions_by_dataset,
        decisions_by_dataset=decisions_by_dataset,
        per_pair_features_by_dataset=per_pair_by_dataset,
        resolved_config={
            "qa": resolved_config.to_dict(),
            "tracker": resolved_tracker.to_dict(),
        },
        output_dir=output_dir,
    )


__all__ = ["QAResult", "run_qa"]
