"""Runner for mm-pipeline score

Applies a trained scorer (a ``FittedScorer`` joblib) to a featurised
candidates parquet, appending the score columns and writing a scored parquet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mm_pipeline.scoring import FittedScorer, load_scorer, score_candidates

from ._outputs import make_run_metadata, write_single_artefact_outputs


@dataclass(frozen=True)
class ScoreResult:
    """In-memory result of a ``run_score`` invocation."""

    scored_df: Any
    resolved_config: dict[str, Any] = field(default_factory=dict)
    output_path: Path | None = None


def _load_features(features: Any) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("run_score requires pandas.") from exc
    if isinstance(features, (str, Path)):
        return pd.read_parquet(features)
    if isinstance(features, pd.DataFrame):
        return features
    raise TypeError(
        f"features must be a DataFrame or path to a parquet; "
        f"got {type(features).__name__}."
    )


def _load_model(model: Any) -> FittedScorer:
    if isinstance(model, FittedScorer):
        return model
    if isinstance(model, (str, Path)):
        return load_scorer(model)
    raise TypeError(
        f"model must be a FittedScorer or path to a joblib; "
        f"got {type(model).__name__}."
    )


def run_score(
    features: Any,
    *,
    model: Any,
    pair_temperature: float = 1.0,
    pair_col: str = "pair_id",
    out_path: str | Path | None = None,
    overwrite: bool = False,
) -> ScoreResult:
    """Apply a trained scorer to a featurised candidates table.

    Parameters
    ----------
    features : either a features DataFrame (output of ``run_featurise``) or
        a path to a parquet file containing one.
    model : either a FittedScorer instance or a path to a joblib produced
        by ``mm_pipeline.scoring.persistence.save_scorer``.
    pair_temperature : temperature for the within-pair softmax (default 1.0).
    pair_col : column name identifying pairs (default ``pair_id``).
    out_path : optional scored-parquet path. When None, no files are
        written and ``result.output_path`` is None.
    overwrite : if True, allow clobbering an existing artefact path.
    """

    features_df = _load_features(features)
    fitted_scorer = _load_model(model)

    scored_df = score_candidates(
        features_df,
        fitted_scorer,
        pair_col=pair_col,
        pair_temperature=pair_temperature,
    )

    model_path = str(model) if isinstance(model, (str, Path)) else None
    resolved_config = {
        "score": {
            "model_path": model_path,
            "model_name": fitted_scorer.model_name,
            "feature_subset": (
                fitted_scorer.feature_subset
                if isinstance(fitted_scorer.feature_subset, str)
                else list(fitted_scorer.feature_subset)
            ),
            "pair_temperature": float(pair_temperature),
            "pair_col": pair_col,
        },
    }

    written_path: Path | None = None
    if out_path is not None:
        metadata = make_run_metadata(
            command="score",
            manifest_path=None,
            resolved_config=resolved_config,
            dataset_ids=sorted(
                str(x) for x in scored_df.get("dataset_id", []).unique()
            ) if "dataset_id" in scored_df.columns else [],
        )
        summary = {
            "n_rows": int(len(scored_df)),
            "n_pairs": int(scored_df[pair_col].nunique()) if pair_col in scored_df.columns else 0,
            "features_source": str(features) if isinstance(features, (str, Path)) else None,
        }
        paths = write_single_artefact_outputs(
            out_path=out_path,
            artefact=scored_df,
            metadata=metadata,
            summary=summary,
            overwrite=overwrite,
        )
        written_path = paths["artefact"]

    return ScoreResult(
        scored_df=scored_df,
        resolved_config=resolved_config,
        output_path=written_path,
    )


__all__ = ["ScoreResult", "run_score"]
