"""Runner for ``mm-pipeline train-scorer``.

Trains a candidate-plausibility scorer from a labelled features parquet
(``is_correct`` column required). Optionally performs LODO CV eval
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mm_pipeline.scoring import (
    DEFAULT_MODEL_NAME,
    FittedScorer,
    fit_lodo_scorers,
    fit_scorer,
    fitted_scorer_metadata,
    save_scorer,
)

from ._outputs import make_run_metadata, write_single_artefact_outputs

CVStrategy = Literal["none", "leave_one_dataset_out"]


@dataclass(frozen=True)
class TrainScorerResult:
    """In-memory result of a ``run_train_scorer`` invocation."""

    fitted_scorer: FittedScorer
    cv_metrics: Any | None = None
    resolved_config: dict[str, Any] = field(default_factory=dict)
    output_path: Path | None = None


def _load_features(features: Any) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("run_train_scorer requires pandas.") from exc
    if isinstance(features, (str, Path)):
        return pd.read_parquet(features)
    if isinstance(features, pd.DataFrame):
        return features
    raise TypeError(
        f"features must be a DataFrame or path to a parquet; "
        f"got {type(features).__name__}."
    )


def _per_fold_metrics(scorers: dict[Any, FittedScorer]) -> Any:
    """Build a per-fold DataFrame from a dict of LODO scorers."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("run_train_scorer requires pandas.") from exc
    rows = []
    for heldout, scorer in scorers.items():
        meta = fitted_scorer_metadata(scorer)
        row = {"heldout": str(heldout), "model_name": scorer.model_name}
        for key, value in dict(meta).items():
            row[str(key)] = value
        rows.append(row)
    return pd.DataFrame(rows)


def run_train_scorer(
    features: Any,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    feature_subset: str | list[str] | tuple[str, ...] = "all_features",
    target_col: str = "is_correct",
    calibrate: bool = False,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 3,
    cv: CVStrategy = "none",
    heldout_col: str = "dataset_id",
    out_path: str | Path | None = None,
    overwrite: bool = False,
) -> TrainScorerResult:
    """Train a candidate-plausibility scorer.

    Always fits a final scorer on all labelled rows. If ``cv`` is
    ``"leave_one_dataset_out"``, also fits one scorer per held-out
    value of ``heldout_col`` and records per-fold metadata.

    Parameters
    ----------
    features : either a features DataFrame with an ``is_correct`` column or
        a path to a parquet file containing one.
    model_name : registry key for the model class (default DEFAULT_MODEL_NAME).
    feature_subset : ``"all_features"`` or an explicit list of columns.
    target_col : binary target column (default ``"is_correct"``).
    calibrate : whether to fit a probability-calibrator alongside the
        base estimator (sigmoid or isotonic).
    cv : ``"none"`` or ``"leave_one_dataset_out"``.
    heldout_col : grouping column for LODO (default ``"dataset_id"``).
    out_path : optional joblib path. When None, no files are written and
        ``result.output_path`` is None.
    overwrite : if True, allow clobbering an existing artefact path.
    """

    features_df = _load_features(features)

    cv_metrics: Any | None = None
    if cv == "leave_one_dataset_out":
        lodo_scorers = fit_lodo_scorers(
            features_df,
            model_name=model_name,
            feature_subset=feature_subset,
            heldout_col=heldout_col,
            target_col=target_col,
            calibrate=calibrate,
            calibration_method=calibration_method,
            calibration_cv=calibration_cv,
        )
        cv_metrics = _per_fold_metrics(lodo_scorers)
    elif cv != "none":
        raise ValueError(f"Unknown cv strategy: {cv!r}. Supported: 'none', 'leave_one_dataset_out'.")

    fitted = fit_scorer(
        features_df,
        model_name=model_name,
        feature_subset=feature_subset,
        target_col=target_col,
        calibrate=calibrate,
        calibration_method=calibration_method,
        calibration_cv=calibration_cv,
    )

    resolved_config = {
        "train_scorer": {
            "model_name": model_name,
            "feature_subset": (
                feature_subset
                if isinstance(feature_subset, str)
                else list(feature_subset)
            ),
            "target_col": target_col,
            "calibrate": bool(calibrate),
            "calibration_method": calibration_method if calibrate else None,
            "calibration_cv": int(calibration_cv) if calibrate else None,
            "cv": cv,
            "heldout_col": heldout_col if cv != "none" else None,
        },
    }

    written_path: Path | None = None
    if out_path is not None:
        out = Path(out_path)
        # save_scorer writes the joblib; we then build the sibling run.json
        # by reusing write_single_artefact_outputs with the just-saved file.
        if out.exists() and not overwrite:
            raise FileExistsError(
                f"{out} already exists. Pass --overwrite or choose a different --out."
            )
        save_scorer(fitted, out)
        metadata = make_run_metadata(
            command="train-scorer",
            manifest_path=None,
            resolved_config=resolved_config,
            dataset_ids=sorted(
                str(x) for x in features_df.get(heldout_col, []).unique()
            ) if heldout_col in features_df.columns else [],
        )
        summary = {
            "n_train_rows": int(features_df[target_col].notna().sum()) if target_col in features_df.columns else 0,
            "n_features": len(fitted.feature_cols),
            "scorer_type": fitted.scorer_type,
            "is_calibrated": bool(fitted.is_calibrated),
            "features_source": str(features) if isinstance(features, (str, Path)) else None,
        }
        if cv_metrics is not None:
            summary["n_cv_folds"] = int(len(cv_metrics))
        # Write only the sibling .run.json — the joblib is already saved above.
        run_json = out.with_suffix(out.suffix + ".run.json")
        import json

        from mm_pipeline.io.run_outputs import _jsonable

        combined = {**dict(metadata), **dict(summary)}
        with run_json.open("w", encoding="utf-8") as fh:
            json.dump(_jsonable(combined), fh, indent=2, sort_keys=True)
            fh.write("\n")
        written_path = out

    return TrainScorerResult(
        fitted_scorer=fitted,
        cv_metrics=cv_metrics,
        resolved_config=resolved_config,
        output_path=written_path,
    )


__all__ = ["TrainScorerResult", "run_train_scorer"]
