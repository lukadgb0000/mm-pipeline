"""Training utilities for candidate plausibility scorers"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from mm_pipeline.features import resolve_feature_subset

from .model_registry import DEFAULT_MODEL_NAME, RawScoreKind, get_model_spec

NBPrior = Literal["empirical", "uniform_within_pair"] | float

PROB_EPS = 1e-9


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Scoring requires pandas. Install the package with the 'scoring' extra.") from exc
    return pd


def _require_sklearn_pipeline_bits():
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Discriminative scoring requires scikit-learn. Install the package with the 'scoring' extra."
        ) from exc
    return Pipeline, SimpleImputer, StandardScaler


def _require_sklearn_calibration_bits():
    try:
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError as exc:
        raise RuntimeError(
            "Calibrated scoring requires scikit-learn. Install the package with the 'scoring' extra."
        ) from exc
    return CalibratedClassifierCV


def _logit(values: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    clipped = np.clip(arr, PROB_EPS, 1.0 - PROB_EPS)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.empty_like(arr, dtype=float)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_vals = np.exp(arr[~pos])
    out[~pos] = exp_vals / (1.0 + exp_vals)
    return out


def _positive_class_index(estimator: Any) -> int:
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        return 1
    classes_arr = np.asarray(classes)
    for label in (1, True):
        locs = np.flatnonzero(classes_arr == label)
        if len(locs) == 1:
            return int(locs[0])
    if len(classes_arr) < 2:
        raise ValueError("Estimator has fewer than two classes.")
    return 1


def _decision_scores(estimator: Any, X: Any) -> np.ndarray:
    if not hasattr(estimator, "decision_function"):
        raise TypeError("Estimator does not provide decision_function.")
    raw = estimator.decision_function(X)
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, _positive_class_index(estimator)]
    raise ValueError(f"Unsupported decision_function output shape: {arr.shape}")


def _probability_scores(estimator: Any, X: Any) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("Estimator does not provide predict_proba.")
    proba = estimator.predict_proba(X)
    arr = np.asarray(proba, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Unsupported predict_proba output shape: {arr.shape}")
    return arr[:, _positive_class_index(estimator)]


def _validate_nb_prior(nb_prior: NBPrior) -> NBPrior:
    if isinstance(nb_prior, str):
        if nb_prior not in {"empirical", "uniform_within_pair"}:
            raise ValueError("nb_prior must be 'empirical', 'uniform_within_pair', or a float in (0, 1).")
        return nb_prior
    value = float(nb_prior)
    if not 0.0 < value < 1.0:
        raise ValueError("Explicit nb_prior must be in (0, 1).")
    return value


def _make_sklearn_pipeline(estimator: Any):
    Pipeline, SimpleImputer, StandardScaler = _require_sklearn_pipeline_bits()
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", estimator),
        ]
    )


def _resolve_actual_raw_kind(estimator: Any, requested: RawScoreKind) -> RawScoreKind:
    if requested == "decision_or_logit_from_probability":
        return "decision" if hasattr(estimator, "decision_function") else "logit_from_probability"
    return requested


def _validate_feature_columns(df: Any, feature_cols: list[str]) -> None:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")


def _training_data(feature_table: Any, feature_cols: list[str], target_col: str):
    pd = _require_pandas()
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError("feature_table must be a pandas DataFrame.")
    if feature_table.empty:
        raise ValueError("feature_table is empty.")
    if target_col not in feature_table.columns:
        raise KeyError(f"Missing target column '{target_col}'.")
    _validate_feature_columns(feature_table, feature_cols)

    labelled = feature_table[feature_table[target_col].notna()].copy()
    if labelled.empty:
        raise ValueError(f"Target column '{target_col}' has no labelled rows.")
    y = labelled[target_col].astype(bool).to_numpy()
    if np.unique(y.astype(int)).size < 2:
        raise ValueError("Training data must contain both correct and incorrect candidates.")
    return labelled, y


def _make_calibrated_classifier(estimator: Any, *, method: str, cv: int):
    CalibratedClassifierCV = _require_sklearn_calibration_bits()
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


def _validate_calibration_request(y: np.ndarray, calibration_cv: int) -> None:
    if calibration_cv < 2:
        raise ValueError("calibration_cv must be >= 2.")
    counts = np.bincount(y.astype(int), minlength=2)
    if int(np.min(counts)) < int(calibration_cv):
        raise ValueError(
            "Calibration requires at least calibration_cv examples per class; "
            f"got class counts {counts.tolist()} and calibration_cv={calibration_cv}."
        )


@dataclass
class FittedScorer:
    """Opaque fitted phase-8 scorer.

    ``raw_score`` is model-specific ranking evidence. ``raw_score_kind`` records
    whether that evidence is a logit, generic decision margin, probability-logit
    transform, or Naive Bayes log-likelihood ratio.
    """

    model_name: str
    scorer_type: str
    feature_cols: tuple[str, ...]
    feature_subset: str | tuple[str, ...]
    raw_score_kind: RawScoreKind
    estimator: Any
    probability_estimator: Any = None
    is_calibrated: bool = False
    calibration_method: str | None = None
    calibration_cv: int | None = None
    nb_prior: NBPrior | None = None
    empirical_prior: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def _features(self, feature_table: Any) -> Any:
        _validate_feature_columns(feature_table, list(self.feature_cols))
        return feature_table.loc[:, list(self.feature_cols)]

    def raw_scores(self, feature_table: Any) -> np.ndarray:
        """Return model-specific ranking scores; higher means more plausible."""

        X = self._features(feature_table)
        if self.scorer_type == "naive_bayes":
            return np.asarray(self.estimator.score(X), dtype=float)

        if self.raw_score_kind in {"logit", "decision"}:
            return _decision_scores(self.estimator, X)
        if self.raw_score_kind == "logit_from_probability":
            return _logit(_probability_scores(self.estimator, X))
        # `decision_or_logit_from_probability` is a registry-level placeholder
        # resolved by `_resolve_actual_raw_kind` at fit time; it never reaches
        # this dispatch on a stored FittedScorer.
        raise ValueError(f"Unsupported raw_score_kind '{self.raw_score_kind}'.")

    def candidate_probabilities(
        self,
        feature_table: Any,
        *,
        raw_scores: np.ndarray | None = None,
        pair_col: str = "pair_id",
    ) -> np.ndarray:
        """Return row-level correctness probabilities where available."""

        X = self._features(feature_table)
        if self.scorer_type == "naive_bayes":
            raw = self.raw_scores(feature_table) if raw_scores is None else np.asarray(raw_scores, dtype=float)
            prior = self._nb_prior_vector(feature_table, pair_col=pair_col)
            return _sigmoid(raw + _logit(prior))

        proba_estimator = self.probability_estimator or self.estimator
        # sklearn's SVC exposes predict_proba via @available_if, so hasattr
        # correctly returns False when probability=False. LinearSVC has no
        # predict_proba at all. Both fall through to the NaN branch.
        if hasattr(proba_estimator, "predict_proba"):
            return _probability_scores(proba_estimator, X)
        return np.full(len(feature_table), np.nan, dtype=float)

    def feature_contributions(self, feature_table: Any) -> Any:
        """Return Naive Bayes per-feature contributions.

        Discriminative models do not expose a common contribution API here.
        """

        if self.scorer_type != "naive_bayes":
            raise ValueError("feature_contributions is only available for Naive Bayes scorers.")
        return self.estimator.feature_contributions(self._features(feature_table))

    def _nb_prior_vector(self, feature_table: Any, *, pair_col: str) -> np.ndarray:
        prior_setting = self.nb_prior
        if prior_setting == "uniform_within_pair":
            if pair_col not in feature_table.columns:
                raise KeyError(f"Missing pair column '{pair_col}' for nb_prior='uniform_within_pair'.")
            return feature_table.groupby(pair_col, sort=False)[pair_col].transform(lambda s: 1.0 / len(s)).to_numpy(
                dtype=float
            )
        if prior_setting == "empirical" or prior_setting is None:
            prior = self.empirical_prior
            if prior is None:
                raise ValueError("Naive Bayes empirical prior is not available.")
            return np.full(len(feature_table), float(prior), dtype=float)
        return np.full(len(feature_table), float(prior_setting), dtype=float)


def fit_scorer(
    feature_table: Any,
    model_name: str = DEFAULT_MODEL_NAME,
    feature_subset: str | list[str] | tuple[str, ...] = "all_features",
    target_col: str = "is_correct",
    calibrate: bool = False,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 3,
    nb_prior: NBPrior = "empirical",
) -> FittedScorer:
    """Fit a candidate plausibility scorer from a labelled feature table."""

    spec = get_model_spec(model_name)
    feature_cols = resolve_feature_subset(feature_subset)
    train_df, y = _training_data(feature_table, feature_cols, target_col)
    X = train_df.loc[:, feature_cols]

    if spec.scorer_type == "naive_bayes":
        from .naive_bayes import NaiveBayesScorer

        prior = _validate_nb_prior(nb_prior)
        if spec.nb_mode is None:
            raise ValueError(f"Naive Bayes model '{model_name}' is missing nb_mode.")
        scorer = NaiveBayesScorer(mode=spec.nb_mode, feature_cols=feature_cols)
        scorer.fit(X, y)
        return FittedScorer(
            model_name=spec.name,
            scorer_type=spec.scorer_type,
            feature_cols=tuple(feature_cols),
            feature_subset=feature_subset if isinstance(feature_subset, str) else tuple(feature_cols),
            raw_score_kind="llr",
            estimator=scorer,
            is_calibrated=False,
            nb_prior=prior,
            empirical_prior=float(scorer.empirical_prior),
            metadata={
                "n_train": int(len(train_df)),
                "target_col": target_col,
            },
        )

    if spec.estimator_factory is None:
        raise ValueError(f"Model '{model_name}' is missing an estimator factory.")

    base_pipeline = _make_sklearn_pipeline(spec.estimator_factory())
    base_pipeline.fit(X, y.astype(int))
    raw_kind = _resolve_actual_raw_kind(base_pipeline, spec.raw_score_kind)

    calibrated = None
    if calibrate:
        _validate_calibration_request(y, calibration_cv)
        calibrated_pipeline = _make_sklearn_pipeline(spec.estimator_factory())
        calibrated = _make_calibrated_classifier(
            calibrated_pipeline,
            method=calibration_method,
            cv=calibration_cv,
        )
        calibrated.fit(X, y.astype(int))

    return FittedScorer(
        model_name=spec.name,
        scorer_type=spec.scorer_type,
        feature_cols=tuple(feature_cols),
        feature_subset=feature_subset if isinstance(feature_subset, str) else tuple(feature_cols),
        raw_score_kind=raw_kind,
        estimator=base_pipeline,
        probability_estimator=calibrated,
        is_calibrated=bool(calibrate),
        calibration_method=calibration_method if calibrate else None,
        calibration_cv=int(calibration_cv) if calibrate else None,
        nb_prior=None,
        empirical_prior=None,
        metadata={
            "n_train": int(len(train_df)),
            "target_col": target_col,
        },
    )


def fit_lodo_scorers(
    feature_table: Any,
    model_name: str = DEFAULT_MODEL_NAME,
    feature_subset: str | list[str] | tuple[str, ...] = "all_features",
    heldout_col: str = "dataset_id",
    **fit_options: Any,
) -> dict[Any, FittedScorer]:
    """Fit one scorer per held-out value, usually one per dataset."""

    pd = _require_pandas()
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError("feature_table must be a pandas DataFrame.")
    if heldout_col not in feature_table.columns:
        raise KeyError(f"Missing heldout column '{heldout_col}'.")
    scorers: dict[Any, FittedScorer] = {}
    for heldout in pd.unique(feature_table[heldout_col]):
        train_df = feature_table[feature_table[heldout_col] != heldout]
        scorers[heldout] = fit_scorer(
            train_df,
            model_name=model_name,
            feature_subset=feature_subset,
            **fit_options,
        )
    return scorers


def fitted_scorer_metadata(scorer: FittedScorer) -> Mapping[str, Any]:
    """Return serialisable metadata for a fitted scorer."""

    return {
        "model_name": scorer.model_name,
        "scorer_type": scorer.scorer_type,
        "feature_cols": list(scorer.feature_cols),
        "feature_subset": scorer.feature_subset,
        "raw_score_kind": scorer.raw_score_kind,
        "is_calibrated": scorer.is_calibrated,
        "calibration_method": scorer.calibration_method,
        "calibration_cv": scorer.calibration_cv,
        "nb_prior": scorer.nb_prior,
        "empirical_prior": scorer.empirical_prior,
        "metadata": dict(scorer.metadata),
    }
