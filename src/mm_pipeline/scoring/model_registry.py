"""Scoring model registry
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

ScorerType = Literal["sklearn", "naive_bayes"]
RawScoreKind = Literal["logit", "decision", "logit_from_probability", "llr", "decision_or_logit_from_probability"]

DEFAULT_MODEL_NAME = "logreg_l2_balanced"


@dataclass(frozen=True)
class ModelSpec:
    """Description of an available candidate scorer"""

    name: str
    scorer_type: ScorerType
    raw_score_kind: RawScoreKind
    estimator_factory: Callable[[], Any] | None = None
    nb_mode: Literal["parametric", "kde"] | None = None
    description: str = ""


def _missing_sklearn_error() -> RuntimeError:
    return RuntimeError(
        "Scikit-learn is required for discriminative scoring models. "
        "Install the package with the 'scoring' extra."
    )


def _logreg_l2():
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise _missing_sklearn_error() from exc
    return LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        max_iter=4000,
        solver="lbfgs",
    )


def _logreg_l1():
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise _missing_sklearn_error() from exc
    return LogisticRegression(
        penalty="l1",
        class_weight="balanced",
        max_iter=4000,
        solver="liblinear",
    )


def _linear_svm():
    try:
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise _missing_sklearn_error() from exc
    return LinearSVC(class_weight="balanced")


def _random_forest():
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise _missing_sklearn_error() from exc
    return RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )


def _hist_gbm():
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:
        raise _missing_sklearn_error() from exc
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )


def _rbf_svm():
    try:
        from sklearn.svm import SVC
    except ImportError as exc:
        raise _missing_sklearn_error() from exc
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=False,
    )


def get_model_registry() -> dict[str, ModelSpec]:
    """Return the supported phase-8 scoring models."""

    specs = (
        ModelSpec(
            name="logreg_l2_balanced",
            scorer_type="sklearn",
            raw_score_kind="logit",
            estimator_factory=_logreg_l2,
            description="Logistic regression (L2, balanced classes).",
        ),
        ModelSpec(
            name="logreg_l1_balanced",
            scorer_type="sklearn",
            raw_score_kind="logit",
            estimator_factory=_logreg_l1,
            description="Logistic regression (L1, balanced classes).",
        ),
        ModelSpec(
            name="linear_svm_balanced",
            scorer_type="sklearn",
            raw_score_kind="decision",
            estimator_factory=_linear_svm,
            description="Linear SVM with balanced class weights.",
        ),
        ModelSpec(
            name="random_forest_balanced",
            scorer_type="sklearn",
            raw_score_kind="logit_from_probability",
            estimator_factory=_random_forest,
            description="Random forest baseline (balanced_subsample).",
        ),
        ModelSpec(
            name="hist_gbm",
            scorer_type="sklearn",
            raw_score_kind="decision_or_logit_from_probability",
            estimator_factory=_hist_gbm,
            description="Histogram gradient boosting baseline.",
        ),
        ModelSpec(
            name="rbf_svm_balanced",
            scorer_type="sklearn",
            raw_score_kind="decision",
            estimator_factory=_rbf_svm,
            description="RBF-kernel SVM baseline.",
        ),
        ModelSpec(
            name="naive_bayes_parametric",
            scorer_type="naive_bayes",
            raw_score_kind="llr",
            nb_mode="parametric",
            description="Naive Bayes scorer with parametric feature densities.",
        ),
        ModelSpec(
            name="naive_bayes_kde",
            scorer_type="naive_bayes",
            raw_score_kind="llr",
            nb_mode="kde",
            description="Naive Bayes scorer with kernel-density feature densities.",
        ),
    )
    return {spec.name: spec for spec in specs}


def get_model_spec(model_name: str = DEFAULT_MODEL_NAME) -> ModelSpec:


    registry = get_model_registry()
    if model_name not in registry:
        raise KeyError(f"Unknown model '{model_name}'. Available models: {sorted(registry)}")
    return registry[model_name]


def list_models() -> list[str]:
    

    return sorted(get_model_registry())
