"""Per-pair model-violation anomaly detection.

Experimental and off by default: the only working detector is the no-op
``NeverAnomalous``; ``HistGBMModelViolationDetector`` needs a trained ``.joblib``
(not bundled). Formerly named "physical error".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from .aggregation import PAIR_ID_COLS, per_pair_feature_columns


def _numeric_feature_columns() -> list[str]:
    """Per-pair feature columns excluding identifier columns."""

    return [c for c in per_pair_feature_columns() if c not in PAIR_ID_COLS]


DEFAULT_MODEL_FILENAME = "model_violation_default.joblib"


class ModelViolationDetector(Protocol):
    name: str

    def detect(self, per_pair_features: Any) -> Any: ...
    """Return a DataFrame with columns ``anomaly_score`` and ``anomaly_flag``."""


class NeverAnomalous:
    """No-op detector. Returns ``anomaly_flag=False`` for every pair."""

    name = "never_anomalous"

    def detect(self, per_pair_features: Any) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("NeverAnomalous requires pandas.") from exc
        out = pd.DataFrame(
            {
                "pair_id": per_pair_features["pair_id"].astype(str).to_list()
                if "pair_id" in per_pair_features.columns
                else [],
                "anomaly_score": [float("nan")] * len(per_pair_features),
                "anomaly_flag": [False] * len(per_pair_features),
            }
        )
        return out


@dataclass
class HistGBMModelViolationDetector:
    """Hist-GBM detector with a recall-targeted decision threshold.

    ``estimator`` should be a fitted sklearn pipeline that exposes
    ``predict_proba`` (the second column is interpreted as P(has_error)).
    ``threshold`` is the calibrated decision threshold on that probability.
    """

    name: str = "hist_gbm"
    estimator: Any = None
    feature_cols: tuple[str, ...] = ()
    threshold: float = 0.5
    target_recall: float = 0.95
    training_summary: dict[str, Any] = field(default_factory=dict)

    def detect(self, per_pair_features: Any) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("HistGBMModelViolationDetector requires pandas.") from exc
        if self.estimator is None:
            raise RuntimeError("HistGBMModelViolationDetector has no fitted estimator.")

        feature_cols = list(self.feature_cols) or _numeric_feature_columns()
        feature_cols = [c for c in feature_cols if c in per_pair_features.columns]
        if not feature_cols:
            raise KeyError("No expected feature columns present on per_pair_features.")
        scores = _predict_score(self.estimator, per_pair_features[feature_cols])
        flags = scores >= float(self.threshold)
        return pd.DataFrame(
            {
                "pair_id": per_pair_features["pair_id"].astype(str).to_list(),
                "anomaly_score": [float(s) for s in scores],
                "anomaly_flag": [bool(f) for f in flags],
            }
        )

    def with_threshold(self, threshold: float) -> "HistGBMModelViolationDetector":
        return HistGBMModelViolationDetector(
            name=self.name,
            estimator=self.estimator,
            feature_cols=self.feature_cols,
            threshold=float(threshold),
            target_recall=self.target_recall,
            training_summary=dict(self.training_summary),
        )

    @classmethod
    def default(cls) -> "HistGBMModelViolationDetector":
        """Load the package-bundled default-trained detector."""

        model_path = _default_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Default model-violation detector model not found at {model_path}. "
                "Train one with scripts/train_model_violation_detector.py."
            )
        return load_detector(model_path)


def _predict_score(estimator: Any, X: Any) -> Any:
    """Return P(has_error=1) per row from a fitted sklearn pipeline."""

    import numpy as np

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        arr = np.asarray(proba, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Unsupported predict_proba shape {arr.shape}.")
        classes = getattr(estimator, "classes_", np.array([0, 1]))
        positive_loc = int(np.flatnonzero(np.asarray(classes) == 1)[0]) if 1 in np.asarray(classes) else 1
        return arr[:, positive_loc]
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=float)
    raise TypeError("Estimator must expose predict_proba or decision_function.")


def _default_model_path() -> Path:
    from importlib import resources

    pkg_root = resources.files("mm_pipeline.modelvio")
    return Path(str(pkg_root.joinpath("_models", DEFAULT_MODEL_FILENAME)))


def save_detector(detector: HistGBMModelViolationDetector, path: str | Path) -> Path:
    """Persist a fitted detector with joblib."""

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Persistence requires joblib.") from exc
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, out_path)
    return out_path


def load_detector(path: str | Path) -> HistGBMModelViolationDetector:
    """Load a detector saved by :func:`save_detector`."""

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Persistence requires joblib.") from exc
    obj = joblib.load(Path(path))
    if not isinstance(obj, HistGBMModelViolationDetector):
        raise TypeError(f"Expected HistGBMModelViolationDetector, got {type(obj).__name__}.")
    return obj


def _make_pipeline():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    random_state=42,
                ),
            ),
        ]
    )


def train_detector(
    per_pair_features: Any,
    has_error_col: str = "has_error",
    *,
    feature_cols: Optional[list[str]] = None,
    target_recall: float = 0.95,
    cv_groups: Any = None,
) -> HistGBMModelViolationDetector:
    """Train a hist-GBM detector with a recall-targeted threshold.

    ``per_pair_features`` is the DataFrame produced by
    :func:`build_per_pair_features` extended with a binary ``has_error`` column.
    When ``cv_groups`` is provided, the threshold is calibrated via grouped CV
    (one fold per unique group). Otherwise the threshold is calibrated on the
    training set (with the usual caveat that this is optimistic).
    """

    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("train_detector requires pandas and numpy.") from exc

    if not isinstance(per_pair_features, pd.DataFrame):
        raise TypeError("per_pair_features must be a pandas DataFrame.")
    if has_error_col not in per_pair_features.columns:
        raise KeyError(f"Missing target column '{has_error_col}'.")

    available = [c for c in _numeric_feature_columns() if c in per_pair_features.columns]
    feature_cols = list(feature_cols) if feature_cols else available
    feature_cols = [c for c in feature_cols if c != has_error_col]
    if not feature_cols:
        raise ValueError("No feature columns available for training.")

    X = per_pair_features[feature_cols]
    y = per_pair_features[has_error_col].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Training data must contain both classes (has_error in {0, 1}).")

    estimator = _make_pipeline()
    estimator.fit(X, y)

    if cv_groups is not None:
        threshold, cv_summary = _threshold_via_grouped_cv(
            per_pair_features, feature_cols, has_error_col, cv_groups, target_recall
        )
    else:
        scores = _predict_score(estimator, X)
        threshold, cv_summary = _threshold_for_target_recall(y, scores, target_recall)
        cv_summary["calibration_data"] = "training_set"

    return HistGBMModelViolationDetector(
        estimator=estimator,
        feature_cols=tuple(feature_cols),
        threshold=float(threshold),
        target_recall=float(target_recall),
        training_summary={
            "n_rows": int(len(per_pair_features)),
            "n_features": len(feature_cols),
            "feature_cols": list(feature_cols),
            **cv_summary,
        },
    )


def _threshold_for_target_recall(y_true, scores, target_recall: float) -> tuple[float, dict[str, Any]]:
    """Lowest decision threshold whose recall on (y_true, scores) ≥ target."""

    import numpy as np
    from sklearn.metrics import precision_recall_curve

    prec, rec, thresholds = precision_recall_curve(y_true, scores)
    mask = rec >= target_recall
    if not bool(mask.any()):
        return 0.0, {"calibration": "fallback_zero", "achieved_recall": float(rec.max())}

    idx_candidates = np.where(mask)[0]
    idx = int(idx_candidates[-1])
    if idx >= len(thresholds):
        threshold = 0.0
    else:
        threshold = float(thresholds[idx])
    return threshold, {
        "calibration_target_recall": float(target_recall),
        "calibration_precision_at_threshold": float(prec[idx]),
        "calibration_recall_at_threshold": float(rec[idx]),
        "calibration_threshold": float(threshold),
    }


def _threshold_via_grouped_cv(
    per_pair_features,
    feature_cols: list[str],
    has_error_col: str,
    cv_groups,
    target_recall: float,
) -> tuple[float, dict[str, Any]]:
    """Average the recall-targeted thresholds across leave-one-group-out folds."""

    import numpy as np
    import pandas as pd

    if not isinstance(cv_groups, pd.Series):
        cv_groups = pd.Series(cv_groups, index=per_pair_features.index)
    fold_thresholds: list[float] = []
    fold_summaries: list[dict[str, Any]] = []
    for held_out in cv_groups.unique():
        train_mask = cv_groups != held_out
        test_mask = cv_groups == held_out
        train = per_pair_features.loc[train_mask]
        test = per_pair_features.loc[test_mask]
        if train.empty or test.empty:
            continue
        y_train = train[has_error_col].astype(int).to_numpy()
        y_test = test[has_error_col].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        fold_estimator = _make_pipeline()
        fold_estimator.fit(train[feature_cols], y_train)
        test_scores = _predict_score(fold_estimator, test[feature_cols])
        thr, summary = _threshold_for_target_recall(y_test, test_scores, target_recall)
        fold_thresholds.append(thr)
        fold_summaries.append({"held_out": str(held_out), **summary})

    if not fold_thresholds:
        return 0.5, {"calibration": "no_folds_usable"}
    median_threshold = float(np.median(fold_thresholds))
    return median_threshold, {
        "calibration": "grouped_cv",
        "n_folds": len(fold_thresholds),
        "fold_thresholds": fold_thresholds,
        "fold_summaries": fold_summaries,
        "median_threshold": median_threshold,
    }


def build_detector(name: str) -> ModelViolationDetector:
    """Resolve a config-string name to a concrete detector instance."""

    if name == "never_anomalous":
        return NeverAnomalous()
    if name == "hist_gbm_default":
        return HistGBMModelViolationDetector.default()
    # Otherwise assume a filesystem path to a saved artefact.
    return load_detector(name)
