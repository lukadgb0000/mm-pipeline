"""Naive Bayes candidate plausibility scorer"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from mm_pipeline.features import FEATURE_COLUMNS

FEATURE_DISTRIBUTIONS: dict[str, str] = {
    "max_shrink_pct": "gamma",
    "total_area_ratio_exit_adjusted": "lognorm",
    "exit_open_end_dist_median_norm": "gamma",
    "link_area_ratio_median": "lognorm",
    "link_area_ratio_max": "lognorm",
    "link_dy_median_norm": "gamma",
    "link_dy_max_norm": "gamma",
    "link_iou_shifted_median": "beta",
    "div_mother_sum_area_ratio_max": "lognorm",
    "div_mother_sum_area_ratio_mean": "lognorm",
    "div_daughter_area_ratio_max": "lognorm",
    "div_daughter_area_ratio_mean": "lognorm",
    "div_mother_daughter_dy_max_norm": "gamma",
    "div_mother_daughter_dy_mean_norm": "gamma",
}

LOGPDF_FLOOR = -30.0
EPS = 1e-9
MIN_OBS_PROB = 0.01


def _require_scipy_stats():
    try:
        from scipy import stats
    except ImportError as exc:
        raise RuntimeError("Naive Bayes scoring requires scipy. Install the package with the 'scoring' extra.") from exc
    return stats


def _fit_parametric(family: str, values: np.ndarray) -> tuple[str, tuple[Any, ...]] | None:
    stats = _require_scipy_stats()
    vals = values[np.isfinite(values)]
    if len(vals) < 3:
        return None
    try:
        if family == "gamma":
            return "gamma", stats.gamma.fit(np.maximum(vals, EPS), floc=0)
        if family == "lognorm":
            return "lognorm", stats.lognorm.fit(np.maximum(vals, EPS), floc=0)
        if family == "beta":
            return "beta", stats.beta.fit(np.clip(vals, EPS, 1.0 - EPS), floc=0, fscale=1)
    except Exception:
        return None
    raise ValueError(f"Unknown distribution family '{family}'.")


def _logpdf_parametric(family: str, params: tuple[Any, ...], values: np.ndarray) -> np.ndarray:
    stats = _require_scipy_stats()
    if family == "gamma":
        vals = np.maximum(values, EPS)
        out = stats.gamma.logpdf(vals, *params)
    elif family == "lognorm":
        vals = np.maximum(values, EPS)
        out = stats.lognorm.logpdf(vals, *params)
    elif family == "beta":
        vals = np.clip(values, EPS, 1.0 - EPS)
        out = stats.beta.logpdf(vals, *params)
    else:
        raise ValueError(f"Unknown distribution family '{family}'.")
    arr = np.asarray(out, dtype=float)
    arr[~np.isfinite(arr)] = LOGPDF_FLOOR
    return np.maximum(arr, LOGPDF_FLOOR)


def _fit_kde(values: np.ndarray) -> Any | None:
    stats = _require_scipy_stats()
    vals = values[np.isfinite(values)]
    if len(vals) < 3 or np.ptp(vals) < EPS:
        return None
    try:
        return stats.gaussian_kde(vals)
    except Exception:
        return None


def _logpdf_kde(kde: Any, values: np.ndarray) -> np.ndarray:
    out = np.asarray(kde.logpdf(values), dtype=float)
    out[~np.isfinite(out)] = LOGPDF_FLOOR
    return np.maximum(out, LOGPDF_FLOOR)


@dataclass
class _FeatureFit:
    obs_prob: float
    family: str | None = None
    params: tuple[Any, ...] | None = None
    kde: Any | None = None


class NaiveBayesScorer:
    """Feature-wise log-likelihood-ratio scorer

    The raw score is:

    ``sum_j log p(x_j | correct) - log p(x_j | incorrect)``

    Missing values are modelled directly through class-conditional observation
    probabilities rather than imputed away - see my report, it's interesting I'd say if you like probabilistic modelling :)
    """

    def __init__(
        self,
        mode: Literal["parametric", "kde"] = "parametric",
        feature_cols: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if mode not in {"parametric", "kde"}:
            raise ValueError("mode must be 'parametric' or 'kde'.")
        self.mode = mode
        self.feature_cols = list(feature_cols or FEATURE_COLUMNS)
        self.empirical_prior: float = float("nan")
        self._fits: dict[str, dict[bool, _FeatureFit]] = {}
        self._is_fitted = False

    def fit(self, X: Any, y: np.ndarray) -> "NaiveBayesScorer":
        y_bool = np.asarray(y).astype(bool)
        if len(X) != len(y_bool):
            raise ValueError("X and y must have equal length.")
        if np.unique(y_bool.astype(int)).size < 2:
            raise ValueError("Naive Bayes training requires both classes.")

        self.empirical_prior = float(np.clip(np.mean(y_bool.astype(float)), EPS, 1.0 - EPS))
        self._fits = {}
        for feature in self.feature_cols:
            if feature not in X.columns:
                raise KeyError(f"Missing feature column '{feature}'.")
            values = X[feature].to_numpy(dtype=float)
            feature_fits: dict[bool, _FeatureFit] = {}
            for cls in (True, False):
                cls_values = values[y_bool == cls]
                n_cls = len(cls_values)
                n_obs = int(np.sum(np.isfinite(cls_values)))
                obs_prob = max(n_obs / max(n_cls, 1), MIN_OBS_PROB)
                observed = cls_values[np.isfinite(cls_values)]

                if self.mode == "parametric":
                    family = FEATURE_DISTRIBUTIONS.get(feature, "gamma")
                    fitted = _fit_parametric(family, observed)
                    if fitted is not None:
                        feature_fits[cls] = _FeatureFit(
                            obs_prob=obs_prob,
                            family=fitted[0],
                            params=fitted[1],
                        )
                        continue

                feature_fits[cls] = _FeatureFit(obs_prob=obs_prob, kde=_fit_kde(observed))
            self._fits[feature] = feature_fits

        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before scoring.")

    def _log_likelihood_feature(self, feature: str, values: np.ndarray, cls: bool) -> np.ndarray:
        fit = self._fits[feature].get(cls)
        out = np.full(len(values), LOGPDF_FLOOR, dtype=float)
        if fit is None:
            return out

        observed = np.isfinite(values)
        out[~observed] = np.log(max(1.0 - fit.obs_prob, MIN_OBS_PROB))
        if not np.any(observed):
            return out

        vals_obs = values[observed]
        if fit.family is not None and fit.params is not None:
            logp = _logpdf_parametric(fit.family, fit.params, vals_obs)
        elif fit.kde is not None:
            logp = _logpdf_kde(fit.kde, vals_obs)
        else:
            logp = np.full(len(vals_obs), LOGPDF_FLOOR, dtype=float)
        out[np.flatnonzero(observed)] = np.log(max(fit.obs_prob, MIN_OBS_PROB)) + logp
        return out

    def score(self, X: Any) -> np.ndarray:
        """Return log-likelihood-ratio raw scores"""

        self._check_fitted()
        out = np.zeros(len(X), dtype=float)
        for feature in self.feature_cols:
            if feature not in X.columns:
                raise KeyError(f"Missing feature column '{feature}'.")
            values = X[feature].to_numpy(dtype=float)
            out += self._log_likelihood_feature(feature, values, True)
            out -= self._log_likelihood_feature(feature, values, False)
        return out

    def feature_contributions(self, X: Any) -> Any:
        """Return per-feature log-likelihood-ratio contributions."""

        self._check_fitted()
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("feature_contributions requires pandas.") from exc

        data = {}
        for feature in self.feature_cols:
            if feature not in X.columns:
                raise KeyError(f"Missing feature column '{feature}'.")
            values = X[feature].to_numpy(dtype=float)
            data[feature] = self._log_likelihood_feature(feature, values, True) - self._log_likelihood_feature(
                feature, values, False
            )
        return pd.DataFrame(data, index=X.index)

    def fitted_params(self) -> Any:
        """Return a tabular summary of fitted feature densities"""

        self._check_fitted()
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("fitted_params requires pandas.") from exc

        rows = []
        for feature in self.feature_cols:
            for class_name, cls in (("correct", True), ("incorrect", False)):
                fit = self._fits[feature].get(cls)
                rows.append(
                    {
                        "feature": feature,
                        "class": class_name,
                        "obs_prob": np.nan if fit is None else fit.obs_prob,
                        "family": None if fit is None else fit.family,
                        "params": None if fit is None or fit.params is None else str(fit.params),
                        "kde_fitted": False if fit is None else fit.kde is not None,
                    }
                )
        return pd.DataFrame(rows)
