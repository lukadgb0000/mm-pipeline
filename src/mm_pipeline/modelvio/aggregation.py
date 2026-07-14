"""Per-pair feature aggregation for the QA anomaly detector.

Aggregates the 14 candidate features (plus DP / classifier score summaries)
into one row per ``pair_id``. The output is consumed by
physical error qa module
"""

from __future__ import annotations

from typing import Any

from mm_pipeline.features import FEATURE_COLUMNS

CANDIDATE_AGG_FUNCS: tuple[str, ...] = ("max", "min", "mean", "std", "best")

PAIR_ID_COLS: tuple[str, ...] = ("dataset_id", "pair_id", "t", "n_candidates")


def candidate_feature_columns() -> list[str]:
    """Return the per-pair feature column order: ``{feature}_{agg}`` cross-product."""

    cols: list[str] = []
    for feature in FEATURE_COLUMNS:
        for agg in CANDIDATE_AGG_FUNCS:
            cols.append(f"{feature}_{agg}")
    return cols


def score_summary_columns() -> list[str]:
    return [
        "score_max", "score_min", "score_mean", "score_std",
        "score_entropy", "score_margin_top1_top2", "score_max_pair_probability",
    ]


def dp_summary_columns() -> list[str]:
    return ["dp_cost_min", "dp_cost_max", "dp_cost_mean", "dp_cost_range"]


def per_pair_feature_columns() -> list[str]:
    """Full ordered column list for the per-pair feature DataFrame."""

    return [
        *PAIR_ID_COLS,
        *candidate_feature_columns(),
        *score_summary_columns(),
        *dp_summary_columns(),
        "disagreement_score",
    ]


def _safe_std(values) -> float:
    import numpy as np
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.nanstd(arr, ddof=0))


def _best_row_idx(pair: Any) -> Any:
    """Index of the 'best' candidate row used for ``_best`` aggregations.

    Prefer ``raw_score`` argmax. Fall back to ``dp_cost`` argmin if scores are
    missing, then to the first row if all signals are missing.
    """

    if "raw_score" in pair.columns and pair["raw_score"].notna().any():
        return pair["raw_score"].astype(float).idxmax()
    if "dp_cost" in pair.columns and pair["dp_cost"].notna().any():
        return pair["dp_cost"].astype(float).idxmin()
    return pair.index[0]


def _entropy_of_pair_probability(pair: Any) -> float:
    import math
    import numpy as np

    if "pair_probability" not in pair.columns:
        return float("nan")
    probs = pair["pair_probability"].astype(float).to_numpy()
    probs = probs[np.isfinite(probs)]
    if probs.size == 0 or float(probs.sum()) == 0.0:
        return float("nan")
    probs = probs / float(probs.sum())
    nz = probs[probs > 0]
    return float(-np.sum(nz * np.log(nz)))


def _disagreement_score(pair: Any) -> float:
    import math

    if "dp_cost" not in pair.columns or "raw_score" not in pair.columns:
        return float("nan")
    if pair["dp_cost"].isna().all() or pair["raw_score"].isna().all():
        return float("nan")
    dp_top1 = pair["dp_cost"].astype(float).idxmin()
    cls_top1 = pair["raw_score"].astype(float).idxmax()
    dp_cost_dp_top1 = float(pair.at[dp_top1, "dp_cost"])
    dp_cost_cls_top1 = float(pair.at[cls_top1, "dp_cost"])
    raw_dp_top1 = float(pair.at[dp_top1, "raw_score"])
    raw_cls_top1 = float(pair.at[cls_top1, "raw_score"])
    dp_gap = max(dp_cost_cls_top1 - dp_cost_dp_top1, 0.0)
    cls_gap = max(raw_cls_top1 - raw_dp_top1, 0.0)
    dp_gap_norm = dp_gap / max(abs(dp_cost_dp_top1), 1e-9)
    return float(math.sqrt(dp_gap_norm * cls_gap))


def _score_summaries(pair: Any) -> dict[str, float]:
    import math
    import numpy as np

    out: dict[str, float] = {}
    if "raw_score" in pair.columns:
        scores = pair["raw_score"].astype(float).to_numpy()
        finite = scores[np.isfinite(scores)]
        if finite.size:
            sorted_desc = np.sort(finite)[::-1]
            out["score_max"] = float(sorted_desc[0])
            out["score_min"] = float(finite.min())
            out["score_mean"] = float(finite.mean())
            out["score_std"] = _safe_std(finite)
            out["score_margin_top1_top2"] = (
                float(sorted_desc[0] - sorted_desc[1]) if sorted_desc.size >= 2 else float("nan")
            )
        else:
            for k in ("score_max", "score_min", "score_mean", "score_std", "score_margin_top1_top2"):
                out[k] = float("nan")
    else:
        for k in ("score_max", "score_min", "score_mean", "score_std", "score_margin_top1_top2"):
            out[k] = float("nan")

    out["score_entropy"] = _entropy_of_pair_probability(pair)
    if "pair_probability" in pair.columns and pair["pair_probability"].notna().any():
        out["score_max_pair_probability"] = float(pair["pair_probability"].astype(float).max())
    else:
        out["score_max_pair_probability"] = float("nan")
    return out


def _dp_summaries(pair: Any) -> dict[str, float]:
    import numpy as np

    if "dp_cost" not in pair.columns:
        return {k: float("nan") for k in dp_summary_columns()}
    costs = pair["dp_cost"].astype(float).to_numpy()
    finite = costs[np.isfinite(costs)]
    if finite.size == 0:
        return {k: float("nan") for k in dp_summary_columns()}
    return {
        "dp_cost_min": float(finite.min()),
        "dp_cost_max": float(finite.max()),
        "dp_cost_mean": float(finite.mean()),
        "dp_cost_range": float(finite.max() - finite.min()),
    }


def _candidate_feature_aggs(pair: Any) -> dict[str, float]:
    import numpy as np

    out: dict[str, float] = {}
    best_idx = _best_row_idx(pair)
    for feature in FEATURE_COLUMNS:
        if feature not in pair.columns:
            for agg in CANDIDATE_AGG_FUNCS:
                out[f"{feature}_{agg}"] = float("nan")
            continue
        values = pair[feature].astype(float).to_numpy()
        finite = values[np.isfinite(values)]
        if finite.size:
            out[f"{feature}_max"] = float(finite.max())
            out[f"{feature}_min"] = float(finite.min())
            out[f"{feature}_mean"] = float(finite.mean())
            out[f"{feature}_std"] = _safe_std(finite)
        else:
            for agg in ("max", "min", "mean", "std"):
                out[f"{feature}_{agg}"] = float("nan")
        out[f"{feature}_best"] = float(pair.at[best_idx, feature])
    return out


def build_per_pair_features(scored_candidates: Any) -> Any:
    """One row per ``pair_id`` with aggregated features for anomaly detection."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("build_per_pair_features requires pandas.") from exc

    if not isinstance(scored_candidates, pd.DataFrame):
        raise TypeError("scored_candidates must be a pandas DataFrame.")
    if scored_candidates.empty:
        return pd.DataFrame(columns=per_pair_feature_columns())
    if "pair_id" not in scored_candidates.columns:
        raise KeyError("scored_candidates must include 'pair_id'.")

    rows: list[dict[str, Any]] = []
    for pair_id, pair in scored_candidates.groupby("pair_id", sort=False):
        row: dict[str, Any] = {
            "dataset_id": str(pair["dataset_id"].iloc[0]) if "dataset_id" in pair.columns else "",
            "pair_id": str(pair_id),
            "t": int(pair["t"].iloc[0]) if "t" in pair.columns else -1,
            "n_candidates": int(len(pair)),
        }
        row.update(_candidate_feature_aggs(pair))
        row.update(_score_summaries(pair))
        row.update(_dp_summaries(pair))
        row["disagreement_score"] = _disagreement_score(pair)
        rows.append(row)

    return pd.DataFrame(rows, columns=per_pair_feature_columns())
