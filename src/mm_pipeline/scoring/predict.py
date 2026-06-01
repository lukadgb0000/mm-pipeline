"""Prediction utilities for fitted scorers"""

from __future__ import annotations

from typing import Any

import numpy as np

from .train import FittedScorer, _require_pandas


def _add_pair_probabilities(out: Any, *, pair_col: str, pair_temperature: float) -> Any:
    if pair_temperature <= 0.0:
        raise ValueError("pair_temperature must be > 0.")
    if pair_col not in out.columns:
        raise KeyError(f"Missing pair column '{pair_col}'.")

    out["pair_probability"] = np.nan
    out["score_rank"] = np.nan

    for _, group in out.groupby(pair_col, sort=False):
        idx = group.index
        scores = group["raw_score"].to_numpy(dtype=float)
        finite = np.isfinite(scores)
        if not np.any(finite):
            continue

        finite_scores = scores[finite] / float(pair_temperature)
        finite_scores = finite_scores - float(np.max(finite_scores))
        exp_scores = np.exp(finite_scores)
        probs = exp_scores / float(np.sum(exp_scores))

        # Stable argsort assigns sequential ranks to ties (1, 2, ... in input
        # order). Continuous feature distributions make exact ties vanishingly
        # rare in practice; switch to a tied-rank scheme here only if a caller
        # needs ties preserved.
        ranks = np.empty(len(finite_scores), dtype=float)
        order = np.argsort(-finite_scores, kind="mergesort")
        ranks[order] = np.arange(1, len(finite_scores) + 1, dtype=float)

        finite_idx = idx[np.flatnonzero(finite)]
        out.loc[finite_idx, "pair_probability"] = probs
        out.loc[finite_idx, "score_rank"] = ranks

    out["pair_prob"] = out["pair_probability"]
    out["pair_score_rank"] = out["score_rank"]
    return out


def score_candidates(
    feature_table: Any,
    fitted_scorer: FittedScorer,
    pair_col: str = "pair_id",
    pair_temperature: float = 1.0,
) -> Any:
    """Score candidate rows abd contract

    Columns added:
      - ``raw_score``: model-specific ranking evidence
      - ``candidate_correctness_probability``: row-level probability when available
      - ``pair_probability``: within-pair softmax over ``raw_score``
      - ``score_rank``: rank within pair, 1 is best
      - ``y_score``: legacy alias; probability when finite, else raw score
    """

    pd = _require_pandas()
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError("feature_table must be a pandas DataFrame.")
    if pair_temperature <= 0.0:
        raise ValueError("pair_temperature must be > 0.")
    if pair_col not in feature_table.columns:
        raise KeyError(f"Missing pair column '{pair_col}'.")

    out = feature_table.copy()
    if out.empty:
        for col in (
            "raw_score",
            "raw_score_kind",
            "candidate_correctness_probability",
            "score_model",
            "score_feature_subset",
            "score_is_calibrated",
            "y_score",
            "pair_probability",
            "score_rank",
            "pair_prob",
            "pair_score_rank",
        ):
            out[col] = []
        return out

    raw = fitted_scorer.raw_scores(out)
    probs = fitted_scorer.candidate_probabilities(out, raw_scores=raw, pair_col=pair_col)

    out["raw_score"] = raw
    out["raw_score_kind"] = fitted_scorer.raw_score_kind
    out["candidate_correctness_probability"] = probs
    out["score_model"] = fitted_scorer.model_name
    out["score_feature_subset"] = (
        fitted_scorer.feature_subset
        if isinstance(fitted_scorer.feature_subset, str)
        else ",".join(fitted_scorer.feature_subset)
    )
    out["score_is_calibrated"] = bool(fitted_scorer.is_calibrated)

    y_score = np.asarray(probs, dtype=float).copy()
    missing_prob = ~np.isfinite(y_score)
    y_score[missing_prob] = raw[missing_prob]
    out["y_score"] = y_score

    return _add_pair_probabilities(out, pair_col=pair_col, pair_temperature=pair_temperature)


def score_with_lodo_scorers(
    feature_table: Any,
    scorers: dict[Any, FittedScorer],
    heldout_col: str = "dataset_id",
    **score_options: Any,
) -> Any:
    """Score rows with a mapping of held-out value to fitted scorer"""

    pd = _require_pandas()
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError("feature_table must be a pandas DataFrame.")
    if heldout_col not in feature_table.columns:
        raise KeyError(f"Missing heldout column '{heldout_col}'.")
    if feature_table.empty:
        return feature_table.copy()

    parts = []
    missing: list[Any] = []
    for heldout, group in feature_table.groupby(heldout_col, sort=False):
        if heldout not in scorers:
            missing.append(heldout)
            continue
        parts.append(score_candidates(group, scorers[heldout], **score_options))
    if missing:
        raise KeyError(f"No fitted scorer for held-out value(s): {missing}")
    scored = pd.concat(parts, axis=0)
    if feature_table.index.is_unique:
        return scored.loc[feature_table.index]
    return scored.sort_index(kind="stable")
