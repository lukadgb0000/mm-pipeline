"""Top-level QA orchestration FOR NOW. COuld rethink this - should I separate within pair and anomaly detection more. Probably"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional

from mm_pipeline.config import QAConfig, TrackerParams

from .aggregation import build_per_pair_features
from .decisions import Action, DropReason, QADecision
from .physical_errors import NeverAnomalous, PhysicalErrorDetector, build_detector
from .within_pair import WithinPairScorer, build_scorer


@dataclass(frozen=True)
class _PairDiagnostics:
    dp_top1_idx: Optional[Any]
    classifier_top1_idx: Optional[Any]
    dp_cost_gap: float
    dp_cost_gap_normalised: float
    classifier_score_gap: float
    within_pair_max_score: float
    within_pair_entropy: float
    within_pair_margin_top1_top2: float
    disagreement_score: float
    classifier_disagrees_with_dp: bool


def apply_qa_workflow(
    scored_candidates: Any,
    *,
    config: QAConfig,
    within_pair_scorer: Optional[WithinPairScorer] = None,
    anomaly_detector: Optional[PhysicalErrorDetector] = None,
    bridge_scorer: Any = None,
    labels: Any = None,
    tracker_params: Optional[TrackerParams] = None,
    open_end: str = "high",
) -> list[QADecision]:
    

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("apply_qa_workflow requires pandas.") from exc

    if not isinstance(scored_candidates, pd.DataFrame):
        raise TypeError("scored_candidates must be a pandas DataFrame.")
    if scored_candidates.empty:
        return []
    if "pair_id" not in scored_candidates.columns:
        raise KeyError("scored_candidates must include 'pair_id'.")

    scorer = within_pair_scorer or build_scorer(
        config.within_pair_scorer,
        ensemble_alpha=config.within_pair_ensemble_alpha,
        ensemble_mode=config.within_pair_ensemble_mode,
    )
    detector = anomaly_detector or build_detector(config.anomaly_detector)

    if config.disagreement_drop != "never" and not config.bridge_enabled:
        warnings.warn(
            "disagreement_drop is set but bridge_enabled=False: drops will be "
            "recorded but no bridge will run; lineage breaks at every flagged pair.",
            stacklevel=2,
        )

    
    pick_rows: list[dict[str, Any]] = []
    for pair_id, pair in scored_candidates.groupby("pair_id", sort=False):
        pick = scorer.pick(pair)
        diag = _pair_diagnostics(pair, config.pair_temperature)
        pick_rows.append(
            {
                "pair_id": str(pair_id),
                "pair_view": pair,
                "pick_idx": pick.chosen_idx,
                "diag": diag,
            }
        )

    
    per_pair_features = build_per_pair_features(scored_candidates)
    anomaly_df = detector.detect(per_pair_features)
    anomaly_lookup: dict[str, tuple[float, bool]] = {
        str(row["pair_id"]): (float(row["anomaly_score"]), bool(row["anomaly_flag"]))
        for _, row in anomaly_df.iterrows()
    }

    
    decisions: list[QADecision] = []
    for entry in pick_rows:
        pair = entry["pair_view"]
        diag: _PairDiagnostics = entry["diag"]
        chosen_idx = entry["pick_idx"]
        chosen_row = pair.loc[chosen_idx]
        anomaly_score, anomaly_flag = anomaly_lookup.get(entry["pair_id"], (float("nan"), False))
        action, drop_reason = _initial_action(
            anomaly_flag=anomaly_flag,
            disagreement_score=diag.disagreement_score,
            disagreement_policy=config.disagreement_drop,
            soft_threshold=config.disagreement_soft_threshold,
            classifier_disagrees=diag.classifier_disagrees_with_dp,
        )

        decisions.append(
            QADecision(
                dataset_id=str(chosen_row.get("dataset_id", "")),
                pair_id=str(entry["pair_id"]),
                t=int(chosen_row["t"]) if "t" in pair.columns else -1,
                n_candidates=int(len(pair)),
                within_pair_scorer=scorer.name,
                chosen_candidate_idx=chosen_idx if action == Action.KEEP else None,
                chosen_ops_json=(
                    str(chosen_row["ops_json"]) if action == Action.KEEP and "ops_json" in pair.columns else None
                ),
                dp_top1_idx=diag.dp_top1_idx,
                classifier_top1_idx=diag.classifier_top1_idx,
                classifier_disagrees_with_dp=diag.classifier_disagrees_with_dp,
                dp_cost_gap=diag.dp_cost_gap,
                dp_cost_gap_normalised=diag.dp_cost_gap_normalised,
                classifier_score_gap=diag.classifier_score_gap,
                within_pair_max_score=diag.within_pair_max_score,
                within_pair_entropy=diag.within_pair_entropy,
                within_pair_margin_top1_top2=diag.within_pair_margin_top1_top2,
                disagreement_score=diag.disagreement_score,
                anomaly_detector=detector.name,
                anomaly_score=float(anomaly_score),
                anomaly_flag=bool(anomaly_flag),
                action=action,
                drop_reason=drop_reason,
                has_correct_candidate=_optional_bool_any(pair, "is_correct"),
                chosen_is_correct=_optional_bool_at(pair, "is_correct", chosen_idx) if action == Action.KEEP else None,
            )
        )

    
    if config.bridge_enabled:
        if labels is None or tracker_params is None or bridge_scorer is None:
            raise ValueError(
                "Bridging requires labels, tracker_params, and bridge_scorer to be provided."
            )
        from .bridge import bridge_drops

        eligible: list[DropReason] = [DropReason.ANOMALY]
        if config.disagreement_drop != "never":
            eligible.append(DropReason.DISAGREEMENT)
        decisions = bridge_drops(
            decisions,
            labels=labels,
            bridge_scorer=bridge_scorer,
            tau_bridge=config.bridge_tau,
            max_gap=config.bridge_max_gap,
            tracker_params=tracker_params,
            top_k=config.bridge_top_k,
            open_end=open_end,
            bridge_drop_reasons=eligible,
        )

    return decisions


def _initial_action(
    *,
    anomaly_flag: bool,
    disagreement_score: float,
    disagreement_policy: str,
    soft_threshold: float,
    classifier_disagrees: bool,
) -> tuple[Action, Optional[DropReason]]:
    if anomaly_flag:
        return Action.DROP, DropReason.ANOMALY
    if disagreement_policy == "hard" and classifier_disagrees:
        return Action.DROP, DropReason.DISAGREEMENT
    if disagreement_policy == "soft" and math.isfinite(disagreement_score) and disagreement_score > soft_threshold:
        return Action.DROP, DropReason.DISAGREEMENT
    return Action.KEEP, None


def _pair_diagnostics(pair: Any, pair_temperature: float) -> _PairDiagnostics:
    import numpy as np

    has_dp = "dp_cost" in pair.columns and pair["dp_cost"].notna().any()
    has_cls = "raw_score" in pair.columns and pair["raw_score"].notna().any()

    dp_top1_idx: Optional[Any] = None
    classifier_top1_idx: Optional[Any] = None
    dp_cost_gap = float("nan")
    dp_cost_gap_norm = float("nan")
    classifier_score_gap = float("nan")
    within_pair_max_score = float("nan")
    within_pair_margin = float("nan")
    disagreement_score = float("nan")
    classifier_disagrees = False

    if has_dp:
        dp_top1_idx = pair["dp_cost"].astype(float).idxmin()
    if has_cls:
        classifier_top1_idx = pair["raw_score"].astype(float).idxmax()
        scores = pair["raw_score"].astype(float).to_numpy()
        finite = scores[np.isfinite(scores)]
        if finite.size:
            sorted_desc = np.sort(finite)[::-1]
            within_pair_max_score = float(sorted_desc[0])
            if sorted_desc.size >= 2:
                within_pair_margin = float(sorted_desc[0] - sorted_desc[1])

    if has_dp and has_cls and dp_top1_idx is not None and classifier_top1_idx is not None:
        classifier_disagrees = bool(dp_top1_idx != classifier_top1_idx)
        dp_dp = float(pair.at[dp_top1_idx, "dp_cost"])
        dp_cls = float(pair.at[classifier_top1_idx, "dp_cost"])
        raw_dp = float(pair.at[dp_top1_idx, "raw_score"])
        raw_cls = float(pair.at[classifier_top1_idx, "raw_score"])
        dp_cost_gap = max(dp_cls - dp_dp, 0.0)
        dp_cost_gap_norm = dp_cost_gap / max(abs(dp_dp), 1e-9)
        classifier_score_gap = max(raw_cls - raw_dp, 0.0)
        disagreement_score = float(math.sqrt(dp_cost_gap_norm * classifier_score_gap))

    entropy = _entropy_pair_probability(pair, pair_temperature)

    return _PairDiagnostics(
        dp_top1_idx=dp_top1_idx,
        classifier_top1_idx=classifier_top1_idx,
        dp_cost_gap=float(dp_cost_gap),
        dp_cost_gap_normalised=float(dp_cost_gap_norm),
        classifier_score_gap=float(classifier_score_gap),
        within_pair_max_score=float(within_pair_max_score),
        within_pair_entropy=float(entropy),
        within_pair_margin_top1_top2=float(within_pair_margin),
        disagreement_score=float(disagreement_score),
        classifier_disagrees_with_dp=classifier_disagrees,
    )


def _entropy_pair_probability(pair: Any, pair_temperature: float) -> float:
    import numpy as np

    if "pair_probability" in pair.columns and pair["pair_probability"].notna().any():
        probs = pair["pair_probability"].astype(float).to_numpy()
    elif "raw_score" in pair.columns and pair["raw_score"].notna().any():
        scores = pair["raw_score"].astype(float).to_numpy() / max(pair_temperature, 1e-9)
        scores = scores - float(np.nanmax(scores))
        exp = np.exp(scores)
        denom = float(np.nansum(exp))
        if denom == 0.0:
            return float("nan")
        probs = exp / denom
    else:
        return float("nan")
    probs = probs[np.isfinite(probs) & (probs > 0)]
    if probs.size == 0:
        return float("nan")
    return float(-np.sum(probs * np.log(probs)))


def _optional_bool_any(pair: Any, col: str) -> Optional[bool]:
    if col not in pair.columns:
        return None
    series = pair[col].dropna()
    if series.empty:
        return None
    return bool(series.astype(bool).any())


def _optional_bool_at(pair: Any, col: str, idx: Any) -> Optional[bool]:
    if col not in pair.columns:
        return None
    value = pair.at[idx, col]
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except ImportError:
        pass
    return bool(value)
