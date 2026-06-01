"""Bridge dropped frame pairs across physical errors.

Bridging is CURRENTLY triggered only for ``QADecision.action == DROP`` rows whose
``drop_reason`` indicates a physical anomaly (or, optionally, a strong
classifier-DP disagreement). The bridge scorer is by default the same Phase 8
candidate classifier; bridging spans up to max_gap adjacent pairs. Need to make some changes here
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol

from mm_pipeline.config import TrackerParams

from .decisions import Action, DropReason, QADecision


class BridgeScorer(Protocol):
    name: str

    def score(self, candidates: Any) -> Any: ...


@dataclass
class ReuseCandidateClassifier:
    """Wrap a fitted Phase 8 scorer for use on bridge (dt > 1) candidates.
    Caveat: the underlying classifier is trained on dt=1 candidate
    distributions. On larger gaps the score scale may shift; calibrate
    tau_bridge per-deployment rather than reusing the dt=1 calibration.
    """

    fitted_scorer: Any
    name: str = "reuse_candidate_classifier"

    def score(self, candidates: Any) -> Any:
        if hasattr(self.fitted_scorer, "raw_scores"):
            return self.fitted_scorer.raw_scores(candidates)
        if hasattr(self.fitted_scorer, "predict_proba"):
            import numpy as np
            proba = self.fitted_scorer.predict_proba(candidates)
            arr = np.asarray(proba, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, 1]
            return arr
        if hasattr(self.fitted_scorer, "decision_function"):
            import numpy as np
            return np.asarray(self.fitted_scorer.decision_function(candidates), dtype=float)
        raise TypeError("BridgeScorer does not know how to score with the provided model.")


@dataclass
class BridgeAttempt:
    t_a: int
    t_b: int
    ops_json: str
    score: float


def bridge_drops(
    decisions: list[QADecision],
    labels: Any,
    *,
    bridge_scorer: BridgeScorer,
    tau_bridge: float,
    max_gap: int,
    tracker_params: TrackerParams,
    top_k: int,
    open_end: str = "high",
    bridge_drop_reasons: Iterable[DropReason] = (DropReason.ANOMALY,),
) -> list[QADecision]:
    """Convert eligible drops into bridges in-place over a copy of decisions.

    Iterates dropped-pair t-values in order, attempts to find a bridge that
    spans the drop at ``delta_t = 2, 3, ..., max_gap``, and accepts the first
    bridge whose score clears ``tau_bridge``. Pairs covered by a successful
    bridge are marked ``Action.BRIDGE``; the lowest-t covered pair carries the
    ops as ``bridge_is_primary=True``. Drops that fail to bridge are kept as
    drops with ``drop_reason=BRIDGE_FAILED``.
    """

    from mm_pipeline.features.pairwise import solve_and_featurize_pair

    eligible_reasons = set(bridge_drop_reasons)
    decisions_by_t = {d.t: d for d in decisions}
    T = int(labels.shape[0])

    drops_to_consider = sorted(
        d.t for d in decisions if d.action == Action.DROP and d.drop_reason in eligible_reasons
    )
    bridged: set[int] = set()
    spans: dict[int, BridgeAttempt] = {}  # primary t_a -> attempt

    for t_drop in drops_to_consider:
        if t_drop in bridged:
            continue
        attempt = _find_bridge(
            t_drop,
            labels,
            T,
            bridge_scorer=bridge_scorer,
            max_gap=max_gap,
            tracker_params=tracker_params,
            top_k=top_k,
            open_end=open_end,
            tau_bridge=tau_bridge,
            solver=solve_and_featurize_pair,
        )
        if attempt is None:
            continue
        spans[attempt.t_a] = attempt
        for ti in range(attempt.t_a, attempt.t_b):
            bridged.add(ti)

    # Build the updated decisions list. The primary bridge row is at t_a (the bridge's start frame), which  not equal the original drop's t.
    out: list[QADecision] = []
    for d in decisions:
        if d.t in bridged:
            covering_span = _find_covering_span(d.t, spans)
            if covering_span is None:
                out.append(d)
                continue
            is_primary = d.t == covering_span.t_a
            out.append(
                _bridge_decision(
                    d,
                    bridge_span=(covering_span.t_a, covering_span.t_b),
                    bridge_ops_json=covering_span.ops_json if is_primary else None,
                    bridge_score=covering_span.score,
                    bridge_is_primary=is_primary,
                )
            )
        elif (
            d.action == Action.DROP
            and d.drop_reason in eligible_reasons
        ):
            # Eligible drop that did not get covered: stayed dropped because no bridge cleared the tau thresh
            out.append(_drop_decision_failed(d))
        else:
            out.append(d)
    return out


def _bridge_decision(
    d: QADecision,
    *,
    bridge_span: tuple[int, int],
    bridge_ops_json: Optional[str],
    bridge_score: float,
    bridge_is_primary: bool,
) -> QADecision:
    from dataclasses import replace
    return replace(
        d,
        action=Action.BRIDGE,
        drop_reason=None,
        bridge_span=bridge_span,
        bridge_ops_json=bridge_ops_json,
        bridge_score=float(bridge_score),
        bridge_is_primary=bridge_is_primary,
    )


def _drop_decision_failed(d: QADecision) -> QADecision:
    from dataclasses import replace
    return replace(d, drop_reason=DropReason.BRIDGE_FAILED)


def _find_covering_span(t: int, spans: dict[int, BridgeAttempt]) -> Optional[BridgeAttempt]:
    for attempt in spans.values():
        if attempt.t_a <= t < attempt.t_b:
            return attempt
    return None


def _find_bridge(
    t_drop: int,
    labels: Any,
    T: int,
    *,
    bridge_scorer: BridgeScorer,
    max_gap: int,
    tracker_params: TrackerParams,
    top_k: int,
    open_end: str,
    tau_bridge: float,
    solver,
) -> Optional[BridgeAttempt]:
    import numpy as np

    for delta_t in range(2, max_gap + 1):
        best: Optional[BridgeAttempt] = None
        for t_a in range(max(0, t_drop - delta_t + 1), min(t_drop + 1, T - delta_t)):
            t_b = t_a + delta_t
            cand_df = solver(
                labels[t_a],
                labels[t_b],
                t=t_a,
                k=t_b,
                open_end=open_end,
                params=tracker_params,
                top_k=top_k,
                store_ops=True,
            )
            if cand_df.empty:
                continue
            scores = np.asarray(bridge_scorer.score(cand_df), dtype=float)
            if scores.size == 0:
                continue
            best_pos = int(np.argmax(scores))
            best_score = float(scores[best_pos])
            if best is None or best_score > best.score:
                best = BridgeAttempt(
                    t_a=int(t_a),
                    t_b=int(t_b),
                    ops_json=str(cand_df.iloc[best_pos]["ops_json"]),
                    score=best_score,
                )
        if best is not None and best.score >= tau_bridge:
            return best
    return None
