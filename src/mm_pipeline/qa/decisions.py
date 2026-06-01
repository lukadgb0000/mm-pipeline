"""CURRENT QA decision contracts"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Action(str, Enum):
    KEEP = "keep"
    DROP = "drop"
    BRIDGE = "bridge"


class DropReason(str, Enum):
    ANOMALY = "anomaly"
    DISAGREEMENT = "disagreement"
    BRIDGE_FAILED = "bridge_failed"


@dataclass
class QADecision:
    """One QA decision per frame pair.
    The within-pair stage records the chosen candidate plus the DP and
    classifier top-1 picks regardless of which scorer drove the choice. The
    per-pair anomaly stage adds an anomaly score and flag. The final action
    is the resolution after applying any anomaly drop / bridge / disagreement
    drop policy.
    """

    dataset_id: str
    pair_id: str
    t: int
    n_candidates: int

    within_pair_scorer: str
    chosen_candidate_idx: Optional[Any]
    chosen_ops_json: Optional[str]
    dp_top1_idx: Optional[Any]
    classifier_top1_idx: Optional[Any]

    classifier_disagrees_with_dp: bool
    dp_cost_gap: float
    dp_cost_gap_normalised: float
    classifier_score_gap: float
    within_pair_max_score: float
    within_pair_entropy: float
    within_pair_margin_top1_top2: float
    disagreement_score: float

    anomaly_detector: str
    anomaly_score: float
    anomaly_flag: bool

    action: Action
    drop_reason: Optional[DropReason] = None

    bridge_span: Optional[tuple[int, int]] = None
    bridge_ops_json: Optional[str] = None
    bridge_score: float = float("nan")
    bridge_is_primary: bool = False

    has_correct_candidate: Optional[bool] = None
    chosen_is_correct: Optional[bool] = None

    def to_row(self) -> dict[str, Any]:
        """Flatten to a row suitable for ``qa_decisions.csv``."""

        data: dict[str, Any] = asdict(self)
        data["action"] = self.action.value
        data["drop_reason"] = self.drop_reason.value if self.drop_reason is not None else None
        span = data.pop("bridge_span")
        data["bridge_t_a"] = span[0] if span is not None else None
        data["bridge_t_b"] = span[1] if span is not None else None
        return data


def validate_decision(d: QADecision) -> None:
    """Check the schema-level invariants on a single decision"""

    if d.action == Action.KEEP:
        if d.chosen_candidate_idx is None:
            raise ValueError(f"action=KEEP requires chosen_candidate_idx; pair_id={d.pair_id}.")
        if d.drop_reason is not None:
            raise ValueError(f"action=KEEP must not have a drop_reason; pair_id={d.pair_id}.")
        if d.bridge_ops_json is not None or d.bridge_is_primary:
            raise ValueError(f"action=KEEP must not have bridge metadata; pair_id={d.pair_id}.")
    elif d.action == Action.DROP:
        if d.drop_reason is None:
            raise ValueError(f"action=DROP requires a drop_reason; pair_id={d.pair_id}.")
        if d.bridge_is_primary:
            raise ValueError(f"action=DROP cannot be a bridge primary; pair_id={d.pair_id}.")
    elif d.action == Action.BRIDGE:
        if d.bridge_span is None:
            raise ValueError(f"action=BRIDGE requires bridge_span; pair_id={d.pair_id}.")
        t_a, t_b = d.bridge_span
        if t_b <= t_a:
            raise ValueError(f"bridge_span must satisfy t_b > t_a; pair_id={d.pair_id}.")
        if d.bridge_is_primary and (d.bridge_ops_json is None or math.isnan(d.bridge_score)):
            raise ValueError(f"primary bridge requires ops_json and score; pair_id={d.pair_id}.")
        if not d.bridge_is_primary and d.bridge_ops_json is not None:
            raise ValueError(f"non-primary bridge must not carry ops_json; pair_id={d.pair_id}.")
