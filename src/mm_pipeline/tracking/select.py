"""Within-pair scoring and per-pair candidate selection

Each scorer takes a pandas DataFrame of scored candidates for one pair and returns the chosen row's index label plus diagnostics. 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class WithinPairPick:
    chosen_idx: Any
    chosen_score: float
    diagnostics: dict[str, Any]


class WithinPairScorer(Protocol):
    name: str

    def pick(self, pair_candidates: Any) -> WithinPairPick: ...


def _require_dp_cost(pair_candidates: Any) -> None:
    if "dp_cost" not in pair_candidates.columns:
        raise KeyError("DPCostMin requires a 'dp_cost' column on candidates.")


def _require_raw_score(pair_candidates: Any) -> None:
    if "raw_score" not in pair_candidates.columns:
        raise KeyError("ClassifierMax requires a 'raw_score' column on candidates.")


class DPCostMin:
    """Pick the candidate with the smallest dp_cost.

    Equivalent to taking is_dpt_best
    """

    name = "dp_cost_min"

    def pick(self, pair_candidates: Any) -> WithinPairPick:
        _require_dp_cost(pair_candidates)
        if pair_candidates.empty:
            raise ValueError("Cannot pick from an empty candidate set.")
        idxmin = pair_candidates["dp_cost"].astype(float).idxmin()
        return WithinPairPick(
            chosen_idx=idxmin,
            chosen_score=float(pair_candidates.at[idxmin, "dp_cost"]),
            diagnostics={"score_basis": "dp_cost"},
        )


class ClassifierMax:
    """Pick the candidate with the largest raw_score"""

    name = "classifier"

    def pick(self, pair_candidates: Any) -> WithinPairPick:
        _require_raw_score(pair_candidates)
        if pair_candidates.empty:
            raise ValueError("Cannot pick from an empty candidate set.")
        idxmax = pair_candidates["raw_score"].astype(float).idxmax()
        return WithinPairPick(
            chosen_idx=idxmax,
            chosen_score=float(pair_candidates.at[idxmax, "raw_score"]),
            diagnostics={"score_basis": "raw_score"},
        )


class Ensemble:
    """Linear combination of DP and classifier picks for the sake of interest

    Two modes
    """

    def __init__(self, alpha: float = 0.5, mode: str = "rank") -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if mode not in {"rank", "zscore"}:
            raise ValueError("mode must be 'rank' or 'zscore'.")
        self.alpha = float(alpha)
        self.mode = mode
        self.name = f"ensemble({mode},alpha={self.alpha:g})"

    def pick(self, pair_candidates: Any) -> WithinPairPick:
        _require_dp_cost(pair_candidates)
        _require_raw_score(pair_candidates)
        if pair_candidates.empty:
            raise ValueError("Cannot pick from an empty candidate set.")

        if self.mode == "rank":
            return self._pick_by_rank(pair_candidates)
        return self._pick_by_zscore(pair_candidates)

    def _pick_by_rank(self, pair_candidates: Any) -> WithinPairPick:

        dp_rank = pair_candidates["dp_cost"].astype(float).rank(method="first", ascending=True)
        cls_rank = pair_candidates["raw_score"].astype(float).rank(method="first", ascending=False)
        combined = self.alpha * dp_rank + (1.0 - self.alpha) * cls_rank
        order = [
            (combined_value, dp_rank_value, pos, idx)
            for pos, (combined_value, dp_rank_value, idx) in enumerate(
                zip(combined.tolist(), dp_rank.tolist(), pair_candidates.index.tolist())
            )
        ]
        order.sort()
        _combined_score, _dp_rank, _pos, chosen_idx = order[0]
        return WithinPairPick(
            chosen_idx=chosen_idx,
            chosen_score=float(combined.loc[chosen_idx]),
            diagnostics={"score_basis": "rank_ensemble", "alpha": self.alpha},
        )

    def _pick_by_zscore(self, pair_candidates: Any) -> WithinPairPick:
        import numpy as np

        dp = -pair_candidates["dp_cost"].astype(float).to_numpy()
        cls = pair_candidates["raw_score"].astype(float).to_numpy()
        z_dp = _zscore(dp)
        z_cls = _zscore(cls)
        combined = self.alpha * z_dp + (1.0 - self.alpha) * z_cls
        chosen_pos = int(np.argmax(combined))
        chosen_idx = pair_candidates.index[chosen_pos]
        return WithinPairPick(
            chosen_idx=chosen_idx,
            chosen_score=float(combined[chosen_pos]),
            diagnostics={"score_basis": "zscore_ensemble", "alpha": self.alpha},
        )


def _zscore(values):
    import numpy as np
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return np.zeros_like(arr)
    std = float(arr.std())
    if std == 0.0:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / std


def build_scorer(
    name: str,
    *,
    ensemble_alpha: float = 0.5,
    ensemble_mode: str = "rank",
) -> WithinPairScorer:


    if name == "dp_cost_min":
        return DPCostMin()
    if name == "classifier":
        return ClassifierMax()
    if name == "ensemble":
        return Ensemble(alpha=ensemble_alpha, mode=ensemble_mode)
    raise ValueError(f"Unknown within_pair_scorer '{name}'.")


# per-pair selection 


@dataclass(frozen=True)
class SelectionResult:
    """One selection per frame-pair: the chosen candidate's ops plus context

    ``chosen_ops_json is None`` means break the lineage 
    ``chosen_idx`` / ``chosen_score`` are carried for CSV/diagnostics only and
    are NOT read by lineage reconstruction
    """

    pair_id: str
    t: int
    chosen_ops_json: str | None
    dataset_id: str = ""
    chosen_idx: Any | None = None
    chosen_score: float = float("nan")


def select_pairs(scored: Any, scorer: WithinPairScorer) -> list[SelectionResult]:
    """Pick the best candidate for each frame-pair"""

    if "pair_id" not in scored.columns:
        raise KeyError("select_pairs requires a 'pair_id' column.")

    selections: list[SelectionResult] = []
    for pair_id, pair in scored.groupby("pair_id", sort=False):
        pick = scorer.pick(pair)
        chosen_row = pair.loc[pick.chosen_idx]
        selections.append(
            SelectionResult(
                pair_id=str(pair_id),
                t=int(chosen_row["t"]) if "t" in pair.columns else -1,
                chosen_ops_json=(
                    str(chosen_row["ops_json"]) if "ops_json" in pair.columns else None
                ),
                dataset_id=str(chosen_row.get("dataset_id", "")),
                chosen_idx=pick.chosen_idx,
                chosen_score=pick.chosen_score,
            )
        )
    return selections
