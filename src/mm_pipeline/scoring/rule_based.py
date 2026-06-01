"""Rule-based plausibility diagnostics

Just ignore these for now they're a bit silly I'll do something about it
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _column(df: Any, preferred: str, aliases: Sequence[str] = ()) -> str:
    for name in (preferred, *aliases):
        if name in df.columns:
            return name
    raise KeyError(f"Missing required column '{preferred}'" + (f" or aliases {list(aliases)}" if aliases else ""))


def classify_max_shrink(df: Any, threshold: float = 10.0, column: str = "max_shrink_pct") -> Any:
    """Return True where max shrinkage is within a plausible threshold"""

    col = _column(df, column)
    return df[col] <= float(threshold)


def classify_area_ratio(
    df: Any,
    low: float = 0.9,
    high: float = 1.2,
    column: str = "total_area_ratio_exit_adjusted",
) -> Any:
    """Return True where total area ratio is within a plausible interval"""

    col = _column(df, column, aliases=("mass_ratio",))
    return (df[col] >= float(low)) & (df[col] <= float(high))


def classify_norm_cost(df: Any, threshold: float = 10.0, column: str = "norm_cost") -> Any:
    """Return True where normalised DP cost is within a plausible threshold"""

    col = _column(df, column)
    return df[col] <= float(threshold)


def ensemble_or(df: Any, pred_cols: Sequence[str]) -> Any:
    """Return row-wise OR across boolean diagnostic columns"""

    if not pred_cols:
        raise ValueError("pred_cols must be non-empty.")
    out = df[pred_cols[0]].astype(bool)
    for col in pred_cols[1:]:
        out = out | df[col].astype(bool)
    return out


def ensemble_and(df: Any, pred_cols: Sequence[str]) -> Any:
    """Return row-wise AND across boolean diagnostic columns."""

    if not pred_cols:
        raise ValueError("pred_cols must be non-empty.")
    out = df[pred_cols[0]].astype(bool)
    for col in pred_cols[1:]:
        out = out & df[col].astype(bool)
    return out


def add_rule_based_diagnostics(
    df: Any,
    *,
    shrink_threshold: float = 10.0,
    area_low: float = 0.9,
    area_high: float = 1.2,
    norm_cost_threshold: float | None = None,
) -> Any:
    """Add rule-based plausibility flags and a coarse probability-style score."""

    out = df.copy()
    diagnostic_cols: list[str] = []

    out["rule_plausible_shrink"] = classify_max_shrink(out, threshold=shrink_threshold)
    diagnostic_cols.append("rule_plausible_shrink")

    out["rule_plausible_area_ratio"] = classify_area_ratio(out, low=area_low, high=area_high)
    diagnostic_cols.append("rule_plausible_area_ratio")

    if norm_cost_threshold is not None and "norm_cost" in out.columns:
        out["rule_plausible_norm_cost"] = classify_norm_cost(out, threshold=norm_cost_threshold)
        diagnostic_cols.append("rule_plausible_norm_cost")

    out["rule_plausible_all"] = ensemble_and(out, diagnostic_cols)
    out["rule_plausible_any"] = ensemble_or(out, diagnostic_cols)
    out["rule_based_probability"] = out["rule_plausible_all"].astype(float)
    return out
