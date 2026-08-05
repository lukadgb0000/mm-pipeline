"""Per-cell-cycle metrics: a registry of reductions over one ``track_id``.

A ``track_id`` is one cell lifespan (from birth to division). A metric receives an explicit
:class:`CycleContext` (the track's frames and, optionally, its per-cell
properties) so what an author may read is legible and the per-track slice is
computed once by the driver rather than re-scanned per metric.

Register a metric with ``@cycle_metric("name")``; it returns a scalar or a
``dict`` (expanded into columns by :func:`metrics`). Metrics that need per-cell
properties raise a clear error unless the driver is called with
``with_properties=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class CycleContext:
    """Everything a cycle metric may read for one ``track_id``."""

    lineage: Any
    track_id: int
    parent_id: Optional[int]
    frames: Any  # this track's tracks_df slice, time-sorted, observed frames only
    properties: Optional[Any]  # this track's cell_properties slice, or None


_REGISTRY: dict[str, Callable[[CycleContext], Any]] = {}

# Built-in metrics that read ``ctx.properties`` — a driver must run them with
# ``with_properties=True`` (which needs the lineage's label images).
PROPERTY_METRICS = frozenset({"birth_length", "added_length", "growth_rate"})


def cycle_metric(name: str) -> Callable[[Callable], Callable]:
    """Decorator: register ``fn`` under ``name`` and return it unchanged."""

    def register(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return register


def get_cycle_metric(name: str) -> Callable[[CycleContext], Any]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown cycle metric '{name}'. Available metrics: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_cycle_metrics() -> list[str]:
    return sorted(_REGISTRY)


# helpers 


def _require_properties(ctx: CycleContext, name: str) -> Any:
    if ctx.properties is None:
        raise ValueError(f"metric '{name}' requires with_properties=True (per-cell properties).")
    if "major_axis_length_px" not in ctx.properties.columns:
        raise ValueError(f"metric '{name}' needs 'major_axis_length_px' in the requested props.")
    return ctx.properties


def _length_at(props: Any, t: int) -> float:
    match = props.loc[props["t"] == t, "major_axis_length_px"]
    return float(match.iloc[0]) if not match.empty else float("nan")


def _loglinear_fit(x: Any, y: Any) -> tuple[float, float, int]:
    """Fit ``log(y) ~ x`` and return ``(slope, r2, n)`` over finite, positive y."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[mask], y[mask]
    n = int(x.size)
    if n < 2:
        return float("nan"), float("nan"), n
    logy = np.log(y)
    slope, intercept = np.polyfit(x, logy, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((logy - pred) ** 2))
    ss_tot = float(np.sum((logy - logy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), r2, n


# built-in metrics


@cycle_metric("cycle_time")
def cycle_time(ctx: CycleContext) -> float:
    """Generation time = ``(end_t - birth_t + 1) * frame_interval_min``.

    The ``+1`` counts the interval from the parent's division — the physical
    birth, one frame before the daughter first appears — to this cell's own
    division. Robust to gaps (gaps are missing observations, not missing time); a
    true generation time only for a ``complete_cycle`` (filter via ``cycles``).
    In frames when ``frame_interval_min`` is unset.

    NOTE: the ``+1`` is the plan's recommendation, but the lab's exact definition
    of generation time is UNCONFIRMED. Revisit before relying on absolute values.
    """
    t = ctx.frames["t"]
    span = int(t.iloc[-1]) - int(t.iloc[0]) + 1
    interval = ctx.lineage.frame_interval_min
    return span * interval if interval is not None else float(span)


@cycle_metric("birth_length")
def birth_length(ctx: CycleContext) -> float:
    """Fitted length (``major_axis_length_px``) at the first observed frame"""
    props = _require_properties(ctx, "birth_length")
    return _length_at(props, int(ctx.frames["t"].iloc[0]))


@cycle_metric("added_length")
def added_length(ctx: CycleContext) -> float:
    """Length at division minus length at birth"""
    props = _require_properties(ctx, "added_length")
    t = ctx.frames["t"]
    return _length_at(props, int(t.iloc[-1])) - _length_at(props, int(t.iloc[0]))


@cycle_metric("growth_rate")
def growth_rate(ctx: CycleContext) -> dict[str, float]:
    """Log-linear (exponential) growth rate of length over observed frames.

    Returns ``growth_rate`` (slope of ``log(length)`` vs time), ``growth_rate_r2``,
    and ``growth_rate_n``, so a two-point fit (r2 undefined / trivially 1) is
    visibly untrustworthy. Time is minutes when ``frame_interval_min`` is set,
    else frames.
    """
    props = _require_properties(ctx, "growth_rate").sort_values("t")
    interval = ctx.lineage.frame_interval_min
    x = props["t"].to_numpy(dtype=float)
    if interval is not None:
        x = x * interval
    slope, r2, n = _loglinear_fit(x, props["major_axis_length_px"].to_numpy(dtype=float))
    return {"growth_rate": slope, "growth_rate_r2": r2, "growth_rate_n": n}


# driver 


def metrics(lin: Any, tracks: Any, names: Any, *, with_properties: bool = False) -> Any:
    """Compute ``names`` for every track in ``tracks``, joined onto ``cycles``

    With ``with_properties=True`` the per-cell properties are computed **once**
    (via :func:`cell_properties`) and sliced per track into each ``CycleContext``.
    Dict-returning metrics (e.g. ``growth_rate``) are expanded into columns.
    """
    import pandas as pd

    fns = [(name, get_cycle_metric(name)) for name in names]
    ids = sorted(int(t) for t in tracks if int(t) in lin.frames_by_track)

    props_by_track: dict[int, Any] = {}
    if with_properties:
        from .properties import cell_properties

        props = cell_properties(lin, tracks)
        props_by_track = {int(tid): sub for tid, sub in props.groupby("track_id")}

    parent = lin.parent_map
    records = []
    for tid in ids:
        ctx = CycleContext(
            lineage=lin,
            track_id=tid,
            parent_id=parent.get(tid),
            frames=lin.frames_by_track[tid],
            properties=props_by_track.get(tid) if with_properties else None,
        )
        row: dict[str, Any] = {"track_id": tid}
        for name, fn in fns:
            result = fn(ctx)
            if isinstance(result, dict):
                row.update(result)
            else:
                row[name] = result
        records.append(row)

    result_df = pd.DataFrame(records) if records else pd.DataFrame(columns=["track_id"])
    cyc = lin.cycles
    return cyc[cyc["track_id"].isin(ids)].merge(result_df, on="track_id", how="left")
