"""Branch selection over a :class:`~mm_pipeline.analysis.tree.Lineage`.

A :class:`TrackSet` is a named, dataset-scoped set of ``track_id``\\ s. It carries
its ``dataset_id`` so a selection built from one lineage cannot be silently
applied to another (whose track IDs restart at 1 and would "match" wrongly).

Selectors are plain functions returning a ``TrackSet``; compose them with set
algebra (``|``, ``&``, ``-``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = [
    "TrackSet",
    "mother_branch",
    "descendants_of",
    "ancestors_of",
    "path_between",
    "roots",
    "leaves",
    "generation",
    "filter_cycles",
]


@dataclass(frozen=True)
class TrackSet:
    """A named set of ``track_id``\\ s scoped to one ``dataset_id``"""

    dataset_id: str
    track_ids: frozenset[int]
    name: str = ""

    def __iter__(self):
        return iter(self.track_ids)

    def __len__(self) -> int:
        return len(self.track_ids)

    def __contains__(self, track_id: int) -> bool:
        return int(track_id) in self.track_ids

    def _combine(self, other: "TrackSet", op: Callable[[frozenset, frozenset], frozenset], sep: str) -> "TrackSet":
        if self.dataset_id != other.dataset_id:
            raise ValueError(f"Cannot combine TrackSets from {self.dataset_id!r} and {other.dataset_id!r}.")
        name = f"{self.name} {sep} {other.name}".strip() if (self.name or other.name) else ""
        return TrackSet(self.dataset_id, op(self.track_ids, other.track_ids), name)

    def __or__(self, other: "TrackSet") -> "TrackSet":
        return self._combine(other, lambda a, b: a | b, "|")

    def __and__(self, other: "TrackSet") -> "TrackSet":
        return self._combine(other, lambda a, b: a & b, "&")

    def __sub__(self, other: "TrackSet") -> "TrackSet":
        return self._combine(other, lambda a, b: a - b, "-")

    def cycles(self, lin: Any) -> Any:
        """The rows of ``lin.cycles`` for these tracks."""
        self._require(lin)
        return lin.cycles[lin.cycles["track_id"].isin(self.track_ids)]

    def detections(self, lin: Any) -> Any:
        """The rows of ``lin.tracks_df`` for these tracks."""
        self._require(lin)
        return lin.tracks_df[lin.tracks_df["track_id"].isin(self.track_ids)]

    def _require(self, lin: Any) -> None:
        if lin.dataset_id != self.dataset_id:
            raise ValueError(f"TrackSet is for {self.dataset_id!r}, not lineage {lin.dataset_id!r}.")


def _closed_daughter(d1: int, d2: int, pos1: Optional[float], pos2: Optional[float], open_end: str) -> int:
    """The daughter nearest the closed end — the continuing mother.

    Positions are along the trench axis; the closed end is opposite ``open_end``.
    """
    if pos1 is None or pos2 is None:
        return d1
    if open_end == "high":
        return d1 if pos1 < pos2 else d2
    return d1 if pos1 > pos2 else d2


def _birth_positions(lin: Any) -> dict[int, tuple[int, float]]:
    """``{track_id: (birth_t, axis position at birth)}``."""
    col = lin.axis_col
    out: dict[int, tuple[int, float]] = {}
    for tid, frames in lin.frames_by_track.items():
        first = frames.iloc[0]
        out[int(tid)] = (int(first["t"]), float(first[col]))
    return out


def mother_branch(lin: Any, start: Optional[int] = None, follow: Optional[Callable] = None) -> TrackSet:
    """The mother lineage: follow the closed-end daughter at each division

    With ``start=None`` it begins at the cell nearest the closed end in the
    earliest frame. ``follow`` overrides the division-following rule (default:
    :func:`_closed_daughter`). 
    """
    follow = follow or _closed_daughter
    births = _birth_positions(lin)
    children = lin.child_map

    if start is None:
        t0 = min(bt for bt, _ in births.values())
        candidates = [(tid, pos) for tid, (bt, pos) in births.items() if bt == t0]
        chooser = min if lin.open_end == "high" else max
        start = chooser(candidates, key=lambda kv: kv[1])[0]

    branch: list[int] = []
    seen: set[int] = set()
    cur: Optional[int] = int(start)
    while cur is not None and cur not in seen and cur in births:
        branch.append(cur)
        seen.add(cur)
        kids = children.get(cur)
        if not kids:
            break
        d1, d2 = kids
        cur = follow(d1, d2, _pos(births, d1), _pos(births, d2), lin.open_end)
    return TrackSet(lin.dataset_id, frozenset(branch), "mother_branch")


def _pos(births: dict[int, tuple[int, float]], track_id: int) -> Optional[float]:
    entry = births.get(track_id)
    return None if entry is None else entry[1]


def descendants_of(lin: Any, track_id: int, *, inclusive: bool = True) -> TrackSet:
    """The subtree rooted at ``track_id`` (inclusive by default)"""
    children = lin.child_map
    found: set[int] = set()
    queue = deque([int(track_id)])
    while queue:
        tid = queue.popleft()
        if tid in found:
            continue
        found.add(tid)
        queue.extend(children.get(tid, ()))
    if not inclusive:
        found.discard(int(track_id))
    return TrackSet(lin.dataset_id, frozenset(found), f"descendants_of({track_id})")


def ancestors_of(lin: Any, track_id: int, *, inclusive: bool = False) -> TrackSet:
    """The chain of mothers above ``track_id`` (exclusive by default)."""
    parent = lin.parent_map
    found: set[int] = {int(track_id)} if inclusive else set()
    cur = int(track_id)
    while cur in parent:
        cur = parent[cur]
        found.add(cur)
    return TrackSet(lin.dataset_id, frozenset(found), f"ancestors_of({track_id})")


def path_between(lin: Any, start: int, end: int) -> TrackSet:
    """The lineage path between two tracks (inclusive) — one branch of the tree.

    Order-independent: whichever of ``start``/``end`` is the ancestor, the result is
    the unique chain of ``track_id``\\ s running from it down to the other. Returns an
    **empty** ``TrackSet`` if the two are not on one ancestor→descendant line (or
    either id is absent), so a caller can detect "not a single branch".
    """
    a, b = int(start), int(end)
    name = f"path_between({start}, {end})"
    frames = lin.frames_by_track
    if a not in frames or b not in frames:
        return TrackSet(lin.dataset_id, frozenset(), name)

    parent = lin.parent_map
    for lo, hi in ((a, b), (b, a)):  # try hi as the descendant of lo, then the reverse
        chain = [hi]
        cur = hi
        while cur != lo and cur in parent:
            cur = parent[cur]
            chain.append(cur)
        if cur == lo:
            return TrackSet(lin.dataset_id, frozenset(chain), name)
    return TrackSet(lin.dataset_id, frozenset(), name)


def roots(lin: Any) -> TrackSet:
    """Tracks with no observed parent (true origins and re-init roots)"""
    ids = set(lin.frames_by_track) - set(lin.parent_map)
    return TrackSet(lin.dataset_id, frozenset(ids), "roots")


def leaves(lin: Any) -> TrackSet:
    """Tracks that never divide"""
    ids = set(lin.frames_by_track) - set(lin.child_map)
    return TrackSet(lin.dataset_id, frozenset(ids), "leaves")


def generation(lin: Any, n: int) -> TrackSet:
    """Tracks at generation ``n`` (0 at every root)"""
    cyc = lin.cycles
    ids = cyc.loc[cyc["generation"] == n, "track_id"]
    return TrackSet(lin.dataset_id, frozenset(int(t) for t in ids), f"generation({n})")


def filter_cycles(lin: Any, predicate: Callable[[Any], Any], name: str = "filter_cycles") -> TrackSet:
    """Tracks whose cycles rows satisfy ``predicate(cycles_df) -> boolean mask``"""
    cyc = lin.cycles
    ids = cyc.loc[predicate(cyc), "track_id"]
    return TrackSet(lin.dataset_id, frozenset(int(t) for t in ids), name)
