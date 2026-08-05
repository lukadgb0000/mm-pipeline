"""The ``Lineage`` model and its derived cell-cycle table

A ``Lineage`` wraps the three reconstruction tables for one dataset plus the
context needed to interpret them (axis, open end, frame interval, label dir).
Its keystone is :attr:`Lineage.cycles`: one row per ``track_id`` carrying the lineage structure that the
selection, metric, and plotting layers all build on.

The model is source-agnostic: a lineage from ``track-select`` (clean KEEP-only)
and one from ``modelvio`` (drop/bridge) are the same kind of object. Two things a
clean run usually lacks are represented as ordinary columns rather than special
cases: tracks with time gaps (``n_gap_frames`` > 0, from bridges) and roots whose
birth was not observed (``birth_observed`` is False, from frame 0, a
drop-not-bridged boundary, or an empty frame)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

# One row per track_id; see module docstring for the column contract.
CYCLE_COLUMNS = [
    "dataset_id",
    "track_id",
    "parent_id",
    "generation",
    "birth_t",
    "end_t",
    "n_frames",
    "n_gap_frames",
    "birth_observed",
    "end_cause",
    "complete_cycle",
]


@dataclass(frozen=True, eq=False)
class Lineage:
    """One dataset's reconstructed lineage

    ``eq=False`` because the default dataclass ``__eq__`` would compare DataFrame
    fields (``df == df`` raises) and the frozen ``__hash__`` cannot hash them;
    identity/hash therefore fall back to ``object``. ``cached_property`` writes
    through ``__dict__`` so it works despite ``frozen=True`` — but adding
    ``slots=True`` later would remove ``__dict__`` and break it
    """

    dataset_id: str
    tracks_df: Any
    divisions_df: Any
    events_df: Any
    axis: str = "y"
    open_end: str = "high"
    frame_interval_min: Optional[float] = None
    labels_dir: Optional[Path] = None

    @classmethod
    def from_result(cls, result: Any, spec: Any) -> "Lineage":
        """Build from an in-memory ``TrackSelectResult`` (the notebook path)."""
        did = spec.dataset_id
        if did not in result.tracks_by_dataset:
            raise KeyError(f"Dataset {did!r} not in result (have {sorted(result.tracks_by_dataset)}).")
        return cls(
            dataset_id=did,
            tracks_df=result.tracks_by_dataset[did],
            divisions_df=result.divisions_by_dataset[did],
            events_df=result.events_by_dataset[did],
            axis=spec.axis,
            open_end=spec.open_end,
            frame_interval_min=spec.frame_interval_min,
            labels_dir=spec.effective_labels_dir,
        )

    @classmethod
    def from_run(cls, run_dir: str | Path, spec: Any) -> "Lineage":
        """Build from a persisted run directory ``<run_dir>/<dataset_id>/``."""
        import pandas as pd

        dataset_dir = Path(run_dir) / spec.dataset_id
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"No dataset directory {dataset_dir!s} in run {run_dir!s}.")
        return cls(
            dataset_id=spec.dataset_id,
            tracks_df=pd.read_csv(dataset_dir / "tracks.csv"),
            divisions_df=pd.read_csv(dataset_dir / "division_events.csv"),
            events_df=pd.read_csv(dataset_dir / "events.csv"),
            axis=spec.axis,
            open_end=spec.open_end,
            frame_interval_min=spec.frame_interval_min,
            labels_dir=spec.effective_labels_dir,
        )

    @property
    def axis_col(self) -> str:
        """The ``tracks_df`` column holding position along the trench axis.

        Selection and plotting read this instead
        of hardcoding ``"y"``, so ``axis="x"`` needs no special-casing
        """
        return "y" if self.axis == "y" else "x"

    @cached_property
    def parent_map(self) -> dict[int, int]:
        """``{daughter_track_id: mother_track_id}`` from the division table"""
        parent: dict[int, int] = {}
        for _, row in self.divisions_df.iterrows():
            mother = int(row["mother_track_id"])
            parent[int(row["d1_track_id"])] = mother
            parent[int(row["d2_track_id"])] = mother
        return parent

    @cached_property
    def child_map(self) -> dict[int, tuple[int, int]]:
        """``{mother_track_id: (d1, d2)}`` from the division table"""
        return {
            int(row["mother_track_id"]): (int(row["d1_track_id"]), int(row["d2_track_id"]))
            for _, row in self.divisions_df.iterrows()
        }

    @cached_property
    def frames_by_track(self) -> dict[int, Any]:
        """``{track_id: time-sorted tracks_df slice}``, computed once

        Lets per-track consumers (metrics) look up a slice in O(1) instead of
        rescanning ``tracks_df`` per track
        """
        return {
            int(tid): sub.sort_values("t").reset_index(drop=True)
            for tid, sub in self.tracks_df.groupby("track_id")
        }

    @cached_property
    def cycles(self) -> Any:
        """One row per ``track_id``; see :data:`CYCLE_COLUMNS`."""
        import pandas as pd

        if self.tracks_df.empty:
            return pd.DataFrame(columns=CYCLE_COLUMNS)

        # Observed span per track
        cyc = (
            self.tracks_df.groupby("track_id")["t"]
            .agg(birth_t="min", end_t="max", n_frames="count")
            .reset_index()
        )
        cyc["n_gap_frames"] = (cyc["end_t"] - cyc["birth_t"] + 1) - cyc["n_frames"]

        # The division table is the only place the tree structure is explicit
        parent, children = self.parent_map, self.child_map
        cyc["parent_id"] = cyc["track_id"].map(parent).astype("Int64")
        cyc["birth_observed"] = cyc["track_id"].isin(parent)  # appeared as a daughter

        # end_cause: divided (authoritative from divisions) > exited > censored
        exited = self._exited_track_ids()
        cyc["end_cause"] = "censored"
        cyc.loc[cyc["track_id"].isin(exited), "end_cause"] = "exited"
        cyc.loc[cyc["track_id"].isin(children), "end_cause"] = "divided"

        cyc["generation"] = cyc["track_id"].map(_generations(cyc["track_id"], parent, children)).astype("Int64")

        cyc["complete_cycle"] = (
            cyc["birth_observed"]
            & (cyc["end_cause"] == "divided")
            & (cyc["n_gap_frames"] == 0)
        )
        cyc["dataset_id"] = self.dataset_id
        return cyc[CYCLE_COLUMNS].sort_values("track_id").reset_index(drop=True)

    def _exited_track_ids(self) -> set[int]:
        """Track IDs that terminate on an exit event (ignoring the -1 sentinel)"""
        events = self.events_df
        if events.empty:
            return set()
        exit_rows = events.loc[events["event"] == "exit", "track_id"]
        return {int(tid) for tid in exit_rows if int(tid) >= 0}


def _generations(track_ids: Any, parent: dict[int, int], children: dict[int, list[int]]) -> dict[int, int]:
    """BFS generation numbers: 0 at every root, +1 per division

    A root is any track with no parent — a true frame-0 origin *or* a
    birth-unobserved re-init root; both legitimately start their own count
    """
    roots = [int(tid) for tid in track_ids if int(tid) not in parent]
    generation: dict[int, int] = {}
    queue = deque((root, 0) for root in roots)
    while queue:
        tid, gen = queue.popleft()
        generation[tid] = gen
        for child in children.get(tid, ()):
            queue.append((child, gen + 1))
    return generation
