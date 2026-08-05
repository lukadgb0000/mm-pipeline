"""Interactive branch selection on a rendered swimlane or dendrogram.

:func:`pick_tracks` attaches a matplotlib pick handler to the per-track lines
(tagged by :mod:`~mm_pipeline.analysis.plotting` with ``set_gid(track_id)``).
A branch is chosen with **two clicks — a start and an end**: the selection is the
unique lineage path between them (:func:`~mm_pipeline.analysis.selection.path_between`),
so it is a single chain ready for time-series analysis, not a subtree.

The plot updates as you go: the first click highlights the start track; the second
highlights the whole branch and stores it as ``selector.selection``. The click also
**prints the equivalent declarative call** (``path_between(lin, 3, 41)``), so a
selection made by clicking stays reproducible.

Works identically on the swimlane and the dendrogram (same tagging), so you can
select on whichever view reads best. Requires an interactive backend
(``%matplotlib widget`` / ``ipympl``); the static plots and the declarative
selectors work without one.

This module pulls in no matplotlib — it operates on the ``Axes`` the caller already
built (only the colour constants come from :mod:`~mm_pipeline.analysis.plotting`).
"""

from __future__ import annotations

from typing import Any, Optional

from .plotting import BASE_COLOR, HIGHLIGHT_COLOR
from .selection import TrackSet, path_between


class TrackSelector:
    """Two-click branch picker on one ``Axes``; ``.selection`` is the latest branch.

    Click the start of a branch, then its end (e.g. a leaf): the selection becomes
    the lineage path between them. Clicks that aren't on one lineage restart from the
    new click. ``.history`` keeps every completed branch.
    """

    def __init__(self, lin: Any, ax: Any, *, echo: bool = True) -> None:
        self.lineage = lin
        self.ax = ax
        self.echo = echo
        self.selection: Optional[TrackSet] = None
        self.history: list[TrackSet] = []
        self._start: Optional[int] = None
        self._cid = ax.figure.canvas.mpl_connect("pick_event", self._on_pick)

    def _on_pick(self, event: Any) -> None:
        gid = event.artist.get_gid()
        if gid is None or int(gid) not in self.lineage.frames_by_track:
            return  # a connector, legend proxy, or stray artist — not a track
        tid = int(gid)

        if self._start is None:
            self._start = tid
            self._render(frozenset({tid}))
            if self.echo:
                print(f"start = {tid}; click the branch end (a descendant or ancestor)")
            return

        branch = path_between(self.lineage, self._start, tid)
        if not branch:
            if self.echo:
                print(f"{tid} is not on one lineage with {self._start}; restarting at {tid}")
            self._start = tid
            self._render(frozenset({tid}))
            return

        self.selection = branch
        self.history.append(branch)
        self._render(branch.track_ids)
        if self.echo:
            print(f"path_between(lin, {self._start}, {tid})")
        self._start = None

    def _render(self, ids: Any) -> None:
        """Restyle the tagged track lines so the current selection is visible."""
        for line in self.ax.get_lines():
            gid = line.get_gid()
            if gid is None:
                continue
            hot = int(gid) in ids
            line.set_color(HIGHLIGHT_COLOR if hot else BASE_COLOR)
            line.set_linewidth(2.0 if hot else 0.6)
            line.set_alpha(0.95 if hot else 0.5)
            line.set_zorder(5 if hot else 1)
        self.ax.figure.canvas.draw_idle()

    def disconnect(self) -> None:
        """Detach the pick handler from the canvas."""
        self.ax.figure.canvas.mpl_disconnect(self._cid)


def pick_tracks(lin: Any, ax: Any, *, echo: bool = True) -> TrackSelector:
    """Enable two-click branch picking on the tagged lines of ``ax``.

    ``ax`` is a swimlane or dendrogram ``Axes``. Only the gid-bearing per-track
    lines are made pickable; division connectors and legend proxies are left alone.
    Returns a :class:`TrackSelector` whose ``.selection`` is the picked branch.
    """
    for line in ax.get_lines():
        if line.get_gid() is not None:
            line.set_picker(True)
            line.set_pickradius(5)
    return TrackSelector(lin, ax, echo=echo)
