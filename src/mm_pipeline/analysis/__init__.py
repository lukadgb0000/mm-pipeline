"""Lineage analysis and plotting.

Entirely downstream of tracking output. Reads the reconstructed lineage tables
(``tracks``/``events``/``divisions``) plus the dataset's ``DatasetSpec``. Does not
import into ``tracking``/``features``/``core``.
"""

from __future__ import annotations

from .consistency import division_length_consistency
from .metrics import (
    PROPERTY_METRICS,
    CycleContext,
    cycle_metric,
    get_cycle_metric,
    list_cycle_metrics,
    metrics,
)
from .picking import TrackSelector, pick_tracks
from .plotting import plot_dendrogram, plot_property_series, plot_swimlane
from .properties import cell_properties
from .selection import (
    TrackSet,
    ancestors_of,
    descendants_of,
    filter_cycles,
    generation,
    leaves,
    mother_branch,
    path_between,
    roots,
)
from .tree import Lineage

__all__ = [
    "PROPERTY_METRICS",
    "CycleContext",
    "Lineage",
    "TrackSelector",
    "TrackSet",
    "ancestors_of",
    "cell_properties",
    "cycle_metric",
    "descendants_of",
    "division_length_consistency",
    "filter_cycles",
    "generation",
    "get_cycle_metric",
    "leaves",
    "list_cycle_metrics",
    "metrics",
    "mother_branch",
    "path_between",
    "pick_tracks",
    "plot_dendrogram",
    "plot_property_series",
    "plot_swimlane",
    "roots",
]
