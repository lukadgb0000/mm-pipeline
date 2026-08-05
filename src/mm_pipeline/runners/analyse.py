"""Runner for mm-pipeline analyse

Consumes a reconstructed run (``<out>/<run_tag>`` from track-select / modelvio)
and emits the biology-facing artifacts: ``cycles.csv`` (plus any requested cycle
metrics), and ``swimlane.png`` / ``dendrogram.png``. The notebook API is primary
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from mm_pipeline.io.manifests import load_dataset_manifest

from ._outputs import make_run_metadata, write_multifile_outputs


@dataclass(frozen=True)
class AnalyseResult:
    """In-memory result of a run_analyse invocation"""

    cycles: Any
    output_dir: Path | None = None


def run_analyse(
    run_dir: str | Path,
    dataset: str,
    manifest: str | Path,
    *,
    metric_names: Sequence[str] = (),
    out_dir: str | Path | None = None,
    overwrite: bool = False,
) -> AnalyseResult:
    """Build a Lineage from a run and emit cycles (+ metrics) and the two plots

    ``run_dir`` is the ``<out>/<run_tag>`` directory (dataset id appended by
    ``Lineage.from_run``). ``manifest`` supplies the ``DatasetSpec`` the library
    constructors require. Requested metrics are computed over every track; the
    plots highlight the mother branch. Outputs go to ``out_dir`` (default
    ``<run_dir>/<dataset>/analysis``).
    """
    from matplotlib.figure import Figure

    from mm_pipeline.analysis import (
        Lineage,
        TrackSet,
        metrics,
        mother_branch,
        plot_dendrogram,
        plot_swimlane,
    )
    from mm_pipeline.analysis.metrics import PROPERTY_METRICS

    spec_by_id = {spec.dataset_id: spec for spec in load_dataset_manifest(manifest)}
    if dataset not in spec_by_id:
        raise ValueError(f"Dataset {dataset!r} not in manifest (have {sorted(spec_by_id)}).")
    lin = Lineage.from_run(run_dir, spec_by_id[dataset])

    names = tuple(metric_names)
    if names:
        all_tracks = TrackSet(lin.dataset_id, frozenset(lin.frames_by_track), "all")
        needs_props = any(name in PROPERTY_METRICS for name in names)
        cycles_df = metrics(lin, all_tracks, names, with_properties=needs_props)
    else:
        cycles_df = lin.cycles

    write_dir = Path(out_dir) if out_dir is not None else Path(run_dir) / dataset / "analysis"
    write_dir.mkdir(parents=True, exist_ok=True)

    branch = mother_branch(lin)
    for name, plotter, size in (
        ("swimlane", plot_swimlane, (12, 5)),
        ("dendrogram", plot_dendrogram, (12, 6)),
    ):
        fig = Figure(figsize=size)
        plotter(lin, highlight=branch, ax=fig.subplots())
        fig.savefig(write_dir / f"{name}.png", dpi=150, bbox_inches="tight")

    metadata = make_run_metadata(
        command="analyse",
        manifest_path=manifest,
        resolved_config={"metrics": list(names)},
        dataset_ids=[dataset],
    )
    summary = {
        "n_tracks": len(lin.frames_by_track),
        "n_divisions": len(lin.child_map),
        "n_complete_cycles": int(lin.cycles["complete_cycle"].sum()),
        "metrics": list(names),
    }
    write_multifile_outputs(
        out_dir=write_dir,
        summary=summary,
        metadata=metadata,
        tables={"cycles": cycles_df},
        title="analyse run",
        overwrite=overwrite,
    )
    return AnalyseResult(cycles=cycles_df, output_dir=write_dir)


__all__ = ["AnalyseResult", "run_analyse"]
