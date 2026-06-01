"""Runner for ``mm-pipeline segment``

Runs the configured segmentation backend over each
raw-image dataset in the manifest, writing label TIFFs + overlays + a
per-dataset run artefact
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from mm_pipeline.config import RawImageDatasetSpec, SegmentationConfig, SegmentationRunArtifact
from mm_pipeline.io.images import collect_image_paths
from mm_pipeline.io.manifests import load_raw_image_manifest
from mm_pipeline.segmentation.base import PrecomputedLabelsBackend, SegmenterBackend
from mm_pipeline.segmentation.batch import run_segmentation

from ._outputs import (
    make_run_metadata,
    resolve_run_tag,
    write_multifile_outputs,
)


@dataclass(frozen=True)
class SegmentResult:
    """In-memory result of a ``run_segment`` invocation."""

    artefacts_by_dataset: dict[str, SegmentationRunArtifact]
    resolved_config: dict[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None


def _normalise_specs(
    datasets: RawImageDatasetSpec | Sequence[RawImageDatasetSpec] | str | Path,
) -> tuple[list[RawImageDatasetSpec], str | None]:
    if isinstance(datasets, (str, Path)):
        return load_raw_image_manifest(datasets), str(datasets)
    if isinstance(datasets, RawImageDatasetSpec):
        return [datasets], None
    specs = list(datasets)
    if not specs:
        raise ValueError("datasets must be non-empty.")
    for spec in specs:
        if not isinstance(spec, RawImageDatasetSpec):
            raise TypeError(
                f"datasets entries must be RawImageDatasetSpec; got {type(spec).__name__}."
            )
    return specs, None


def _resolve_backend(backend: str | SegmenterBackend) -> SegmenterBackend:
    if not isinstance(backend, str):
        return backend
    if backend == "cpsam":
        from mm_pipeline.segmentation.cpsam import CPSAMBackend
        return CPSAMBackend()
    if backend == "precomputed":
        raise ValueError(
            "precomputed backend requires a labels_dir; pass a "
            "PrecomputedLabelsBackend instance directly."
        )
    raise ValueError(f"Unknown segmentation backend: {backend!r}")


def run_segment(
    datasets: RawImageDatasetSpec | Sequence[RawImageDatasetSpec] | str | Path,
    *,
    backend: str | SegmenterBackend = "cpsam",
    config: SegmentationConfig | None = None,
    out_dir: str | Path | None = None,
    run_tag: str | None = None,
    overwrite: bool = False,
) -> SegmentResult:
    """Run segmentation over a manifest of raw-image datasets.

    Parameters
    ----------
    datasets : a RawImageDatasetSpec, a list of them, or a path to a CSV
        manifest. When a path is given, the manifest is loaded internally.
    backend : either ``"cpsam"`` (resolved to a CPSAMBackend instance) or a
        SegmenterBackend instance (useful for tests with PrecomputedLabelsBackend).
    config : optional SegmentationConfig. Defaults to SegmentationConfig().
    out_dir : optional output directory. When None, no files are written and
        ``result.output_dir`` is None (notebook-friendly mode).
    run_tag : optional run tag (default: UTC timestamp).
    overwrite : if True, allows clobbering an existing run_tag directory.
    """

    specs, manifest_path = _normalise_specs(datasets)
    resolved_config = config or SegmentationConfig()
    backend_obj = _resolve_backend(backend)
    if backend == "cpsam" and resolved_config.backend != "cpsam":
        # Keep the resolved config consistent with the actual backend in use.
        pass

    artefacts: dict[str, SegmentationRunArtifact] = {}
    write_dir: Path | None = None
    if out_dir is not None:
        write_dir = Path(out_dir) / resolve_run_tag(run_tag)

    for spec in specs:
        image_paths = collect_image_paths(spec.images_dir, spec.image_pattern)
        if write_dir is not None:
            dataset_out = write_dir / spec.dataset_id
            artefact = run_segmentation(
                backend_obj, image_paths, dataset_out, resolved_config,
                dataset_id=spec.dataset_id,
            )
        else:
            # In notebook mode we still need a temporary place for the
            # backend to write label TIFFs — but most backends require a
            # writable output_dir. Use a tmp directory.
            import tempfile

            with tempfile.TemporaryDirectory(prefix=f"mm_segment_{spec.dataset_id}_") as tmp:
                artefact = run_segmentation(
                    backend_obj, image_paths, tmp, resolved_config,
                    dataset_id=spec.dataset_id,
                )
        artefacts[spec.dataset_id] = artefact

    output_dir: Path | None = None
    if write_dir is not None:
        metadata = make_run_metadata(
            command="segment",
            manifest_path=manifest_path,
            resolved_config=resolved_config.to_dict(),
            dataset_ids=[spec.dataset_id for spec in specs],
        )
        summary = {
            "n_datasets": len(specs),
            "backend": backend if isinstance(backend, str) else backend_obj.name,
        }
        write_multifile_outputs(
            out_dir=write_dir,
            summary=summary,
            metadata=metadata,
            title="Segmentation run",
            overwrite=overwrite,
        )
        output_dir = write_dir

    return SegmentResult(
        artefacts_by_dataset=artefacts,
        resolved_config=resolved_config.to_dict(),
        output_dir=output_dir,
    )


__all__ = ["SegmentResult", "run_segment"]
