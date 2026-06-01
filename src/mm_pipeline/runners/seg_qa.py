"""Runner for ``mm-pipeline seg-qa``.

Runs the headless segmentation QA checks over each dataset's labels
directory and writes a findings CSV per dataset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from mm_pipeline.config import DatasetSpec, SegmentationQAConfig, SegmentationQAFinding
from mm_pipeline.io.labels import load_labels_from_folder
from mm_pipeline.io.manifests import load_dataset_manifest
from mm_pipeline.segmentation_qa.checks import run_basic_checks
from mm_pipeline.segmentation_qa.reports import write_qa_report_csv

from ._outputs import (
    make_run_metadata,
    resolve_run_tag,
    write_multifile_outputs,
)


@dataclass(frozen=True)
class SegQAResult:
    """In-memory result of a ``run_seg_qa`` invocation."""

    findings_by_dataset: dict[str, list[SegmentationQAFinding]]
    resolved_config: dict[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None


def _normalise_specs(
    datasets: DatasetSpec | Sequence[DatasetSpec] | str | Path,
) -> tuple[list[DatasetSpec], str | None]:
    if isinstance(datasets, (str, Path)):
        return load_dataset_manifest(datasets), str(datasets)
    if isinstance(datasets, DatasetSpec):
        return [datasets], None
    specs = list(datasets)
    if not specs:
        raise ValueError("datasets must be non-empty.")
    for spec in specs:
        if not isinstance(spec, DatasetSpec):
            raise TypeError(
                f"datasets entries must be DatasetSpec; got {type(spec).__name__}."
            )
    return specs, None


def run_seg_qa(
    datasets: DatasetSpec | Sequence[DatasetSpec] | str | Path,
    *,
    config: SegmentationQAConfig | None = None,
    out_dir: str | Path | None = None,
    run_tag: str | None = None,
    overwrite: bool = False,
) -> SegQAResult:
    """Run headless segmentation QA checks over a manifest of datasets.

    For each dataset the function loads labels from
    ``dataset.effective_labels_dir`` (preferring approved labels), runs the
    basic per-frame checks ([segmentation_qa.checks.run_basic_checks]),
    and writes a findings CSV when ``out_dir`` is given.

    Parameters
   
    datasets : a DatasetSpec, a list of them, or a path to a CSV manifest
    config : optional SegmentationQAConfig, defaults to the package defaults
    out_dir : optional output directory. When None, no files are written
    run_tag 
    overwrite : if True, allows clobbering an existing run_tag directory
    """

    specs, manifest_path = _normalise_specs(datasets)
    resolved_config = config or SegmentationQAConfig()

    findings_by_dataset: dict[str, list[SegmentationQAFinding]] = {}
    write_dir: Path | None = None
    if out_dir is not None:
        write_dir = Path(out_dir) / resolve_run_tag(run_tag)
        write_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        labels_dir = spec.effective_labels_dir
        if labels_dir is None:
            raise ValueError(
                f"Dataset {spec.dataset_id!r} has no labels directory. "
                "seg-qa requires approved_labels_dir or labels_dir; "
                "run 'mm-pipeline segment' first."
            )
        labels = load_labels_from_folder(labels_dir)
        findings = run_basic_checks(labels, spec.dataset_id, resolved_config)
        findings_by_dataset[spec.dataset_id] = findings

        if write_dir is not None:
            dataset_dir = write_dir / spec.dataset_id
            dataset_dir.mkdir(parents=True, exist_ok=True)
            write_qa_report_csv(findings, dataset_dir / "seg_qa_findings.csv")

    output_dir: Path | None = None
    if write_dir is not None:
        metadata = make_run_metadata(
            command="seg-qa",
            manifest_path=manifest_path,
            resolved_config=resolved_config.to_dict(),
            dataset_ids=[spec.dataset_id for spec in specs],
        )
        n_findings = sum(len(v) for v in findings_by_dataset.values())
        summary = {
            "n_datasets": len(specs),
            "n_findings_total": n_findings,
            "n_findings_per_dataset": {
                k: len(v) for k, v in findings_by_dataset.items()
            },
        }
        write_multifile_outputs(
            out_dir=write_dir,
            summary=summary,
            metadata=metadata,
            title="Segmentation QA",
            overwrite=overwrite,
        )
        output_dir = write_dir

    return SegQAResult(
        findings_by_dataset=findings_by_dataset,
        resolved_config=resolved_config.to_dict(),
        output_dir=output_dir,
    )


__all__ = ["SegQAResult", "run_seg_qa"]
