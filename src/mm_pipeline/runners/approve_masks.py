"""Runner for mm-pipeline approve-masks.

Launches napari for interactive label review and writes edited labels to
``out_dir`` on save. Unlike the other runners, ``out_dir`` is required — there is no notebook-friendly
"in-memory" mode because the napari workflow inherently produces files. Maybe I should change that
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mm_pipeline.segmentation_qa.review import review_and_approve_masks

from ._outputs import make_run_metadata


@dataclass(frozen=True)
class ApproveMasksResult:
    """In-memory result of a ``run_approve_masks`` invocation."""

    output_dir: Path
    dataset_id: str
    resolved_config: dict[str, Any] = field(default_factory=dict)


def run_approve_masks(
    *,
    images_dir: str | Path,
    labels_dir: str | Path,
    out_dir: str | Path | None = None,
    overwrite: bool = False,
    dataset_id: str = "",
    source_segmentation_run: str | Path | None = None,
    qa_report_path: str | Path | None = None,
) -> ApproveMasksResult:
    """Launch napari to review and approve masks for one dataset.

    Wraps :func:`mm_pipeline.segmentation_qa.review.review_and_approve_masks`.
    If ``out_dir`` is omitted, edited labels are written to a sibling
    directory named ``<labels_dir>_edited``.

    A ``run_metadata.json`` is written into the output directory after
    napari saves the approved labels. The user may close napari without
    saving, in which case the output directory may be empty or missing —
    the runner still returns the resolved output path.
    """

    output_dir = review_and_approve_masks(
        labels_dir,
        images_dir,
        out_dir,
        overwrite=overwrite,
        dataset_id=dataset_id,
        source_segmentation_run=source_segmentation_run,
        qa_report_path=qa_report_path,
    )

    metadata = make_run_metadata(
        command="approve-masks",
        manifest_path=None,
        resolved_config={
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "overwrite": overwrite,
            "source_segmentation_run": str(source_segmentation_run) if source_segmentation_run else None,
            "qa_report_path": str(qa_report_path) if qa_report_path else None,
        },
        dataset_ids=[dataset_id] if dataset_id else [],
    )

    if output_dir.exists():
        run_json = output_dir / "run_metadata.json"
        with run_json.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
            fh.write("\n")

    return ApproveMasksResult(
        output_dir=output_dir,
        dataset_id=dataset_id,
        resolved_config=dict(metadata["resolved_config"]),
    )


__all__ = ["ApproveMasksResult", "run_approve_masks"]
