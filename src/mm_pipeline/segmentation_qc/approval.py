"""Approved-label export helpers"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from mm_pipeline.config import ApprovedLabelStack
from mm_pipeline.io.labels import save_label_stack


def save_approved_labels(
    labels: np.ndarray,
    file_names: Iterable[str | Path],
    out_dir: str | Path,
    *,
    dataset_id: str = "",
    overwrite: bool = False,
    source_segmentation_run: Optional[str | Path] = None,
    qa_report_path: Optional[str | Path] = None,
) -> ApprovedLabelStack:
    """Save reviewed labels and return the approved-label contract"""

    save_label_stack(labels, file_names, out_dir, overwrite=overwrite)
    return ApprovedLabelStack(
        dataset_id=dataset_id,
        labels_dir=Path(out_dir),
        source_segmentation_run=None if source_segmentation_run is None else Path(source_segmentation_run),
        qa_report_path=None if qa_report_path is None else Path(qa_report_path),
    )
