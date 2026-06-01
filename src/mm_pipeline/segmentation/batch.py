"""Batch segmentation runner helpers"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from mm_pipeline.config import SegmentationConfig, SegmentationRunArtifact

from .base import SegmenterBackend


def run_segmentation(
    backend: SegmenterBackend,
    image_paths: Sequence[str | Path],
    output_dir: str | Path,
    config: SegmentationConfig,
    *,
    dataset_id: str,
) -> SegmentationRunArtifact:
    return backend.segment_images(image_paths, output_dir, config, dataset_id=dataset_id)
