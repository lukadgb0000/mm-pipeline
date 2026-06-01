"""Segmentation backend interfaces and simple adapters"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence

from mm_pipeline.config import SegmentationConfig, SegmentationRunArtifact
from mm_pipeline.io.artifacts import write_json_artifact

from .validation import validate_label_directory


class SegmenterBackend(Protocol):
    name: str

    def segment_images(
        self,
        image_paths: Sequence[str | Path],
        output_dir: str | Path,
        config: SegmentationConfig,
        *,
        dataset_id: str,
    ) -> SegmentationRunArtifact:
        """Segment images and return a run artifact."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_segmentation_metadata(artifact: SegmentationRunArtifact, output_dir: str | Path) -> Path:
    return write_json_artifact(artifact, Path(output_dir) / "segmentation_run.json")


class PrecomputedLabelsBackend:
    """Adapter for existing label TIFF folders

    This is not a segmentation model. It validates already-produced labels and
    returns the same artifact contract as a real segmenter.
    """

    name = "precomputed"

    def __init__(self, labels_dir: str | Path, *, allow_empty_frames: bool = False):
        self.labels_dir = Path(labels_dir)
        self.allow_empty_frames = allow_empty_frames

    def segment_images(
        self,
        image_paths: Sequence[str | Path],
        output_dir: str | Path,
        config: SegmentationConfig,
        *,
        dataset_id: str,
    ) -> SegmentationRunArtifact:
        validation = validate_label_directory(self.labels_dir, allow_empty_frames=self.allow_empty_frames)
        validation.raise_for_errors()
        artifact = SegmentationRunArtifact(
            dataset_id=dataset_id,
            backend=self.name,
            label_tifs_dir=self.labels_dir,
            raw_images_dir=Path(image_paths[0]).parent if image_paths else None,
            model_type=None,
            config=config.to_dict(),
            image_count=len(image_paths),
            label_count=validation.frame_count,
            frame_shape=validation.frame_shape,
            created_at=utc_now_iso(),
            metadata={"validation": validation.__dict__},
        )
        write_segmentation_metadata(artifact, output_dir)
        return artifact
