from pathlib import Path

import pytest

from mm_pipeline.config import SegmentationConfig
from mm_pipeline.io.labels import save_label_stack
from mm_pipeline.segmentation import PrecomputedLabelsBackend, validate_label_directory, validate_label_stack


np = pytest.importorskip("numpy")
pytest.importorskip("tifffile")


def _label_stack() -> np.ndarray:
    labels = np.zeros((2, 5, 5), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 2:4, 2:4] = 2
    return labels


def test_validate_label_stack_accepts_valid_integer_stack():
    result = validate_label_stack(_label_stack())

    assert result.is_valid
    assert result.frame_count == 2
    assert result.frame_shape == (5, 5)
    assert result.errors == ()


def test_validate_label_stack_rejects_empty_frames():
    labels = _label_stack()
    labels[1] = 0

    result = validate_label_stack(labels)

    assert not result.is_valid
    assert "empty label frames found" in result.errors[0]


def test_precomputed_backend_validates_label_directory_and_writes_metadata(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    out_dir = tmp_path / "artifact"
    save_label_stack(_label_stack(), ["frame1.tif", "frame2.tif"], labels_dir)

    validation = validate_label_directory(labels_dir)
    assert validation.is_valid

    backend = PrecomputedLabelsBackend(labels_dir)
    artifact = backend.segment_images([], out_dir, SegmentationConfig(backend="precomputed"), dataset_id="trench_a")

    assert artifact.backend == "precomputed"
    assert artifact.label_tifs_dir == labels_dir
    assert artifact.label_count == 2
    assert artifact.frame_shape == (5, 5)
    assert (out_dir / "segmentation_run.json").exists()
