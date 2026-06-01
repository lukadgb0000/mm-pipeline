"""Tests for mm-pipeline segment and run_segment"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_pipeline.cli.main import main
from mm_pipeline.config import RawImageDatasetSpec, SegmentationConfig
from mm_pipeline.runners.segment import SegmentResult, run_segment
from mm_pipeline.segmentation.base import PrecomputedLabelsBackend


def _np():
    return pytest.importorskip("numpy")


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, RawImageDatasetSpec]:
    """Create a tiny images dir and labels dir for one synthetic dataset."""

    np = _np()
    tiff = pytest.importorskip("tifffile")

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    # Two frames: simple 8x8 stacks with one labelled cell each.
    for t in range(2):
        img = np.zeros((8, 8), dtype=np.uint8)
        img[1:3, 1:3] = 200
        tiff.imwrite(images_dir / f"frame_{t:03d}.tif", img)

        labels = np.zeros((8, 8), dtype=np.uint32)
        labels[1 + t : 3 + t, 1:3] = 1
        tiff.imwrite(labels_dir / f"frame_{t:03d}.tif", labels)

    spec = RawImageDatasetSpec(
        dataset_id="synth_a",
        images_dir=images_dir,
    )
    return images_dir, labels_dir, spec


def test_cli_segment_help_succeeds(capsys):
    try:
        main(["segment", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "segmentation backend" in captured.out.lower()
    assert "--backend" in captured.out


def test_run_segment_with_precomputed_backend_writes_outputs(tmp_path: Path):
    images_dir, labels_dir, spec = _build_fixture(tmp_path)
    out_dir = tmp_path / "seg_run"

    result = run_segment(
        spec,
        backend=PrecomputedLabelsBackend(labels_dir=labels_dir),
        out_dir=out_dir,
        run_tag="v1",
    )

    assert isinstance(result, SegmentResult)
    assert result.output_dir == out_dir / "v1"
    assert (out_dir / "v1" / "summary.json").exists()
    summary = json.loads((out_dir / "v1" / "summary.json").read_text())
    assert summary["command"] == "segment"
    assert summary["n_datasets"] == 1
    assert summary["dataset_ids"] == ["synth_a"]
    assert "synth_a" in result.artefacts_by_dataset


def test_run_segment_overwrite_protection(tmp_path: Path):
    images_dir, labels_dir, spec = _build_fixture(tmp_path)
    out_dir = tmp_path / "seg_run"

    run_segment(
        spec,
        backend=PrecomputedLabelsBackend(labels_dir=labels_dir),
        out_dir=out_dir,
        run_tag="v1",
    )
    with pytest.raises(FileExistsError):
        run_segment(
            spec,
            backend=PrecomputedLabelsBackend(labels_dir=labels_dir),
            out_dir=out_dir,
            run_tag="v1",
        )


def test_run_segment_overwrite_succeeds(tmp_path: Path):
    images_dir, labels_dir, spec = _build_fixture(tmp_path)
    out_dir = tmp_path / "seg_run"

    run_segment(
        spec,
        backend=PrecomputedLabelsBackend(labels_dir=labels_dir),
        out_dir=out_dir,
        run_tag="v1",
    )
    result = run_segment(
        spec,
        backend=PrecomputedLabelsBackend(labels_dir=labels_dir),
        out_dir=out_dir,
        run_tag="v1",
        overwrite=True,
    )
    assert result.output_dir is not None


def test_run_segment_in_memory_mode_returns_none_output(tmp_path: Path):
    images_dir, labels_dir, spec = _build_fixture(tmp_path)

    result = run_segment(
        spec,
        backend=PrecomputedLabelsBackend(labels_dir=labels_dir),
        out_dir=None,
    )
    assert result.output_dir is None
    assert "synth_a" in result.artefacts_by_dataset


def test_run_segment_accepts_manifest_list(tmp_path: Path):
    images_dir, labels_dir, spec = _build_fixture(tmp_path)

    result = run_segment(
        [spec],
        backend=PrecomputedLabelsBackend(labels_dir=labels_dir),
        out_dir=None,
    )
    assert len(result.artefacts_by_dataset) == 1


def test_run_segment_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        run_segment([])


def test_run_segment_unknown_backend_string_raises(tmp_path: Path):
    images_dir, labels_dir, spec = _build_fixture(tmp_path)
    with pytest.raises(ValueError, match="Unknown segmentation backend"):
        run_segment(spec, backend="not_a_real_backend")
