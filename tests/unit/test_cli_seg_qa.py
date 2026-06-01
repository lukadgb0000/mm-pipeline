"""Tests for mm-pipeline seg-qa and run_seg_qa"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_pipeline.cli.main import main
from mm_pipeline.config import DatasetSpec, SegmentationQAConfig
from mm_pipeline.runners.seg_qa import SegQAResult, run_seg_qa


def _np():
    return pytest.importorskip("numpy")


def _build_labels_dir(tmp_path: Path, *, name: str = "labels") -> Path:
    np = _np()
    tiff = pytest.importorskip("tifffile")
    labels_dir = tmp_path / name
    labels_dir.mkdir()
    for t in range(3):
        labels = np.zeros((8, 8), dtype=np.uint32)
        labels[1 + t : 3 + t, 1:3] = 1
        tiff.imwrite(labels_dir / f"frame_{t:03d}.tif", labels)
    return labels_dir


def _spec(labels_dir: Path, dataset_id: str = "synth_a") -> DatasetSpec:
    return DatasetSpec(dataset_id=dataset_id, labels_dir=labels_dir)


def test_cli_seg_qa_help_succeeds(capsys):
    try:
        main(["seg-qa", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "qa" in captured.out.lower()
    assert "napari" in captured.out.lower()  # documents the headless-only design


def test_run_seg_qa_writes_findings_csv(tmp_path: Path):
    labels_dir = _build_labels_dir(tmp_path)
    spec = _spec(labels_dir)
    out_dir = tmp_path / "seg_review"

    result = run_seg_qa(spec, out_dir=out_dir, run_tag="v1")

    assert isinstance(result, SegQAResult)
    assert result.output_dir == out_dir / "v1"
    assert (out_dir / "v1" / "synth_a" / "seg_qa_findings.csv").exists()
    summary = json.loads((out_dir / "v1" / "summary.json").read_text())
    assert summary["command"] == "seg-qa"
    assert summary["dataset_ids"] == ["synth_a"]
    assert "synth_a" in result.findings_by_dataset


def test_run_seg_qa_in_memory_mode(tmp_path: Path):
    labels_dir = _build_labels_dir(tmp_path)
    spec = _spec(labels_dir)

    result = run_seg_qa(spec, out_dir=None)
    assert result.output_dir is None
    assert "synth_a" in result.findings_by_dataset


def test_run_seg_qa_overwrite_protection(tmp_path: Path):
    labels_dir = _build_labels_dir(tmp_path)
    spec = _spec(labels_dir)
    out_dir = tmp_path / "seg_review"

    run_seg_qa(spec, out_dir=out_dir, run_tag="v1")
    with pytest.raises(FileExistsError):
        run_seg_qa(spec, out_dir=out_dir, run_tag="v1")


def test_run_seg_qa_accepts_custom_config(tmp_path: Path):
    labels_dir = _build_labels_dir(tmp_path)
    spec = _spec(labels_dir)

    cfg = SegmentationQAConfig(min_label_size=100)  # everything is too small
    result = run_seg_qa(spec, config=cfg, out_dir=None)
    findings = result.findings_by_dataset["synth_a"]
    # All cells get flagged as small.
    assert any(f.check_name == "small_label" for f in findings)


def test_run_seg_qa_missing_labels_raises(tmp_path: Path):
    spec = DatasetSpec(
        dataset_id="orphan",
        images_dir=tmp_path,  # only images, no labels
    )
    with pytest.raises(ValueError, match="labels"):
        run_seg_qa(spec)


def test_run_seg_qa_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        run_seg_qa([])
