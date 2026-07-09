from pathlib import Path

import pytest

from mm_pipeline.config import SegmentationQCConfig
from mm_pipeline.io.labels import load_labels_from_folder, save_label_stack
from mm_pipeline.segmentation_qc import (
    collect_label_image_pairs,
    default_edited_labels_dir,
    find_small_labels,
    normalize_stem,
    resolve_review_output_dir,
    run_basic_checks,
    save_approved_labels,
    write_qc_report_csv,
)


def _array_deps():
    pytest.importorskip("tifffile")
    return pytest.importorskip("numpy")


def test_normalize_stem_strips_known_label_suffixes():
    assert normalize_stem("frame001_cp_masks.tif") == "frame001"
    assert normalize_stem("frame001_masks.tif") == "frame001"
    assert normalize_stem("frame001_labels.tif") == "frame001"
    assert normalize_stem("frame001_seg.tif") == "frame001"


def test_collect_label_image_pairs_uses_normalized_stems(tmp_path: Path):
    np = _array_deps()
    labels_dir = tmp_path / "labels"
    images_dir = tmp_path / "images"
    labels_dir.mkdir()
    images_dir.mkdir()
    save_label_stack(np.ones((1, 3, 3), dtype=np.uint32), ["frame001_cp_masks.tif"], labels_dir)
    (images_dir / "frame001.png").write_text("")

    pairing = collect_label_image_pairs(labels_dir, images_dir)

    assert pairing.stems_match
    assert pairing.save_names == ("frame001_cp_masks.tif",)


def test_review_output_defaults_to_sibling_edited_dir(tmp_path: Path):
    labels_dir = tmp_path / "labels"

    assert default_edited_labels_dir(labels_dir) == tmp_path / "labels_edited"
    assert resolve_review_output_dir(labels_dir) == tmp_path / "labels_edited"


def test_review_output_refuses_in_place_write_without_overwrite(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        resolve_review_output_dir(labels_dir, labels_dir)


def test_review_output_allows_in_place_write_with_overwrite(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    assert resolve_review_output_dir(labels_dir, labels_dir, overwrite=True) == labels_dir


def test_find_small_labels_matches_segqa_tuple_shape():
    np = _array_deps()
    labels = np.zeros((2, 4, 4), dtype=np.uint32)
    labels[0, 0, 0] = 1
    labels[0, 1:3, 1:3] = 2
    labels[1, 2, 2] = 5

    assert find_small_labels(labels, min_size=2) == [(0, 1, 1), (1, 5, 1)]


def test_run_basic_checks_and_write_report(tmp_path: Path):
    np = _array_deps()
    labels = np.zeros((3, 5, 5), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 1
    labels[1] = 0
    labels[2, 0, 0] = 1
    labels[2, 1, 1] = 2
    labels[2, 2, 2] = 3
    labels[2, 3, 3] = 4

    findings = run_basic_checks(
        labels,
        "trench_a",
        SegmentationQCConfig(min_label_size=2, cell_count_jump_threshold=2, total_area_jump_fraction=0.5),
    )

    names = {finding.check_name for finding in findings}
    assert "empty_frame" in names
    assert "small_label" in names
    assert "cell_count_jump" in names
    assert "total_area_jump" in names

    out_csv = write_qc_report_csv(findings, tmp_path / "qa.csv")
    assert out_csv.exists()
    assert "check_name" in out_csv.read_text()


def test_save_approved_labels_round_trip(tmp_path: Path):
    np = _array_deps()
    labels = np.zeros((1, 3, 3), dtype=np.uint32)
    labels[0, 1, 1] = 1

    approved = save_approved_labels(labels, ["frame001.tif"], tmp_path / "approved", dataset_id="trench_a")

    assert approved.dataset_id == "trench_a"
    loaded = load_labels_from_folder(approved.labels_dir)
    np.testing.assert_array_equal(loaded, labels)
