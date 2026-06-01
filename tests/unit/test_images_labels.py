from pathlib import Path

import pytest

from mm_pipeline.io.images import collect_image_paths
from mm_pipeline.io.labels import collect_label_paths, load_labels_from_folder, save_label_stack


np = pytest.importorskip("numpy")
tiff = pytest.importorskip("tifffile")


def test_collect_image_paths_natural_sort_and_extension_filter(tmp_path: Path):
    for name in ["img10.png", "img2.tif", "img1.bmp", "notes.txt"]:
        (tmp_path / name).write_text("")

    paths = collect_image_paths(tmp_path)

    assert [p.name for p in paths] == ["img1.bmp", "img2.tif", "img10.png"]


def test_label_stack_save_collect_and_load(tmp_path: Path):
    labels = np.zeros((2, 4, 4), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 1:3, 1:3] = 2

    written = save_label_stack(labels, ["frame2.tif", "frame10.tif"], tmp_path)

    assert [p.name for p in written] == ["frame2.tif", "frame10.tif"]
    assert [p.name for p in collect_label_paths(tmp_path)] == ["frame2.tif", "frame10.tif"]
    loaded = load_labels_from_folder(tmp_path)
    np.testing.assert_array_equal(loaded, labels)
