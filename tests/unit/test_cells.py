import pytest

from mm_pipeline.core import (
    cell_axis_len,
    cells_by_label,
    extract_cell_instances,
    sort_cells_along_trench,
)


np = pytest.importorskip("numpy")


def test_extract_cell_instances_matches_regionprops_geometry_contract():
    labels = np.zeros((6, 7), dtype=np.uint32)
    labels[0:2, 0] = 1
    labels[3:5, 4:6] = 2

    cells = extract_cell_instances(labels, dataset_id="trench_a", frame=3)

    assert [cell.label for cell in cells] == [1, 2]
    by_label = cells_by_label(cells)

    c1 = by_label[1]
    assert c1.dataset_id == "trench_a"
    assert c1.frame == 3
    assert c1.area == 2.0
    assert c1.x == 0.0
    assert c1.y == 0.5
    assert (c1.bbox_minr, c1.bbox_minc, c1.bbox_maxr, c1.bbox_maxc) == (0, 0, 2, 1)

    c2 = by_label[2]
    assert c2.area == 4.0
    assert c2.x == 4.5
    assert c2.y == 3.5
    assert (c2.bbox_minr, c2.bbox_minc, c2.bbox_maxr, c2.bbox_maxc) == (3, 4, 5, 6)


def test_sort_cells_along_trench_respects_open_end_and_axis():
    labels = np.zeros((6, 7), dtype=np.uint32)
    labels[0:2, 0] = 1
    labels[3:5, 4:6] = 2
    cells = extract_cell_instances(labels)

    assert [c.label for c in sort_cells_along_trench(cells, axis="y", open_end="low")] == [1, 2]
    assert [c.label for c in sort_cells_along_trench(cells, axis="y", open_end="high")] == [2, 1]
    assert [c.label for c in sort_cells_along_trench(cells, axis="x", open_end="low")] == [1, 2]
    assert [c.label for c in sort_cells_along_trench(cells, axis="x", open_end="high")] == [2, 1]


def test_cell_axis_len_uses_exclusive_bbox_extent():
    labels = np.zeros((6, 7), dtype=np.uint32)
    labels[3:5, 4:6] = 2
    cell = extract_cell_instances(labels)[0]

    assert cell_axis_len(cell, "y") == 2.0
    assert cell_axis_len(cell, "x") == 2.0


def test_extract_cell_instances_rejects_non_2d_labels():
    with pytest.raises(ValueError, match="Expected a 2D label image"):
        extract_cell_instances(np.zeros((1, 2, 2), dtype=np.uint32))
