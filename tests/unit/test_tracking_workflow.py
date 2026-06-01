import pytest

from mm_pipeline.config import TrackerParams
from mm_pipeline.core import CellInstance, FramePair
from mm_pipeline.tracking import (
    extract_sorted_cells_for_stack,
    generate_pair_candidates,
    generate_tracking_candidates_for_stack,
)


def _array_deps():
    return pytest.importorskip("numpy")


def make_cell(label: int, y: float) -> CellInstance:
    return CellInstance(
        label=label,
        x=5.0,
        y=y,
        area=4.0,
        bbox_minr=int(y),
        bbox_minc=4,
        bbox_maxr=int(y) + 2,
        bbox_maxc=6,
    )


def test_generate_pair_candidates_rejects_unknown_mode():
    pair = FramePair("trench_a", 0, 1, (20, 10), "y", "high")

    with pytest.raises(ValueError, match="mode"):
        generate_pair_candidates([], [], pair, mode="unknown")  # type: ignore[arg-type]


def test_extract_sorted_cells_for_stack_preserves_frame_metadata_and_order():
    np = _array_deps()
    labels = np.zeros((2, 8, 8), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 1
    labels[0, 5:7, 1:3] = 2
    labels[1, 2:4, 1:3] = 10

    cells_by_frame = extract_sorted_cells_for_stack(
        labels,
        dataset_id="trench_a",
        axis="y",
        open_end="high",
    )

    assert [cell.label for cell in cells_by_frame[0]] == [2, 1]
    assert cells_by_frame[0][0].dataset_id == "trench_a"
    assert cells_by_frame[0][0].frame == 0
    assert cells_by_frame[1][0].frame == 1


def test_generate_tracking_candidates_for_stack_returns_pair_grouped_run():
    np = _array_deps()
    labels = np.zeros((2, 8, 8), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 2:4, 1:3] = 10

    run = generate_tracking_candidates_for_stack(
        labels,
        dataset_id="trench_a",
        axis="y",
        open_end="low",
        params=TrackerParams(),
        mode="best",
    )

    assert run.dataset_id == "trench_a"
    assert run.frame_shape == (8, 8)
    assert len(run.cells_by_frame) == 2
    assert len(run.pair_results) == 1

    result = run.pair_results[0]
    assert result.frame_pair.pair_id == "trench_a:0->1"
    assert [candidate.pair_id for candidate in result.candidates] == ["trench_a:0->1"]
    assert [candidate.generator for candidate in run.candidates] == ["dp_best"]
