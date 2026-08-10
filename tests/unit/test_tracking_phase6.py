import pytest

from mm_pipeline.config import TrackerParams
from mm_pipeline.core import CandidateSolution, CellInstance, FramePair, canonical_ops_key
from mm_pipeline.tracking import assert_ops_valid, solve_pair_best, solve_pair_topk
from mm_pipeline.tracking.costs import INFEASIBLE_DIVISION_COST, divide_cost


def make_cell(
    label: int,
    y: float,
    *,
    area: float = 10.0,
    minr: int = 0,
    maxr: int = 10,
    x: float = 5.0,
    minc: int = 0,
    maxc: int = 4,
) -> CellInstance:
    return CellInstance(
        label=label,
        x=x,
        y=y,
        area=area,
        bbox_minr=minr,
        bbox_minc=minc,
        bbox_maxr=maxr,
        bbox_maxc=maxc,
    )


def test_frame_pair_pair_id_and_validation():
    pair = FramePair(
        dataset_id="trench_a",
        t=2,
        k=3,
        frame_shape=(64, 16),
        axis="y",
        open_end="high",
    )

    assert pair.pair_id == "trench_a:2->3"
    assert pair.frame_shape == (64, 16)

    with pytest.raises(ValueError, match="k must be greater than t"):
        FramePair("trench_a", 2, 2, (64, 16), "y", "high")

    with pytest.raises(ValueError, match="axis"):
        FramePair("trench_a", 2, 3, (64, 16), "z", "high")  # type: ignore[arg-type]


def test_solve_pair_best_returns_candidate_solution_with_exact_link_cost():
    params = TrackerParams()
    pair = FramePair("trench_a", 0, 1, (100, 20), "y", "low")
    cells_t = [make_cell(1, 10.0, area=10.0, minr=5, maxr=15)]
    cells_k = [make_cell(11, 12.0, area=10.0, minr=7, maxr=17)]

    candidate = solve_pair_best(cells_t, cells_k, pair, params)

    assert candidate.pair_id == "trench_a:0->1"
    assert candidate.generator == "dp_best"
    assert candidate.rank == 1
    assert candidate.cost == pytest.approx(params.wy * 2.0)
    assert [op.to_tuple() for op in candidate.ops] == [("link", 1, 11, None)]
    assert_ops_valid(cells_t, cells_k, candidate)


def test_solve_pair_best_rejects_axis_mismatch():
    pair = FramePair("trench_a", 0, 1, (100, 20), "x", "low")
    cells_t = [make_cell(1, 10.0)]
    cells_k = [make_cell(11, 10.0)]

    with pytest.raises(ValueError, match="must match FramePair.axis"):
        solve_pair_best(cells_t, cells_k, pair, TrackerParams())


def test_prefix_exit_is_allowed_only_at_open_end_prefix():
    params = TrackerParams()
    pair = FramePair("trench_a", 0, 1, (100, 20), "y", "high")
    cells_t = [
        make_cell(1, 90.0, area=10.0, minr=88, maxr=99),
        make_cell(2, 50.0, area=10.0, minr=45, maxr=55),
    ]
    cells_k = [make_cell(20, 50.0, area=10.0, minr=45, maxr=55)]

    candidate = solve_pair_best(cells_t, cells_k, pair, params)

    assert [op.to_tuple() for op in candidate.ops] == [
        ("exit", 1, None, None),
        ("link", 2, 20, None),
    ]
    assert candidate.cost == pytest.approx(params.exit_lin + params.exit_quad)

    with pytest.raises(ValueError, match="prefix"):
        assert_ops_valid(
            cells_t,
            cells_k,
            [("link", 1, 20, None), ("exit", 2, None, None)],
        )


@pytest.mark.parametrize(
    ("source", "dest1", "dest2"),
    [
        (
            make_cell(1, 10.0, area=10.0, minr=0, maxr=10),
            make_cell(10, 8.0, area=20.0, minr=0, maxr=8),
            make_cell(11, 12.0, area=20.0, minr=8, maxr=16),
        ),
        (
            make_cell(1, 10.0, area=100.0, minr=0, maxr=10),
            make_cell(10, 8.0, area=40.0, minr=0, maxr=10),
            make_cell(11, 12.0, area=40.0, minr=10, maxr=20),
        ),
    ],
)
def test_division_hard_gates_return_legacy_infeasible_cost(source, dest1, dest2):
    params = TrackerParams(
        div_tol_sum_area=0.2,
        div_tol_ind_area=0.2,
        div_tol_sum_len=0.2,
        div_tol_ind_len=0.2,
    )
    pair = FramePair("trench_a", 0, 1, (100, 20), "y", "low")

    assert divide_cost(source, dest1, dest2, frame_pair=pair, params=params) == pytest.approx(
        INFEASIBLE_DIVISION_COST
    )


def test_default_tracker_does_not_hard_gate_divisions():
    params = TrackerParams()
    pair = FramePair("trench_a", 0, 1, (100, 20), "y", "low")
    source = make_cell(1, 10.0, area=10.0, minr=0, maxr=10)
    dest1 = make_cell(10, 8.0, area=20.0, minr=0, maxr=8)
    dest2 = make_cell(11, 12.0, area=20.0, minr=8, maxr=16)

    assert divide_cost(source, dest1, dest2, frame_pair=pair, params=params) < (
        INFEASIBLE_DIVISION_COST
    )


def test_solve_pair_best_preserves_legacy_infeasible_division_cost():
    params = TrackerParams(
        div_tol_sum_area=0.2,
        div_tol_ind_area=0.2,
        div_tol_sum_len=0.2,
        div_tol_ind_len=0.2,
    )
    pair = FramePair("trench_a", 0, 1, (100, 20), "y", "low")
    source = make_cell(1, 10.0, area=10.0, minr=0, maxr=10)
    dests = [
        make_cell(10, 8.0, area=20.0, minr=0, maxr=8),
        make_cell(11, 12.0, area=20.0, minr=8, maxr=16),
    ]

    candidate = solve_pair_best([source], dests, pair, params)

    assert [op.to_tuple() for op in candidate.ops] == [("divide", 1, 10, 11)]
    assert candidate.cost == pytest.approx(INFEASIBLE_DIVISION_COST)


def test_solve_pair_topk_returns_unique_ranked_candidates():
    params = TrackerParams()
    pair = FramePair("trench_a", 0, 1, (100, 20), "y", "low")
    cells_t = [
        make_cell(1, 10.0, area=20.0, minr=5, maxr=15),
        make_cell(2, 30.0, area=40.0, minr=20, maxr=60),
    ]
    cells_k = [
        make_cell(10, 20.0, area=20.0, minr=15, maxr=25),
        make_cell(11, 40.0, area=20.0, minr=35, maxr=45),
    ]

    candidates = solve_pair_topk(cells_t, cells_k, pair, params, top_k=10)

    assert len(candidates) == 2
    assert [candidate.generator for candidate in candidates] == ["dp_topk", "dp_topk"]
    assert [candidate.rank for candidate in candidates] == [1, 2]
    assert [candidate.cost for candidate in candidates] == sorted(
        candidate.cost for candidate in candidates if candidate.cost is not None
    )
    assert len({canonical_ops_key(candidate.ops) for candidate in candidates}) == len(candidates)
    for candidate in candidates:
        assert candidate.pair_id == pair.pair_id
        assert_ops_valid(cells_t, cells_k, candidate)


def test_solve_pair_topk_returns_empty_for_non_positive_top_k():
    pair = FramePair("trench_a", 0, 1, (100, 20), "y", "low")

    assert solve_pair_topk([], [], pair, TrackerParams(), top_k=0) == []


def test_assert_ops_valid_accepts_candidate_solution_input():
    cells_t = [make_cell(1, 10.0)]
    cells_k = [make_cell(11, 10.0)]
    candidate = CandidateSolution.from_ops(
        pair_id="trench_a:0->1",
        ops=[("link", 1, 11, None)],
        generator="manual",
    )

    assert_ops_valid(cells_t, cells_k, candidate)


def test_assert_ops_valid_rejects_invalid_operations():
    cells_t = [make_cell(1, 10.0), make_cell(2, 20.0)]
    cells_k = [make_cell(11, 10.0), make_cell(12, 20.0)]

    with pytest.raises(ValueError, match="consumed more than once"):
        assert_ops_valid(cells_t, cells_k, [("link", 1, 11, None), ("link", 1, 12, None)])

    with pytest.raises(ValueError, match="assigned more than once"):
        assert_ops_valid(cells_t, cells_k, [("link", 1, 11, None), ("link", 2, 11, None)])

    with pytest.raises(ValueError, match="src label 99"):
        assert_ops_valid(cells_t, cells_k, [("link", 99, 11, None), ("link", 2, 12, None)])

    with pytest.raises(ValueError, match="requires dst1_label"):
        assert_ops_valid(cells_t[:1], cells_k[:1], [("link", 1, None, None)])
