"""Tests for exhaustive per-pair enumeration and exact operation costing."""

from __future__ import annotations

import pytest

from mm_pipeline.config import TrackerParams
from mm_pipeline.core import CellInstance, FramePair, canonical_ops_key
from mm_pipeline.tracking import (
    candidate_ops_cost,
    count_pair_candidates,
    enumerate_pair_candidates,
    solve_pair_topk,
)


def _cell(label, y, *, area=20, minr=0, maxr=10):
    return CellInstance(
        label=label,
        x=3.0,
        y=float(y),
        area=float(area),
        bbox_minr=minr,
        bbox_minc=1,
        bbox_maxr=maxr,
        bbox_maxc=6,
    )


def test_closed_form_candidate_counts():
    assert count_pair_candidates(0, 0) == 1
    assert count_pair_candidates(2, 2) == 2  # ll; ed
    assert count_pair_candidates(3, 4) == 4
    assert count_pair_candidates(11, 19) == 175
    assert count_pair_candidates(1, 3) == 0
    with pytest.raises(ValueError, match="non-negative"):
        count_pair_candidates(-1, 2)


def test_exhaustive_matches_topk_when_small_set_is_independently_bounded():
    cells_t = [_cell(1, 90), _cell(2, 50)]
    cells_k = [_cell(10, 88), _cell(11, 48)]
    pair = FramePair("d1", 0, 1, (100, 10), "y", "high")
    params = TrackerParams()

    exhaustive = enumerate_pair_candidates(cells_t, cells_k, pair, params)
    topk = solve_pair_topk(cells_t, cells_k, pair, params, top_k=10)

    assert len(exhaustive) == count_pair_candidates(2, 2) == 2
    assert [canonical_ops_key(c.ops) for c in exhaustive] == [
        canonical_ops_key(c.ops) for c in topk
    ]
    assert [c.cost for c in exhaustive] == pytest.approx([c.cost for c in topk])


def test_exhaustive_safety_ceiling_raises_before_traversal():
    cells_t = [_cell(i + 1, 100 - i * 10) for i in range(11)]
    cells_k = [_cell(i + 20, 100 - i * 5) for i in range(19)]
    pair = FramePair("d1", 0, 1, (120, 10), "y", "high")

    with pytest.raises(ValueError, match="175 structural candidates"):
        enumerate_pair_candidates(
            cells_t, cells_k, pair, TrackerParams(), max_candidates=100
        )


def test_exact_cost_matches_dp_with_multiple_exits_and_border_link():
    sources = [
        _cell(1, 96, area=20, minr=92, maxr=100),
        _cell(2, 85, area=20, minr=81, maxr=89),
        _cell(3, 60, area=20, minr=55, maxr=65),
    ]
    dests = [_cell(10, 96, area=8, minr=94, maxr=100)]
    pair = FramePair("d1", 0, 1, (100, 10), "y", "high")
    params = TrackerParams()
    candidates = enumerate_pair_candidates(sources, dests, pair, params)
    candidate = next(
        c for c in candidates if [op.kind for op in c.ops] == ["exit", "exit", "link"]
    )

    assert candidate_ops_cost(sources, dests, pair, params, candidate.ops) == pytest.approx(
        candidate.cost
    )
