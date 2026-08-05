"""Unit tests for mm_pipeline.analysis.selection (TrackSet + selectors)."""

from __future__ import annotations

import pytest

from mm_pipeline.analysis import (
    Lineage,
    ancestors_of,
    descendants_of,
    filter_cycles,
    generation,
    leaves,
    mother_branch,
    path_between,
    roots,
)
from mm_pipeline.analysis.selection import TrackSet

pd = pytest.importorskip("pandas")


# --- a dividing lineage with distinct birth positions -------------------------
#
#   1 (y=5) --div@1--> 2 (y=2), 3 (y=8)
#   2       --div@3--> 4 (y=1), 5 (y=4)
# open_end="high" => closed end at low y => mother lineage = 1 -> 2 -> 4.


def _rows(track_id, ty_pairs):
    return [
        {"track_id": track_id, "t": t, "label": 1, "x": 2.0, "y": float(y), "area": 10.0, "axis_len": 5.0}
        for t, y in ty_pairs
    ]


def _dividing_lineage(axis="y", open_end="high"):
    tracks = pd.DataFrame(
        _rows(1, [(0, 5), (1, 5)])
        + _rows(2, [(2, 2), (3, 2)])
        + _rows(3, [(2, 8), (3, 8), (4, 8)])
        + _rows(4, [(4, 1), (5, 1)])
        + _rows(5, [(4, 4), (5, 4)])
    )
    divisions = pd.DataFrame(
        [
            {"t_div": 1, "mother_track_id": 1, "d1_track_id": 2, "d2_track_id": 3},
            {"t_div": 3, "mother_track_id": 2, "d1_track_id": 4, "d2_track_id": 5},
        ]
    )
    events = pd.DataFrame(columns=["t", "event", "src_label", "dst1_label", "dst2_label", "track_id", "cost", "dst_t"])
    return Lineage(dataset_id="d1", tracks_df=tracks, divisions_df=divisions, events_df=events, axis=axis, open_end=open_end)


# --- notebook trace_branch, copied verbatim as a parity oracle ----------------


def _notebook_build_tree(tracks_df, div_df, open_end):
    children_map, parent_map = {}, {}
    for _, r in div_df.iterrows():
        m, d1, d2 = int(r["mother_track_id"]), int(r["d1_track_id"]), int(r["d2_track_id"])
        children_map[m] = (d1, d2)
        parent_map[d1] = m
        parent_map[d2] = m
    birth = {}
    for tid, seg in tracks_df.groupby("track_id"):
        row = seg.loc[seg["t"].idxmin()]
        birth[int(tid)] = {"t": int(row["t"]), "y": float(row["y"])}
    return children_map, birth


def _notebook_pick_closed(d1, d2, y1, y2, open_end):
    if y1 is None or y2 is None:
        return d1
    if open_end == "high":
        return d1 if y1 < y2 else d2
    return d1 if y1 > y2 else d2


def _notebook_trace_branch(tracks_df, div_df, open_end):
    children, birth = _notebook_build_tree(tracks_df, div_df, open_end)
    t0 = min(b["t"] for b in birth.values())
    cands = [(tid, b["y"]) for tid, b in birth.items() if b["t"] == t0]
    chooser = min if open_end == "high" else max
    cur = chooser(cands, key=lambda kv: kv[1])[0]
    branch, seen = [], set()
    while cur is not None and cur not in seen and cur in birth:
        branch.append(cur)
        seen.add(cur)
        if cur not in children:
            break
        d1, d2 = children[cur]
        cur = _notebook_pick_closed(d1, d2, birth.get(d1, {}).get("y"), birth.get(d2, {}).get("y"), open_end)
    return branch


def test_mother_branch_matches_expected():
    lin = _dividing_lineage()
    assert set(mother_branch(lin)) == {1, 2, 4}


@pytest.mark.parametrize("open_end", ["high", "low"])
def test_mother_branch_parity_with_notebook(open_end):
    lin = _dividing_lineage(open_end=open_end)
    reference = _notebook_trace_branch(lin.tracks_df, lin.divisions_df, open_end)
    assert set(mother_branch(lin).track_ids) == set(reference)


def test_mother_branch_low_open_end_follows_high_position():
    # Mirror image: closed end at high y => mother lineage = 1 -> 3.
    lin = _dividing_lineage(open_end="low")
    assert 3 in mother_branch(lin)
    assert 2 not in mother_branch(lin)


def test_descendants_of_is_inclusive_subtree():
    lin = _dividing_lineage()
    assert set(descendants_of(lin, 1)) == {1, 2, 3, 4, 5}
    assert set(descendants_of(lin, 2)) == {2, 4, 5}
    assert set(descendants_of(lin, 2, inclusive=False)) == {4, 5}
    assert set(descendants_of(lin, 4)) == {4}  # leaf


def test_ancestors_of_walks_up():
    lin = _dividing_lineage()
    assert set(ancestors_of(lin, 4)) == {1, 2}
    assert set(ancestors_of(lin, 4, inclusive=True)) == {1, 2, 4}
    assert set(ancestors_of(lin, 1)) == set()  # root


def test_path_between_is_the_inclusive_lineage_path():
    lin = _dividing_lineage()
    # 1 -> 2 -> 4 is a single ancestor->descendant line.
    assert set(path_between(lin, 1, 4)) == {1, 2, 4}
    assert set(path_between(lin, 4, 1)) == {1, 2, 4}  # order-independent
    assert path_between(lin, 1, 4).name == "path_between(1, 4)"


def test_path_between_single_cell_and_direct_child():
    lin = _dividing_lineage()
    assert set(path_between(lin, 3, 3)) == {3}  # same cell
    assert set(path_between(lin, 2, 5)) == {2, 5}  # mother -> daughter


def test_path_between_off_lineage_or_unknown_is_empty():
    lin = _dividing_lineage()
    assert set(path_between(lin, 3, 4)) == set()  # different branches
    assert set(path_between(lin, 4, 5)) == set()  # sibling leaves
    assert set(path_between(lin, 1, 999)) == set()  # unknown id


def test_roots_and_leaves():
    lin = _dividing_lineage()
    assert set(roots(lin)) == {1}
    assert set(leaves(lin)) == {3, 4, 5}


def test_generation_selector():
    lin = _dividing_lineage()
    assert set(generation(lin, 0)) == {1}
    assert set(generation(lin, 1)) == {2, 3}
    assert set(generation(lin, 2)) == {4, 5}


def test_filter_cycles():
    lin = _dividing_lineage()
    divided = filter_cycles(lin, lambda c: c["end_cause"] == "divided")
    assert set(divided) == {1, 2}


def test_set_algebra_and_slices():
    lin = _dividing_lineage()
    combined = descendants_of(lin, 2) & generation(lin, 2)
    assert set(combined) == {4, 5}
    assert set(descendants_of(lin, 1) - descendants_of(lin, 2)) == {1, 3}
    assert set(roots(lin) | leaves(lin)) == {1, 3, 4, 5}
    # slices resolve against the lineage
    assert set(generation(lin, 2).cycles(lin)["track_id"]) == {4, 5}
    assert set(generation(lin, 2).detections(lin)["track_id"]) == {4, 5}


def test_dataset_id_guard_blocks_cross_lineage_use():
    a = TrackSet("A", frozenset({1}))
    b = TrackSet("B", frozenset({1}))
    with pytest.raises(ValueError):
        _ = a | b
    lin_b = _dividing_lineage()  # dataset_id "d1"
    with pytest.raises(ValueError):
        a.cycles(lin_b)
