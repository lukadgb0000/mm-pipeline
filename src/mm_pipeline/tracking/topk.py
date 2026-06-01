"""Top-K DP candidate generation"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from mm_pipeline.config import TrackerParams
from mm_pipeline.core import CandidateSolution, CellInstance, FramePair, canonical_ops_key

from .costs import divide_cost, exit_increment, link_cost, validate_tracker_context
from .validation import assert_ops_valid

Op = tuple[str, int, int | None, int | None]


@dataclass(frozen=True)
class _PathEntry:
    cost: float
    prev_i: int | None
    prev_j: int | None
    prev_rank: int | None
    op: Op | None


def _push_entry(bucket: list[_PathEntry], entry: _PathEntry, limit: int) -> None:
    bucket.append(entry)
    bucket.sort(key=lambda item: item.cost)
    if len(bucket) > limit:
        del bucket[limit:]


def _solve_pair_topk_raw(
    cells_t: Sequence[CellInstance],
    cells_k: Sequence[CellInstance],
    frame_pair: FramePair,
    params: TrackerParams,
    raw_limit: int,
) -> list[tuple[list[Op], float]]:
    sources = tuple(cells_t)
    dests = tuple(cells_k)
    n = len(sources)
    m = len(dests)
    bottom_label_dest = dests[0].label if dests else None

    table: list[list[list[_PathEntry]]] = [[[] for _ in range(m + 1)] for _ in range(n + 1)]
    table[0][0] = [_PathEntry(cost=0.0, prev_i=None, prev_j=None, prev_rank=None, op=None)]

    for i in range(n + 1):
        for j in range(m + 1):
            if not table[i][j]:
                continue

            for rank, entry in enumerate(table[i][j]):
                base = entry.cost

                if j == 0 and i < n:
                    k_exit = i + 1
                    op: Op = ("exit", sources[i].label, None, None)
                    _push_entry(
                        table[i + 1][0],
                        _PathEntry(base + exit_increment(k_exit, params), i, j, rank, op),
                        raw_limit,
                    )

                if i < n and j < m:
                    op = ("link", sources[i].label, dests[j].label, None)
                    _push_entry(
                        table[i + 1][j + 1],
                        _PathEntry(
                            base
                            + link_cost(
                                sources[i],
                                dests[j],
                                bottom_label_dest=bottom_label_dest,
                                frame_pair=frame_pair,
                                params=params,
                            ),
                            i,
                            j,
                            rank,
                            op,
                        ),
                        raw_limit,
                    )

                if i < n and (j + 1) < m:
                    op = ("divide", sources[i].label, dests[j].label, dests[j + 1].label)
                    _push_entry(
                        table[i + 1][j + 2],
                        _PathEntry(
                            base
                            + divide_cost(
                                sources[i],
                                dests[j],
                                dests[j + 1],
                                frame_pair=frame_pair,
                                params=params,
                            ),
                            i,
                            j,
                            rank,
                            op,
                        ),
                        raw_limit,
                    )

    out: list[tuple[list[Op], float]] = []
    finals = table[n][m]
    for final_rank, final_entry in enumerate(finals):
        if not math.isfinite(final_entry.cost):
            continue

        i = n
        j = m
        rank = final_rank
        ops_rev: list[Op] = []

        while not (i == 0 and j == 0):
            cur = table[i][j][rank]
            if cur.op is None or cur.prev_i is None or cur.prev_j is None or cur.prev_rank is None:
                raise RuntimeError(f"Backtrace failed at state ({i},{j},{rank}).")
            ops_rev.append(cur.op)
            i, j, rank = int(cur.prev_i), int(cur.prev_j), int(cur.prev_rank)

        out.append((list(reversed(ops_rev)), float(final_entry.cost)))

    return out


def solve_pair_topk(
    cells_t: Sequence[CellInstance],
    cells_k: Sequence[CellInstance],
    frame_pair: FramePair,
    params: TrackerParams,
    top_k: int,
) -> list[CandidateSolution]:
    """Return up to ``top_k`` unique DP candidates for one sorted frame pair."""

    if top_k < 1:
        return []

    validate_tracker_context(frame_pair, params)
    raw_limit = max(32, 4 * int(top_k))
    raw_candidates = _solve_pair_topk_raw(cells_t, cells_k, frame_pair, params, raw_limit)

    out: list[CandidateSolution] = []
    seen: set[tuple[tuple[str, int, int, int], ...]] = set()
    for ops, cost in raw_candidates:
        key = canonical_ops_key(ops)
        if key in seen:
            continue
        seen.add(key)
        candidate = CandidateSolution.from_ops(
            pair_id=frame_pair.pair_id,
            ops=ops,
            generator="dp_topk",
            rank=len(out) + 1,
            cost=float(cost),
        )
        assert_ops_valid(cells_t, cells_k, candidate)
        out.append(candidate)
        if len(out) >= top_k:
            break

    return out
