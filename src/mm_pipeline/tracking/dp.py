"""Best-path DP tracker"""

from __future__ import annotations

import math
from collections.abc import Sequence

from mm_pipeline.config import TrackerParams
from mm_pipeline.core import CandidateSolution, CellInstance, FramePair

from .costs import divide_cost, exit_increment, link_cost, validate_tracker_context
from .validation import assert_ops_valid

Op = tuple[str, int, int | None, int | None]


def solve_pair_best(
    cells_t: Sequence[CellInstance],
    cells_k: Sequence[CellInstance],
    frame_pair: FramePair,
    params: TrackerParams,
) -> CandidateSolution:
    """Solve one sorted frame pair with the old dynamic programming algo"""

    validate_tracker_context(frame_pair, params)
    sources = tuple(cells_t)
    dests = tuple(cells_k)
    n = len(sources)
    m = len(dests)
    bottom_label_dest = dests[0].label if dests else None

    dp = [[math.inf for _ in range(m + 1)] for _ in range(n + 1)]
    prev: list[list[tuple[int, int, Op] | None]] = [[None for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            base = dp[i][j]
            if not math.isfinite(base):
                continue

            if j == 0 and i < n:
                k_exit = i + 1
                cand = base + exit_increment(k_exit, params)
                if cand < dp[i + 1][0]:
                    dp[i + 1][0] = cand
                    prev[i + 1][0] = (i, 0, ("exit", sources[i].label, None, None))

            if i < n and j < m:
                cand = base + link_cost(
                    sources[i],
                    dests[j],
                    bottom_label_dest=bottom_label_dest,
                    frame_pair=frame_pair,
                    params=params,
                )
                if cand < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = cand
                    prev[i + 1][j + 1] = (
                        i,
                        j,
                        ("link", sources[i].label, dests[j].label, None),
                    )

            if i < n and (j + 1) < m:
                cand = base + divide_cost(
                    sources[i],
                    dests[j],
                    dests[j + 1],
                    frame_pair=frame_pair,
                    params=params,
                )
                if cand < dp[i + 1][j + 2]:
                    dp[i + 1][j + 2] = cand
                    prev[i + 1][j + 2] = (
                        i,
                        j,
                        ("divide", sources[i].label, dests[j].label, dests[j + 1].label),
                    )

    total_cost = float(dp[n][m])
    if not math.isfinite(total_cost):
        raise ValueError(
            f"No valid alignment between frames: n={n}, m={m}. "
        )

    ops_rev: list[Op] = []
    i, j = n, m
    while (i, j) != (0, 0):
        back = prev[i][j]
        if back is None:
            raise RuntimeError(f"Backtrace failed at state (i={i}, j={j}).")
        pi, pj, op = back
        ops_rev.append(op)
        i, j = int(pi), int(pj)

    ops = list(reversed(ops_rev))
    candidate = CandidateSolution.from_ops(
        pair_id=frame_pair.pair_id,
        ops=ops,
        generator="dp_best",
        rank=1,
        cost=total_cost,
    )
    assert_ops_valid(sources, dests, candidate)
    return candidate
