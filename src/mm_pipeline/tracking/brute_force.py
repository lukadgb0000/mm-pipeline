"""Bounded exhaustive candidate generation for one frame pair."""

from __future__ import annotations

import math
from collections.abc import Sequence

from mm_pipeline.config import TrackerParams
from mm_pipeline.core import CandidateSolution, CellInstance, FramePair, canonical_ops_key

from .costs import candidate_ops_cost, validate_tracker_context
from .validation import assert_ops_valid

Op = tuple[str, int, int | None, int | None]


def count_pair_candidates(n_sources: int, n_destinations: int) -> int:
    """Return the exact number of default-model structural solutions.

    For ``e`` open-end-prefix exits, the remaining sources each link or divide.
    The destination count fixes the number of divides, leaving only the positions
    of those divides to choose.
    """

    n = int(n_sources)
    m = int(n_destinations)
    if n < 0 or m < 0:
        raise ValueError("Cell counts must be non-negative.")

    total = 0
    for n_exits in range(n + 1):
        remaining = n - n_exits
        n_divides = m - remaining
        n_links = remaining - n_divides
        if n_links < 0 or n_divides < 0:
            continue
        total += math.comb(remaining, n_divides)
    return int(total)


def enumerate_pair_candidates(
    cells_t: Sequence[CellInstance],
    cells_k: Sequence[CellInstance],
    frame_pair: FramePair,
    params: TrackerParams,
    *,
    max_candidates: int | None = 100_000,
) -> list[CandidateSolution]:
    """Enumerate every valid default-model solution for one sorted pair.

    Unlike :func:`solve_pair_topk`, this traversal performs no cost-based
    per-state pruning. ``max_candidates`` protects callers from accidentally
    materialising a combinatorially large set; pass ``None`` only when the caller
    has independently bounded the pair.
    """

    validate_tracker_context(frame_pair, params)
    sources = tuple(cells_t)
    dests = tuple(cells_k)
    predicted = count_pair_candidates(len(sources), len(dests))
    if max_candidates is not None:
        limit = int(max_candidates)
        if limit < 0:
            raise ValueError("max_candidates must be non-negative or None.")
        if predicted > limit:
            raise ValueError(
                f"Pair has {predicted} structural candidates, exceeding "
                f"max_candidates={limit}."
            )

    raw: list[tuple[Op, ...]] = []

    def walk(i: int, j: int, ops: tuple[Op, ...]) -> None:
        if i == len(sources) and j == len(dests):
            raw.append(ops)
            return
        if i >= len(sources):
            return

        # Exits are permitted only while no destination has been consumed, so
        # they form the internal open-end prefix.
        if j == 0:
            walk(i + 1, j, ops + (("exit", sources[i].label, None, None),))
        if j < len(dests):
            walk(
                i + 1,
                j + 1,
                ops + (("link", sources[i].label, dests[j].label, None),),
            )
        if j + 1 < len(dests):
            walk(
                i + 1,
                j + 2,
                ops
                + ((
                    "divide",
                    sources[i].label,
                    dests[j].label,
                    dests[j + 1].label,
                ),),
            )

    walk(0, 0, tuple())
    if len(raw) != predicted:
        raise RuntimeError(
            f"Exhaustive traversal produced {len(raw)} candidates; expected {predicted}."
        )

    costed = [
        (
            candidate_ops_cost(sources, dests, frame_pair, params, ops),
            canonical_ops_key(ops),
            ops,
        )
        for ops in raw
    ]
    costed.sort(key=lambda item: (item[0], item[1]))

    out: list[CandidateSolution] = []
    for rank, (cost, _key, ops) in enumerate(costed, start=1):
        candidate = CandidateSolution.from_ops(
            pair_id=frame_pair.pair_id,
            ops=ops,
            generator="brute_force",
            rank=rank,
            cost=float(cost),
        )
        assert_ops_valid(sources, dests, candidate)
        out.append(candidate)
    return out


__all__ = ["count_pair_candidates", "enumerate_pair_candidates"]
