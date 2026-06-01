import json

from mm_pipeline.core import CandidateSolution, TrackingOperation, deserialize_ops_json, serialize_ops_json
from mm_pipeline.core.operations import canonical_ops_key


def test_ops_json_round_trip_matches_legacy_shape():
    ops = [
        TrackingOperation("link", 1, 10, None),
        TrackingOperation("divide", 2, 11, 12),
        TrackingOperation("exit", 3, None, None),
    ]

    payload = serialize_ops_json(ops)

    assert json.loads(payload) == [
        ["link", 1, 10, None],
        ["divide", 2, 11, 12],
        ["exit", 3, None, None],
    ]
    assert [op.to_tuple() for op in deserialize_ops_json(payload)] == [op.to_tuple() for op in ops]


def test_candidate_solution_ops_json_round_trip():
    candidate = CandidateSolution.from_ops(
        pair_id="dataset:0->1",
        ops=[("link", 1, 2, None), ("exit", 3, None, None)],
        generator="dp_topk",
        rank=1,
        cost=0.5,
    )

    restored = CandidateSolution.from_ops_json(
        pair_id=candidate.pair_id,
        ops_json=candidate.to_ops_json(),
        generator=candidate.generator,
        rank=candidate.rank,
        cost=candidate.cost,
    )

    assert restored == candidate


def test_canonical_ops_key_sorts_division_daughters():
    key_a = canonical_ops_key([("divide", 5, 7, 6)])
    key_b = canonical_ops_key([("divide", 5, 6, 7)])

    assert key_a == key_b == (("divide", 5, 6, 7),)
