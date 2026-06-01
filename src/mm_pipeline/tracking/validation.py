"""Tracking operation validation"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from mm_pipeline.core import CandidateSolution, CellInstance, TrackingOperation
from mm_pipeline.core.operations import normalize_operation

OperationLike = TrackingOperation | Sequence[object]


def _normalise_ops(ops: CandidateSolution | Iterable[OperationLike]) -> list[TrackingOperation]:
    raw_ops: Iterable[OperationLike]
    if isinstance(ops, CandidateSolution):
        raw_ops = ops.ops
    else:
        raw_ops = ops

    try:
        return [normalize_operation(op) for op in raw_ops]
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def assert_ops_valid(
    cells_t: Sequence[CellInstance],
    cells_k: Sequence[CellInstance],
    ops: CandidateSolution | Iterable[OperationLike],
) -> None:
    """Validate source/destination coverage and DP prefix-exit semantics.

    ``cells_t`` and ``cells_k`` must be in sorted trench order with the open-end
    prefix first. The prefix-exit check relies on that ordering. CHANGE THIS TO SUFFIX ASAP!!!
    """

    source_labels = [int(cell.label) for cell in cells_t]
    dest_labels = [int(cell.label) for cell in cells_k]
    source_set = set(source_labels)
    dest_set = set(dest_labels)

    if len(source_set) != len(source_labels):
        raise ValueError("Frame t contains duplicate labels.")
    if len(dest_set) != len(dest_labels):
        raise ValueError("Frame k contains duplicate labels.")

    normalised = _normalise_ops(ops)
    used_sources: list[int] = []
    used_dests: list[int] = []
    exit_sources: list[int] = []

    for op in normalised:
        src = int(op.src_label)
        if src not in source_set:
            raise ValueError(f"Op references src label {src} not in frame t.")
        used_sources.append(src)

        if op.kind == "link":
            if op.dst1_label is None or op.dst2_label is not None:
                raise ValueError(f"Bad link operation: {op.to_tuple()}.")
            if op.dst1_label not in dest_set:
                raise ValueError(f"Link references dst label {op.dst1_label} not in frame k.")
            used_dests.append(int(op.dst1_label))
        elif op.kind == "divide":
            if op.dst1_label is None or op.dst2_label is None:
                raise ValueError(f"Bad divide operation: {op.to_tuple()}.")
            if op.dst1_label not in dest_set or op.dst2_label not in dest_set:
                raise ValueError(
                    f"Divide references dst labels {(op.dst1_label, op.dst2_label)} not in frame k."
                )
            used_dests.extend([int(op.dst1_label), int(op.dst2_label)])
        elif op.kind == "exit":
            if op.dst1_label is not None or op.dst2_label is not None:
                raise ValueError(f"Bad exit operation: {op.to_tuple()}.")
            exit_sources.append(src)
        else:
            raise ValueError(f"Unknown op kind: {op.kind}.")

    if len(used_sources) != len(source_labels):
        raise ValueError(f"Expected exactly {len(source_labels)} ops consuming frame t labels.")
    if len(set(used_sources)) != len(used_sources):
        raise ValueError("A frame t label was consumed more than once.")
    if set(used_sources) != source_set:
        raise ValueError("Each frame t label must be consumed exactly once.")

    if len(used_dests) != len(dest_labels):
        raise ValueError(f"Expected to consume exactly {len(dest_labels)} frame k labels.")
    if len(set(used_dests)) != len(used_dests):
        raise ValueError("A frame k label was assigned more than once.")
    if set(used_dests) != dest_set:
        raise ValueError("Each frame k label must be assigned exactly once.")

    expected_exit_sources = set(source_labels[: len(exit_sources)])
    if set(exit_sources) != expected_exit_sources:
        raise ValueError("Exit operations are only valid for the open-end prefix of frame t.")
