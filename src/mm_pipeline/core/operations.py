"""Tracking operation contracts and JSON compatibility helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, TypeAlias

OperationKind: TypeAlias = Literal["link", "divide", "exit"]
OpTuple: TypeAlias = tuple[str, int, Optional[int], Optional[int]]


@dataclass(frozen=True)
class TrackingOperation:
    """One frame-pair operation

    Note to self. This dataclass preserves compatibility with the old tuple format I used:
    ``("link", src, dst, None)``, ``("divide", src, d1, d2)``, and
    ``("exit", src, None, None)``.
    """

    kind: OperationKind
    src_label: int
    dst1_label: Optional[int] = None
    dst2_label: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in {"link", "divide", "exit"}:
            raise ValueError(f"Unknown operation kind: {self.kind}")
        if self.kind == "link" and self.dst1_label is None:
            raise ValueError("link operation requires dst1_label.")
        if self.kind == "divide" and (self.dst1_label is None or self.dst2_label is None):
            raise ValueError("divide operation requires both daughter labels.")
        if self.kind == "exit" and (self.dst1_label is not None or self.dst2_label is not None):
            raise ValueError("exit operation cannot have destination labels.")

    @classmethod
    def from_tuple(cls, op: Sequence[object]) -> "TrackingOperation":
        if len(op) != 4:
            raise ValueError(f"Operation must have 4 elements, got {len(op)}.")
        kind = str(op[0])
        if kind not in {"link", "divide", "exit"}:
            raise ValueError(f"Unknown operation kind: {kind}")
        return cls(
            kind=kind,  # type: ignore[arg-type]
            src_label=int(op[1]),
            dst1_label=None if op[2] is None else int(op[2]),
            dst2_label=None if op[3] is None else int(op[3]),
        )

    def to_tuple(self) -> OpTuple:
        return (self.kind, int(self.src_label), self.dst1_label, self.dst2_label)

    def to_json_compatible(self) -> list[object]:
        return list(self.to_tuple())


def normalize_operation(op: TrackingOperation | Sequence[object]) -> TrackingOperation:
    if isinstance(op, TrackingOperation):
        return op
    return TrackingOperation.from_tuple(op)


def serialize_ops_json(ops: Iterable[TrackingOperation | Sequence[object]]) -> str:
    """Serialise operations in the same JSON shape used by old featuretest2 code."""

    return json.dumps([normalize_operation(op).to_json_compatible() for op in ops])


def deserialize_ops_json(payload: str) -> list[TrackingOperation]:
    raw = json.loads(payload)
    if not isinstance(raw, list):
        raise ValueError("ops_json must decode to a list.")
    return [TrackingOperation.from_tuple(op) for op in raw]


def canonical_ops_key(ops: Iterable[TrackingOperation | Sequence[object]]) -> tuple[tuple[str, int, int, int], ...]:
    """Return an order-insensitive key compatible with legacy candidate dedupe"""

    out: list[tuple[str, int, int, int]] = []
    for item in ops:
        op = normalize_operation(item)
        src = int(op.src_label)
        if op.kind == "link":
            assert op.dst1_label is not None
            out.append(("link", src, int(op.dst1_label), -1))
        elif op.kind == "divide":
            assert op.dst1_label is not None and op.dst2_label is not None
            d1, d2 = sorted((int(op.dst1_label), int(op.dst2_label)))
            out.append(("divide", src, d1, d2))
        elif op.kind == "exit":
            out.append(("exit", src, -1, -1))
    out.sort(key=lambda x: (x[1], x[0], x[2], x[3]))
    return tuple(out)
