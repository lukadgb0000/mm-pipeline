"""Candidate-solution contracts"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .operations import TrackingOperation, deserialize_ops_json, normalize_operation, serialize_ops_json


@dataclass(frozen=True)
class CandidateSolution:
    pair_id: str
    ops: tuple[TrackingOperation, ...]
    generator: str
    rank: Optional[int] = None
    cost: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ops(
        cls,
        pair_id: str,
        ops: Sequence[TrackingOperation | Sequence[object]],
        generator: str,
        rank: Optional[int] = None,
        cost: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "CandidateSolution":
        return cls(
            pair_id=pair_id,
            ops=tuple(normalize_operation(op) for op in ops),
            generator=generator,
            rank=rank,
            cost=cost,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_ops_json(
        cls,
        pair_id: str,
        ops_json: str,
        generator: str,
        rank: Optional[int] = None,
        cost: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "CandidateSolution":
        return cls(
            pair_id=pair_id,
            ops=tuple(deserialize_ops_json(ops_json)),
            generator=generator,
            rank=rank,
            cost=cost,
            metadata=dict(metadata or {}),
        )

    def to_ops_json(self) -> str:
        return serialize_ops_json(self.ops)
