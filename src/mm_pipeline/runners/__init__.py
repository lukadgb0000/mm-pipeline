"""pipeline entry points

Each runner is the canonical orchestrator for one pipeline stage and is
shared by the CLI handler and notebook users
importing directly from mm_pipeline.runners.
"""

from __future__ import annotations

from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ApproveMasksResult": (".approve_masks", "ApproveMasksResult"),
    "CandidatesResult": (".candidates", "CandidatesResult"),
    "FeaturiseResult": (".featurise", "FeaturiseResult"),
    "QAResult": (".qa", "QAResult"),
    "ScoreResult": (".score", "ScoreResult"),
    "SegmentResult": (".segment", "SegmentResult"),
    "SegQAResult": (".seg_qa", "SegQAResult"),
    "TrainScorerResult": (".train_scorer", "TrainScorerResult"),
    "run_approve_masks": (".approve_masks", "run_approve_masks"),
    "run_candidates": (".candidates", "run_candidates"),
    "run_featurise": (".featurise", "run_featurise"),
    "run_qa": (".qa", "run_qa"),
    "run_score": (".score", "run_score"),
    "run_segment": (".segment", "run_segment"),
    "run_seg_qa": (".seg_qa", "run_seg_qa"),
    "run_train_scorer": (".train_scorer", "run_train_scorer"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_LAZY_EXPORTS)
