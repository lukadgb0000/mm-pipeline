"""Phase 9 tracking-QA package.

Public API. Modules with optional dependencies (pandas, sklearn, scipy,
joblib) are imported lazily so importing the package does not require them.
"""

from __future__ import annotations

from typing import Any

from .decisions import Action, DropReason, QADecision, validate_decision

_LAZY_EXPORTS = {
    "build_per_pair_features": (".aggregation", "build_per_pair_features"),
    "per_pair_feature_columns": (".aggregation", "per_pair_feature_columns"),
    "DPCostMin": (".within_pair", "DPCostMin"),
    "ClassifierMax": (".within_pair", "ClassifierMax"),
    "Ensemble": (".within_pair", "Ensemble"),
    "WithinPairPick": (".within_pair", "WithinPairPick"),
    "WithinPairScorer": (".within_pair", "WithinPairScorer"),
    "build_scorer": (".within_pair", "build_scorer"),
    "NeverAnomalous": (".physical_errors", "NeverAnomalous"),
    "HistGBMPhysicalErrorDetector": (".physical_errors", "HistGBMPhysicalErrorDetector"),
    "PhysicalErrorDetector": (".physical_errors", "PhysicalErrorDetector"),
    "build_detector": (".physical_errors", "build_detector"),
    "load_detector": (".physical_errors", "load_detector"),
    "save_detector": (".physical_errors", "save_detector"),
    "train_detector": (".physical_errors", "train_detector"),
    "ReuseCandidateClassifier": (".bridge", "ReuseCandidateClassifier"),
    "BridgeScorer": (".bridge", "BridgeScorer"),
    "bridge_drops": (".bridge", "bridge_drops"),
    "apply_qa_workflow": (".workflow", "apply_qa_workflow"),
    "decisions_to_dataframe": (".reports", "decisions_to_dataframe"),
    "write_qa_decisions_csv": (".reports", "write_qa_decisions_csv"),
    "write_lineage_outputs": (".reports", "write_lineage_outputs"),
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


__all__ = [
    "Action",
    "BridgeScorer",
    "ClassifierMax",
    "DPCostMin",
    "DropReason",
    "Ensemble",
    "HistGBMPhysicalErrorDetector",
    "NeverAnomalous",
    "PhysicalErrorDetector",
    "QADecision",
    "ReuseCandidateClassifier",
    "WithinPairPick",
    "WithinPairScorer",
    "apply_qa_workflow",
    "bridge_drops",
    "build_detector",
    "build_per_pair_features",
    "build_scorer",
    "decisions_to_dataframe",
    "load_detector",
    "per_pair_feature_columns",
    "save_detector",
    "train_detector",
    "validate_decision",
    "write_lineage_outputs",
    "write_qa_decisions_csv",
]
