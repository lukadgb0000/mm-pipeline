"""Model-violation detection package (experimental)

Go-forward home for anomaly / model-violation detection + drop/bridge handling
and the per-pair decision contract, relocated from ``qa`` (which now re-exports
these via shims). Off by default: the only working detector is the no-op
``NeverAnomalous``. Modules with optional dependencies (pandas, sklearn, scipy,
joblib) are imported lazily so importing the package does not require them
"""

from __future__ import annotations

from typing import Any

from .decisions import Action, DropReason, QADecision, validate_decision

_LAZY_EXPORTS = {
    "build_per_pair_features": (".aggregation", "build_per_pair_features"),
    "per_pair_feature_columns": (".aggregation", "per_pair_feature_columns"),
    "NeverAnomalous": (".detectors", "NeverAnomalous"),
    "ModelViolationDetector": (".detectors", "ModelViolationDetector"),
    "HistGBMModelViolationDetector": (".detectors", "HistGBMModelViolationDetector"),
    "build_detector": (".detectors", "build_detector"),
    "load_detector": (".detectors", "load_detector"),
    "save_detector": (".detectors", "save_detector"),
    "train_detector": (".detectors", "train_detector"),
    "ReuseCandidateClassifier": (".bridge", "ReuseCandidateClassifier"),
    "BridgeScorer": (".bridge", "BridgeScorer"),
    "BridgeAttempt": (".bridge", "BridgeAttempt"),
    "bridge_drops": (".bridge", "bridge_drops"),
    "apply_qa_workflow": (".workflow", "apply_qa_workflow"),
    "decisions_to_dataframe": (".reports", "decisions_to_dataframe"),
    "write_qa_decisions_csv": (".reports", "write_qa_decisions_csv"),
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
    "BridgeAttempt",
    "BridgeScorer",
    "DropReason",
    "HistGBMModelViolationDetector",
    "ModelViolationDetector",
    "NeverAnomalous",
    "QADecision",
    "ReuseCandidateClassifier",
    "apply_qa_workflow",
    "bridge_drops",
    "build_detector",
    "build_per_pair_features",
    "decisions_to_dataframe",
    "load_detector",
    "per_pair_feature_columns",
    "save_detector",
    "train_detector",
    "validate_decision",
    "write_qa_decisions_csv",
]
