"""Candidate scoring APIs
"""

from __future__ import annotations

from typing import Any

from .model_registry import DEFAULT_MODEL_NAME, ModelSpec, get_model_registry, get_model_spec, list_models

_LAZY_EXPORTS = {
    "FittedScorer": (".train", "FittedScorer"),
    "NaiveBayesScorer": (".naive_bayes", "NaiveBayesScorer"),
    "add_rule_based_diagnostics": (".rule_based", "add_rule_based_diagnostics"),
    "classify_area_ratio": (".rule_based", "classify_area_ratio"),
    "classify_max_shrink": (".rule_based", "classify_max_shrink"),
    "classify_norm_cost": (".rule_based", "classify_norm_cost"),
    "ensemble_and": (".rule_based", "ensemble_and"),
    "ensemble_or": (".rule_based", "ensemble_or"),
    "fit_lodo_scorers": (".train", "fit_lodo_scorers"),
    "fit_scorer": (".train", "fit_scorer"),
    "fitted_scorer_metadata": (".train", "fitted_scorer_metadata"),
    "load_scorer": (".persistence", "load_scorer"),
    "save_scorer": (".persistence", "save_scorer"),
    "score_candidates": (".predict", "score_candidates"),
    "score_with_lodo_scorers": (".predict", "score_with_lodo_scorers"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "DEFAULT_MODEL_NAME",
    "FittedScorer",
    "ModelSpec",
    "NaiveBayesScorer",
    "add_rule_based_diagnostics",
    "classify_area_ratio",
    "classify_max_shrink",
    "classify_norm_cost",
    "ensemble_and",
    "ensemble_or",
    "fit_lodo_scorers",
    "fit_scorer",
    "fitted_scorer_metadata",
    "get_model_registry",
    "get_model_spec",
    "list_models",
    "load_scorer",
    "save_scorer",
    "score_candidates",
    "score_with_lodo_scorers",
]
