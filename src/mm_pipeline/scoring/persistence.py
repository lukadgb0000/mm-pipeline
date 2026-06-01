"""Persistence helpers for fitted scorers"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .train import FittedScorer


def _require_joblib():
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Scorer persistence requires joblib. Install the package with the 'scoring' extra.") from exc
    return joblib


def save_scorer(fitted_scorer: FittedScorer, path: str | Path) -> Path:
    """Serialise a fitted scorer with joblib."""

    joblib = _require_joblib()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_scorer, out_path)
    return out_path


def load_scorer(path: str | Path) -> FittedScorer:
    """Load a scorer saved by :func:`save_scorer`."""

    joblib = _require_joblib()
    loaded: Any = joblib.load(Path(path))
    if not isinstance(loaded, FittedScorer):
        raise TypeError(f"Expected FittedScorer, loaded {type(loaded).__name__}.")
    return loaded
