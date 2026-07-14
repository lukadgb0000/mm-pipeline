"""Re-export shim: detectors now live in ``modelvio/detectors.py`` (renamed
physical_error -> model_violation). The old names are kept here as back-compat
aliases so ``qa``'s public API is unchanged. Do not add logic here
"""

from __future__ import annotations

from mm_pipeline.modelvio.detectors import (  # noqa: F401
    HistGBMModelViolationDetector as HistGBMPhysicalErrorDetector,
    ModelViolationDetector as PhysicalErrorDetector,
    NeverAnomalous,
    build_detector,
    load_detector,
    save_detector,
    train_detector,
)

__all__ = [
    "HistGBMPhysicalErrorDetector",
    "NeverAnomalous",
    "PhysicalErrorDetector",
    "build_detector",
    "load_detector",
    "save_detector",
    "train_detector",
]
