"""Segmentation backend package"""

from .base import PrecomputedLabelsBackend, SegmenterBackend
from .batch import run_segmentation
from .cpsam import CPSAMBackend
from .evaluation import (
    InstanceMatch,
    InstanceSegmentationEvaluation,
    InstanceSegmentationSummary,
    aggregate_instance_evaluations,
    evaluate_instance_labels,
)
from .validation import LabelStackValidation, validate_label_directory, validate_label_stack

__all__ = [
    "CPSAMBackend",
    "InstanceMatch",
    "InstanceSegmentationEvaluation",
    "InstanceSegmentationSummary",
    "LabelStackValidation",
    "PrecomputedLabelsBackend",
    "SegmenterBackend",
    "aggregate_instance_evaluations",
    "evaluate_instance_labels",
    "run_segmentation",
    "validate_label_directory",
    "validate_label_stack",
]
