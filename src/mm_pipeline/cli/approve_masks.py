"""mm-pipeline approve-masks CLI handler. Note: this launches napari"""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace


def configure_parser(parser: ArgumentParser) -> None:
    parser.description = (
        "Launch napari to review and approve segmentation masks for one "
        "dataset. Press 's' in napari to save edited labels."
    )
    parser.add_argument("--images", required=True, help="Directory of raw images.")
    parser.add_argument("--labels", required=True, help="Directory of label TIFFs.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for approved labels (default: <labels>_edited sibling).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing back into --labels.")
    parser.add_argument("--dataset-id", default="", help="Optional dataset identifier for metadata.")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.set_defaults(_handler=run)


def _log_level(verbosity: int) -> int:
    if verbosity >= 2:
        return logging.DEBUG
    if verbosity >= 1:
        return logging.INFO
    return logging.WARNING


def run(args: Namespace) -> int:
    logging.basicConfig(level=_log_level(args.verbose))
    from mm_pipeline.runners.approve_masks import run_approve_masks

    result = run_approve_masks(
        images_dir=args.images,
        labels_dir=args.labels,
        out_dir=args.out,
        overwrite=args.overwrite,
        dataset_id=args.dataset_id,
    )
    print(f"Wrote {result.output_dir}")
    return 0
