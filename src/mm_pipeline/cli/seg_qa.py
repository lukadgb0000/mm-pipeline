"""mm-pipeline seg-qa CLI handler. Headless segmentation QA checks (ie no napari GUI launch)"""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace

from mm_pipeline.config import SegmentationQAConfig

from ._config import load_yaml_config, resolve, section


def configure_parser(parser: ArgumentParser) -> None:
    parser.description = (
        "Run headless segmentation QA checks over a dataset manifest. "
        "No napari; for interactive review use 'mm-pipeline approve-masks'."
    )
    parser.add_argument("--manifest", required=True, help="Path to a dataset manifest CSV/YAML.")
    parser.add_argument("--out", required=True, help="Output directory for the run.")
    parser.add_argument("--config", default=None, help="Optional YAML config file.")
    parser.add_argument("--run-tag", default=None, help="Run tag (default: UTC timestamp).")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing run tag.")
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
    from mm_pipeline.runners.seg_qa import run_seg_qa

    config_data = load_yaml_config(args.config)
    qa_config = resolve(
        defaults=SegmentationQAConfig(),
        config_section=section(config_data, "seg_qa"),
    )

    result = run_seg_qa(
        datasets=args.manifest,
        config=qa_config,
        out_dir=args.out,
        run_tag=args.run_tag,
        overwrite=args.overwrite,
    )
    print(f"Wrote {result.output_dir}")
    return 0
