"""mm-pipeline segment CLI handler."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace

from mm_pipeline.config import SegmentationConfig

from ._config import load_yaml_config, resolve, section


def configure_parser(parser: ArgumentParser) -> None:
    parser.description = "Run a segmentation backend over raw-image datasets."
    parser.add_argument("--manifest", required=True, help="Path to a raw-image manifest CSV/YAML.")
    parser.add_argument("--out", required=True, help="Output directory for the run.")
    parser.add_argument("--backend", default="cpsam", help="Segmentation backend (default: cpsam).")
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
    from mm_pipeline.runners.segment import run_segment

    config_data = load_yaml_config(args.config)
    seg_config = resolve(
        defaults=SegmentationConfig(),
        config_section=section(config_data, "segment"),
    )

    result = run_segment(
        datasets=args.manifest,
        backend=args.backend,
        config=seg_config,
        out_dir=args.out,
        run_tag=args.run_tag,
        overwrite=args.overwrite,
    )
    print(f"Wrote {result.output_dir}")
    return 0
