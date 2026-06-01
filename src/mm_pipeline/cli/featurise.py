"""mm-pipeline featurise CLI handler"""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace

from mm_pipeline.config import TrackerParams

from ._config import load_yaml_config, resolve, section


def configure_parser(parser: ArgumentParser) -> None:
    parser.description = (
        "Compute the 14 pairwise features for an existing candidates parquet, "
        "using labels resolved per dataset from the manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to a dataset manifest CSV/YAML.")
    parser.add_argument("--candidates", required=True, help="Path to a candidates parquet.")
    parser.add_argument("--out", required=True, help="Output features-parquet path.")
    parser.add_argument("--config", default=None, help="Optional YAML config file.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing artefact.")
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
    from mm_pipeline.runners.featurise import run_featurise

    config_data = load_yaml_config(args.config)
    tracker_params = TrackerParams.from_mapping(section(config_data, "tracker"))

    result = run_featurise(
        datasets=args.manifest,
        candidates=args.candidates,
        tracker_params=tracker_params,
        out_path=args.out,
        overwrite=args.overwrite,
    )
    print(f"Wrote {result.output_path}")
    return 0
