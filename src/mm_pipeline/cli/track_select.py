"""mm-pipeline track-select CLI handler"""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace

from ._config import load_yaml_config, section


def configure_parser(parser: ArgumentParser) -> None:
    parser.description = (
        "Pick the best candidate per frame-pair and reconstruct tracks. "
        "no anomaly / bridge handling - see modelvio"
    )
    parser.add_argument("--manifest", required=True, help="Path to a dataset manifest CSV/YAML.")
    parser.add_argument("--scored", default=None, help="Scored candidates parquet (from 'score').")
    parser.add_argument("--features", default=None, help="Featurised candidates parquet (from 'featurise').")
    parser.add_argument("--candidates", default=None, help="Raw candidates parquet (from 'track-generate').")
    parser.add_argument(
        "--scorer",
        choices=["dp_cost_min", "classifier", "ensemble"],
        default="dp_cost_min",
        help="Within-pair scorer (default: dp_cost_min).",
    )
    parser.add_argument("--out", required=True, help="Output run directory.")
    parser.add_argument("--run-tag", default=None, help="Run tag (default: UTC timestamp).")
    parser.add_argument("--config", default=None, help="Optional YAML config file.")
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
    from mm_pipeline.runners.track_select import run_track_select

    config_data = load_yaml_config(args.config)
    ts_section = section(config_data, "track_select")

    # CLI flag wins over config; config wins over default.
    scorer = args.scorer if args.scorer != "dp_cost_min" else str(ts_section.get("scorer", "dp_cost_min"))
    ensemble_alpha = float(ts_section.get("ensemble_alpha", 0.5))
    ensemble_mode = str(ts_section.get("ensemble_mode", "rank"))

    result = run_track_select(
        datasets=args.manifest,
        scored=args.scored,
        features=args.features,
        candidates=args.candidates,
        scorer=scorer,  # type: ignore[arg-type]
        ensemble_alpha=ensemble_alpha,
        ensemble_mode=ensemble_mode,
        out_dir=args.out,
        run_tag=args.run_tag,
        overwrite=args.overwrite,
    )
    print(f"Wrote {result.output_dir}")
    return 0
