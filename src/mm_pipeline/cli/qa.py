"""mm-pipeline qa CLI handler"""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace

from mm_pipeline.config import QAConfig, TrackerParams

from ._config import load_yaml_config, resolve, section


def configure_parser(parser: ArgumentParser) -> None:
    parser.description = (
        "End-to-end QA workflow: within-pair picking + per-pair anomaly "
        "detection + drop/bridge + lineage reconstruction. Input artefact "
        "can be a scored, featurised, or unscored candidates parquet — the "
        "runner validates required columns against the resolved QAConfig."
    )
    parser.add_argument("--manifest", required=True, help="Path to a dataset manifest CSV/YAML.")
    parser.add_argument("--scored", default=None, help="Path to a scored parquet (preferred).")
    parser.add_argument("--features", default=None, help="Path to a features parquet.")
    parser.add_argument("--candidates", default=None, help="Path to a candidates parquet.")
    parser.add_argument("--model", default=None, help="Path to a FittedScorer joblib (used for bridging).")
    parser.add_argument("--anomaly-model", default=None, help="Anomaly-detector name or path (overrides config).")
    parser.add_argument("--config", default=None, help="Optional YAML config file.")
    parser.add_argument("--out", required=True, help="Output directory for the qa run.")
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
    from mm_pipeline.runners.qa import run_qa

    config_data = load_yaml_config(args.config)
    qa_config = QAConfig.from_mapping(section(config_data, "qa"))
    tracker_params = TrackerParams.from_mapping(section(config_data, "tracker"))

    result = run_qa(
        datasets=args.manifest,
        scored=args.scored,
        features=args.features,
        candidates=args.candidates,
        qa_config=qa_config,
        model=args.model,
        anomaly_model=args.anomaly_model,
        tracker_params=tracker_params,
        out_dir=args.out,
        run_tag=args.run_tag,
        overwrite=args.overwrite,
    )
    print(f"Wrote {result.output_dir}")
    return 0
