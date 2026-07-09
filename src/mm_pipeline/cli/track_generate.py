"""mm-pipeline candidates CLI handler."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace

from mm_pipeline.config import HypothesisModel, TrackerParams

from ._config import load_yaml_config, resolve, section


def configure_parser(parser: ArgumentParser) -> None:
    parser.description = (
        "Generate top-K candidate (t, t+1) mappings from approved labels. "
        "Default sampler is DP; --sampler brute_force is exposed but raises "
        "NotImplementedError until that sampler lands."
    )
    parser.add_argument("--manifest", required=True, help="Path to a dataset manifest CSV/YAML.")
    parser.add_argument("--out", required=True, help="Output parquet path for candidates.")
    parser.add_argument("--top-k", type=int, default=16, help="Candidates per pair (default: 16).")
    parser.add_argument(
        "--sampler",
        choices=["dp", "brute_force"],
        default="dp",
        help="Candidate sampler (default: dp).",
    )
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
    from mm_pipeline.runners.track_generate import run_track_generate

    config_data = load_yaml_config(args.config)
    tracker_params = TrackerParams.from_mapping(section(config_data, "tracker"))

    track_generate_section = section(config_data, "track_generate")
    hm_section = track_generate_section.get("hypothesis_model") or {}
    if not isinstance(hm_section, dict):
        raise ValueError("'track_generate.hypothesis_model' must be a mapping.")
    hypothesis_model = HypothesisModel.from_mapping(hm_section)

    # CLI flag wins over config; config wins over default 16.
    top_k = args.top_k if args.top_k != 16 else int(track_generate_section.get("top_k", 16))
    sampler = args.sampler if args.sampler != "dp" else str(track_generate_section.get("sampler", "dp"))

    result = run_track_generate(
        datasets=args.manifest,
        tracker_params=tracker_params,
        hypothesis_model=hypothesis_model,
        sampler=sampler,  # type: ignore[arg-type]
        top_k=top_k,
        out_path=args.out,
        overwrite=args.overwrite,
    )
    print(f"Wrote {result.output_path}")
    return 0
