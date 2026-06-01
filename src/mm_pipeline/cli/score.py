"""mm-pipeline score CLI handler"""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace


def configure_parser(parser: ArgumentParser) -> None:
    parser.description = (
        "Apply a trained scorer (FittedScorer joblib) to a featurised "
        "candidates parquet. Appends raw_score, candidate_correctness_"
        "probability, pair_probability, score_rank, and other score columns."
    )
    parser.add_argument("--features", required=True, help="Path to a features parquet.")
    parser.add_argument("--model", required=True, help="Path to a FittedScorer joblib.")
    parser.add_argument("--out", required=True, help="Output scored-parquet path.")
    parser.add_argument(
        "--pair-temperature",
        type=float,
        default=1.0,
        help="Within-pair softmax temperature (default 1.0).",
    )
    parser.add_argument("--pair-col", default="pair_id", help="Column identifying pairs (default pair_id).")
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
    from mm_pipeline.runners.score import run_score

    result = run_score(
        features=args.features,
        model=args.model,
        pair_temperature=args.pair_temperature,
        pair_col=args.pair_col,
        out_path=args.out,
        overwrite=args.overwrite,
    )
    print(f"Wrote {result.output_path}")
    return 0
