"""mm-pipeline analyse CLI handler

Turns a reconstructed run into the biology-facing artifacts: cycles (+ optional
metrics) and the swimlane / dendrogram plots
"""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace


def configure_parser(parser: ArgumentParser) -> None:
    parser.description = (
        "Cycles, optional cycle metrics, and swimlane/dendrogram plots from a "
        "reconstructed run (track-select / modelvio)"
    )
    parser.add_argument("--run", required=True, help="Run directory (<out>/<run_tag>).")
    parser.add_argument("--dataset", required=True, help="Dataset id within the run.")
    parser.add_argument("--manifest", required=True, help="Path to a dataset manifest CSV/YAML.")
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated cycle metrics, e.g. cycle_time,growth_rate.",
    )
    parser.add_argument(
        "--out", default=None, help="Output directory (default: <run>/<dataset>/analysis)."
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs.")
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
    from mm_pipeline.runners.analyse import run_analyse

    metric_names = tuple(m.strip() for m in args.metrics.split(",")) if args.metrics else ()

    result = run_analyse(
        run_dir=args.run,
        dataset=args.dataset,
        manifest=args.manifest,
        metric_names=metric_names,
        out_dir=args.out,
        overwrite=args.overwrite,
    )
    print(f"Wrote {result.output_dir}")
    return 0
