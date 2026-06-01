"""mm-pipeline train-scorer CLI handler"""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace

from mm_pipeline.scoring import DEFAULT_MODEL_NAME, list_models


def configure_parser(parser: ArgumentParser) -> None:
    parser.description = (
        "Train a candidate-plausibility scorer from a labelled features parquet. "
        "Requires an 'is_correct' column. Optionally runs leave-one-dataset-out "
        "cross-validation alongside the final fit."
    )
    parser.add_argument("--features", required=True, help="Path to a labelled features parquet.")
    parser.add_argument("--out", required=True, help="Output joblib path for the trained scorer.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Model registry key (default: {DEFAULT_MODEL_NAME}). Use --list-models to see options.",
    )
    parser.add_argument("--feature-subset", default="all_features", help="Feature subset name or 'all_features'.")
    parser.add_argument("--target-col", default="is_correct", help="Binary target column (default 'is_correct').")
    parser.add_argument("--calibrate", action="store_true", help="Fit a probability calibrator alongside the base estimator.")
    parser.add_argument(
        "--cv",
        choices=["none", "leave_one_dataset_out"],
        default="none",
        help="Cross-validation strategy (default: none).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing artefact.")
    parser.add_argument("--list-models", action="store_true", help="Print the registry of available model names and exit.")
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
    if args.list_models:
        for name in list_models():
            print(name)
        return 0

    from mm_pipeline.runners.train_scorer import run_train_scorer

    result = run_train_scorer(
        features=args.features,
        model_name=args.model,
        feature_subset=args.feature_subset,
        target_col=args.target_col,
        calibrate=args.calibrate,
        cv=args.cv,
        out_path=args.out,
        overwrite=args.overwrite,
    )
    print(f"Wrote {result.output_path}")
    return 0
