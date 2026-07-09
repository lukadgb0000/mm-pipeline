"""Command-line main place"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence

from mm_pipeline import __version__


_COMMAND_MODULES: list[tuple[str, str, str]] = [
    # (cli command name, module name under cli/, help text)
    ("segment", "segment", "Run a segmentation backend over raw images."),
    ("seg-qc", "seg_qc", "Run headless segmentation QC checks."),
    ("approve-masks", "approve_masks", "Launch napari to review and approve masks."),
    ("track-generate", "track_generate", "Generate tracking candidates from labels."),
    ("featurise", "featurise", "Compute the 14 pairwise features for candidates."),
    ("score", "score", "Apply a trained scorer to featurised candidates."),
    ("qa", "qa", "Run the QA workflow: pick / detect / drop / bridge / reconstruct."),
    ("track-select", "track_select", "Pick best candidate per pair and reconstruct tracks."),
    ("train-scorer", "train", "Train a candidate-plausibility scorer."),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mm-pipeline",
        description="Mother-machine segmentation, tracking, and QA pipeline.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")
    for cli_name, module_name, help_text in _COMMAND_MODULES:
        cmd = subparsers.add_parser(cli_name, help=help_text)
        try:
            module = importlib.import_module(f".{module_name}", __package__)
            if hasattr(module, "configure_parser"):
                module.configure_parser(cmd)
            else:
                cmd.set_defaults(_placeholder=cli_name)
        except ImportError:
            cmd.set_defaults(_placeholder=cli_name)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "_handler"):
        return int(args._handler(args))
    if getattr(args, "_placeholder", None):
        parser.error(f"command '{args.command}' is not implemented yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
