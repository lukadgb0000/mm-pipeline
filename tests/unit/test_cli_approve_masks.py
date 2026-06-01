"""Tests for the approve masks argparse surface"""

from __future__ import annotations

import pytest

from mm_pipeline.cli.main import build_parser, main


def test_cli_approve_masks_help_succeeds(capsys):
    try:
        main(["approve-masks", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "napari" in captured.out.lower()
    assert "--images" in captured.out
    assert "--labels" in captured.out
    assert "--out" in captured.out


def test_cli_approve_masks_requires_images_and_labels(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["approve-masks"])


def test_cli_approve_masks_parses_minimum_args():
    parser = build_parser()
    args = parser.parse_args(
        ["approve-masks", "--images", "/a", "--labels", "/b"],
    )
    assert args.images == "/a"
    assert args.labels == "/b"
    assert args.out is None
    assert args.overwrite is False


def test_cli_approve_masks_parses_optional_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "approve-masks",
            "--images", "/a",
            "--labels", "/b",
            "--out", "/c",
            "--overwrite",
            "--dataset-id", "trench84",
        ],
    )
    assert args.out == "/c"
    assert args.overwrite is True
    assert args.dataset_id == "trench84"


def test_run_approve_masks_module_imports():
    """Confirm the runner module loads without napari at import time."""

    from mm_pipeline.runners import approve_masks as runner_module

    assert hasattr(runner_module, "run_approve_masks")
    assert hasattr(runner_module, "ApproveMasksResult")


def test_run_approve_masks_via_public_api():
    """Confirm the lazy-loaded export works."""

    from mm_pipeline.runners import ApproveMasksResult, run_approve_masks  # noqa: F401
