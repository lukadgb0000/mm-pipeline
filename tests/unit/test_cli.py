from mm_pipeline.cli.main import main


def test_cli_help_exits_successfully(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "Mother-machine segmentation, tracking, and QA pipeline" in captured.out


def test_all_subcommands_have_help(capsys):
    """Every registered subcommand should respond to --help with exit 0."""

    for cmd in [
        "segment",
        "seg-qc",
        "approve-masks",
        "track-generate",
        "featurise",
        "score",
        "qa",
        "train-scorer",
        "analyse",
    ]:
        try:
            main([cmd, "--help"])
        except SystemExit as exc:
            assert exc.code == 0, f"{cmd} --help did not exit 0"
