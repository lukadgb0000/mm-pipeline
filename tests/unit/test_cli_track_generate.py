"""Tests for mm-pipeline track-generate and run_track_generate"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_pipeline.cli.main import build_parser, main
from mm_pipeline.config import DatasetSpec, HypothesisModel, TrackerParams
from mm_pipeline.runners.track_generate import TrackGenerateResult, run_track_generate


def _np():
    return pytest.importorskip("numpy")


def _build_dataset(tmp_path: Path) -> DatasetSpec:
    np = _np()
    tiff = pytest.importorskip("tifffile")

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    for t in range(3):
        labels = np.zeros((8, 8), dtype=np.uint32)
        labels[1 + t : 3 + t, 1:3] = 1
        tiff.imwrite(labels_dir / f"frame_{t:03d}.tif", labels)

    return DatasetSpec(
        dataset_id="trench_x",
        labels_dir=labels_dir,
        axis="y",
        open_end="high",
    )


def test_cli_track_generate_help_succeeds(capsys):
    try:
        main(["track-generate", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "candidate" in captured.out.lower()
    assert "--manifest" in captured.out
    assert "--sampler" in captured.out
    assert "--top-k" in captured.out


def test_cli_track_generate_sampler_argparse_validates():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["track-generate", "--manifest", "/m.csv", "--out", "/o.parquet", "--sampler", "bogus"],
        )


def test_run_track_generate_writes_parquet_and_run_json(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    spec = _build_dataset(tmp_path)
    out_path = tmp_path / "candidates.parquet"

    result = run_track_generate(spec, top_k=4, out_path=out_path)

    assert isinstance(result, TrackGenerateResult)
    assert result.output_path == out_path
    assert out_path.exists()

    sibling = out_path.with_suffix(out_path.suffix + ".run.json")
    assert sibling.exists()
    metadata = json.loads(sibling.read_text())
    assert metadata["command"] == "track-generate"
    assert metadata["dataset_ids"] == ["trench_x"]
    assert metadata["n_pairs_total"] == 2
    assert metadata["resolved_config"]["track_generate"]["sampler"] == "dp"
    assert metadata["resolved_config"]["track_generate"]["top_k"] == 4

    df = pd.read_parquet(out_path)
    assert len(df) > 0
    assert "ops_json" in df.columns
    assert "pair_id" in df.columns
    assert "dp_cost" in df.columns
    # No feature columns.
    assert "max_shrink_pct" not in df.columns


def test_run_track_generate_in_memory_mode(tmp_path: Path):
    spec = _build_dataset(tmp_path)

    result = run_track_generate(spec, top_k=4, out_path=None)
    assert result.output_path is None
    assert len(result.candidates_df) > 0
    assert "trench_x" in result.runs_by_dataset


def test_run_track_generate_brute_force_raises_notimplemented(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    with pytest.raises(NotImplementedError, match="brute_force"):
        run_track_generate(spec, sampler="brute_force")


def test_run_track_generate_unknown_sampler_raises(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    with pytest.raises(ValueError, match="Unknown sampler"):
        run_track_generate(spec, sampler="nonsense")  # type: ignore[arg-type]


def test_run_track_generate_overwrite_protection(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    out_path = tmp_path / "candidates.parquet"

    run_track_generate(spec, top_k=4, out_path=out_path)
    with pytest.raises(FileExistsError):
        run_track_generate(spec, top_k=4, out_path=out_path)


def test_run_track_generate_overwrite_succeeds(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    out_path = tmp_path / "candidates.parquet"

    run_track_generate(spec, top_k=4, out_path=out_path)
    result = run_track_generate(spec, top_k=4, out_path=out_path, overwrite=True)
    assert result.output_path == out_path


def test_run_track_generate_missing_labels_raises(tmp_path: Path):
    spec = DatasetSpec(
        dataset_id="orphan",
        images_dir=tmp_path,  # only images, no labels
    )
    with pytest.raises(ValueError, match="labels"):
        run_track_generate(spec)


def test_run_track_generate_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        run_track_generate([])


def test_run_track_generate_unknown_hypothesis_model_raises(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    with pytest.raises(ValueError, match="Unknown hypothesis model"):
        run_track_generate(
            spec,
            hypothesis_model=HypothesisModel.from_mapping({"name": "lysis"}),
        )


def test_run_track_generate_via_public_api():
    """Confirm the lazy-loaded export works."""

    from mm_pipeline.runners import TrackGenerateResult, run_track_generate  # noqa: F401


def test_run_track_generate_accepts_manifest_csv(tmp_path: Path):
    """Confirm str/Path manifest input works and metadata captures path."""

    spec = _build_dataset(tmp_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "dataset_id,labels_dir,axis,open_end\n"
        f"{spec.dataset_id},{spec.labels_dir},{spec.axis},{spec.open_end}\n"
    )
    out_path = tmp_path / "out.parquet"
    result = run_track_generate(manifest, top_k=4, out_path=out_path)

    sibling = out_path.with_suffix(out_path.suffix + ".run.json")
    metadata = json.loads(sibling.read_text())
    assert metadata["manifest_path"] == str(manifest)
    assert result.output_path == out_path
