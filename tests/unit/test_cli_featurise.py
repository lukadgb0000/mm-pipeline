"""Tests for mm-pipeline featurise and run_featurise"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_pipeline.cli.main import build_parser, main
from mm_pipeline.config import DatasetSpec, TrackerParams
from mm_pipeline.features import FEATURE_COLUMNS, SAMPLE_META_COLUMNS, build_feature_table_for_stack
from mm_pipeline.runners.candidates import run_candidates
from mm_pipeline.runners.featurise import FeaturiseResult, run_featurise


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


def test_cli_featurise_help_succeeds(capsys):
    try:
        main(["featurise", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "feature" in captured.out.lower()
    assert "--manifest" in captured.out
    assert "--candidates" in captured.out


def test_cli_featurise_requires_args():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["featurise"])


def test_run_featurise_writes_parquet_and_run_json(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    spec = _build_dataset(tmp_path)
    cand_result = run_candidates(spec, top_k=4)
    out_path = tmp_path / "features.parquet"

    result = run_featurise(
        spec,
        candidates=cand_result.candidates_df,
        out_path=out_path,
    )

    assert isinstance(result, FeaturiseResult)
    assert result.output_path == out_path
    assert out_path.exists()

    sibling = out_path.with_suffix(out_path.suffix + ".run.json")
    assert sibling.exists()
    metadata = json.loads(sibling.read_text())
    assert metadata["command"] == "featurise"
    assert metadata["dataset_ids"] == ["trench_x"]
    assert metadata["n_rows"] > 0

    df = pd.read_parquet(out_path)
    expected = set(SAMPLE_META_COLUMNS) | set(FEATURE_COLUMNS) | {"ops_json"}
    assert set(df.columns) == expected


def test_run_featurise_matches_build_feature_table_for_stack(tmp_path: Path):
    """End-to-end: featurise on candidates parquet == build_feature_table on labels."""

    pd = pytest.importorskip("pandas")
    np = _np()
    spec = _build_dataset(tmp_path)
    cand_result = run_candidates(spec, top_k=4)

    via_featurise = run_featurise(spec, candidates=cand_result.candidates_df)

    from mm_pipeline.io.labels import load_labels_from_folder
    labels = load_labels_from_folder(spec.labels_dir)
    via_full = build_feature_table_for_stack(
        labels,
        dataset_id="trench_x",
        axis="y",
        open_end="high",
        params=TrackerParams(),
        top_k=4,
        store_ops=True,
    )

    assert len(via_featurise.features_df) == len(via_full)
    for col in FEATURE_COLUMNS:
        a = via_featurise.features_df[col].astype(float).reset_index(drop=True)
        b = via_full[col].astype(float).reset_index(drop=True)
        pd.testing.assert_series_equal(a, b, check_names=False)


def test_run_featurise_in_memory_mode(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    cand_result = run_candidates(spec, top_k=4)
    result = run_featurise(spec, candidates=cand_result.candidates_df, out_path=None)
    assert result.output_path is None
    assert len(result.features_df) > 0


def test_run_featurise_reads_candidates_from_path(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    cand_path = tmp_path / "cand.parquet"
    run_candidates(spec, top_k=4, out_path=cand_path)

    result = run_featurise(spec, candidates=cand_path, out_path=None)
    assert len(result.features_df) > 0


def test_run_featurise_overwrite_protection(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    cand_result = run_candidates(spec, top_k=4)
    out_path = tmp_path / "features.parquet"

    run_featurise(spec, candidates=cand_result.candidates_df, out_path=out_path)
    with pytest.raises(FileExistsError):
        run_featurise(spec, candidates=cand_result.candidates_df, out_path=out_path)


def test_run_featurise_overwrite_succeeds(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    cand_result = run_candidates(spec, top_k=4)
    out_path = tmp_path / "features.parquet"

    run_featurise(spec, candidates=cand_result.candidates_df, out_path=out_path)
    result = run_featurise(
        spec,
        candidates=cand_result.candidates_df,
        out_path=out_path,
        overwrite=True,
    )
    assert result.output_path == out_path


def test_run_featurise_unknown_dataset_raises(tmp_path: Path):
    sub_a = tmp_path / "a"
    sub_a.mkdir()
    spec_a = _build_dataset(sub_a)
    spec_b = DatasetSpec(dataset_id="trench_other", labels_dir=sub_a / "labels")

    cand_result = run_candidates(spec_a, top_k=4)
    # Use a manifest that doesn't include 'trench_x' (the dataset in the candidates).
    with pytest.raises(ValueError, match="not in manifest"):
        run_featurise([spec_b], candidates=cand_result.candidates_df)


def test_run_featurise_via_public_api():
    from mm_pipeline.runners import FeaturiseResult, run_featurise  # noqa: F401


def test_run_featurise_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        run_featurise([], candidates=None)


def test_run_featurise_invalid_candidates_type_raises(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    with pytest.raises(TypeError, match="candidates"):
        run_featurise(spec, candidates=42)  # type: ignore[arg-type]
