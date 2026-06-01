"""Tests for mm-pipeline qa and run_qa"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_pipeline.cli.main import build_parser, main
from mm_pipeline.config import DatasetSpec, QAConfig
from mm_pipeline.features import FEATURE_COLUMNS
from mm_pipeline.runners.candidates import run_candidates
from mm_pipeline.runners.featurise import run_featurise
from mm_pipeline.runners.qa import QAResult, run_qa


def _np():
    return pytest.importorskip("numpy")


def _build_dataset(tmp_path: Path, *, n_frames: int = 4) -> DatasetSpec:
    np = _np()
    tiff = pytest.importorskip("tifffile")

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for t in range(n_frames):
        labels = np.zeros((8, 8), dtype=np.uint32)
        labels[1 + t : 3 + t, 1:3] = 1
        tiff.imwrite(labels_dir / f"frame_{t:03d}.tif", labels)

    return DatasetSpec(dataset_id="trench_x", labels_dir=labels_dir, axis="y", open_end="high")


def test_cli_qa_help_succeeds(capsys):
    try:
        main(["qa", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "qa" in captured.out.lower()
    assert "--manifest" in captured.out
    assert "--scored" in captured.out
    assert "--features" in captured.out
    assert "--candidates" in captured.out


def test_cli_qa_requires_manifest_and_out():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["qa"])


def test_run_qa_dp_baseline_with_candidates(tmp_path: Path):
    """Default QAConfig + candidates input → DP baseline lineage."""

    spec = _build_dataset(tmp_path)
    cand = run_candidates(spec, top_k=4)

    result = run_qa(spec, candidates=cand.candidates_df)

    assert isinstance(result, QAResult)
    assert "trench_x" in result.tracks_by_dataset
    tracks = result.tracks_by_dataset["trench_x"]
    # Three frames, one cell → 3 track rows with same track_id.
    assert len(tracks) > 0
    assert tracks["track_id"].nunique() == 1


def test_run_qa_writes_per_dataset_outputs(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    spec = _build_dataset(tmp_path)
    cand = run_candidates(spec, top_k=4)
    out_dir = tmp_path / "qa_run"

    result = run_qa(spec, candidates=cand.candidates_df, out_dir=out_dir, run_tag="v1")

    dataset_dir = out_dir / "v1" / "trench_x"
    for filename in ("tracks.csv", "events.csv", "division_events.csv", "qa_decisions.csv"):
        assert (dataset_dir / filename).exists()

    summary = json.loads((out_dir / "v1" / "summary.json").read_text())
    assert summary["command"] == "qa"
    assert summary["dataset_ids"] == ["trench_x"]
    assert summary["n_decisions_total"] > 0


def test_run_qa_in_memory_mode(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    cand = run_candidates(spec, top_k=4)
    result = run_qa(spec, candidates=cand.candidates_df, out_dir=None)
    assert result.output_dir is None


def test_run_qa_overwrite_protection(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    cand = run_candidates(spec, top_k=4)
    out_dir = tmp_path / "qa_run"

    run_qa(spec, candidates=cand.candidates_df, out_dir=out_dir, run_tag="v1")
    with pytest.raises(FileExistsError):
        run_qa(spec, candidates=cand.candidates_df, out_dir=out_dir, run_tag="v1")


def test_run_qa_overwrite_succeeds(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    cand = run_candidates(spec, top_k=4)
    out_dir = tmp_path / "qa_run"

    run_qa(spec, candidates=cand.candidates_df, out_dir=out_dir, run_tag="v1")
    result = run_qa(
        spec, candidates=cand.candidates_df, out_dir=out_dir, run_tag="v1", overwrite=True,
    )
    assert result.output_dir == out_dir / "v1"


def test_run_qa_missing_required_column_dp_cost(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    spec = _build_dataset(tmp_path)
    cand = run_candidates(spec, top_k=4)

    # Drop dp_cost and try DP-baseline scoring.
    broken = cand.candidates_df.drop(columns=["dp_cost"])
    with pytest.raises(ValueError, match="dp_cost"):
        run_qa(spec, candidates=broken)


def test_run_qa_classifier_requires_raw_score(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    cand = run_candidates(spec, top_k=4)
    cfg = QAConfig(within_pair_scorer="classifier")

    with pytest.raises(ValueError, match="raw_score"):
        run_qa(spec, candidates=cand.candidates_df, qa_config=cfg)


def test_run_qa_bridge_enabled_requires_model_and_raw_score(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    cand = run_candidates(spec, top_k=4)
    cfg = QAConfig(bridge_enabled=True)

    # No model → raises about raw_score first (column validation happens first).
    with pytest.raises(ValueError, match="raw_score"):
        run_qa(spec, candidates=cand.candidates_df, qa_config=cfg)


def test_run_qa_missing_pair_id_raises(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    spec = _build_dataset(tmp_path)
    bogus = pd.DataFrame({"dataset_id": ["trench_x"], "dp_cost": [1.0]})
    with pytest.raises(ValueError, match="pair_id"):
        run_qa(spec, candidates=bogus)


def test_run_qa_requires_some_input(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    with pytest.raises(ValueError, match="scored, features, candidates"):
        run_qa(spec)


def test_run_qa_unknown_dataset_in_input_returns_no_rows(tmp_path: Path):
    """Dataset in manifest but absent from candidates → silently skipped."""

    spec = _build_dataset(tmp_path)
    cand = run_candidates(spec, top_k=4)

    # Use a different DatasetSpec that doesn't match the candidates.
    spec_b = DatasetSpec(dataset_id="orphan", labels_dir=tmp_path / "labels")
    result = run_qa([spec_b], candidates=cand.candidates_df)
    assert "orphan" not in result.tracks_by_dataset


def test_run_qa_with_anomaly_detector_requires_features(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    cand = run_candidates(spec, top_k=4)
    cfg = QAConfig(anomaly_detector="hist_gbm_default")

    with pytest.raises(ValueError, match="feature columns"):
        run_qa(spec, candidates=cand.candidates_df, qa_config=cfg)


def test_run_qa_features_input_works(tmp_path: Path):
    """qa accepts a featurised parquet directly (DP baseline)."""

    spec = _build_dataset(tmp_path)
    cand = run_candidates(spec, top_k=4)
    feat = run_featurise(spec, candidates=cand.candidates_df)

    result = run_qa(spec, features=feat.features_df)
    assert "trench_x" in result.tracks_by_dataset


def test_run_qa_records_per_pair_features_when_features_present(tmp_path: Path):
    spec = _build_dataset(tmp_path)
    cand = run_candidates(spec, top_k=4)
    feat = run_featurise(spec, candidates=cand.candidates_df)

    result = run_qa(spec, features=feat.features_df)
    assert "trench_x" in result.per_pair_features_by_dataset


def test_run_qa_via_public_api():
    from mm_pipeline.runners import QAResult, run_qa  # noqa: F401


def test_run_qa_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        run_qa([])
