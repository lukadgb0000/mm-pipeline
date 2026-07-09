"""Tests for mm-pipeline score and run_score"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_pipeline.cli.main import build_parser, main
from mm_pipeline.config import DatasetSpec
from mm_pipeline.features import FEATURE_COLUMNS
from mm_pipeline.runners.track_generate import run_track_generate
from mm_pipeline.runners.featurise import run_featurise
from mm_pipeline.runners.score import ScoreResult, run_score
from mm_pipeline.scoring import fit_scorer, save_scorer


def _np():
    return pytest.importorskip("numpy")


def _build_synthetic_features():
    """Build a synthetic features DataFrame with mixed is_correct values."""

    pd = pytest.importorskip("pandas")
    rows = []
    for pair_idx in range(6):
        for cand_idx in range(3):
            row = {
                "dataset_id": "trench_x",
                "labels_dir": "/tmp",
                "t": pair_idx,
                "pair_id": f"trench_x:{pair_idx}->{pair_idx + 1}",
                "delta_t": 1,
                "sample_rank": cand_idx + 1,
                "dp_rank_global": cand_idx + 1,
                "dp_cost": 0.1 * (cand_idx + 1),
                "is_dpt_best": cand_idx == 0,
                "candidate_source": "dp_topk",
                "n_candidates_pair": 3,
                "sample_class": "correct" if cand_idx == 0 else "incorrect",
                "is_correct": cand_idx == 0,
                "n_links": 1,
                "n_exits": 0,
                "n_divides": 0,
            }
            for i, feature in enumerate(FEATURE_COLUMNS):
                # Make the feature value depend on cand_idx so the classifier
                # actually has signal.
                row[feature] = float(cand_idx) + 0.01 * i + 0.001 * pair_idx
            rows.append(row)
    return pd.DataFrame(rows)


def _build_dataset_and_features(tmp_path: Path):
    return None, _build_synthetic_features()


def test_cli_score_help_succeeds(capsys):
    try:
        main(["score", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "score" in captured.out.lower()
    assert "--features" in captured.out
    assert "--model" in captured.out


def test_cli_score_requires_args():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["score"])


def test_run_score_applies_model_and_adds_columns(tmp_path: Path):
    spec, features = _build_dataset_and_features(tmp_path)

    scorer = fit_scorer(features, model_name="logreg_l2_balanced")
    result = run_score(features, model=scorer)

    assert isinstance(result, ScoreResult)
    assert result.output_path is None
    # Score columns appended by score_candidates.
    for col in (
        "raw_score",
        "candidate_correctness_probability",
        "pair_probability",
        "score_rank",
        "y_score",
        "score_model",
        "score_feature_subset",
    ):
        assert col in result.scored_df.columns


def test_run_score_writes_parquet_and_run_json(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    spec, features = _build_dataset_and_features(tmp_path)

    scorer = fit_scorer(features, model_name="logreg_l2_balanced")
    model_path = tmp_path / "model.joblib"
    save_scorer(scorer, model_path)

    out_path = tmp_path / "scored.parquet"
    result = run_score(features, model=model_path, out_path=out_path)

    assert result.output_path == out_path
    assert out_path.exists()

    sibling = out_path.with_suffix(out_path.suffix + ".run.json")
    assert sibling.exists()
    metadata = json.loads(sibling.read_text())
    assert metadata["command"] == "score"
    assert metadata["resolved_config"]["score"]["model_name"] == scorer.model_name
    assert metadata["resolved_config"]["score"]["model_path"] == str(model_path)
    assert metadata["n_rows"] > 0


def test_run_score_loads_features_from_path(tmp_path: Path):
    spec, features = _build_dataset_and_features(tmp_path)
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)

    scorer = fit_scorer(features, model_name="logreg_l2_balanced")
    result = run_score(features_path, model=scorer)
    assert len(result.scored_df) == len(features)


def test_run_score_overwrite_protection(tmp_path: Path):
    spec, features = _build_dataset_and_features(tmp_path)
    scorer = fit_scorer(features, model_name="logreg_l2_balanced")
    out_path = tmp_path / "scored.parquet"
    run_score(features, model=scorer, out_path=out_path)
    with pytest.raises(FileExistsError):
        run_score(features, model=scorer, out_path=out_path)


def test_run_score_overwrite_succeeds(tmp_path: Path):
    spec, features = _build_dataset_and_features(tmp_path)
    scorer = fit_scorer(features, model_name="logreg_l2_balanced")
    out_path = tmp_path / "scored.parquet"
    run_score(features, model=scorer, out_path=out_path)
    result = run_score(features, model=scorer, out_path=out_path, overwrite=True)
    assert result.output_path == out_path


def test_run_score_invalid_features_type_raises(tmp_path: Path):
    spec, features = _build_dataset_and_features(tmp_path)
    scorer = fit_scorer(features, model_name="logreg_l2_balanced")
    with pytest.raises(TypeError, match="features"):
        run_score(42, model=scorer)  # type: ignore[arg-type]


def test_run_score_invalid_model_type_raises(tmp_path: Path):
    spec, features = _build_dataset_and_features(tmp_path)
    with pytest.raises(TypeError, match="model"):
        run_score(features, model=42)  # type: ignore[arg-type]


def test_run_score_via_public_api():
    from mm_pipeline.runners import ScoreResult, run_score  # noqa: F401
