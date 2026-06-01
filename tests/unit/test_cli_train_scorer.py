"""Tests for mm-pipeline train-scorer and run_train_scorer"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_pipeline.cli.main import build_parser, main
from mm_pipeline.features import FEATURE_COLUMNS
from mm_pipeline.runners.train_scorer import TrainScorerResult, run_train_scorer
from mm_pipeline.scoring import FittedScorer, load_scorer


def _build_synthetic_features(*, n_pairs: int = 6, n_candidates: int = 3, dataset_id: str = "trench_x"):
    """Build a synthetic features DataFrame with mixed is_correct values."""

    pd = pytest.importorskip("pandas")
    rows = []
    for pair_idx in range(n_pairs):
        for cand_idx in range(n_candidates):
            row = {
                "dataset_id": dataset_id,
                "labels_dir": "/tmp",
                "t": pair_idx,
                "pair_id": f"{dataset_id}:{pair_idx}->{pair_idx + 1}",
                "delta_t": 1,
                "sample_rank": cand_idx + 1,
                "dp_rank_global": cand_idx + 1,
                "dp_cost": 0.1 * (cand_idx + 1),
                "is_dpt_best": cand_idx == 0,
                "candidate_source": "dp_topk",
                "n_candidates_pair": n_candidates,
                "sample_class": "correct" if cand_idx == 0 else "incorrect",
                "is_correct": cand_idx == 0,
                "n_links": 1,
                "n_exits": 0,
                "n_divides": 0,
            }
            for i, feature in enumerate(FEATURE_COLUMNS):
                row[feature] = float(cand_idx) + 0.01 * i + 0.001 * pair_idx
            rows.append(row)
    return pd.DataFrame(rows)


def _build_two_dataset_features():
    pd = pytest.importorskip("pandas")
    df_a = _build_synthetic_features(dataset_id="trench_a")
    df_b = _build_synthetic_features(dataset_id="trench_b")
    return pd.concat([df_a, df_b], ignore_index=True)


def test_cli_train_scorer_help_succeeds(capsys):
    try:
        main(["train-scorer", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "train" in captured.out.lower()
    assert "--features" in captured.out
    assert "--model" in captured.out
    assert "--cv" in captured.out


def test_cli_train_scorer_list_models_succeeds(capsys):
    assert main(["train-scorer", "--list-models", "--features", "x", "--out", "y"]) == 0
    captured = capsys.readouterr()
    assert "logreg_l2_balanced" in captured.out or "random_forest_balanced" in captured.out


def test_cli_train_scorer_requires_args():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train-scorer"])


def test_run_train_scorer_fits_default_model():
    features = _build_synthetic_features()
    result = run_train_scorer(features, model_name="logreg_l2_balanced")

    assert isinstance(result, TrainScorerResult)
    assert isinstance(result.fitted_scorer, FittedScorer)
    assert result.fitted_scorer.model_name == "logreg_l2_balanced"
    assert result.output_path is None
    assert result.cv_metrics is None


def test_run_train_scorer_writes_joblib_and_run_json(tmp_path: Path):
    features = _build_synthetic_features()
    out_path = tmp_path / "model.joblib"

    result = run_train_scorer(
        features,
        model_name="logreg_l2_balanced",
        out_path=out_path,
    )

    assert result.output_path == out_path
    assert out_path.exists()

    sibling = out_path.with_suffix(out_path.suffix + ".run.json")
    assert sibling.exists()
    metadata = json.loads(sibling.read_text())
    assert metadata["command"] == "train-scorer"
    assert metadata["resolved_config"]["train_scorer"]["model_name"] == "logreg_l2_balanced"
    assert metadata["n_features"] == len(result.fitted_scorer.feature_cols)

    # Round-trip via load_scorer.
    loaded = load_scorer(out_path)
    assert loaded.model_name == result.fitted_scorer.model_name


def test_run_train_scorer_overwrite_protection(tmp_path: Path):
    features = _build_synthetic_features()
    out_path = tmp_path / "model.joblib"

    run_train_scorer(features, model_name="logreg_l2_balanced", out_path=out_path)
    with pytest.raises(FileExistsError):
        run_train_scorer(features, model_name="logreg_l2_balanced", out_path=out_path)


def test_run_train_scorer_overwrite_succeeds(tmp_path: Path):
    features = _build_synthetic_features()
    out_path = tmp_path / "model.joblib"

    run_train_scorer(features, model_name="logreg_l2_balanced", out_path=out_path)
    result = run_train_scorer(
        features,
        model_name="logreg_l2_balanced",
        out_path=out_path,
        overwrite=True,
    )
    assert result.output_path == out_path


def test_run_train_scorer_lodo_cv():
    features = _build_two_dataset_features()
    result = run_train_scorer(
        features,
        model_name="logreg_l2_balanced",
        cv="leave_one_dataset_out",
    )

    assert result.cv_metrics is not None
    # Two datasets → two LODO folds.
    assert len(result.cv_metrics) == 2
    assert {"heldout", "model_name"}.issubset(set(result.cv_metrics.columns))
    # The final scorer is also fit.
    assert isinstance(result.fitted_scorer, FittedScorer)


def test_run_train_scorer_unknown_cv_raises():
    features = _build_synthetic_features()
    with pytest.raises(ValueError, match="Unknown cv"):
        run_train_scorer(features, cv="invalid")  # type: ignore[arg-type]


def test_run_train_scorer_loads_features_from_path(tmp_path: Path):
    features = _build_synthetic_features()
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)

    result = run_train_scorer(features_path, model_name="logreg_l2_balanced")
    assert isinstance(result.fitted_scorer, FittedScorer)


def test_run_train_scorer_invalid_features_type_raises():
    with pytest.raises(TypeError, match="features"):
        run_train_scorer(42)  # type: ignore[arg-type]


def test_run_train_scorer_via_public_api():
    from mm_pipeline.runners import TrainScorerResult, run_train_scorer  # noqa: F401
