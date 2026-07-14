"""Tests for the physical-error detector implementations"""

from __future__ import annotations

import math

import pytest

from mm_pipeline.modelvio.aggregation import build_per_pair_features, per_pair_feature_columns
from mm_pipeline.modelvio.detectors import (
    HistGBMModelViolationDetector,
    NeverAnomalous,
    build_detector,
    load_detector,
    save_detector,
    train_detector,
)


def _pd():
    return pytest.importorskip("pandas")


def _per_pair_fixture(n_pairs: int = 6, seed: int = 0):
    pd = _pd()
    rng = pytest.importorskip("numpy").random.default_rng(seed)
    rows = []
    for i in range(n_pairs):
        row = {col: float(rng.normal(0.0, 1.0)) for col in per_pair_feature_columns()}
        row["pair_id"] = f"p{i}"
        row["dataset_id"] = "d"
        row["t"] = i
        row["n_candidates"] = 3
        rows.append(row)
    return pd.DataFrame(rows, columns=["dataset_id", "pair_id", "t", "n_candidates", *per_pair_feature_columns()[4:]])


def test_never_anomalous_returns_no_flags():
    pd = _pd()
    df = _per_pair_fixture()
    detector = NeverAnomalous()
    out = detector.detect(df)
    assert (out["anomaly_flag"] == False).all()  # noqa: E712
    assert out["anomaly_score"].isna().all()
    assert detector.name == "never_anomalous"


def test_build_detector_resolves_never_anomalous():
    detector = build_detector("never_anomalous")
    assert isinstance(detector, NeverAnomalous)


def test_train_detector_trains_and_calibrates(tmp_path):
    pytest.importorskip("sklearn")
    pd = _pd()
    rng = pytest.importorskip("numpy").random.default_rng(0)
    rows = []
    family_ids = []
    for fam in ("a", "b", "c"):
        for j in range(60):
            has_error = bool(j % 2 == 0)
            shift = 1.5 if has_error else -1.5
            row = {col: float(rng.normal(shift, 0.7)) for col in per_pair_feature_columns()[4:]}
            row["pair_id"] = f"{fam}:{j}"
            row["dataset_id"] = fam
            row["t"] = j
            row["n_candidates"] = 3
            row["has_error"] = int(has_error)
            family_ids.append(fam)
            rows.append(row)
    df = pd.DataFrame(rows)
    df["family_id"] = family_ids

    detector = train_detector(df, target_recall=0.9, cv_groups=df["family_id"])
    assert isinstance(detector, HistGBMModelViolationDetector)
    assert 0.0 <= detector.threshold <= 1.0
    assert detector.estimator is not None
    assert detector.training_summary["n_rows"] == len(df)

    # Round-trip via save/load.
    path = save_detector(detector, tmp_path / "phys.joblib")
    loaded = load_detector(path)
    out_a = detector.detect(df)
    out_b = loaded.detect(df)
    assert out_a["anomaly_flag"].tolist() == out_b["anomaly_flag"].tolist()


def test_default_loader_errors_when_artifact_missing():
    with pytest.raises(FileNotFoundError, match="Default model-violation detector"):
        HistGBMModelViolationDetector.default()
