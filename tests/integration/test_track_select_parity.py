"""Parity harness: run_track_select == run_qa on the mainline

Both notebook examples run with anomaly and bridge OFF, so track-select
(within-pair pick + pure KEEP/DROP reconstruction) must reproduce qa
exactly. Self-contained synthetic labels (per the repo's integration-test
convention) rather than the untracked example_data/ stack. Covers the DP and
classifier within-pair paths plus the written CSVs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mm_pipeline.config import DatasetSpec, QAConfig
from mm_pipeline.runners.featurise import run_featurise
from mm_pipeline.runners.qa import run_qa
from mm_pipeline.runners.track_generate import run_track_generate
from mm_pipeline.runners.track_select import run_track_select


def _pd():
    return pytest.importorskip("pandas")


def _spec(tmp_path: Path) -> DatasetSpec:
    tiff = pytest.importorskip("tifffile")
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for t in range(5):
        frame = np.zeros((40, 8), dtype=np.int32)
        frame[2:8, 1:6] = 1
        frame[14:22, 1:6] = 2
        tiff.imwrite(labels_dir / f"frame_{t:03d}.tif", frame)
    return DatasetSpec(dataset_id="d1", labels_dir=labels_dir, axis="y", open_end="high")


def _features(spec: DatasetSpec):
    cand = run_track_generate(spec, top_k=8, out_path=None)
    feat = run_featurise(spec, candidates=cand.candidates_df, out_path=None)
    return feat.features_df


def _assert_lineage_equal(a, b) -> None:
    pd = _pd()
    for key in ("tracks_by_dataset", "events_by_dataset", "divisions_by_dataset"):
        da = getattr(a, key)["d1"].reset_index(drop=True)
        db = getattr(b, key)["d1"].reset_index(drop=True)
        pd.testing.assert_frame_equal(da, db)


def test_track_select_matches_qa_dp_path(tmp_path: Path):
    spec = _spec(tmp_path)
    features = _features(spec)

    qa_res = run_qa(spec, features=features)
    ts_res = run_track_select(spec, features=features, scorer="dp_cost_min")

    _assert_lineage_equal(qa_res, ts_res)
    assert not qa_res.tracks_by_dataset["d1"].empty  # the mainline produced tracks


def test_track_select_matches_qa_classifier_path(tmp_path: Path):
    spec = _spec(tmp_path)
    scored = _features(spec).copy()
    # Without a fitted classifier, mirror DP ordering so both paths agree on a well-defined raw_score (the equality is what matters, not the values
    scored["raw_score"] = -scored["dp_cost"].astype(float)

    qa_res = run_qa(spec, scored=scored, qa_config=QAConfig(within_pair_scorer="classifier"))
    ts_res = run_track_select(spec, scored=scored, scorer="classifier")

    _assert_lineage_equal(qa_res, ts_res)


def test_track_select_matches_qa_written_csvs(tmp_path: Path):
    spec = _spec(tmp_path)
    features = _features(spec)

    run_qa(spec, features=features, out_dir=tmp_path / "qa_out", run_tag="v1")
    run_track_select(
        spec, features=features, scorer="dp_cost_min",
        out_dir=tmp_path / "ts_out", run_tag="v1",
    )

    for name in ("tracks.csv", "events.csv", "division_events.csv"):
        qa_csv = (tmp_path / "qa_out" / "v1" / "d1" / name).read_text()
        ts_csv = (tmp_path / "ts_out" / "v1" / "d1" / name).read_text()
        assert qa_csv == ts_csv, f"{name} differs between qa and track-select"
