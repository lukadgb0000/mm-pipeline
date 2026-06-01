"""Tests for mm_pipeline.runners._outputs"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_pipeline.runners._outputs import (
    make_run_metadata,
    resolve_run_tag,
    write_multifile_outputs,
    write_single_artefact_outputs,
)


def test_resolve_run_tag_explicit_passes_through():
    assert resolve_run_tag("v1") == "v1"


def test_resolve_run_tag_none_generates_timestamp():
    tag = resolve_run_tag(None)
    # Format: YYYY-MM-DDTHHMMSSZ (length 18)
    assert len(tag) == 18
    assert tag[4] == "-"
    assert tag[7] == "-"
    assert tag[10] == "T"
    assert tag.endswith("Z")


def test_make_run_metadata_basic():
    meta = make_run_metadata(
        command="qa",
        manifest_path="/abs/manifest.csv",
        resolved_config={"within_pair_scorer": "dp_cost_min"},
        dataset_ids=["trench84", "trench31"],
    )
    assert meta["command"] == "qa"
    assert meta["manifest_path"] == "/abs/manifest.csv"
    assert meta["dataset_ids"] == ["trench84", "trench31"]
    assert meta["resolved_config"] == {"within_pair_scorer": "dp_cost_min"}
    assert "git_commit" in meta  # may be None
    assert "created_at" in meta
    # created_at is ISO-ish: YYYY-MM-DDTHHMMSSZ
    assert meta["created_at"].endswith("Z")


def test_make_run_metadata_none_manifest():
    meta = make_run_metadata(
        command="approve-masks",
        manifest_path=None,
        resolved_config={},
        dataset_ids=["one"],
    )
    assert meta["manifest_path"] is None


def test_write_multifile_outputs_creates_summary_and_report(tmp_path: Path):
    out = tmp_path / "run"
    metadata = make_run_metadata(
        command="qa", manifest_path=None,
        resolved_config={"bridge_enabled": False},
        dataset_ids=["d1"],
    )
    summary = {"n_pairs": 10, "n_dropped": 1}
    paths = write_multifile_outputs(
        out_dir=out, summary=summary, metadata=metadata, title="Test Run",
    )
    assert paths["summary"].exists()
    assert paths["report"].exists()
    data = json.loads(paths["summary"].read_text())
    assert data["n_pairs"] == 10
    assert data["command"] == "qa"
    assert data["dataset_ids"] == ["d1"]
    text = paths["report"].read_text()
    assert "Test Run" in text


def test_write_multifile_outputs_no_overwrite_raises(tmp_path: Path):
    out = tmp_path / "run"
    metadata = make_run_metadata(
        command="qa", manifest_path=None, resolved_config={}, dataset_ids=[],
    )
    write_multifile_outputs(out_dir=out, summary={}, metadata=metadata, title="t")
    with pytest.raises(FileExistsError):
        write_multifile_outputs(out_dir=out, summary={}, metadata=metadata, title="t")


def test_write_multifile_outputs_overwrite_succeeds(tmp_path: Path):
    out = tmp_path / "run"
    metadata = make_run_metadata(
        command="qa", manifest_path=None, resolved_config={}, dataset_ids=[],
    )
    write_multifile_outputs(out_dir=out, summary={"v": 1}, metadata=metadata, title="t")
    write_multifile_outputs(
        out_dir=out, summary={"v": 2}, metadata=metadata, title="t", overwrite=True,
    )
    data = json.loads((out / "summary.json").read_text())
    assert data["v"] == 2


def test_write_multifile_outputs_with_tables(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    out = tmp_path / "run"
    metadata = make_run_metadata(
        command="evaluate", manifest_path=None, resolved_config={}, dataset_ids=[],
    )
    table = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    paths = write_multifile_outputs(
        out_dir=out, summary={}, metadata=metadata, title="t",
        tables={"per_dataset": table},
    )
    assert "per_dataset" in paths
    assert paths["per_dataset"].exists()
    loaded = pd.read_csv(paths["per_dataset"])
    assert loaded["a"].tolist() == [1, 2]


def test_write_single_artefact_outputs_dataframe(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    out = tmp_path / "candidates.parquet"
    metadata = make_run_metadata(
        command="candidates", manifest_path=None, resolved_config={}, dataset_ids=["d1"],
    )
    df = pd.DataFrame({"pair_id": ["d1__t000_t001"], "dp_cost": [1.0]})
    paths = write_single_artefact_outputs(
        out_path=out, artefact=df, metadata=metadata, summary={"n_rows": 1},
    )
    assert paths["artefact"].exists()
    assert paths["run_metadata"].exists()
    assert paths["run_metadata"].name == "candidates.parquet.run.json"
    data = json.loads(paths["run_metadata"].read_text())
    assert data["command"] == "candidates"
    assert data["n_rows"] == 1
    loaded = pd.read_parquet(out)
    assert loaded["dp_cost"].tolist() == [1.0]


def test_write_single_artefact_outputs_no_overwrite_raises(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    out = tmp_path / "x.parquet"
    out.write_bytes(b"placeholder")
    metadata = make_run_metadata(
        command="candidates", manifest_path=None, resolved_config={}, dataset_ids=[],
    )
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(FileExistsError):
        write_single_artefact_outputs(out_path=out, artefact=df, metadata=metadata)


def test_write_single_artefact_outputs_overwrite_succeeds(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    out = tmp_path / "x.parquet"
    out.write_bytes(b"placeholder")
    metadata = make_run_metadata(
        command="candidates", manifest_path=None, resolved_config={}, dataset_ids=[],
    )
    df = pd.DataFrame({"a": [1, 2]})
    paths = write_single_artefact_outputs(
        out_path=out, artefact=df, metadata=metadata, overwrite=True,
    )
    loaded = pd.read_parquet(paths["artefact"])
    assert loaded["a"].tolist() == [1, 2]
