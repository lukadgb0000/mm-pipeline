"""Tests for pure tracking-review helpers and the notebook session wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from mm_pipeline.analysis import Lineage
from mm_pipeline.config import DatasetSpec, TrackerParams
from mm_pipeline.core import extract_cell_instances, sort_cells_along_trench
from mm_pipeline.io.labels import save_label_stack
from mm_pipeline.runners import run_track_generate, run_track_select
from mm_pipeline.tracking import (
    TrackingReviewSession,
    format_compact_ops,
    infer_ops_from_compact,
    parse_compact_kinds,
)


def _labels() -> np.ndarray:
    labels = np.zeros((3, 30, 8), dtype=np.uint32)
    for t in range(3):
        labels[t, 20:27, 1:7] = 1  # open-end cell for open_end=high
        labels[t, 5:13, 1:7] = 2
    return labels


def _build_results(tmp_path: Path, *, top_k: int = 4):
    labels_dir = tmp_path / "labels"
    save_label_stack(_labels(), ["f0.tif", "f1.tif", "f2.tif"], labels_dir)
    spec = DatasetSpec(
        dataset_id="d1",
        approved_labels_dir=labels_dir,
        axis="y",
        open_end="high",
    )
    candidates = run_track_generate(spec, tracker_params=TrackerParams(), top_k=top_k)
    selected = run_track_select(spec, candidates=candidates.candidates_df)
    return spec, candidates, selected


@pytest.mark.parametrize("open_end", ["low", "high"])
def test_human_and_internal_orders_are_always_exact_reverses(open_end):
    img_t = np.zeros((30, 8), dtype=np.uint32)
    img_t[4:10, 1:7] = 1
    img_t[20:27, 1:7] = 2
    img_k = np.zeros_like(img_t)
    # Only the closed-end cell remains: human closed->open is link,exit.
    if open_end == "high":
        img_k[4:10, 1:7] = 10
    else:
        img_k[20:27, 1:7] = 10
    cells_t = sort_cells_along_trench(extract_cell_instances(img_t), "y", open_end)
    cells_k = sort_cells_along_trench(extract_cell_instances(img_k), "y", open_end)

    ops = infer_ops_from_compact("le", cells_t, cells_k)

    assert [op.kind for op in ops] == ["exit", "link"]
    assert format_compact_ops(ops) == "le"


def test_compact_parser_accepts_letters_words_and_tuple_syntax():
    expected = ("link", "divide", "exit")
    assert parse_compact_kinds("lde") == expected
    assert parse_compact_kinds("(l, d, e)") == expected
    assert parse_compact_kinds(["link", "divide", "exit"]) == expected
    with pytest.raises(ValueError, match="Unknown operation token"):
        parse_compact_kinds("lx")


def test_manual_inference_rejects_bad_exit_suffix_and_consumption():
    cells_t = sort_cells_along_trench(extract_cell_instances(_labels()[0]), "y", "high")
    one_dest = cells_t[:1]
    with pytest.raises(ValueError, match="open-end suffix"):
        infer_ops_from_compact("el", cells_t, one_dest)
    with pytest.raises(ValueError, match="Expected 2 operations"):
        infer_ops_from_compact("l", cells_t, one_dest)
    with pytest.raises(ValueError, match="consumes 0 destination"):
        infer_ops_from_compact("ee", cells_t, one_dest)


def test_session_presents_and_finds_candidates_exactly(tmp_path):
    spec, candidates, selected = _build_results(tmp_path)
    session = TrackingReviewSession(spec, candidates, selected)

    table = session.pair_candidates(0)
    assert set(table["operations"]) == {"ll", "de"}
    assert table["selected"].sum() == 1
    match = session.find_candidate(0, "de")
    assert match is not None
    assert int(match["sample_rank"]) in set(table["rank"])


def test_rank_and_manual_selection_record_exact_candidate_metadata(tmp_path):
    spec, candidates, selected = _build_results(tmp_path)
    session = TrackingReviewSession(spec, candidates, selected)
    matched = session.find_candidate(0, "de")
    assert matched is not None
    rank = int(matched["sample_rank"])

    by_rank = session.select_candidate(0, rank, note="candidate choice")
    assert by_rank.source == "candidate"
    assert by_rank.candidate_rank == rank
    assert by_rank.dp_cost == pytest.approx(float(matched["dp_cost"]))

    typed = session.select_manual(0, "(d,e)", note="typed after review")
    assert typed.source == "manual"
    assert typed.candidate_rank == rank
    assert typed.dp_cost == pytest.approx(float(matched["dp_cost"]))
    assert session.corrections.iloc[0]["corrected_operations"] == "de"

    session.clear_correction(0)
    assert session.corrections.empty


def test_non_generated_manual_solution_gets_exact_recomputed_cost(tmp_path):
    spec, candidates, selected = _build_results(tmp_path, top_k=1)
    session = TrackingReviewSession(spec, candidates, selected)
    assert session.find_candidate(0, "de") is None

    correction = session.select_manual(0, "de")

    assert correction.source == "manual"
    assert correction.candidate_rank is None
    assert np.isfinite(correction.dp_cost)


def test_reconstruct_applies_all_later_label_ops_after_track_ids_change(tmp_path):
    spec, candidates, selected = _build_results(tmp_path)
    session = TrackingReviewSession(spec, candidates, selected)
    session.select_manual(0, "de")

    corrected = session.reconstruct()
    lin = Lineage.from_result(corrected, spec)

    assert len(lin.divisions_df) == 1
    assert set(lin.tracks_df["t"]) == {0, 1, 2}
    assert len(lin.events_df.loc[lin.events_df["t"] == 1]) == 2
    assert set(lin.events_df.loc[lin.events_df["t"] == 1, "event"]) == {"link"}


def test_save_writes_new_run_with_only_one_additional_dataset_artifact(tmp_path):
    spec, candidates, selected = _build_results(tmp_path / "inputs")
    session = TrackingReviewSession(spec, candidates, selected)
    session.select_manual(0, "de", note="audited")
    original = tmp_path / "outputs" / "original"
    original.mkdir(parents=True)
    (original / "marker.txt").write_text("unchanged")

    result = session.reconstruct(out_dir=tmp_path / "outputs")

    assert result.output_dir == tmp_path / "outputs" / "tracking_corrected"
    dataset_dir = result.output_dir / "d1"
    assert {path.name for path in dataset_dir.iterdir()} == {
        "tracks.csv",
        "events.csv",
        "division_events.csv",
        "corrections.csv",
    }
    assert (original / "marker.txt").read_text() == "unchanged"
    with pytest.raises(FileExistsError):
        session.reconstruct(out_dir=tmp_path / "outputs")


def test_save_refuses_nonempty_run_even_without_summary(tmp_path):
    spec, candidates, selected = _build_results(tmp_path / "inputs")
    session = TrackingReviewSession(spec, candidates, selected)
    incomplete_run = tmp_path / "outputs" / "tracking_corrected"
    incomplete_run.mkdir(parents=True)
    (incomplete_run / "unrelated.txt").write_text("preserve me")

    with pytest.raises(FileExistsError, match="is not empty"):
        session.reconstruct(out_dir=tmp_path / "outputs")

    assert (incomplete_run / "unrelated.txt").read_text() == "preserve me"


def test_session_rejects_candidate_geometry_from_different_labels(tmp_path):
    spec, candidates, selected = _build_results(tmp_path / "original")
    changed = _labels()
    changed[1][changed[1] == 2] = 0
    changed_dir = tmp_path / "changed"
    save_label_stack(changed, ["f0.tif", "f1.tif", "f2.tif"], changed_dir)
    changed_spec = DatasetSpec(
        dataset_id="d1",
        approved_labels_dir=changed_dir,
        axis="y",
        open_end="high",
    )
    table_without_source_path = candidates.candidates_df.drop(columns=["labels_dir"])

    with pytest.raises(ValueError, match="no longer matches labels"):
        TrackingReviewSession(
            changed_spec,
            table_without_source_path,
            selected,
            tracker_params=TrackerParams(),
        )


def test_table_only_session_requires_exact_tracker_params(tmp_path):
    spec, candidates, selected = _build_results(tmp_path)

    with pytest.raises(ValueError, match="exact parameters"):
        TrackingReviewSession(spec, candidates.candidates_df, selected)

    session = TrackingReviewSession(
        spec,
        candidates.candidates_df,
        selected,
        tracker_params=TrackerParams(),
    )
    assert session.tracker_params == TrackerParams()
