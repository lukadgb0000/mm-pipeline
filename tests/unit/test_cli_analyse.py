"""Tests for the analyse CLI command and its runner."""

from __future__ import annotations

import pytest

pytest.importorskip("skimage")
pd = pytest.importorskip("pandas")

import numpy as np

from mm_pipeline.cli.main import build_parser, main
from mm_pipeline.config import TrackerParams
from mm_pipeline.io.labels import save_label_stack
from mm_pipeline.io.tracks import write_lineage_outputs
from mm_pipeline.tracking.lineage import reconstruct_lineage
from mm_pipeline.tracking.select import DPCostMin, select_pairs

DATASET = "d1"


def _stable_label_stack() -> np.ndarray:
    labels = np.zeros((5, 40, 8), dtype=np.int32)
    for t in range(5):
        labels[t, 2:8, 1:6] = 1
        labels[t, 14:22, 1:6] = 2
    return labels


def _build_run(tmp_path):
    """Write a labels dir, a reconstructed run, and a manifest; return their paths."""
    from mm_pipeline.features import build_feature_table_for_stack

    labels = _stable_label_stack()
    labels_dir = tmp_path / "labels"
    save_label_stack(labels, [f"f{t:03d}.tif" for t in range(labels.shape[0])], labels_dir)

    features = build_feature_table_for_stack(
        labels, dataset_id=DATASET, axis="y", open_end="high",
        params=TrackerParams(), top_k=4, store_ops=True,
    )
    tracks, events, divisions = reconstruct_lineage(
        select_pairs(features, DPCostMin()), labels, open_end="high", axis="y"
    )
    run_dir = tmp_path / "run"
    write_lineage_outputs(tracks, events, divisions, run_dir / DATASET)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "dataset_id,labels_dir,axis,open_end,frame_interval_min\n"
        f"{DATASET},{labels_dir},y,high,1.0\n"
    )
    return run_dir, manifest


def test_analyse_help_exits_zero():
    try:
        main(["analyse", "--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_analyse_requires_run_and_dataset_and_manifest():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["analyse", "--run", "x"])  # missing --dataset/--manifest


def test_run_analyse_writes_cycles_metrics_and_plots(tmp_path):
    from mm_pipeline.runners.analyse import run_analyse

    run_dir, manifest = _build_run(tmp_path)
    out = tmp_path / "analysis"
    result = run_analyse(
        run_dir=run_dir, dataset=DATASET, manifest=manifest,
        metric_names=("cycle_time", "growth_rate"), out_dir=out, overwrite=True,
    )
    assert result.output_dir == out
    for name in ("cycles.csv", "swimlane.png", "dendrogram.png", "summary.json"):
        assert (out / name).exists(), f"missing {name}"

    cycles = pd.read_csv(out / "cycles.csv")
    for col in ("cycle_time", "growth_rate", "growth_rate_r2", "growth_rate_n"):
        assert col in cycles.columns
    assert (cycles["cycle_time"] == 5.0).all()  # 5 frames * 1.0 min


def test_run_analyse_without_metrics_writes_plain_cycles(tmp_path):
    from mm_pipeline.runners.analyse import run_analyse

    run_dir, manifest = _build_run(tmp_path)
    run_analyse(run_dir=run_dir, dataset=DATASET, manifest=manifest, out_dir=tmp_path / "a2")
    cycles = pd.read_csv(tmp_path / "a2" / "cycles.csv")
    assert "cycle_time" not in cycles.columns  # no metrics requested
    assert "generation" in cycles.columns


def test_run_analyse_unknown_dataset_raises(tmp_path):
    from mm_pipeline.runners.analyse import run_analyse

    run_dir, manifest = _build_run(tmp_path)
    with pytest.raises(ValueError, match="not in manifest"):
        run_analyse(run_dir=run_dir, dataset="nope", manifest=manifest, out_dir=tmp_path / "a3")
