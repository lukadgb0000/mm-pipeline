"""Unit tests for mm_pipeline.analysis.metrics (registry + built-ins + driver)."""

from __future__ import annotations

import math

import pytest

pd = pytest.importorskip("pandas")

import numpy as np

from mm_pipeline.analysis import Lineage, cycle_metric, get_cycle_metric, list_cycle_metrics, metrics, roots
from mm_pipeline.analysis.metrics import _REGISTRY, _loglinear_fit
from mm_pipeline.config import TrackerParams
from mm_pipeline.tracking.lineage import reconstruct_lineage
from mm_pipeline.tracking.select import DPCostMin, select_pairs


def _stable_label_stack() -> np.ndarray:
    labels = np.zeros((5, 40, 8), dtype=np.int32)
    for t in range(5):
        labels[t, 2:8, 1:6] = 1
        labels[t, 14:22, 1:6] = 2
    return labels


def _labelled_lineage(tmp_path, frame_interval_min=None) -> Lineage:
    from mm_pipeline.features import build_feature_table_for_stack
    from mm_pipeline.io.labels import save_label_stack

    labels = _stable_label_stack()
    names = [f"f{t:03d}.tif" for t in range(labels.shape[0])]
    save_label_stack(labels, names, tmp_path, overwrite=True)

    features = build_feature_table_for_stack(
        labels, dataset_id="d1", axis="y", open_end="high",
        params=TrackerParams(), top_k=4, store_ops=True,
    )
    selections = select_pairs(features, DPCostMin())
    tracks, events, divisions = reconstruct_lineage(selections, labels, open_end="high", axis="y")
    return Lineage(
        dataset_id="d1", tracks_df=tracks, divisions_df=divisions, events_df=events,
        axis="y", open_end="high", frame_interval_min=frame_interval_min, labels_dir=tmp_path,
    )


# --- registry -----------------------------------------------------------------


def test_builtins_are_registered():
    assert set(list_cycle_metrics()) >= {"cycle_time", "birth_length", "added_length", "growth_rate"}


def test_unknown_metric_raises():
    with pytest.raises(KeyError, match="Unknown cycle metric"):
        get_cycle_metric("does_not_exist")


def test_decorator_registers_and_returns_unchanged():
    @cycle_metric("__unit_test_metric__")
    def _m(ctx):
        return 1

    try:
        assert get_cycle_metric("__unit_test_metric__") is _m
    finally:
        _REGISTRY.pop("__unit_test_metric__", None)


# --- log-linear fit -----------------------------------------------------------


def test_loglinear_fit_recovers_exponential():
    x = np.arange(5, dtype=float)
    slope, r2, n = _loglinear_fit(x, np.exp(0.1 * x))
    assert slope == pytest.approx(0.1)
    assert r2 == pytest.approx(1.0)
    assert n == 5


def test_loglinear_fit_too_few_points_is_nan():
    slope, r2, n = _loglinear_fit(np.array([0.0]), np.array([5.0]))
    assert n == 1 and math.isnan(slope) and math.isnan(r2)


# --- cycle_time (does not need properties) ------------------------------------


def test_cycle_time_plus_one_convention(tmp_path):
    lin = _labelled_lineage(tmp_path, frame_interval_min=2.0)
    out = metrics(lin, roots(lin), ["cycle_time"])
    # Each track spans t=0..4 -> (4 - 0 + 1) * 2.0 = 10.0 min.
    assert (out["cycle_time"] == 10.0).all()
    assert "generation" in out.columns  # joined onto cycles


def test_property_metric_raises_without_properties(tmp_path):
    lin = _labelled_lineage(tmp_path)
    with pytest.raises(ValueError, match="requires with_properties=True"):
        metrics(lin, roots(lin), ["birth_length"], with_properties=False)


# --- property-based metrics ---------------------------------------------------


def test_property_metrics_with_properties(tmp_path):
    lin = _labelled_lineage(tmp_path, frame_interval_min=1.0)
    out = metrics(
        lin, roots(lin), ["birth_length", "added_length", "growth_rate"], with_properties=True
    )
    for col in ("birth_length", "added_length", "growth_rate", "growth_rate_r2", "growth_rate_n"):
        assert col in out.columns
    assert (out["birth_length"] > 0).all()
    # Constant-size cells: no length added, ~zero growth, all 5 frames used.
    assert out["added_length"].abs().max() == pytest.approx(0.0, abs=1e-9)
    assert (out["growth_rate_n"] == 5).all()
    assert out["growth_rate"].abs().max() == pytest.approx(0.0, abs=1e-9)
