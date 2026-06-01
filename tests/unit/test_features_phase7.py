import math

import pytest

from mm_pipeline.config import DatasetSpec, TrackerParams
from mm_pipeline.core import CandidateSolution, FramePair, extract_cell_instances, sort_cells_along_trench
from mm_pipeline.features import (
    FEATURE_COLUMNS,
    SAMPLE_META_COLUMNS,
    FeatureContext,
    build_feature_dataframe,
    build_feature_table_for_stack,
    compute_solution_features,
    get_feature_subsets,
    resolve_feature_subset,
    solve_and_featurize_pair,
)


LEGACY_FEATURE_COLUMNS = (
    "max_shrink_pct",
    "total_area_ratio_exit_adjusted",
    "exit_open_end_dist_median_norm",
    "link_area_ratio_median",
    "link_area_ratio_max",
    "link_dy_median_norm",
    "link_dy_max_norm",
    "link_iou_shifted_median",
    "div_mother_sum_area_ratio_max",
    "div_mother_sum_area_ratio_mean",
    "div_daughter_area_ratio_max",
    "div_daughter_area_ratio_mean",
    "div_mother_daughter_dy_max_norm",
    "div_mother_daughter_dy_mean_norm",
)


def _np():
    return pytest.importorskip("numpy")


def _pd():
    return pytest.importorskip("pandas")


def _sorted_cells(label_img, *, dataset_id="trench_a", frame=0, axis="y", open_end="high"):
    return tuple(
        sort_cells_along_trench(
            extract_cell_instances(label_img, dataset_id=dataset_id, frame=frame),
            axis=axis,
            open_end=open_end,
        )
    )


def test_feature_columns_match_legacy_order():
    assert FEATURE_COLUMNS == LEGACY_FEATURE_COLUMNS


def test_feature_set_resolution_supports_named_and_custom_sets():
    subsets = get_feature_subsets()

    assert subsets["all_features"] == list(FEATURE_COLUMNS)
    assert subsets["non_division_features"] == list(FEATURE_COLUMNS[:8])
    assert subsets["division_features"] == list(FEATURE_COLUMNS[8:])
    assert resolve_feature_subset("reduced_v1") == subsets["reduced_v1"]
    assert resolve_feature_subset(["max_shrink_pct"]) == ["max_shrink_pct"]

    with pytest.raises(KeyError, match="Unknown feature subset"):
        resolve_feature_subset("missing")
    with pytest.raises(ValueError, match="empty"):
        resolve_feature_subset([])


def test_compute_solution_features_link_metrics_and_no_division_nans():
    np = _np()
    label_t = np.zeros((12, 10), dtype=np.uint32)
    label_k = np.zeros((12, 10), dtype=np.uint32)
    label_t[3:5, 2:4] = 1
    label_k[6:7, 2:4] = 10

    cells_t = _sorted_cells(label_t, frame=0, open_end="high")
    cells_k = _sorted_cells(label_k, frame=1, open_end="high")
    pair = FramePair("trench_a", 0, 1, (12, 10), "y", "high")
    candidate = CandidateSolution.from_ops(pair.pair_id, [("link", 1, 10, None)], "manual")

    features = compute_solution_features(
        FeatureContext(label_t, label_k, cells_t, cells_k, candidate, pair, TrackerParams())
    )

    assert features["n_links"] == 1.0
    assert features["n_exits"] == 0.0
    assert features["n_divides"] == 0.0
    assert features["max_shrink_pct"] == pytest.approx(50.0)
    assert features["total_area_ratio_exit_adjusted"] == pytest.approx(0.5)
    assert features["link_area_ratio_median"] == pytest.approx(2.0)
    assert features["link_area_ratio_max"] == pytest.approx(2.0)
    assert features["link_dy_median_norm"] == pytest.approx(2.5 / 12.0)
    assert features["link_dy_max_norm"] == pytest.approx(2.5 / 12.0)
    assert features["link_iou_shifted_median"] == pytest.approx(0.5)
    assert math.isnan(features["div_mother_sum_area_ratio_max"])
    assert math.isnan(features["div_mother_daughter_dy_mean_norm"])


def test_compute_solution_features_exit_metrics():
    np = _np()
    label_t = np.zeros((12, 10), dtype=np.uint32)
    label_k = np.zeros((12, 10), dtype=np.uint32)
    label_t[8:9, 2:4] = 1
    label_t[3:5, 2:4] = 2
    label_k[3:5, 2:4] = 20

    cells_t = _sorted_cells(label_t, frame=0, open_end="high")
    cells_k = _sorted_cells(label_k, frame=1, open_end="high")
    pair = FramePair("trench_a", 0, 1, (12, 10), "y", "high")
    candidate = CandidateSolution.from_ops(
        pair.pair_id,
        [("exit", 1, None, None), ("link", 2, 20, None)],
        "manual",
    )

    features = compute_solution_features(
        FeatureContext(label_t, label_k, cells_t, cells_k, candidate, pair, TrackerParams())
    )

    assert features["n_exits"] == 1.0
    assert features["total_area_ratio_exit_adjusted"] == pytest.approx(1.0)
    assert features["exit_open_end_dist_median_norm"] == pytest.approx((11.0 - 8.0) / 12.0)


def test_compute_solution_features_division_metrics():
    np = _np()
    label_t = np.zeros((12, 10), dtype=np.uint32)
    label_k = np.zeros((12, 10), dtype=np.uint32)
    label_t[2:4, 1:5] = 1
    label_k[2:4, 1:3] = 10
    label_k[4:5, 1:3] = 11

    cells_t = _sorted_cells(label_t, frame=0, open_end="low")
    cells_k = _sorted_cells(label_k, frame=1, open_end="low")
    pair = FramePair("trench_a", 0, 1, (12, 10), "y", "low")
    candidate = CandidateSolution.from_ops(pair.pair_id, [("divide", 1, 10, 11)], "manual")

    features = compute_solution_features(
        FeatureContext(label_t, label_k, cells_t, cells_k, candidate, pair, TrackerParams())
    )

    assert features["n_divides"] == 1.0
    assert features["div_mother_sum_area_ratio_max"] == pytest.approx(8.0 / 6.0)
    assert features["div_mother_sum_area_ratio_mean"] == pytest.approx(8.0 / 6.0)
    assert features["div_daughter_area_ratio_max"] == pytest.approx(2.0)
    assert features["div_daughter_area_ratio_mean"] == pytest.approx(2.0)
    assert features["div_mother_daughter_dy_max_norm"] == pytest.approx(1.5 / 12.0)
    assert features["div_mother_daughter_dy_mean_norm"] == pytest.approx(0.75 / 12.0)
    assert math.isnan(features["link_area_ratio_median"])


def test_solve_and_featurize_pair_returns_stable_table_with_ops_json():
    np = _np()
    pd = _pd()
    label_t = np.zeros((8, 8), dtype=np.uint32)
    label_k = np.zeros((8, 8), dtype=np.uint32)
    label_t[1:3, 1:3] = 1
    label_k[2:4, 1:3] = 10

    df = solve_and_featurize_pair(
        label_t,
        label_k,
        dataset_id="trench_a",
        t=2,
        k=5,
        axis="y",
        open_end="low",
        top_k=1,
        store_ops=True,
    )

    assert list(df.columns) == list(SAMPLE_META_COLUMNS + FEATURE_COLUMNS) + ["ops_json"]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["pair_id"] == "trench_a:2->5"
    assert row["delta_t"] == 3
    assert row["dp_rank_global"] == 1
    assert row["candidate_source"] == "dp_topk"
    assert row["n_links"] == 1
    assert pd.isna(row["is_correct"])
    assert row["sample_class"] == "unknown"
    assert row["ops_json"] == '[["link", 1, 10, null]]'


def test_build_feature_table_for_stack_defaults_to_unknown_correctness():
    np = _np()
    pd = _pd()
    labels = np.zeros((2, 8, 8), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 2:4, 1:3] = 10

    df = build_feature_table_for_stack(
        labels,
        dataset_id="trench_a",
        axis="y",
        open_end="low",
        top_k=1,
        store_ops=False,
    )

    assert len(df) == 1
    assert "ops_json" not in df.columns
    assert pd.isna(df.iloc[0]["is_correct"])
    assert df.iloc[0]["sample_class"] == "unknown"


def test_build_feature_dataframe_saved_gt_can_inject_missing_gt(tmp_path):
    np = _np()
    pd = _pd()
    tiff = pytest.importorskip("tifffile")

    labels = np.zeros((2, 10, 10), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 1
    labels[0, 6:8, 1:3] = 2
    labels[1, 1:3, 1:3] = 10
    labels[1, 6:8, 1:3] = 20
    tiff.imwrite(tmp_path / "frame0.tif", labels[0])
    tiff.imwrite(tmp_path / "frame1.tif", labels[1])

    tracks_csv = tmp_path / "tracks.csv"
    divisions_csv = tmp_path / "divisions.csv"
    pd.DataFrame(
        [
            {"track_id": 1, "t": 0, "label": 1},
            {"track_id": 2, "t": 0, "label": 2},
            {"track_id": 2, "t": 1, "label": 10},
            {"track_id": 1, "t": 1, "label": 20},
        ]
    ).to_csv(tracks_csv, index=False)
    pd.DataFrame(columns=["t_div", "mother_track_id", "d1_track_id", "d2_track_id"]).to_csv(
        divisions_csv,
        index=False,
    )

    spec = DatasetSpec(
        dataset_id="trench_a",
        axis="y",
        open_end="low",
        labels_dir=tmp_path,
        gt_tracks_csv=tracks_csv,
        gt_divisions_csv=divisions_csv,
    )

    samples, failures = build_feature_dataframe(
        [spec],
        gt_mode="saved",
        top_k_candidates=1,
        include_gt_if_missing=True,
        store_ops=True,
        strict=True,
    )

    assert failures.empty
    assert len(samples) == 2
    assert set(samples["candidate_source"]) == {"dp_topk", "gt_injected"}
    assert samples["n_candidates_pair"].tolist() == [2, 2]
    gt_row = samples[samples["candidate_source"] == "gt_injected"].iloc[0]
    dp_row = samples[samples["candidate_source"] == "dp_topk"].iloc[0]
    assert bool(gt_row["is_correct"]) is True
    assert bool(dp_row["is_correct"]) is False
    assert bool(dp_row["is_dpt_best"]) is True
    assert "ops_json" in samples.columns
