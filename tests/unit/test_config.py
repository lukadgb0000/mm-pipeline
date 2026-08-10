import math

from mm_pipeline.config import DEFAULT_TRACKER_PARAMS, SegmentationConfig, SegmentationQCConfig, TrackerParams


def test_tracker_defaults_use_legacy_costs_without_division_hard_gates():
    assert DEFAULT_TRACKER_PARAMS == {
        "wy": 0.02,
        "wa": 0.2,
        "exit_lin": 3.0,
        "exit_quad": 4.5,
        "axis": "y",
        "wshrink": 50.0,
        "wshrink_border": 0.0,
        "shrink_tol": 0.05,
        "c_div0": 0.1,
        "wsym": 2.0,
        "w_divshrink": 10.0,
        "border_margin": 2,
        "div_tol_sum_area": math.inf,
        "div_tol_ind_area": math.inf,
        "div_tol_sum_len": math.inf,
        "div_tol_ind_len": math.inf,
    }
    assert TrackerParams().to_dict(include_extra=False) == DEFAULT_TRACKER_PARAMS


def test_segmentation_configs_construct_with_defaults():
    assert SegmentationConfig().backend == "cpsam"
    assert SegmentationConfig().model_type == "cpsam"
    assert SegmentationQCConfig().min_label_size == 25
