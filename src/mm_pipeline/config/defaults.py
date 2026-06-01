"""Canonical defaults for the new package"""

from __future__ import annotations

from typing import Any

DEFAULT_TRACKER_PARAMS: dict[str, Any] = {
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
    "div_tol_sum_area": 0.2,
    "div_tol_ind_area": 0.2,
    "div_tol_sum_len": 0.2,
    "div_tol_ind_len": 0.2,
}

DEFAULT_SEGMENTATION_CONFIG: dict[str, Any] = {
    "backend": "cpsam",
    "model_type": "cpsam",
    "chan": 0,
    "chan2": 0,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "use_gpu": False,
    "save_tif": True,
    "save_pngs": False,
    "overlays": False,
}

DEFAULT_SEGMENTATION_QA_CONFIG: dict[str, Any] = {
    "min_label_size": 25,
    "cell_count_jump_threshold": 5,
    "total_area_jump_fraction": 0.25,
    "small_area_quantile": 0.01,
    "large_area_quantile": 0.99,
}
