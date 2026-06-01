"""Feature set definitions."""

from __future__ import annotations

from collections.abc import Sequence

from .pairwise import COUNT_COLUMNS, FEATURE_COLUMNS


NON_DIVISION_FEATURES: tuple[str, ...] = (
    "max_shrink_pct",
    "total_area_ratio_exit_adjusted",
    "exit_open_end_dist_median_norm",
    "link_area_ratio_median",
    "link_area_ratio_max",
    "link_dy_median_norm",
    "link_dy_max_norm",
    "link_iou_shifted_median",
)

DIVISION_FEATURES: tuple[str, ...] = (
    "div_mother_sum_area_ratio_max",
    "div_mother_sum_area_ratio_mean",
    "div_daughter_area_ratio_max",
    "div_daughter_area_ratio_mean",
    "div_mother_daughter_dy_max_norm",
    "div_mother_daughter_dy_mean_norm",
)

REDUCED_V1_FEATURES: tuple[str, ...] = (
    "max_shrink_pct",
    "total_area_ratio_exit_adjusted",
    "exit_open_end_dist_median_norm",
    "link_area_ratio_median",
    "link_dy_max_norm",
    "link_iou_shifted_median",
    "div_mother_sum_area_ratio_mean",
    "div_daughter_area_ratio_mean",
    "div_mother_daughter_dy_mean_norm",
)


def get_feature_subsets() -> dict[str, list[str]]:

    full = list(FEATURE_COLUMNS)
    known_features = set(full)
    for subset_name, subset in {
        "non_division_features": NON_DIVISION_FEATURES,
        "division_features": DIVISION_FEATURES,
        "reduced_v1": REDUCED_V1_FEATURES,
    }.items():
        missing = [name for name in subset if name not in known_features]
        if missing:
            raise ValueError(f"{subset_name} references unknown feature column(s): {missing}")
    return {
        "all_features": full,
        "reduced_v1": list(REDUCED_V1_FEATURES),
        "non_division_features": list(NON_DIVISION_FEATURES),
        "division_features": list(DIVISION_FEATURES),
        "all_plus_counts": full + list(COUNT_COLUMNS),
    }


def resolve_feature_subset(feature_subset: str | Sequence[str]) -> list[str]:

    subsets = get_feature_subsets()
    if isinstance(feature_subset, str):
        if feature_subset not in subsets:
            raise KeyError(
                f"Unknown feature subset '{feature_subset}'. "
                f"Available subsets: {sorted(subsets.keys())}"
            )
        out = list(subsets[feature_subset])
    else:
        out = list(feature_subset)
    if not out:
        raise ValueError("Resolved feature subset is empty.")
    return out
