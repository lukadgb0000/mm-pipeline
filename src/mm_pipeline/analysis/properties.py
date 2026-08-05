"""Per-cell properties from scikit-image ``regionprops`` over the label TIFFs.

The tracker records only ``x, y, area, axis_len`` per detection. Anything richer
(a fitted ``major_axis_length``, ``orientation``, shape descriptors) requires
running ``regionprops`` over the label images and joining on ``(t, label)``.

``t`` is the 0-based index into the natsorted label stack and ``label`` is the
label-image pixel value, so the join is exact as long as labels are resolved the
same way the tracker did — via ``spec.effective_labels_dir``, already stored on
``Lineage.labels_dir``. This module reuses ``io.labels`` (io, not core/tracking),
keeping analysis purely downstream
"""

from __future__ import annotations

from typing import Any, Optional

# the pixel measurements are named with _px to accomodate other units eg microns in future)
_LENGTH_PROPS = frozenset(
    {
        "major_axis_length",
        "minor_axis_length",
        "perimeter",
        "perimeter_crofton",
        "equivalent_diameter",
        "equivalent_diameter_area",
        "feret_diameter_max",
    }
)

DEFAULT_PROPS = ("major_axis_length", "orientation", "eccentricity", "solidity")


def _px_name(prop: str) -> str:
    return f"{prop}_px" if prop in _LENGTH_PROPS else prop


def cell_properties(
    lin: Any,
    tracks: Any = None,
    props: tuple[str, ...] = DEFAULT_PROPS,
    *,
    extra_properties: tuple = (),
) -> Any:
    """Per-cell ``regionprops`` for the selected tracks, joined onto ``tracks_df``.

    ``tracks`` is a :class:`TrackSet` (or ``None`` for the whole lineage — the
    expensive full-stack case). Returns one row per observed detection with
    columns ``dataset_id, track_id, t, label, <props>, bbox_len, area``, where
    length-dimensioned props carry a ``_px`` suffix (e.g. ``major_axis_length_px``)
    and ``bbox_len`` is the tracker's ``axis_len`` (a bbox extent, *not* the fitted
    length). Extend by passing skimage's own ``extra_properties=(fn,)``.
    """
    import pandas as pd

    try:
        from skimage.measure import regionprops_table
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "cell_properties requires scikit-image. Install the 'analysis' extra."
        ) from exc

    from mm_pipeline.io.labels import collect_label_paths, read_label

    if lin.labels_dir is None:
        raise ValueError(
            "cell_properties requires labels; build the Lineage from a spec with a "
            "labels directory (spec.effective_labels_dir)."
        )

    rows = lin.tracks_df
    if tracks is not None:
        ids = {int(t) for t in tracks}
        rows = rows[rows["track_id"].isin(ids)]
    sel = rows[["track_id", "t", "label", "area", "axis_len"]]

    paths = collect_label_paths(lin.labels_dir)
    requested = ("label", *props)
    per_frame = []
    for t in sorted(int(x) for x in sel["t"].unique()):
        table = regionprops_table(
            read_label(paths[t]), properties=requested, extra_properties=extra_properties
        )
        frame = pd.DataFrame(table)
        frame["t"] = t
        per_frame.append(frame)

    prop_cols = [c for c in (per_frame[0].columns if per_frame else []) if c not in ("label", "t")]
    measured = (
        pd.concat(per_frame, ignore_index=True)
        if per_frame
        else pd.DataFrame(columns=["label", "t", *prop_cols])
    )

    out = sel.merge(measured, on=["t", "label"], how="inner")
    out = out.rename(columns={"axis_len": "bbox_len", **{c: _px_name(c) for c in prop_cols}})
    out["dataset_id"] = lin.dataset_id

    ordered = ["dataset_id", "track_id", "t", "label", *(_px_name(c) for c in prop_cols), "bbox_len", "area"]
    return out[ordered].sort_values(["track_id", "t"]).reset_index(drop=True)
