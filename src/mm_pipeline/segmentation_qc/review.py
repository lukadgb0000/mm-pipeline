"""Segmentation review and image/label pairing helpers"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mm_pipeline.io.images import collect_image_paths, load_image_stack
from mm_pipeline.io.labels import collect_label_paths, load_label_stack

if TYPE_CHECKING:
    import numpy as np


def normalize_stem(path: str | Path) -> str:
    stem = Path(path).stem
    return re.sub(r"(?:_cp_masks|_masks|_labels|_seg)$", "", stem, flags=re.IGNORECASE)


@dataclass(frozen=True)
class LabelImagePairing:
    label_paths: tuple[Path, ...]
    image_paths: tuple[Path, ...]
    save_names: tuple[str, ...]
    stems_match: bool


def default_edited_labels_dir(labels_dir: str | Path) -> Path:
    """Return the default output directory for edited labels"""

    labels_path = Path(labels_dir)
    return labels_path.with_name(f"{labels_path.name}_edited")


def resolve_review_output_dir(
    labels_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Resolve review output path and guard against accidental in-place writes"""

    labels_path = Path(labels_dir)
    output_path = Path(out_dir) if out_dir is not None else default_edited_labels_dir(labels_path)
    if labels_path.resolve() == output_path.resolve() and not overwrite:
        raise ValueError(
            "Refusing to overwrite labels_dir without overwrite=True; "
            "choose out_dir or pass overwrite=True."
        )
    return output_path


def collect_label_image_pairs(labels_dir: str | Path, images_dir: str | Path) -> LabelImagePairing:
    """Pair label TIFFs and raw images by sorted order, checking normalised stems."""

    label_paths = collect_label_paths(labels_dir)
    image_paths = collect_image_paths(images_dir)
    if not label_paths:
        raise ValueError(f"No label TIFFs found in {labels_dir}")
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if len(label_paths) != len(image_paths):
        raise ValueError(f"Found {len(label_paths)} labels and {len(image_paths)} images; counts must match.")

    label_stems = [normalize_stem(p) for p in label_paths]
    image_stems = [normalize_stem(p) for p in image_paths]
    return LabelImagePairing(
        label_paths=tuple(label_paths),
        image_paths=tuple(image_paths),
        save_names=tuple(p.name for p in label_paths),
        stems_match=label_stems == image_stems,
    )


def load_review_stacks(pairing: LabelImagePairing) -> tuple[np.ndarray, np.ndarray]:
    """Load labels and images for review, verifying image/label dimensions"""

    labels = load_label_stack(pairing.label_paths)
    images = load_image_stack(pairing.image_paths)
    if images.shape[1:3] != labels.shape[1:3]:
        raise ValueError(f"Image shape {images.shape[1:3]} does not match labels {labels.shape[1:3]}")
    return labels, images


def launch_napari_review(
    labels: np.ndarray,
    images: np.ndarray,
    save_names: tuple[str, ...],
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    dataset_id: str = "",
    source_segmentation_run: str | Path | None = None,
    qa_report_path: str | Path | None = None,
) -> None:
    """Launch napari for manual review.

     Just be careful because it opens a GUI
    """

    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("napari is required for manual review.") from exc

    from .approval import save_approved_labels

    viewer = napari.Viewer()
    viewer.add_image(images, name="raw", rgb=images.ndim == 4 and images.shape[-1] in (3, 4))
    labels_layer = viewer.add_labels(labels, name="labels")
    viewer.layers.selection.active = labels_layer

    @viewer.bind_key("s")
    def _save(_viewer):  # pragma: no cover - GUI callback
        save_approved_labels(
            labels_layer.data,
            save_names,
            out_dir,
            dataset_id=dataset_id,
            overwrite=overwrite,
            source_segmentation_run=source_segmentation_run,
            qa_report_path=qa_report_path,
        )

    napari.run()


def review_and_approve_masks(
    labels_dir: str | Path,
    images_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
    dataset_id: str = "",
    source_segmentation_run: str | Path | None = None,
    qa_report_path: str | Path | None = None,
) -> Path:
    """Open labels/images in napari and return the directory used for saved edits

    If ``out_dir`` is omitted, edited labels are written to a sibling directory
    named ``<labels_dir>_edited``. Writing back into ``labels_dir`` is refused
    unless ``overwrite=True``
    """

    output_dir = resolve_review_output_dir(labels_dir, out_dir, overwrite=overwrite)
    pairing = collect_label_image_pairs(labels_dir, images_dir)
    labels, images = load_review_stacks(pairing)
    launch_napari_review(
        labels,
        images,
        pairing.save_names,
        output_dir,
        overwrite=overwrite,
        dataset_id=dataset_id,
        source_segmentation_run=source_segmentation_run,
        qa_report_path=qa_report_path,
    )
    return output_dir
