"""Segmentation overlay helpers. Ignore this it's not useful will clear it"""

from __future__ import annotations

from pathlib import Path


def to_rgb01(img: np.ndarray) -> np.ndarray:
    """Return an HxWx3 image scaled to [0, 1]"""

    import numpy as np

    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] != 3:
        arr = np.stack([arr.mean(axis=-1)] * 3, axis=-1)
    arr = arr.astype(np.float32)
    mx = float(arr.max()) if arr.size else 1.0
    if mx > 1.0:
        arr = arr / (65535.0 if mx > 255.0 else 255.0)
    return np.clip(arr, 0.0, 1.0)


def save_mask_overlays(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    base_name: str,
    filled_dir: str | Path,
    outlines_dir: str | Path,
) -> tuple[Path, Path]:
    """Save filled and outline overlays using lazy optional imports"""

    import numpy as np

    try:
        import imageio.v2 as imageio
        from cellpose import plot
        from skimage.segmentation import find_boundaries
    except ImportError as exc:
        raise RuntimeError("Overlay generation requires cellpose, scikit-image, and imageio.") from exc

    filled_root = Path(filled_dir)
    outlines_root = Path(outlines_dir)
    filled_root.mkdir(parents=True, exist_ok=True)
    outlines_root.mkdir(parents=True, exist_ok=True)

    rgb = to_rgb01(image)
    filled = plot.mask_overlay(rgb, labels)
    filled_path = filled_root / f"{base_name}_overlay.png"
    imageio.imwrite(filled_path, (np.clip(filled, 0, 1) * 255).astype(np.uint8))

    edges = find_boundaries(labels, mode="outer")
    outline = (rgb * 255).astype(np.uint8).copy()
    outline[edges] = [255, 0, 0]
    outline_path = outlines_root / f"{base_name}_outlines_on_image.png"
    imageio.imwrite(outline_path, outline)
    return filled_path, outline_path
