"""Raw image collection and loading helpers"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

IMG_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def natsort_key(value: str | Path) -> list[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(value))]


def collect_image_paths(
    images_dir: str | Path,
    image_pattern: str | None = None,
    extensions: Sequence[str] = IMG_EXTS,
) -> list[Path]:
    """Collect raw image paths with natural sorting. Mirrors the extension filtering used in ``01_cpsam_batch.py`` from the old codebase while returning
    Path objects and allowing an optional glob pattern.
    """

    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {root}")

    if image_pattern:
        candidates = [p for p in root.glob(image_pattern) if p.is_file()]
    else:
        allowed = {ext.lower() for ext in extensions}
        candidates = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in allowed]

    candidates.sort(key=natsort_key)
    return candidates


def read_image(path: str | Path) -> np.ndarray:
    """Read one image using imageio imported lazily"""

    import numpy as np

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("Reading images requires imageio.") from exc
    return np.asarray(imageio.imread(path))


def load_image_stack(paths: Iterable[str | Path]) -> np.ndarray:
    """Load images and require identical shapes"""

    import numpy as np

    path_list = list(paths)
    arrays = [read_image(p) for p in path_list]
    if not arrays:
        raise ValueError("No image paths provided.")
    expected = arrays[0].shape
    for path, arr in zip(path_list, arrays):
        if arr.shape != expected:
            raise ValueError(f"Image shape mismatch: {path} has {arr.shape}, expected {expected}")
    return np.stack(arrays, axis=0)
