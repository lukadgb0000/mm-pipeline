"""Label TIFF IO helpers"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .images import natsort_key


def collect_label_paths(labels_dir: str | Path) -> list[Path]:
    root = Path(labels_dir)
    if not root.exists():
        raise FileNotFoundError(f"Label directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Label path is not a directory: {root}")
    paths = [p for p in root.glob("*.tif*") if p.is_file()]
    paths.sort(key=natsort_key)
    return paths


def read_label(path: str | Path) -> np.ndarray:
    """Read one 2D label TIFF tolerating singleton dimensions"""

    import numpy as np

    try:
        import tifffile as tiff
    except ImportError as exc:
        raise RuntimeError("Reading label TIFFs requires tifffile.") from exc

    arr = tiff.imread(path)
    if arr.ndim > 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Expected 2D label image, got shape {arr.shape} in {path}")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label image, got shape {arr.shape} in {path}")
    return np.asarray(arr, dtype=np.uint32)


def load_label_stack(paths: Iterable[str | Path]) -> np.ndarray:
    import numpy as np

    path_list = list(paths)
    labels = [read_label(p) for p in path_list]
    if not labels:
        raise ValueError("No label paths provided.")
    expected = labels[0].shape
    for path, arr in zip(path_list, labels):
        if arr.shape != expected:
            raise ValueError(f"Label shape mismatch: {path} has {arr.shape}, expected {expected}")
    return np.stack(labels, axis=0)


def load_labels_from_folder(labels_dir: str | Path) -> np.ndarray:
    paths = collect_label_paths(labels_dir)
    if not paths:
        raise ValueError(f"No TIFF files found in {labels_dir}")
    return load_label_stack(paths)


def save_label_stack(
    labels: np.ndarray,
    file_names: Iterable[str | Path],
    out_dir: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    """Save a 2D or 3D label stack as per-frame TIFFs."""

    import numpy as np

    try:
        import tifffile as tiff
    except ImportError as exc:
        raise RuntimeError("Saving label TIFFs requires tifffile.") from exc

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(labels)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Labels must be 2D or 3D (T,H,W); got shape {arr.shape}")

    names = [Path(p).name for p in file_names]
    if arr.shape[0] != len(names):
        raise ValueError(f"Cannot map {arr.shape[0]} frames to {len(names)} filenames.")

    written: list[Path] = []
    for frame, name in zip(arr, names):
        root = Path(name).stem
        suffix = Path(name).suffix or ".tif"
        out_path = out_root / f"{root}{suffix}"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"{out_path} exists; pass overwrite=True to replace it.")
        tiff.imwrite(out_path, np.asarray(frame, dtype=np.uint32))
        written.append(out_path)
    return written
