"""Manifest readers for dataset-level configuration"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from mm_pipeline.config.schemas import DatasetSpec, RawImageDatasetSpec


def read_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a CSV or YAML manifest into raw row dictionaries
    """

    manifest_path = Path(path)
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        with manifest_path.open(newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("YAML manifests require PyYAML to be installed.") from exc
        with manifest_path.open() as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict) and "datasets" in data:
            data = data["datasets"]
        if not isinstance(data, list):
            raise ValueError("YAML manifest must be a list or contain a 'datasets' list.")
        return [dict(row) for row in data]
    raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")


def load_dataset_manifest(path: str | Path) -> list[DatasetSpec]:
    """Load a dataset manifest into validated DatasetSpec rows"""

    return [DatasetSpec.from_mapping(row) for row in read_manifest_rows(path)]


def load_raw_image_manifest(path: str | Path) -> list[RawImageDatasetSpec]:
    """Load a raw-image manifest for segmentation runs"""

    return [RawImageDatasetSpec.from_mapping(row) for row in read_manifest_rows(path)]
