"""YAML config loader and override-precedence helper for CLI handlers. PyYAML is imported lazily so the package remains importable without it.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, TypeVar

T = TypeVar("T")


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config file or return ``{}`` if ``path`` is ``None``."""

    if path is None:
        return {}
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "YAML config files require PyYAML. Install the `cli` extra: "
            "pip install 'mothermachine-pipeline[cli]'."
        ) from exc
    with Path(path).open() as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML must be a mapping; got {type(data).__name__}.")
    return {str(k): v for k, v in data.items()}


def section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return ``config[name]`` as a dict, or ``{}`` if absent."""

    value = config.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {name!r} must be a mapping; got {type(value).__name__}.")
    return {str(k): v for k, v in value.items()}


def resolve(
    *,
    defaults: T,
    config_section: Mapping[str, Any] | None = None,
    flag_overrides: Mapping[str, Any] | None = None,
) -> T:

    if not is_dataclass(defaults):
        raise TypeError("defaults must be a dataclass instance.")
    valid = {f.name for f in fields(defaults)}
    merged: dict[str, Any] = {}
    for key, value in (config_section or {}).items():
        if key in valid:
            merged[key] = value
    for key, value in (flag_overrides or {}).items():
        if value is None:
            continue
        if key in valid:
            merged[key] = value
    return replace(defaults, **merged)
