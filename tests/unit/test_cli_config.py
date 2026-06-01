"""Tests for mm_pipeline.cli._config"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from mm_pipeline.cli._config import load_yaml_config, resolve, section


def test_load_yaml_config_none_returns_empty():
    assert load_yaml_config(None) == {}


def test_load_yaml_config_reads_file(tmp_path: Path):
    yaml = pytest.importorskip("yaml")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("tracker:\n  wshrink: 50.0\nqa:\n  bridge_enabled: true\n")
    data = load_yaml_config(cfg)
    assert data == {"tracker": {"wshrink": 50.0}, "qa": {"bridge_enabled": True}}


def test_load_yaml_config_empty_file_returns_empty(tmp_path: Path):
    pytest.importorskip("yaml")
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("")
    assert load_yaml_config(cfg) == {}


def test_load_yaml_config_non_mapping_raises(tmp_path: Path):
    pytest.importorskip("yaml")
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml_config(cfg)


def test_section_missing_returns_empty():
    assert section({}, "qa") == {}


def test_section_present_returns_dict():
    assert section({"qa": {"bridge_enabled": True}}, "qa") == {"bridge_enabled": True}


def test_section_non_mapping_raises():
    with pytest.raises(ValueError, match="mapping"):
        section({"qa": [1, 2]}, "qa")


@dataclass
class _SampleConfig:
    a: int = 1
    b: str = "hello"
    c: float = 2.5


def test_resolve_no_overrides_returns_defaults():
    defaults = _SampleConfig()
    result = resolve(defaults=defaults)
    assert result == _SampleConfig(a=1, b="hello", c=2.5)


def test_resolve_config_section_applies():
    result = resolve(
        defaults=_SampleConfig(),
        config_section={"a": 10, "b": "world"},
    )
    assert result == _SampleConfig(a=10, b="world", c=2.5)


def test_resolve_flag_overrides_win():
    result = resolve(
        defaults=_SampleConfig(),
        config_section={"a": 10},
        flag_overrides={"a": 99},
    )
    assert result.a == 99


def test_resolve_none_flag_skipped():
    result = resolve(
        defaults=_SampleConfig(),
        config_section={"a": 10},
        flag_overrides={"a": None, "b": "from_flag"},
    )
    assert result.a == 10
    assert result.b == "from_flag"


def test_resolve_unknown_field_ignored():
    result = resolve(
        defaults=_SampleConfig(),
        config_section={"a": 5, "unknown": "ignored"},
    )
    assert result == _SampleConfig(a=5)


def test_resolve_non_dataclass_raises():
    with pytest.raises(TypeError, match="dataclass"):
        resolve(defaults="not a dataclass")  # type: ignore[arg-type]
