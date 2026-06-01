"""Tests for HypothesisModel and the new from_mapping classmethods"""

from __future__ import annotations

import pytest

from mm_pipeline.config.schemas import (
    HypothesisModel,
    QAConfig,
    SegmentationConfig,
    SegmentationQAConfig,
)


def test_hypothesis_model_default_constructs():
    hm = HypothesisModel()
    assert hm.name == "default"


def test_hypothesis_model_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown hypothesis model"):
        HypothesisModel(name="lysis_extended")  # type: ignore[arg-type]


def test_hypothesis_model_from_mapping_default():
    hm = HypothesisModel.from_mapping({})
    assert hm.name == "default"


def test_hypothesis_model_from_mapping_explicit_default():
    hm = HypothesisModel.from_mapping({"name": "default"})
    assert hm.name == "default"


def test_hypothesis_model_from_mapping_unknown_raises():
    with pytest.raises(ValueError, match="Unknown hypothesis model"):
        HypothesisModel.from_mapping({"name": "lysis"})


def test_hypothesis_model_to_dict():
    hm = HypothesisModel()
    assert hm.to_dict() == {"name": "default"}


def test_qa_config_from_mapping_empty_returns_defaults():
    cfg = QAConfig.from_mapping({})
    assert cfg == QAConfig()


def test_qa_config_from_mapping_applies_known_fields():
    cfg = QAConfig.from_mapping({
        "within_pair_scorer": "classifier",
        "bridge_enabled": True,
        "bridge_tau": 0.7,
    })
    assert cfg.within_pair_scorer == "classifier"
    assert cfg.bridge_enabled is True
    assert cfg.bridge_tau == 0.7


def test_qa_config_from_mapping_ignores_unknown_fields():
    cfg = QAConfig.from_mapping({
        "within_pair_scorer": "dp_cost_min",
        "unknown_field": "ignored",
    })
    assert cfg.within_pair_scorer == "dp_cost_min"


def test_qa_config_from_mapping_invalid_value_raises():
    with pytest.raises(ValueError):
        QAConfig.from_mapping({"within_pair_scorer": "not_a_real_scorer"})


def test_segmentation_config_from_mapping_empty_returns_defaults():
    cfg = SegmentationConfig.from_mapping({})
    assert cfg == SegmentationConfig()


def test_segmentation_config_from_mapping_applies_known_fields():
    cfg = SegmentationConfig.from_mapping({
        "flow_threshold": 0.8,
        "use_gpu": True,
    })
    assert cfg.flow_threshold == 0.8
    assert cfg.use_gpu is True


def test_segmentation_qa_config_from_mapping_empty_returns_defaults():
    cfg = SegmentationQAConfig.from_mapping({})
    assert cfg == SegmentationQAConfig()


def test_segmentation_qa_config_from_mapping_applies_known_fields():
    cfg = SegmentationQAConfig.from_mapping({
        "min_label_size": 50,
        "small_area_quantile": 0.05,
    })
    assert cfg.min_label_size == 50
    assert cfg.small_area_quantile == 0.05
