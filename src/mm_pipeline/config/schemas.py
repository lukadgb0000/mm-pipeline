"""Central configuration and data-contract schemas"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

from .defaults import DEFAULT_SEGMENTATION_CONFIG, DEFAULT_SEGMENTATION_QA_CONFIG, DEFAULT_TRACKER_PARAMS

AXES = {"x", "y"}
OPEN_ENDS = {"low", "high"}
QA_SEVERITIES = {"info", "warning", "error"}


def _none_if_blank(value: Any) -> Any:
    if value == "":
        return None
    return value


def _optional_path(value: Any) -> Optional[Path]:
    value = _none_if_blank(value)
    if value is None:
        return None
    return Path(str(value))


def _optional_float(value: Any) -> Optional[float]:
    value = _none_if_blank(value)
    if value is None:
        return None
    return float(value)


@dataclass(frozen=True)
class DatasetSpec:
    """This is the manifest row describing one dataset.

    For tracking, ``approved_labels_dir`` is preferred over ``labels_dir``.
    If neither label directory is present, ``images_dir`` can be used by a
    full pipeline run that performs segmentation before tracking.
    """

    dataset_id: str
    axis: str = "y"
    open_end: str = "high"
    approved_labels_dir: Optional[Path] = None
    labels_dir: Optional[Path] = None
    images_dir: Optional[Path] = None
    gt_tracks_csv: Optional[Path] = None
    gt_divisions_csv: Optional[Path] = None
    frame_interval_min: Optional[float] = None
    dataset_kind: Optional[str] = None
    segmentation_run_dir: Optional[Path] = None
    segmentation_qa_report: Optional[Path] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty.")
        if self.axis not in AXES:
            raise ValueError("axis must be 'x' or 'y'.")
        if self.open_end not in OPEN_ENDS:
            raise ValueError("open_end must be 'low' or 'high'.")
        if self.approved_labels_dir is None and self.labels_dir is None and self.images_dir is None:
            raise ValueError("DatasetSpec requires approved_labels_dir, labels_dir, or images_dir.")

    @property
    def effective_labels_dir(self) -> Optional[Path]:
        """Return the label directory to use for tracking, if available"""

        return self.approved_labels_dir or self.labels_dir

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "DatasetSpec":
        clean = {str(k): _none_if_blank(v) for k, v in row.items()}
        return cls(
            dataset_id=str(clean.get("dataset_id") or ""),
            axis=str(clean.get("axis") or "y"),
            open_end=str(clean.get("open_end") or "high"),
            approved_labels_dir=_optional_path(clean.get("approved_labels_dir")),
            labels_dir=_optional_path(clean.get("labels_dir")),
            images_dir=_optional_path(clean.get("images_dir")),
            gt_tracks_csv=_optional_path(clean.get("gt_tracks_csv")),
            gt_divisions_csv=_optional_path(clean.get("gt_divisions_csv")),
            frame_interval_min=_optional_float(clean.get("frame_interval_min")),
            dataset_kind=clean.get("dataset_kind"),
            segmentation_run_dir=_optional_path(clean.get("segmentation_run_dir")),
            segmentation_qa_report=_optional_path(clean.get("segmentation_qa_report")),
            notes=clean.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawImageDatasetSpec:
    """Manifest row for raw image segmentation input"""

    dataset_id: str
    images_dir: Path
    image_pattern: Optional[str] = None
    channel: Optional[int] = None
    frame_interval_min: Optional[float] = None
    dataset_kind: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty.")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "RawImageDatasetSpec":
        clean = {str(k): _none_if_blank(v) for k, v in row.items()}
        images_dir = _optional_path(clean.get("images_dir"))
        if images_dir is None:
            raise ValueError("RawImageDatasetSpec requires images_dir.")
        channel = clean.get("channel")
        return cls(
            dataset_id=str(clean.get("dataset_id") or ""),
            images_dir=images_dir,
            image_pattern=clean.get("image_pattern"),
            channel=None if channel is None else int(channel),
            frame_interval_min=_optional_float(clean.get("frame_interval_min")),
            dataset_kind=clean.get("dataset_kind"),
            notes=clean.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawImageFrame:
    dataset_id: str
    frame: int
    path: Path
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class SegmentationRunArtifact:
    dataset_id: str
    backend: str
    label_tifs_dir: Path
    raw_images_dir: Optional[Path] = None
    model_type: Optional[str] = None
    overlays_filled_dir: Optional[Path] = None
    overlays_outlines_dir: Optional[Path] = None
    config: dict[str, Any] = field(default_factory=dict)
    image_count: int = 0
    label_count: int = 0
    frame_shape: Optional[tuple[int, int]] = None
    created_at: Optional[str] = None
    software_versions: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationQAFinding:
    dataset_id: str
    frame: Optional[int]
    severity: Literal["info", "warning", "error"]
    check_name: str
    message: str
    label_id: Optional[int] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    review_status: str = "unreviewed"
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in QA_SEVERITIES:
            raise ValueError("severity must be one of: info, warning, error.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ApprovedLabelStack:
    dataset_id: str
    labels_dir: Path
    source_segmentation_run: Optional[Path] = None
    approval_status: str = "approved"
    qa_report_path: Optional[Path] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationConfig:
    backend: str = str(DEFAULT_SEGMENTATION_CONFIG["backend"])
    model_type: str = str(DEFAULT_SEGMENTATION_CONFIG["model_type"])
    chan: int = int(DEFAULT_SEGMENTATION_CONFIG["chan"])
    chan2: int = int(DEFAULT_SEGMENTATION_CONFIG["chan2"])
    flow_threshold: float = float(DEFAULT_SEGMENTATION_CONFIG["flow_threshold"])
    cellprob_threshold: float = float(DEFAULT_SEGMENTATION_CONFIG["cellprob_threshold"])
    use_gpu: bool = bool(DEFAULT_SEGMENTATION_CONFIG["use_gpu"])
    save_tif: bool = bool(DEFAULT_SEGMENTATION_CONFIG["save_tif"])
    save_pngs: bool = bool(DEFAULT_SEGMENTATION_CONFIG["save_pngs"])
    overlays: bool = bool(DEFAULT_SEGMENTATION_CONFIG["overlays"])

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty.")
        if not self.model_type:
            raise ValueError("model_type must be non-empty.")
        if self.chan < 0 or self.chan2 < 0:
            raise ValueError("chan and chan2 must be non-negative.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SegmentationConfig":
        known = {f for f in DEFAULT_SEGMENTATION_CONFIG}
        typed = {str(k): v for k, v in values.items() if k in known}
        return cls(**typed)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationQAConfig:
    min_label_size: int = int(DEFAULT_SEGMENTATION_QA_CONFIG["min_label_size"])
    cell_count_jump_threshold: int = int(DEFAULT_SEGMENTATION_QA_CONFIG["cell_count_jump_threshold"])
    total_area_jump_fraction: float = float(DEFAULT_SEGMENTATION_QA_CONFIG["total_area_jump_fraction"])
    small_area_quantile: float = float(DEFAULT_SEGMENTATION_QA_CONFIG["small_area_quantile"])
    large_area_quantile: float = float(DEFAULT_SEGMENTATION_QA_CONFIG["large_area_quantile"])

    def __post_init__(self) -> None:
        if self.min_label_size < 0:
            raise ValueError("min_label_size must be non-negative.")
        if self.cell_count_jump_threshold < 0:
            raise ValueError("cell_count_jump_threshold must be non-negative.")
        if self.total_area_jump_fraction < 0:
            raise ValueError("total_area_jump_fraction must be non-negative.")
        if not 0.0 <= self.small_area_quantile <= 1.0:
            raise ValueError("small_area_quantile must be in [0, 1].")
        if not 0.0 <= self.large_area_quantile <= 1.0:
            raise ValueError("large_area_quantile must be in [0, 1].")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SegmentationQAConfig":
        known = {f for f in DEFAULT_SEGMENTATION_QA_CONFIG}
        typed = {str(k): v for k, v in values.items() if k in known}
        return cls(**typed)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrackerParams:
    wy: float = float(DEFAULT_TRACKER_PARAMS["wy"])
    wa: float = float(DEFAULT_TRACKER_PARAMS["wa"])
    exit_lin: float = float(DEFAULT_TRACKER_PARAMS["exit_lin"])
    exit_quad: float = float(DEFAULT_TRACKER_PARAMS["exit_quad"])
    axis: str = str(DEFAULT_TRACKER_PARAMS["axis"])
    wshrink: float = float(DEFAULT_TRACKER_PARAMS["wshrink"])
    wshrink_border: float = float(DEFAULT_TRACKER_PARAMS["wshrink_border"])
    shrink_tol: float = float(DEFAULT_TRACKER_PARAMS["shrink_tol"])
    c_div0: float = float(DEFAULT_TRACKER_PARAMS["c_div0"])
    wsym: float = float(DEFAULT_TRACKER_PARAMS["wsym"])
    w_divshrink: float = float(DEFAULT_TRACKER_PARAMS["w_divshrink"])
    border_margin: int = int(DEFAULT_TRACKER_PARAMS["border_margin"])
    div_tol_sum_area: float = float(DEFAULT_TRACKER_PARAMS["div_tol_sum_area"])
    div_tol_ind_area: float = float(DEFAULT_TRACKER_PARAMS["div_tol_ind_area"])
    div_tol_sum_len: float = float(DEFAULT_TRACKER_PARAMS["div_tol_sum_len"])
    div_tol_ind_len: float = float(DEFAULT_TRACKER_PARAMS["div_tol_ind_len"])
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError("axis must be 'x' or 'y'.")
        if self.border_margin < 0:
            raise ValueError("border_margin must be non-negative.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TrackerParams":
        known = {k for k in DEFAULT_TRACKER_PARAMS}
        typed: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in values.items():
            if key in known:
                typed[key] = value
            else:
                extra[str(key)] = value
        return cls(**typed, extra=extra)

    def to_dict(self, include_extra: bool = True) -> dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra", {})
        if include_extra:
            data.update(extra)
        return data


WITHIN_PAIR_SCORERS = {"dp_cost_min", "classifier", "ensemble"}
ENSEMBLE_MODES = {"rank", "zscore"}
ANOMALY_DETECTORS_BUILTIN = {"never_anomalous", "hist_gbm_default"}
DISAGREEMENT_DROP_POLICIES = {"never", "hard", "soft"}


@dataclass(frozen=True)
class QAConfig:
    """
    Defaults CURRENTLY give the most conservative production behaviour: trust the DP
    top-1, no anomaly detection, no bridging, disagreement diagnostics
    report-only. Enabling anomaly detection and bridging is opt-in CURRENTLY (!double check).
    """

    within_pair_scorer: str = "dp_cost_min"
    within_pair_ensemble_alpha: float = 0.5
    within_pair_ensemble_mode: str = "rank"

    anomaly_detector: str = "never_anomalous"
    anomaly_threshold: Optional[float] = None

    bridge_enabled: bool = False
    bridge_tau: float = 0.5
    bridge_max_gap: int = 3
    bridge_top_k: int = 16

    disagreement_drop: str = "never"
    disagreement_soft_threshold: float = 1.0

    pair_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.within_pair_scorer not in WITHIN_PAIR_SCORERS:
            raise ValueError(f"within_pair_scorer must be one of {sorted(WITHIN_PAIR_SCORERS)}.")
        if not 0.0 <= self.within_pair_ensemble_alpha <= 1.0:
            raise ValueError("within_pair_ensemble_alpha must be in [0, 1].")
        if self.within_pair_ensemble_mode not in ENSEMBLE_MODES:
            raise ValueError(f"within_pair_ensemble_mode must be one of {sorted(ENSEMBLE_MODES)}.")
        if self.disagreement_drop not in DISAGREEMENT_DROP_POLICIES:
            raise ValueError(f"disagreement_drop must be one of {sorted(DISAGREEMENT_DROP_POLICIES)}.")
        if self.bridge_max_gap < 2:
            raise ValueError("bridge_max_gap must be >= 2.")
        if self.bridge_top_k < 1:
            raise ValueError("bridge_top_k must be >= 1.")
        if self.pair_temperature <= 0:
            raise ValueError("pair_temperature must be > 0.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "QAConfig":
        fields = {
            "within_pair_scorer", "within_pair_ensemble_alpha", "within_pair_ensemble_mode",
            "anomaly_detector", "anomaly_threshold",
            "bridge_enabled", "bridge_tau", "bridge_max_gap", "bridge_top_k",
            "disagreement_drop", "disagreement_soft_threshold",
            "pair_temperature",
        }
        typed = {str(k): v for k, v in values.items() if k in fields}
        return cls(**typed)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


HYPOTHESIS_MODELS = {"default"}


@dataclass(frozen=True)
class HypothesisModel:
    """Hypothesis-model config placeholder for future model expansion
    """

    name: Literal["default"] = "default"

    def __post_init__(self) -> None:
        if self.name not in HYPOTHESIS_MODELS:
            raise ValueError(
                f"Unknown hypothesis model: {self.name!r}. "
                f"Supported in Phase 11: {sorted(HYPOTHESIS_MODELS)}."
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HypothesisModel":
        name = str(values.get("name", "default"))
        return cls(name=name)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
