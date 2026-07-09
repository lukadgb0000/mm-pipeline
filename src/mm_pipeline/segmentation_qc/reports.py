"""Segmentation QC report writers"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from mm_pipeline.config import SegmentationQCFinding


REPORT_COLUMNS = [
    "dataset_id",
    "frame",
    "severity",
    "check_name",
    "message",
    "label_id",
    "metric_name",
    "metric_value",
    "threshold",
    "review_status",
    "metrics",
]


def write_qc_report_csv(findings: Iterable[SegmentationQCFinding], out_csv: str | Path) -> Path:
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for finding in findings:
            row = finding.to_dict()
            row["metrics"] = json.dumps(row.get("metrics", {}), sort_keys=True)
            writer.writerow({col: row.get(col) for col in REPORT_COLUMNS})
    return path
