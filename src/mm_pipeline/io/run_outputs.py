"""Run-output writers for runner result artefacts
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence


def write_summary_json(summary: dict, out_path: str | Path) -> Path:
    """Write a JSON summary with stable formatting"""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(summary), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def write_per_fold_csv(per_fold: Any, out_path: str | Path) -> Path:
    """Write per-fold metrics to CSV"""

    pd = _require_pandas()
    if not isinstance(per_fold, pd.DataFrame):
        raise TypeError("per_fold must be a pandas DataFrame.")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(path, index=False)
    return path


def write_markdown_report(
    summary: dict,
    per_fold: Any | None,
    out_path: str | Path,
    *,
    title: str,
    sections: Sequence[str] | None = None,
) -> Path:
    """Write a compact human-readable markdown report"""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Summary", ""]
    for key, value in _jsonable(summary).items():
        lines.append(f"- `{key}`: {value}")

    if sections:
        for section in sections:
            lines.extend(["", str(section).rstrip(), ""])

    if per_fold is not None:
        pd = _require_pandas()
        if not isinstance(per_fold, pd.DataFrame):
            raise TypeError("per_fold must be a pandas DataFrame or None.")
        if not per_fold.empty:
            lines.extend(["", "## Per-Fold Preview", "", "```csv"])
            lines.append(per_fold.head(20).to_csv(index=False).strip())
            lines.append("```")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def write_evaluation_run(
    summary: dict,
    per_fold: Any | None,
    out_dir: str | Path,
    *,
    title: str,
    per_dataset: Any | None = None,
) -> dict[str, Path]:
    """Write standard summary/report/CSV files for one run"""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": write_summary_json(summary, out / "summary.json"),
        "report": write_markdown_report(summary, per_fold, out / "report.md", title=title),
    }
    if per_fold is not None:
        paths["per_fold"] = write_per_fold_csv(per_fold, out / "per_fold.csv")
    if per_dataset is not None:
        paths["per_dataset"] = write_per_fold_csv(per_dataset, out / "per_dataset.csv")
    return paths


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Run-output writers require pandas.") from exc
    return pd
