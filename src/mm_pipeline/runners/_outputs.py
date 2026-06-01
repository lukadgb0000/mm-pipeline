"""Shared output-writing helpers for Phase 11 runners.

Two output conventions

- Multi-file commands
- Single-artefact commands

summary.json and <out>.run.json follow the same schema, carrying
both headline metrics and run metadata (git commit, timestamp, manifest
path, resolved config, dataset ids).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mm_pipeline.io.git import current_git_commit
from mm_pipeline.io.run_outputs import (
    _jsonable,
    write_markdown_report,
    write_summary_json,
)


def resolve_run_tag(run_tag: str | None) -> str:
    """Return ``run_tag`` if given, else a UTC timestamp ``YYYY-MM-DDTHHMMSSZ``."""

    if run_tag:
        return run_tag
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def make_run_metadata(
    *,
    command: str,
    manifest_path: str | Path | None,
    resolved_config: Mapping[str, Any],
    dataset_ids: Sequence[str],
    git_root: str | Path = ".",
) -> dict[str, Any]:
   

    return {
        "command": command,
        "git_commit": current_git_commit(git_root),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "dataset_ids": list(dataset_ids),
        "resolved_config": _jsonable(dict(resolved_config)),
    }


def write_multifile_outputs(
    *,
    out_dir: str | Path,
    summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
    tables: Mapping[str, Any] | None = None,
    title: str,
    report_sections: Sequence[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    
    

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(
            f"{summary_path} already exists. Pass --overwrite or choose a different --run-tag."
        )

    combined: dict[str, Any] = {**dict(metadata), **dict(summary)}
    paths: dict[str, Path] = {
        "summary": write_summary_json(combined, summary_path),
        "report": write_markdown_report(
            combined, None, out / "report.md", title=title, sections=report_sections
        ),
    }

    if tables:
        import pandas as pd  # local import — pandas is an optional dep at the top level

        for name, table in tables.items():
            if not isinstance(table, pd.DataFrame):
                raise TypeError(f"Table {name!r} is not a pandas DataFrame.")
            path = out / f"{name}.csv"
            table.to_csv(path, index=False)
            paths[name] = path

    return paths


def write_single_artefact_outputs(
    *,
    out_path: str | Path,
    artefact: Any,
    metadata: Mapping[str, Any],
    summary: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"{out} already exists. Pass --overwrite or choose a different --out."
        )

    import pandas as pd  

    if isinstance(artefact, pd.DataFrame):
        artefact.to_parquet(out, index=False)
    elif isinstance(artefact, (bytes, bytearray)):
        out.write_bytes(bytes(artefact))
    elif isinstance(artefact, (str, Path)):
        src = Path(artefact)
        if src.resolve() != out.resolve():
            out.write_bytes(src.read_bytes())
    else:
        raise TypeError(
            f"Unsupported artefact type: {type(artefact).__name__}. "
            "Expected pandas.DataFrame, bytes, or Path."
        )

    run_json_path = out.with_suffix(out.suffix + ".run.json")
    combined: dict[str, Any] = {**dict(metadata), **dict(summary or {})}
    with run_json_path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(combined), fh, indent=2, sort_keys=True)
        fh.write("\n")

    return {"artefact": out, "run_metadata": run_json_path}
