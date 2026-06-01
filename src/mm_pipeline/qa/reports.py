"""QA decision report writers"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .decisions import QADecision


def decisions_to_dataframe(decisions: Iterable[QADecision]):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("decisions_to_dataframe requires pandas.") from exc
    rows = [d.to_row() for d in decisions]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def write_qa_decisions_csv(decisions: Iterable[QADecision], path: str | Path) -> Path:
    """Write a flattened QA decisions table to CSV."""

    df = decisions_to_dataframe(decisions)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def write_lineage_outputs(
    tracks_df,
    events_df,
    divisions_df,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write the three lineage CSVs to out_dir and return their paths"""

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "tracks": out_root / "tracks.csv",
        "events": out_root / "events.csv",
        "divisions": out_root / "division_events.csv",
    }
    tracks_df.to_csv(paths["tracks"], index=False)
    events_df.to_csv(paths["events"], index=False)
    divisions_df.to_csv(paths["divisions"], index=False)
    return paths
