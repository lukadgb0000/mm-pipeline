"""Track-table IO

Shared output destination for tracking stages (qa and newer track-select)
write_lineage_outputs moved here from qa/reports.py, which now re-exports it for back-compatibility
"""

from __future__ import annotations

from pathlib import Path


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


__all__ = ["write_lineage_outputs"]
