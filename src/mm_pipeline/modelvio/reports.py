"""Model-violation decision report writers"""

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


__all__ = ["decisions_to_dataframe", "write_qa_decisions_csv"]
