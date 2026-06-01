"""Git-commit retrieval helper for run metadata
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def current_git_commit(path: str | Path = ".", *, short: bool = True) -> str | None:
    """Return HEAD SHA or ``None`` if not in a git repo / git absent."""

    args = ["git", "rev-parse"]
    if short:
        args.append("--short")
    args.append("HEAD")
    try:
        result = subprocess.run(
            args,
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None
