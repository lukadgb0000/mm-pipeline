"""Tests for mm_pipeline.io.git.current_git_commit"""

from __future__ import annotations

import tempfile
from pathlib import Path

from mm_pipeline.io.git import current_git_commit


def test_current_git_commit_in_repo_returns_string():
    sha = current_git_commit(".")
    assert sha is None or isinstance(sha, str)
    if sha is not None:
        assert sha.strip() == sha
        assert " " not in sha
        assert len(sha) >= 4


def test_current_git_commit_outside_repo_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        # /tmp is not a git repo
        assert current_git_commit(tmp) is None


def test_current_git_commit_nonexistent_path_returns_none():
    sha = current_git_commit("/nonexistent/path/does/not/exist")
    assert sha is None


def test_current_git_commit_long_form_returns_string():
    sha_short = current_git_commit(".", short=True)
    sha_long = current_git_commit(".", short=False)
    if sha_short is not None and sha_long is not None:
        assert len(sha_long) >= len(sha_short)
