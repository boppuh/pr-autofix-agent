from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pr_agent.models import ReviewComment, ReviewThread


def make_thread(
    *,
    thread_id: str = "T_1",
    path: str | None = "src/foo.py",
    line: int | None = 10,
    body: str = "missing null check on user.email",
    author: str = "cursor[bot]",
) -> ReviewThread:
    return ReviewThread(
        id=thread_id,
        is_resolved=False,
        comments=[
            ReviewComment(
                id="1",
                author=author,
                body=body,
                path=path,
                line=line,
                diff_hunk=None,
                created_at="2024-01-01T00:00:00Z",
            )
        ],
    )


@pytest.fixture
def thread_factory():
    return make_thread


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "foo.py").write_text("def f(x):\n    return x\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    return tmp_path
