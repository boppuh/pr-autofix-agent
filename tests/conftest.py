from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pr_agent.models import BugbotComment, ReviewThread


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
        path=path,
        line=line,
        is_resolved=False,
        is_outdated=False,
        comments=[
            BugbotComment(
                id="1",
                author_login=author,
                body=body,
                path=path,
                line=line,
                created_at=datetime.now(UTC),
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
