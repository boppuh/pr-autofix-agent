from __future__ import annotations

import pytest

from pr_agent.models import Patch, PatchFile
from pr_agent.patcher import Patcher, UnsafePatchError


def _patch(thread_id: str, files: list[tuple[str, str]]) -> Patch:
    return Patch(
        thread_id=thread_id,
        summary="test",
        files=[PatchFile(path=p, new_content=c, rationale="r") for p, c in files],
    )


def test_apply_writes_files_inside_repo(git_repo):
    p = Patcher(git_repo, exclude_paths=[], max_files_per_patch=5)
    patch = _patch("T1", [("src/foo.py", "def f(x):\n    return x or 0\n")])
    written = p.apply(patch)
    assert (git_repo / "src/foo.py").read_text().endswith("x or 0\n")
    assert written[0] == git_repo / "src/foo.py"


def test_rejects_path_traversal(git_repo):
    p = Patcher(git_repo, exclude_paths=[], max_files_per_patch=5)
    patch = _patch("T1", [("../etc/passwd", "x")])
    with pytest.raises(UnsafePatchError):
        p.apply(patch)


def test_rejects_forbidden_workflow_path(git_repo):
    p = Patcher(git_repo, exclude_paths=[], max_files_per_patch=5)
    patch = _patch("T1", [(".github/workflows/ci.yml", "evil: true")])
    with pytest.raises(UnsafePatchError):
        p.apply(patch)


def test_rejects_lockfile(git_repo):
    p = Patcher(git_repo, exclude_paths=[], max_files_per_patch=5)
    patch = _patch("T1", [("package-lock.json", "{}")])
    with pytest.raises(UnsafePatchError):
        p.apply(patch)


def test_rejects_excluded_path(git_repo):
    p = Patcher(git_repo, exclude_paths=["vendor/**"], max_files_per_patch=5)
    patch = _patch("T1", [("vendor/lib.py", "x")])
    with pytest.raises(UnsafePatchError):
        p.apply(patch)


def test_max_files_per_patch_enforced(git_repo):
    p = Patcher(git_repo, exclude_paths=[], max_files_per_patch=1)
    patch = _patch("T1", [("a.py", "x"), ("b.py", "y")])
    report = p.check_safe(patch)
    assert not report.ok
    assert any("too many files" in r for r in report.reasons)


def test_revert_restores_committed_content(git_repo):
    p = Patcher(git_repo, exclude_paths=[], max_files_per_patch=5)
    patch = _patch("T1", [("src/foo.py", "BROKEN\n")])
    written = p.apply(patch)
    assert (git_repo / "src/foo.py").read_text() == "BROKEN\n"
    p.revert_uncommitted(written)
    assert "return x" in (git_repo / "src/foo.py").read_text()


def test_stage_and_commit_returns_sha(git_repo):
    p = Patcher(git_repo, exclude_paths=[], max_files_per_patch=5)
    patch = _patch("T1", [("src/foo.py", "def f(x):\n    return x or 0\n")])
    written = p.apply(patch)
    sha = p.stage_and_commit(patch, written, author_email="a@b")
    assert len(sha) >= 7
