"""End-to-end tests: real ``git init`` repo, real ``git diff`` output,
real ``git apply`` via ``Patcher.apply_diff``.

Every other test in the suite verifies what the patcher *intends* to
do. These tests verify it composes correctly with the actual ``git``
binary — line endings, temp-file handling, the ``--check`` /
``--reverse`` dance, ordering, and the bits of unified-diff format
(``index`` rows, mode lines, similarity scores) that we don't
synthesize ourselves but that real diffs always carry.

Each test creates an isolated temp repo, commits a baseline, generates
the diff with ``git diff`` (so the format is exactly what the agent
will see in production), then runs the patcher against it.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from pr_agent.patcher import Patcher, UnsafePatchError


def _git(*args: str, cwd: Path) -> str:
    """Run git, capture stdout, fail loud on non-zero exit."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed in {cwd}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[Path]:
    """A real git repo with one committed file. Returns its root.

    Uses per-command ``-c user.email=... -c user.name=...`` so we don't
    touch any global git config.
    """
    if shutil.which("git") is None:
        pytest.skip("git binary not on PATH")
    root = tmp_path / "repo"
    root.mkdir()
    _git("init", "-q", "-b", "main", cwd=root)
    (root / "src").mkdir()
    (root / "src" / "hello.py").write_text("def hello():\n    return 1\n")
    _git("add", "src/hello.py", cwd=root)
    _git(
        "-c", "user.email=test@example.com",
        "-c", "user.name=Test",
        "commit", "-q", "-m", "init",
        cwd=root,
    )
    yield root


def _diff(repo_root: Path) -> str:
    """Return ``git diff`` of the working tree against HEAD."""
    return _git("diff", "HEAD", cwd=repo_root)


def _patcher(repo_root: Path, *, max_files: int = 5, max_lines: int = 500) -> Patcher:
    return Patcher(
        repo_root=repo_root,
        protected_paths=["secrets/", "infra/"],
        max_files_touched=max_files,
        max_patch_lines=max_lines,
    )


# --- Happy path ------------------------------------------------------------


def test_apply_real_single_file_diff(repo: Path) -> None:
    """A real ``git diff`` for a single-file edit applies cleanly and
    the working tree contains the new content."""
    target = repo / "src" / "hello.py"
    target.write_text("def hello():\n    return 42\n")
    diff = _diff(repo)
    # Reset working tree so apply_diff has to do the work.
    _git("checkout", "--", "src/hello.py", cwd=repo)
    assert target.read_text() == "def hello():\n    return 1\n"

    written = _patcher(repo).apply_diff(diff, ["T1"])
    assert written == [target]
    assert target.read_text() == "def hello():\n    return 42\n"


def test_apply_real_multi_file_diff(repo: Path) -> None:
    (repo / "src" / "world.py").write_text("def world():\n    return 'hi'\n")
    target = repo / "src" / "hello.py"
    target.write_text("def hello():\n    return 99\n")
    _git("add", "src/world.py", cwd=repo)
    diff = _diff(repo)
    # Stash both changes (untracked file too) to start clean.
    (repo / "src" / "world.py").unlink()
    _git("checkout", "--", "src/hello.py", cwd=repo)
    _git("rm", "--cached", "-q", "src/world.py", cwd=repo)

    written = _patcher(repo).apply_diff(diff, ["T1", "T2"])
    assert {p.name for p in written} == {"hello.py", "world.py"}
    assert target.read_text() == "def hello():\n    return 99\n"
    assert (repo / "src" / "world.py").read_text() == "def world():\n    return 'hi'\n"


def test_apply_real_diff_with_non_ascii_path(repo: Path) -> None:
    """A path with non-ASCII bytes triggers git's quoted-path output
    (``"b/src/caf\\303\\251.py"``, with C-style octal escapes). The
    patcher must decode those escapes back to real UTF-8 so the safety
    check sees what ``git apply`` will actually mutate, and the apply
    pipeline must succeed end-to-end."""
    weird = repo / "src" / "café.py"
    weird.write_text("x = 1\n")
    _git("add", "--", "src/café.py", cwd=repo)
    _git(
        "-c", "user.email=test@example.com",
        "-c", "user.name=Test",
        "commit", "-q", "-m", "add non-ascii",
        cwd=repo,
    )
    weird.write_text("x = 2\n")
    diff = _diff(repo)
    # Modern git, with default core.quotePath=true, quotes non-ASCII
    # paths with C-style octal escapes. Confirm we're actually
    # exercising that code path.
    assert "\\303\\251" in diff, (
        f"expected git to emit octal-escaped non-ASCII path but got:\n{diff}"
    )
    _git("checkout", "--", "src/café.py", cwd=repo)

    written = _patcher(repo).apply_diff(diff, ["T1"])
    assert written == [weird]
    assert weird.read_text() == "x = 2\n"


def test_apply_real_diff_with_payload_resembling_header(repo: Path) -> None:
    """A real diff whose hunk *contains* a line starting with ``+++ b/``
    (added as content, e.g. a doc example showing diff syntax) must
    apply cleanly. The state-machine tokenizer doesn't extract that
    sentinel as a touched path; the apply pipeline doesn't reject it
    either."""
    target = repo / "src" / "hello.py"
    target.write_text(
        "def hello():\n"
        "    return 1\n"
        "# doc: a unified-diff line looks like:\n"
        "# +++ b/some/other/file.py\n"
        "# --- a/some/other/file.py\n"
    )
    diff = _diff(repo)
    _git("checkout", "--", "src/hello.py", cwd=repo)

    written = _patcher(repo).apply_diff(diff, ["T1"])
    assert written == [target]
    # The sentinel content survived as literal text in the file.
    assert "+++ b/some/other/file.py" in target.read_text()
    # ...but no spurious files were created for those payload lines.
    assert not (repo / "some" / "other" / "file.py").exists()


def test_apply_real_rename_diff(repo: Path) -> None:
    src_old = repo / "src" / "hello.py"
    src_new = repo / "src" / "renamed.py"
    _git("mv", "src/hello.py", "src/renamed.py", cwd=repo)
    diff = _diff(repo)
    # Reset.
    _git("mv", "src/renamed.py", "src/hello.py", cwd=repo)

    written = _patcher(repo).apply_diff(diff, ["T1"])
    written_names = {p.name for p in written}
    assert "renamed.py" in written_names
    assert src_new.exists()
    assert not src_old.exists()


# --- Rejection paths -------------------------------------------------------


def test_apply_real_rename_into_protected_dir_rejected(repo: Path) -> None:
    """A rename moving a file INTO a protected directory must be
    blocked. The b-side path is what the safety check catches."""
    (repo / "infra").mkdir()
    _git("mv", "src/hello.py", "infra/hello.py", cwd=repo)
    diff = _diff(repo)
    _git("mv", "infra/hello.py", "src/hello.py", cwd=repo)
    (repo / "infra").rmdir()

    with pytest.raises(UnsafePatchError, match="protected path"):
        _patcher(repo).apply_diff(diff, ["T1"])
    # Working tree was not mutated.
    assert (repo / "src" / "hello.py").exists()
    assert not (repo / "infra" / "hello.py").exists()


def test_apply_real_rename_out_of_protected_dir_rejected(repo: Path) -> None:
    """The reciprocal: a rename moving a protected file OUT of its
    protected directory must also be blocked. The a-side (source)
    path is what catches this — exactly the rename-bypass scenario the
    a-side extraction was added for."""
    secrets = repo / "secrets"
    secrets.mkdir()
    (secrets / "key.py").write_text("API_KEY = 'x'\n")
    _git("add", "secrets/key.py", cwd=repo)
    _git(
        "-c", "user.email=test@example.com",
        "-c", "user.name=Test",
        "commit", "-q", "-m", "add secret",
        cwd=repo,
    )
    _git("mv", "secrets/key.py", "src/key.py", cwd=repo)
    diff = _diff(repo)
    _git("mv", "src/key.py", "secrets/key.py", cwd=repo)

    with pytest.raises(UnsafePatchError, match="protected path"):
        _patcher(repo).apply_diff(diff, ["T1"])
    assert (repo / "secrets" / "key.py").exists()


def test_apply_real_diff_revert_on_failed_check(repo: Path) -> None:
    """A diff that introduces a conflict marker passes ``git apply
    --check`` (it's syntactically valid) but fails ``git diff
    --check`` post-apply. The pipeline must reverse-apply so the
    working tree returns to the pre-apply state."""
    target = repo / "src" / "hello.py"
    target.write_text(
        "def hello():\n"
        "<<<<<<< HEAD\n"
        "    return 1\n"
        "=======\n"
        "    return 2\n"
        ">>>>>>> theirs\n"
    )
    diff = _diff(repo)
    _git("checkout", "--", "src/hello.py", cwd=repo)
    pre = target.read_text()

    with pytest.raises(UnsafePatchError):
        _patcher(repo).apply_diff(diff, ["T1"])
    # Reverted: the file content is exactly what it was before the call.
    assert target.read_text() == pre


def test_apply_real_diff_too_many_files_rejected(repo: Path) -> None:
    for i in range(4):
        (repo / "src" / f"f{i}.py").write_text(f"x = {i}\n")
    _git("add", "-A", cwd=repo)
    diff = _diff(repo)
    for i in range(4):
        (repo / "src" / f"f{i}.py").unlink()
    _git("reset", "-q", "HEAD", cwd=repo)

    with pytest.raises(UnsafePatchError, match="too many files"):
        _patcher(repo, max_files=2).apply_diff(diff, ["T1"])
    assert not any((repo / "src" / f"f{i}.py").exists() for i in range(4))


def test_apply_real_diff_unrelated_to_committed_state_fails_check(repo: Path) -> None:
    """A diff that doesn't apply to the current tree (e.g. line counts
    don't match) is rejected by ``git apply --check``. Working tree
    untouched."""
    bogus = (
        "diff --git a/src/hello.py b/src/hello.py\n"
        "--- a/src/hello.py\n+++ b/src/hello.py\n"
        "@@ -100,3 +100,3 @@\n"
        " context-that-doesnt-exist\n"
        "-old\n"
        "+new\n"
    )
    pre = (repo / "src" / "hello.py").read_text()
    with pytest.raises(UnsafePatchError):
        _patcher(repo).apply_diff(bogus, ["T1"])
    assert (repo / "src" / "hello.py").read_text() == pre
