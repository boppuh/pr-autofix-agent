from __future__ import annotations

import pytest

from pr_agent.models import Patch, PatchFile
from pr_agent.patcher import (
    Patcher,
    UnsafePatchError,
    apply_unified_diff,
    count_patch_lines,
    extract_touched_files,
    violates_protected_paths,
)


def _patch(thread_id: str, files: list[tuple[str, str]]) -> Patch:
    return Patch(
        thread_id=thread_id,
        summary="test",
        files=[PatchFile(path=p, new_content=c, rationale="r") for p, c in files],
    )


def _mk(git_repo, **kw):
    defaults = dict(protected_paths=[], max_files_touched=15, max_patch_lines=800)
    defaults.update(kw)
    return Patcher(git_repo, **defaults)


def test_apply_writes_files_inside_repo(git_repo):
    p = _mk(git_repo)
    patch = _patch("T1", [("src/foo.py", "def f(x):\n    return x or 0\n")])
    written = p.apply(patch)
    assert (git_repo / "src/foo.py").read_text().endswith("x or 0\n")
    assert written[0] == git_repo / "src/foo.py"


def test_rejects_path_traversal(git_repo):
    p = _mk(git_repo)
    patch = _patch("T1", [("../etc/passwd", "x")])
    with pytest.raises(UnsafePatchError):
        p.apply(patch)


def test_rejects_forbidden_workflow_path(git_repo):
    p = _mk(git_repo)
    patch = _patch("T1", [(".github/workflows/ci.yml", "evil: true")])
    with pytest.raises(UnsafePatchError):
        p.apply(patch)


def test_rejects_lockfile(git_repo):
    p = _mk(git_repo)
    patch = _patch("T1", [("package-lock.json", "{}")])
    with pytest.raises(UnsafePatchError):
        p.apply(patch)


def test_rejects_protected_dir_prefix(git_repo):
    p = _mk(git_repo, protected_paths=["infra/"])
    patch = _patch("T1", [("infra/main.tf", "x")])
    with pytest.raises(UnsafePatchError):
        p.apply(patch)


def test_max_files_touched_enforced(git_repo):
    p = _mk(git_repo, max_files_touched=1)
    patch = _patch("T1", [("a.py", "x"), ("b.py", "y")])
    report = p.check_safe(patch)
    assert not report.ok
    assert any("too many files" in r for r in report.reasons)


def test_max_patch_lines_enforced(git_repo):
    p = _mk(git_repo, max_patch_lines=10)
    big = "\n".join(f"line {i}" for i in range(50))
    patch = _patch("T1", [("a.py", big)])
    report = p.check_safe(patch)
    assert not report.ok
    assert any("patch too large" in r for r in report.reasons)


def test_revert_restores_committed_content(git_repo):
    p = _mk(git_repo)
    patch = _patch("T1", [("src/foo.py", "BROKEN\n")])
    written = p.apply(patch)
    assert (git_repo / "src/foo.py").read_text() == "BROKEN\n"
    p.revert_uncommitted(written)
    assert "return x" in (git_repo / "src/foo.py").read_text()


def test_stage_and_commit_returns_sha(git_repo):
    p = _mk(git_repo)
    patch = _patch("T1", [("src/foo.py", "def f(x):\n    return x or 0\n")])
    written = p.apply(patch)
    sha = p.stage_and_commit(patch, written, author_email="a@b")
    assert len(sha) >= 7


# --- apply_diff (Phase 8 batched path) ----------------------------------


def _diff(headers: list[tuple[str, str]], hunks: str) -> str:
    """Build a minimal unified diff for tests."""
    parts = []
    for old, new in headers:
        parts.append(f"diff --git a/{old} b/{new}")
        parts.append(f"--- a/{old}")
        parts.append(f"+++ b/{new}")
    parts.append(hunks)
    return "\n".join(parts) + "\n"


def test_apply_diff_applies_a_simple_diff(git_repo):
    p = _mk(git_repo)
    diff = _diff(
        [("src/foo.py", "src/foo.py")],
        "@@ -1,2 +1,2 @@\n def f(x):\n-    return x\n+    return x or 0\n",
    )
    written = p.apply_diff(diff, thread_ids=["T1"])
    assert (git_repo / "src/foo.py").read_text() == "def f(x):\n    return x or 0\n"
    assert written == [git_repo / "src/foo.py"]


def test_apply_diff_rejects_protected_path(git_repo):
    p = _mk(git_repo, protected_paths=["infra/"])
    diff = _diff(
        [("infra/main.tf", "infra/main.tf")],
        "@@ -0,0 +1 @@\n+resource\n",
    )
    with pytest.raises(UnsafePatchError, match="protected path"):
        p.apply_diff(diff, thread_ids=["T1"])


def test_apply_diff_rejects_forbidden_glob(git_repo):
    p = _mk(git_repo)
    diff = _diff(
        [(".github/workflows/ci.yml", ".github/workflows/ci.yml")],
        "@@ -0,0 +1 @@\n+evil: true\n",
    )
    with pytest.raises(UnsafePatchError, match="forbidden path"):
        p.apply_diff(diff, thread_ids=["T1"])


def test_apply_diff_rejects_too_many_files(git_repo):
    p = _mk(git_repo, max_files_touched=1)
    diff = _diff(
        [("src/foo.py", "src/foo.py"), ("src/bar.py", "src/bar.py")],
        "@@ -1 +1 @@\n-x\n+y\n",
    )
    with pytest.raises(UnsafePatchError, match="too many files"):
        p.apply_diff(diff, thread_ids=["T1"])


def test_apply_diff_rejects_too_many_lines(git_repo):
    p = _mk(git_repo, max_patch_lines=2)
    big_hunk = "@@ -1,5 +1,5 @@\n" + "\n".join(["-old"] * 5 + ["+new"] * 5) + "\n"
    diff = _diff([("src/foo.py", "src/foo.py")], big_hunk)
    with pytest.raises(UnsafePatchError, match="patch too large"):
        p.apply_diff(diff, thread_ids=["T1"])


def test_apply_diff_rejects_unparseable_diff(git_repo):
    p = _mk(git_repo)
    with pytest.raises(UnsafePatchError):
        p.apply_diff("this is not a diff", thread_ids=["T1"])


def test_apply_diff_rejects_diff_that_does_not_apply(git_repo):
    """git apply --check fails when context lines don't match the working tree.
    Patcher.apply_diff now raises a generic message (specific reason in logs)
    since the actual git invocation is delegated to the free function."""
    p = _mk(git_repo)
    diff = _diff(
        [("src/foo.py", "src/foo.py")],
        "@@ -1,2 +1,2 @@\n totally\n-different\n+content\n",
    )
    with pytest.raises(UnsafePatchError, match="git apply or git diff --check failed"):
        p.apply_diff(diff, thread_ids=["T1"])


def test_apply_diff_validates_plus_plus_path_when_diff_git_header_lies(git_repo):
    """Regression: a malicious / confused diff might claim a safe path in
    its `diff --git` header but actually target a forbidden path via the
    `+++ b/` line. `git apply` follows the `+++` line; our safety check
    must too. The union of both header sources must be validated."""
    p = _mk(git_repo)
    # diff --git claims src/foo.py, but +++ targets the workflow file.
    diff = (
        "diff --git a/src/foo.py b/src/foo.py\n"
        "--- a/src/foo.py\n"
        "+++ b/.github/workflows/ci.yml\n"
        "@@ -0,0 +1 @@\n+evil: true\n"
    )
    with pytest.raises(UnsafePatchError, match="forbidden path"):
        p.apply_diff(diff, thread_ids=["T1"])


# --- Phase 9 free functions ---------------------------------------------


def test_extract_touched_files_union_of_headers():
    diff = (
        "diff --git a/src/a.py b/src/a.py\n"
        "--- a/src/a.py\n+++ b/src/a.py\n"
        "@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/src/b.py b/src/b.py\n"
        "--- a/src/b.py\n+++ b/src/b.py\n"
        "@@ -1 +1 @@\n-x\n+y\n"
    )
    assert extract_touched_files(diff) == ["src/a.py", "src/b.py"]


def test_extract_touched_files_filters_dev_null():
    diff = (
        "diff --git a/src/new.py b/src/new.py\n"
        "--- /dev/null\n+++ b/src/new.py\n"
        "@@ -0,0 +1 @@\n+content\n"
    )
    assert extract_touched_files(diff) == ["src/new.py"]


def test_extract_touched_files_includes_plus_only_when_diff_git_absent():
    diff = "--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    assert extract_touched_files(diff) == ["src/x.py"]


def test_count_patch_lines_excludes_headers():
    diff = (
        "diff --git a/x b/x\n"
        "--- a/x\n+++ b/x\n"
        "@@ -1,2 +1,2 @@\n"
        "-old1\n-old2\n+new1\n+new2\n"
    )
    # Only 4 payload lines (--- and +++ headers excluded).
    assert count_patch_lines(diff) == 4


def test_count_patch_lines_counts_payload_starting_with_double_dash():
    """A removed line whose content begins with '--' is still a payload line."""
    diff = (
        "diff --git a/x b/x\n"
        "--- a/x\n+++ b/x\n"
        "@@ -1 +1 @@\n"
        "---some-arg\n"
        "+++some-arg\n"
    )
    # Both lines are payload (content beginning with -- and ++), not headers.
    # The header rows are at the top of the diff.
    assert count_patch_lines(diff) == 2


def test_violates_protected_paths_directory_prefix():
    assert violates_protected_paths(["infra/main.tf"], ["infra/"]) is True
    assert violates_protected_paths(["src/foo.py"], ["infra/"]) is False


def test_violates_protected_paths_glob():
    assert violates_protected_paths(["a/b.generated.py"], ["**/*.generated.*"]) is True


def test_violates_protected_paths_empty_protected_list():
    assert violates_protected_paths(["any/path"], []) is False


# --- apply_unified_diff (Phase 9 spec) ----------------------------------


def _kwargs(git_repo, **overrides):
    base = dict(
        repo_root=git_repo,
        protected_paths=[],
        max_files=15,
        max_patch_lines=800,
    )
    base.update(overrides)
    return base


def test_apply_unified_diff_happy_path(git_repo):
    diff = _diff(
        [("src/foo.py", "src/foo.py")],
        "@@ -1,2 +1,2 @@\n def f(x):\n-    return x\n+    return x or 0\n",
    )
    assert apply_unified_diff(diff, **_kwargs(git_repo)) is True
    assert (git_repo / "src/foo.py").read_text() == "def f(x):\n    return x or 0\n"


def test_apply_unified_diff_returns_false_on_empty_diff(git_repo):
    assert apply_unified_diff("", **_kwargs(git_repo)) is False
    assert apply_unified_diff("   \n  ", **_kwargs(git_repo)) is False


def test_apply_unified_diff_returns_false_on_too_many_files(git_repo):
    diff = _diff(
        [("a.py", "a.py"), ("b.py", "b.py")],
        "@@ -1 +1 @@\n-x\n+y\n",
    )
    assert apply_unified_diff(diff, **_kwargs(git_repo, max_files=1)) is False


def test_apply_unified_diff_returns_false_on_protected_path(git_repo):
    diff = _diff(
        [("infra/main.tf", "infra/main.tf")],
        "@@ -0,0 +1 @@\n+resource\n",
    )
    assert apply_unified_diff(diff, **_kwargs(git_repo, protected_paths=["infra/"])) is False


def test_apply_unified_diff_returns_false_on_too_many_lines(git_repo):
    big = "@@ -1,5 +1,5 @@\n" + "\n".join(["-old"] * 5 + ["+new"] * 5) + "\n"
    diff = _diff([("src/foo.py", "src/foo.py")], big)
    assert apply_unified_diff(diff, **_kwargs(git_repo, max_patch_lines=2)) is False


def test_apply_unified_diff_returns_false_on_unparseable(git_repo):
    assert apply_unified_diff("this is not a diff", **_kwargs(git_repo)) is False


def test_apply_unified_diff_returns_false_on_unappliable(git_repo):
    """git apply --check fails on context that doesn't match the working tree."""
    diff = _diff(
        [("src/foo.py", "src/foo.py")],
        "@@ -1,2 +1,2 @@\n totally\n-different\n+content\n",
    )
    assert apply_unified_diff(diff, **_kwargs(git_repo)) is False


def test_apply_unified_diff_reverts_on_git_diff_check_failure(git_repo):
    """When the applied diff introduces a merge-conflict marker (or similar),
    `git diff --check` reports it. The free function must revert the apply
    via `git apply --reverse` and return False, leaving the working tree
    exactly as it was before the call."""
    target = git_repo / "src/foo.py"
    pre_apply = target.read_text()
    # This diff is structurally valid and applies cleanly, but the added
    # content contains a merge-conflict marker which `git diff --check`
    # reports as an error.
    diff = (
        "diff --git a/src/foo.py b/src/foo.py\n"
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,2 +1,3 @@\n"
        " def f(x):\n"
        "+<<<<<<< HEAD\n"
        "     return x\n"
    )
    assert apply_unified_diff(diff, **_kwargs(git_repo)) is False
    # File content must be exactly what it was before the call.
    assert target.read_text() == pre_apply
