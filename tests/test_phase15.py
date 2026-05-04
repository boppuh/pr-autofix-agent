"""Phase 15 spec-traceable test checklist.

One test per Phase 15 spec bullet, plus an integration test for the
max-rounds stop condition (the for/else clause in :func:`pr_agent.run.main`
that has no other direct coverage).

Each smoke test exercises a single canonical case so a reviewer can match
the spec line-by-line. Per-bucket exhaustive coverage lives in the
matching `tests/test_<topic>.py` file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

from pr_agent.classifier import Classifier
from pr_agent.github_client import get_unresolved_bugbot_threads
from pr_agent.models import (
    Classification,
    ClassificationCategory,
    PullRequest,
    ReviewComment,
    ReviewThread,
)
from pr_agent.patcher import (
    apply_unified_diff,
    count_patch_lines,
    extract_touched_files,
)
from pr_agent.run import main
from pr_agent.state import AgentState
from pr_agent.validator import run_validation


def _comment(author: str = "cursor", body: str = "fix it") -> ReviewComment:
    return ReviewComment(
        id="C1",
        author=author,
        body=body,
        path="src/foo.py",
        line=10,
        diff_hunk=None,
        created_at="2024-01-01T00:00:00Z",
    )


def _thread(thread_id: str = "T1", *, author: str = "cursor") -> ReviewThread:
    return ReviewThread(
        id=thread_id,
        is_resolved=False,
        comments=[_comment(author=author)],
    )


# --- 1. Bugbot author detection ----------------------------------------


def test_phase15_bugbot_author_detection():
    """`get_unresolved_bugbot_threads` keeps threads authored by any login
    in `bugbot_logins` (case-insensitive) and drops the rest."""
    import httpx
    import respx

    def thread_node(tid: str, login: str) -> dict:
        return {
            "id": tid,
            "isResolved": False,
            "isOutdated": False,
            "path": "src/foo.py",
            "line": 1,
            "comments": {
                "nodes": [
                    {
                        "id": "C",
                        "databaseId": 1,
                        "author": {"login": login},
                        "body": "x",
                        "path": "src/foo.py",
                        "line": 1,
                        "originalLine": 1,
                        "diffHunk": "",
                        "createdAt": "2024-01-01T00:00:00Z",
                    }
                ]
            },
        }

    payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "id": "PR_1",
                    "title": "t",
                    "body": "b",
                    "headRefName": "main",
                    "headRefOid": "x",
                    "baseRefName": "main",
                    "reviewThreads": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [
                            thread_node("T1", "cursor"),
                            thread_node("T2", "Cursor-Bugbot"),  # case-insensitive
                            thread_node("T3", "human-reviewer"),  # not Bugbot
                        ],
                    },
                }
            }
        }
    }
    with respx.mock(base_url="https://api.github.com") as mock:
        mock.post("/graphql").mock(return_value=httpx.Response(200, json=payload))
        out = get_unresolved_bugbot_threads(
            "o", "r", 1, ["cursor", "cursor-bugbot"], token="t"
        )
    assert sorted(t.id for t in out) == ["T1", "T2"]


# --- 2. Comment hashing ------------------------------------------------


def test_phase15_comment_hashing():
    """Spec hash is `sha256(author + body + path + line)`, full hex."""
    c = _comment()
    h = AgentState.comment_hash(c)
    assert len(h) == 64
    assert all(ch in "0123456789abcdef" for ch in h)
    # Sensitivity check (full coverage in test_state.py).
    h2 = AgentState.comment_hash(_comment(body="different"))
    assert h != h2


# --- 3. Classification -------------------------------------------------


def test_phase15_classification():
    """`Classifier.triage` routes threads into AUTO_FIX / NEEDS_HUMAN /
    IGNORE buckets based on rule layer + LLM fallback."""
    llm = MagicMock()
    # LLM never reached for these because each thread hits a rule bucket.
    classifier = Classifier(llm=llm, protected_paths=[])
    threads = [
        _thread("auto", author="cursor"),  # body 'fix it' -> no rule match
        _thread("human", author="cursor"),  # we'll override body
        _thread("ignore", author="cursor"),
    ]
    threads[0].comments[0].body = "missing null check on user.email"  # AUTO_FIX rule
    threads[1].comments[0].body = "Consider refactoring this auth handler."  # NEEDS_HUMAN
    threads[2].comments[0].body = "LGTM!"  # IGNORE

    out = classifier.triage(
        threads, file_excerpts={t.id: None for t in threads}
    )
    assert [t.id for t in out.fixable] == ["auto"]
    assert [t.id for t, _ in out.skipped] == ["human"]
    assert [t.id for t, _ in out.ignored] == ["ignore"]
    llm.classify.assert_not_called()


# --- 4. Protected path rejection ---------------------------------------


def test_phase15_protected_path_rejection(tmp_path: Path):
    """`apply_unified_diff` returns False for any diff touching a
    `protected_paths` entry (defense-in-depth alongside FORBIDDEN_GLOBS)."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@e.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "infra").mkdir()
    (tmp_path / "infra" / "main.tf").write_text("# placeholder\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    diff = (
        "diff --git a/infra/main.tf b/infra/main.tf\n"
        "--- a/infra/main.tf\n"
        "+++ b/infra/main.tf\n"
        "@@ -1 +1 @@\n"
        "-# placeholder\n"
        "+resource \"x\" \"y\" {}\n"
    )
    assert apply_unified_diff(
        diff,
        repo_root=tmp_path,
        protected_paths=["infra/"],
        max_files=10,
        max_patch_lines=100,
    ) is False


# --- 5. Patch line counting --------------------------------------------


def test_phase15_patch_line_counting():
    """`count_patch_lines` counts payload +/- lines, excludes ---/+++ headers."""
    diff = (
        "diff --git a/x b/x\n"
        "--- a/x\n+++ b/x\n"
        "@@ -1,2 +1,2 @@\n"
        "-old1\n-old2\n+new1\n+new2\n"
    )
    # 4 payload lines (--- and +++ headers are excluded).
    assert count_patch_lines(diff) == 4


# --- 6. Touched file extraction ----------------------------------------


def test_phase15_touched_file_extraction():
    """`extract_touched_files` returns the union of `diff --git a/X b/Y`
    and `+++ b/Y` headers."""
    diff = (
        "diff --git a/safe.py b/safe.py\n"
        "--- a/safe.py\n"
        "+++ b/.github/workflows/ci.yml\n"  # mismatched header
        "@@ -1 +1 @@\n-x\n+y\n"
    )
    paths = extract_touched_files(diff)
    # Both header sources are validated; the spoofed +++ target is included.
    assert "safe.py" in paths
    assert ".github/workflows/ci.yml" in paths


# --- 7. Validation command failure -------------------------------------


def test_phase15_validation_command_failure(tmp_path: Path):
    """`run_validation` aggregates per-command results; a non-zero exit
    flips `success` to False and surfaces the failing command."""
    out = run_validation(["true", "false"], repo_root=tmp_path)
    assert out.success is False
    assert out.first_failure is not None
    assert out.first_failure.name == "false"
    assert out.first_failure.exit_code != 0


# --- 8. Max rounds stop condition --------------------------------------


def test_phase15_max_rounds_stop_condition(tmp_path: Path, monkeypatch):
    """When the round loop completes its full `range(1, max_rounds + 1)`
    without a `break`, the for/else clause in main() escalates with
    EscalationReason.MAX_ROUNDS and applies the agent:max-rounds label.
    """
    # Fixture git repo so Patcher() can construct.
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@e.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "src.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    # No-validation config so we don't need npm/pytest.
    (tmp_path / ".pr-agent.yml").write_text("validation:\n  commands: []\n")

    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    # Decreasing thread counts each fetch so the no-progress guard never fires.
    threads_seq = [
        [_thread("T1"), _thread("T2")],  # round 1 fetch -> count=2
        [_thread("T1")],  # round 2 fetch -> count=1 (decreased)
        [_thread("T1")],  # else clause's get_unresolved_bugbot_threads
    ]

    fake_gh = MagicMock()
    fake_gh.get_unresolved_bugbot_threads.side_effect = lambda *a, **kw: threads_seq.pop(0)
    fake_gh.get_pr.return_value = PullRequest(
        id="PR_1",
        number=1,
        title="t",
        body="b",
        head_ref_name="main",
        head_ref_oid="x",
        base_ref_name="main",
    )
    fake_gh.get_pr_diff.return_value = ""
    fake_gh.get_file_contents.return_value = "x = 1\n"

    fake_llm = MagicMock()
    # Triage routes to AUTO_FIX so triage.fixable is non-empty (avoids the
    # all-needs-human early exit). Then the batch returns ESCALATE so no
    # commit happens, but the loop still continues to the next round.
    fake_llm.classify.return_value = Classification(
        thread_id="T",
        category=cast(ClassificationCategory, "AUTO_FIX"),
        reason="r",
        confidence=0.95,
    )
    fake_llm.generate_patch.return_value = "ESCALATE: simulated"

    monkeypatch.setattr(
        "pr_agent.run.GitHubClient.from_full_name", lambda *a, **kw: fake_gh
    )
    monkeypatch.setattr("pr_agent.run.make_provider", lambda *a, **kw: fake_llm)

    rc = main(["--repo", "o/r", "--pr", "1", "--max-rounds", "2"])

    assert rc == 0
    # Verify the for/else MAX_ROUNDS escalation by checking that
    # gh.add_labels was called with the agent:max-rounds label.
    flat_labels: list[str] = []
    for c in fake_gh.add_labels.call_args_list:
        # Calls look like: add_labels(pr_number, [labels...])
        args = c.args
        if len(args) >= 2 and isinstance(args[1], list):
            flat_labels.extend(args[1])
    assert "agent:max-rounds" in flat_labels, f"got labels: {flat_labels}"
    assert "agent:needs-human" in flat_labels  # phase 14 universal label


