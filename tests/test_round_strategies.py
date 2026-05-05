"""Unit tests for the round-strategy helpers in ``pr_agent.run``.

``_attempt_batch_round`` and ``_attempt_per_thread_round`` are the two
sides of the strategy split that hides the batched / per-thread
duplication from the main loop. These tests verify each one in
isolation: what it returns, what it leaves on ``state`` /
``round_result``, and the cases where it falls through (returns
``None``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pr_agent.llm import LLMResponseError
from pr_agent.models import (
    Patch,
    PatchFile,
    PullRequest,
    ReviewComment,
    ReviewThread,
    RoundResult,
    SafetyLimits,
)
from pr_agent.patcher import Patcher, UnsafePatchError
from pr_agent.run import (
    RoundAttempt,
    _attempt_batch_round,
    _attempt_per_thread_round,
)
from pr_agent.state import AgentState

# --- Fixtures -------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Bare-but-real git repo so Patcher() and apply_diff have a home.
    Strategy tests don't actually invoke git unless the patcher path
    runs, but Patcher() resolves repo_root, so a real path is simpler
    than mocking."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    (root / "src").mkdir()
    (root / "src" / "f.py").write_text("x = 1\n")
    return root


@pytest.fixture
def patcher(repo: Path) -> Patcher:
    return Patcher(
        repo_root=repo,
        protected_paths=[],
        max_files_touched=10,
        max_patch_lines=500,
    )


@pytest.fixture
def state(tmp_path: Path) -> AgentState:
    return AgentState(pr_number=1, persist_path=tmp_path / "state.json")


@pytest.fixture
def pr() -> PullRequest:
    return PullRequest(
        id="PR_1",
        number=1,
        title="t",
        body="b",
        head_ref_name="main",
        head_ref_oid="abc",
        base_ref_name="main",
    )


def _thread(thread_id: str) -> ReviewThread:
    return ReviewThread(
        id=thread_id,
        is_resolved=False,
        comments=[
            ReviewComment(
                id="c1",
                author="cursor[bot]",
                body="missing null check",
                path="src/f.py",
                line=1,
                diff_hunk=None,
                created_at="2024-01-01T00:00:00Z",
            )
        ],
    )


# --- _attempt_batch_round -------------------------------------------------


def test_batch_returns_none_when_llm_raises_response_error(repo, patcher, state, pr):
    """LLMResponseError → log + fall through, no attempt incremented."""
    llm = MagicMock()
    llm.generate_patch.side_effect = LLMResponseError("bad json")

    out = _attempt_batch_round(
        live_fixable=[_thread("T1")],
        llm=llm,
        patcher=patcher,
        state=state,
        repo_root=repo,
        pr=pr,
        pr_diff="",
        validation_commands=[],
        last_failure=None,
        dry_run=False,
        round_no=1,
    )
    assert out is None
    assert state.attempts_for("T1") == 0


def test_batch_returns_none_when_llm_raises_generic_exception(repo, patcher, state, pr):
    llm = MagicMock()
    llm.generate_patch.side_effect = RuntimeError("rate limit")

    out = _attempt_batch_round(
        live_fixable=[_thread("T1")],
        llm=llm,
        patcher=patcher,
        state=state,
        repo_root=repo,
        pr=pr,
        pr_diff="",
        validation_commands=[],
        last_failure=None,
        dry_run=False,
        round_no=1,
    )
    assert out is None
    assert state.attempts_for("T1") == 0


def test_batch_returns_none_on_escalate(repo, patcher, state, pr):
    """ESCALATE means fall through to per-thread; don't burn attempts
    on the batch path (per-thread will increment as it consumes them)."""
    llm = MagicMock()
    llm.generate_patch.return_value = "ESCALATE: too big"

    out = _attempt_batch_round(
        live_fixable=[_thread("T1"), _thread("T2")],
        llm=llm,
        patcher=patcher,
        state=state,
        repo_root=repo,
        pr=pr,
        pr_diff="",
        validation_commands=[],
        last_failure=None,
        dry_run=False,
        round_no=1,
    )
    assert out is None
    assert state.attempts_for("T1") == 0
    assert state.attempts_for("T2") == 0


def test_batch_dry_run_increments_attempts_and_signals_terminate(repo, patcher, state, pr):
    """Dry-run: don't mutate the tree, but still consume attempts and
    return a terminator so the main loop can _finish() cleanly."""
    llm = MagicMock()
    llm.generate_patch.return_value = (
        "diff --git a/src/f.py b/src/f.py\n"
        "--- a/src/f.py\n+++ b/src/f.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    )

    out = _attempt_batch_round(
        live_fixable=[_thread("T1"), _thread("T2")],
        llm=llm,
        patcher=patcher,
        state=state,
        repo_root=repo,
        pr=pr,
        pr_diff="",
        validation_commands=[],
        last_failure=None,
        dry_run=True,
        round_no=1,
    )
    assert isinstance(out, RoundAttempt)
    assert out.dry_run_terminate is True
    assert out.applied_paths == []
    assert state.attempts_for("T1") == 1
    assert state.attempts_for("T2") == 1


def test_batch_returns_none_when_apply_diff_rejects(repo, patcher, state, pr, monkeypatch):
    """Patcher rejects the diff (e.g. exceeds size cap). Fall through
    to per-thread; do NOT increment attempts here (per-thread will)."""
    llm = MagicMock()
    llm.generate_patch.return_value = (
        "diff --git a/src/f.py b/src/f.py\n"
        "--- a/src/f.py\n+++ b/src/f.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    )

    def _raise(*a, **kw):
        raise UnsafePatchError("too many files")

    monkeypatch.setattr(patcher, "apply_diff", _raise)

    out = _attempt_batch_round(
        live_fixable=[_thread("T1")],
        llm=llm,
        patcher=patcher,
        state=state,
        repo_root=repo,
        pr=pr,
        pr_diff="",
        validation_commands=[],
        last_failure=None,
        dry_run=False,
        round_no=1,
    )
    assert out is None
    assert state.attempts_for("T1") == 0


def test_batch_success_produces_attempt_with_one_summary_line(
    repo, patcher, state, pr, monkeypatch
):
    """Batched apply collapses to a single summary line listing every
    thread — no per-thread duplication in the commit body."""
    llm = MagicMock()
    llm.generate_patch.return_value = "valid-diff"
    monkeypatch.setattr(
        patcher, "apply_diff", lambda diff, ids: [repo / "src" / "f.py"]
    )

    out = _attempt_batch_round(
        live_fixable=[_thread("T1"), _thread("T2"), _thread("T3")],
        llm=llm,
        patcher=patcher,
        state=state,
        repo_root=repo,
        pr=pr,
        pr_diff="",
        validation_commands=[],
        last_failure=None,
        dry_run=False,
        round_no=4,
    )
    assert isinstance(out, RoundAttempt)
    assert out.dry_run_terminate is False
    assert out.applied_paths == [repo / "src" / "f.py"]
    assert len(out.patches) == 3
    # All three threads share the same synthetic Patch (one per round).
    assert {id(p) for _, p in out.patches} == {id(out.patches[0][1])}
    assert out.patches[0][1].thread_id == "batch-r4"
    # ONE summary line, listing all thread IDs.
    assert len(out.summary_lines) == 1
    assert "T1" in out.summary_lines[0]
    assert "T2" in out.summary_lines[0]
    assert "T3" in out.summary_lines[0]
    # Attempts incremented once per thread.
    assert state.attempts_for("T1") == 1
    assert state.attempts_for("T2") == 1
    assert state.attempts_for("T3") == 1


def test_batch_forwards_prior_failure_to_llm(repo, patcher, state, pr):
    llm = MagicMock()
    llm.generate_patch.return_value = None  # any not-applicable response is fine
    _attempt_batch_round(
        live_fixable=[_thread("T1")],
        llm=llm,
        patcher=patcher,
        state=state,
        repo_root=repo,
        pr=pr,
        pr_diff="",
        validation_commands=[],
        last_failure="ruff: F401 unused import",
        dry_run=False,
        round_no=1,
    )
    assert (
        llm.generate_patch.call_args.kwargs["prior_failure"]
        == "ruff: F401 unused import"
    )


# --- _attempt_per_thread_round -------------------------------------------


def _safety() -> SafetyLimits:
    return SafetyLimits(max_files_touched=5, max_patch_lines=500)


def test_per_thread_returns_none_when_no_safe_patches(repo, patcher, state, pr):
    """Every thread errors out → no patches → None signals "break"."""
    llm = MagicMock()
    llm.propose_patch.side_effect = LLMResponseError("unparseable")
    rr = RoundResult(round_no=1)

    out = _attempt_per_thread_round(
        live_fixable=[_thread("T1"), _thread("T2")],
        llm=llm,
        patcher=patcher,
        state=state,
        safety=_safety(),
        repo_root=repo,
        pr=pr,
        pr_diff="",
        last_failure=None,
        dry_run=False,
        round_result=rr,
    )
    assert out is None
    # Skip reasons recorded on round_result.
    assert {tid for tid, _ in rr.skipped} == {"T1", "T2"}
    assert all("llm output unusable" in reason for _, reason in rr.skipped)
    # Attempts ARE consumed on the per-thread path (intent: each thread
    # got its own dedicated try, even if it failed).
    assert state.attempts_for("T1") == 1
    assert state.attempts_for("T2") == 1


def test_per_thread_dry_run_returns_terminate_after_one_safe_patch(
    repo, patcher, state, pr
):
    llm = MagicMock()
    llm.propose_patch.return_value = Patch(
        thread_id="T1",
        files=[PatchFile(path="src/f.py", new_content="x = 2\n", rationale="r")],
        summary="fix",
    )
    rr = RoundResult(round_no=1)

    out = _attempt_per_thread_round(
        live_fixable=[_thread("T1")],
        llm=llm,
        patcher=patcher,
        state=state,
        safety=_safety(),
        repo_root=repo,
        pr=pr,
        pr_diff="",
        last_failure=None,
        dry_run=True,
        round_result=rr,
    )
    assert isinstance(out, RoundAttempt)
    assert out.dry_run_terminate is True
    assert out.applied_paths == []


def test_per_thread_success_one_summary_line_per_thread(repo, patcher, state, pr):
    """The per-thread path produces one summary line per thread, with
    that thread's specific patch summary — no batch collapsing."""
    llm = MagicMock()

    def _propose(*, thread, **kw):
        return Patch(
            thread_id=thread.id,
            files=[
                PatchFile(
                    path="src/f.py",
                    new_content=f"x = {thread.id}\n",
                    rationale="r",
                )
            ],
            summary=f"fix for {thread.id}",
        )

    llm.propose_patch.side_effect = _propose
    rr = RoundResult(round_no=1)

    out = _attempt_per_thread_round(
        live_fixable=[_thread("T1"), _thread("T2")],
        llm=llm,
        patcher=patcher,
        state=state,
        safety=_safety(),
        repo_root=repo,
        pr=pr,
        pr_diff="",
        last_failure=None,
        dry_run=False,
        round_result=rr,
    )
    assert isinstance(out, RoundAttempt)
    # NB: both patches touch src/f.py so the second .apply() will overwrite
    # the first — applied_paths still records both.
    assert len(out.patches) == 2
    assert len(out.summary_lines) == 2
    assert "T1" in out.summary_lines[0] and "fix for T1" in out.summary_lines[0]
    assert "T2" in out.summary_lines[1] and "fix for T2" in out.summary_lines[1]


def test_per_thread_records_unsafe_patch_in_skipped(repo, state, pr):
    """A safe-check rejection routes to round_result.skipped and the
    thread doesn't enter the patches list."""
    # Patcher with strict file count so any patch trips check_safe.
    strict_patcher = Patcher(
        repo_root=repo, protected_paths=[], max_files_touched=0, max_patch_lines=500
    )
    llm = MagicMock()
    llm.propose_patch.return_value = Patch(
        thread_id="T1",
        files=[PatchFile(path="src/f.py", new_content="x = 2\n", rationale="r")],
        summary="fix",
    )
    rr = RoundResult(round_no=1)

    out = _attempt_per_thread_round(
        live_fixable=[_thread("T1")],
        llm=llm,
        patcher=strict_patcher,
        state=state,
        safety=_safety(),
        repo_root=repo,
        pr=pr,
        pr_diff="",
        last_failure=None,
        dry_run=False,
        round_result=rr,
    )
    assert out is None
    assert len(rr.skipped) == 1
    assert rr.skipped[0][0] == "T1"
    assert "unsafe" in rr.skipped[0][1]
