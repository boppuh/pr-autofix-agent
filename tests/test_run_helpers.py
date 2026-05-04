"""Unit tests for the small pure helpers in pr_agent/run.py.

The full main() loop has integration risk that's not covered here — see the
PR description for the known-gap follow-up. This file just exercises the
deterministic string-building helpers.
"""

from __future__ import annotations

from pr_agent.models import (
    AgentRunReport,
    CommandResult,
    EscalatedThread,
    EscalationReason,
    HandledThread,
    ReviewComment,
    ReviewThread,
    RoundResult,
    ValidationResult,
)
from pr_agent.run import _format_run_summary, _summarize_human_threads


def _thread(thread_id: str, path: str | None, line: int | None) -> ReviewThread:
    return ReviewThread(
        id=thread_id,
        is_resolved=False,
        comments=[
            ReviewComment(
                id="c1",
                author="cursor",
                body="b",
                path=path,
                line=line,
                diff_hunk=None,
                created_at="2024-01-01T00:00:00Z",
            )
        ],
    )


def test_summarize_lists_each_thread_with_path_and_reason():
    skipped = [
        (_thread("T1", "src/a.py", 10), "rule: needs-human keyword 'payments'"),
        (_thread("T2", "src/b.py", 42), "llm: ambiguous architectural change"),
    ]
    out = _summarize_human_threads(skipped)
    assert "Threads:" in out
    assert "`src/a.py:10`" in out
    assert "rule: needs-human keyword 'payments'" in out
    assert "`src/b.py:42`" in out
    assert "llm: ambiguous architectural change" in out


def test_summarize_handles_missing_path_and_line():
    skipped = [(_thread("T1", None, None), "rule: protected path")]
    out = _summarize_human_threads(skipped)
    assert "(no path)" in out


def test_summarize_handles_path_without_line():
    skipped = [(_thread("T1", "src/x.py", None), "llm: too long")]
    out = _summarize_human_threads(skipped)
    # No trailing colon when line is unknown.
    assert "`src/x.py`" in out
    assert "`src/x.py:" not in out


def test_summarize_truncates_past_twenty_threads():
    skipped = [
        (_thread(f"T{i}", f"src/{i}.py", i), f"reason {i}")
        for i in range(25)
    ]
    out = _summarize_human_threads(skipped)
    # First 20 listed, the 21st mentioned only via the truncation marker.
    assert "src/0.py" in out
    assert "src/19.py" in out
    assert "src/20.py" not in out
    assert "and 5 more" in out


# --- _format_run_summary (Phase 13) -------------------------------------


def test_format_run_summary_empty_when_no_rounds():
    report = AgentRunReport(pr_number=42)
    assert _format_run_summary(report) == ""


def test_format_run_summary_single_round_full():
    report = AgentRunReport(pr_number=42)
    report.rounds.append(
        RoundResult(
            round_no=1,
            handled=[
                HandledThread(thread_id="T1", location="src/foo.ts:42",
                              summary="added null guard"),
                HandledThread(thread_id="T2", location="src/bar.ts:18",
                              summary="fixed incorrect type"),
            ],
            validation=ValidationResult(
                success=True,
                command_results=[
                    CommandResult(name="npm test", ok=True, exit_code=0),
                    CommandResult(name="npm run lint", ok=True, exit_code=0),
                    CommandResult(name="npm run typecheck", ok=True, exit_code=0),
                ],
            ),
            escalated_to_human=[
                EscalatedThread(thread_id="T3", location="src/auth/session.ts:120",
                                reason="rule: needs-human keyword 'auth'"),
            ],
        )
    )
    out = _format_run_summary(report)
    assert "## PR Autofix Agent — Run Summary" in out
    assert "### Round 1" in out
    # Handled section.
    assert "Handled:" in out
    assert "`src/foo.ts:42` — added null guard" in out
    assert "`src/bar.ts:18` — fixed incorrect type" in out
    # Validation section.
    assert "Validation:" in out
    assert "npm test: passed" in out
    assert "npm run lint: passed" in out
    assert "npm run typecheck: passed" in out
    # Escalated section.
    assert "Escalated:" in out
    assert "`src/auth/session.ts:120` — rule: needs-human keyword 'auth'" in out


def test_format_run_summary_validation_failure_status():
    report = AgentRunReport(pr_number=1)
    report.rounds.append(
        RoundResult(
            round_no=1,
            validation=ValidationResult(
                success=False,
                command_results=[
                    CommandResult(name="pytest", ok=False, exit_code=1,
                                  stderr_tail="F401"),
                ],
            ),
        )
    )
    out = _format_run_summary(report)
    assert "pytest: failed (exit 1)" in out


def test_format_run_summary_multi_round_with_empty_round():
    report = AgentRunReport(pr_number=1)
    report.rounds.append(
        RoundResult(
            round_no=1,
            handled=[HandledThread(thread_id="T1", location="x.py:1", summary="fix")],
        )
    )
    report.rounds.append(RoundResult(round_no=2))  # no actions
    out = _format_run_summary(report)
    assert "### Round 1" in out
    assert "### Round 2" in out
    assert "(no actions taken this round)" in out


def test_format_run_summary_includes_escalation_footer():
    report = AgentRunReport(pr_number=1)
    report.rounds.append(RoundResult(round_no=1))
    report.escalated = True
    report.escalation_reason = EscalationReason.MAX_ROUNDS
    report.final_unresolved_thread_ids = ["T1", "T2", "T3"]
    out = _format_run_summary(report)
    assert "Final status: escalated (`max_rounds`) — 3 unresolved thread(s)." in out
