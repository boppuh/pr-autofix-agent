"""Unit tests for the small pure helpers in pr_agent/run.py.

The full main() loop has integration risk that's not covered here — see the
PR description for the known-gap follow-up. This file just exercises the
deterministic string-building helpers.
"""

from __future__ import annotations

from pr_agent.models import ReviewComment, ReviewThread
from pr_agent.run import _summarize_human_threads


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
