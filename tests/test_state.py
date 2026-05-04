from __future__ import annotations

import hashlib
import json

from pr_agent.models import EscalationReason, ReviewComment, RoundResult
from pr_agent.state import AgentState


def _comment(**overrides) -> ReviewComment:
    base = dict(
        id="C1",
        author="cursor",
        body="missing null check",
        path="src/foo.py",
        line=10,
        diff_hunk=None,
        created_at="2024-01-01T00:00:00Z",
    )
    base.update(overrides)
    return ReviewComment(**base)  # type: ignore[arg-type]


# --- Hash spec ----------------------------------------------------------


def test_comment_hash_matches_spec_formula():
    """sha256(author + body + path + line) per the Phase 7 spec."""
    c = _comment()
    expected = hashlib.sha256(
        "|".join([c.author, c.body, c.path or "", str(c.line)]).encode("utf-8")
    ).hexdigest()
    assert AgentState.comment_hash(c) == expected
    assert len(AgentState.comment_hash(c)) == 64  # full sha256


def test_comment_hash_is_stable_for_fixed_input():
    h1 = AgentState.comment_hash(_comment())
    h2 = AgentState.comment_hash(_comment())
    assert h1 == h2


def test_comment_hash_changes_when_any_input_changes():
    base = AgentState.comment_hash(_comment())
    assert AgentState.comment_hash(_comment(author="bugbot")) != base
    assert AgentState.comment_hash(_comment(body="other body")) != base
    assert AgentState.comment_hash(_comment(path="src/bar.py")) != base
    assert AgentState.comment_hash(_comment(line=11)) != base


def test_comment_hash_handles_none_path_and_line():
    c = _comment(path=None, line=None)
    h = AgentState.comment_hash(c)
    assert len(h) == 64  # doesn't crash


# --- Dedupe API ---------------------------------------------------------


def test_already_processed_round_trip(tmp_path):
    s = AgentState(pr_number=42, persist_path=tmp_path / ".pr-agent-state.json")
    c = _comment()
    assert not s.already_processed(c)
    s.mark_processed(c)
    assert s.already_processed(c)
    # A different body at the same location is a different hash.
    assert not s.already_processed(_comment(body="different body"))


def test_mark_processed_is_idempotent():
    s = AgentState(pr_number=1)
    c = _comment()
    s.mark_processed(c)
    s.mark_processed(c)
    assert s.report.processed_comment_hashes.count(AgentState.comment_hash(c)) == 1


# --- Per-thread attempts -----------------------------------------------


def test_increment_attempt_counts_per_thread():
    s = AgentState(pr_number=1)
    assert s.increment_attempt("T1") == 1
    assert s.increment_attempt("T1") == 2
    assert s.increment_attempt("T2") == 1
    assert s.attempts_for("T1") == 2
    assert s.attempts_for("T2") == 1
    assert s.attempts_for("T3") == 0


# --- start_round / no-progress guard -----------------------------------


def test_start_round_first_call_records_baseline():
    s = AgentState(pr_number=1)
    assert s.start_round(unresolved_count=5) is True
    assert s.report.previous_unresolved_count == 5
    assert s.report.round == 1


def test_start_round_returns_true_when_count_decreased():
    s = AgentState(pr_number=1)
    assert s.start_round(5) is True
    assert s.start_round(4) is True
    assert s.report.previous_unresolved_count == 4
    assert s.report.round == 2


def test_start_round_returns_false_when_count_did_not_decrease():
    """Strict stop: count flat or up = stop the loop."""
    s = AgentState(pr_number=1)
    s.start_round(5)
    # Flat: same number of unresolved threads.
    assert s.start_round(5) is False
    # The previous count is unchanged so we can't accidentally "make progress" later.
    assert s.report.previous_unresolved_count == 5


def test_start_round_returns_false_when_count_increased():
    s = AgentState(pr_number=1)
    s.start_round(3)
    assert s.start_round(7) is False


# --- Persistence (atomic + spec surface) -------------------------------


def test_state_file_top_level_has_spec_keys(tmp_path):
    path = tmp_path / ".pr-agent-state.json"
    s = AgentState(pr_number=42, persist_path=path)
    s.start_round(5)
    s.mark_processed(_comment())
    s.increment_attempt("T1")
    data = json.loads(path.read_text())
    # Spec surface keys present.
    assert "round" in data
    assert "processed_comment_hashes" in data
    assert "thread_attempt_counts" in data
    assert "previous_unresolved_count" in data
    assert data["round"] == 1
    assert len(data["processed_comment_hashes"]) == 1
    assert data["thread_attempt_counts"] == {"T1": 1}
    assert data["previous_unresolved_count"] == 5


def test_persistence_is_atomic_no_tmp_left_behind(tmp_path):
    path = tmp_path / ".pr-agent-state.json"
    s = AgentState(pr_number=1, persist_path=path)
    s.record_round(RoundResult(round_no=1))
    assert path.exists()
    # The .tmp file must have been atomically replaced, not left behind.
    assert not (tmp_path / ".pr-agent-state.json.tmp").exists()


def test_record_round_persists(tmp_path):
    path = tmp_path / ".pr-agent-state.json"
    s = AgentState(pr_number=42, persist_path=path)
    s.record_round(RoundResult(round_no=1, fixed_thread_ids=["T1"]))
    s.record_round(RoundResult(round_no=2, fixed_thread_ids=["T2"]))
    data = json.loads(path.read_text())
    assert data["pr_number"] == 42
    assert len(data["rounds"]) == 2
    assert data["rounds"][0]["fixed_thread_ids"] == ["T1"]


# --- Validation-failure signatures (unchanged) -------------------------


def test_repeated_failure_signature():
    s = AgentState(pr_number=1)
    sig = AgentState.signature_for("ruff failed: foo.py:1: F401\n")
    assert s.record_validation_failure(sig) == 1
    assert s.record_validation_failure(sig) == 2
    other = AgentState.signature_for("pytest failed elsewhere")
    assert s.record_validation_failure(other) == 1


# --- Escalation --------------------------------------------------------


def test_escalate_records_reason(tmp_path):
    path = tmp_path / ".pr-agent-state.json"
    s = AgentState(pr_number=1, persist_path=path)
    s.escalate(EscalationReason.NO_PROGRESS, ["T1", "T2"])
    data = json.loads(path.read_text())
    assert data["escalated"] is True
    assert data["escalation_reason"] == "no_progress"
    assert data["final_unresolved_thread_ids"] == ["T1", "T2"]
