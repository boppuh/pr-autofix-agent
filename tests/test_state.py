from __future__ import annotations

import json

from pr_agent.models import EscalationReason, RoundResult
from pr_agent.state import AgentState


def test_idempotency_round_trip(tmp_path):
    s = AgentState(pr_number=42, persist_path=tmp_path / "state.json")
    assert not s.already_handled("T1", "body")
    s.mark_handled("T1", "body")
    assert s.already_handled("T1", "body")
    assert not s.already_handled("T1", "body modified")


def test_record_round_persists(tmp_path):
    path = tmp_path / "state.json"
    s = AgentState(pr_number=42, persist_path=path)
    s.record_round(RoundResult(round_no=1, fixed_thread_ids=["T1"]))
    s.record_round(RoundResult(round_no=2, fixed_thread_ids=["T2"]))
    data = json.loads(path.read_text())
    assert data["pr_number"] == 42
    assert len(data["rounds"]) == 2
    assert data["rounds"][0]["fixed_thread_ids"] == ["T1"]


def test_repeated_failure_signature():
    s = AgentState(pr_number=1)
    sig = AgentState.signature_for("ruff failed: foo.py:1: F401\n")
    assert s.record_validation_failure(sig) == 1
    assert s.record_validation_failure(sig) == 2
    other = AgentState.signature_for("pytest failed elsewhere")
    assert s.record_validation_failure(other) == 1


def test_escalate_records_reason(tmp_path):
    path = tmp_path / "state.json"
    s = AgentState(pr_number=1, persist_path=path)
    s.escalate(EscalationReason.MAX_ROUNDS, ["T1", "T2"])
    data = json.loads(path.read_text())
    assert data["escalated"] is True
    assert data["escalation_reason"] == "max_rounds"
    assert data["final_unresolved_thread_ids"] == ["T1", "T2"]
