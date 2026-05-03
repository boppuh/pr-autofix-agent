from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from .models import AgentRunReport, EscalationReason, RoundResult

log = logging.getLogger(__name__)


class AgentState:
    """In-memory state with optional JSON persistence for the run.

    Tracks idempotency keys so the agent never replies twice to the same
    (thread_id, comment_hash) pair, and tracks repeated validation failures.
    """

    def __init__(self, pr_number: int, persist_path: Path | None = None):
        self._report = AgentRunReport(pr_number=pr_number)
        self._handled: set[str] = set()  # idempotency keys
        self._failure_signatures: list[str] = []
        self._persist_path = persist_path

    @staticmethod
    def idempotency_key(thread_id: str, body: str) -> str:
        h = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        return f"{thread_id}:{h}"

    def already_handled(self, thread_id: str, body: str) -> bool:
        return self.idempotency_key(thread_id, body) in self._handled

    def mark_handled(self, thread_id: str, body: str) -> None:
        self._handled.add(self.idempotency_key(thread_id, body))

    def record_round(self, round_result: RoundResult) -> None:
        self._report.rounds.append(round_result)
        self._persist()

    def record_validation_failure(self, signature: str) -> int:
        """Record a validation failure signature; return how many times we've now seen it."""
        self._failure_signatures.append(signature)
        return self._failure_signatures.count(signature)

    @staticmethod
    def signature_for(failure_text: str) -> str:
        # Strip absolute paths/timestamps for stability.
        return hashlib.sha256(failure_text.strip().encode()).hexdigest()[:16]

    def escalate(self, reason: EscalationReason, unresolved: list[str]) -> None:
        self._report.escalated = True
        self._report.escalation_reason = reason
        self._report.final_unresolved_thread_ids = unresolved
        self._persist()

    @property
    def report(self) -> AgentRunReport:
        return self._report

    def _persist(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.write_text(self._report.model_dump_json(indent=2))
        except OSError as e:
            log.warning("Could not persist state: %s", e)


