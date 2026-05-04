"""Per-run state: dedupe + loop control.

The persisted file shape (`.pr-agent-state.json`, gitignored) starts with
the four spec keys and then carries the existing audit trail:

    {
      "round": 3,
      "processed_comment_hashes": [...],
      "thread_attempt_counts": {...},
      "previous_unresolved_count": null,
      ...rest of AgentRunReport...
    }

Persistence is atomic (write to `.tmp`, then `os.replace`) so a crash
mid-write doesn't leave a half-flushed file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from .models import AgentRunReport, EscalationReason, ReviewComment, RoundResult

log = logging.getLogger(__name__)


class AgentState:
    def __init__(self, pr_number: int, persist_path: Path | None = None):
        self._report = AgentRunReport(pr_number=pr_number)
        self._failure_signatures: list[str] = []
        self._persist_path = persist_path

    # --- Hashing / dedupe --------------------------------------------------

    @staticmethod
    def comment_hash(c: ReviewComment) -> str:
        """sha256(author + body + path + line) per the Phase 7 spec."""
        parts = "|".join(
            [
                c.author,
                c.body,
                c.path or "",
                "" if c.line is None else str(c.line),
            ]
        )
        return hashlib.sha256(parts.encode("utf-8")).hexdigest()

    def already_processed(self, comment: ReviewComment) -> bool:
        return self.comment_hash(comment) in self._report.processed_comment_hashes

    def mark_processed(self, comment: ReviewComment) -> None:
        h = self.comment_hash(comment)
        if h not in self._report.processed_comment_hashes:
            self._report.processed_comment_hashes.append(h)
            self._persist()

    # --- Per-thread attempt counts ----------------------------------------

    def increment_attempt(self, thread_id: str) -> int:
        n = self._report.thread_attempt_counts.get(thread_id, 0) + 1
        self._report.thread_attempt_counts[thread_id] = n
        self._persist()
        return n

    def attempts_for(self, thread_id: str) -> int:
        return self._report.thread_attempt_counts.get(thread_id, 0)

    # --- Round / loop control ---------------------------------------------

    def start_round(self, unresolved_count: int) -> bool:
        """Begin a new round. Returns False if the loop should stop because
        the unresolved count did not decrease since the last round.

        On round 1 there is no prior count, so the call always returns True
        and just records ``previous_unresolved_count`` for the next round to
        compare against.
        """
        prev = self._report.previous_unresolved_count
        # First round: nothing to compare against yet.
        if prev is None:
            self._report.previous_unresolved_count = unresolved_count
            self._report.round += 1
            self._persist()
            return True
        if unresolved_count >= prev:
            log.info(
                "No progress: unresolved count %d >= previous %d. Stopping.",
                unresolved_count,
                prev,
            )
            return False
        self._report.previous_unresolved_count = unresolved_count
        self._report.round += 1
        self._persist()
        return True

    def record_round(self, round_result: RoundResult) -> None:
        self._report.rounds.append(round_result)
        self._persist()

    # --- Validation-failure signatures (used by the repeated-failure guard)

    def record_validation_failure(self, signature: str) -> int:
        self._failure_signatures.append(signature)
        return self._failure_signatures.count(signature)

    @staticmethod
    def signature_for(failure_text: str) -> str:
        return hashlib.sha256(failure_text.strip().encode()).hexdigest()[:16]

    # --- Escalation -------------------------------------------------------

    def escalate(self, reason: EscalationReason, unresolved: list[str]) -> None:
        self._report.escalated = True
        self._report.escalation_reason = reason
        self._report.final_unresolved_thread_ids = unresolved
        self._persist()

    @property
    def report(self) -> AgentRunReport:
        return self._report

    # --- Persistence ------------------------------------------------------

    def _persist(self) -> None:
        if not self._persist_path:
            return
        try:
            payload = json.dumps(asdict(self._report), indent=2, default=str)
            tmp = self._persist_path.with_suffix(self._persist_path.suffix + ".tmp")
            tmp.write_text(payload)
            os.replace(tmp, self._persist_path)
        except OSError as e:
            log.warning("Could not persist state: %s", e)
