from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ._paths import matches_any_protected
from .llm import LLMProvider
from .models import Classification, ReviewThread

log = logging.getLogger(__name__)

ARCHITECTURAL_PATTERNS = [
    r"\bconsider refactor",
    r"\barchitectur",
    r"\bdesign\b.*\b(decision|choice|smell|review)",
    r"\babstraction",
    r"\bapi contract",
    r"\bpublic api\b",
    r"\bbreaking change",
    r"\bsecurity (policy|model|posture)",
    r"\bschema (migration|change)",
    r"\bperformance (regression|tradeoff)",
    r"\bcoupling\b",
    r"\bseparation of concerns",
]

_ARCH_RE = re.compile("|".join(ARCHITECTURAL_PATTERNS), re.IGNORECASE)


@dataclass
class TriageOutcome:
    fixable: list[ReviewThread]
    skipped: list[tuple[ReviewThread, str]]   # routed to NEEDS_HUMAN bucket
    ignored: list[tuple[ReviewThread, str]] = field(default_factory=list)


class Classifier:
    def __init__(
        self,
        llm: LLMProvider,
        protected_paths: list[str],
        confidence_threshold: float = 0.7,
    ):
        self._llm = llm
        self._protected_paths = protected_paths
        self._confidence_threshold = confidence_threshold

    def triage(
        self,
        threads: list[ReviewThread],
        file_excerpts: dict[str, str | None],
    ) -> TriageOutcome:
        fixable: list[ReviewThread] = []
        skipped: list[tuple[ReviewThread, str]] = []
        ignored: list[tuple[ReviewThread, str]] = []
        for t in threads:
            rule_skip = self._rule_layer(t)
            if rule_skip:
                # Rule-layer skips are NEEDS_HUMAN, not IGNORE — protected paths
                # and architectural comments still warrant human attention.
                skipped.append((t, rule_skip))
                continue
            cls = self._llm.classify(t, file_excerpts.get(t.id))
            decision = self._apply_threshold(cls)
            if decision.category == "AUTO_FIX":
                fixable.append(t)
            elif decision.category == "IGNORE":
                ignored.append((t, f"ignored: {decision.reason}"))
            else:  # NEEDS_HUMAN
                skipped.append((t, f"llm: {decision.reason}"))
        log.info(
            "Triage: %d fixable, %d needs-human, %d ignored",
            len(fixable),
            len(skipped),
            len(ignored),
        )
        return TriageOutcome(fixable=fixable, skipped=skipped, ignored=ignored)

    def _rule_layer(self, thread: ReviewThread) -> str | None:
        if thread.path and self._is_protected(thread.path):
            return f"protected path: {thread.path}"
        body = thread.body_text
        if _ARCH_RE.search(body):
            return "architectural keyword match"
        if len(body) > 4000:
            return "comment too long for safe auto-fix"
        return None

    def _is_protected(self, path: str) -> bool:
        return matches_any_protected(path, self._protected_paths)

    def _apply_threshold(self, cls: Classification) -> Classification:
        # Only AUTO_FIX is gated by confidence; IGNORE / NEEDS_HUMAN pass through
        # unchanged (the harm of low-confidence IGNORE is just a missed silent-skip,
        # whereas low-confidence AUTO_FIX risks a bad code change).
        if cls.category == "AUTO_FIX" and cls.confidence < self._confidence_threshold:
            return Classification(
                thread_id=cls.thread_id,
                category="NEEDS_HUMAN",
                reason=f"low confidence ({cls.confidence:.2f}): {cls.reason}",
                confidence=cls.confidence,
            )
        return cls
