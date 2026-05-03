from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass

from .llm_client import LLMClient
from .models import Classification, ClassificationLabel, ReviewThread

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
    skipped: list[tuple[ReviewThread, str]]


class Classifier:
    def __init__(
        self,
        llm: LLMClient,
        exclude_paths: list[str],
        confidence_threshold: float = 0.7,
    ):
        self._llm = llm
        self._exclude_paths = exclude_paths
        self._confidence_threshold = confidence_threshold

    def triage(
        self,
        threads: list[ReviewThread],
        file_excerpts: dict[str, str | None],
    ) -> TriageOutcome:
        fixable: list[ReviewThread] = []
        skipped: list[tuple[ReviewThread, str]] = []
        for t in threads:
            rule_skip = self._rule_layer(t)
            if rule_skip:
                skipped.append((t, rule_skip))
                continue
            cls = self._llm.classify(t, file_excerpts.get(t.id))
            decision = self._apply_threshold(cls)
            if decision.label is ClassificationLabel.AUTO_FIXABLE:
                fixable.append(t)
            else:
                skipped.append((t, f"llm: {decision.reason}"))
        log.info("Triage: %d fixable, %d skipped", len(fixable), len(skipped))
        return TriageOutcome(fixable=fixable, skipped=skipped)

    def _rule_layer(self, thread: ReviewThread) -> str | None:
        if thread.path and self._is_excluded(thread.path):
            return f"path excluded by config: {thread.path}"
        body = thread.body_text
        if _ARCH_RE.search(body):
            return "architectural keyword match"
        if len(body) > 4000:
            return "comment too long for safe auto-fix"
        return None

    def _is_excluded(self, path: str) -> bool:
        return any(fnmatch.fnmatch(path, pat) for pat in self._exclude_paths)

    def _apply_threshold(self, cls: Classification) -> Classification:
        if (
            cls.label is ClassificationLabel.AUTO_FIXABLE
            and cls.confidence < self._confidence_threshold
        ):
            return Classification(
                label=ClassificationLabel.HUMAN_REQUIRED,
                confidence=cls.confidence,
                reason=f"low confidence ({cls.confidence:.2f}): {cls.reason}",
            )
        return cls
