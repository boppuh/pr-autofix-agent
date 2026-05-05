"""Triage Bugbot review threads into AUTO_FIX / NEEDS_HUMAN / IGNORE.

The rule layer fires first and short-circuits when a keyword matches.
The LLM is consulted only when no rule fires. Match priority:

  1. Path-based: protected_paths -> NEEDS_HUMAN
  2. Comment too long -> NEEDS_HUMAN
  3. NEEDS_HUMAN keyword
  4. AUTO_FIX keyword
  5. IGNORE keyword (whole-comment anchored)
  6. Fall through to LLM

NEEDS_HUMAN is checked before AUTO_FIX so "missing null check in payment
processor" routes to a human. IGNORE is checked last so "LGTM, but add a
null check" still routes to AUTO_FIX.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ._paths import matches_any_protected
from .llm import LLMProvider
from .models import Classification, ReviewThread

log = logging.getLogger(__name__)

# --- Pattern lists ---------------------------------------------------------

NEEDS_HUMAN_PATTERNS: list[str] = [
    # architecture
    r"\barchitectur",
    r"\b(consider|needs?)\s+(a\s+)?refactor",
    r"\babstraction\b",
    r"\bcoupling\b",
    r"\bseparation of concerns\b",
    r"\b(rewrite|redesign)\s+(this|the)\b",
    # product / contract
    r"\b(product|business)\s+(behavior|behaviour|decision|requirement)\b",
    r"\bapi contract\b",
    r"\bpublic api\b",
    r"\bbreaking change\b",
    # database / schema
    r"\b(database|schema)\s+migration\b",
    r"\bmigrate\s+(the|a)\s+(table|schema|column)\b",
    # auth / security
    r"\bauthentication\b",
    r"\bauthorization\b",
    r"\bauth\s+(flow|handler|middleware|provider|context|guard|check)\b",
    r"\bsecurity\s+(policy|model|posture|implication|review|risk|concern)\b",
    r"\b(rbac|access\s+control)\b",
    r"\bpermissions?\s+model\b",
    # payments
    r"\b(payment|billing|charge|invoice|stripe|checkout)\b",
    # secrets
    r"\b(secret|credential)s?\b",
    r"\bapi[\s_-]?key\b",
    r"\btoken\s+leak\b",
    # infra
    r"\binfrastructure\b",
    r"\b(terraform|kubernetes|k8s|helm)\b",
    r"\bdeploy(ment)?\s+(config|pipeline)\b",
    # large rewrites
    r"\blarge\s+rewrite\b",
]

AUTO_FIX_PATTERNS: list[str] = [
    # lint / formatting / style
    r"\blint(er|ing)?\b",
    r"\b(eslint|ruff|flake8|pylint|black|prettier|gofmt|rustfmt)\b",
    r"\bformat(ting)?\b",
    r"\b(indent(ation)?|whitespace)\b",
    r"\btrailing\s+(comma|whitespace|newline)\b",
    # type errors
    r"\btype\s+(error|annotation|hint|mismatch)\b",
    r"\b(mypy|pyright|tsc)\b",
    r"\bwrong\s+type\b",
    # null / undefined / optional checks
    r"\b(null|none|undefined|nil)\s+check\b",
    r"\bnullable\b",
    r"\boptional\s+chaining\b",
    r"\bmissing\s+(check|guard)\b",
    # guard clauses
    r"\bguard\s+clause\b",
    r"\bearly\s+return\b",
    # missing tests
    r"\bmissing\s+tests?\b",
    r"\bno\s+tests?\b",
    r"\badd\s+(a\s+)?test\b",
    r"\btest\s+coverage\b",
    # variable usage / naming
    r"\bunused\s+\w+\b",
    r"\btypo\b",
    r"\bwrong\s+(name|variable|identifier|type|enum|order|reason)\b",
    r"\bshadow(s|ed|ing)\s+(a\s+)?\w+\b",
    # small logic bugs
    r"\boff[\s-]by[\s-]one\b",
    r"\bmissing\s+await\b",
    r"\bdead\s+\w+\b",
    r"\bunreachable\s+code\b",
    r"\bduplicat(ed?|ion|ing)\s+\w+\b",
]

# Whole-comment anchors so that "LGTM" inside a longer comment doesn't
# mark the thread as ignorable. Each pattern matches a comment whose entire
# body is the snippet plus optional trailing punctuation/whitespace.
# NOTE: compiled WITHOUT re.MULTILINE so that ^ and $ anchor to the start/end
# of the whole string, not individual lines.
IGNORE_PATTERNS: list[str] = [
    r"^\s*lgtm[!.\s]*$",
    r"^\s*thanks?(\s+you)?[!.\s]*$",
    r"^\s*looks?\s+good[!.\s]*$",
    r"^\s*ship\s+it[!.\s]*$",
    r"^\s*nit:\s*$",
    r"^\s*nice[!.\s]*$",
    r"^\s*approved?[!.\s]*$",
    r"^\s*no\s+changes?\s+needed[!.\s]*$",
]


def _compile(patterns: list[str], *, multiline: bool = True) -> list[re.Pattern[str]]:
    flags = re.IGNORECASE
    if multiline:
        flags |= re.MULTILINE
    return [re.compile(p, flags) for p in patterns]


_NEEDS_HUMAN_RE = _compile(NEEDS_HUMAN_PATTERNS)
_AUTO_FIX_RE = _compile(AUTO_FIX_PATTERNS)
# IGNORE patterns must NOT use MULTILINE — anchors must match the whole string.
_IGNORE_RE = _compile(IGNORE_PATTERNS, multiline=False)


def _first_match(body: str, regexes: list[re.Pattern[str]]) -> str | None:
    for r in regexes:
        m = r.search(body)
        if m:
            return m.group(0).strip()
    return None


# --- Triage ----------------------------------------------------------------


@dataclass
class TriageOutcome:
    fixable: list[ReviewThread]
    skipped: list[tuple[ReviewThread, str]]   # NEEDS_HUMAN bucket
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
        prior_failure: str | None = None,
    ) -> TriageOutcome:
        fixable: list[ReviewThread] = []
        skipped: list[tuple[ReviewThread, str]] = []
        ignored: list[tuple[ReviewThread, str]] = []
        for t in threads:
            ruled = self._rule_classify(t)
            if ruled is not None:
                self._route(t, ruled, fixable, skipped, ignored)
                continue
            # When a previous round's patch failed validation, pass the
            # failure to the classifier — a thread that looked like an
            # easy AUTO_FIX may turn out to need human judgment. Rule-
            # classified threads (protected paths, etc.) skip this path
            # by design; their answer doesn't depend on prior attempts.
            cls = self._llm.classify(t, file_excerpts.get(t.id), prior_failure)
            decision = self._apply_threshold(cls)
            self._route(t, decision, fixable, skipped, ignored)
        log.info(
            "Triage: %d fixable, %d needs-human, %d ignored",
            len(fixable),
            len(skipped),
            len(ignored),
        )
        return TriageOutcome(fixable=fixable, skipped=skipped, ignored=ignored)

    # --- Internals ---------------------------------------------------------

    def _rule_classify(self, thread: ReviewThread) -> Classification | None:
        """Return a deterministic Classification, or None to fall through to the LLM."""
        # 1. Protected path
        if thread.path and matches_any_protected(thread.path, self._protected_paths):
            return Classification(
                thread_id=thread.id,
                category="NEEDS_HUMAN",
                reason=f"rule: protected path {thread.path!r}",
                confidence=1.0,
            )
        body = thread.body_text
        # 2. Comment too long
        if len(body) > 4000:
            return Classification(
                thread_id=thread.id,
                category="NEEDS_HUMAN",
                reason="rule: comment too long for safe auto-fix",
                confidence=1.0,
            )
        # 3. NEEDS_HUMAN keyword (highest priority — safety wins)
        hit = _first_match(body, _NEEDS_HUMAN_RE)
        if hit:
            return Classification(
                thread_id=thread.id,
                category="NEEDS_HUMAN",
                reason=f"rule: needs-human keyword {hit!r}",
                confidence=1.0,
            )
        # 4. AUTO_FIX keyword
        hit = _first_match(body, _AUTO_FIX_RE)
        if hit:
            return Classification(
                thread_id=thread.id,
                category="AUTO_FIX",
                reason=f"rule: auto-fix keyword {hit!r}",
                confidence=1.0,
            )
        # 5. IGNORE keyword (whole-comment anchored)
        hit = _first_match(body, _IGNORE_RE)
        if hit:
            return Classification(
                thread_id=thread.id,
                category="IGNORE",
                reason=f"rule: ignore-only comment {hit!r}",
                confidence=1.0,
            )
        return None

    def _apply_threshold(self, cls: Classification) -> Classification:
        # Only AUTO_FIX from the LLM is gated by confidence.
        if cls.category == "AUTO_FIX" and cls.confidence < self._confidence_threshold:
            return Classification(
                thread_id=cls.thread_id,
                category="NEEDS_HUMAN",
                reason=f"low confidence ({cls.confidence:.2f}): {cls.reason}",
                confidence=cls.confidence,
            )
        return cls

    @staticmethod
    def _route(
        thread: ReviewThread,
        decision: Classification,
        fixable: list[ReviewThread],
        skipped: list[tuple[ReviewThread, str]],
        ignored: list[tuple[ReviewThread, str]],
    ) -> None:
        if decision.category == "AUTO_FIX":
            fixable.append(thread)
        elif decision.category == "IGNORE":
            ignored.append((thread, f"ignored: {decision.reason}"))
        else:  # NEEDS_HUMAN
            skipped.append((thread, decision.reason))
