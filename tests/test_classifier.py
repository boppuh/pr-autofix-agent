from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from pr_agent.classifier import Classifier
from pr_agent.models import Classification, ClassificationCategory


def _llm_returning(category: str, confidence: float) -> MagicMock:
    llm = MagicMock()
    llm.classify.return_value = Classification(
        thread_id="T_1",
        category=cast(ClassificationCategory, category),
        reason="ok",
        confidence=confidence,
    )
    return llm


def _classifier(*, protected_paths: list[str] | None = None, threshold: float = 0.7) -> tuple[Classifier, MagicMock]:
    llm = _llm_returning("AUTO_FIX", 0.99)  # default; tests override category as needed
    c = Classifier(
        llm=llm,
        protected_paths=protected_paths or [],
        confidence_threshold=threshold,
    )
    return c, llm


# ---------- Path / length rules -------------------------------------------


def test_rule_layer_skips_protected_dir_prefix(thread_factory):
    c, llm = _classifier(protected_paths=["migrations/"])
    t = thread_factory(path="migrations/0001_init.py", body="missing null check")
    out = c.triage([t], file_excerpts={t.id: ""})
    assert not out.fixable
    assert "protected path" in out.skipped[0][1]
    llm.classify.assert_not_called()


def test_rule_layer_skips_overly_long_comment(thread_factory):
    c, llm = _classifier()
    t = thread_factory(body="x" * 4001)
    out = c.triage([t], file_excerpts={t.id: ""})
    assert not out.fixable
    assert "too long" in out.skipped[0][1]
    llm.classify.assert_not_called()


# ---------- NEEDS_HUMAN keyword bucket ------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "Consider refactoring this module to reduce coupling.",
        "This needs an architecture review.",
        "Touches the public API",
        "This is a breaking change",
        "Database migration required for this column.",
        "Migrate the table to use the new schema.",
        "Add an authentication check before this branch.",
        "The auth middleware should validate scopes.",
        "Security policy violation here.",
        "RBAC is broken — non-admins can hit this.",
        "Don't log the api_key like this.",
        "Hard-coded secret in the config.",
        "Stripe integration needs idempotency keys.",
        "Move this to terraform; infrastructure shouldn't live in app code.",
        "Schema migration required",
        "This needs a large rewrite",
    ],
)
def test_rule_layer_routes_needs_human_keyword(thread_factory, body):
    c, llm = _classifier()
    t = thread_factory(body=body)
    out = c.triage([t], file_excerpts={t.id: ""})
    assert not out.fixable
    assert not out.ignored
    assert "needs-human keyword" in out.skipped[0][1]
    llm.classify.assert_not_called()


# ---------- AUTO_FIX keyword bucket ---------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "Lint failure on this file",
        "ESLint complains about this",
        "Trailing whitespace here",
        "Indentation off by two",
        "Type mismatch — this should be int",
        "mypy is unhappy with this signature",
        "Missing null check on user.email",
        "Use optional chaining here",
        "Add a guard clause for the empty list",
        "Early return when n == 0",
        "Missing tests for this branch",
        "Add a test for the error path",
        "Unused import in the top of the file",
        "Typo in identifier name",
        "Off-by-one in the loop bound",
        "Missing await on the async call",
        "Dead code below the return statement",
        "Unreachable code after the throw",
    ],
)
def test_rule_layer_routes_auto_fix_keyword(thread_factory, body):
    c, llm = _classifier()
    t = thread_factory(body=body)
    out = c.triage([t], file_excerpts={t.id: ""})
    assert len(out.fixable) == 1
    assert not out.skipped
    assert not out.ignored
    llm.classify.assert_not_called()


# ---------- IGNORE bucket -------------------------------------------------


@pytest.mark.parametrize(
    "body",
    ["LGTM", "lgtm!", "Thanks!", "thank you", "Looks good", "ship it!", "nit:", "nice!", "approved", "No changes needed"],
)
def test_rule_layer_routes_ignore_when_whole_comment(thread_factory, body):
    c, llm = _classifier()
    t = thread_factory(body=body)
    out = c.triage([t], file_excerpts={t.id: ""})
    assert not out.fixable
    assert not out.skipped
    assert len(out.ignored) == 1
    llm.classify.assert_not_called()


# ---------- Conflict / priority rules -------------------------------------


def test_needs_human_wins_over_auto_fix(thread_factory):
    """'Missing null check in payment processor' must route to NEEDS_HUMAN —
    the safety bucket beats the AUTO_FIX bucket whenever both match."""
    c, llm = _classifier()
    t = thread_factory(body="Missing null check in payment processor.")
    out = c.triage([t], file_excerpts={t.id: ""})
    assert not out.fixable
    assert "needs-human keyword" in out.skipped[0][1]
    llm.classify.assert_not_called()


def test_auto_fix_wins_over_ignore_when_both_match(thread_factory):
    """'LGTM, but please add a null check' must route to AUTO_FIX —
    actionable signal beats the LGTM framing."""
    c, llm = _classifier()
    t = thread_factory(body="LGTM, but please add a null check.")
    out = c.triage([t], file_excerpts={t.id: ""})
    assert len(out.fixable) == 1
    assert not out.ignored
    llm.classify.assert_not_called()


def test_ignore_does_not_match_inside_longer_comment(thread_factory):
    """LGTM as a substring inside a longer non-actionable comment shouldn't
    trigger IGNORE — only a whole-comment LGTM does. Falls through to LLM."""
    c, llm = _classifier()
    llm.classify.return_value = Classification(
        thread_id="T_1", category="NEEDS_HUMAN", reason="ambiguous", confidence=0.9
    )
    t = thread_factory(body="LGTM in spirit, but I have some thoughts on the design.")
    out = c.triage([t], file_excerpts={t.id: ""})
    # No rule fires; LLM is called and returns NEEDS_HUMAN.
    llm.classify.assert_called_once()
    assert "ambiguous" in out.skipped[0][1]


# ---------- Fall-through to LLM -------------------------------------------


def test_no_rule_match_falls_through_to_llm(thread_factory):
    c, llm = _classifier()
    llm.classify.return_value = Classification(
        thread_id="T_1", category="AUTO_FIX", reason="llm-decided", confidence=0.95
    )
    t = thread_factory(body="This function is interesting; could you double-check it?")
    out = c.triage([t], file_excerpts={t.id: ""})
    llm.classify.assert_called_once()
    assert len(out.fixable) == 1


def test_low_confidence_llm_auto_fix_routed_to_human(thread_factory):
    """The threshold gate still applies to LLM-emitted AUTO_FIX (rule-emitted
    AUTO_FIX is at confidence 1.0 and bypasses the gate)."""
    c, llm = _classifier(threshold=0.7)
    llm.classify.return_value = Classification(
        thread_id="T_1", category="AUTO_FIX", reason="ok", confidence=0.5
    )
    t = thread_factory(body="This function is interesting; could you double-check it?")
    out = c.triage([t], file_excerpts={t.id: ""})
    assert not out.fixable
    assert "low confidence" in out.skipped[0][1]


def test_llm_ignore_passes_through_threshold(thread_factory):
    """A low-confidence IGNORE from the LLM stays IGNORE — the threshold gate
    only downgrades AUTO_FIX (we don't want to escalate vague chatter)."""
    c, llm = _classifier(threshold=0.7)
    llm.classify.return_value = Classification(
        thread_id="T_1", category="IGNORE", reason="praise", confidence=0.4
    )
    t = thread_factory(body="This function is interesting; could you double-check it?")
    out = c.triage([t], file_excerpts={t.id: ""})
    assert not out.fixable
    assert not out.skipped
    assert len(out.ignored) == 1
