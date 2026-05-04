from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

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


def test_rule_layer_skips_architectural(thread_factory):
    llm = _llm_returning("AUTO_FIX", 0.99)
    c = Classifier(llm=llm, protected_paths=[])
    t = thread_factory(body="Consider refactoring this module to reduce coupling.")
    out = c.triage([t], file_excerpts={t.id: ""})
    assert not out.fixable
    assert out.skipped[0][1].startswith("architectural")
    llm.classify.assert_not_called()


def test_rule_layer_skips_protected_dir_prefix(thread_factory):
    llm = _llm_returning("AUTO_FIX", 0.99)
    c = Classifier(llm=llm, protected_paths=["migrations/"])
    t = thread_factory(path="migrations/0001_init.py", body="missing null check")
    out = c.triage([t], file_excerpts={t.id: ""})
    assert not out.fixable
    assert out.skipped[0][1].startswith("protected path")


def test_low_confidence_routed_to_human(thread_factory):
    llm = _llm_returning("AUTO_FIX", 0.5)
    c = Classifier(llm=llm, protected_paths=[], confidence_threshold=0.7)
    t = thread_factory(body="missing null check on user.email")
    out = c.triage([t], file_excerpts={t.id: "code"})
    assert not out.fixable
    assert out.skipped[0][1].startswith("llm: low confidence")


def test_high_confidence_auto_fixable(thread_factory):
    llm = _llm_returning("AUTO_FIX", 0.95)
    c = Classifier(llm=llm, protected_paths=[])
    t = thread_factory(body="off-by-one in loop bound")
    out = c.triage([t], file_excerpts={t.id: "code"})
    assert len(out.fixable) == 1
    assert not out.skipped
    assert not out.ignored


def test_human_required_is_skipped(thread_factory):
    llm = _llm_returning("NEEDS_HUMAN", 0.9)
    c = Classifier(llm=llm, protected_paths=[])
    t = thread_factory(body="missing null check")
    out = c.triage([t], file_excerpts={t.id: ""})
    assert not out.fixable
    assert out.skipped[0][1].startswith("llm:")
    assert not out.ignored


def test_ignore_routed_separately(thread_factory):
    """IGNORE threads must land in `ignored`, not `skipped` (no escalation)."""
    llm = _llm_returning("IGNORE", 0.95)
    c = Classifier(llm=llm, protected_paths=[])
    t = thread_factory(body="LGTM, thanks!")
    out = c.triage([t], file_excerpts={t.id: ""})
    assert not out.fixable
    assert not out.skipped
    assert len(out.ignored) == 1
    assert out.ignored[0][1].startswith("ignored:")
