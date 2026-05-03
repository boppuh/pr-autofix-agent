from __future__ import annotations

from unittest.mock import MagicMock

from pr_agent.classifier import Classifier
from pr_agent.models import Classification, ClassificationLabel


def _llm_returning(label: ClassificationLabel, confidence: float) -> MagicMock:
    llm = MagicMock()
    llm.classify.return_value = Classification(
        label=label, confidence=confidence, reason="ok"
    )
    return llm


def test_rule_layer_skips_architectural(thread_factory):
    llm = _llm_returning(ClassificationLabel.AUTO_FIXABLE, 0.99)
    c = Classifier(llm=llm, exclude_paths=[])
    t = thread_factory(body="Consider refactoring this module to reduce coupling.")
    out = c.triage([t], file_excerpts={t.id: ""})
    assert not out.fixable
    assert out.skipped[0][1].startswith("architectural")
    llm.classify.assert_not_called()


def test_rule_layer_skips_excluded_path(thread_factory):
    llm = _llm_returning(ClassificationLabel.AUTO_FIXABLE, 0.99)
    c = Classifier(llm=llm, exclude_paths=["migrations/**"])
    t = thread_factory(path="migrations/0001_init.py", body="missing null check")
    out = c.triage([t], file_excerpts={t.id: ""})
    assert not out.fixable
    assert "excluded" in out.skipped[0][1]


def test_low_confidence_routed_to_human(thread_factory):
    llm = _llm_returning(ClassificationLabel.AUTO_FIXABLE, 0.5)
    c = Classifier(llm=llm, exclude_paths=[], confidence_threshold=0.7)
    t = thread_factory(body="missing null check on user.email")
    out = c.triage([t], file_excerpts={t.id: "code"})
    assert not out.fixable
    assert out.skipped[0][1].startswith("llm: low confidence")


def test_high_confidence_auto_fixable(thread_factory):
    llm = _llm_returning(ClassificationLabel.AUTO_FIXABLE, 0.95)
    c = Classifier(llm=llm, exclude_paths=[])
    t = thread_factory(body="off-by-one in loop bound")
    out = c.triage([t], file_excerpts={t.id: "code"})
    assert len(out.fixable) == 1
    assert not out.skipped


def test_human_required_label_is_skipped(thread_factory):
    llm = _llm_returning(ClassificationLabel.HUMAN_REQUIRED, 0.9)
    c = Classifier(llm=llm, exclude_paths=[])
    t = thread_factory(body="missing null check")
    out = c.triage([t], file_excerpts={t.id: ""})
    assert not out.fixable
    assert out.skipped[0][1].startswith("llm:")
