from __future__ import annotations

import pytest

from pr_agent.llm import LLMResponseError
from pr_agent.llm._base import extract_json as _extract_json


def test_extract_json_raises_typed_error_on_truncated_input():
    """Regression: a Claude response truncated by max_tokens used to bubble
    up as a bare ValueError that crashed the agent loop. It must now raise
    LLMResponseError so run.py can catch and skip the thread."""
    truncated = '{"summary": "fix", "files": [{"path": "a", "new_content": "...'
    with pytest.raises(LLMResponseError):
        _extract_json(truncated)


def test_extract_json_accepts_valid_object():
    out = _extract_json('{"summary": "fix", "files": []}')
    assert out["summary"] == "fix"


def test_extract_json_strips_code_fence():
    out = _extract_json('```json\n{"x": 1}\n```')
    assert out == {"x": 1}


def test_patch_prompt_no_leading_newline_without_pr_context(thread_factory):
    """Regression: when no PR context (title/body/diff) is provided the
    formatted user prompt must start at 'Thread path: ...', not with a stray
    leading blank line."""
    from pr_agent.llm._base import format_patch_user

    out = format_patch_user(
        thread_factory(),
        file_contents={"src/foo.py": "x"},
        max_files=5,
        prior_failure=None,
    )
    assert not out.startswith("\n")
    assert out.startswith("Thread path:")


def test_parse_classification_handles_null_confidence(thread_factory):
    """End-to-end regression for the TypeError leak: an LLM reply with
    ``"confidence": null`` must fall back to NEEDS_HUMAN, not crash."""
    from pr_agent.llm._base import parse_classification

    raw = '{"category": "AUTO_FIX", "confidence": null, "reason": "x"}'
    cls = parse_classification(raw, thread_id="T_1")
    assert cls.category == "NEEDS_HUMAN"
    assert "failed validation" in cls.reason


def test_classify_returns_needs_human_on_truncated_response(monkeypatch, thread_factory):
    """Regression: when the classifier model returns non-JSON, classify must
    fall back to NEEDS_HUMAN instead of letting LLMResponseError escape and
    crash the agent loop."""
    from unittest.mock import MagicMock

    from pr_agent.llm import anthropic as a_mod
    from pr_agent.llm._factory import make_provider

    fake = MagicMock()
    # Truncated/garbage output the classifier might return.
    fake.messages.create.return_value = MagicMock(
        content=[MagicMock(type="text", text='{"category": "AUTO_FIX')]
    )
    monkeypatch.setattr(a_mod, "Anthropic", lambda **kw: fake)

    provider = make_provider("anthropic", model="claude-sonnet-4-6", api_key="k")
    cls = provider.classify(thread_factory(), file_excerpt=None)
    assert cls.category == "NEEDS_HUMAN"
    assert "failed validation" in cls.reason
    assert cls.thread_id == "T_1"
