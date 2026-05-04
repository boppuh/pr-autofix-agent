from __future__ import annotations

import pytest

from pr_agent.llm_client import LLMResponseError, _extract_json


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


def test_classify_returns_human_required_on_truncated_response(monkeypatch, thread_factory):
    """Regression: when the classifier model returns non-JSON, classify must
    fall back to HUMAN_REQUIRED instead of letting LLMResponseError escape and
    crash the agent loop."""
    from unittest.mock import MagicMock

    from pr_agent import llm_client as m
    from pr_agent.models import ClassificationLabel

    fake = MagicMock()
    # Truncated/garbage output the classifier might return.
    fake.messages.create.return_value = MagicMock(
        content=[MagicMock(type="text", text='{"label": "auto_fix')]
    )
    monkeypatch.setattr(m, "Anthropic", lambda **kw: fake)

    client = m.LLMClient(model="claude-sonnet-4-6", api_key="k")
    cls = client.classify(thread_factory(), file_excerpt=None)
    assert cls.label is ClassificationLabel.HUMAN_REQUIRED
    assert "failed validation" in cls.reason
