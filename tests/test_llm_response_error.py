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
