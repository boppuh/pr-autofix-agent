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


# --- validate_diff_response (Phase 8 syntactic checks) -----------------


def test_validate_diff_response_passes_through_escalate():
    from pr_agent.llm._base import validate_diff_response

    out = validate_diff_response("ESCALATE: needs product input")
    assert out.startswith("ESCALATE:")


def test_validate_diff_response_strips_whitespace():
    from pr_agent.llm._base import validate_diff_response

    raw = "\n\n  diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    out = validate_diff_response(raw)
    assert out.startswith("diff --git")


def test_validate_diff_response_rejects_markdown_fence():
    from pr_agent.llm._base import LLMResponseError, validate_diff_response

    raw = "```diff\ndiff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n```"
    with pytest.raises(LLMResponseError, match="markdown fence"):
        validate_diff_response(raw)


def test_validate_diff_response_rejects_prose():
    from pr_agent.llm._base import LLMResponseError, validate_diff_response

    with pytest.raises(LLMResponseError, match="not a unified diff"):
        validate_diff_response("Sure, here is the fix: change line 5 to do X.")


def test_validate_diff_response_accepts_payload_lines_containing_backticks():
    """Regression: a valid diff that adds, removes, or has as context any
    line containing triple backticks (e.g. a markdown README, a Python
    docstring with code-fence examples) must NOT be rejected as a fence.
    The fence check is line-anchored — only lines that START with ``` are
    treated as wrapping fences."""
    from pr_agent.llm._base import validate_diff_response

    # Payload-line variants: + line containing ```, - line containing ```,
    # context line ' ```...' (the leading space is the diff context marker).
    raw = (
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1,4 +1,4 @@\n"
        " # Title\n"
        "-```python\n"
        "+```bash\n"
        " example\n"
        " ```\n"
    )
    out = validate_diff_response(raw)
    assert "diff --git" in out
    # Confirm the test really exercises payload lines containing fences.
    assert "```python" in out
    assert "```bash" in out
