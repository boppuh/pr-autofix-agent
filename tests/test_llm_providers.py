"""Tests for the pluggable LLM providers.

Mocks each SDK's call surface so we don't depend on having API keys.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pr_agent.llm import LLMProvider, make_provider
from pr_agent.llm._factory import default_model_for, env_var_for
from pr_agent.models import ReviewComment, ReviewThread


def _thread() -> ReviewThread:
    return ReviewThread(
        id="T1",
        is_resolved=False,
        comments=[
            ReviewComment(
                id="1",
                author="cursor",
                body="missing null check",
                path="src/foo.py",
                line=10,
                diff_hunk=None,
                created_at="2024-01-01T00:00:00Z",
            )
        ],
    )


# --- Factory --------------------------------------------------------------


def test_factory_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        make_provider("cohere", model="x", api_key="k")


def test_default_model_for():
    assert default_model_for("anthropic") == "claude-sonnet-4-6"
    assert default_model_for("openai") == "gpt-5-codex"
    # Unknown name falls back to anthropic.
    assert default_model_for("xyz") == "claude-sonnet-4-6"


def test_env_var_for():
    assert env_var_for("anthropic") == "ANTHROPIC_API_KEY"
    assert env_var_for("openai") == "OPENAI_API_KEY"


# --- Anthropic provider ---------------------------------------------------


def _anthropic_text_response(text: str):
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block])


def test_anthropic_classify_parses_json(monkeypatch):
    from pr_agent.llm import anthropic as a_mod

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _anthropic_text_response(
        '{"category": "AUTO_FIX", "confidence": 0.9, "reason": "ok"}'
    )
    monkeypatch.setattr(a_mod, "Anthropic", lambda **kw: fake_client)

    p = make_provider("anthropic", model="m", api_key="k")
    cls = p.classify(_thread(), file_excerpt="snippet")
    assert cls.category == "AUTO_FIX"
    assert cls.thread_id == "T1"
    assert cls.confidence == 0.9
    # Confirm system-prompt caching is wired.
    args = fake_client.messages.create.call_args
    assert args.kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_classify_legacy_label_mapping(monkeypatch):
    """Backward-compat: a model that emits the old `label` field should still parse."""
    from pr_agent.llm import anthropic as a_mod

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _anthropic_text_response(
        '{"label": "auto_fixable", "confidence": 0.9, "reason": "ok"}'
    )
    monkeypatch.setattr(a_mod, "Anthropic", lambda **kw: fake_client)
    p = make_provider("anthropic", model="m", api_key="k")
    cls = p.classify(_thread(), file_excerpt=None)
    assert cls.category == "AUTO_FIX"


def test_anthropic_classify_includes_prior_failure_in_prompt(monkeypatch):
    """When a previous round's patch failed validation, the failure
    text must reach the classifier — a thread that looked auto-fixable
    last round may need re-routing to NEEDS_HUMAN."""
    from pr_agent.llm import anthropic as a_mod

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _anthropic_text_response(
        '{"category": "NEEDS_HUMAN", "confidence": 0.9, "reason": "x"}'
    )
    monkeypatch.setattr(a_mod, "Anthropic", lambda **kw: fake_client)

    p = make_provider("anthropic", model="m", api_key="k")
    p.classify(_thread(), file_excerpt=None, prior_failure="ruff: F401 unused import")
    user_msg = fake_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Validation failure from a previous round" in user_msg
    assert "ruff: F401 unused import" in user_msg


def test_anthropic_propose_patch(monkeypatch):
    from pr_agent.llm import anthropic as a_mod

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _anthropic_text_response(
        '{"summary": "fix null", "files": [{"path": "src/foo.py", '
        '"new_content": "def f(x):\\n    return x or 0\\n", "rationale": "guard"}]}'
    )
    monkeypatch.setattr(a_mod, "Anthropic", lambda **kw: fake_client)

    p = make_provider("anthropic", model="m", api_key="k")
    patch = p.propose_patch(_thread(), {"src/foo.py": "old"}, max_files=5)
    assert patch.thread_id == "T1"
    assert patch.summary == "fix null"
    assert patch.files[0].path == "src/foo.py"


# --- OpenAI provider ------------------------------------------------------


def _openai_response(text: str):
    return SimpleNamespace(output_text=text)


def test_openai_classify_parses_json(monkeypatch):
    from pr_agent.llm import openai as o_mod

    fake_client = MagicMock()
    fake_client.responses.create.return_value = _openai_response(
        '{"category": "NEEDS_HUMAN", "confidence": 0.95, "reason": "arch"}'
    )
    monkeypatch.setattr(o_mod, "OpenAI", lambda **kw: fake_client)

    p = make_provider("openai", model="gpt-5-codex", api_key="k")
    cls = p.classify(_thread(), file_excerpt=None)
    assert cls.category == "NEEDS_HUMAN"
    args = fake_client.responses.create.call_args
    # JSON mode + prompt cache key are both passed.
    assert args.kwargs["text"] == {"format": {"type": "json_object"}}
    assert args.kwargs["prompt_cache_key"] == "pr-agent/classify"


def test_openai_propose_patch_includes_pr_context(monkeypatch):
    from pr_agent.llm import openai as o_mod

    fake_client = MagicMock()
    fake_client.responses.create.return_value = _openai_response(
        '{"summary": "fix", "files": []}'
    )
    monkeypatch.setattr(o_mod, "OpenAI", lambda **kw: fake_client)

    p = make_provider("openai", model="gpt-5-codex", api_key="k")
    p.propose_patch(
        _thread(),
        {"src/foo.py": "old"},
        max_files=5,
        pr_title="feat: foo",
        pr_body_excerpt="body",
        pr_diff_excerpt="diff",
    )
    user_input = fake_client.responses.create.call_args.kwargs["input"]
    assert "PR title: feat: foo" in user_input
    assert "PR description:" in user_input
    assert "PR diff (truncated):" in user_input


_VALID_DIFF = """diff --git a/src/foo.py b/src/foo.py
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,2 +1,2 @@
-old
+new
"""


def test_anthropic_generate_patch_passes_through_diff(monkeypatch):
    from pr_agent.llm import anthropic as a_mod

    fake = MagicMock()
    fake.messages.create.return_value = _anthropic_text_response(_VALID_DIFF)
    monkeypatch.setattr(a_mod, "Anthropic", lambda **kw: fake)

    p = make_provider("anthropic", model="m", api_key="k")
    out = p.generate_patch(
        pr_title="feat: foo",
        pr_body="body",
        pr_diff="diff goes here",
        comments=[_thread()],
        repo_context="ctx",
        validation_commands=["pytest -x"],
    )
    assert "diff --git a/src/foo.py" in out
    # System prompt is the Phase 8 verbatim text; cache_control still set.
    args = fake.messages.create.call_args
    sys_block = args.kwargs["system"][0]
    assert "autonomous PR repair agent" in sys_block["text"]
    assert sys_block["cache_control"] == {"type": "ephemeral"}
    user = args.kwargs["messages"][0]["content"]
    assert "feat: foo" in user
    assert "pytest -x" in user


def test_anthropic_generate_patch_passes_through_escalate(monkeypatch):
    from pr_agent.llm import anthropic as a_mod

    fake = MagicMock()
    fake.messages.create.return_value = _anthropic_text_response(
        "ESCALATE: needs product input on retention policy"
    )
    monkeypatch.setattr(a_mod, "Anthropic", lambda **kw: fake)
    p = make_provider("anthropic", model="m", api_key="k")
    out = p.generate_patch(
        pr_title="t",
        pr_body="b",
        pr_diff="",
        comments=[_thread()],
        repo_context="",
        validation_commands=[],
    )
    assert out.startswith("ESCALATE:")
    assert "retention policy" in out


def test_anthropic_generate_patch_rejects_markdown_fence(monkeypatch):
    from pr_agent.llm import _base
    from pr_agent.llm import anthropic as a_mod

    fake = MagicMock()
    fake.messages.create.return_value = _anthropic_text_response(
        "```diff\n" + _VALID_DIFF + "```"
    )
    monkeypatch.setattr(a_mod, "Anthropic", lambda **kw: fake)
    p = make_provider("anthropic", model="m", api_key="k")
    with pytest.raises(_base.LLMResponseError, match="markdown fence"):
        p.generate_patch(
            pr_title="t",
            pr_body="b",
            pr_diff="",
            comments=[_thread()],
            repo_context="",
            validation_commands=[],
        )


def test_anthropic_generate_patch_rejects_non_diff(monkeypatch):
    from pr_agent.llm import _base
    from pr_agent.llm import anthropic as a_mod

    fake = MagicMock()
    fake.messages.create.return_value = _anthropic_text_response(
        "Sure, here's some prose explaining the bug fix..."
    )
    monkeypatch.setattr(a_mod, "Anthropic", lambda **kw: fake)
    p = make_provider("anthropic", model="m", api_key="k")
    with pytest.raises(_base.LLMResponseError, match="not a unified diff"):
        p.generate_patch(
            pr_title="t",
            pr_body="b",
            pr_diff="",
            comments=[_thread()],
            repo_context="",
            validation_commands=[],
        )


def test_openai_generate_patch_disables_json_mode(monkeypatch):
    """generate_patch wants raw text (a diff), not JSON. The provider must
    NOT pass `text={'format': {'type': 'json_object'}}` to responses.create
    on this path."""
    from pr_agent.llm import openai as o_mod

    fake = MagicMock()
    fake.responses.create.return_value = _openai_response(_VALID_DIFF)
    monkeypatch.setattr(o_mod, "OpenAI", lambda **kw: fake)
    p = make_provider("openai", model="gpt-5-codex", api_key="k")
    out = p.generate_patch(
        pr_title="t",
        pr_body="b",
        pr_diff="d",
        comments=[_thread()],
        repo_context="ctx",
        validation_commands=["pytest -x"],
    )
    assert "diff --git" in out
    args = fake.responses.create.call_args
    assert "text" not in args.kwargs
    assert args.kwargs["prompt_cache_key"] == "pr-agent/generate-patch"


def test_protocol_runtime_check():
    """make_provider should return something that satisfies the Protocol."""
    from pr_agent.llm import anthropic as a_mod
    from pr_agent.llm import openai as o_mod

    a_mod.Anthropic = lambda **kw: MagicMock()  # type: ignore[assignment]
    o_mod.OpenAI = lambda **kw: MagicMock()  # type: ignore[assignment]
    assert isinstance(make_provider("anthropic", model="m", api_key="k"), LLMProvider)
    assert isinstance(make_provider("openai", model="m", api_key="k"), LLMProvider)
