"""Provider factory."""

from __future__ import annotations

import logging

from ._base import LLMProvider

log = logging.getLogger(__name__)


def make_provider(name: str, model: str, api_key: str | None) -> LLMProvider:
    """Construct a concrete provider by short name.

    Imports are lazy so installing one provider's SDK doesn't require the other.
    """
    n = name.lower()
    if n == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(model=model, api_key=api_key)
    if n == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(model=model, api_key=api_key)
    raise ValueError(f"Unknown LLM provider: {name!r}. Supported: anthropic, openai.")


DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-5-codex",
}


def default_model_for(name: str) -> str:
    return DEFAULT_MODELS.get(name.lower(), DEFAULT_MODELS["anthropic"])


def env_var_for(name: str) -> str:
    return {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(name.lower(), "ANTHROPIC_API_KEY")
