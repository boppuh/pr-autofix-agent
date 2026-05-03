"""Pluggable LLM providers.

Public API:
- `LLMProvider` — protocol every provider implements.
- `LLMClient` — backwards-compatible alias kept for existing call sites
  (`run.py`, tests). It is `LLMProvider` underneath; the factory chooses the
  concrete implementation.
- `make_provider(name, model, api_key)` — factory.
"""

from __future__ import annotations

from ._base import LLMProvider, LLMResponseError
from ._factory import make_provider

# Backwards-compatible alias — code that imported `LLMClient` from
# `pr_agent.llm_client` keeps working by importing from `pr_agent.llm`.
LLMClient = LLMProvider

__all__ = ["LLMProvider", "LLMClient", "LLMResponseError", "make_provider"]
