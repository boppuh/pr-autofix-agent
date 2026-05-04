"""OpenAI provider (GPT / Codex via the Responses API)."""

from __future__ import annotations

import logging

from openai import OpenAI

from ..models import Classification, Patch, ReviewThread
from ._base import (
    CLASSIFY_SYSTEM,
    PATCH_SYSTEM,
    format_classify_user,
    format_patch_user,
    parse_classification,
    parse_patch,
)

log = logging.getLogger(__name__)


class OpenAIProvider:
    """Implements `LLMProvider` against the OpenAI Responses API.

    Uses `response_format={"type": "json_object"}` to force JSON output and
    `prompt_cache_key` for system-prompt caching (analogous to Anthropic's
    `cache_control: ephemeral`).
    """

    def __init__(self, model: str, api_key: str | None = None):
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._model = model

    def classify(self, thread: ReviewThread, file_excerpt: str | None) -> Classification:
        text = self._call(
            system=CLASSIFY_SYSTEM,
            user=format_classify_user(thread, file_excerpt),
            cache_key="pr-agent/classify",
            max_output_tokens=400,
        )
        return parse_classification(text, thread_id=thread.id)

    def propose_patch(
        self,
        thread: ReviewThread,
        file_contents: dict[str, str],
        max_files: int,
        prior_failure: str | None = None,
        pr_title: str | None = None,
        pr_body_excerpt: str | None = None,
        pr_diff_excerpt: str | None = None,
    ) -> Patch:
        user = format_patch_user(
            thread,
            file_contents,
            max_files,
            prior_failure,
            pr_title=pr_title,
            pr_body_excerpt=pr_body_excerpt,
            pr_diff_excerpt=pr_diff_excerpt,
        )
        text = self._call(
            system=PATCH_SYSTEM,
            user=user,
            cache_key="pr-agent/patch",
            max_output_tokens=16000,
        )
        return parse_patch(text, thread.id)

    def _call(
        self, *, system: str, user: str, cache_key: str, max_output_tokens: int
    ) -> str:
        resp = self._client.responses.create(
            model=self._model,
            instructions=system,
            input=user,
            max_output_tokens=max_output_tokens,
            prompt_cache_key=cache_key,
            text={"format": {"type": "json_object"}},
        )
        # Prefer the SDK's convenience attribute when available.
        text: str | None = getattr(resp, "output_text", None)
        if text:
            return text.strip()
        # Fallback: walk the structured output.
        parts: list[str] = []
        for item in getattr(resp, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                t = getattr(content, "text", None)
                if t:
                    parts.append(t)
        return "".join(parts).strip()
