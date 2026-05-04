"""Anthropic Claude provider."""

from __future__ import annotations

import logging

from anthropic import Anthropic

from ..models import Classification, Patch, ReviewThread
from ._base import (
    CLASSIFY_SYSTEM,
    GENERATE_PATCH_SYSTEM,
    PATCH_SYSTEM,
    format_classify_user,
    format_generate_patch_user,
    format_patch_user,
    parse_classification,
    parse_patch,
    validate_diff_response,
)

log = logging.getLogger(__name__)


class AnthropicProvider:
    """Implements `LLMProvider` against the Anthropic Messages API."""

    def __init__(self, model: str, api_key: str | None = None):
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self._model = model

    def classify(self, thread: ReviewThread, file_excerpt: str | None) -> Classification:
        text = self._call(
            system=CLASSIFY_SYSTEM,
            user=format_classify_user(thread, file_excerpt),
            max_tokens=400,
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
        # Full file contents need headroom; 4000 was hitting truncation on
        # real-sized source files (then crashing patch JSON parsing).
        text = self._call(system=PATCH_SYSTEM, user=user, max_tokens=16000)
        return parse_patch(text, thread.id)

    def generate_patch(
        self,
        *,
        pr_title: str,
        pr_body: str,
        pr_diff: str,
        comments: list[ReviewThread],
        repo_context: str,
        validation_commands: list[str],
        prior_failure: str | None = None,
    ) -> str:
        user = format_generate_patch_user(
            pr_title=pr_title,
            pr_body=pr_body,
            pr_diff=pr_diff,
            comments=comments,
            repo_context=repo_context,
            validation_commands=validation_commands,
            prior_failure=prior_failure,
        )
        text = self._call(system=GENERATE_PATCH_SYSTEM, user=user, max_tokens=16000)
        return validate_diff_response(text)

    def _call(self, *, system: str, user: str, max_tokens: int) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
        parts: list[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "".join(parts).strip()
