from __future__ import annotations

import json
import logging
import re
from typing import Any

from anthropic import Anthropic
from pydantic import ValidationError

from .models import Classification, ClassificationLabel, Patch, PatchFile, ReviewThread

log = logging.getLogger(__name__)


class LLMResponseError(Exception):
    """LLM returned text that did not decode to a JSON object."""

_CLASSIFY_SYSTEM = """You triage Cursor Bugbot PR review comments.

Decide whether a comment is AUTO-FIXABLE (small, mechanical, scoped) or HUMAN-REQUIRED
(architectural, ambiguous, multi-file design discussion, contract change, security policy).

Auto-fixable examples: missing null check, unused import, off-by-one, wrong type, typo
in identifier, missing await, dead code removal, fixing a regex, adding a guard.

Human-required examples: "consider refactoring this module", "the abstraction here is
leaky", API contract changes, schema migrations, security model changes, anything that
requires product judgement or touches >2 files conceptually.

Output strictly valid JSON: {"label": "auto_fixable"|"human_required",
"confidence": 0.0-1.0, "reason": "<one sentence>"}.
"""

_PATCH_SYSTEM = """You generate minimal patches for Cursor Bugbot PR review comments.

Hard rules:
- Touch ONLY files referenced by the thread or directly required by the fix.
- Never exceed the file budget.
- Never modify CI workflows, lockfiles, or .pr-agent.yml.
- Output ENTIRE new file contents (not diffs) for each modified file.
- Do NOT add unrelated cleanup, comments, or refactors.
- If the comment cannot be safely auto-fixed, return an empty files array.

Output strictly valid JSON:
{"summary": "<one-line>", "files": [{"path": "<repo-relative>",
"new_content": "<full file>", "rationale": "<why>"}]}
"""


class LLMClient:
    def __init__(self, model: str, api_key: str | None = None):
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self._model = model

    def classify(self, thread: ReviewThread, file_excerpt: str | None) -> Classification:
        user = _format_classify_user(thread, file_excerpt)
        text = self._call(
            system=_CLASSIFY_SYSTEM,
            user=user,
            max_tokens=400,
        )
        try:
            data = _extract_json(text)
            return Classification.model_validate(data)
        except (LLMResponseError, ValidationError) as e:
            log.warning("Classifier returned invalid JSON: %s; raw=%r", e, text[:300])
            return Classification(
                label=ClassificationLabel.HUMAN_REQUIRED,
                confidence=0.0,
                reason="classifier output failed validation",
            )

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
        user = _format_patch_user(
            thread,
            file_contents,
            max_files,
            prior_failure,
            pr_title=pr_title,
            pr_body_excerpt=pr_body_excerpt,
            pr_diff_excerpt=pr_diff_excerpt,
        )
        # Patches must include full file contents, so give the model headroom.
        # 4000 was hitting truncation on real-sized source files.
        text = self._call(system=_PATCH_SYSTEM, user=user, max_tokens=16000)
        data = _extract_json(text)
        files_raw = data.get("files") or []
        files = [PatchFile.model_validate(f) for f in files_raw]
        return Patch(
            thread_id=thread.id,
            files=files,
            summary=data.get("summary", "autofix"),
        )

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


def _format_classify_user(thread: ReviewThread, file_excerpt: str | None) -> str:
    lines = [
        f"Path: {thread.path or '(none)'}",
        f"Line: {thread.line if thread.line is not None else '(none)'}",
        "",
        "Comment thread:",
        thread.body_text,
    ]
    if file_excerpt:
        lines += ["", "File excerpt around the comment:", "```", file_excerpt, "```"]
    return "\n".join(lines)


def _format_patch_user(
    thread: ReviewThread,
    file_contents: dict[str, str],
    max_files: int,
    prior_failure: str | None,
    *,
    pr_title: str | None = None,
    pr_body_excerpt: str | None = None,
    pr_diff_excerpt: str | None = None,
) -> str:
    parts: list[str] = []
    if pr_title:
        parts += [f"PR title: {pr_title}"]
    if pr_body_excerpt:
        parts += ["", "PR description:", pr_body_excerpt]
    if pr_diff_excerpt:
        parts += [
            "",
            "PR diff (truncated):",
            "```diff",
            pr_diff_excerpt,
            "```",
        ]
    if parts:
        parts.append("")  # blank line separator only when prior context exists
    parts += [
        f"Thread path: {thread.path or '(none)'}",
        f"Thread line: {thread.line if thread.line is not None else '(none)'}",
        f"File budget: at most {max_files} file(s).",
        "",
        "Comment thread:",
        thread.body_text,
    ]
    if thread.comments and thread.comments[0].diff_hunk:
        parts += ["", "Diff hunk for the thread:", "```diff", thread.comments[0].diff_hunk, "```"]
    if prior_failure:
        parts += [
            "",
            "Validation failure from a previous attempt (do not repeat it):",
            "```",
            prior_failure[-2000:],
            "```",
        ]
    parts += ["", "Current file contents:"]
    for path, content in file_contents.items():
        parts += [f"--- {path} ---", "```", content, "```"]
    return "\n".join(parts)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    candidates = _FENCE_RE.findall(text) or [text]
    for c in candidates:
        c = c.strip()
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise LLMResponseError(f"LLM did not return valid JSON object: {text[:200]!r}")
