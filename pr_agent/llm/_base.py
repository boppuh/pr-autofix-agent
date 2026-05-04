"""Provider-agnostic base: shared prompts, JSON parsing, and the Protocol."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from ..models import (
    Classification,
    ClassificationLabel,
    Patch,
    PatchFile,
    ReviewThread,
)

log = logging.getLogger(__name__)


CLASSIFY_SYSTEM = """You triage Cursor Bugbot PR review comments.

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

PATCH_SYSTEM = """You generate minimal patches for Cursor Bugbot PR review comments.

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


class LLMResponseError(Exception):
    """Raised when the provider returns text that doesn't decode to JSON."""


@runtime_checkable
class LLMProvider(Protocol):
    """Every concrete provider exposes the same triage/patch surface."""

    def classify(self, thread: ReviewThread, file_excerpt: str | None) -> Classification: ...

    def propose_patch(
        self,
        thread: ReviewThread,
        file_contents: dict[str, str],
        max_files: int,
        prior_failure: str | None = None,
        pr_title: str | None = None,
        pr_body_excerpt: str | None = None,
        pr_diff_excerpt: str | None = None,
    ) -> Patch: ...


# --- Shared helpers reused by every concrete provider ---------------------


def parse_classification(raw_text: str) -> Classification:
    try:
        data = extract_json(raw_text)
        return Classification.model_validate(data)
    except (LLMResponseError, ValidationError) as e:
        log.warning("Classifier returned invalid JSON: %s; raw=%r", e, raw_text[:300])
        return Classification(
            label=ClassificationLabel.HUMAN_REQUIRED,
            confidence=0.0,
            reason="classifier output failed validation",
        )


def parse_patch(raw_text: str, thread_id: str) -> Patch:
    data = extract_json(raw_text)
    files_raw = data.get("files") or []
    files = [PatchFile.model_validate(f) for f in files_raw]
    return Patch(
        thread_id=thread_id,
        files=files,
        summary=data.get("summary", "autofix"),
    )


def format_classify_user(thread: ReviewThread, file_excerpt: str | None) -> str:
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


def format_patch_user(
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
        parts += ["", "PR diff (truncated):", "```diff", pr_diff_excerpt, "```"]
    if parts:
        parts.append("")
    parts += [
        f"Thread path: {thread.path or '(none)'}",
        f"Thread line: {thread.line if thread.line is not None else '(none)'}",
        f"File budget: at most {max_files} file(s).",
        "",
        "Comment thread:",
        thread.body_text,
    ]
    if thread.comments and thread.comments[0].diff_hunk:
        parts += [
            "",
            "Diff hunk for the thread:",
            "```diff",
            thread.comments[0].diff_hunk,
            "```",
        ]
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


def extract_json(text: str) -> dict[str, Any]:
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
