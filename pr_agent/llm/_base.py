"""Provider-agnostic base: shared prompts, JSON parsing, and the Protocol."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, runtime_checkable

from ..models import Classification, Patch, ReviewThread

log = logging.getLogger(__name__)


CLASSIFY_SYSTEM = """You triage Cursor Bugbot PR review comments into one of three buckets.

AUTO_FIX — small, mechanical, scoped change that an LLM can produce safely.
  Examples: missing null check, unused import, off-by-one, wrong type, typo
  in identifier, missing await, dead code removal, fixing a regex, adding
  a guard.

NEEDS_HUMAN — the comment requires product/architectural judgement, multi-file
  design, contract changes, schema migrations, security model changes, or
  anything ambiguous.
  Examples: "consider refactoring this module", "the abstraction here is leaky",
  "this should be split", "API contract change".

IGNORE — the comment is not actionable as a code change.
  Examples: "LGTM", "thanks for the fix", praise, status updates, questions
  the author can answer in chat, `nit:` items the author has already
  addressed, conversation about non-code topics.

Output strictly valid JSON: {"category": "AUTO_FIX"|"NEEDS_HUMAN"|"IGNORE",
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


GENERATE_PATCH_SYSTEM = """You are an autonomous PR repair agent.

Fix only the Cursor Bugbot comments listed below.

Rules:
- Return only a unified git diff.
- Do not include prose.
- Do not modify unrelated files.
- Do not exceed the requested scope.
- Preserve existing behavior unless the comment explicitly identifies a bug.
- Add or update tests when needed.
- If the fix is unsafe, ambiguous, or too large, return exactly:
  ESCALATE: <reason>
"""


class LLMResponseError(Exception):
    """Raised when the provider returns text that doesn't decode as expected."""


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

    def generate_patch(
        self,
        *,
        pr_title: str,
        pr_body: str,
        pr_diff: str,
        comments: list[ReviewThread],
        repo_context: str,
        validation_commands: list[str],
    ) -> str:
        """Phase 8 batched entry point.

        Returns either a unified git diff (no markdown fences, no prose) or
        the literal string ``"ESCALATE: <reason>"``.
        """
        ...


# --- Shared helpers reused by every concrete provider ---------------------


def parse_classification(raw_text: str, thread_id: str) -> Classification:
    """Decode the classifier's JSON output, falling back to NEEDS_HUMAN on
    any parse / validation failure (truncated output, schema drift, etc.)."""
    try:
        data = extract_json(raw_text)
        return Classification.from_json(data, thread_id=thread_id)
    except (LLMResponseError, ValueError) as e:
        log.warning("Classifier returned invalid JSON: %s; raw=%r", e, raw_text[:300])
        return Classification(
            thread_id=thread_id,
            category="NEEDS_HUMAN",
            reason="classifier output failed validation",
            confidence=0.0,
        )


def parse_patch(raw_text: str, thread_id: str) -> Patch:
    data = extract_json(raw_text)
    return Patch.from_json(data, thread_id=thread_id)


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


def format_generate_patch_user(
    *,
    pr_title: str,
    pr_body: str,
    pr_diff: str,
    comments: list[ReviewThread],
    repo_context: str,
    validation_commands: list[str],
) -> str:
    """Render the Phase 8 batched user message.

    Field names mirror the spec template (`{{pr_title}}`, etc.).
    """
    rendered_comments = "\n".join(
        _format_thread_for_batch(i, t) for i, t in enumerate(comments, start=1)
    )
    rendered_validators = (
        "\n".join(f"  - {cmd}" for cmd in validation_commands) if validation_commands else "  (none)"
    )
    return (
        f"PR title:\n{pr_title}\n\n"
        f"PR body:\n{pr_body or '(none)'}\n\n"
        f"Current PR diff:\n```diff\n{pr_diff}\n```\n\n"
        f"Repo context:\n{repo_context}\n\n"
        f"Unresolved Bugbot comments:\n{rendered_comments}\n\n"
        f"Validation commands:\n{rendered_validators}\n"
    )


def _format_thread_for_batch(idx: int, thread: ReviewThread) -> str:
    parts = [
        f"[{idx}] thread {thread.id} @ {thread.path or '(none)'}:"
        f"{thread.line if thread.line is not None else '?'}",
        thread.body_text,
    ]
    hunk = thread.comments[0].diff_hunk if thread.comments else None
    if hunk:
        parts += ["Diff hunk:", hunk]
    return "\n".join(parts)


_DIFF_HEADER_RE = re.compile(r"^(diff --git |--- |\+\+\+ |@@ )", re.MULTILINE)
_FENCE_TOKEN = "```"


def validate_diff_response(text: str) -> str:
    """Run the Phase 8 syntactic checks and return the cleaned text.

    Raises ``LLMResponseError`` on any violation. Pass-through for the
    ``ESCALATE: <reason>`` form so the caller can recognise it.
    """
    stripped = text.strip()
    if stripped.startswith("ESCALATE:"):
        return stripped
    if _FENCE_TOKEN in stripped:
        raise LLMResponseError("response contained a markdown fence")
    if not _DIFF_HEADER_RE.search(stripped):
        raise LLMResponseError("response is not a unified diff")
    return stripped


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
