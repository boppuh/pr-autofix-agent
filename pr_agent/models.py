"""Domain models — pure stdlib dataclasses, no pydantic.

Each call site that used to lean on pydantic validation gets an explicit
`from_*` factory or normaliser here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, get_args

# --- Review thread ---------------------------------------------------------


@dataclass
class ReviewComment:
    id: str
    author: str
    body: str
    path: str | None
    line: int | None
    diff_hunk: str | None
    created_at: str


@dataclass
class ReviewThread:
    id: str
    is_resolved: bool
    comments: list[ReviewComment]

    @property
    def root_comment(self) -> ReviewComment:
        return self.comments[0]

    @property
    def body_text(self) -> str:
        return "\n\n".join(c.body for c in self.comments)

    @property
    def path(self) -> str | None:
        """Convenience: the path of the root comment."""
        return self.comments[0].path if self.comments else None

    @property
    def line(self) -> int | None:
        """Convenience: the line of the root comment."""
        return self.comments[0].line if self.comments else None


# --- Classification --------------------------------------------------------

ClassificationCategory = Literal["AUTO_FIX", "NEEDS_HUMAN", "IGNORE"]
_CATEGORY_VALUES: tuple[str, ...] = get_args(ClassificationCategory)


@dataclass
class Classification:
    thread_id: str
    category: ClassificationCategory
    reason: str
    confidence: float

    def __post_init__(self) -> None:
        if self.category not in _CATEGORY_VALUES:
            raise ValueError(
                f"Classification.category must be one of {_CATEGORY_VALUES}, got {self.category!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Classification.confidence must be in [0,1], got {self.confidence}"
            )

    @classmethod
    def from_json(cls, data: dict[str, Any], thread_id: str) -> Classification:
        """Build a Classification from a parsed JSON dict.

        Any malformed input (wrong category, non-numeric confidence,
        ``"confidence": null``, a list / dict where a scalar was expected)
        raises ``ValueError`` so callers can use a single except clause for
        all LLM-output validation failures.
        """
        category = str(data.get("category") or data.get("label") or "").upper()
        # Backwards-compat with the legacy two-bucket schema.
        if category == "AUTO_FIXABLE":
            category = "AUTO_FIX"
        elif category == "HUMAN_REQUIRED":
            category = "NEEDS_HUMAN"
        if category not in _CATEGORY_VALUES:
            raise ValueError(f"Unknown classification category {category!r}")
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"confidence must be numeric, got {data.get('confidence')!r}"
            ) from e
        return cls(
            thread_id=thread_id,
            category=category,  # type: ignore[arg-type]
            reason=str(data.get("reason", "")),
            confidence=confidence,
        )


# --- Patches ---------------------------------------------------------------


@dataclass
class PatchFile:
    path: str
    new_content: str
    rationale: str


@dataclass
class Patch:
    thread_id: str
    files: list[PatchFile]
    summary: str

    def touched_paths(self) -> list[str]:
        return [f.path for f in self.files]

    @classmethod
    def from_json(cls, data: dict[str, Any], thread_id: str) -> Patch:
        """Build a Patch from a parsed JSON dict.

        Any malformed input raises ``ValueError`` so callers can use a single
        except clause for all LLM-output validation failures.
        """
        files_raw = data.get("files") or []
        files: list[PatchFile] = []
        for f in files_raw:
            if not isinstance(f, dict):
                raise ValueError(
                    f"patch file entry must be a mapping, got {type(f).__name__}"
                )
            try:
                path = str(f["path"])
                new_content = str(f["new_content"])
            except KeyError as e:
                raise ValueError(f"patch file missing required field: {e}") from e
            files.append(
                PatchFile(path=path, new_content=new_content, rationale=str(f.get("rationale", "")))
            )
        return cls(
            thread_id=thread_id,
            files=files,
            summary=str(data.get("summary", "autofix")),
        )


# --- Validation ------------------------------------------------------------


@dataclass
class ValidateCommand:
    name: str
    run: str


@dataclass
class CommandResult:
    """Result of running a single validation command."""

    name: str
    ok: bool
    exit_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_s: float = 0.0


@dataclass
class ValidationResult:
    """Aggregate result of running all configured validation commands.

    The Phase 10 spec wraps individual command results in this top-level
    object with an explicit ``success`` flag. ``success`` is False if any
    command failed.
    """

    success: bool
    command_results: list[CommandResult] = field(default_factory=list)

    @property
    def first_failure(self) -> CommandResult | None:
        return next((c for c in self.command_results if not c.ok), None)


# --- Round / report --------------------------------------------------------


@dataclass
class HandledThread:
    """A thread the agent successfully fixed in a given round.

    Used by the end-of-run summary comment to render the 'Handled:' section.
    """

    thread_id: str
    location: str  # e.g. "src/foo.py:42" or "(no path)"
    summary: str  # the LLM's per-patch summary line


@dataclass
class EscalatedThread:
    """A thread routed to NEEDS_HUMAN by triage in a given round.

    Used by the end-of-run summary comment to render the 'Escalated:' section.
    Distinct from IGNORE / dedupe / LLM-error skips, which are not surfaced.
    """

    thread_id: str
    location: str
    reason: str  # rule keyword or LLM reason


@dataclass
class RoundResult:
    round_no: int
    fixed_thread_ids: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    handled: list[HandledThread] = field(default_factory=list)
    escalated_to_human: list[EscalatedThread] = field(default_factory=list)
    validation: ValidationResult = field(
        default_factory=lambda: ValidationResult(success=True)
    )
    commit_sha: str | None = None
    error: str | None = None

    @property
    def validation_ok(self) -> bool:
        return self.validation.success


class EscalationReason(StrEnum):
    MAX_ROUNDS = "max_rounds"
    REPEATED_VALIDATION_FAILURE = "repeated_validation_failure"
    VALIDATION_FAILED = "validation_failed"
    NO_FIXABLE_THREADS = "no_fixable_threads"
    UNSAFE_PATCH = "unsafe_patch"
    RUNTIME_BUDGET_EXHAUSTED = "runtime_budget_exhausted"
    MISSING_LLM_CREDENTIAL = "missing_llm_credential"
    NO_PROGRESS = "no_progress"


@dataclass(kw_only=True)
class AgentRunReport:
    """Persisted state for a single agent run.

    The first four fields are the spec surface for `.pr-agent-state.json`:
    `round`, `processed_comment_hashes`, `thread_attempt_counts`,
    `previous_unresolved_count`. The rest is the audit trail kept for
    debugging and the escalation comment.

    `kw_only=True` lets us put the spec fields (with defaults) before the
    required `pr_number` field; `dataclasses.asdict` then serialises them
    in that order at the top of the JSON file.
    """

    round: int = 0
    processed_comment_hashes: list[str] = field(default_factory=list)
    thread_attempt_counts: dict[str, int] = field(default_factory=dict)
    previous_unresolved_count: int | None = None
    pr_number: int
    escalated: bool = False
    escalation_reason: EscalationReason | None = None
    final_unresolved_thread_ids: list[str] = field(default_factory=list)
    rounds: list[RoundResult] = field(default_factory=list)


# --- PR + checks -----------------------------------------------------------


@dataclass
class PullRequest:
    id: str
    number: int
    title: str
    body: str
    head_ref_name: str
    head_ref_oid: str
    base_ref_name: str
    threads: list[ReviewThread] = field(default_factory=list)


@dataclass
class CheckRun:
    name: str
    status: str  # queued | in_progress | completed
    conclusion: str | None = None  # success | failure | neutral | cancelled | timed_out | ...


# --- Config ----------------------------------------------------------------


@dataclass
class SafetyLimits:
    max_rounds: int = 5
    max_comments_per_round: int = 20
    max_patch_lines: int = 800
    max_files_touched: int = 15
    max_runtime_minutes: int = 20
    exit_on_validation_failure: bool = True
    post_per_thread_replies: bool = True


@dataclass
class TargetRepoConfig:
    """Schema for the target repo's `.pr-agent.yml`."""

    validate_: list[ValidateCommand] = field(default_factory=list)
    protected_paths: list[str] = field(default_factory=list)
    safety: SafetyLimits = field(default_factory=SafetyLimits)
    bugbot_logins: list[str] = field(
        default_factory=lambda: ["cursor", "bugbot", "cursor-bugbot"]
    )


# --- Workflow inputs -------------------------------------------------------

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
_LOG_LEVELS: tuple[str, ...] = get_args(LogLevel)
ProviderName = Literal["anthropic", "openai"]
_PROVIDERS: tuple[str, ...] = get_args(ProviderName)


@dataclass
class WorkflowInputs:
    pr_number: int
    repo_full_name: str
    max_rounds: int = 5
    provider: ProviderName = "anthropic"
    model: str | None = None
    dry_run: bool = False
    needs_human_label: str = "needs-human"
    confidence_threshold: float = 0.7
    log_level: LogLevel = "INFO"

    def __post_init__(self) -> None:
        if self.provider not in _PROVIDERS:
            raise ValueError(f"provider must be one of {_PROVIDERS}, got {self.provider!r}")
        if self.log_level not in _LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {_LOG_LEVELS}, got {self.log_level!r}"
            )
