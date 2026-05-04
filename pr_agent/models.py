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
        category = str(data.get("category") or data.get("label") or "").upper()
        # Backwards-compat with the legacy two-bucket schema.
        if category == "AUTO_FIXABLE":
            category = "AUTO_FIX"
        elif category == "HUMAN_REQUIRED":
            category = "NEEDS_HUMAN"
        if category not in _CATEGORY_VALUES:
            raise ValueError(f"Unknown classification category {category!r}")
        return cls(
            thread_id=thread_id,
            category=category,  # type: ignore[arg-type]
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 0.0)),
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
        files_raw = data.get("files") or []
        files: list[PatchFile] = []
        for f in files_raw:
            if not isinstance(f, dict):
                raise ValueError(
                    f"patch file entry must be a mapping, got {type(f).__name__}"
                )
            files.append(
                PatchFile(
                    path=str(f["path"]),
                    new_content=str(f["new_content"]),
                    rationale=str(f.get("rationale", "")),
                )
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
class ValidationResult:
    name: str
    ok: bool
    exit_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_s: float = 0.0


# --- Round / report --------------------------------------------------------


@dataclass
class RoundResult:
    round_no: int
    fixed_thread_ids: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    validation: list[ValidationResult] = field(default_factory=list)
    commit_sha: str | None = None
    error: str | None = None

    @property
    def validation_ok(self) -> bool:
        return all(v.ok for v in self.validation)


class EscalationReason(StrEnum):
    MAX_ROUNDS = "max_rounds"
    REPEATED_VALIDATION_FAILURE = "repeated_validation_failure"
    NO_FIXABLE_THREADS = "no_fixable_threads"
    UNSAFE_PATCH = "unsafe_patch"
    RUNTIME_BUDGET_EXHAUSTED = "runtime_budget_exhausted"
    MISSING_LLM_CREDENTIAL = "missing_llm_credential"


@dataclass
class AgentRunReport:
    pr_number: int
    rounds: list[RoundResult] = field(default_factory=list)
    escalated: bool = False
    escalation_reason: EscalationReason | None = None
    final_unresolved_thread_ids: list[str] = field(default_factory=list)


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
