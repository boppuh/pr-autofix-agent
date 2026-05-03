from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class BugbotComment(BaseModel):
    id: str
    author_login: str
    body: str
    path: str | None = None
    line: int | None = None
    created_at: datetime


class ReviewThread(BaseModel):
    id: str
    path: str | None = None
    line: int | None = None
    is_resolved: bool = False
    is_outdated: bool = False
    comments: list[BugbotComment] = Field(default_factory=list)

    @property
    def root_comment(self) -> BugbotComment:
        return self.comments[0]

    @property
    def body_text(self) -> str:
        return "\n\n".join(c.body for c in self.comments)


class ClassificationLabel(StrEnum):
    AUTO_FIXABLE = "auto_fixable"
    HUMAN_REQUIRED = "human_required"


class Classification(BaseModel):
    label: ClassificationLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    skipped_by_rule: str | None = None


class PatchFile(BaseModel):
    path: str
    new_content: str
    rationale: str


class Patch(BaseModel):
    thread_id: str
    files: list[PatchFile]
    summary: str

    def touched_paths(self) -> list[str]:
        return [f.path for f in self.files]


class ValidationResult(BaseModel):
    name: str
    ok: bool
    exit_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_s: float = 0.0


class RoundResult(BaseModel):
    round_no: int
    fixed_thread_ids: list[str] = Field(default_factory=list)
    skipped: list[tuple[str, str]] = Field(default_factory=list)
    validation: list[ValidationResult] = Field(default_factory=list)
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


class AgentRunReport(BaseModel):
    pr_number: int
    rounds: list[RoundResult] = Field(default_factory=list)
    escalated: bool = False
    escalation_reason: EscalationReason | None = None
    final_unresolved_thread_ids: list[str] = Field(default_factory=list)


class ValidateCommand(BaseModel):
    name: str
    run: str


class TargetRepoConfig(BaseModel):
    """Schema for the target repo's `.pr-autofix.yml`."""

    validate_: list[ValidateCommand] = Field(default_factory=list, alias="validate")
    exclude_paths: list[str] = Field(default_factory=list)
    max_files_per_patch: int = 5
    bugbot_logins: list[str] = Field(default_factory=lambda: ["cursor[bot]"])

    model_config = {"populate_by_name": True}


class WorkflowInputs(BaseModel):
    pr_number: int
    max_rounds: int = 5
    model: str = "claude-sonnet-4-6"
    dry_run: bool = False
    repo_full_name: str
    needs_human_label: str = "needs-human"
    confidence_threshold: float = 0.7
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
