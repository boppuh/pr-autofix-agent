"""Config constants, defaults, and `.pr-agent.yml` loader."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError

from .models import SafetyLimits, TargetRepoConfig, ValidateCommand, WorkflowInputs

# --- Hard-coded defaults (overridable via .pr-agent.yml) ---

MAX_ROUNDS = 5
MAX_COMMENTS_PER_ROUND = 20
MAX_PATCH_LINES = 800
MAX_FILES_TOUCHED = 15
MAX_RUNTIME_MINUTES = 20

BUGBOT_AUTHOR_MATCHES: list[str] = [
    "cursor",
    "bugbot",
    "cursor-bugbot",
]

DEFAULT_VALIDATION_COMMANDS: list[str] = [
    "npm test",
    "npm run lint",
    "npm run typecheck",
]

DEFAULT_PROTECTED_PATHS: list[str] = [
    ".github/workflows/",
    "infra/",
    "migrations/",
    "secrets/",
]

DEFAULT_CONFIG_FILENAME = ".pr-agent.yml"


class ConfigError(Exception):
    pass


def load_target_repo_config(repo_root: Path) -> TargetRepoConfig:
    """Load `.pr-agent.yml` from the target repo, applying spec defaults.

    The file is optional: if missing, defaults are used (npm validation commands,
    standard protected paths, BUGBOT_AUTHOR_MATCHES). All fields under
    `validation:` and `safety:` may be overridden.
    """
    path = repo_root / DEFAULT_CONFIG_FILENAME
    raw: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{DEFAULT_CONFIG_FILENAME} must be a YAML mapping.")
        raw = cast(dict[str, Any], loaded)

    validation_section: dict[str, Any] = raw.get("validation") or {}
    commands_raw = validation_section.get("commands") or DEFAULT_VALIDATION_COMMANDS
    commands = [_normalize_command(c, idx) for idx, c in enumerate(commands_raw)]

    safety_raw: dict[str, Any] = raw.get("safety") or {}
    safety = SafetyLimits(
        max_rounds=safety_raw.get("max_rounds", MAX_ROUNDS),
        max_comments_per_round=safety_raw.get("max_comments_per_round", MAX_COMMENTS_PER_ROUND),
        max_patch_lines=safety_raw.get("max_patch_lines", MAX_PATCH_LINES),
        max_files_touched=safety_raw.get("max_files_touched", MAX_FILES_TOUCHED),
        max_runtime_minutes=safety_raw.get("max_runtime_minutes", MAX_RUNTIME_MINUTES),
    )

    protected: list[str] = list(raw.get("protected_paths") or DEFAULT_PROTECTED_PATHS)
    bugbot_logins: list[str] = list(raw.get("bugbot_logins") or BUGBOT_AUTHOR_MATCHES)

    try:
        return TargetRepoConfig.model_validate(
            {
                "validate": commands,
                "protected_paths": protected,
                "safety": safety,
                "bugbot_logins": bugbot_logins,
            }
        )
    except ValidationError as e:
        raise ConfigError(f"Invalid {DEFAULT_CONFIG_FILENAME}: {e}") from e


def _normalize_command(c: object, idx: int) -> ValidateCommand:
    if isinstance(c, str):
        return ValidateCommand(name=f"step{idx + 1}", run=c)
    if isinstance(c, dict) and "run" in c:
        return ValidateCommand(name=str(c.get("name") or f"step{idx + 1}"), run=str(c["run"]))
    raise ConfigError(f"validation.commands[{idx}] must be a string or {{name, run}} mapping.")


def load_workflow_inputs(argv: list[str] | None = None) -> WorkflowInputs:
    parser = argparse.ArgumentParser(prog="pr-autofix-agent")
    parser.add_argument("--pr", dest="pr_number", type=int, required=False)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo", dest="repo_full_name", type=str, default=None)
    parser.add_argument("--label", dest="needs_human_label", default=None)
    parser.add_argument("--confidence", dest="confidence_threshold", type=float, default=None)
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    pr_number = args.pr_number or _int_env("PR_NUMBER")
    if pr_number is None:
        raise ConfigError("PR number required (--pr or PR_NUMBER env).")

    repo = args.repo_full_name or os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        raise ConfigError("Repo full name required (--repo or GITHUB_REPOSITORY env).")

    return WorkflowInputs(
        pr_number=pr_number,
        max_rounds=args.max_rounds or _int_env("MAX_ROUNDS") or MAX_ROUNDS,
        model=args.model or os.environ.get("AGENT_MODEL") or "claude-sonnet-4-6",
        dry_run=args.dry_run or _bool_env("DRY_RUN"),
        repo_full_name=repo,
        needs_human_label=args.needs_human_label
        or os.environ.get("NEEDS_HUMAN_LABEL")
        or "needs-human",
        confidence_threshold=args.confidence_threshold
        or _float_env("CONFIDENCE_THRESHOLD")
        or 0.7,
        log_level=(args.log_level or os.environ.get("LOG_LEVEL") or "INFO").upper(),  # type: ignore[arg-type]
    )


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise ConfigError(f"Required environment variable {name} is not set.")
    return val


def _int_env(name: str) -> int | None:
    v = os.environ.get(name)
    return int(v) if v else None


def _float_env(name: str) -> float | None:
    v = os.environ.get(name)
    return float(v) if v else None


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}
