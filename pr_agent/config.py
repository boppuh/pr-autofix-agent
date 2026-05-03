from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import TargetRepoConfig, WorkflowInputs

DEFAULT_CONFIG_FILENAME = ".pr-autofix.yml"


class ConfigError(Exception):
    pass


def load_target_repo_config(repo_root: Path) -> TargetRepoConfig:
    path = repo_root / DEFAULT_CONFIG_FILENAME
    if not path.exists():
        raise ConfigError(
            f"Missing {DEFAULT_CONFIG_FILENAME} at repo root ({repo_root}). "
            "Create one with at least a `validate:` list."
        )
    raw = yaml.safe_load(path.read_text()) or {}
    try:
        cfg = TargetRepoConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"Invalid {DEFAULT_CONFIG_FILENAME}: {e}") from e
    if not cfg.validate_:
        raise ConfigError(
            f"{DEFAULT_CONFIG_FILENAME} must declare at least one `validate:` command."
        )
    return cfg


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
        max_rounds=args.max_rounds or _int_env("MAX_ROUNDS") or 5,
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
