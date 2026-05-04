"""Run configured validation commands.

Phase 10 spec exposes a free function:

    run_validation(commands: list[str]) -> ValidationResult

returning an aggregate ``ValidationResult{success, command_results}``.
The :class:`Validator` class wraps the same primitives and accepts named
:class:`ValidateCommand` objects from the YAML config so per-command
``name`` survives the config round-trip.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

from .models import CommandResult, ValidateCommand, ValidationResult

log = logging.getLogger(__name__)

_TAIL_BYTES = 4000
_TIMEOUT_SECONDS = 15 * 60


def run_validation(
    commands: list[str],
    *,
    repo_root: Path | None = None,
) -> ValidationResult:
    """Run each shell command in order; short-circuit on first failure.

    Returns a :class:`ValidationResult` aggregate. ``success`` is False if
    any command returned a non-zero exit code or timed out.
    """
    cwd = repo_root or Path.cwd()
    results: list[CommandResult] = []
    for cmd in commands:
        result = _run_one(name=cmd, command=cmd, cwd=cwd)
        results.append(result)
        if not result.ok:
            log.warning("Validator %s failed (exit %d)", cmd, result.exit_code)
            break
    success = all(r.ok for r in results)
    return ValidationResult(success=success, command_results=results)


def _run_one(*, name: str, command: str, cwd: Path) -> CommandResult:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            executable=_shell_path(),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
        duration = time.monotonic() - start
        return CommandResult(
            name=name,
            ok=proc.returncode == 0,
            exit_code=proc.returncode,
            stdout_tail=_tail(proc.stdout),
            stderr_tail=_tail(proc.stderr),
            duration_s=duration,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            name=name,
            ok=False,
            exit_code=-1,
            stderr_tail=f"timed out after {_TIMEOUT_SECONDS // 60} minutes",
            duration_s=time.monotonic() - start,
        )


class Validator:
    """Stateful wrapper that holds repo_root + named commands.

    Internally delegates command execution to :func:`run_validation` so the
    free-function path and the class path produce identical results.
    """

    def __init__(self, repo_root: Path, commands: list[ValidateCommand]):
        self._root = repo_root
        self._commands = commands

    def run(self) -> ValidationResult:
        results: list[CommandResult] = []
        for cmd in self._commands:
            result = _run_one(name=cmd.name, command=cmd.run, cwd=self._root)
            results.append(result)
            if not result.ok:
                log.warning("Validator %s failed (exit %d)", cmd.name, result.exit_code)
                break
        success = all(r.ok for r in results)
        return ValidationResult(success=success, command_results=results)

    @staticmethod
    def format_failure(result: ValidationResult) -> str:
        """Format the first failing CommandResult for the LLM prompt / PR comment.

        Returns the empty string if the aggregate was successful.
        """
        failure = result.first_failure
        if failure is None:
            return ""
        return (
            f"[{failure.name}] exit={failure.exit_code}\n"
            f"--- stderr ---\n{failure.stderr_tail}\n"
            f"--- stdout ---\n{failure.stdout_tail}"
        )


def _tail(s: str) -> str:
    if len(s) <= _TAIL_BYTES:
        return s
    return "...\n" + s[-_TAIL_BYTES:]


def _shell_path() -> str:
    return os.environ.get("SHELL", "/bin/sh")
