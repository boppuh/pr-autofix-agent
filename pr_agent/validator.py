from __future__ import annotations

import logging
import shlex
import subprocess
import time
from pathlib import Path

from .models import ValidateCommand, ValidationResult

log = logging.getLogger(__name__)

_TAIL_BYTES = 4000


class Validator:
    def __init__(self, repo_root: Path, commands: list[ValidateCommand]):
        self._root = repo_root
        self._commands = commands

    def run(self) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        for cmd in self._commands:
            result = self._run_one(cmd)
            results.append(result)
            if not result.ok:
                log.warning("Validator %s failed (exit %d)", cmd.name, result.exit_code)
                break
        return results

    def _run_one(self, cmd: ValidateCommand) -> ValidationResult:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd.run,
                shell=True,
                executable=_shell_path(),
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=15 * 60,
            )
            duration = time.monotonic() - start
            return ValidationResult(
                name=cmd.name,
                ok=proc.returncode == 0,
                exit_code=proc.returncode,
                stdout_tail=_tail(proc.stdout),
                stderr_tail=_tail(proc.stderr),
                duration_s=duration,
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                name=cmd.name,
                ok=False,
                exit_code=-1,
                stderr_tail="timed out after 15 minutes",
                duration_s=time.monotonic() - start,
            )

    @staticmethod
    def format_failure(results: list[ValidationResult]) -> str:
        for r in results:
            if not r.ok:
                return (
                    f"[{r.name}] exit={r.exit_code}\n"
                    f"--- stderr ---\n{r.stderr_tail}\n"
                    f"--- stdout ---\n{r.stdout_tail}"
                )
        return ""


def _tail(s: str) -> str:
    if len(s) <= _TAIL_BYTES:
        return s
    return "...\n" + s[-_TAIL_BYTES:]


def _shell_path() -> str:
    import os

    return os.environ.get("SHELL", "/bin/sh")


# Re-export for convenience.
__all__ = ["Validator", "ValidationResult", "ValidateCommand", "shlex"]
