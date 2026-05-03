from __future__ import annotations

import fnmatch
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ._paths import matches_any_protected
from .models import Patch

log = logging.getLogger(__name__)

FORBIDDEN_GLOBS = [
    ".github/workflows/*",
    ".pr-agent.yml",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Cargo.lock",
    "Gemfile.lock",
    "go.sum",
]


class UnsafePatchError(Exception):
    pass


@dataclass
class PatchSafetyReport:
    ok: bool
    reasons: list[str]


class Patcher:
    def __init__(
        self,
        repo_root: Path,
        protected_paths: list[str],
        max_files_touched: int,
        max_patch_lines: int,
    ):
        self._root = repo_root.resolve()
        self._protected = protected_paths
        self._max_files = max_files_touched
        self._max_patch_lines = max_patch_lines

    def check_safe(self, patch: Patch) -> PatchSafetyReport:
        reasons: list[str] = []
        if not patch.files:
            reasons.append("empty patch")
        if len(patch.files) > self._max_files:
            reasons.append(f"too many files ({len(patch.files)} > {self._max_files})")
        total_lines = sum(f.new_content.count("\n") + 1 for f in patch.files)
        if total_lines > self._max_patch_lines:
            reasons.append(f"patch too large ({total_lines} > {self._max_patch_lines} lines)")
        for f in patch.files:
            if Path(f.path).is_absolute() or ".." in Path(f.path).parts:
                reasons.append(f"non-relative path: {f.path}")
                continue
            target = (self._root / f.path).resolve()
            try:
                target.relative_to(self._root)
            except ValueError:
                reasons.append(f"path escapes repo root: {f.path}")
                continue
            if any(fnmatch.fnmatch(f.path, pat) for pat in FORBIDDEN_GLOBS):
                reasons.append(f"forbidden path: {f.path}")
            if matches_any_protected(f.path, self._protected):
                reasons.append(f"protected path: {f.path}")
        return PatchSafetyReport(ok=not reasons, reasons=reasons)

    def apply(self, patch: Patch) -> list[Path]:
        report = self.check_safe(patch)
        if not report.ok:
            raise UnsafePatchError("; ".join(report.reasons))
        written: list[Path] = []
        for f in patch.files:
            target = self._root / f.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f.new_content)
            written.append(target)
        log.info("Applied patch %s: %d files", patch.thread_id, len(written))
        return written

    def stage_and_commit(self, patch: Patch, paths: list[Path], author_email: str) -> str:
        rel = [str(p.relative_to(self._root)) for p in paths]
        self._git("add", "--", *rel)
        msg = f"fix(autofix): {patch.summary} [thread {patch.thread_id}]"
        env_args = [
            "-c",
            f"user.email={author_email}",
            "-c",
            "user.name=pr-autofix-agent",
        ]
        self._git(*env_args, "commit", "-m", msg)
        return self._git("rev-parse", "HEAD").strip()

    def push(self, branch: str) -> None:
        self._git("push", "origin", f"HEAD:{branch}")

    def revert_uncommitted(self, paths: list[Path]) -> None:
        rel = [str(p.relative_to(self._root)) for p in paths]
        if not rel:
            return
        self._git("checkout", "--", *rel)

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self._root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )
        return result.stdout


