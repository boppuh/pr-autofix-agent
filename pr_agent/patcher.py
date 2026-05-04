from __future__ import annotations

import fnmatch
import logging
import re
import subprocess
import tempfile
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

    def apply_diff(self, diff_text: str, thread_ids: list[str]) -> list[Path]:
        """Apply a unified-diff string from the Phase 8 batched LLM call.

        Validates the diff (forbidden globs, protected paths, file/line caps),
        runs ``git apply --check`` to confirm patchability, then ``git apply``
        to apply. Raises :class:`UnsafePatchError` on any guard rejection or
        patch-tool failure.
        """
        if not diff_text.strip():
            raise UnsafePatchError("empty diff")
        paths = _paths_from_diff(diff_text)
        if not paths:
            raise UnsafePatchError("diff does not name any files")
        # File-count cap.
        if len(paths) > self._max_files:
            raise UnsafePatchError(
                f"too many files ({len(paths)} > {self._max_files})"
            )
        # Path-safety checks (forbidden globs + protected paths + traversal).
        for p in paths:
            if p.startswith("/") or ".." in Path(p).parts:
                raise UnsafePatchError(f"non-relative path: {p}")
            if any(fnmatch.fnmatch(p, pat) for pat in FORBIDDEN_GLOBS):
                raise UnsafePatchError(f"forbidden path: {p}")
            if matches_any_protected(p, self._protected):
                raise UnsafePatchError(f"protected path: {p}")
        # Line cap (count added + removed payload lines, ignore headers).
        changed = _count_changed_lines(diff_text)
        if changed > self._max_patch_lines:
            raise UnsafePatchError(
                f"patch too large ({changed} > {self._max_patch_lines} lines)"
            )
        # Stage diff to a temp file and check + apply.
        with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as tmp:
            tmp.write(diff_text if diff_text.endswith("\n") else diff_text + "\n")
            tmp_path = tmp.name
        try:
            try:
                self._git("apply", "--check", tmp_path)
            except RuntimeError as e:
                raise UnsafePatchError(f"git apply --check failed: {e}") from e
            self._git("apply", tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        log.info(
            "Applied batched diff for threads %s: %d files",
            thread_ids,
            len(paths),
        )
        return [self._root / p for p in paths]

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


_DIFF_GIT_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)
_PLUS_FILE_RE = re.compile(r"^\+\+\+ b/(\S+)$", re.MULTILINE)


def _paths_from_diff(diff_text: str) -> list[str]:
    """Extract repo-relative paths touched by a unified diff.

    Prefers the b-side of ``diff --git`` headers; falls back to ``+++ b/``
    lines for diffs that omit the ``diff --git`` preamble.
    """
    paths: list[str] = []
    seen: set[str] = set()
    for _, b in _DIFF_GIT_RE.findall(diff_text):
        if b not in seen:
            paths.append(b)
            seen.add(b)
    if not paths:
        for b in _PLUS_FILE_RE.findall(diff_text):
            if b not in seen:
                paths.append(b)
                seen.add(b)
    return paths


def _count_changed_lines(diff_text: str) -> int:
    """Count payload +/- lines, excluding the +++ / --- header rows."""
    count = 0
    for line in diff_text.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            count += 1
    return count
