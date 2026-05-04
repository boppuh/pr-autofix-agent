"""Patch the working tree.

The Phase 9 spec exposes four module-level free functions:

- :func:`extract_touched_files` — every path the diff names (union of
  ``diff --git`` and ``+++ b/`` headers).
- :func:`count_patch_lines` — payload +/- line count.
- :func:`violates_protected_paths` — predicate against a protected-path list.
- :func:`apply_unified_diff` — full apply pipeline (guards, ``git apply
  --check``, ``git apply``, post-apply ``git diff --check``, revert on
  failure). Returns ``True`` on success, ``False`` on any rejection.

The :class:`Patcher` class wraps the same primitives and translates
rejections into typed :class:`UnsafePatchError` exceptions so the run
loop can log specific reasons. Both APIs work; the class also enforces
the project-specific :data:`FORBIDDEN_GLOBS` allowlist (CI workflows,
lockfiles, ``.pr-agent.yml``) which the spec's free functions know
nothing about.
"""

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


# --- Free functions (Phase 9 spec) ---------------------------------------


_DIFF_GIT_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)
_PLUS_FILE_RE = re.compile(r"^\+\+\+ b/(\S+)$", re.MULTILINE)
# Matches the +++ and --- file-header lines in unified diff format.
# These are exactly three characters followed by a space and a path.
_HEADER_LINE_RE = re.compile(r"^(?:\+\+\+|---) ")


def extract_touched_files(diff_text: str) -> list[str]:
    """Every repo-relative path the diff names.

    Returns the union of paths from ``diff --git a/X b/Y`` headers and
    ``+++ b/Y`` lines. ``git apply`` decides target files from the
    ``---``/``+++`` headers, not from ``diff --git`` — so a malicious /
    confused diff with mismatched headers (e.g. ``diff --git a/safe.py
    b/safe.py`` but ``+++ b/.github/workflows/ci.yml``) would otherwise
    pass safety checks against ``safe.py`` while ``git apply`` mutates
    the workflow file. Validating the union closes that gap.
    """
    paths: list[str] = []
    seen: set[str] = set()
    for _, b in _DIFF_GIT_RE.findall(diff_text):
        if b and b != "/dev/null" and b not in seen:
            paths.append(b)
            seen.add(b)
    for b in _PLUS_FILE_RE.findall(diff_text):
        if b and b != "/dev/null" and b not in seen:
            paths.append(b)
            seen.add(b)
    return paths


def count_patch_lines(diff_text: str) -> int:
    """Count payload +/- lines, excluding the ``+++`` / ``---`` header rows.

    Header rows are exactly ``+++ <path>`` or ``--- <path>`` (three marker
    characters followed by a space). Payload lines that happen to start with
    ``--`` or ``++`` (e.g. a removed/added line whose content begins with
    dashes) are still counted.
    """
    count = 0
    for line in diff_text.splitlines():
        if _HEADER_LINE_RE.match(line):
            continue
        if line.startswith(("+", "-")):
            count += 1
    return count


def violates_protected_paths(files: list[str], protected_paths: list[str]) -> bool:
    """True iff any of ``files`` matches a ``protected_paths`` entry.

    Uses the shared :func:`pr_agent._paths.matches_any_protected` helper:
    trailing-slash entries are directory prefixes; everything else is fnmatch.
    """
    return any(matches_any_protected(p, protected_paths) for p in files)


def apply_unified_diff(
    diff_text: str,
    *,
    repo_root: Path,
    protected_paths: list[str],
    max_files: int,
    max_patch_lines: int,
) -> bool:
    """Run the full Phase 9 apply pipeline against ``repo_root``.

    Returns ``True`` on success and leaves the working tree mutated.
    Returns ``False`` on any rejection (logs the reason at INFO):

    1. empty diff
    2. diff names no files
    3. ``len(files) > max_files``
    4. :func:`violates_protected_paths`
    5. ``count_patch_lines > max_patch_lines``
    6. ``git apply --check`` fails
    7. ``git apply`` fails
    8. ``git diff --check`` fails after apply (whitespace errors / conflict
       markers) — runs ``git apply --reverse`` to undo, then returns False
    """
    if not diff_text.strip():
        log.info("apply_unified_diff: empty diff")
        return False
    files = extract_touched_files(diff_text)
    if not files:
        log.info("apply_unified_diff: diff does not name any files")
        return False
    if len(files) > max_files:
        log.info("apply_unified_diff: too many files (%d > %d)", len(files), max_files)
        return False
    if violates_protected_paths(files, protected_paths):
        log.info("apply_unified_diff: protected path touched")
        return False
    n_lines = count_patch_lines(diff_text)
    if n_lines > max_patch_lines:
        log.info(
            "apply_unified_diff: patch too large (%d > %d lines)", n_lines, max_patch_lines
        )
        return False

    # Stage diff to a temp file; check + apply + post-apply check.
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as tmp:
        tmp.write(diff_text if diff_text.endswith("\n") else diff_text + "\n")
        tmp_path = tmp.name
    try:
        try:
            _run_git(["apply", "--check", tmp_path], cwd=repo_root)
        except RuntimeError as e:
            log.info("apply_unified_diff: git apply --check failed: %s", e)
            return False
        try:
            _run_git(["apply", tmp_path], cwd=repo_root)
        except RuntimeError as e:
            log.info("apply_unified_diff: git apply failed: %s", e)
            return False

        # Post-apply: git diff --check catches whitespace errors and merge-
        # conflict markers (<<<<<<<, =======, >>>>>>>) that --check at the
        # patch level can't see because they're valid diff hunks.
        try:
            _run_git(["diff", "--check"], cwd=repo_root)
        except RuntimeError as e:
            log.info("apply_unified_diff: git diff --check failed; reverting: %s", e)
            try:
                _run_git(["apply", "--reverse", tmp_path], cwd=repo_root)
            except RuntimeError as revert_err:  # working tree now in unknown state
                log.warning(
                    "apply_unified_diff: revert via git apply --reverse FAILED (%s); "
                    "working tree may be inconsistent",
                    revert_err,
                )
            return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    log.info("apply_unified_diff: applied %d file(s)", len(files))
    return True


def _run_git(args: list[str], *, cwd: Path) -> str:
    """Run ``git <args>`` in ``cwd``. Raise :class:`RuntimeError` on non-zero exit."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
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


# --- Patcher class wrapper -----------------------------------------------


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

        Runs the same guards as :func:`apply_unified_diff` but raises
        :class:`UnsafePatchError` with a specific reason on any rejection
        (rather than returning ``False``). Also enforces the project-specific
        :data:`FORBIDDEN_GLOBS` allowlist on top of ``protected_paths``.
        """
        if not diff_text.strip():
            raise UnsafePatchError("empty diff")
        files = extract_touched_files(diff_text)
        if not files:
            raise UnsafePatchError("diff does not name any files")
        if len(files) > self._max_files:
            raise UnsafePatchError(
                f"too many files ({len(files)} > {self._max_files})"
            )
        for p in files:
            if p.startswith("/") or ".." in Path(p).parts:
                raise UnsafePatchError(f"non-relative path: {p}")
            if any(fnmatch.fnmatch(p, pat) for pat in FORBIDDEN_GLOBS):
                raise UnsafePatchError(f"forbidden path: {p}")
            if matches_any_protected(p, self._protected):
                raise UnsafePatchError(f"protected path: {p}")
        n_lines = count_patch_lines(diff_text)
        if n_lines > self._max_patch_lines:
            raise UnsafePatchError(
                f"patch too large ({n_lines} > {self._max_patch_lines} lines)"
            )

        # All guards passed at the class level. Delegate the actual apply
        # (including the post-apply git diff --check step) to the free
        # function. Translate False into a generic UnsafePatchError; the
        # specific reason is in the logs from apply_unified_diff itself.
        ok = apply_unified_diff(
            diff_text,
            repo_root=self._root,
            protected_paths=self._protected,
            max_files=self._max_files,
            max_patch_lines=self._max_patch_lines,
        )
        if not ok:
            raise UnsafePatchError(
                "git apply or git diff --check failed (see logs for details)"
            )
        log.info(
            "Applied batched diff for threads %s: %d files",
            thread_ids,
            len(files),
        )
        return [self._root / p for p in files]

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
        return _run_git(list(args), cwd=self._root)
