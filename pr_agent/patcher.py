"""Patch the working tree.

Module-level free functions:

- :func:`extract_touched_files` — every path the diff names (union of
  ``diff --git`` headers, ``--- a/X`` lines, and ``+++ b/Y`` lines).
- :func:`count_patch_lines` — payload +/- line count.
- :func:`violates_protected_paths` — predicate against a protected-path list.
- :func:`apply_unified_diff` — full Phase 9 apply pipeline: runs the
  safety guards (file count, protected paths, line limits) then
  delegates the git work to :func:`apply_diff_to_repo`. Returns
  ``True`` on success, ``False`` on any rejection.
- :func:`apply_diff_to_repo` — git-only helper: ``git apply --check``,
  ``git apply``, post-apply ``git diff --check``, revert on failure.
  Callers must run their own guards. Used by both
  :func:`apply_unified_diff` and :meth:`Patcher.apply_diff` so the
  temp-file handling and revert ordering live in one place.

The :class:`Patcher` class wraps the same primitives and translates
rejections into typed :class:`UnsafePatchError` exceptions so the run
loop can log specific reasons. Both APIs work; the class also enforces
the project-specific :data:`FORBIDDEN_GLOBS` allowlist (CI workflows,
lockfiles, ``.pr-agent.yml``) which the spec's free functions know
nothing about.
"""

from __future__ import annotations

import contextlib
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


# A "side" of a diff header is either ``a/path`` / ``b/path`` plain, or
# ``"a/path with space"`` / ``"b/path with space"`` quoted (git emits the
# quoted form when a path contains spaces, tabs, control chars, or
# non-ASCII bytes — see ``core.quotePath``). The wrapping quotes appear
# *outside* the ``a/`` / ``b/`` prefix.
_DIFF_GIT_RE = re.compile(
    r'^diff --git (?:a/(\S+)|"a/((?:[^"\\]|\\.)*)")'
    r' (?:b/(\S+)|"b/((?:[^"\\]|\\.)*)")$'
)
# ``--- a/X`` / ``+++ b/Y`` lines: in the plain-path form, modern git
# (default ``core.quotePath=true``) emits a trailing ``\t<timestamp>``
# even for a clean diff (the timestamp is empty for tracked files but
# the tab separator is still there). ``[^\t]+`` captures the path up
# to the tab; ``(?:\t.*)?`` consumes the optional trailing field.
# Plain ASCII paths-with-spaces parse here too, since we no longer
# require ``\S+``.
_PLUS_HEADER_RE = re.compile(
    r'^\+\+\+ (?:b/([^\t]+)|"b/((?:[^"\\]|\\.)*)")(?:\t.*)?$'
)
_MINUS_HEADER_RE = re.compile(
    r'^--- (?:a/([^\t]+)|"a/((?:[^"\\]|\\.)*)")(?:\t.*)?$'
)


def _parse_diff_git_line(line: str) -> tuple[str, str] | None:
    """Extract ``(a_path, b_path)`` from a ``diff --git`` header line.

    Two cases:

    1. Strict regex match: handles plain unspaced paths and the
       quoted-path form.
    2. Fallback for the unquoted-with-ASCII-space case: modern git
       emits ``diff --git a/PATH b/PATH`` literally, with no quoting
       and no escaping. The space-containing path makes
       ``\\S+ \\S+`` ambiguous — but git always repeats the same
       path twice for non-rename diffs. Find a ``" b/"`` split point
       where the a-side equals the b-side and use that.
    """
    m = _DIFF_GIT_RE.match(line)
    if m:
        return (
            _path_from_match(m.group(1), m.group(2)),
            _path_from_match(m.group(3), m.group(4)),
        )
    body = line.removeprefix("diff --git a/")
    if body == line:
        return None
    sep = " b/"
    idx = body.find(sep)
    while idx != -1:
        a, b = body[:idx], body[idx + len(sep) :]
        if a and a == b:
            return (a, b)
        idx = body.find(sep, idx + 1)
    return None

# Standard C-style single-char escapes git emits in quoted paths.
_QUOTED_SHORT_ESCAPES = {
    "a": b"\a",
    "b": b"\b",
    "f": b"\f",
    "n": b"\n",
    "r": b"\r",
    "t": b"\t",
    "v": b"\v",
    "\\": b"\\",
    '"': b'"',
}


def _decode_quoted_inner(inner: str) -> str:
    """Decode the inside of a git quoted-path token to a real path string.

    Git's quoted form is C-style: backslash-octal sequences encode raw
    bytes (e.g. ``\\303\\251`` for the UTF-8 bytes of ``é``). Walk the
    string character by character, accumulating real bytes, then decode
    the byte buffer as UTF-8. ``unicode_escape`` would interpret
    ``\\303`` as codepoint U+00C3, producing mojibake for any non-ASCII
    path — and a path that round-trips to mojibake silently bypasses the
    ``protected_paths`` check.
    """
    out = bytearray()
    i = 0
    n = len(inner)
    while i < n:
        ch = inner[i]
        if ch != "\\":
            out.extend(ch.encode("utf-8"))
            i += 1
            continue
        # Escape: look at the next character.
        if i + 1 >= n:
            out.append(ord("\\"))
            i += 1
            continue
        nxt = inner[i + 1]
        if nxt in _QUOTED_SHORT_ESCAPES:
            out.extend(_QUOTED_SHORT_ESCAPES[nxt])
            i += 2
            continue
        # Three-digit octal byte: \NNN
        if nxt.isdigit() and i + 3 < n + 1:
            octal = inner[i + 1 : i + 4]
            if len(octal) == 3 and all(c in "01234567" for c in octal):
                out.append(int(octal, 8))
                i += 4
                continue
        # Unknown escape — keep both characters literal.
        out.extend(("\\" + nxt).encode("utf-8"))
        i += 2
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return out.decode("utf-8", errors="replace")


def _path_from_match(plain: str, quoted: str) -> str:
    return plain if plain else _decode_quoted_inner(quoted)


# --- Single diff tokenizer -----------------------------------------------
#
# Every diff-introspection function in this module reads off the same
# per-file token stream: paths from headers, +/- payload counts from
# inside hunks. Keeping one walker means every parser sees the same view
# — no class of bug where ``count_patch_lines`` and
# ``extract_touched_files`` disagree about what is a header vs. payload.


@dataclass(frozen=True)
class FileSection:
    """One file's worth of a unified diff.

    ``a_paths`` and ``b_paths`` are kept ordered (insertion order) and
    are the union of the ``diff --git`` header sides and the
    corresponding ``--- a/X`` / ``+++ b/Y`` rows that appear before the
    first ``@@`` for this file. ``payload_lines`` counts ``+`` and ``-``
    rows inside ``@@`` hunks for this file only.
    """

    a_paths: tuple[str, ...]
    b_paths: tuple[str, ...]
    payload_lines: int


def _iter_diff_sections(diff_text: str) -> list[FileSection]:
    """Walk a unified diff once and emit per-file sections.

    Designed to be the single source of truth for diff parsing in this
    module. Behavior:

    - A ``diff --git`` line opens a new section. If a section was being
      built, it is closed and emitted first.
    - Inside the file header (between ``diff --git`` / start-of-input
      and the first ``@@``), ``--- a/X`` and ``+++ b/Y`` rows contribute
      paths.
    - The first ``@@`` line in a file flips to hunk mode; ``+`` / ``-``
      rows from that point until the next ``diff --git`` (or EOF) are
      counted as payload.
    - A diff that lacks any ``diff --git`` header (plain ``--- a/x`` /
      ``+++ b/x``) is treated as one anonymous section; the in-header
      default is True so the leading ``+++`` row is still a header.
    - ``/dev/null`` paths are dropped from both sides.
    """
    sections: list[FileSection] = []
    a_paths: list[str] = []
    b_paths: list[str] = []
    payload_lines = 0
    in_header = True
    started = False  # True once we have any content for a section

    def _add(seq: list[str], path: str) -> None:
        if path and path != "/dev/null" and path not in seq:
            seq.append(path)

    def _flush() -> None:
        nonlocal a_paths, b_paths, payload_lines, started
        if not started:
            return
        sections.append(
            FileSection(
                a_paths=tuple(a_paths),
                b_paths=tuple(b_paths),
                payload_lines=payload_lines,
            )
        )
        a_paths = []
        b_paths = []
        payload_lines = 0
        started = False

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            _flush()
            started = True
            in_header = True
            ab = _parse_diff_git_line(line)
            if ab is not None:
                _add(a_paths, ab[0])
                _add(b_paths, ab[1])
            continue
        if line.startswith("@@"):
            in_header = False
            started = True
            continue
        if in_header and line.startswith("+++ "):
            started = True
            mp = _PLUS_HEADER_RE.match(line)
            if mp:
                _add(b_paths, _path_from_match(mp.group(1), mp.group(2)))
            continue
        if in_header and line.startswith("--- "):
            started = True
            mm = _MINUS_HEADER_RE.match(line)
            if mm:
                _add(a_paths, _path_from_match(mm.group(1), mm.group(2)))
            continue
        if not in_header and line.startswith(("+", "-")):
            payload_lines += 1
            started = True
    _flush()
    return sections


def extract_touched_files(diff_text: str) -> list[str]:
    """Every repo-relative path the diff names.

    Returns the union of paths from ``diff --git a/X b/Y`` headers
    (both sides), ``+++ b/Y`` lines, and ``--- a/X`` lines. ``git apply``
    decides target files from the ``---``/``+++`` headers, not from
    ``diff --git`` — so a malicious / confused diff with mismatched
    headers would otherwise pass safety checks against the wrong path.
    Validating the union closes that gap.

    Both *source* and *destination* sides are extracted so a rename or
    copy diff (``diff --git a/secrets/key.py b/src/utils.py``) can't
    move a protected file out of a protected directory without
    triggering ``violates_protected_paths``.

    Handles git's quoted-path form (``"b/path with spaces.py"``) so a
    diff targeting a protected path that git happens to quote can't slip
    past the safety guards. Reads off the shared
    :func:`_iter_diff_sections` walker so payload lines beginning with
    ``+++ b/...`` or ``--- a/...`` (which would otherwise look like
    headers) are correctly classified as content.
    """
    seen: set[str] = set()
    paths: list[str] = []
    # Within a section, a-side first then b-side — matches the prior
    # walker's emission order (the ``diff --git`` header lists a/ before
    # b/).
    for section in _iter_diff_sections(diff_text):
        for p in (*section.a_paths, *section.b_paths):
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


def count_patch_lines(diff_text: str) -> int:
    """Count payload ``+``/``-`` lines across all hunks.

    The ``--- <path>`` and ``+++ <path>`` rows that wrap each file's
    hunks are *not* payload; they are file-headers and excluded. A
    removed line whose content happens to start with ``-- `` (which
    appears in the diff as ``--- <content>``) IS payload — the shared
    walker distinguishes header from hunk-content by position, not by
    regex on the line.
    """
    return sum(s.payload_lines for s in _iter_diff_sections(diff_text))


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

    return apply_diff_to_repo(diff_text, repo_root=repo_root, files=files)


def apply_diff_to_repo(
    diff_text: str, *, repo_root: Path, files: list[str]
) -> bool:
    """Apply a unified diff to a repository's working tree.

    Runs the git-level pipeline only:

    1. ``git apply --check`` — pre-flight, no mutation on failure.
    2. ``git apply`` — apply for real. On failure, best-effort
       ``git apply --reverse`` to clean up any partial mutation
       (``git apply`` is documented atomic, but the revert is cheap
       insurance).
    3. ``git diff --check`` (post-apply, scoped to ``files``) — catches
       whitespace errors and merge-conflict markers (``<<<<<<<``,
       ``=======``, ``>>>>>>>``) that ``--check`` at the patch level
       can't see because they parse as valid diff hunks. On failure,
       reverse-applies and returns False.

    Returns ``True`` on a clean apply (working tree mutated), ``False``
    on any git-level rejection (logs the reason at INFO).

    Safety guards (file count, protected paths, line limits, …) are
    explicitly NOT this function's responsibility — callers must run
    their own. Two callers exist:

    - :func:`apply_unified_diff` — the spec free function with its
      own guard list.
    - :meth:`Patcher.apply_diff` — the class wrapper with project-
      specific :data:`FORBIDDEN_GLOBS` on top.

    Both run their guards then delegate here. Keeping the git work in
    one place means there's a single owner of the temp-file handling,
    revert ordering, and ``--check`` scoping.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as tmp:
        tmp.write(diff_text if diff_text.endswith("\n") else diff_text + "\n")
        tmp_path = tmp.name
    try:
        try:
            _run_git(["apply", "--check", tmp_path], cwd=repo_root)
        except RuntimeError as e:
            log.info("apply_diff_to_repo: git apply --check failed: %s", e)
            return False
        try:
            _run_git(["apply", tmp_path], cwd=repo_root)
        except RuntimeError as e:
            log.info("apply_diff_to_repo: git apply failed: %s", e)
            # git apply is documented as atomic but defensive: if anything
            # did mutate before the failure, reverse-apply so the per-thread
            # fallback starts from a clean tree. Best-effort — a failed
            # revert doesn't change the return.
            with contextlib.suppress(RuntimeError):
                _run_git(["apply", "--reverse", tmp_path], cwd=repo_root)
            return False

        # Post-apply: git diff --check catches whitespace errors and merge-
        # conflict markers (<<<<<<<, =======, >>>>>>>) that --check at the
        # patch level can't see because they're valid diff hunks.
        # Scope to only the files touched by this patch to avoid spurious
        # failures from pre-existing uncommitted changes in other files.
        try:
            _run_git(["diff", "--check", "--"] + files, cwd=repo_root)
        except RuntimeError as e:
            log.info("apply_diff_to_repo: git diff --check failed; reverting: %s", e)
            try:
                _run_git(["apply", "--reverse", tmp_path], cwd=repo_root)
            except RuntimeError as revert_err:  # working tree now in unknown state
                log.warning(
                    "apply_diff_to_repo: revert via git apply --reverse FAILED (%s); "
                    "working tree may be inconsistent",
                    revert_err,
                )
            return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    log.info("apply_diff_to_repo: applied %d file(s)", len(files))
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

        # All guards passed at the class level. Delegate the actual git
        # work (apply --check / apply / post-apply diff --check / revert)
        # to the shared pipeline. We bypass apply_unified_diff to avoid
        # re-running the safety guards we just performed (file count,
        # protected paths, line limits) — the pipeline only touches git.
        ok = apply_diff_to_repo(
            diff_text, repo_root=self._root, files=files
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

    def stage_and_commit(
        self,
        summary_lines: list[str],
        paths: list[Path],
        author_email: str,
    ) -> str | None:
        """Stage paths and commit if there's anything to commit.

        Returns the new commit sha, or ``None`` when ``git status --porcelain``
        is empty after staging (the patch was a no-op against current content).
        Callers treat ``None`` as "commit was skipped, don't push or reply".

        ``summary_lines`` becomes the commit body (one bullet per thread,
        already formatted by the caller). The header is fixed per the
        Phase 12 spec: ``fix: address Cursor Bugbot comments``.
        """
        rel = [str(p.relative_to(self._root)) for p in paths]
        self._git("add", "--", *rel)
        if not self._has_changes_to_commit():
            log.info("stage_and_commit: nothing to commit (working tree clean).")
            return None
        header = "fix: address Cursor Bugbot comments"
        body = "\n".join(summary_lines)
        msg = f"{header}\n\n{body}" if body else header
        env_args = [
            "-c",
            f"user.email={author_email}",
            "-c",
            "user.name=pr-autofix-agent[bot]",
        ]
        self._git(*env_args, "commit", "-m", msg)
        return self._git("rev-parse", "HEAD").strip()

    def _has_changes_to_commit(self) -> bool:
        """True iff there are staged changes ready for ``git commit``.

        Checks the index directly via ``git diff --cached --name-only``.
        We deliberately do NOT use ``git status --porcelain`` because that
        includes untracked files (``??`` lines) by default. The agent
        writes its own ``.pr-agent-state.json`` to the repo root during the
        run; target repos won't have it in their ``.gitignore``, so a
        porcelain check would always report changes (the untracked state
        file) and the no-op guard would never fire — letting ``git commit``
        fail with ``nothing added to commit but untracked files present``.
        """
        return bool(self._git("diff", "--cached", "--name-only").strip())

    def push(self, branch: str) -> None:
        self._git("push", "origin", f"HEAD:{branch}")

    def revert_uncommitted(self, paths: list[Path]) -> None:
        rel = [str(p.relative_to(self._root)) for p in paths]
        if not rel:
            return
        self._git("checkout", "--", *rel)

    def _git(self, *args: str) -> str:
        return _run_git(list(args), cwd=self._root)
