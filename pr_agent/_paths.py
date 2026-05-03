"""Shared path-matching helpers used by classifier and patcher."""

from __future__ import annotations

import fnmatch


def matches_any_protected(path: str, protected: list[str]) -> bool:
    """Return True if `path` is matched by any entry in `protected`.

    Trailing-slash entries are treated as directory prefixes (e.g. `infra/`
    matches `infra/main.tf` and `infra`). All other entries are matched as
    fnmatch globs.
    """
    for pat in protected:
        if pat.endswith("/"):
            if path == pat.rstrip("/") or path.startswith(pat):
                return True
        elif fnmatch.fnmatch(path, pat):
            return True
    return False
