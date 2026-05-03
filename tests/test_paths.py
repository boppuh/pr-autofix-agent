from __future__ import annotations

from pr_agent._paths import matches_any_protected


def test_directory_prefix():
    assert matches_any_protected("infra/main.tf", ["infra/"])
    assert matches_any_protected("infra", ["infra/"])
    assert not matches_any_protected("infrastructure/main.tf", ["infra/"])


def test_glob_match():
    assert matches_any_protected("a/b.generated.py", ["**/*.generated.*"])
    assert matches_any_protected("file.lock", ["*.lock"])


def test_no_match():
    assert not matches_any_protected("src/foo.py", ["infra/", "*.lock"])


def test_empty_protected_list():
    assert not matches_any_protected("any/path", [])
