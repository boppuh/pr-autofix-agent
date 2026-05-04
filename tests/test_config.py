from __future__ import annotations

import textwrap

from pr_agent.config import (
    BUGBOT_AUTHOR_MATCHES,
    DEFAULT_PROTECTED_PATHS,
    DEFAULT_VALIDATION_COMMANDS,
    MAX_COMMENTS_PER_ROUND,
    MAX_FILES_TOUCHED,
    MAX_PATCH_LINES,
    MAX_ROUNDS,
    MAX_RUNTIME_MINUTES,
    load_target_repo_config,
)


def test_constants_match_spec():
    assert MAX_ROUNDS == 5
    assert MAX_COMMENTS_PER_ROUND == 20
    assert MAX_PATCH_LINES == 800
    assert MAX_FILES_TOUCHED == 15
    assert MAX_RUNTIME_MINUTES == 20
    assert BUGBOT_AUTHOR_MATCHES == ["cursor", "bugbot", "cursor-bugbot"]
    assert DEFAULT_VALIDATION_COMMANDS == [
        "npm test",
        "npm run lint",
        "npm run typecheck",
    ]
    assert DEFAULT_PROTECTED_PATHS == [
        ".github/workflows/",
        "infra/",
        "migrations/",
        "secrets/",
    ]


def test_loads_defaults_when_file_missing(tmp_path):
    cfg = load_target_repo_config(tmp_path)
    assert [c.run for c in cfg.validate_] == DEFAULT_VALIDATION_COMMANDS
    assert cfg.protected_paths == DEFAULT_PROTECTED_PATHS
    assert cfg.safety.max_rounds == MAX_ROUNDS
    assert cfg.safety.max_runtime_minutes == MAX_RUNTIME_MINUTES
    # Phase 10 default: exit on first validation failure (per spec).
    assert cfg.safety.exit_on_validation_failure is True
    assert cfg.bugbot_logins == BUGBOT_AUTHOR_MATCHES


def test_safety_exit_on_validation_failure_can_be_disabled(tmp_path):
    """Override the spec default to opt back into the retry loop."""
    (tmp_path / ".pr-agent.yml").write_text(
        textwrap.dedent(
            """
            safety:
              exit_on_validation_failure: false
            """
        )
    )
    cfg = load_target_repo_config(tmp_path)
    assert cfg.safety.exit_on_validation_failure is False
    # Other safety fields keep their defaults.
    assert cfg.safety.max_rounds == MAX_ROUNDS


def test_safety_post_per_thread_replies_default_and_override(tmp_path):
    """Phase 13 toggle defaults to True; YAML override to false works."""
    cfg = load_target_repo_config(tmp_path)
    assert cfg.safety.post_per_thread_replies is True

    (tmp_path / ".pr-agent.yml").write_text(
        textwrap.dedent(
            """
            safety:
              post_per_thread_replies: false
            """
        )
    )
    cfg2 = load_target_repo_config(tmp_path)
    assert cfg2.safety.post_per_thread_replies is False


def test_empty_lists_are_respected(tmp_path):
    """Explicitly empty lists must not silently fall back to defaults."""
    (tmp_path / ".pr-agent.yml").write_text(
        textwrap.dedent(
            """
            validation:
              commands: []
            protected_paths: []
            bugbot_logins: []
            """
        )
    )
    cfg = load_target_repo_config(tmp_path)
    assert cfg.validate_ == []
    assert cfg.protected_paths == []
    assert cfg.bugbot_logins == []


def test_overrides_via_yaml(tmp_path):
    (tmp_path / ".pr-agent.yml").write_text(
        textwrap.dedent(
            """
            validation:
              commands:
                - pytest
                - {name: lint, run: ruff check .}

            safety:
              max_rounds: 2
              max_patch_lines: 100
              max_files_touched: 3

            protected_paths:
              - "secrets/"
              - "**/*.generated.*"
            """
        )
    )
    cfg = load_target_repo_config(tmp_path)
    assert [c.run for c in cfg.validate_] == ["pytest", "ruff check ."]
    assert cfg.validate_[1].name == "lint"
    assert cfg.safety.max_rounds == 2
    assert cfg.safety.max_patch_lines == 100
    assert cfg.safety.max_files_touched == 3
    # Unspecified safety fields fall back to constants.
    assert cfg.safety.max_runtime_minutes == MAX_RUNTIME_MINUTES
    assert cfg.protected_paths == ["secrets/", "**/*.generated.*"]
