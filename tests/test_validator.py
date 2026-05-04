from __future__ import annotations

from unittest.mock import patch

from pr_agent.models import CommandResult, ValidateCommand, ValidationResult
from pr_agent.validator import Validator, run_validation


def test_run_validation_empty_list_succeeds(tmp_path):
    result = run_validation([], repo_root=tmp_path)
    assert isinstance(result, ValidationResult)
    assert result.success is True
    assert result.command_results == []


def test_run_validation_single_success(tmp_path):
    result = run_validation(["true"], repo_root=tmp_path)
    assert result.success is True
    assert len(result.command_results) == 1
    assert result.command_results[0].ok is True
    assert result.command_results[0].exit_code == 0
    assert result.command_results[0].name == "true"


def test_run_validation_single_failure(tmp_path):
    result = run_validation(["false"], repo_root=tmp_path)
    assert result.success is False
    assert result.first_failure is not None
    assert result.first_failure.exit_code != 0


def test_run_validation_short_circuits_on_first_failure(tmp_path):
    """Spec: stop running commands at the first non-zero exit."""
    result = run_validation(["true", "false", "true"], repo_root=tmp_path)
    assert result.success is False
    # Only two commands ran (the third is never executed).
    assert len(result.command_results) == 2
    assert [c.name for c in result.command_results] == ["true", "false"]
    assert result.command_results[0].ok is True
    assert result.command_results[1].ok is False


def test_run_validation_captures_stderr_tail(tmp_path):
    """The CommandResult should carry the stderr tail for the failure summary."""
    result = run_validation(['sh -c "echo BANG >&2; exit 7"'], repo_root=tmp_path)
    assert result.success is False
    failure = result.first_failure
    assert failure is not None
    assert failure.exit_code == 7
    assert "BANG" in failure.stderr_tail


def test_run_validation_timeout_is_failure(tmp_path):
    """Mock the subprocess timeout so we don't actually sleep 15 minutes."""
    import subprocess as sp

    real_run = sp.run

    def fake_run(*args, **kwargs):
        # Match the call signature run_validation uses.
        if kwargs.get("shell") and "sleep" in args[0]:
            raise sp.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 0))
        return real_run(*args, **kwargs)

    with patch("pr_agent.validator.subprocess.run", side_effect=fake_run):
        result = run_validation(["sleep 999"], repo_root=tmp_path)
    assert result.success is False
    failure = result.first_failure
    assert failure is not None
    assert failure.exit_code == -1
    assert "timed out" in failure.stderr_tail


# --- Validator class wrapper ---------------------------------------------


def test_validator_class_returns_aggregate(tmp_path):
    v = Validator(
        repo_root=tmp_path,
        commands=[
            ValidateCommand(name="ok", run="true"),
            ValidateCommand(name="fail", run="false"),
            ValidateCommand(name="never", run="true"),
        ],
    )
    result = v.run()
    assert isinstance(result, ValidationResult)
    assert result.success is False
    assert [c.name for c in result.command_results] == ["ok", "fail"]


def test_validator_format_failure_returns_empty_on_success(tmp_path):
    result = ValidationResult(
        success=True,
        command_results=[CommandResult(name="ok", ok=True, exit_code=0)],
    )
    assert Validator.format_failure(result) == ""


def test_validator_format_failure_includes_first_failure(tmp_path):
    result = ValidationResult(
        success=False,
        command_results=[
            CommandResult(name="ok", ok=True, exit_code=0),
            CommandResult(
                name="lint",
                ok=False,
                exit_code=1,
                stderr_tail="F401 unused import",
                stdout_tail="",
            ),
        ],
    )
    out = Validator.format_failure(result)
    assert "[lint] exit=1" in out
    assert "F401 unused import" in out


# --- Aggregate model -----------------------------------------------------


def test_validation_result_first_failure_returns_first_non_ok():
    r = ValidationResult(
        success=False,
        command_results=[
            CommandResult(name="a", ok=True, exit_code=0),
            CommandResult(name="b", ok=False, exit_code=2),
            CommandResult(name="c", ok=False, exit_code=3),
        ],
    )
    assert r.first_failure is not None
    assert r.first_failure.name == "b"


def test_validation_result_first_failure_returns_none_when_all_ok():
    r = ValidationResult(
        success=True,
        command_results=[CommandResult(name="a", ok=True, exit_code=0)],
    )
    assert r.first_failure is None
