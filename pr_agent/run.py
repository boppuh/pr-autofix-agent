from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from .classifier import Classifier
from .config import ConfigError, load_target_repo_config, load_workflow_inputs, require_env
from .github_client import GitHubClient
from .llm_client import LLMClient
from .models import (
    EscalationReason,
    Patch,
    ReviewThread,
    RoundResult,
    WorkflowInputs,
)
from .patcher import Patcher, UnsafePatchError
from .state import AgentState
from .validator import Validator

log = logging.getLogger("pr_agent")

REPEATED_FAILURE_LIMIT = 2


def main(argv: list[str] | None = None) -> int:
    inputs = load_workflow_inputs(argv)
    logging.basicConfig(
        level=getattr(logging, inputs.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    repo_root = Path(os.environ.get("GITHUB_WORKSPACE") or os.getcwd()).resolve()
    repo_cfg = load_target_repo_config(repo_root)

    state_path = repo_root / "pr_agent_state.json"
    state = AgentState(inputs.pr_number, persist_path=state_path)

    gh_token = require_env("GITHUB_TOKEN")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    gh = GitHubClient(gh_token, inputs.repo_full_name, repo_cfg.bugbot_logins)
    llm = LLMClient(model=inputs.model, api_key=anthropic_key)
    classifier = Classifier(
        llm=llm,
        exclude_paths=repo_cfg.exclude_paths,
        confidence_threshold=inputs.confidence_threshold,
    )
    patcher = Patcher(
        repo_root=repo_root,
        exclude_paths=repo_cfg.exclude_paths,
        max_files_per_patch=repo_cfg.max_files_per_patch,
    )
    validator = Validator(repo_root=repo_root, commands=repo_cfg.validate_)

    pr = gh.get_pull(inputs.pr_number)
    head_ref = pr.head.ref
    last_failure: str | None = None

    for round_no in range(1, inputs.max_rounds + 1):
        log.info("=== round %d/%d ===", round_no, inputs.max_rounds)
        threads = gh.list_unresolved_bugbot_threads(inputs.pr_number)
        if not threads:
            log.info("No unresolved Bugbot threads. Done.")
            break

        excerpts = _gather_excerpts(gh, threads, head_ref)
        triage = classifier.triage(threads, excerpts)
        round_result = RoundResult(round_no=round_no)
        round_result.skipped.extend((t.id, r) for t, r in triage.skipped)

        if not triage.fixable:
            log.info("No auto-fixable threads remaining.")
            state.record_round(round_result)
            break

        patches: list[tuple[ReviewThread, Patch]] = []
        for thread in triage.fixable:
            if state.already_handled(thread.id, thread.body_text):
                round_result.skipped.append((thread.id, "already handled"))
                continue
            file_contents = _collect_file_contents(repo_root, thread)
            patch = llm.propose_patch(
                thread=thread,
                file_contents=file_contents,
                max_files=repo_cfg.max_files_per_patch,
                prior_failure=last_failure,
            )
            try:
                report = patcher.check_safe(patch)
            except UnsafePatchError as e:
                round_result.skipped.append((thread.id, f"unsafe: {e}"))
                continue
            if not report.ok:
                round_result.skipped.append((thread.id, f"unsafe: {'; '.join(report.reasons)}"))
                continue
            patches.append((thread, patch))

        if not patches:
            log.info("Round %d produced no safe patches.", round_no)
            state.record_round(round_result)
            break

        if inputs.dry_run:
            for thread, patch in patches:
                log.info(
                    "[DRY RUN] would patch %s for thread %s: %s",
                    patch.touched_paths(),
                    thread.id,
                    patch.summary,
                )
            state.record_round(round_result)
            return 0

        applied_paths: list[Path] = []
        for _, patch in patches:
            applied_paths.extend(patcher.apply(patch))

        round_result.validation = validator.run()
        if not round_result.validation_ok:
            failure_text = Validator.format_failure(round_result.validation)
            last_failure = failure_text
            patcher.revert_uncommitted(applied_paths)
            sig = AgentState.signature_for(failure_text)
            seen = state.record_validation_failure(sig)
            round_result.error = f"validation failed (sig {sig}, seen {seen}x)"
            state.record_round(round_result)
            if seen >= REPEATED_FAILURE_LIMIT:
                _escalate(
                    gh,
                    inputs,
                    state,
                    EscalationReason.REPEATED_VALIDATION_FAILURE,
                    [t.id for t in threads],
                )
                return 0
            continue

        author_email = os.environ.get("GIT_AUTHOR_EMAIL", "pr-autofix-agent@users.noreply.github.com")
        sha = patcher.stage_and_commit(patches[0][1], applied_paths, author_email)
        patcher.push(head_ref)
        round_result.commit_sha = sha
        round_result.fixed_thread_ids = [t.id for t, _ in patches]

        for thread, patch in patches:
            body = _format_reply(patch, sha)
            try:
                gh.reply_to_thread(inputs.pr_number, thread.root_comment.id, body)
                gh.resolve_thread(thread.id)
            except Exception as e:
                log.warning("Failed to resolve thread %s: %s", thread.id, e)
            state.mark_handled(thread.id, thread.body_text)

        state.record_round(round_result)
        last_failure = None
    else:
        unresolved = [t.id for t in gh.list_unresolved_bugbot_threads(inputs.pr_number)]
        _escalate(gh, inputs, state, EscalationReason.MAX_ROUNDS, unresolved)

    gh.close()
    return 0


def _gather_excerpts(
    gh: GitHubClient, threads: list[ReviewThread], ref: str
) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for t in threads:
        if not t.path:
            out[t.id] = None
            continue
        full = gh.get_file_contents(t.path, ref) or ""
        out[t.id] = _excerpt_around(full, t.line)
    return out


def _excerpt_around(content: str, line: int | None, window: int = 25) -> str:
    if not content:
        return ""
    lines = content.splitlines()
    if line is None:
        return "\n".join(lines[: window * 2])
    lo = max(0, line - window)
    hi = min(len(lines), line + window)
    return "\n".join(lines[lo:hi])


def _collect_file_contents(repo_root: Path, thread: ReviewThread) -> dict[str, str]:
    contents: dict[str, str] = {}
    if thread.path:
        p = repo_root / thread.path
        if p.exists() and p.is_file():
            contents[thread.path] = p.read_text(errors="replace")
    return contents


def _format_reply(patch: Patch, sha: str) -> str:
    files = "\n".join(f"- `{f.path}`" for f in patch.files)
    return (
        f"🤖 **pr-autofix-agent** applied a fix in `{sha[:7]}`:\n\n"
        f"{patch.summary}\n\n"
        f"Files changed:\n{files}\n"
    )


def _escalate(
    gh: GitHubClient,
    inputs: WorkflowInputs,
    state: AgentState,
    reason: EscalationReason,
    unresolved: list[str],
) -> None:
    state.escalate(reason, unresolved)
    try:
        gh.add_label(inputs.pr_number, inputs.needs_human_label)
        gh.post_pr_comment(
            inputs.pr_number,
            f"🤖 **pr-autofix-agent** is escalating to a human reviewer.\n\n"
            f"Reason: `{reason.value}`\n"
            f"Unresolved Bugbot threads: {len(unresolved)}\n",
        )
    except Exception as e:
        log.warning("Failed to post escalation: %s", e)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        sys.exit(2)
