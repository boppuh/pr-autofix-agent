from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from pathlib import Path

from .classifier import Classifier
from .config import ConfigError, load_target_repo_config, load_workflow_inputs, require_env
from .github_client import GitHubClient
from .llm import LLMResponseError, make_provider
from .llm._factory import default_model_for, env_var_for
from .models import (
    AgentRunReport,
    CommandResult,
    EscalatedThread,
    EscalationReason,
    HandledThread,
    Patch,
    PatchFile,
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

    state_path = repo_root / ".pr-agent-state.json"
    state = AgentState(inputs.pr_number, persist_path=state_path)

    gh_token = require_env("GITHUB_TOKEN")
    provider_env = env_var_for(inputs.provider)
    llm_key = os.environ.get(provider_env)
    model = inputs.model or default_model_for(inputs.provider)

    safety = repo_cfg.safety
    max_rounds = min(inputs.max_rounds, safety.max_rounds)
    runtime_deadline = time.monotonic() + safety.max_runtime_minutes * 60

    gh = GitHubClient.from_full_name(gh_token, inputs.repo_full_name)

    # Short-circuit before constructing any LLM-dependent component, so the
    # provider SDK is never instantiated without a key (defense in depth:
    # current SDKs defer key validation to first request, but future versions
    # may raise at construction).
    if not llm_key:
        threads = gh.get_unresolved_bugbot_threads(inputs.pr_number, repo_cfg.bugbot_logins)
        if threads:
            log.warning(
                "%s is not set; cannot triage %d Bugbot thread(s) with provider=%s. "
                "Escalating without LLM calls.",
                provider_env,
                len(threads),
                inputs.provider,
            )
            _escalate(
                gh,
                inputs,
                state,
                EscalationReason.MISSING_LLM_CREDENTIAL,
                [t.id for t in threads],
            )
        else:
            log.info(
                "%s is not set, but no Bugbot threads to triage. Done.", provider_env
            )
        return _finish(gh, inputs.pr_number, state)

    log.info("Using LLM provider=%s model=%s", inputs.provider, model)
    llm = make_provider(inputs.provider, model=model, api_key=llm_key)
    classifier = Classifier(
        llm=llm,
        protected_paths=repo_cfg.protected_paths,
        confidence_threshold=inputs.confidence_threshold,
    )
    patcher = Patcher(
        repo_root=repo_root,
        protected_paths=repo_cfg.protected_paths,
        max_files_touched=safety.max_files_touched,
        max_patch_lines=safety.max_patch_lines,
    )
    validator = Validator(repo_root=repo_root, commands=repo_cfg.validate_)

    pr = gh.get_pr(inputs.pr_number)
    head_ref = pr.head_ref_name
    pr_diff = gh.get_pr_diff(inputs.pr_number)
    last_failure: str | None = None

    for round_no in range(1, max_rounds + 1):
        if time.monotonic() > runtime_deadline:
            log.warning("Runtime budget (%dm) exhausted.", safety.max_runtime_minutes)
            unresolved = [t.id for t in gh.get_unresolved_bugbot_threads(inputs.pr_number, repo_cfg.bugbot_logins)]
            _escalate(gh, inputs, state, EscalationReason.RUNTIME_BUDGET_EXHAUSTED, unresolved)
            return _finish(gh, inputs.pr_number, state)

        log.info("=== round %d/%d ===", round_no, max_rounds)
        threads = gh.get_unresolved_bugbot_threads(inputs.pr_number, repo_cfg.bugbot_logins)
        if not threads:
            log.info("No unresolved Bugbot threads. Done.")
            _post_clean(gh, inputs.pr_number)
            return _finish(gh, inputs.pr_number, state)

        # Stop-on-no-progress guard: from round 2 onward, escalate if the
        # unresolved count didn't decrease since the previous round.
        if not state.start_round(len(threads)):
            _escalate(
                gh,
                inputs,
                state,
                EscalationReason.NO_PROGRESS,
                [t.id for t in threads],
            )
            return _finish(gh, inputs.pr_number, state)

        if len(threads) > safety.max_comments_per_round:
            log.info(
                "Capping %d threads to max_comments_per_round=%d",
                len(threads),
                safety.max_comments_per_round,
            )
            threads = threads[: safety.max_comments_per_round]

        excerpts = _gather_excerpts(gh, threads, head_ref)
        triage = classifier.triage(threads, excerpts, prior_failure=last_failure)
        round_result = RoundResult(round_no=round_no)
        # NEEDS_HUMAN threads are tracked as skipped (they keep the PR labelled
        # for review). IGNORE threads are recorded but won't drive escalation.
        round_result.skipped.extend((t.id, r) for t, r in triage.skipped)
        round_result.skipped.extend((t.id, r) for t, r in triage.ignored)
        # Surface NEEDS_HUMAN routes for the end-of-run summary's
        # 'Escalated:' section. IGNORE entries are silent (non-actionable).
        round_result.escalated_to_human.extend(
            EscalatedThread(thread_id=t.id, location=_loc(t), reason=r)
            for t, r in triage.skipped
        )

        if not triage.fixable:
            # Phase 11 spec: if every thread routed to NEEDS_HUMAN, escalate
            # explicitly (label + summary comment + return). If everything is
            # IGNORE-only (or both lists empty), break out without escalation
            # — IGNORE means non-actionable, no human attention needed.
            if triage.skipped:
                log.info("All threads routed to NEEDS_HUMAN; escalating.")
                body = _summarize_human_threads(triage.skipped)
                # Each call gets its own try/except so a comment-API hiccup
                # doesn't drop the label (and vice versa) — humans need at
                # least one of the two signals to discover the PR.
                try:
                    gh.create_pr_comment(inputs.pr_number, body)
                except Exception as e:
                    log.warning("Could not post all-needs-human comment: %s", e)
                try:
                    gh.add_labels(
                        inputs.pr_number,
                        [inputs.needs_human_label, "agent:needs-human"],
                    )
                except Exception as e:
                    log.warning("Could not apply all-needs-human label: %s", e)
                state.record_round(round_result)
                state.escalate(
                    EscalationReason.NO_FIXABLE_THREADS,
                    [t.id for t in threads],
                )
                return _finish(gh, inputs.pr_number, state)
            log.info("No auto-fixable threads remaining.")
            state.record_round(round_result)
            break

        # Skip already-processed threads (dedupe across the whole run).
        live_fixable: list[ReviewThread] = []
        for thread in triage.fixable:
            if state.already_processed(thread.root_comment):
                round_result.skipped.append((thread.id, "already processed (dedupe)"))
            else:
                live_fixable.append(thread)
        if not live_fixable:
            log.info("Round %d: nothing new to fix after dedupe.", round_no)
            state.record_round(round_result)
            break

        applied_paths: list[Path] = []
        patches: list[tuple[ReviewThread, Patch]] = []
        batch_used = False

        # --- Phase 8: batched generate_patch first --------------------------
        try:
            repo_context = _build_repo_context(repo_root, live_fixable, budget=32_000)
            diff = llm.generate_patch(
                pr_title=pr.title,
                pr_body=pr.body or "",
                pr_diff=pr_diff,
                comments=live_fixable,
                repo_context=repo_context,
                validation_commands=[c.run for c in repo_cfg.validate_],
                prior_failure=last_failure,
            )
        except LLMResponseError as e:
            log.info("Batch generate_patch rejected (%s); falling back to per-thread.", e)
            diff = None
        except Exception as e:  # provider-side errors
            log.info("Batch generate_patch failed (%s); falling back to per-thread.", e)
            diff = None

        if diff and diff.startswith("ESCALATE:"):
            # The batch model may decline for whole-batch reasons ("too many
            # comments", "needs cross-file refactor") that don't apply to
            # individual threads. Fall through to the per-thread path so
            # each thread gets its own chance — don't burn attempts and
            # skip the round.
            reason = diff.removeprefix("ESCALATE:").strip() or "model returned ESCALATE"
            log.info("Model escalated batch (%s); falling through to per-thread.", reason)
            diff = None

        if diff is not None:
            # Dry-run check BEFORE we mutate anything via git apply.
            if inputs.dry_run:
                log.info(
                    "[DRY RUN] would apply batched diff for %d thread(s)",
                    len(live_fixable),
                )
                for t in live_fixable:
                    state.increment_attempt(t.id)
                state.record_round(round_result)
                return _finish(gh, inputs.pr_number, state)
            try:
                applied_paths = patcher.apply_diff(diff, [t.id for t in live_fixable])
            except UnsafePatchError as e:
                # Don't increment attempts here — the per-thread fallback below
                # will increment per thread as it actually consumes attempts.
                log.info("Batched diff rejected by Patcher (%s); falling back.", e)
                applied_paths = []
                patches = []
            else:
                for t in live_fixable:
                    state.increment_attempt(t.id)
                # Synthesize a single Patch for the commit message + replies.
                rel_paths = [str(p.relative_to(repo_root)) for p in applied_paths]
                synthetic = Patch(
                    thread_id=f"batch-r{round_no}",
                    files=[PatchFile(path=p, new_content="", rationale="batched") for p in rel_paths],
                    summary=f"batched fix for {len(live_fixable)} thread(s)",
                )
                patches = [(t, synthetic) for t in live_fixable]
                batch_used = True
                log.info("Batched patch applied: %d files", len(applied_paths))

        # --- Per-thread fallback (Phase 5 path) -----------------------------
        if not batch_used:
            for thread in live_fixable:
                file_contents = _collect_file_contents(repo_root, thread)
                attempt_no = state.increment_attempt(thread.id)
                log.info("Thread %s: per-thread attempt #%d", thread.id, attempt_no)
                try:
                    patch = llm.propose_patch(
                        thread=thread,
                        file_contents=file_contents,
                        max_files=safety.max_files_touched,
                        prior_failure=last_failure,
                        pr_title=pr.title,
                        pr_body_excerpt=pr.body[:2000] if pr.body else None,
                        pr_diff_excerpt=pr_diff,
                    )
                except LLMResponseError as e:
                    log.warning("LLM patch output unusable for thread %s: %s", thread.id, e)
                    round_result.skipped.append((thread.id, f"llm output unusable: {e}"))
                    continue
                except Exception as e:
                    log.warning("LLM call failed for thread %s: %s", thread.id, e)
                    round_result.skipped.append((thread.id, f"llm error: {e}"))
                    continue
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
                return _finish(gh, inputs.pr_number, state)

            for _, patch in patches:
                applied_paths.extend(patcher.apply(patch))

        round_result.validation = validator.run()
        if not round_result.validation_ok:
            failure_text = Validator.format_failure(round_result.validation)
            last_failure = failure_text
            patcher.revert_uncommitted(applied_paths)

            # The patch is reverted at this point, so EVERY Bugbot thread on
            # the PR is still unresolved — not just the live_fixable subset.
            # Re-fetch from the API rather than reusing local ``threads``,
            # which may have been capped to ``max_comments_per_round``.
            # Consistent with NO_PROGRESS, MAX_ROUNDS, and
            # RUNTIME_BUDGET_EXHAUSTED escalation paths.
            unresolved_ids = [
                t.id
                for t in gh.get_unresolved_bugbot_threads(
                    inputs.pr_number, repo_cfg.bugbot_logins
                )
            ]

            if safety.exit_on_validation_failure:
                # Phase 10 default: post a PR comment summarising the failed
                # command, apply the agent:failed-validation label, and exit.
                first = round_result.validation.first_failure
                _post_validation_failure(gh, inputs.pr_number, first)
                round_result.error = "validation failed (exit-on-failure)"
                state.record_round(round_result)
                _escalate(
                    gh,
                    inputs,
                    state,
                    EscalationReason.VALIDATION_FAILED,
                    unresolved_ids,
                )
                return _finish(gh, inputs.pr_number, state)

            # Retry path: feed the failure into the next LLM round; escalate
            # only if we see the same failure signature twice in a row.
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
                    unresolved_ids,
                )
                return _finish(gh, inputs.pr_number, state)
            continue

        author_email = os.environ.get(
            "GIT_AUTHOR_EMAIL", "pr-autofix-agent[bot]@users.noreply.github.com"
        )
        # Aggregate one summary line per thread across the entire round so
        # the commit body covers every fix, not just patches[0]. The batched
        # path pairs the same synthetic Patch with every thread; collapse
        # into one line per unique patch (keyed by thread_id) so the commit
        # body doesn't repeat "batched fix for N thread(s)" N times.
        seen_patch_ids: set[str] = set()
        summary_lines: list[str] = []
        for thread, patch in patches:
            if patch.thread_id in seen_patch_ids:
                continue
            seen_patch_ids.add(patch.thread_id)
            if batch_used:
                ids = ", ".join(t.id for t, _ in patches)
                summary_lines.append(f"- threads {ids}: {patch.summary}")
            else:
                summary_lines.append(f"- thread {thread.id}: {patch.summary}")
        sha = patcher.stage_and_commit(summary_lines, applied_paths, author_email)
        if sha is None:
            log.info(
                "Round %d: nothing to commit (patch was a no-op vs. working tree); "
                "skipping push and thread replies.",
                round_no,
            )
            state.record_round(round_result)
            last_failure = None
            continue
        patcher.push(head_ref)
        round_result.commit_sha = sha
        round_result.fixed_thread_ids = [t.id for t, _ in patches]

        for thread, patch in patches:
            if safety.post_per_thread_replies:
                body = _format_reply(patch, sha)
                try:
                    gh.reply_to_thread(inputs.pr_number, thread.root_comment.id, body)
                except Exception as e:
                    log.warning("Failed to reply to thread %s: %s", thread.id, e)
            try:
                gh.resolve_thread(thread.id)
            except Exception as e:
                log.warning("Failed to resolve thread %s: %s", thread.id, e)
            state.mark_processed(thread.root_comment)
            round_result.handled.append(
                HandledThread(
                    thread_id=thread.id,
                    location=_loc(thread),
                    summary=patch.summary,
                )
            )

        state.record_round(round_result)
        last_failure = None
    else:
        unresolved = [t.id for t in gh.get_unresolved_bugbot_threads(inputs.pr_number, repo_cfg.bugbot_logins)]
        _escalate(gh, inputs, state, EscalationReason.MAX_ROUNDS, unresolved)

    return _finish(gh, inputs.pr_number, state)


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


def _build_repo_context(
    repo_root: Path,
    threads: list[ReviewThread],
    *,
    budget: int = 32_000,
) -> str:
    """Build the `repo_context` string for the batched generate_patch call.

    Includes (in order, each truncated to its own sub-budget):
    1. file tree from `git ls-files`
    2. README.md / CLAUDE.md excerpt if present
    3. excerpts of files referenced by any thread
    Total truncated to ``budget`` bytes.
    """
    sections: list[str] = []

    # File tree
    try:
        import subprocess as _sp

        out = _sp.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        tree = out.stdout if out.returncode == 0 else ""
    except OSError:
        tree = ""
    if tree:
        sections.append("--- file tree ---\n" + _truncate(tree, 5000))

    # README / CLAUDE.md — break only on successful read
    for fname in ("README.md", "CLAUDE.md"):
        f = repo_root / fname
        if f.exists() and f.is_file():
            try:
                sections.append(
                    f"--- {fname} ---\n" + _truncate(f.read_text(errors="replace"), 5000)
                )
                break
            except OSError:
                pass

    # File excerpts referenced by the threads
    excerpts: list[str] = []
    seen: set[str] = set()
    for t in threads:
        if not t.path or t.path in seen:
            continue
        seen.add(t.path)
        f = repo_root / t.path
        if f.exists() and f.is_file():
            with contextlib.suppress(OSError):
                excerpts.append(
                    f"=== {t.path} ===\n" + _truncate(f.read_text(errors="replace"), 4000)
                )
    if excerpts:
        sections.append("--- file excerpts ---\n" + "\n\n".join(excerpts))

    out_text = "\n\n".join(sections)
    return _truncate(out_text, budget)


def _truncate(s: str, max_bytes: int) -> str:
    raw = s.encode("utf-8")
    if len(raw) <= max_bytes:
        return s
    return raw[:max_bytes].decode("utf-8", errors="replace") + f"\n... [truncated, original {len(raw)} bytes] ..."


def _post_validation_failure(
    gh: GitHubClient,
    pr_number: int,
    failure: CommandResult | None,
) -> None:
    """Post a PR comment summarising the failed validation command and
    apply the ``agent:failed-validation`` label.

    Both calls are best-effort — failures are logged but never raise so
    they don't block the escalation path that follows.
    """
    if failure is None:
        body = "🤖 **pr-autofix-agent** validation failed (no per-command result captured)."
    else:
        duration = f"{failure.duration_s:.1f}s"
        stderr = failure.stderr_tail or "(no stderr captured)"
        body = (
            f"🤖 **pr-autofix-agent** validation failed.\n\n"
            f"**Failed command:** `{failure.name}` "
            f"(exit {failure.exit_code}, {duration})\n\n"
            f"```\n{stderr}\n```\n"
        )
    try:
        gh.create_pr_comment(pr_number, body)
    except Exception as e:
        log.warning("Could not post validation-failure comment: %s", e)
    try:
        gh.add_labels(pr_number, ["agent:failed-validation"])
    except Exception as e:
        log.warning("Could not apply agent:failed-validation label: %s", e)


def _format_reply(patch: Patch, sha: str) -> str:
    files = "\n".join(f"- `{f.path}`" for f in patch.files)
    return (
        f"\U0001f916 **pr-autofix-agent** applied a fix in `{sha[:7]}`:\n\n"
        f"{patch.summary}\n\n"
        f"Files changed:\n{files}\n"
    )


def _loc(thread: ReviewThread) -> str:
    """Thin alias retained for call-site readability."""
    return thread.location


def _format_run_summary(report: AgentRunReport) -> str:
    """Render the end-of-run PR-level summary comment.

    Returns the empty string when there are no rounds to summarise so
    callers can short-circuit.
    """
    if not report.rounds:
        return ""
    lines: list[str] = ["## PR Autofix Agent — Run Summary"]
    for r in report.rounds:
        lines.append("")
        lines.append(f"### Round {r.round_no}")
        had_section = False
        if r.handled:
            lines.append("")
            lines.append("Handled:")
            for h in r.handled:
                lines.append(f"- `{h.location}` — {h.summary}")
            had_section = True
        if r.validation.command_results:
            lines.append("")
            lines.append("Validation:")
            for c in r.validation.command_results:
                status = "passed" if c.ok else f"failed (exit {c.exit_code})"
                lines.append(f"- {c.name}: {status}")
            had_section = True
        if r.escalated_to_human:
            lines.append("")
            lines.append("Escalated:")
            for e in r.escalated_to_human:
                lines.append(f"- `{e.location}` — {e.reason}")
            had_section = True
        if r.error:
            lines.append("")
            lines.append(f"Error: {r.error}")
            had_section = True
        if not had_section:
            lines.append("")
            lines.append("(no actions taken this round)")
    if report.escalated:
        reason = report.escalation_reason.value if report.escalation_reason else "unknown"
        n = len(report.final_unresolved_thread_ids)
        lines.append("")
        lines.append("---")
        lines.append(f"Final status: escalated (`{reason}`) — {n} unresolved thread(s).")
    return "\n".join(lines)


def _post_run_summary(gh: GitHubClient, pr_number: int, report: AgentRunReport) -> None:
    """Post the end-of-run summary comment. Best-effort + no-op on empty."""
    body = _format_run_summary(report)
    if not body:
        return
    try:
        gh.create_pr_comment(pr_number, body)
    except Exception as e:
        log.warning("Could not post run-summary comment: %s", e)


def _finish(gh: GitHubClient, pr_number: int, state: AgentState) -> int:
    """Post the run summary, close the GitHub client, and return exit code 0.

    Single funnel for every clean exit path in :func:`main` so we can't
    accidentally drop the summary or the ``agent:autofixed`` label on one
    branch. The label is applied once if at least one round produced a
    commit; additive — never removed, coexists with escalation labels.
    """
    _post_run_summary(gh, pr_number, state.report)
    if any(r.commit_sha for r in state.report.rounds):
        try:
            gh.add_labels(pr_number, ["agent:autofixed"])
        except Exception as e:
            log.warning("Could not apply agent:autofixed label: %s", e)
    gh.close()
    return 0


def _post_clean(gh: GitHubClient, pr_number: int) -> None:
    """Post the all-clean comment and apply the ``agent:clean`` label.

    Called when the agent finds no unresolved Bugbot threads on the PR.
    Both calls are best-effort.
    """
    try:
        gh.create_pr_comment(
            pr_number,
            "🤖 **pr-autofix-agent** found no unresolved Cursor Bugbot comments.",
        )
    except Exception as e:
        log.warning("Could not post clean comment: %s", e)
    try:
        gh.add_labels(pr_number, ["agent:clean"])
    except Exception as e:
        log.warning("Could not apply agent:clean label: %s", e)


def _summarize_human_threads(skipped: list[tuple[ReviewThread, str]]) -> str:
    """Build the all-needs-human escalation comment body.

    Lists up to 20 NEEDS_HUMAN threads with their path/line and the rule
    or LLM reason. The comment is a single Markdown block; truncated
    with a "... and N more" tail past 20 entries.
    """
    lines = [
        "🤖 **pr-autofix-agent** is escalating because every Bugbot comment "
        "this round needs human review.",
        "",
        "Threads:",
    ]
    cap = 20
    for thread, reason in skipped[:cap]:
        # Wrap path-bearing locations in backticks; keep the bare "(no path)"
        # marker readable.
        location = f"`{thread.location}`" if thread.path else thread.location
        lines.append(f"- {location} — {reason}")
    if len(skipped) > cap:
        lines.append(f"... and {len(skipped) - cap} more.")
    return "\n".join(lines)


_REASON_LABEL_MAP: dict[EscalationReason, list[str]] = {
    EscalationReason.MAX_ROUNDS: ["agent:max-rounds"],
}


def _escalate(
    gh: GitHubClient,
    inputs: WorkflowInputs,
    state: AgentState,
    reason: EscalationReason,
    unresolved: list[str],
) -> None:
    state.escalate(reason, unresolved)
    # Phase 14: every human-needed escalation also carries the fixed
    # `agent:needs-human` label alongside the configurable
    # ``inputs.needs_human_label`` (defaults to ``needs-human``). GitHub
    # add_labels is additive so duplicates collapse.
    labels = [
        inputs.needs_human_label,
        "agent:needs-human",
        *_REASON_LABEL_MAP.get(reason, []),
    ]
    try:
        gh.add_labels(inputs.pr_number, labels)
        gh.create_pr_comment(
            inputs.pr_number,
            f"\U0001f916 **pr-autofix-agent** is escalating to a human reviewer.\n\n"
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
