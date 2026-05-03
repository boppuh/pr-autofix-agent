# pr-autofix-agent

Autonomous GitHub PR autofix agent for Cursor Bugbot review threads.

It runs as a **reusable GitHub Actions workflow** plus a small Python agent that:

1. Lists unresolved Bugbot threads on a PR (GraphQL).
2. Triages each thread: rule-layer skip for architectural / protected paths, then an LLM classifier (Anthropic Claude or OpenAI GPT/Codex — pluggable) with a confidence threshold.
3. Generates minimal patches for auto-fixable threads, restricted to a configured file/line budget and forbidden-path allowlist.
4. Runs target-repo validators (`.pr-agent.yml` → `validation.commands`); reverts the patch on failure and feeds the failure into the next round.
5. Commits, pushes, replies to the thread, and resolves it via GraphQL.
6. Loops up to `safety.max_rounds` (default 5); on hitting the limit, exhausting `safety.max_runtime_minutes`, or repeated identical validation failures it labels the PR `needs-human` and posts an escalation comment.

## Installing in a target repo

1. Optionally create `.pr-agent.yml` at the repo root (see `.pr-agent.yml.example`). Defaults: npm validation commands, standard `protected_paths`, Bugbot logins `cursor` / `bugbot` / `cursor-bugbot`.
2. Add a calling workflow:

```yaml
# .github/workflows/pr-autofix.yml
name: PR Autofix

on:
  pull_request_review_comment:
    types: [created]
  workflow_dispatch:
    inputs:
      pr_number:
        required: true
        type: number

permissions:
  contents: write
  pull-requests: write
  issues: write

jobs:
  autofix:
    uses: <your-org>/pr-autofix-agent/.github/workflows/pr-autofix-agent.yml@v1
    with:
      pr_number: ${{ github.event.pull_request.number || inputs.pr_number }}
    with:
      pr_number: ${{ github.event.pull_request.number || inputs.pr_number }}
      provider: anthropic  # or "openai"
    secrets:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

> Set whichever secret matches your `provider:`. The other can be omitted.

The reusable workflow uses the standard `GITHUB_TOKEN`. **It never uses `pull_request_target`** — patches are applied to the PR head ref under least-privilege scopes.

## Safety rails

- Patch path allowlist; rejects `.github/workflows/*`, lockfiles, `.pr-agent.yml` itself, anything in `protected_paths`, and any `..` traversal.
- `safety.max_files_touched` cap (default 15).
- `safety.max_patch_lines` cap on total new content (default 800).
- `safety.max_comments_per_round` cap on threads handled per round (default 20).
- `safety.max_runtime_minutes` overall budget (default 20); escalates on exhaustion.
- Architectural-keyword regex skip (rule layer, not LLM-trustable).
- Confidence threshold on LLM triage (default 0.7).
- No push if any validator fails — the patch is reverted.
- Idempotency: never re-replies to the same `(thread_id, comment_hash)` twice.
- Concurrency group keyed on PR number.
- Repeated identical validation failures trigger immediate escalation.

## Configuration

| Workflow input | Default | Description |
| --- | --- | --- |
| `pr_number` | _required_ | PR to operate on |
| `max_rounds` | `5` | Cap on autofix rounds (further capped by `safety.max_rounds`) |
| `provider` | `anthropic` | LLM provider — `anthropic` or `openai` |
| `model` | provider default | Provider-specific model id (defaults: `claude-sonnet-4-6` / `gpt-5-codex`) |
| `dry_run` | `false` | Generate patches but skip commit/push |
| `needs_human_label` | `needs-human` | Label applied on escalation |
| `confidence_threshold` | `0.7` | Minimum classifier confidence to attempt a fix |

`.pr-agent.yml` (in target repo, all sections optional):

| Key | Description |
| --- | --- |
| `validation.commands` | Ordered list of strings or `{name, run}` mappings; first failure stops the round (default: `npm test`, `npm run lint`, `npm run typecheck`) |
| `safety.max_rounds` | Round cap (default 5) |
| `safety.max_comments_per_round` | Threads handled per round (default 20) |
| `safety.max_patch_lines` | Total new lines per patch (default 800) |
| `safety.max_files_touched` | Files per patch (default 15) |
| `safety.max_runtime_minutes` | Hard wall-clock budget (default 20) |
| `protected_paths` | Trailing-slash dir prefixes or fnmatch globs the agent must never touch |
| `bugbot_logins` | Author logins counted as Bugbot (default `["cursor", "bugbot", "cursor-bugbot"]`) |

## Local dev

```sh
pip install -e ".[dev]"
pytest -x
ruff check .
mypy pr_agent
```

Dry run against a real PR (read-only):

```sh
# Anthropic (default)
GITHUB_TOKEN=... ANTHROPIC_API_KEY=... \
  python -m pr_agent.run --repo owner/repo --pr 123 --dry-run

# OpenAI
GITHUB_TOKEN=... OPENAI_API_KEY=... \
  python -m pr_agent.run --repo owner/repo --pr 123 --provider openai --dry-run
```
