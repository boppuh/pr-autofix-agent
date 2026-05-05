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

1. **Add the secret.** In the target repo: Settings → Secrets and variables → Actions → New repository secret. Add `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` if you'll set `provider: openai`).
2. **Optional: drop `.pr-agent.yml`** at the repo root. See [`.pr-agent.yml.example`](.pr-agent.yml.example) for the full schema. Defaults: npm validation commands, standard `protected_paths`, Bugbot logins `cursor` / `bugbot` / `cursor-bugbot` / `cursor[bot]`. Only the `validation.commands` field is project-specific in practice — set them to whatever your CI runs.
3. **Add a calling workflow** that fires on Bugbot activity and delegates to the reusable workflow:

```yaml
# .github/workflows/pr-autofix.yml
name: PR Autofix

on:
  pull_request:
    types: [opened, synchronize, reopened]
  pull_request_review:
    types: [submitted]
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
    # Skip fork PRs: secrets aren't available there, and the agent cannot
    # push back to the fork's head ref.
    if: github.event_name == 'workflow_dispatch' || github.event.pull_request.head.repo.full_name == github.repository
    uses: boppuh/pr-autofix-agent/.github/workflows/pr-autofix-agent.yml@main
    with:
      pr_number: ${{ github.event.pull_request.number || inputs.pr_number }}
      provider: anthropic   # or "openai"
      # Optional: pin the agent to a specific tag/sha for reproducibility.
      # agent_ref: v1
    secrets:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

> Set whichever secret matches your `provider:`. The other can be omitted.

That's it. Bugbot review activity → workflow fires → agent triages, patches, validates, and pushes a commit back to the PR branch.

**Alternative: direct install** without the reusable workflow. If you'd rather vendor the steps into the calling repo (e.g. to customise the Python version or pre/post steps), copy [`.github/workflows/autofix-on-pr.yml`](.github/workflows/autofix-on-pr.yml) and replace `pip install -e ".[dev]"` with `pip install "git+https://github.com/boppuh/pr-autofix-agent@main"`.

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
| `agent_ref` | `main` | Git ref of `pr-autofix-agent` to install — branch, tag, or sha. Pin to a tag for reproducibility. |

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
