# pr-autofix-agent

Autonomous GitHub PR autofix agent for Cursor Bugbot review threads.

It runs as a **reusable GitHub Actions workflow** plus a small Python agent that:

1. Lists unresolved Bugbot threads on a PR (GraphQL).
2. Triages each thread: rule-layer skip for architectural / excluded paths, then an LLM classifier (Anthropic Claude) with a confidence threshold.
3. Generates minimal patches for auto-fixable threads, restricted to a configured file budget and forbidden-path allowlist.
4. Runs target-repo validators (`.pr-autofix.yml` → `validate:`); reverts the patch on failure and feeds the failure into the next round.
5. Commits, pushes, replies to the thread, and resolves it via GraphQL.
6. Loops up to `max_rounds` (default 5); on hitting the limit or repeated identical validation failures it labels the PR `needs-human` and posts an escalation comment.

## Installing in a target repo

1. Create `.pr-autofix.yml` at the repo root (see `.pr-autofix.yml.example`).
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
    secrets:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

The reusable workflow uses the standard `GITHUB_TOKEN`. **It never uses `pull_request_target`** — patches are applied to the PR head ref under least-privilege scopes.

## Safety rails

- Patch path allowlist; rejects `.github/workflows/*`, lockfiles, `.pr-autofix.yml` itself, anything in `exclude_paths`, and any `..` traversal.
- `max_files_per_patch` cap (default 5).
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
| `max_rounds` | `5` | Max autofix rounds before escalation |
| `model` | `claude-sonnet-4-6` | Anthropic model id |
| `dry_run` | `false` | Generate patches but skip commit/push |
| `needs_human_label` | `needs-human` | Label applied on escalation |
| `confidence_threshold` | `0.7` | Minimum classifier confidence to attempt a fix |

`.pr-autofix.yml` (in target repo):

| Key | Description |
| --- | --- |
| `validate` | Ordered list of `{name, run}` shell commands; first failure stops the round |
| `exclude_paths` | fnmatch globs the agent must never touch |
| `max_files_per_patch` | Hard cap on files modified per patch (default 5) |
| `bugbot_logins` | Author logins counted as Bugbot (default `["cursor[bot]"]`) |

## Local dev

```sh
pip install -e ".[dev]"
pytest -x
ruff check .
mypy pr_agent
```

Dry run against a real PR (read-only):

```sh
GITHUB_TOKEN=... ANTHROPIC_API_KEY=... \
  python -m pr_agent.run --repo owner/repo --pr 123 --dry-run
```
