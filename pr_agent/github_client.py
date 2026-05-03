"""GitHub client — free functions plus a thin class wrapper.

The free functions are the canonical API; `GitHubClient` is a stateful
convenience that holds owner/repo/token. Both share the same httpx Client.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .models import BugbotComment, CheckRun, PullRequest, ReviewThread

log = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.github.com/graphql"
REST_URL = "https://api.github.com"

_PR_QUERY = """
query($owner: String!, $repo: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      id
      title
      body
      headRefName
      headRefOid
      baseRefName
      reviewThreads(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          isResolved
          isOutdated
          path
          line
          comments(first: 20) {
            nodes {
              id
              databaseId
              author { login }
              body
              path
              line
              originalLine
              diffHunk
              createdAt
            }
          }
        }
      }
    }
  }
}
"""

_RESOLVE_MUTATION = """
mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread { id isResolved }
  }
}
"""


def _token(token: str | None) -> str:
    if token:
        return token
    env = os.environ.get("GITHUB_TOKEN")
    if not env:
        raise RuntimeError("GITHUB_TOKEN not set")
    return env


def _http(token: str | None) -> httpx.Client:
    return httpx.Client(
        headers={
            "Authorization": f"bearer {_token(token)}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "pr-autofix-agent",
        },
        timeout=httpx.Timeout(30.0),
    )


# --- Free functions ---------------------------------------------------------


def get_pr(
    owner: str,
    repo: str,
    pr_number: int,
    *,
    token: str | None = None,
    http: httpx.Client | None = None,
) -> PullRequest:
    """Fetch PR metadata + all review threads in a single GraphQL traversal."""
    client = http or _http(token)
    own_close = http is None
    try:
        threads, pr_meta = _fetch_pr(client, owner, repo, pr_number)
        return PullRequest(
            id=pr_meta["id"],
            number=pr_number,
            title=pr_meta["title"],
            body=pr_meta.get("body") or "",
            head_ref_name=pr_meta["headRefName"],
            head_ref_oid=pr_meta["headRefOid"],
            base_ref_name=pr_meta["baseRefName"],
            threads=threads,
        )
    finally:
        if own_close:
            client.close()


def get_pr_diff(
    owner: str,
    repo: str,
    pr_number: int,
    *,
    token: str | None = None,
    http: httpx.Client | None = None,
    max_bytes: int = 64_000,
) -> str:
    """Fetch the unified diff for a PR (truncated to max_bytes)."""
    client = http or _http(token)
    own_close = http is None
    try:
        resp = client.get(
            f"{REST_URL}/repos/{owner}/{repo}/pulls/{pr_number}",
            headers={"Accept": "application/vnd.github.v3.diff"},
        )
        resp.raise_for_status()
        text = resp.text
        if len(text) <= max_bytes:
            return text
        return text[:max_bytes] + f"\n... [truncated, original {len(text)} bytes] ..."
    finally:
        if own_close:
            client.close()


def get_review_threads(
    owner: str,
    repo: str,
    pr_number: int,
    *,
    token: str | None = None,
    http: httpx.Client | None = None,
) -> list[ReviewThread]:
    """All review threads on a PR (resolved and unresolved)."""
    client = http or _http(token)
    own_close = http is None
    try:
        threads, _ = _fetch_pr(client, owner, repo, pr_number)
        return threads
    finally:
        if own_close:
            client.close()


def get_unresolved_bugbot_threads(
    owner: str,
    repo: str,
    pr_number: int,
    bugbot_logins: list[str],
    *,
    token: str | None = None,
    http: httpx.Client | None = None,
) -> list[ReviewThread]:
    """Filter threads to unresolved + authored by a Bugbot identity."""
    matches = {login.lower() for login in bugbot_logins}
    out: list[ReviewThread] = []
    for t in get_review_threads(owner, repo, pr_number, token=token, http=http):
        if t.is_resolved or t.is_outdated:
            continue
        if not any(c.author_login.lower() in matches for c in t.comments):
            continue
        out.append(t)
    log.info("Found %d unresolved Bugbot threads on %s/%s#%d", len(out), owner, repo, pr_number)
    return out


def reply_to_thread(
    owner: str,
    repo: str,
    pr_number: int,
    root_comment_id: str,
    body: str,
    *,
    token: str | None = None,
    http: httpx.Client | None = None,
) -> None:
    """Post a reply to a review thread by replying to its root review comment."""
    try:
        db_id = int(root_comment_id)
    except ValueError:
        log.warning("Skipping reply: non-numeric comment id %s", root_comment_id)
        return
    client = http or _http(token)
    own_close = http is None
    try:
        resp = client.post(
            f"{REST_URL}/repos/{owner}/{repo}/pulls/{pr_number}/comments/{db_id}/replies",
            json={"body": body},
        )
        resp.raise_for_status()
    finally:
        if own_close:
            client.close()


def resolve_thread(
    thread_id: str,
    *,
    token: str | None = None,
    http: httpx.Client | None = None,
) -> None:
    """Mark a review thread as resolved via GraphQL."""
    _graphql(http or _http(token), _RESOLVE_MUTATION, {"threadId": thread_id}, own_close=http is None)


def create_pr_comment(
    owner: str,
    repo: str,
    pr_number: int,
    body: str,
    *,
    token: str | None = None,
    http: httpx.Client | None = None,
) -> None:
    client = http or _http(token)
    own_close = http is None
    try:
        resp = client.post(
            f"{REST_URL}/repos/{owner}/{repo}/issues/{pr_number}/comments",
            json={"body": body},
        )
        resp.raise_for_status()
    finally:
        if own_close:
            client.close()


def add_labels(
    owner: str,
    repo: str,
    pr_number: int,
    labels: list[str],
    *,
    token: str | None = None,
    http: httpx.Client | None = None,
) -> None:
    """Add labels to a PR (REST endpoint is additive — already-applied labels are no-ops)."""
    if not labels:
        return
    client = http or _http(token)
    own_close = http is None
    try:
        resp = client.post(
            f"{REST_URL}/repos/{owner}/{repo}/issues/{pr_number}/labels",
            json={"labels": labels},
        )
        resp.raise_for_status()
    finally:
        if own_close:
            client.close()


def get_check_runs(
    owner: str,
    repo: str,
    sha: str,
    *,
    token: str | None = None,
    http: httpx.Client | None = None,
) -> list[CheckRun]:
    client = http or _http(token)
    own_close = http is None
    try:
        resp = client.get(
            f"{REST_URL}/repos/{owner}/{repo}/commits/{sha}/check-runs",
            headers={"Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
        payload = resp.json()
        return [
            CheckRun(
                name=run["name"],
                status=run.get("status", "completed"),
                conclusion=run.get("conclusion"),
            )
            for run in payload.get("check_runs", [])
        ]
    finally:
        if own_close:
            client.close()


# --- Class wrapper ----------------------------------------------------------


class GitHubClient:
    """Stateful convenience wrapper. All methods delegate to the free functions."""

    def __init__(self, token: str, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        self._token = token
        self._http = _http(token)

    @classmethod
    def from_full_name(cls, token: str, repo_full_name: str) -> GitHubClient:
        owner, repo = repo_full_name.split("/", 1)
        return cls(token, owner, repo)

    def get_pr(self, pr_number: int) -> PullRequest:
        return get_pr(self.owner, self.repo, pr_number, http=self._http)

    def get_pr_diff(self, pr_number: int, max_bytes: int = 64_000) -> str:
        return get_pr_diff(self.owner, self.repo, pr_number, http=self._http, max_bytes=max_bytes)

    def get_review_threads(self, pr_number: int) -> list[ReviewThread]:
        return get_review_threads(self.owner, self.repo, pr_number, http=self._http)

    def get_unresolved_bugbot_threads(
        self, pr_number: int, bugbot_logins: list[str]
    ) -> list[ReviewThread]:
        return get_unresolved_bugbot_threads(
            self.owner, self.repo, pr_number, bugbot_logins, http=self._http
        )

    def reply_to_thread(self, pr_number: int, root_comment_id: str, body: str) -> None:
        reply_to_thread(
            self.owner, self.repo, pr_number, root_comment_id, body, http=self._http
        )

    def resolve_thread(self, thread_id: str) -> None:
        resolve_thread(thread_id, http=self._http)

    def create_pr_comment(self, pr_number: int, body: str) -> None:
        create_pr_comment(self.owner, self.repo, pr_number, body, http=self._http)

    def add_labels(self, pr_number: int, labels: list[str]) -> None:
        add_labels(self.owner, self.repo, pr_number, labels, http=self._http)

    def get_check_runs(self, sha: str) -> list[CheckRun]:
        return get_check_runs(self.owner, self.repo, sha, http=self._http)

    # Convenience: single-call file fetch (used by run.py for excerpts).
    def get_file_contents(self, path: str, ref: str) -> str | None:
        try:
            resp = self._http.get(
                f"{REST_URL}/repos/{self.owner}/{self.repo}/contents/{path}",
                params={"ref": ref},
                headers={"Accept": "application/vnd.github.raw"},
            )
            if resp.status_code != 200:
                return None
            return resp.text
        except httpx.HTTPError:
            return None

    def close(self) -> None:
        self._http.close()


# --- Internals --------------------------------------------------------------


def _fetch_pr(
    client: httpx.Client, owner: str, repo: str, pr_number: int
) -> tuple[list[ReviewThread], dict[str, Any]]:
    """Fetch PR metadata and all review threads (paginated)."""
    cursor: str | None = None
    threads: list[ReviewThread] = []
    pr_meta: dict[str, Any] = {}
    while True:
        data = _graphql(
            client,
            _PR_QUERY,
            {"owner": owner, "repo": repo, "number": pr_number, "cursor": cursor},
        )
        pr = data["repository"]["pullRequest"]
        if not pr_meta:
            pr_meta = {
                "id": pr["id"],
                "title": pr["title"],
                "body": pr.get("body") or "",
                "headRefName": pr["headRefName"],
                "headRefOid": pr["headRefOid"],
                "baseRefName": pr["baseRefName"],
            }
        page = pr["reviewThreads"]
        for node in page["nodes"]:
            threads.append(_thread_from_node(node))
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    return threads, pr_meta


@retry(
    retry=retry_if_exception_type((httpx.HTTPError,)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _graphql(
    client: httpx.Client,
    query: str,
    variables: dict[str, Any],
    *,
    own_close: bool = False,
) -> dict[str, Any]:
    try:
        resp = client.post(GRAPHQL_URL, json={"query": query, "variables": variables})
        resp.raise_for_status()
        payload = resp.json()
        if "errors" in payload:
            raise RuntimeError(f"GraphQL errors: {payload['errors']}")
        data: dict[str, Any] = payload["data"]
        return data
    finally:
        if own_close:
            client.close()


def _thread_from_node(node: dict[str, Any]) -> ReviewThread:
    comments = [
        BugbotComment(
            id=str(c.get("databaseId") or c["id"]),
            author_login=(c.get("author") or {}).get("login", "") or "",
            body=c["body"],
            path=c.get("path") or node.get("path"),
            line=c.get("line") if c.get("line") is not None else node.get("line"),
            original_line=c.get("originalLine"),
            diff_hunk=c.get("diffHunk"),
            created_at=_parse_dt(c["createdAt"]),
        )
        for c in node["comments"]["nodes"]
    ]
    return ReviewThread(
        id=node["id"],
        path=node.get("path"),
        line=node.get("line"),
        is_resolved=node["isResolved"],
        is_outdated=node.get("isOutdated", False),
        comments=comments,
    )


def _parse_dt(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
