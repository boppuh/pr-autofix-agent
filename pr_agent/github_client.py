from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from github import Github
from github.PullRequest import PullRequest
from github.Repository import Repository
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .models import BugbotComment, ReviewThread

log = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.github.com/graphql"

_THREADS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 50, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          isResolved
          isOutdated
          path
          line
          comments(first: 50) {
            nodes {
              id
              databaseId
              body
              createdAt
              author { login }
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


class GitHubClient:
    def __init__(self, token: str, repo_full_name: str, bugbot_logins: list[str]):
        self._token = token
        self._repo_full_name = repo_full_name
        self._bugbot_logins = {login.lower() for login in bugbot_logins}
        self._gh = Github(token)
        self._http = httpx.Client(
            headers={
                "Authorization": f"bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "pr-autofix-agent",
            },
            timeout=httpx.Timeout(30.0),
        )
        self._repo: Repository | None = None

    @property
    def repo(self) -> Repository:
        if self._repo is None:
            self._repo = self._gh.get_repo(self._repo_full_name)
        return self._repo

    def get_pull(self, number: int) -> PullRequest:
        return self.repo.get_pull(number)

    def list_unresolved_bugbot_threads(self, pr_number: int) -> list[ReviewThread]:
        owner, name = self._repo_full_name.split("/", 1)
        cursor: str | None = None
        out: list[ReviewThread] = []
        while True:
            data = self._graphql(
                _THREADS_QUERY,
                {"owner": owner, "name": name, "number": pr_number, "cursor": cursor},
            )
            pr = data["repository"]["pullRequest"]
            page = pr["reviewThreads"]
            for node in page["nodes"]:
                thread = self._thread_from_node(node)
                if thread.is_resolved or thread.is_outdated:
                    continue
                if not any(c.author_login.lower() in self._bugbot_logins for c in thread.comments):
                    continue
                out.append(thread)
            if not page["pageInfo"]["hasNextPage"]:
                break
            cursor = page["pageInfo"]["endCursor"]
        log.info("Found %d unresolved Bugbot threads on PR #%d", len(out), pr_number)
        return out

    def reply_to_thread(self, pr_number: int, root_comment_id: str, body: str) -> None:
        pr = self.get_pull(pr_number)
        try:
            db_id = int(root_comment_id)
        except ValueError:
            log.warning("Skipping reply: non-numeric comment id %s", root_comment_id)
            return
        pr.create_review_comment_reply(db_id, body)

    def resolve_thread(self, thread_id: str) -> None:
        self._graphql(_RESOLVE_MUTATION, {"threadId": thread_id})

    def add_label(self, pr_number: int, label: str) -> None:
        issue = self.repo.get_issue(pr_number)
        existing = {lbl.name for lbl in issue.labels}
        if label in existing:
            return
        issue.add_to_labels(label)

    def post_pr_comment(self, pr_number: int, body: str) -> None:
        self.repo.get_issue(pr_number).create_comment(body)

    def get_file_contents(self, path: str, ref: str) -> str | None:
        try:
            f = self.repo.get_contents(path, ref=ref)
            if isinstance(f, list):
                return None
            return f.decoded_content.decode("utf-8", errors="replace")
        except Exception:
            return None

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        resp = self._http.post(GRAPHQL_URL, json={"query": query, "variables": variables})
        resp.raise_for_status()
        payload = resp.json()
        if "errors" in payload:
            raise RuntimeError(f"GraphQL errors: {payload['errors']}")
        data: dict[str, Any] = payload["data"]
        return data

    def _thread_from_node(self, node: dict[str, Any]) -> ReviewThread:
        comments = [
            BugbotComment(
                id=str(c.get("databaseId") or c["id"]),
                author_login=(c.get("author") or {}).get("login", "") or "",
                body=c["body"],
                path=node.get("path"),
                line=node.get("line"),
                created_at=_parse_dt(c["createdAt"]),
            )
            for c in node["comments"]["nodes"]
        ]
        return ReviewThread(
            id=node["id"],
            path=node.get("path"),
            line=node.get("line"),
            is_resolved=node["isResolved"],
            is_outdated=node["isOutdated"],
            comments=comments,
        )

    def close(self) -> None:
        self._http.close()


def _parse_dt(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
