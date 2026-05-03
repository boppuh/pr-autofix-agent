from __future__ import annotations

import httpx
import pytest
import respx

from pr_agent.github_client import (
    GitHubClient,
    add_labels,
    create_pr_comment,
    get_check_runs,
    get_pr,
    get_pr_diff,
    get_unresolved_bugbot_threads,
    reply_to_thread,
    resolve_thread,
)


@pytest.fixture
def token():
    return "ghp_test"


def _pr_payload(threads, has_next=False, end_cursor=None):
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "id": "PR_1",
                    "title": "feat: foo",
                    "body": "describes the change",
                    "headRefName": "feature/foo",
                    "headRefOid": "deadbeef",
                    "baseRefName": "main",
                    "reviewThreads": {
                        "pageInfo": {"hasNextPage": has_next, "endCursor": end_cursor},
                        "nodes": threads,
                    },
                }
            }
        }
    }


def _thread_node(*, id_, login, resolved=False, body="x", line=10):
    return {
        "id": id_,
        "isResolved": resolved,
        "isOutdated": False,
        "path": "src/foo.py",
        "line": line,
        "comments": {
            "nodes": [
                {
                    "id": "C_1",
                    "databaseId": 12345,
                    "author": {"login": login},
                    "body": body,
                    "path": "src/foo.py",
                    "line": line,
                    "originalLine": line,
                    "diffHunk": "@@ -1 +1 @@\n-x\n+y",
                    "createdAt": "2024-01-01T00:00:00Z",
                }
            ]
        },
    }


def test_get_pr_parses_metadata_and_threads(token):
    with respx.mock(base_url="https://api.github.com") as mock:
        mock.post("/graphql").mock(
            return_value=httpx.Response(
                200,
                json=_pr_payload([_thread_node(id_="T1", login="cursor")]),
            )
        )
        pr = get_pr("o", "r", 1, token=token)
        assert pr.title == "feat: foo"
        assert pr.head_ref_name == "feature/foo"
        assert pr.head_ref_oid == "deadbeef"
        assert pr.base_ref_name == "main"
        assert len(pr.threads) == 1
        c = pr.threads[0].comments[0]
        assert c.author_login == "cursor"
        assert c.original_line == 10
        assert c.diff_hunk and "+y" in c.diff_hunk


def test_get_pr_paginates_threads(token):
    page1 = _pr_payload([_thread_node(id_="T1", login="cursor")], has_next=True, end_cursor="C")
    page2 = _pr_payload([_thread_node(id_="T2", login="other")], has_next=False)
    with respx.mock(base_url="https://api.github.com") as mock:
        mock.post("/graphql").mock(
            side_effect=[httpx.Response(200, json=page1), httpx.Response(200, json=page2)]
        )
        pr = get_pr("o", "r", 1, token=token)
        assert [t.id for t in pr.threads] == ["T1", "T2"]


def test_unresolved_bugbot_filter(token):
    threads = [
        _thread_node(id_="T1", login="cursor"),
        _thread_node(id_="T2", login="cursor", resolved=True),
        _thread_node(id_="T3", login="some-other-bot"),
        _thread_node(id_="T4", login="bugbot"),
    ]
    with respx.mock(base_url="https://api.github.com") as mock:
        mock.post("/graphql").mock(return_value=httpx.Response(200, json=_pr_payload(threads)))
        out = get_unresolved_bugbot_threads(
            "o", "r", 1, ["cursor", "bugbot"], token=token
        )
        assert [t.id for t in out] == ["T1", "T4"]


def test_get_pr_diff_truncates(token):
    big = "diff --git a/x b/x\n" + ("+line\n" * 10000)
    with respx.mock(base_url="https://api.github.com") as mock:
        mock.get("/repos/o/r/pulls/1").mock(return_value=httpx.Response(200, text=big))
        out = get_pr_diff("o", "r", 1, token=token, max_bytes=200)
        assert len(out) <= 300  # 200 + truncation marker
        assert "truncated" in out


def test_resolve_thread_sends_mutation(token):
    with respx.mock(base_url="https://api.github.com") as mock:
        route = mock.post("/graphql").mock(
            return_value=httpx.Response(
                200,
                json={"data": {"resolveReviewThread": {"thread": {"id": "T1", "isResolved": True}}}},
            )
        )
        resolve_thread("T1", token=token)
        body = route.calls.last.request.read().decode()
        assert "resolveReviewThread" in body
        assert "T1" in body


def test_reply_to_thread_skips_non_numeric(token, caplog):
    with respx.mock(base_url="https://api.github.com", assert_all_called=False) as mock:
        route = mock.post("/repos/o/r/pulls/1/comments/1/replies")
        reply_to_thread("o", "r", 1, "not-numeric", "hi", token=token)
        assert not route.called


def test_reply_to_thread_posts_reply(token):
    with respx.mock(base_url="https://api.github.com") as mock:
        route = mock.post("/repos/o/r/pulls/1/comments/12345/replies").mock(
            return_value=httpx.Response(201, json={"id": 999})
        )
        reply_to_thread("o", "r", 1, "12345", "hello", token=token)
        assert route.called


def test_create_pr_comment(token):
    with respx.mock(base_url="https://api.github.com") as mock:
        route = mock.post("/repos/o/r/issues/1/comments").mock(
            return_value=httpx.Response(201, json={"id": 1})
        )
        create_pr_comment("o", "r", 1, "body", token=token)
        assert route.called


def test_add_labels_noop_on_empty(token):
    with respx.mock(base_url="https://api.github.com", assert_all_called=False) as mock:
        route = mock.post("/repos/o/r/issues/1/labels")
        add_labels("o", "r", 1, [], token=token)
        assert not route.called


def test_add_labels_posts(token):
    with respx.mock(base_url="https://api.github.com") as mock:
        route = mock.post("/repos/o/r/issues/1/labels").mock(
            return_value=httpx.Response(200, json=[{"name": "needs-human"}])
        )
        add_labels("o", "r", 1, ["needs-human"], token=token)
        assert route.called


def test_get_check_runs(token):
    with respx.mock(base_url="https://api.github.com") as mock:
        mock.get("/repos/o/r/commits/abc/check-runs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "check_runs": [
                        {"name": "test", "status": "completed", "conclusion": "success"},
                        {"name": "lint", "status": "in_progress", "conclusion": None},
                    ]
                },
            )
        )
        runs = get_check_runs("o", "r", "abc", token=token)
        assert len(runs) == 2
        assert runs[0].name == "test"
        assert runs[0].conclusion == "success"
        assert runs[1].conclusion is None


def test_resolve_thread_retries_through_transient_errors(token):
    """Regression: a transient HTTPError on the first attempt must not close
    the client mid-retry. Tenacity should reuse the client and the second
    attempt should succeed (not blow up with 'client has been closed')."""
    success = {
        "data": {"resolveReviewThread": {"thread": {"id": "T1", "isResolved": True}}}
    }
    with respx.mock(base_url="https://api.github.com") as mock:
        route = mock.post("/graphql").mock(
            side_effect=[
                httpx.Response(502, text="bad gateway"),
                httpx.Response(200, json=success),
            ]
        )
        resolve_thread("T1", token=token)
        assert route.call_count == 2


def test_class_wrapper_delegates(token):
    with respx.mock(base_url="https://api.github.com") as mock:
        mock.post("/graphql").mock(
            return_value=httpx.Response(
                200, json=_pr_payload([_thread_node(id_="T1", login="cursor")])
            )
        )
        c = GitHubClient.from_full_name(token, "o/r")
        pr = c.get_pr(1)
        assert pr.title == "feat: foo"
        c.close()
