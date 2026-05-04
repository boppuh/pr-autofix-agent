"""Smoke tests for the dataclass models and their JSON factories."""

from __future__ import annotations

import pytest

from pr_agent.models import (
    Classification,
    Patch,
    ReviewComment,
    ReviewThread,
    WorkflowInputs,
)

# --- ReviewThread / ReviewComment -----------------------------------------


def _comment(**overrides) -> ReviewComment:
    base = dict(
        id="C1",
        author="cursor",
        body="missing null check",
        path="src/foo.py",
        line=10,
        diff_hunk=None,
        created_at="2024-01-01T00:00:00Z",
    )
    base.update(overrides)
    return ReviewComment(**base)  # type: ignore[arg-type]


def test_review_thread_path_and_line_proxy_root_comment():
    t = ReviewThread(id="T1", is_resolved=False, comments=[_comment()])
    assert t.path == "src/foo.py"
    assert t.line == 10
    assert t.root_comment.body == "missing null check"


def test_review_thread_body_text_joins_all_comments():
    t = ReviewThread(
        id="T1",
        is_resolved=False,
        comments=[_comment(body="a"), _comment(id="C2", body="b")],
    )
    assert t.body_text == "a\n\nb"


# --- Classification -------------------------------------------------------


def test_classification_rejects_unknown_category():
    with pytest.raises(ValueError, match="must be one of"):
        Classification(thread_id="T1", category="MAYBE", reason="r", confidence=0.5)  # type: ignore[arg-type]


def test_classification_rejects_out_of_range_confidence():
    with pytest.raises(ValueError, match="confidence"):
        Classification(thread_id="T1", category="AUTO_FIX", reason="r", confidence=1.5)


def test_classification_from_json_handles_legacy_label():
    cls = Classification.from_json(
        {"label": "human_required", "confidence": 0.9, "reason": "arch"},
        thread_id="T1",
    )
    assert cls.category == "NEEDS_HUMAN"
    assert cls.thread_id == "T1"


def test_classification_from_json_handles_new_category():
    cls = Classification.from_json(
        {"category": "IGNORE", "confidence": 0.95, "reason": "LGTM"},
        thread_id="T2",
    )
    assert cls.category == "IGNORE"


def test_classification_from_json_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown classification category"):
        Classification.from_json({"category": "POTATO"}, thread_id="T1")


def test_classification_from_json_null_confidence_raises_value_error():
    """Regression: ``"confidence": null`` from the model used to TypeError out
    of float(None). It must surface as ValueError so the parse_classification
    fallback handler (catches LLMResponseError, ValueError) absorbs it."""
    with pytest.raises(ValueError, match="confidence"):
        Classification.from_json(
            {"category": "AUTO_FIX", "confidence": None, "reason": "x"},
            thread_id="T1",
        )


def test_classification_from_json_list_confidence_raises_value_error():
    with pytest.raises(ValueError, match="confidence"):
        Classification.from_json(
            {"category": "AUTO_FIX", "confidence": [0.9], "reason": "x"},
            thread_id="T1",
        )


# --- Patch ----------------------------------------------------------------


def test_patch_from_json_round_trip():
    p = Patch.from_json(
        {
            "summary": "fix null",
            "files": [
                {"path": "src/a.py", "new_content": "x", "rationale": "guard"},
                {"path": "src/b.py", "new_content": "y", "rationale": ""},
            ],
        },
        thread_id="T1",
    )
    assert p.thread_id == "T1"
    assert p.summary == "fix null"
    assert p.touched_paths() == ["src/a.py", "src/b.py"]


def test_patch_from_json_rejects_non_mapping_file():
    with pytest.raises(ValueError, match="must be a mapping"):
        Patch.from_json(
            {"summary": "x", "files": ["just a string"]}, thread_id="T1"
        )


def test_patch_from_json_empty_files_ok():
    p = Patch.from_json({"summary": "no fix needed"}, thread_id="T1")
    assert p.files == []


# --- WorkflowInputs -------------------------------------------------------


def test_workflow_inputs_validates_provider():
    with pytest.raises(ValueError, match="provider"):
        WorkflowInputs(pr_number=1, repo_full_name="o/r", provider="cohere")  # type: ignore[arg-type]


def test_workflow_inputs_validates_log_level():
    with pytest.raises(ValueError, match="log_level"):
        WorkflowInputs(pr_number=1, repo_full_name="o/r", log_level="VERBOSE")  # type: ignore[arg-type]


def test_workflow_inputs_defaults():
    w = WorkflowInputs(pr_number=42, repo_full_name="o/r")
    assert w.provider == "anthropic"
    assert w.model is None
    assert w.confidence_threshold == 0.7
    assert w.log_level == "INFO"
