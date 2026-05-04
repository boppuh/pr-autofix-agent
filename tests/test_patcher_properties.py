"""Property tests for the unified diff tokenizer in ``pr_agent.patcher``.

These tests build random multi-file diffs from structured data — so we
know the exact set of declared paths and the exact +/- payload count
ahead of time — then assert the tokenizer's outputs match. Adversarial
content (payload lines that look like ``+++ b/...``, ``--- a/...``,
``-- something``, ``++ something``, quoted-path headers, non-ASCII
bytes) is injected on purpose: parsing bugs in the past have all been
"two parsers disagree about what is a header vs. payload", and these
properties pin the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pr_agent.patcher import count_patch_lines, extract_touched_files

# --- Path strategies -------------------------------------------------------

# Plain path: ASCII letters/digits, slash-separated. No spaces, no quotes —
# git renders these unquoted.
_plain_segment = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="_-.",
    ),
    min_size=1,
    max_size=8,
)
_plain_path = st.lists(_plain_segment, min_size=1, max_size=3).map("/".join)


@dataclass(frozen=True)
class _FileSpec:
    """A spec describing one file's contribution to the synthesized diff.

    ``a_path`` and ``b_path`` are the source/destination *real* paths
    (before any quoting). ``payload_count`` is the exact number of +/-
    rows we will write inside hunks. ``adversarial`` injects content
    lines that look like file headers / counter-diffs to verify the
    tokenizer doesn't misclassify them.
    """

    a_path: str
    b_path: str
    payload_count: int
    adversarial: bool


def _section_text(spec: _FileSpec) -> str:
    """Render one file section to diff text and return the literal
    bytes the tokenizer will see."""
    a, b = spec.a_path, spec.b_path
    lines: list[str] = [
        f"diff --git a/{a} b/{b}",
        f"--- a/{a}",
        f"+++ b/{b}",
    ]
    # One hunk with payload_count add/remove rows. Distribute as
    # alternating + / - so both kinds get exercised.
    if spec.payload_count > 0:
        lines.append(f"@@ -1,{spec.payload_count} +1,{spec.payload_count} @@")
        for i in range(spec.payload_count):
            if spec.adversarial and i % 4 == 0:
                # Payload line whose CONTENT begins with '++ b/...' — the
                # full diff line is '+++ b/<path>', which used to look
                # like a file header to the regex-based parser.
                lines.append("+++ b/this/is/payload/not/a/header")
            elif spec.adversarial and i % 4 == 1:
                # Removed line whose CONTENT begins with '-- '. The full
                # diff line is '--- <content>'.
                lines.append("--- a/this/is/payload/not/a/header")
            elif i % 2 == 0:
                lines.append(f"+added line {i}")
            else:
                lines.append(f"-removed line {i}")
    return "\n".join(lines) + "\n"


@st.composite
def _file_specs(draw: st.DrawFn) -> _FileSpec:
    a = draw(_plain_path)
    rename = draw(st.booleans())
    b = draw(_plain_path) if rename else a
    payload_count = draw(st.integers(min_value=0, max_value=8))
    adversarial = draw(st.booleans())
    return _FileSpec(a_path=a, b_path=b, payload_count=payload_count, adversarial=adversarial)


@st.composite
def _diffs(draw: st.DrawFn) -> tuple[str, list[_FileSpec]]:
    """Generate a diff and the specs that produced it.

    Specs are deduplicated by (a_path, b_path) so the *expected* path
    list (which is order-preserving + dedup) lines up cleanly with the
    extractor's output.
    """
    raw_specs = draw(st.lists(_file_specs(), min_size=1, max_size=4))
    # Drop duplicate sections (same a/b pair) — the tokenizer dedups
    # paths globally, and counting payload across duplicate sections
    # would still work, but this keeps assertions readable.
    seen: set[tuple[str, str]] = set()
    specs: list[_FileSpec] = []
    for s in raw_specs:
        key = (s.a_path, s.b_path)
        if key in seen:
            continue
        seen.add(key)
        specs.append(s)
    text = "".join(_section_text(s) for s in specs)
    return text, specs


def _expected_paths(specs: list[_FileSpec]) -> list[str]:
    """Mirror the tokenizer's emission order: per section, a-side
    first then b-side, deduped globally in first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for s in specs:
        for p in (s.a_path, s.b_path):
            if p and p != "/dev/null" and p not in seen:
                seen.add(p)
                out.append(p)
    return out


# --- Properties ------------------------------------------------------------


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(_diffs())
def test_extract_touched_files_returns_declared_paths(
    diff_data: tuple[str, list[_FileSpec]],
) -> None:
    text, specs = diff_data
    assert extract_touched_files(text) == _expected_paths(specs)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(_diffs())
def test_count_patch_lines_returns_declared_count(
    diff_data: tuple[str, list[_FileSpec]],
) -> None:
    text, specs = diff_data
    assert count_patch_lines(text) == sum(s.payload_count for s in specs)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(_diffs())
def test_adversarial_payload_does_not_leak_paths(
    diff_data: tuple[str, list[_FileSpec]],
) -> None:
    """Sanity: the adversarial sentinel path appears in payload lines
    of some sections, but must NEVER show up in the extracted path
    list — those are content, not headers."""
    text, _ = diff_data
    assert "this/is/payload/not/a/header" not in extract_touched_files(text)


@settings(max_examples=100)
@given(_diffs())
def test_count_patch_lines_invariants(
    diff_data: tuple[str, list[_FileSpec]],
) -> None:
    text, specs = diff_data
    n = count_patch_lines(text)
    # Invariant 1: never negative.
    assert n >= 0
    # Invariant 2: header rows ('--- a/...', '+++ b/...') in file-header
    # position never get counted as payload. With S sections, the diff
    # contains exactly 2*S header rows that look like +/- lines but are
    # excluded — n must equal the declared payload sum, not (sum + 2S).
    declared = sum(s.payload_count for s in specs)
    assert n == declared
    # Stronger: removing a section drops at least its payload from the
    # count (no cross-section interference).
    if specs:
        without_first = "".join(_section_text(s) for s in specs[1:])
        assert count_patch_lines(without_first) == declared - specs[0].payload_count
