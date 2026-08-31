"""Candidate extraction and the checks that can fail a run.

The corpus under cache/ is the fixture. These tests pin two things the rewrite
depends on: that the candidate filter never loses a heading the book prints
(the LLM can only choose from candidates, so an absent boundary is unreachable
no matter how good the model is), and that the residue check separates the one
book whose structure is wrong from the ten whose structure is right.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gutenberg_reader import candidates, structure_checks
from gutenberg_reader.models import ChapterInfo
from tests.fixtures import body_lines as _fixture_body

CACHE = Path(__file__).resolve().parent.parent / "cache"

# Every cached book except 6400, whose shipped discovery.json is the known-bad
# structure these checks exist to reject.
CORRECT_BOOKS = ["1184", "1260", "1342", "1661", "1727", "2131", "2641", "2701", "3296", "37106"]


def _body(book_id: str) -> list[str]:
    return _fixture_body(book_id)


def _discovered(book_id: str) -> tuple[list[ChapterInfo], int]:
    """Chapters from the committed golden, not from cache/02-discovery.

    That file is rewritten by any run and deleted by any cache clear, so reading
    it made these tests skip silently the moment the cache was cleaned — which is
    precisely the disappearing-coverage problem the goldens exist to end. The
    golden stores body-relative line numbers, so body_start is 0 here.
    """
    from tests.fixtures import golden

    g = golden(book_id)
    body = _body(book_id)
    entries = g.get("chapters") or [
        {**c, "kind": "body", "words": 0} for c in g["required_body"]
    ]
    infos: list[ChapterInfo] = []
    for i, c in enumerate(entries):
        end = (entries[i + 1]["line"] - 1) if i + 1 < len(entries) else len(body) - 1
        infos.append(ChapterInfo(
            number=i + 1, title=c["title"], start_line=c["line"] + 1,
            end_line=end + 1, word_count=c.get("words", 0),
            start_marker="" if c.get("synthetic") else c["title"],
            kind=c.get("kind", "body"),
        ))
    return infos, 0


def _info(number, words, start=None, end=None, title="T"):
    start = number if start is None else start
    return ChapterInfo(number=number, title=title, start_line=start,
                       end_line=start if end is None else end, word_count=words)


# ── Candidate extraction ─────────────────────────────────────────────────────

# 2131 is excluded: Herodotus prints no chapter heading anywhere, so it has no
# printed heading to find. That is asserted directly below instead.
@pytest.mark.parametrize("book_id", [b for b in CORRECT_BOOKS if b != "2131"] + ["6400"])
def test_every_printed_heading_is_a_candidate(book_id):
    """The model can only choose from candidates, so a heading that is not one
    is unreachable however good the model is. This is the real contract on
    candidates.extract, and it runs without an LLM or a network."""
    body = _body(book_id)
    chapters, body_start = _discovered(book_id)
    lines = {c.line for c in candidates.extract(body)}

    # A synthetic chapter has no heading line to find: the detector invented it
    # because the book prints none (P&P's chapter one, 2131's parts). Those
    # carry an empty start_marker, which is exactly what marks them synthetic.
    printed = [ci for ci in chapters if ci.start_marker]
    missed = [
        (ci.number, ci.title) for ci in printed
        if ci.start_line - body_start - 1 not in lines
    ]
    assert not missed, f"{book_id}: printed headings absent from candidates: {missed}"
    assert printed, f"{book_id}: no printed headings to check"


def test_herodotus_prints_no_headings():
    """2131's 13 chapters are all synthetic parts cut at paragraph boundaries —
    the source carries no heading to detect. The candidate list is nearly empty,
    which is the signal a structure pass should read as "no chapter structure"
    rather than as a reason to split on the title page."""
    chapters, _ = _discovered("2131")
    assert chapters and not any(ci.start_marker for ci in chapters)
    assert len(candidates.extract(_body("2131"))) < 20


@pytest.mark.parametrize("book_id", CORRECT_BOOKS + ["6400"])
def test_candidate_list_fits_one_prompt(book_id):
    """The whole-book view is the point; it has to fit in one call. ~1.4 tokens
    per word measured against the served tokenizer."""
    rendered = candidates.render(candidates.extract(_body(book_id)))
    assert len(rendered.split()) * 1.4 < 24_000


def test_two_line_heading_is_one_candidate():
    """2641 prints the number directly above the title with no blank between,
    so a line-based filter scores 0 of 19 on A Room with a View."""
    body = ["", "Chapter I", "The Bertolini", "", "Prose begins here and runs on."]
    cands = candidates.extract(body)
    assert cands[0].text == "Chapter I / The Bertolini"
    assert cands[0].n_lines == 2


def test_dialogue_and_prose_are_not_candidates():
    body = ['"Go now," he said.', "", "CHAPTER I.", "",
            "he walked down the road and did not look back at all"]
    texts = [c.text for c in candidates.extract(body)]
    assert texts == ["CHAPTER I."]


def test_shape_separates_a_series_from_a_one_off():
    """The whole 6400 argument: no regex can tell 'A.  SALVIUS OTHO.' from the
    inscription 'M. AGRIPPA. L. F. COS: TERTIUM. FECIT.' — same shape to a
    pattern. A shape census can, because eleven others match one and none the
    other."""
    body = _body("6400")
    by_text = {c.text: c for c in candidates.extract(body)}
    otho = by_text["A.  SALVIUS OTHO."]
    agrippa = by_text["M. AGRIPPA. L. F. COS: TERTIUM. FECIT."]
    assert "shape:A A A" in otho.flags
    assert "shape:A A A" not in agrippa.flags
    # And the regex that caused the defect still votes for the inscription —
    # it is kept as evidence rather than as a decision.
    assert "regex:chapter" in agrippa.flags


# ── Checks ───────────────────────────────────────────────────────────────────

def test_tiling_gap_fails():
    chapters = [_info(1, 500, start=1, end=100), _info(2, 500, start=140, end=200)]
    codes = [f.code for f in structure_checks.check(chapters)]
    assert "tiling" in codes


def test_degenerate_toc_fails():
    """37106 shipped 46 chapters holding only their own title."""
    chapters = [_info(i, 3) for i in range(1, 11)] + [_info(11, 188_656)]
    fails = [f for f in structure_checks.check(chapters) if f.severity == "fail"]
    assert any(f.code == "degenerate" for f in fails)


def test_pair_lopsided_fails():
    codes = [f.code for f in structure_checks.check([_info(1, 647), _info(2, 36_916)])]
    assert "pair_lopsided" in codes
    assert not structure_checks.check([_info(1, 5_000), _info(2, 7_000)])


def test_size_outlier_only_warns():
    """Moby-Dick's Cetology is genuinely several times the median. A long
    chapter is a long chapter; this must never stop a run."""
    chapters = [_info(i, 1_000) for i in range(1, 10)] + [_info(10, 8_000)]
    findings = structure_checks.check(chapters)
    assert [f.code for f in findings] == ["size_outlier"]
    assert all(f.severity == "warn" for f in findings)


@pytest.mark.parametrize("book_id", CORRECT_BOOKS)
def test_correct_books_pass_clean(book_id):
    """Zero false positives is what makes failing hard affordable."""
    chapters, body_start = _discovered(book_id)
    cands = candidates.extract(_body(book_id))
    fails = [f for f in structure_checks.check(chapters, cands, body_start)
             if f.severity == "fail"]
    assert not fails, [str(f) for f in fails]


# The structure PG 6400 actually shipped with, recorded here rather than read
# from cache/: 9,984 / 173,461 / 29,196 words over three chapters, because the
# bare-numeral pattern matched the Latin initials in "D. OCTAVIUS CAESAR
# AUGUSTUS." and in the monumental inscription "M. AGRIPPA. L. F. COS: TERTIUM.
# FECIT." Keeping it as data means the test still fails the bad structure once
# the cache holds a good one.
SHIPPED_6400 = [
    (1, "Chapter I", 1536, 2461, 9_984),
    (2, "D. OCTAVIUS CAESAR AUGUSTUS.", 2462, 18702, 173_461),
    (3, "M. AGRIPPA. L. F. COS: TERTIUM. FECIT.", 18703, 22459, 29_196),
]


def test_6400_shipped_structure_is_rejected():
    """The live failure. Nothing acted on the 17x size warning, so it shipped.
    The twelve Caesars were still sitting in that middle chapter as
    unclassified, evenly spaced, same-shaped blocks — which is detectable."""
    chapters = [
        ChapterInfo(number=n, title=t, start_line=a, end_line=b, word_count=w,
                    start_marker=t)
        for n, t, a, b, w in SHIPPED_6400
    ]
    cands = candidates.extract(_body("6400"))
    fails = [f for f in structure_checks.check(chapters, cands, 29)
             if f.severity == "fail"]

    assert fails, "6400's shipped structure must not pass"
    residue = [f for f in fails if f.code == "unexplained_structure"]
    assert residue
    found = {t for f in residue for t in f.evidence.get("examples", [])}
    assert any("CAESAR" in t.upper() for t in found), found


def test_residue_catches_a_swallowed_series_not_every_missing_division():
    """The sensitivity boundary, pinned honestly.

    The residue check finds an evenly spaced series of same-shaped headings
    swallowed into one chapter — the failure PG 6400 actually shipped, with
    eleven Caesars inside a 173,461-word chapter. It does NOT find every missing
    division: build a structure from only the twelve Caesars and the appended
    Lives of the Grammarians, Rhetoricians and Poets are left inside the last
    chapter without complaint, because they are irregular in shape and as little
    as 184 lines apart. Catching those is the structure pass's job, not the
    check's; the check is the backstop against the catastrophic case."""
    from gutenberg_reader.stages.s02_discovery import _build_chapter_infos
    from tests.fixtures import golden

    body = _body("6400")
    body_start = 31
    g = golden("6400")
    raw = [
        {"number": i + 1, "title": c["title"], "start_line": c["line"] + 1,
         "start_marker": c["title"], "kind": "body"}
        for i, c in enumerate(sorted(g["required_body"], key=lambda c: c["line"]))
    ]
    chapters = _build_chapter_infos(raw, body, body_start)
    fails = [f for f in structure_checks.check(
        chapters, candidates.extract(body), body_start) if f.severity == "fail"]
    assert not fails, [str(f) for f in fails]


def test_page_markers_are_never_residue_evidence():
    """A bare parenthesised number recurring every few hundred lines is a page
    marker, not a missed chapter. PG 6400 prints them throughout, and a long
    chapter can hold five 200+ lines apart with nothing wrong."""
    assert "shape:9" in structure_checks.RESIDUE_IGNORED_SHAPES
