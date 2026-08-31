"""The golden set, and the contract it places on candidate extraction."""

from __future__ import annotations

import pytest

from gutenberg_reader import candidates
from tests.fixtures import GOLDEN_BOOKS, body_lines, candidate_fixture, golden


def test_every_cached_book_has_a_golden():
    assert set(GOLDEN_BOOKS) >= {
        "1184", "1260", "1342", "1661", "1727", "2131", "2641", "2701",
        "3296", "37106", "6400",
    }


@pytest.mark.parametrize("book_id", GOLDEN_BOOKS)
def test_candidates_match_the_committed_fixture(book_id):
    """The candidate list is what a structure pass is shown, so a change to the
    filter must show up as a reviewable diff rather than as a silent shift in
    what the model sees."""
    rendered = candidates.render(candidates.extract(body_lines(book_id))) + "\n"
    assert rendered == candidate_fixture(book_id), (
        f"{book_id}: candidate extraction changed — re-run "
        "scripts/build_fixtures.py and review the diff"
    )


@pytest.mark.parametrize("book_id", GOLDEN_BOOKS)
def test_every_golden_boundary_is_reachable(book_id):
    """The contract. A structure pass can only choose from candidates, so a
    boundary that is not a candidate is unreachable however good the model is.
    Synthetic boundaries are exempt: the book prints no heading there."""
    g = golden(book_id)
    lines = {c.line for c in candidates.extract(body_lines(book_id))}

    wanted = [c for c in g.get("chapters", []) if not c.get("synthetic")]
    wanted += g.get("required_body", [])
    missed = [(c.get("title"), c["line"]) for c in wanted if c["line"] not in lines]
    assert not missed, f"{book_id}: unreachable boundaries {missed}"


@pytest.mark.parametrize("book_id", GOLDEN_BOOKS)
def test_golden_matches_the_book_it_describes(book_id):
    g = golden(book_id)
    assert len(body_lines(book_id)) == g["body_lines"]


def test_6400_golden_records_the_defect_not_the_output():
    """6400's golden is the truth from the book's own contents, not what the
    detector produces. It is what Phase 3 has to satisfy."""
    g = golden("6400")
    assert g["exact_count"] is None          # the appended Lives are unsettled
    assert len(g["required_body"]) == 12
    assert "M. AGRIPPA. L. F. COS: TERTIUM. FECIT." in g["forbidden_titles"]
    assert g["max_chapter_words"] == 30_000


def test_2131_golden_records_a_book_with_no_headings():
    g = golden("2131")
    assert g["has_chapter_structure"] is False
    assert g["exact_count"] == 13
    assert g["total_words"] == 36_916
    assert all(c["synthetic"] for c in g["chapters"])
