"""The LLM structure path, exercised without a model.

Everything between the model's answer and the chapter list is deterministic:
ordinals become candidates, candidates become titles and line numbers. That part
is where a hallucinated position or a lost chapter would actually do damage, so
it is tested directly with hand-written verdicts.
"""

from __future__ import annotations

import pytest

from gutenberg_reader import candidates
from gutenberg_reader.config import Config
from gutenberg_reader.models import SCHEMA_VERSION, DiscoveryResult, BookMetadata
from gutenberg_reader.schemas import structure_schema
from gutenberg_reader.stages.s02_discovery import _structure_to_raw, _detector_id
from tests.fixtures import body_lines


def _cfg(**kw):
    return Config(book_id="0", **kw)


def _cands(book_id):
    return candidates.extract(body_lines(book_id))


def _verdict(cands, picks, **kw):
    by_text = {c.text: c for c in cands}
    return {
        "work_type": kw.get("work_type", "history"),
        "has_chapter_structure": kw.get("has_chapter_structure", True),
        "body_starts_before_first_heading": kw.get("body_first", False),
        "headings": [{"ordinal": by_text[t].ordinal, "kind": k} for t, k in picks],
    }


# ── Schema ───────────────────────────────────────────────────────────────────

def test_ordinal_is_bounded_by_the_candidate_count():
    """A bounded integer makes a hallucinated position impossible, the way the
    speaker enum makes an invented character impossible. The model never emits a
    line number or a title at all."""
    s = structure_schema(351)
    item = s["properties"]["headings"]["items"]["properties"]
    assert item["ordinal"] == {"type": "integer", "minimum": 0, "maximum": 350}
    assert set(item) == {"ordinal", "kind"}
    assert "title" not in item and "start_line" not in item


def test_schema_survives_a_book_with_no_candidates():
    assert structure_schema(0)["properties"]["headings"]["items"]["properties"]["ordinal"]["maximum"] == 0


# ── Ordinals to chapters ─────────────────────────────────────────────────────

def test_titles_come_from_the_book_not_the_model():
    cands = _cands("6400")
    raw = _structure_to_raw(
        _verdict(cands, [("CAIUS JULIUS CASAR.", "body"),
                         ("A.  SALVIUS OTHO.", "body")]),
        cands, body_lines("6400"), _cfg())
    assert [r["title"] for r in raw] == ["CAIUS JULIUS CASAR.", "A.  SALVIUS OTHO."]
    # And the lines are the candidates' own, not anything the model said.
    by_text = {c.text: c for c in cands}
    assert raw[0]["start_line"] == by_text["CAIUS JULIUS CASAR."].line + 1


def test_headings_are_sorted_into_document_order():
    """Chapter ends are derived from the next chapter's start, so an out-of-order
    entry empties one chapter and balloons its neighbour — PG 1727's LLM path did
    exactly that: 25 entries in, 24 chapters out, no warning."""
    cands = _cands("6400")
    raw = _structure_to_raw(
        _verdict(cands, [("TITUS FLAVIUS DOMITIANUS.", "body"),
                         ("CAIUS JULIUS CASAR.", "body"),
                         ("NERO CLAUDIUS CAESAR.", "body")]),
        cands, body_lines("6400"), _cfg())
    starts = [r["start_line"] for r in raw]
    assert starts == sorted(starts)
    assert [r["number"] for r in raw] == [1, 2, 3]


@pytest.mark.parametrize("kind", ["front", "back", "title_page", "toc", "section_marker"])
def test_non_body_kinds_are_dropped_by_default(kind):
    cands = _cands("6400")
    raw = _structure_to_raw(
        _verdict(cands, [("CAIUS JULIUS CASAR.", "body"),
                         ("A.  SALVIUS OTHO.", kind)]),
        cands, body_lines("6400"), _cfg())
    assert [r["title"] for r in raw] == ["CAIUS JULIUS CASAR."]


def test_front_matter_kept_with_the_flag():
    cands = _cands("6400")
    raw = _structure_to_raw(
        _verdict(cands, [("PREFACE", "front"), ("CAIUS JULIUS CASAR.", "body")]),
        cands, body_lines("6400"), _cfg(include_front_matter=True))
    assert [r["kind"] for r in raw] == ["front", "body"]


def test_unlabelled_opening_chapter_starts_at_the_prose():
    """PG 1342 prints no heading over chapter one. Without this the text before
    the first heading belongs to no chapter — but the synthetic chapter must
    begin where the story does, not at line 1, or it swallows the title page,
    the publisher's imprint and the list of illustrations."""
    cands = _cands("1342")
    body = body_lines("1342")
    picks = [(c.text, "body") for c in cands if c.text == "CHAPTER II."]
    raw = _structure_to_raw(_verdict(cands, picks, body_first=True), cands, body, _cfg())

    assert raw[0]["title"] == "Chapter I"
    assert raw[0]["start_marker"] == ""       # synthetic: no heading in the source
    assert body[raw[0]["start_line"] - 1].startswith(
        "It is a truth universally acknowledged")


@pytest.mark.parametrize("book_id,wants_synthetic", [
    ("1342", True),    # a real unlabelled chapter one
    ("1260", False),   # Currer Bell's preface, not the story
    ("2641", False),   # the title page
])
def test_body_starts_before_first_heading_is_checked_not_believed(book_id, wants_synthetic):
    """Nearly every book has *some* text above its first heading, so the flag on
    its own gave 1260 and 2641 a spurious extra chapter. The claim is verified
    against the text: walk back only as far as the contents listing, veto the
    block if it carries a front-matter heading, and require sentences rather
    than captions."""
    from gutenberg_reader.stages.s02_discovery import _maybe_prepend_chapter_one
    from tests.fixtures import golden

    body = body_lines(book_id)
    printed = [c for c in golden(book_id)["chapters"] if not c["synthetic"]]
    raw = [
        {"number": i + 1, "title": c["title"], "start_line": c["line"] + 1,
         "start_marker": c["title"], "kind": "body"}
        for i, c in enumerate(printed)
    ]
    assert (len(_maybe_prepend_chapter_one(list(raw), body)) > len(raw)) is wants_synthetic


def test_an_ordinal_outside_the_list_is_ignored():
    cands = _cands("3296")
    v = _verdict(cands, [(cands[0].text, "body")])
    v["headings"].append({"ordinal": 99_999, "kind": "body"})
    raw = _structure_to_raw(v, cands, body_lines("3296"), _cfg())
    assert len(raw) == 1


# ── Cache invalidation ───────────────────────────────────────────────────────

def test_detector_id_tracks_the_choice():
    assert _detector_id(_cfg(structure_detector="llm")) == "llm-v1"
    assert _detector_id(_cfg(structure_detector="regex")) == "regex-v1"


def test_old_discovery_files_declare_themselves_stale():
    """1727 and 2641 both have cached discovery.json that disagrees with the
    detector supposedly behind them. A file with no version is version 1."""
    old = DiscoveryResult.from_dict({
        "metadata": BookMetadata(title="t", author="a", language="en",
                                 gutenberg_id="1").to_dict(),
        "chapters": [],
    })
    assert old.schema_version == 1 != SCHEMA_VERSION
    assert old.detector == "regex-v1"


def test_structure_verdict_round_trips():
    d = DiscoveryResult(
        metadata=BookMetadata(title="t", author="a", language="en", gutenberg_id="1"),
        chapters=[], detector="llm-v1", work_type="history",
        has_chapter_structure=False,
    )
    back = DiscoveryResult.from_dict(d.to_dict())
    assert (back.detector, back.work_type, back.has_chapter_structure) == \
           ("llm-v1", "history", False)
