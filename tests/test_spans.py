"""Offsets, coverage, and what a quoted span actually is.

Two defects motivate this file. The coverage check used to rebuild the chapter
with `" ".join(segment texts)` and diff it, so a quoted word segmented on its own
came back as '"Deserters" :' against an original '"Deserters":' and was reported
as altered text — a bug in the check. And the segmenter treats every quoted span
as dialogue, so in a history a scare-quoted term becomes dialogue that then
demands a speaker: 12 of PG 2131's 29 dialogue segments landed on "Unknown".
"""

from __future__ import annotations

import pytest

from gutenberg_reader import segmenter, text_utils
from tests.fixtures import body_lines, golden

CURLY = ("“", "”")


def _seg(text, quote_pair=None):
    return segmenter.segment_text(text, quote_pair)


# ── The canonical text ───────────────────────────────────────────────────────

def test_reading_text_only_changes_whitespace():
    text = "Wrapped over\ntwo lines here.\n\nAnd a second\nparagraph."
    reading, spans = segmenter.normalize_chapter(text)
    assert reading == "Wrapped over two lines here.\n\nAnd a second paragraph."
    assert [reading[a:b] for a, b in spans] == [
        "Wrapped over two lines here.", "And a second paragraph."]
    assert text_utils.verify_reading_text(text, reading) == (True, [])


def test_reading_text_check_catches_a_real_change():
    reading, _ = segmenter.normalize_chapter("One two three.")
    ok, issues = text_utils.verify_reading_text("One two THREE.", reading)
    assert not ok and issues


@pytest.mark.parametrize("book_id", ["1342", "2701", "2131", "1661"])
def test_every_segment_is_a_slice_of_the_reading_text(book_id):
    """Not "matches" — *is*. The text is cut by code from one canonical string,
    so there is nothing for a model or a rejoin to alter."""
    body = body_lines(book_id)
    g = golden(book_id)
    c = g["chapters"][1] if len(g["chapters"]) > 1 else g["chapters"][0]
    nxt = next((x["line"] for x in g["chapters"] if x["line"] > c["line"]), len(body))
    reading, segs = _seg("\n".join(body[c["line"]:nxt]))

    assert text_utils.verify_reading_text("\n".join(body[c["line"]:nxt]), reading)[0]
    assert all(reading[s["start"]:s["end"]] == s["text"] for s in segs)
    assert text_utils.verify_span_coverage(reading, segs) == (True, [])


def test_coverage_catches_a_gap_and_an_overlap():
    reading, segs = _seg("First part here. Second part there.")
    assert text_utils.verify_span_coverage(reading, segs)[0]
    assert not text_utils.verify_span_coverage(reading, segs[1:])[0] or len(segs) == 1

    overlapping = [dict(segs[0]), dict(segs[0])]
    assert not text_utils.verify_span_coverage(reading, overlapping)[0]


def test_quoted_term_no_longer_reports_altered_text():
    """The exact false positive this replaces."""
    text = 'They were called "Deserters": and this is the reason.'
    reading, segs = _seg(text)
    assert text_utils.verify_span_coverage(reading, segs) == (True, [])


# ── What a quoted span is ────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("“Go now,” said the priest, “before the tide turns.”", "speech"),
    ("“Yes,” she said.", "speech"),
    ("The priests call it the “sacred way” and it runs east.", "term"),
    ("They were called “Deserters”: and this is the reason.", "term"),
])
def test_the_plain_cases_need_no_model(text, expected):
    """A speech verb beside a quoted span settles it; so does a naming verb
    beside a short one. Only what has evidence both ways, or none, is asked."""
    reading, segs = _seg(text, CURLY)
    settled, ask = segmenter.classify_spans_deterministically(segs, reading)
    assert set(settled.values()) == {expected}, (settled, ask)


def test_a_genuinely_ambiguous_span_is_asked_about():
    """"read the 'Odyssey' aloud" could be either; guessing is worse than asking."""
    reading, segs = _seg(
        "He read the “Odyssey” aloud to them every evening.", CURLY)
    settled, ask = segmenter.classify_spans_deterministically(segs, reading)
    assert ask and not settled


def test_a_term_verdict_removes_a_boundary_rather_than_editing_text():
    """Nothing is repaired because nothing was rebuilt: the merged segment is a
    wider slice of the same string, so the colon comes back attached."""
    text = "They were called “Deserters”: and this is the reason."
    reading, segs = _seg(text, CURLY)
    out = segmenter.apply_span_labels(segs, reading, {1: "term"})

    assert [s["type"] for s in out] == ["narration"]
    assert out[0]["text"] == text
    assert text_utils.verify_span_coverage(reading, out) == (True, [])


def test_demotion_does_not_undo_the_narration_length_limit():
    """Merging every adjacent narration pair would defeat split_long_narration,
    which exists to keep the TTS voice stable over a segment."""
    long_text = ("This sentence is exactly fifty characters long!!! " * 12).strip()
    reading, segs = _seg(long_text)
    out = segmenter.apply_span_labels(segs, reading, {})
    assert len(out) > 1
    assert max(s["end"] - s["start"] for s in out) <= segmenter.MAX_NARRATION_CHARS


def test_speech_is_left_alone():
    text = "“Go now,” said the priest, “before the tide turns.”"
    reading, segs = _seg(text, CURLY)
    settled, _ = segmenter.classify_spans_deterministically(segs, reading)
    out = segmenter.apply_span_labels(segs, reading, settled)
    assert sum(1 for s in out if s["type"] == "dialogue") == 2


def test_offsets_survive_a_speaker_correction():
    """The critic rewrites a speaker label; a hand-listed constructor would drop
    whatever was added to the model since it was written."""
    from dataclasses import replace
    from gutenberg_reader.models import Segment

    seg = Segment(type="dialogue", text="x", speaker=None, start=5, end=9)
    assert replace(seg, speaker="Ahab").start == 5


# ── Quote style is a property of the edition ─────────────────────────────────

def test_a_single_long_speech_has_too_few_quotes_to_call_a_style():
    """Which is why the style is detected once over the whole body and carried
    on DiscoveryResult, rather than re-derived per chapter."""
    one_speech = "“" + ("I will talk. " * 20).strip() + "”"
    assert segmenter.detect_quote_pair(one_speech) is None
    reading, segs = _seg(one_speech, CURLY)
    assert any(s["type"] == "dialogue" for s in segs)
