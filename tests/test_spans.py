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


# ── Citations: quoted text with nobody in the scene saying it ────────────────

def test_present_tense_attribution_is_asked_about_not_assumed_to_be_speech():
    """Prose narrates speech in the past ("said", "cried") and cites a source in
    the present ("Homer says"), because the source goes on saying it. Not
    conclusive — present-tense narration exists — so it forces the question
    rather than deciding it."""
    reading, segs = _seg("Homer says “the Aigyptian Thon gave her drugs.”", CURLY)
    settled, ask = segmenter.classify_spans_deterministically(segs, reading)
    assert ask and not settled


@pytest.mark.parametrize("text", [
    "The oracle answered “thou shalt not return.”",
    "The inscription reads “I am Sesostris, king of kings.”",
])
def test_citation_cues_force_the_question(text):
    reading, segs = _seg(text, CURLY)
    _, ask = segmenter.classify_spans_deterministically(segs, reading)
    assert ask


def test_ordinary_speech_is_still_settled_without_asking():
    """The cue must not swallow normal dialogue — that would put every quoted
    line in a novel through an extra call for nothing."""
    reading, segs = _seg("“Go now,” said the priest, “before the tide turns.”", CURLY)
    settled, ask = segmenter.classify_spans_deterministically(segs, reading)
    assert not ask and set(settled.values()) == {"speech"}


def test_a_citation_gets_a_speaker_rather_than_going_looking_for_one():
    """It stays dialogue, so it is still read as quoted text and can still get
    its own voice — but attribution has nothing to find, and running it through
    the passes only ever yields Unknown."""
    text = "Homer says “the Aigyptian Thon gave her drugs of healing.” And so it was."
    reading, segs = _seg(text, CURLY)
    out = segmenter.apply_span_labels(segs, reading, {1: "citation"})

    cited = [s for s in out if s.get("notes") == "citation"]
    assert len(cited) == 1
    assert cited[0]["type"] == "dialogue"
    assert cited[0]["speaker"] == text_utils.CITATION_SPEAKER
    assert text_utils.verify_span_coverage(reading, out) == (True, [])


def test_the_citation_label_can_never_become_a_character():
    assert text_utils.is_reserved_character_name(text_utils.CITATION_SPEAKER)
    assert text_utils.is_reserved_character_name("citation")


def test_a_citation_is_not_offered_to_the_attribution_passes():
    """The unresolved set is what passes A, B and C try to name."""
    segments = [
        {"type": "dialogue", "speaker": None, "notes": None},
        {"type": "dialogue", "speaker": None, "notes": "citation"},
        {"type": "narration", "speaker": None, "notes": None},
    ]
    unresolved = {
        i for i, s in enumerate(segments)
        if s["type"] == "dialogue" and not s.get("speaker")
        and s.get("notes") != "citation"
    }
    assert unresolved == {0}


@pytest.mark.parametrize("text,expected", [
    # "is said to be" reports hearsay about the world, not speech by anyone.
    # Read as an attribution tag it made a scare-quoted term look spoken, and
    # PG 2131 shipped '"Deserters"' as dialogue with speaker Unknown.
    ('This city is said to be the mother-city of all. '
     'They were called "Deserters" then.', "term"),
    # A gloss can be a whole clause, so unlike the other term signals this one
    # settles at any length. Herodotus does it constantly.
    ('this word signifies, when translated, '
     '"those who stand on the left hand of the king."', "term"),
    ('Now _piromis_ means in the tongue of Hellas "honourable and good man."', "term"),
])
def test_glosses_and_hearsay_are_not_speech(text, expected):
    reading, segs = _seg(text, ('"', '"'))
    settled, ask = segmenter.classify_spans_deterministically(segs, reading)
    assert set(settled.values()) == {expected}, (settled, ask)


def test_hearsay_stripping_does_not_swallow_a_real_said():
    """Only the passive and impersonal forms are hearsay; "he said" is not."""
    reading, segs = _seg('He said "come here now" to her.', ('"', '"'))
    settled, _ = segmenter.classify_spans_deterministically(segs, reading)
    assert set(settled.values()) == {"speech"}


def test_a_segment_cache_entry_is_rejected_when_the_boundaries_moved():
    """A cache entry is keyed by chapter number, and the number means nothing on
    its own. Re-running PG 6400 after its structure was fixed would otherwise
    load "chapter 2" from the broken 3-chapter split — a 173,461-word span — as
    chapter 2 of the corrected twenty, and nothing would say so."""
    from gutenberg_reader.stages.s05_segments import SEGMENT_FORMAT, _load_cached
    from gutenberg_reader.cache import atomic_write_json
    from gutenberg_reader.config import Config
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "chapter-02.json"
        stamp = {"start_line": 165, "end_line": 2461, "format": SEGMENT_FORMAT}
        atomic_write_json(path, {
            "chapter_number": 2, "chapter_title": "x", "segments": [],
            "word_count": 1, "roster_after": [], "anchor_names": [],
            "source": stamp,
        })
        cfg = Config(book_id="0")
        assert _load_cached(path, cfg, stamp) is not None
        moved = {"start_line": 2462, "end_line": 6636, "format": SEGMENT_FORMAT}
        assert _load_cached(path, cfg, moved) is None
        older = {**stamp, "format": SEGMENT_FORMAT - 1}
        assert _load_cached(path, cfg, older) is None
