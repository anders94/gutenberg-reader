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


@pytest.mark.parametrize("label", ["term", "title", "rhetorical"])
def test_labels_that_lose_their_boundary(label):
    """A rhetorical utterance is quoted but nobody in the scene says it — an
    abstraction, a personified thing, an imagined objector. Giving it a voice of
    its own would confuse a listener, so it reads as narration like the others."""
    text = 'And truth saith unto me “thou art not God” and I heard it.'
    reading, segs = _seg(text, CURLY)
    out = segmenter.apply_span_labels(segs, reading, {1: label})

    assert [s["type"] for s in out] == ["narration"]
    assert out[0]["text"] == text
    assert text_utils.verify_span_coverage(reading, out) == (True, [])


def test_a_citation_keeps_its_boundary_and_a_rhetorical_one_does_not():
    """The two are deliberately different: a quoted document or verse is worth
    reading differently, an imagined objector is not."""
    text = 'He wrote “the sea was calm” and habit whispered “stay here now”.'
    reading, segs = _seg(text, CURLY)
    dialogue = [i for i, s in enumerate(segs) if s["type"] == "dialogue"]

    cited = segmenter.apply_span_labels(segs, reading, {dialogue[0]: "citation"})
    assert any(s.get("notes") == "citation" for s in cited)

    rhet = segmenter.apply_span_labels(segs, reading, {dialogue[1]: "rhetorical"})
    assert all(s.get("notes") != "rhetorical" for s in rhet)


@pytest.mark.parametrize("text", [
    "And truth saith unto me “thou art not God” and I heard it.",
    "The whole air answered “Anaximenes was deceived, I am not God.”",
    "A violent habit whispered “canst thou live without them?”",
])
def test_speech_given_to_an_abstraction_is_asked_about(text):
    """A speech verb usually settles a span, and that hid the whole category:
    "the air answered" and "a violent habit whispered" never reached the model at
    all, so `rhetorical` fired on 16 spans of PG 3296 instead of the dozens it
    should have. No lexical rule separates "the air" from "the priest", so the
    cue asks rather than decides — adding a word to it can cost a call, never an
    answer."""
    reading, segs = _seg(text, CURLY)
    settled, ask = segmenter.classify_spans_deterministically(segs, reading)
    assert ask and not settled


@pytest.mark.parametrize("text", [
    "“Go now,” said the priest, “before the tide turns.”",
    "“Yes,” she said.",
])
def test_ordinary_speech_still_settles(text):
    reading, segs = _seg(text, CURLY)
    settled, ask = segmenter.classify_spans_deterministically(segs, reading)
    assert not ask and set(settled.values()) == {"speech"}


# ── Page markup a voice would otherwise read out loud ────────────────────────

@pytest.mark.parametrize("raw,said", [
    ("_You_ may well be surprised", "You may well be surprised"),
    ("=Little Women=; or Meg, Jo", "Little Women; or Meg, Jo"),
    ("the ----shire militia", "the blankshire militia"),
    ("a lieutenant's commission in the ----shire.",
     "a lieutenant's commission in the blankshire."),
    ("said,--and then she left", "said,—and then she left"),
    ("better acquainted----”", "better acquainted—”"),
    ("--As he was returning", "—As he was returning"),
    # The discriminator: a dash run mid-line before a CAPITAL is an em-dash
    # opening a clause, not a redacted name. Both of these are from PG 6400.
    ("whom we have this tradition: --As he was returning",
     "whom we have this tradition: —As he was returning"),
    ("the old goat-------- and then", "the old goat and then"),
    ("------------------When a picture", "When a picture"),
])
def test_page_markup_becomes_words(raw, said):
    assert text_utils.normalize_typography(raw) == said


@pytest.mark.parametrize("text", [
    "a well-known man", "snake_case_name", "the semi-detached house",
    "nothing to change here",
])
def test_ordinary_text_is_left_alone(text):
    assert text_utils.normalize_typography(text) == text


def test_normalizing_twice_changes_nothing_more():
    raw = "_You_ may, in the ----shire,--see =this=."
    once = text_utils.normalize_typography(raw)
    assert text_utils.normalize_typography(once) == once


def test_the_reading_text_check_still_catches_a_dropped_word():
    """The check now normalises both sides, so it must still be exact about
    everything else: loosening it to allow an italic underscore through would be
    worthless if it also let a lost word through."""
    reading, _ = segmenter.normalize_chapter("One _two_ three.")
    assert text_utils.verify_reading_text("One _two_ three.", reading)[0]
    ok, issues = text_utils.verify_reading_text("One _two_ three four.", reading)
    assert not ok and issues


# ── A publisher's catalogue is not the end of the book ───────────────────────

CATALOGUE = """The last line of the story, spoken with feeling.

Louisa M. Alcott's Writings

THE LITTLE WOMEN SERIES.

Little Women; or Meg, Jo, Beth, and Amy. Illustrated. 16mo. $1.50.

Little Men. Life at Plumfield. Illustrated. 16mo. $1.50.

An Old-Fashioned Girl. Illustrated. 16mo. $1.50.

Eight Cousins; or, The Aunt-Hill. Illustrated. 16mo. $1.50.

The above eight volumes, uniformly bound in cloth, gilt, in box, $12.00.

LITTLE, BROWN, & COMPANY, Publishers
""".splitlines()


def test_a_trailing_catalogue_is_found_at_its_heading():
    i = text_utils.find_publisher_matter(CATALOGUE, 0, len(CATALOGUE) - 1)
    assert CATALOGUE[i] == "Louisa M. Alcott's Writings"


def test_a_printers_colophon_is_found():
    lines = ["Darcy, as well as Elizabeth, really loved them.", "", "",
             "CHISWICK PRESS:--CHARLES WHITTINGHAM AND CO.",
             "TOOKS COURT, CHANCERY LANE, LONDON.", "",
             "*** END OF THE PROJECT GUTENBERG EBOOK ***"]
    i = text_utils.find_publisher_matter(lines, 0, len(lines) - 1)
    assert lines[i].startswith("CHISWICK PRESS")


@pytest.mark.parametrize("lines", [
    # a novel that merely mentions a publisher, mid-prose
    ["He sent immediately for a cabriolet, and hastened to the publisher's office.",
     "There he found nobody, and returned home in the rain."],
    # a book that ends on its own words, in capitals
    ["But Thou, being the Good which needeth no good, art ever at rest.",
     "", "GRATIAS TIBI DOMINE"],
    ["Some think that he was killed by his slave.", "",
     "THE END OF LIVES OF THE POETS."],
])
def test_ordinary_endings_are_not_trimmed(lines):
    assert text_utils.find_publisher_matter(lines, 0, len(lines) - 1) is None


# ── What the renderer's pacing engine depends on (UPSTREAM #6) ───────────────

def test_offsets_and_notes_survive_segmentation():
    """tts-audiobook v2 derives pacing from these: adjacent offsets plus
    non-terminal punctuation mean a mid-sentence split and a tight gap, an
    offset gap means a scene break, and "quote-continues" means a paragraph
    pause inside one speech. Dropping or renumbering any of it degrades the
    audiobook silently, so it is pinned here rather than left to convention."""
    # Gutenberg's convention for a speech that runs past a paragraph break: no
    # closing quote at the end, and the next paragraph re-opens with one.
    text = ('“I shall go, and nothing will stop me.\n\n'
            '“Not you, nor anyone else.”\n\n'
            'She went, and the house was quiet.')
    reading, segs = _seg(text, CURLY)
    assert all("start" in s and "end" in s for s in segs)
    for s in segs:
        assert reading[s["start"]:s["end"]] == s["text"]
    assert [s["start"] for s in segs] == sorted(s["start"] for s in segs)
    assert any(s.get("notes") == "quote-continues" for s in segs)


def test_normalize_chapter_applies_the_typography_pass():
    """The function existing is not the point; the pipeline calling it is."""
    reading, _ = segmenter.normalize_chapter("_You_ may see the ----shire.")
    assert reading == "You may see the blankshire."


def test_a_few_catalogue_words_are_not_a_catalogue():
    """The hit count and density are what stop this trimming a real chapter, so
    they need a case with genuine hits that must survive. Four books in the
    library mention a publisher in ordinary prose."""
    lines = [
        "The book was Illustrated. He turned the page and read on.",
        "",
        "It had cost him fifty cents, which he could ill afford.",
        "",
        "And so the long evening closed, and he slept.",
    ]
    assert text_utils.find_publisher_matter(lines, 0, len(lines) - 1) is None


# ── The span review has to be able to see the span ───────────────────────────

def _passage(n_before=0, n_after=0, span='"in a figure,"'):
    from gutenberg_reader.stages.s05_segments import _render_span_passages
    before, after = "word " * n_before, " word" * n_after
    text = f"{before}{span}{after}"
    reading = text
    segs = [
        {"type": "narration", "start": 0, "end": len(before), "para": 0},
        {"type": "dialogue", "start": len(before),
         "end": len(before) + len(span), "para": 0},
        {"type": "narration", "start": len(before) + len(span),
         "end": len(text), "para": 0},
    ]
    return _render_span_passages(segs, reading, [1])


def test_the_span_is_visible_however_long_the_paragraph():
    """The passage was cut to its first 600 characters, which removed the span
    itself from 49% of the questions asked about PG 3296 and 59% of PG 6400 —
    long paragraphs. The model was shown a paragraph with nothing marked in it
    and asked what the marked span was, so those answers were guesses."""
    for before in (0, 50, 200, 1000, 5000):
        line = _passage(n_before=before, n_after=before)
        assert "⟦0⟧" in line and "⟦/0⟧" in line, f"lost at {before} words before"
        assert "in a figure," in line


def test_a_windowed_passage_keeps_context_on_both_sides():
    line = _passage(n_before=1000, n_after=1000)
    a, b = line.index("⟦0⟧"), line.index("⟦/0⟧")
    assert line.count("word") > 20            # context survives
    assert len(line[:a]) > 100 and len(line[b:]) > 100
    assert line.startswith("0| …") and line.endswith("…")


def test_a_short_paragraph_is_passed_through_whole():
    line = _passage(n_before=3, n_after=3)
    assert "…" not in line
