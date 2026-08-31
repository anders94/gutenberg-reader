"""Chapter boundary detection — front matter, back matter, and LLM validation.

Regression tests for the Odyssey (PG 1727) defects: the final chapter swallowing
the footnote appendix, and the prefaces being promoted to a synthetic chapter 1.
The cached raw texts under cache/ serve as fixtures when present.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from gutenberg_reader import text_utils
from gutenberg_reader.stages.s02_discovery import (
    _build_chapter_infos,
    _drop_leading_front_matter,
    _maybe_prepend_chapter_one,
    _split_headless_body,
    _validate_llm_chapters,
    TARGET_PART_WORDS,
)
from gutenberg_reader.config import Config
from gutenberg_reader.models import ChapterInfo

CACHE = Path(__file__).resolve().parent.parent / "cache"


def _discover(body_lines: list[str], include_back_matter: bool = False):
    """Run the regex discovery path the way stage 02 does."""
    raw = text_utils.detect_chapters_regex(body_lines)
    raw = _maybe_prepend_chapter_one(raw, body_lines)
    raw = _drop_leading_front_matter(raw, body_lines)
    raw = _split_headless_body(raw, body_lines)
    return _build_chapter_infos(raw, body_lines, 0,
                                include_back_matter=include_back_matter)


def _body_lines(book_id: str) -> list[str]:
    raw_path = CACHE / book_id / "01-raw" / "book.txt"
    if not raw_path.exists():
        pytest.skip(f"cache/{book_id} fixture not present")
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    start, end = text_utils.find_body_bounds(lines)
    return lines[start:end]


# ── Heading classification ────────────────────────────────────────────────────

@pytest.mark.parametrize("title,kind", [
    ("PREFACE TO FIRST EDITION", "front"),
    ("PREFACE.", "front"),
    ("INTRODUCTION", "front"),
    ("DEDICATION", "front"),
    ("Contents", "front"),
    ("LIST OF ILLUSTRATIONS", "front"),
    ("FOOTNOTES:", "back"),
    ("APPENDIX", "back"),
    ("INDEX.", "back"),
    ("GLOSSARY", "back"),
    ("ERRATA", "back"),
    ("BOOK XXIV", "body"),
    ("CHAPTER VII.", "body"),
    ("Chapter 1. Marseilles—The Arrival", "body"),
    ("The captain took notes on the voyage.", "body"),
    ("An index of his character emerged.", "body"),
])
def test_classify_heading(title, kind):
    assert text_utils.classify_heading(title) == kind


# ── Synthetic book: back-matter trim and front-matter guard ──────────────────

# Sentence-shaped: _maybe_prepend_chapter_one only promotes a block that reads
# as prose, so punctuation-free filler would make these fixtures pass (or fail)
# for reasons unrelated to what each test is checking.
PREFACE_WORDS = " ".join(["preface"] * 30) + ". " + " ".join(["preface"] * 30) + "."
CHAPTER_WORDS = " ".join(["story"] * 30) + ". " + " ".join(["story"] * 30) + "."
FOOTNOTE_WORDS = " ".join(["footnote"] * 30) + ". " + " ".join(["footnote"] * 30) + "."

SYNTHETIC = f"""DEDICATION

To someone dear.

PREFACE TO FIRST EDITION

{PREFACE_WORDS}

CHAPTER I.

{CHAPTER_WORDS}

CHAPTER II.

{CHAPTER_WORDS}

CHAPTER III.

{CHAPTER_WORDS}

FOOTNOTES:

{FOOTNOTE_WORDS}
""".splitlines()


def test_synthetic_back_matter_trimmed():
    infos = _discover(SYNTHETIC)
    assert [ci.title for ci in infos] == ["CHAPTER I.", "CHAPTER II.", "CHAPTER III."]
    last_text = "\n".join(SYNTHETIC[infos[-1].start_line - 1:infos[-1].end_line])
    assert "FOOTNOTES" not in last_text
    assert "footnote" not in last_text


def test_synthetic_front_matter_not_promoted():
    infos = _discover(SYNTHETIC)
    # The dedication + preface block must not become a synthetic chapter 1.
    assert infos[0].title == "CHAPTER I."
    assert all("preface" not in ci.title.lower() for ci in infos)


def test_synthetic_include_back_matter_keeps_own_chapter():
    infos = _discover(SYNTHETIC, include_back_matter=True)
    assert [ci.title for ci in infos][:3] == ["CHAPTER I.", "CHAPTER II.", "CHAPTER III."]
    assert infos[-1].kind == "back"
    assert infos[-1].title == "FOOTNOTES:"
    body_text = "\n".join(SYNTHETIC[infos[2].start_line - 1:infos[2].end_line])
    assert "footnote" not in body_text


def test_unlabeled_chapter_one_still_promoted():
    # P&P-style: real narrative before the first heading, no front-matter heading.
    lines = f"""{CHAPTER_WORDS}

CHAPTER II.

{CHAPTER_WORDS}

CHAPTER III.

{CHAPTER_WORDS}
""".splitlines()
    infos = _discover(lines)
    assert len(infos) == 3
    assert infos[0].title == "Chapter I"
    assert infos[0].start_line == 1


# ── LLM entry validation ─────────────────────────────────────────────────────

def _cfg(**kw) -> Config:
    return Config(book_id="0", **kw)


def test_llm_out_of_order_sorted_not_dropped():
    entries = [
        {"title": "FOOTNOTES:", "start_line": 900},
        {"title": "Chapter One", "start_line": 10},
        {"title": "Chapter Two", "start_line": 500},
    ]
    valid = _validate_llm_chapters(entries, 1000, _cfg())
    assert [ch["title"] for ch in valid] == ["Chapter One", "Chapter Two"]
    assert [ch["start_line"] for ch in valid] == [10, 500]


def test_llm_bad_entries_dropped():
    entries = [
        {"title": "Chapter One", "start_line": 10},
        {"title": "Ghost", "start_line": 0},
        {"title": "Beyond", "start_line": 5000},
        {"title": "Duplicate", "start_line": 10},
        {"title": "Chapter Two", "start_line": "not-a-number"},
    ]
    valid = _validate_llm_chapters(entries, 1000, _cfg())
    assert [ch["title"] for ch in valid] == ["Chapter One"]


def test_llm_front_back_kept_with_flags():
    entries = [
        {"title": "PREFACE", "start_line": 5},
        {"title": "Chapter One", "start_line": 10},
        {"title": "FOOTNOTES:", "start_line": 900},
    ]
    valid = _validate_llm_chapters(
        entries, 1000, _cfg(include_front_matter=True, include_back_matter=True))
    assert [ch["kind"] for ch in valid] == ["front", "body", "back"]


# ── PG 1727 (The Odyssey) — the shipped failure ──────────────────────────────

def test_odyssey_structure():
    body = _body_lines("1727")
    infos = _discover(body)

    assert len(infos) == 24
    assert infos[0].title.startswith("BOOK I")
    assert infos[-1].title.startswith("BOOK XXIV")

    # No synthetic preface-chapter in front.
    assert all(ci.title.startswith("BOOK") for ci in infos)

    # The last chapter must not have swallowed the footnote appendix.
    sizes = [ci.word_count for ci in infos]
    med = statistics.median(sizes)
    assert sizes[-1] <= 1.5 * med, f"last chapter {sizes[-1]} words vs median {med}"

    last_text = "\n".join(body[infos[-1].start_line - 1:infos[-1].end_line])
    assert "FOOTNOTES" not in last_text

    # No chapter title is front or back matter.
    assert all(ci.kind == "body" for ci in infos)
    assert all(text_utils.classify_heading(ci.title) == "body" for ci in infos)


# ── Regressions on the other cached books ────────────────────────────────────

def test_pride_and_prejudice_unchanged_except_kind():
    """P&P's unlabeled chapter 1 must still be detected, preface excluded.

    Compared against the committed golden rather than cache/02-discovery: that
    file is rewritten by any run, so an LLM-detected structure landing there made
    this test compare the regex detector against the model's answer.
    """
    from tests.fixtures import golden

    body = _body_lines("1342")
    infos = _discover(body)
    g = golden("1342")

    assert len(infos) == g["exact_count"] == 61
    assert infos[0].title == "Chapter I"
    assert [ci.start_line - 1 for ci in infos] == [c["line"] for c in g["chapters"]]


def test_monte_cristo_footnotes_trimmed():
    """1184 has the same swallowed-FOOTNOTES defect; the trim must fix it too."""
    body = _body_lines("1184")
    infos = _discover(body)
    assert len(infos) == 117
    last_text = "\n".join(body[infos[-1].start_line - 1:infos[-1].end_line])
    assert "FOOTNOTES" not in last_text


def test_jane_eyre_preface_not_promoted():
    """1260's dedication + Currer Bell preface used to become a bogus chapter 1."""
    body = _body_lines("1260")
    infos = _discover(body)
    assert len(infos) == 38
    assert infos[0].title == "CHAPTER I"
    first_text = "\n".join(body[infos[0].start_line - 1:infos[0].end_line])
    assert "THACKERAY" not in first_text


def test_moby_dick_cetology_not_split():
    """Chapter 32's whale taxonomy ("BOOK I. (_Folio_), CHAPTER I. (_Sperm
    Whale_)...") is prose that starts like a heading; it must stay one chapter."""
    body = _body_lines("2701")
    infos = _discover(body)
    assert len(infos) == 136  # 135 numbered chapters + the Epilogue, nothing synthetic
    # ETYMOLOGY and the Sub-Sub-Librarian's EXTRACTS are front matter; promoting
    # them to chapter 1 used to shift every chapter number by one.
    assert infos[0].title == "CHAPTER 1. Loomings."
    assert not any("Folio" in ci.title for ci in infos)
    # Wrapped two-line titles are still detected as headings.
    assert any(ci.title.startswith("CHAPTER 56.") for ci in infos)


def test_sherlock_holmes_bare_numeral_headings():
    """1661 titles its stories "I. A SCANDAL IN BOHEMIA" — a bare roman numeral
    with no CHAPTER/PART/BOOK keyword. No pattern matched, so regex detection
    returned zero, stage 02 fell back to the LLM, and that fallback only sees
    body_lines[:500] — leaving one 100,984-word "chapter" holding 11 stories."""
    body = _body_lines("1661")
    infos = _discover(body)

    assert len(infos) == 12
    assert infos[0].title == "I. A SCANDAL IN BOHEMIA"
    assert infos[-1].title == "XII. THE ADVENTURE OF THE COPPER BEECHES"
    # The title page + contents block must not become a synthetic chapter one.
    assert not any("Sherlock Holmes" == ci.title for ci in infos)
    # No story swallows the rest of the book.
    assert max(ci.word_count for ci in infos) < 4 * statistics.median(
        ci.word_count for ci in infos
    )


def test_bare_numeral_pattern_requires_all_caps_title():
    """The bare-numeral shape is weak evidence — 'M.' is a roman numeral, so an
    unrestricted pattern eats French "M. Morrel..." prose (1184 gained 8 phantom
    chapters, 2701 four). A title is required, and it must be all caps."""
    matches = text_utils.looks_like_chapter_heading

    assert matches("I. A SCANDAL IN BOHEMIA")
    assert matches("XII. THE ADVENTURE OF THE COPPER BEECHES")
    # Bare numeral with no title is an in-story section marker, not a heading.
    assert not matches("I.")
    assert not matches("II.")
    # Prose that happens to start with a numeral-shaped token.
    assert not matches("M. Morrel, and this day and a half was lost from pure whim, for the")
    assert not matches("I. But it is a common name in Nantucket, they say, and I suppose this")
    assert not matches("I. THE FOLIO WHALE; II. the OCTAVO WHALE; III. the DUODECIMO WHALE.")


def test_indented_numeral_contents_entries_are_toc():
    """Contents entries for the bare-numeral form are title case, so the heading
    patterns will not see them; the front-matter walk needs its own check or a
    12-entry contents block reads as narrative and becomes chapter one."""
    assert text_utils.TOC_NUMERAL_ENTRY_RE.match("   I.     A Scandal in Bohemia")
    assert text_utils.TOC_NUMERAL_ENTRY_RE.match("  XII.  The Copper Beeches")
    # Not indented → a real heading line, not a contents entry.
    assert not text_utils.TOC_NUMERAL_ENTRY_RE.match("I. A SCANDAL IN BOHEMIA")


@pytest.mark.parametrize("book_id,expected", [
    ("1184", 117), ("1260", 38), ("1342", 61), ("1661", 12),
    ("1727", 24), ("2641", 19), ("2701", 136), ("3296", 13),
    ("37106", 47),
])
def test_chapter_counts_pinned(book_id, expected):
    """Every cached book's chapter count, pinned. The bare-numeral pattern is
    the loosest in CHAPTER_PATTERNS; this is what catches it over-matching."""
    assert len(_discover(_body_lines(book_id))) == expected


def test_moby_dick_epilogue_is_its_own_chapter():
    """Melville's Epilogue is where Ishmael explains how a drowned crew has a
    narrator. It carries a name instead of a number, so no pattern matched it
    and it was glued onto the end of "The Chase—Third Day" — never its own
    track."""
    body = _body_lines("2701")
    infos = _discover(body)

    assert infos[-1].title == "Epilogue"
    assert infos[-1].kind == "body"          # part of the story, not apparatus
    assert infos[-2].title == "CHAPTER 135. The Chase.—Third Day."

    epilogue = "\n".join(body[infos[-1].start_line - 1:infos[-1].end_line])
    assert "ESCAPED ALONE TO TELL THEE" in epilogue
    assert "another orphan" in epilogue      # runs to its real end
    # The preceding chapter must no longer carry it.
    chase = "\n".join(body[infos[-2].start_line - 1:infos[-2].end_line])
    assert "ESCAPED ALONE" not in chase
    # Back-matter trimming still applies to whatever ends up last.
    assert "Transcriber" not in epilogue


def test_named_division_headings():
    matches = text_utils.looks_like_chapter_heading
    for title in ("Epilogue", "EPILOGUE", "epilogue", "Prologue",
                  "EPILOGUE.", "Epilogue: The Return"):
        assert matches(title), title
    # Body words, not headings.
    for title in ("The epilogue was brief.", "Prologue to the affair, he said."):
        assert not matches(title), title
    # A named division belongs to the story, never the apparatus.
    assert text_utils.classify_heading("Epilogue") == "body"
    assert text_utils.classify_heading("Prologue") == "body"


def test_little_women_two_line_headings():
    """37106 centres the numeral and its title on separate lines:

            I.

        PLAYING PILGRIMS.

    Neither line is a heading alone, so none of the 47 body headings matched,
    drop_toc_clusters discarded the contents listing that did, and the
    "never discard everything" fallback handed the whole TOC back — 46 chapters
    holding just their own title and one holding all 188,656 remaining words."""
    body = _body_lines("37106")
    infos = _discover(body)

    assert len(infos) == 47
    assert infos[0].title == "I. PLAYING PILGRIMS."
    assert infos[-1].title == "XLVII. HARVEST TIME."
    # Every chapter is a real chapter, not a contents entry.
    assert min(ci.word_count for ci in infos) > 500
    med = statistics.median(ci.word_count for ci in infos)
    assert max(ci.word_count for ci in infos) < 3 * med


def test_bare_numeral_needs_all_caps_title_below():
    """A bare numeral followed by prose is an in-story section marker (1661 has
    "I." mid-story); followed by an all-caps title it is a chapter heading."""
    heading = ["I.", "", "PLAYING PILGRIMS.", "", "Prose begins here."]
    assert text_utils._two_line_heading(heading, 0) == ("I. PLAYING PILGRIMS.", 2)

    marker = ["I.", "", "To Sherlock Holmes she is always the woman.", ""]
    assert text_utils._two_line_heading(marker, 0) is None

    # Same line, not a two-line pair.
    assert text_utils._two_line_heading(["I. PLAYING PILGRIMS."], 0) is None
    # A caps line that opens a paragraph is not a standalone heading.
    running = ["I.", "", "PLAYING PILGRIMS", "and then the story continued on."]
    assert text_utils._two_line_heading(running, 0) is None


def test_prose_gate_separates_narrative_from_caption_lists():
    from gutenberg_reader.stages.s02_discovery import _reads_as_prose

    narrative = [
        "It is a truth universally acknowledged, that a single man in "
        "possession of a good fortune, must be in want of a wife.",
        "However little known the feelings of such a man may be.",
    ]
    captions = [
        "Mr. Laurence waving his hat",
        "Now, Miss Jo, I'll settle you",
        "A very merry lunch it was",
        "He went prancing down a quiet street",
    ]
    assert _reads_as_prose(narrative)
    assert not _reads_as_prose(captions)
    assert not _reads_as_prose([])


# ── PG 2131 (Herodotus) — a source with no chapter headings ──────────────────

# 2131's body carries no chapter heading anywhere, so the LLM fallback matches
# the front page instead. Both splits below are runs it actually produced:
# the title page (with the editor's NOTE under it), the title repeated over the
# text, and the subtitle line beneath that.
def _herodotus(*starts: int) -> list[dict]:
    titles = {
        11: "AN ACCOUNT OF EGYPT",
        88: "AN ACCOUNT OF EGYPT",
        95: "BEING THE SECOND BOOK OF HIS HISTORIES CALLED EUTERPE",
    }
    return [
        {"number": i + 1, "title": titles[s], "start_line": s,
         "start_marker": titles[s], "kind": "body"}
        for i, s in enumerate(starts)
    ]


@pytest.mark.parametrize("starts,body_start", [
    ((11, 88), 88),        # title page + NOTE, then the text
    ((11, 88, 95), 95),    # ...and the subtitle split off too
])
def test_herodotus_front_page_dropped(starts, body_start):
    """2131 shipped as a 647-word "chapter 1" of title page and editor's NOTE
    plus a 36,916-word chapter 2 holding the rest. Every stub carries the book's
    own title, so classify_heading saw body matter in each."""
    body = _body_lines("2131")
    raw = _drop_leading_front_matter(_herodotus(*starts), body)

    assert len(raw) == 1
    assert raw[0]["number"] == 1          # renumbered; chapter numbers key the caches
    assert raw[0]["start_line"] == body_start

    infos = _build_chapter_infos(raw, body, 0)
    text = "\n".join(body[infos[0].start_line - 1:infos[0].end_line])
    assert "HERODOTUS was born at Halicarnassus" not in text   # the NOTE is gone
    assert "When Cyrus had brought his life to an end" in text  # the book is not


def test_herodotus_front_page_kept_with_flag():
    body = _body_lines("2131")
    raw = _drop_leading_front_matter(
        _herodotus(11, 88, 95), body, include_front_matter=True)

    assert [ch["kind"] for ch in raw] == ["front", "front", "body"]
    assert [ch["number"] for ch in raw] == [1, 2, 3]


def test_herodotus_headless_body_split_into_parts():
    """With the front page gone 2131 is one 36,916-word chapter — correct, and
    useless: stage 05 reads it in a single pass, so there is no parallelism, the
    cast is discovered from one window, and it assembles as one track."""
    body = _body_lines("2131")
    raw = _drop_leading_front_matter(_herodotus(11, 88), body)
    parts = _split_headless_body(raw, body)

    assert len(parts) == 13
    assert [ch["number"] for ch in parts] == list(range(1, 14))
    assert [ch["title"] for ch in parts[:2]] == ["Part 1", "Part 2"]
    assert parts[0]["start_line"] == raw[0]["start_line"]  # nothing lost off the front
    # Every cut lands on a paragraph, never mid-sentence.
    assert all(body[ch["start_line"] - 1].strip() for ch in parts)

    infos = _build_chapter_infos(parts, body, 0)
    sizes = [ci.word_count for ci in infos]
    assert min(sizes) > TARGET_PART_WORDS // 2      # no runt tail
    assert max(sizes) < 2 * TARGET_PART_WORDS
    # The parts tile the body: no line dropped, none read twice.
    assert [ci.start_line for ci in infos[1:]] == [ci.end_line + 1 for ci in infos[:-1]]
    assert sum(sizes) == 36916


def test_headless_split_leaves_front_matter_alone():
    body = _body_lines("2131")
    raw = _drop_leading_front_matter(
        _herodotus(11, 88), body, include_front_matter=True)
    parts = _split_headless_body(raw, body)

    assert parts[0]["kind"] == "front"
    assert parts[0]["title"] == "AN ACCOUNT OF EGYPT"
    assert [ch["title"] for ch in parts[1:3]] == ["Part 1", "Part 2"]
    assert [ch["number"] for ch in parts] == list(range(1, len(parts) + 1))


def test_headless_split_only_when_there_is_no_structure():
    """A long chapter inside a book that has chapters is a long chapter —
    Moby-Dick's "Cetology" runs several times the median and is genuinely one."""
    body = _body_lines("2701")
    raw = text_utils.detect_chapters_regex(body)
    assert _split_headless_body(raw, body) == raw


def test_short_headless_body_left_whole():
    """A long short story is not an unbroken book; splitting it invents structure."""
    lines = (f"{CHAPTER_WORDS}\n\n" * 4).splitlines()
    raw = [{"number": 1, "title": "T", "start_line": 1,
            "start_marker": "", "kind": "body"}]
    assert _split_headless_body(raw, lines) == raw


def test_leading_block_kept_when_it_reads_as_a_chapter():
    """A short opening chapter is ordinary — only a repeated title or an
    apparatus heading marks the leading block as the edition's own matter."""
    lines = f"""CHAPTER I.

{CHAPTER_WORDS}

CHAPTER II.

{CHAPTER_WORDS}
{CHAPTER_WORDS}
{CHAPTER_WORDS}
{CHAPTER_WORDS}
{CHAPTER_WORDS}
""".splitlines()
    raw = text_utils.detect_chapters_regex(lines)
    assert _drop_leading_front_matter(raw, lines) == raw


def test_two_chapter_split_fails_the_structure_check():
    """The median test cannot see a two-chapter book: each chapter sits at the
    median. 2131 shipped silently because of it — and because a warning was all
    it would have produced. This now fails the run."""
    from gutenberg_reader import structure_checks

    def _info(number, words):
        return ChapterInfo(number=number, title="T", start_line=number,
                           end_line=number, word_count=words)

    codes = [f.code for f in structure_checks.check([_info(1, 647), _info(2, 36916)])]
    assert "pair_lopsided" in codes

    # An ordinary two-chapter split passes clean.
    assert not structure_checks.check([_info(1, 5000), _info(2, 7000)])
