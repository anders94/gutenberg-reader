"""Candidate heading blocks — the condensed whole-book view.

Chapter structure is a *global* property of a book. You cannot tell a table of
contents from the body, or know that "I." / "PLAYING PILGRIMS." is *the* heading
pattern, from a local window — which is how the PG 2701, 37106 and 1661 defects
happened. So rather than stream raw text past a reader chunk by chunk, this
module reduces a whole book to the few hundred blocks that could possibly be
headings, cheaply and with high recall, so the entire structure is judged at once.

Measured over the eleven cached books: 9-706 candidates, at most ~8k tokens
rendered, retaining every heading the book actually prints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from gutenberg_reader import text_utils

# A heading stands alone or wraps its title once. Same block-length rule
# detect_chapters_regex already applies, and it must be block-based rather than
# line-based: PG 2641 prints "Chapter I" directly above "The Bertolini" with no
# blank between them, so a filter requiring a blank line below scores 0 of 19 on
# A Room with a View.
MAX_BLOCK_LINES = 2
MAX_BLOCK_LINE_CHARS = 70
MAX_BLOCK_CHARS = 120

# Running prose that happens to sit in a short paragraph. Headings are titles;
# they capitalize.
_PROSE_MIN_WORDS = 3
_PROSE_LOWER_RATIO = 0.5

# A block opening with a quotation mark is a line of dialogue. This is most of
# the difference between 2,055 and ~700 candidates on PG 1184, a novel built of
# short quoted exchanges; no heading in the corpus opens with a quote.
_OPENS_QUOTED = ('"', "'", "“", "‘")

_ROMAN = re.compile(r"^[IVXLCDM]+\.?$", re.IGNORECASE)
# Lines from a two-line heading's numeral to its title, at most.
_TWO_LINE_SPAN = 4

# An illustration's file name, left behind when Gutenberg's HTML edition was
# flattened to text: PG 1184 prints "0023m", "0025m", "0027m" on their own
# lines 92 times, one per picture. Shown as candidates, the model chose three
# of them as chapter headings and cut chapter one into four pieces named after
# image files ("30053m" further in — the run of digits is the picture's
# sequence number and grows). No heading is a run of digits with one lower-case
# letter after it.
_IMAGE_MARKER_RE = re.compile(r"^\d{3,6}[a-z]$")

# A table of contents lists every heading a line or two apart; real chapters are
# hundreds of lines apart. Moby Dick's contents run 135 entries at a gap of 2.
TOC_RUN_MAX_GAP = 4
# Long, because short clusters occur naturally: PG 37106 prints
# "[Illustration: A Merry Christmas]" two lines above "II." / "A MERRY
# CHRISTMAS.", which is a three-candidate cluster and a real heading. A contents
# listing is not three entries long — Moby Dick's is 136, Little Women's 47.
TOC_RUN_MIN_LEN = 6
# How far apart two entries of a run may be when only short lines — captions
# the prose filter dropped — sit between them. Little Women's illustration
# list has stretches of eight such captions, nineteen lines, between two
# candidates.
TOC_RUN_MAX_GAP_NO_PROSE = 24


@dataclass(frozen=True)
class Candidate:
    """One block that could be a heading, and every cheap signal about it."""

    ordinal: int          # 0..N-1 — what a model addresses, never a line number
    line: int             # 0-based index into body_lines
    n_lines: int          # 1 or 2
    text: str             # the block, lines joined by " / "
    flags: tuple[str, ...]
    gap_before: int       # lines since the previous candidate
    prose_before: int = 0   # paragraphs (blocks too big to be headings) since the previous candidate
    prose_after: bool = False  # the next block is a paragraph, not another short line

    def render(self) -> str:
        flags = " ".join(self.flags)
        return f"{self.ordinal}| {self.line + 1}: {self.text}" + (f"  [{flags}]" if flags else "")


def _shape(text: str) -> str:
    """Token-class signature: 'CHAPTER 9 T T', 'A A A', 'R'.

    What separates a series from a one-off. PG 6400's twelve Caesars share a
    shape at ~1,200-line intervals; the monumental inscription
    'M. AGRIPPA. L. F. COS: TERTIUM. FECIT.' is a singleton that no regex can
    tell apart from 'A.  SALVIUS OTHO.' — but a shape census can.
    """
    out = []
    for tok in text.split()[:8]:
        bare = tok.strip(".,:;—–-()[]\"'")
        if not bare:
            out.append("p")
        elif bare.isdigit():
            out.append("9")
        elif _ROMAN.match(bare):
            out.append("R")
        elif bare.isupper():
            out.append("A")
        elif bare[:1].isupper():
            out.append("T")
        else:
            out.append("w")
    return " ".join(out)


def _flags(block: list[str], text: str, body_lines: list[str], idx: int) -> tuple[str, ...]:
    """Annotate a block with what every existing regex thinks of it.

    This is how the accumulated per-book knowledge survives the rewrite. Each
    pattern stops being a *decider* and becomes a *feature*: today the regex has
    a vote and the model has none, which is why one bad match on a Latin initial
    turned Suetonius into three chapters.
    """
    f: list[str] = []
    if text_utils.looks_like_chapter_heading(text):
        f.append("regex:chapter")
    if text_utils._two_line_heading(body_lines, idx) is not None:
        f.append("regex:two-line")
    if text_utils.BARE_NUMERAL_RE.match(block[0]):
        f.append("regex:bare-numeral")
    if text_utils.FRONT_MATTER_RE.match(text):
        f.append("front-matter-word")
    if text_utils.BACK_MATTER_RE.match(text):
        f.append("back-matter-word")
    if text_utils.TOC_NUMERAL_ENTRY_RE.match(body_lines[idx]):
        f.append("toc-entry")
    if text.isupper():
        f.append("all-caps")
    if body_lines[idx][:1].isspace():
        f.append("centred")
    if text.lower().startswith("[illustration"):
        f.append("illustration")
    f.append(f"shape:{_shape(text)}")
    return tuple(f)


# A short sentence: "Albert laughed.", "Franz continued:", "Valentine
# screamed." PG 1184 prints hundreds of these as one-line paragraphs, the
# structure pass chose sixteen of them as chapter headings, and each cut a
# chapter in two under a title nobody would read aloud. Too short for the
# ratio test above; the tell is a lower-case word after the first, with the
# block ending the way a sentence ends. A heading that ends in a period
# ("CHAPTER II.") has no lower-case word in it, and "Chapter 2. Father and
# Son" does not end like a sentence. Particles do not count as the lower-case
# word — "Story of the Door." and "The Whiteness of the Whale." are titles
# — so the tell is a lower-case word that carries meaning: a verb, mostly.
_SENTENCE_END = (".", "!", "?", ":", "\u201d", '"', "\u2019", "'")
_SHORT_SENTENCE_MAX_WORDS = 4
_TITLE_PARTICLES = frozenset({
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for", "by",
    "with", "from", "into", "upon", "de", "du", "da", "di", "la", "le", "von",
    "van", "is", "as", "vs",
})


_WORD_WRAP = ".,;:!?\"'\u201c\u201d\u2018\u2019_()[]"


def _bare(word: str) -> str:
    """The word without the punctuation, underscores and brackets around it:
    "(_grimly_)." is "grimly"."""
    return word.strip(_WORD_WRAP)


def _lower_content_words(words: list[str]) -> int:
    return sum(
        1 for w in words
        if _bare(w)[:1].islower() and _bare(w).lower() not in _TITLE_PARTICLES
    )


def _short_sentence(text: str) -> bool:
    words = text.split()
    if (not 2 <= len(words) <= _SHORT_SENTENCE_MAX_WORDS
            or not text.endswith(_SENTENCE_END)):
        return False
    return _lower_content_words(words[1:]) > 0


# A sentence of any length: several lower-case words that carry meaning, and
# a full stop. "Thus ends BOOK I. (_Folio_), and now begins BOOK II.
# (_Octavo_)." is half capitals by the ratio test and was a chapter of
# Moby-Dick twice; so was a Latin epigraph with its citation in Suetonius.
_SENTENCE_MIN_LOWER_WORDS = 2

# Blocks that are never a heading whatever their words: the tail of an
# illustration block that a blank line split ("_Reading Jane's Letters._ /
# ]" was chapter one of Pride and Prejudice), a parenthesis ("(_As told at
# the Golden Inn._)"), and a rule of dashes opening an epigraph.
_NEVER_HEADING_RE = re.compile(
    r"^\s*[\(\[].*[\)\]]\s*$"      # wrapped in brackets
    r"|^[^\[]*\]"                     # closes a bracket it never opened
    r"|^\s*-{3,}"                     # a rule of dashes
    r"|^\s*\*(?:\s+\*)+\s*$",       # a row of asterisks
    re.DOTALL,
)


def _reads_as_prose(text: str) -> bool:
    words = text.split()
    if _short_sentence(text):
        return True
    if _NEVER_HEADING_RE.search(text):
        return True
    if text.endswith(_SENTENCE_END) and _lower_content_words(words) >= _SENTENCE_MIN_LOWER_WORDS:
        return True
    if len(words) <= _PROSE_MIN_WORDS or not re.search(r"[a-z]", text):
        return False
    lower = sum(1 for w in words if w[:1].islower())
    return lower / len(words) > _PROSE_LOWER_RATIO


def extract(body_lines: list[str]) -> list[Candidate]:
    """Every block in body_lines that could be a heading, in document order."""
    out: list[Candidate] = []
    prev_line = 0
    prose_since = 0
    i, n = 0, len(body_lines)

    while i < n:
        if not body_lines[i].strip():
            i += 1
            continue
        j = i
        while j < n and body_lines[j].strip():
            j += 1
        block = [body_lines[k].strip() for k in range(i, j)]
        emitted = False
        if out and _is_paragraph(block):
            # A paragraph after the last candidate: it is followed by prose.
            if prose_since == 0:
                out[-1] = replace(out[-1], prose_after=True)
            prose_since += 1

        if _heading_sized(block):
            text = " ".join(block)
            flags = _flags(block, text, body_lines, i)
            # A block the regexes recognize is kept whatever it looks like, so
            # the noise filters can never cost recall on a known heading shape.
            known = any(
                f.startswith(("regex:", "front-matter", "back-matter", "toc-entry"))
                for f in flags
            )
            noise = (_reads_as_prose(text) or text.startswith(_OPENS_QUOTED)
                     or bool(_IMAGE_MARKER_RE.match(text)))
            if known or not noise:
                out.append(Candidate(
                    ordinal=len(out),
                    line=i,
                    n_lines=len(block),
                    text=" / ".join(block),
                    flags=flags,
                    gap_before=i - prev_line,
                    prose_before=prose_since,
                ))
                prev_line = i
                prose_since = 0
                emitted = True
        i = j

    return _mark_toc_runs(out)


def _heading_sized(block: list[str]) -> bool:
    return (len(block) <= MAX_BLOCK_LINES
            and max(len(x) for x in block) <= MAX_BLOCK_LINE_CHARS
            and sum(len(x) for x in block) <= MAX_BLOCK_CHARS)


def _is_paragraph(block: list[str]) -> bool:
    """A block of running text, as opposed to a caption or a heading.

    Anything too big to be a heading; or a line of dialogue, which opens
    with a quote; or a wrapped block that reads as prose and ends the way a
    sentence ends ("...grumbled Jo, / lying on the rug." — two lines, short
    enough to pass the size test). A caption is none of these: "The
    procession set out" is one line, and a caption that wrapped ("I used
    to be so frightened when it was my turn to sit in / the big chair")
    has no full stop, so a listing of them is not broken by them.
    """
    if not _heading_sized(block):
        return True
    text = " ".join(block)
    if text.startswith(_OPENS_QUOTED):
        return True
    return (len(block) >= 2 and _reads_as_prose(text)
            and text.endswith(_SENTENCE_END))


def _packed(c: Candidate) -> bool:
    """Whether this candidate continues a listing from the previous one.

    By line gap, or — further apart — with nothing but short lines between
    them. PG 37106's List of Illustrations runs 200 entries two lines apart,
    but most captions read as prose ("The procession set out") and are not
    candidates, so by candidate-to-candidate gap the run broke into pieces
    too short to mark, and the model took forty entries for chapters. What
    never sits between two entries of a listing is a paragraph.
    """
    if c.gap_before <= TOC_RUN_MAX_GAP:
        return True
    return c.prose_before == 0 and c.gap_before <= TOC_RUN_MAX_GAP_NO_PROSE


def _is_heading_over_text(cands: list[Candidate], k: int) -> bool:
    """A heading the regex recognizes with a paragraph directly under it.

    That is a chapter, whatever is packed above it: PG 3296's title page
    sits five lines over "BOOK I", six short blocks in a row, and "BOOK I"
    has the Confessions under it; PG 6400's sits over "PREFACE" the same
    way. An entry in a listing has another entry under it. Such a candidate
    is never part of a run.

    A two-line heading ("I." over "PLAYING PILGRIMS.") has its title under
    the numeral and the paragraph under the title, so it is judged on the
    title line — and the title line goes with it.
    """
    c = cands[k]
    if not any(f.startswith("regex:") or f in ("front-matter-word", "back-matter-word")
               for f in c.flags):
        return False
    if c.prose_after:
        return True
    if "regex:two-line" in c.flags and k + 1 < len(cands):
        title = cands[k + 1]
        return title.line - c.line <= _TWO_LINE_SPAN and title.prose_after
    return False


def _heading_span(cands: list[Candidate], k: int) -> int:
    """How many candidates the heading at k occupies: 2 for a two-line one."""
    c = cands[k]
    if ("regex:two-line" in c.flags and k + 1 < len(cands)
            and cands[k + 1].line - c.line <= _TWO_LINE_SPAN):
        return 2
    return 1


def _mark_toc_runs(cands: list[Candidate]) -> list[Candidate]:
    """Flag candidates belonging to a densely packed run — a contents listing.

    Told in prose that a contents block is densely packed, a model still picks it:
    on PG 2701 it selected all 135 contents entries *as well as* the 135 body
    headings. The density is measurable, so measure it and say so on the line.
    """
    out = list(cands)
    i = 0
    while i < len(out):
        j = i
        if _is_heading_over_text(out, i):
            i += _heading_span(out, i)
            continue
        while (j + 1 < len(out)
               and _packed(out[j + 1])
               and not _is_heading_over_text(out, j + 1)
               and "illustration" not in out[j + 1].flags):
            j += 1
        if j - i + 1 >= TOC_RUN_MIN_LEN:
            for k in range(i, j + 1):
                out[k] = replace(out[k], flags=out[k].flags + ("toc-run",))
        i = j + 1
    return out


def render(cands: list[Candidate]) -> str:
    """The exact view a model is shown."""
    return "\n".join(c.render() for c in cands)
