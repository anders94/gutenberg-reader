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
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Candidate:
    """One block that could be a heading, and every cheap signal about it."""

    ordinal: int          # 0..N-1 — what a model addresses, never a line number
    line: int             # 0-based index into body_lines
    n_lines: int          # 1 or 2
    text: str             # the block, lines joined by " / "
    flags: tuple[str, ...]
    gap_before: int       # lines since the previous candidate

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


def _reads_as_prose(text: str) -> bool:
    words = text.split()
    if len(words) <= _PROSE_MIN_WORDS or not re.search(r"[a-z]", text):
        return False
    lower = sum(1 for w in words if w[:1].islower())
    return lower / len(words) > _PROSE_LOWER_RATIO


def extract(body_lines: list[str]) -> list[Candidate]:
    """Every block in body_lines that could be a heading, in document order."""
    out: list[Candidate] = []
    prev_line = 0
    i, n = 0, len(body_lines)

    while i < n:
        if not body_lines[i].strip():
            i += 1
            continue
        j = i
        while j < n and body_lines[j].strip():
            j += 1
        block = [body_lines[k].strip() for k in range(i, j)]

        if (len(block) <= MAX_BLOCK_LINES
                and max(len(x) for x in block) <= MAX_BLOCK_LINE_CHARS
                and sum(len(x) for x in block) <= MAX_BLOCK_CHARS):
            text = " ".join(block)
            flags = _flags(block, text, body_lines, i)
            # A block the regexes recognize is kept whatever it looks like, so
            # the noise filters can never cost recall on a known heading shape.
            known = any(
                f.startswith(("regex:", "front-matter", "back-matter", "toc-entry"))
                for f in flags
            )
            noise = _reads_as_prose(text) or text.startswith(_OPENS_QUOTED)
            if known or not noise:
                out.append(Candidate(
                    ordinal=len(out),
                    line=i,
                    n_lines=len(block),
                    text=" / ".join(block),
                    flags=flags,
                    gap_before=i - prev_line,
                ))
                prev_line = i
        i = j

    return out


def render(cands: list[Candidate]) -> str:
    """The exact view a model is shown."""
    return "\n".join(c.render() for c in cands)
