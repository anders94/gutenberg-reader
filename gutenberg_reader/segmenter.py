"""Deterministic narration/dialogue segmentation based on quotation marks.

Quotation marks delimit dialogue mechanically — no LLM is needed (or wanted)
for the splitting step. This guarantees:
  - every character of the source text appears in exactly one segment
  - reported speech ("Mr. Bennet replied that he had not.") is narration,
    because it contains no quotes
  - attribution tags ("said his wife,") between two quoted spans are their
    own narration segments

Speaker attribution is done afterwards (deterministic anchors + LLM review).
"""

from __future__ import annotations
import re

# note value set on a dialogue segment whose quotation continues into the
# next paragraph (Gutenberg convention: no closing quote at paragraph end,
# continuation paragraph re-opens with a quote). Downstream alternation
# logic treats the next dialogue segment as the same speaker.
QUOTE_CONTINUES = "quote-continues"


def split_paragraphs(text: str) -> list[str]:
    """Split into paragraphs, unwrapping Gutenberg's ~70-char line wrapping."""
    paras = re.split(r"\n\n+", text.strip())
    return [" ".join(p.split()) for p in paras if p.strip()]


def detect_quote_pair(text: str) -> tuple[str, str] | None:
    """Pick the dominant dialogue quote style for this text.

    Preference order: curly double, straight double, curly single.
    Returns (open_char, close_char) or None if the text has no dialogue.
    """
    if text.count("“") >= 2:
        return ("“", "”")
    if text.count('"') >= 2:
        return ('"', '"')
    if text.count("‘") >= 2:
        return ("‘", "’")
    return None


def _is_close_quote(text: str, i: int, close_q: str) -> bool:
    """True if text[i] acts as a closing quote (not an apostrophe)."""
    if text[i] != close_q:
        return False
    if close_q != "’":
        return True
    # Curly single close is also the apostrophe: don’t, Bennet’s.
    # An apostrophe sits between two letters; a closing quote does not.
    prev_alpha = i > 0 and text[i - 1].isalpha()
    next_alpha = i + 1 < len(text) and text[i + 1].isalpha()
    return not (prev_alpha and next_alpha)


def segment_paragraph(para: str, open_q: str, close_q: str) -> list[dict]:
    """Split one (unwrapped) paragraph into narration/dialogue segments."""
    segments: list[dict] = []
    straight = open_q == close_q
    in_quote = False
    span_start = 0

    def emit(kind: str, start: int, end: int, note: str | None = None) -> None:
        chunk = para[start:end].strip()
        if chunk:
            segments.append({
                "type": kind,
                "text": chunk,
                "speaker": None,
                "pronunciation_hints": [],
                "notes": note,
            })

    i = 0
    while i < len(para):
        c = para[i]
        if not in_quote:
            if c == open_q:
                emit("narration", span_start, i)
                span_start = i
                in_quote = True
        else:
            closes = (c == close_q) if straight else _is_close_quote(para, i, close_q)
            if closes:
                emit("dialogue", span_start, i + 1)
                span_start = i + 1
                in_quote = False
        i += 1

    if in_quote:
        # Unclosed quote: the quotation continues into the next paragraph
        emit("dialogue", span_start, len(para), note=QUOTE_CONTINUES)
    else:
        emit("narration", span_start, len(para))

    return segments


def segment_text(text: str, quote_pair: tuple[str, str] | None = None) -> list[dict]:
    """Segment a chapter's text into narration/dialogue segment dicts."""
    if quote_pair is None:
        quote_pair = detect_quote_pair(text)

    segments: list[dict] = []
    for para in split_paragraphs(text):
        if quote_pair is None:
            segments.append({
                "type": "narration",
                "text": para,
                "speaker": None,
                "pronunciation_hints": [],
                "notes": None,
            })
        else:
            segments.extend(segment_paragraph(para, *quote_pair))
    return segments
