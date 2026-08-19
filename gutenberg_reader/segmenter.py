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

# TTS voice drifts over long spans: past a few hundred characters the voice at
# the end of a segment no longer matches its beginning. Narration longer than
# this is split at sentence boundaries into chunks packed up to the limit —
# but never mid-sentence, so a single monstrous sentence stays whole.
MAX_NARRATION_CHARS = 400

# Tokens whose trailing period does not end a sentence. Lowercase, no dot.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "esq", "capt", "col", "gen", "lieut",
    "sergt", "rev", "hon", "prof", "messrs", "mme", "mlle", "viz", "etc",
    "vs", "jr", "sr", "no", "vol", "chap", "op",
}

# A sentence ends at terminal punctuation (plus any closing quotes/brackets)
# followed by whitespace.
_SENTENCE_END_RE = re.compile(r"[.!?…]+[”’\"')\]]*\s+")


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


def split_sentences(text: str) -> list[str]:
    """Split unwrapped text into sentences, conservatively.

    A candidate break is terminal punctuation followed by whitespace. It is
    vetoed when the preceding token is a known abbreviation ("Mr.", "Capt."),
    a single-letter initial ("J. Ross Browne"), or when what follows starts
    lowercase (ellipses and interrobang-ish constructions mid-sentence).
    Wrong-but-conservative beats eager: a missed break merely leaves a chunk
    longer, while a false break cuts a name in half.
    """
    sentences: list[str] = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(text):
        # Token carrying the terminal punctuation
        head = text[start:m.start() + 1]
        last_token = head.rsplit(None, 1)[-1] if head.split() else ""
        word = last_token.rstrip(".!?…”’\"')]").lstrip("“‘\"'([").lower()
        if last_token.endswith(".") and word in _ABBREVIATIONS:
            continue
        if last_token.endswith(".") and len(word) == 1 and word.isalpha():
            continue
        nxt = text[m.end():m.end() + 1]
        if nxt.islower():
            continue
        sentences.append(text[start:m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_long_narration(segments: list[dict], max_chars: int = MAX_NARRATION_CHARS) -> list[dict]:
    """Split narration segments longer than max_chars at sentence boundaries.

    Sentences pack greedily up to max_chars per chunk; a single sentence over
    the limit stays whole (leniency over mid-sentence cuts). Dialogue is left
    alone: its speaker labels, QUOTE_CONTINUES notes, and adjacency to
    attribution tags all assume the quoted span is one segment.
    """
    out: list[dict] = []
    for seg in segments:
        if seg["type"] != "narration" or len(seg["text"]) <= max_chars:
            out.append(seg)
            continue
        chunks: list[str] = []
        cur = ""
        for sentence in split_sentences(seg["text"]):
            if cur and len(cur) + 1 + len(sentence) > max_chars:
                chunks.append(cur)
                cur = sentence
            else:
                cur = f"{cur} {sentence}" if cur else sentence
        if cur:
            chunks.append(cur)
        for chunk in chunks:
            out.append({**seg, "text": chunk})
    return out


def segment_text(text: str, quote_pair: tuple[str, str] | None = None) -> list[dict]:
    """Segment a chapter's text into narration/dialogue segment dicts."""
    if quote_pair is None:
        quote_pair = detect_quote_pair(text)

    segments: list[dict] = []
    for para_idx, para in enumerate(split_paragraphs(text)):
        if quote_pair is None:
            para_segments = [{
                "type": "narration",
                "text": para,
                "speaker": None,
                "pronunciation_hints": [],
                "notes": None,
            }]
        else:
            para_segments = segment_paragraph(para, *quote_pair)
        # Which paragraph a segment came from is real typographic evidence for
        # attribution (narration and a quote sharing a paragraph usually share
        # a subject); kept on the working dicts, dropped from the final model.
        for seg in para_segments:
            seg["para"] = para_idx
        segments.extend(para_segments)
    return split_long_narration(segments)
