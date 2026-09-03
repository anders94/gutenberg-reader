"""Deterministic narration/dialogue segmentation based on quotation marks.

Quotation marks delimit dialogue mechanically — no LLM is needed (or wanted)
for the splitting step. This guarantees:
  - every character of the source text appears in exactly one segment
  - reported speech ("Mr. Bennet replied that he had not.") is narration,
    because it contains no quotes
  - attribution tags ("said his wife,") between two quoted spans are their
    own narration segments

Speaker attribution is done afterwards (deterministic anchors + LLM review).

Segments carry offsets into a canonical `reading_text`, and their text is always
a slice of it rather than a string anyone assembled. That is what makes coverage
checkable exactly: spans either tile the chapter or they do not. The previous
check rebuilt the chapter with `" ".join(segment texts)` and compared, which is
lossy — a quoted word segmented on its own came back as `"Deserters" :` against
an original `"Deserters":`, and reported text corruption where there was none.
"""

from __future__ import annotations
import re

from gutenberg_reader import text_utils

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


def normalize_chapter(chapter_text: str) -> tuple[str, list[tuple[int, int]]]:
    """Return (reading_text, paragraph spans).

    reading_text is the chapter with Gutenberg's line wrapping undone and
    paragraphs joined by a blank line. It is the canonical string every offset
    indexes, because the wrapping means a segment is not a substring of the file
    on disk. Written once per chapter so the offsets have something stable and
    inspectable to point at.
    """
    # Illustration captions are not read aloud, and stage 03 already strips them
    # from the chapter files. Doing it here too makes the invariant hold for any
    # caller rather than only for the pipeline's own path, and it is idempotent.
    chapter_text = text_utils.strip_illustration_blocks(chapter_text)
    paras = split_paragraphs(chapter_text)
    spans: list[tuple[int, int]] = []
    pos = 0
    for para in paras:
        spans.append((pos, pos + len(para)))
        pos += len(para) + 2          # the "\n\n" join
    return "\n\n".join(paras), spans


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


def segment_paragraph(
    para: str, open_q: str, close_q: str, base: int = 0
) -> list[dict]:
    """Split one (unwrapped) paragraph into narration/dialogue segments.

    Offsets are absolute: base is where this paragraph begins in reading_text.
    """
    segments: list[dict] = []
    straight = open_q == close_q
    in_quote = False
    span_start = 0

    def emit(kind: str, start: int, end: int, note: str | None = None) -> None:
        # Trim to the non-space content, but record where it actually sits
        # rather than handing back a detached string.
        chunk = para[start:end]
        lead = len(chunk) - len(chunk.lstrip())
        trail = len(chunk) - len(chunk.rstrip())
        s_, e_ = start + lead, end - trail
        if e_ > s_:
            segments.append({
                "type": kind,
                "start": base + s_,
                "end": base + e_,
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
    return [text[a:b] for a, b in sentence_spans(text)]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence boundaries as (start, end) offsets into text, whitespace trimmed.

    Same rules as split_sentences, which is now a thin wrapper: a break needs
    terminal punctuation and whitespace, and is vetoed after an abbreviation, a
    single-letter initial, or before a lowercase word.
    """
    spans: list[tuple[int, int]] = []
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
        spans.append(_trimmed(text, start, m.end()))
        start = m.end()
    if text[start:].strip():
        spans.append(_trimmed(text, start, len(text)))
    return [sp for sp in spans if sp[1] > sp[0]]


def _trimmed(text: str, start: int, end: int) -> tuple[int, int]:
    chunk = text[start:end]
    lead = len(chunk) - len(chunk.lstrip())
    trail = len(chunk) - len(chunk.rstrip())
    return start + lead, end - trail


def split_long_narration(
    segments: list[dict],
    reading_text: str,
    max_chars: int = MAX_NARRATION_CHARS,
) -> list[dict]:
    """Split narration segments longer than max_chars at sentence boundaries.

    Sentences pack greedily up to max_chars per chunk; a single sentence over
    the limit stays whole (leniency over mid-sentence cuts). Dialogue is left
    alone: its speaker labels, QUOTE_CONTINUES notes, and adjacency to
    attribution tags all assume the quoted span is one segment.
    """
    out: list[dict] = []
    for seg in segments:
        length = seg["end"] - seg["start"]
        if seg["type"] != "narration" or length <= max_chars:
            out.append(seg)
            continue
        base = seg["start"]
        body = reading_text[seg["start"]:seg["end"]]
        chunk: tuple[int, int] | None = None
        for a, b in sentence_spans(body):
            if chunk and (b - chunk[0]) > max_chars:
                out.append({**seg, "start": base + chunk[0], "end": base + chunk[1]})
                chunk = (a, b)
            else:
                chunk = (chunk[0], b) if chunk else (a, b)
        if chunk:
            out.append({**seg, "start": base + chunk[0], "end": base + chunk[1]})
    return out


def segment_text(
    text: str, quote_pair: tuple[str, str] | None = None
) -> tuple[str, list[dict]]:
    """Segment a chapter into (reading_text, segments).

    Every segment carries start/end offsets into reading_text and a "text" that
    is exactly that slice — derived here, never assembled.
    """
    if quote_pair is None:
        quote_pair = detect_quote_pair(text)

    reading_text, para_spans = normalize_chapter(text)
    # Paragraphs are sliced from reading_text, not re-derived from the input:
    # normalize_chapter strips illustration blocks, so splitting the original
    # again would hand back paragraphs that no longer line up with the spans.
    segments: list[dict] = []
    for para_idx, (base, end) in enumerate(para_spans):
        para = reading_text[base:end]
        if quote_pair is None:
            para_segments = [{
                "type": "narration",
                "start": base,
                "end": end,
                "speaker": None,
                "pronunciation_hints": [],
                "notes": None,
            }]
        else:
            para_segments = segment_paragraph(para, *quote_pair, base=base)
        # Which paragraph a segment came from is real typographic evidence for
        # attribution (narration and a quote sharing a paragraph usually share
        # a subject); kept on the working dicts, dropped from the final model.
        for seg in para_segments:
            seg["para"] = para_idx
        segments.extend(para_segments)

    segments = split_long_narration(segments, reading_text)
    for seg in segments:
        seg["text"] = reading_text[seg["start"]:seg["end"]]
    return reading_text, segments


# A speech verb next to a quoted span is strong evidence it is speech; a very
# short span with no speech verb anywhere in its paragraph is almost always a
# term. Only what falls between the two is worth asking about.
_SPEECH_VERB_RE = re.compile(
    r"\b(said|says|say|saying|cried|cries|replied|answered|asked|asks|"
    r"exclaimed|shouted|whispered|murmured|added|continued|declared|"
    r"remarked|observed|responded|retorted|urged|begged|inquired)\b",
    re.IGNORECASE,
)
# Being the object of one of these marks the quoted span as a name, not speech.
_NAMING_VERB_RE = re.compile(
    r"\b(called|call|calls|named|names|termed|term|terms|styled|"
    r"known as|word|words|phrase|meaning|means)\b",
    re.IGNORECASE,
)
# A speech verb usually settles a span as speech, but these turn the same
# sentence into a citation: "Homer says" in a history is a source being quoted,
# not a character speaking in the scene.
# Present-tense attribution is the tell. Prose narrates speech in the past
# ("said", "cried", "replied") and cites a source in the present ("Homer says",
# "Herodotus tells us"), because the source goes on saying it. Not conclusive —
# present-tense narration exists — so this only forces the question rather than
# deciding it.
_CITATION_CUE_RE = re.compile(
    r"\b(says|say|writes|write|tells|relates|records|reports|calls it|"
    r"wrote|written|recorded|according to|"
    r"oracle|inscription|epitaph|verse|verses|poem|poet|quoted|quotes|"
    r"scripture|proverb|epigram|letter ran|runs thus|as follows)\b",
    re.IGNORECASE,
)
# A gloss: the quoted span is what a foreign word means. Herodotus does this
# constantly — "_Asmach_ signifies 'those who stand on the left hand of the
# king'" — and the span can be a whole clause, so unlike the other term signals
# this one settles at any length.
_GLOSS_RE = re.compile(
    r"\b(signifies|signifying|means|meaning|translated|rendered|"
    r"in the tongue of|is to say|equivalent to)\b",
    re.IGNORECASE,
)
# "This city is said to be the mother-city" is not an attribution tag. Passive
# and impersonal forms of "say" report hearsay about the world, not speech by
# anyone, and reading them as speech verbs made a scare-quoted term look spoken.
_HEARSAY_RE = re.compile(
    r"\b(is|are|was|were|it\s+is|they\s+are)\s+said\b", re.IGNORECASE
)
# Things that get given speech in devotional and philosophical writing without
# anyone being present to say it: "truth saith unto me", "a violent habit
# whispered", "the whole air answered". A speech verb beside one of these settles
# nothing — the span goes to the model, which decides whether it is rhetorical.
# Deliberately a cue to ASK rather than a cue to decide: adding a word here can
# only cost a call, never produce a wrong answer, which is what separates this
# from the pattern list that broke chapter detection.
_ABSTRACT_SPEAKER_RE = re.compile(
    r"\b(truth|habit|reason|wisdom|conscience|memory|soul|spirit|nature|"
    r"world|heaven|earth|air|sea|sky|light|darkness|sense|senses|flesh|"
    r"law|justice|beauty|time|death|life|voice|creation|things)\b",
    re.IGNORECASE,
)
UNAMBIGUOUS_TERM_WORDS = 2


def classify_spans_deterministically(
    segments: list[dict], reading_text: str
) -> tuple[dict[int, str], list[int]]:
    """Split dialogue segments into the settled ones and the ones worth asking.

    Returns ({segment index: label}, [indices needing judgment]). Cheap and
    conservative: it only settles a span when the sentence around it says plainly
    what the span is.
    """
    settled: dict[int, str] = {}
    ask: list[int] = []
    by_para: dict[int, list[int]] = {}
    for i, seg in enumerate(segments):
        by_para.setdefault(seg.get("para", 0), []).append(i)

    for i, seg in enumerate(segments):
        if seg["type"] != "dialogue":
            continue
        para_text = " ".join(
            reading_text[segments[j]["start"]:segments[j]["end"]]
            for j in by_para.get(seg.get("para", 0), [])
        )
        words = len(reading_text[seg["start"]:seg["end"]].split())
        # Hearsay first: strip "is said to be" before looking for a speech verb,
        # or the passive supplies one that nobody spoke.
        speech_verb = bool(_SPEECH_VERB_RE.search(_HEARSAY_RE.sub(" ", para_text)))
        naming_verb = bool(_NAMING_VERB_RE.search(para_text))

        # Only the unambiguous cases are settled here; anything with evidence
        # pointing both ways, or none, is worth a question.
        abstract_subject = bool(_ABSTRACT_SPEAKER_RE.search(para_text))

        if _GLOSS_RE.search(para_text) and not speech_verb:
            # A definition, whatever its length.
            settled[i] = "term"
        elif _CITATION_CUE_RE.search(para_text):
            ask.append(i)
        elif speech_verb and not naming_verb and not abstract_subject:
            settled[i] = "speech"
        elif naming_verb and not speech_verb and words <= UNAMBIGUOUS_TERM_WORDS:
            settled[i] = "term"
        else:
            ask.append(i)
    return settled, ask


def apply_span_labels(
    segments: list[dict], reading_text: str, labels: dict[int, str]
) -> list[dict]:
    """Turn a `term`/`title` verdict into the removal of a boundary.

    The span stops being its own segment and re-forms with the narration around
    it, punctuation and all — so `"Deserters":` comes back whole. Nothing is
    repaired because nothing was rebuilt: the merged segment is simply a wider
    slice of the same reading_text.

    Only the narration touching a demoted span is joined. Merging every adjacent
    narration pair would undo split_long_narration, which exists to keep the TTS
    voice stable, so the length limit is re-applied afterwards in case a merge
    produced an over-long run.
    """
    out: list[dict] = []
    for i, seg in enumerate(segments):
        # A rhetorical utterance joins term and title in losing its boundary:
        # it is quoted, but nobody in the scene is saying it, and giving an
        # abstraction its own voice would confuse a listener rather than help.
        # It reads in the narrator's voice, which is whose rhetoric it is.
        demote = seg["type"] == "dialogue" and labels.get(i) in (
            "term", "title", "rhetorical")
        if not demote:
            if labels.get(i) == "citation" and seg["type"] == "dialogue":
                # Quoted, so still read as quoted text — but there is nobody in
                # the scene saying it, so it is settled here rather than being
                # sent through attribution to come back "Unknown".
                seg = {**seg, "notes": "citation",
                       "speaker": text_utils.CITATION_SPEAKER}
            # Absorb narration that follows a just-demoted span.
            if (out and out[-1].pop("_demoted", False)
                    and seg["type"] == "narration"
                    and seg["para"] == out[-1]["para"]
                    and not reading_text[out[-1]["end"]:seg["start"]].strip()):
                out[-1] = {**out[-1], "end": seg["end"]}
                continue
            out.append(dict(seg))
            continue

        if (out and out[-1]["type"] == "narration"
                and out[-1]["para"] == seg["para"]):
            out[-1] = {**out[-1], "end": seg["end"], "_demoted": True}
        else:
            out.append({**seg, "type": "narration", "speaker": None, "_demoted": True})

    for seg in out:
        seg.pop("_demoted", None)

    out = split_long_narration(out, reading_text)
    for seg in out:
        seg["text"] = reading_text[seg["start"]:seg["end"]]
    return out
