"""Text processing utilities: boilerplate stripping, chunking, integrity check."""

from __future__ import annotations
import difflib
import re
from typing import Generator


# ── Gutenberg boilerplate markers ────────────────────────────────────────────

START_MARKER_RE = re.compile(
    r"\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE
)
END_MARKER_RE = re.compile(
    r"\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE
)

ILLUSTRATION_RE = re.compile(r"\[Illustration[^\[\]]*\]", re.IGNORECASE)


def strip_illustration_blocks(text: str) -> str:
    """Remove [Illustration: ...] blocks, handling nested brackets."""
    result = []
    depth = 0
    i = 0
    while i < len(text):
        if text[i] == "[":
            # Check if this is an illustration block
            snippet = text[i:i + 15].lower()
            if snippet.startswith("[illustration"):
                depth = 1
                i += 1
                while i < len(text) and depth > 0:
                    if text[i] == "[":
                        depth += 1
                    elif text[i] == "]":
                        depth -= 1
                    i += 1
                continue
        result.append(text[i])
        i += 1
    return "".join(result)


def find_body_bounds(lines: list[str]) -> tuple[int, int]:
    """Return (start_line_idx, end_line_idx) for the body after boilerplate removal.

    Indices are 0-based into the lines list.
    The start is the line after the START marker, end is the line before END marker.
    """
    start_idx = 0
    end_idx = len(lines)

    for i, line in enumerate(lines):
        if START_MARKER_RE.search(line):
            start_idx = i + 1
            break

    for i in range(len(lines) - 1, -1, -1):
        if END_MARKER_RE.search(lines[i]):
            end_idx = i
            break

    return start_idx, end_idx


def extract_preamble_metadata(preamble: str) -> dict[str, str]:
    """Extract metadata fields from the Gutenberg preamble."""
    meta: dict[str, str] = {}

    # Single-line patterns (capture only to end of line)
    single_line_patterns = {
        "title": r"^Title:\s*(.+)$",
        "author": r"^Author:\s*(.+)$",
        "language": r"^Language:\s*(.+)$",
    }
    for key, pattern in single_line_patterns.items():
        m = re.search(pattern, preamble, re.IGNORECASE | re.MULTILINE)
        if m:
            meta[key] = m.group(1).strip()

    # Release date: capture up to first bracket or end of line
    m = re.search(r"^Release [Dd]ate:\s*(.+?)(?:\s*\[|$)", preamble, re.MULTILINE)
    if m:
        meta["release_date"] = m.group(1).strip()

    # Credits: multi-line, from "Produced by" to next blank line
    m = re.search(r"(?:Produced by|Credits?:)\s*(.+?)(?:\n\n|\Z)", preamble, re.IGNORECASE | re.DOTALL)
    if m:
        meta["credits"] = m.group(1).strip().replace("\n", " ")

    return meta


# ── Chapter detection ─────────────────────────────────────────────────────────

# Many books carry the title on the heading line itself ("Chapter 1. Marseilles—The
# Arrival"), so the numeral may be followed by an optional same-line title. Two
# guards keep prose from being mistaken for a heading: the title must begin like a
# title (capital, digit, or opening quote), and after a roman numeral it must be
# introduced by punctuation — otherwise "Chapter I am late" would qualify.
_TITLE_BODY = r"[A-Z0-9“‘\"'(\[].*"
_TITLE_AFTER_PUNCT = rf"(?:\s*[.:—–-]\s*(?:{_TITLE_BODY})?)?"
_TITLE_AFTER_SPACE = rf"(?:\s*[.:—–-]?\s*(?:{_TITLE_BODY})?)?"

# Some books number their divisions with a bare roman numeral and no keyword
# at all ("I. A SCANDAL IN BOHEMIA"). That shape is far weaker evidence than a
# CHAPTER/PART/BOOK line — 'M.' is a roman numeral, so it matches French
# "M. Morrel, and this day and a half was lost..." — so two extra guards apply:
# the title is REQUIRED (bare "I." is a section marker inside a story, not a
# story heading) and must be ALL CAPS, which is what makes such a line read as
# a heading typographically. Without them, PG 1184 gains 8 phantom chapters
# and PG 2701 four.
_ALL_CAPS_TITLE = r"[A-Z][A-Z0-9\s—–\-’'\".,:;!?()]*[A-Z0-9.!?)\"’]"

# A few divisions are named rather than numbered. They are part of the story —
# Moby Dick's Epilogue ("AND I ONLY AM ESCAPED ALONE TO TELL THEE") is where
# Ishmael explains how a drowned crew has a narrator, and a listener expects it
# as its own track — so they are body chapters, not front or back apparatus.
# Deliberately just these two: "Conclusion" and "Afterword" are ordinary enough
# words that matching them costs more than it gains.
_NAMED_DIVISIONS = r"EPILOGUE|PROLOGUE"

CHAPTER_PATTERNS = [
    re.compile(rf"^(CHAPTER\s+[IVXLCDM]+{_TITLE_AFTER_PUNCT})\s*$"),
    re.compile(rf"^(CHAPTER\s+\d+{_TITLE_AFTER_SPACE})\s*$"),
    re.compile(rf"^(Chapter\s+\d+{_TITLE_AFTER_SPACE})\s*$"),
    re.compile(rf"^(Chapter\s+[IVXLCDM]+{_TITLE_AFTER_PUNCT})\s*$"),
    re.compile(rf"^(PART\s+[IVXLCDM]+{_TITLE_AFTER_PUNCT})\s*$"),
    re.compile(rf"^(BOOK\s+[IVXLCDM]+{_TITLE_AFTER_PUNCT})\s*$"),
    re.compile(rf"^([IVXLCDM]+\.\s+{_ALL_CAPS_TITLE})\s*$"),
    re.compile(rf"^((?:{_NAMED_DIVISIONS}){_TITLE_AFTER_PUNCT})\s*$", re.IGNORECASE),
]

# A contents entry for the bare-numeral form above: indented, numeral, then a
# title in any case ("   I.     A Scandal in Bohemia"). The heading pattern
# deliberately will not match these — they are title case — so the front-matter
# walk needs its own way to see them, or a 12-entry contents block reads as
# narrative and becomes a synthetic chapter one.
TOC_NUMERAL_ENTRY_RE = re.compile(r"^\s+[IVXLCDM]+\.\s+\S")

# Editions carry their own apparatus around the story: prefaces and dedications
# before it, footnotes and indexes after. Both read as prose to the chapter
# detector, so they are recognized by their headings instead. Anchored to the
# whole line: a heading is a line that says only this.
FRONT_MATTER_RE = re.compile(
    r"^\s*(PREFACE|INTRODUCTION|DEDICATION|FOREWORD|ETYMOLOGY|EXTRACTS|"
    r"TRANSLATOR.?S?\s+NOTE|CONTENTS?|LIST OF ILLUSTRATIONS|ILLUSTRATIONS)"
    r"\b[\w\s’']*[:.]?\s*$",
    re.IGNORECASE,
)
# A transcriber's note usually carries a qualifier ("Original Transcriber's
# Notes:"), and the heading is anchored, so allow one leading word there.
BACK_MATTER_RE = re.compile(
    r"^\s*(FOOTNOTES?|ENDNOTES?|NOTES?|APPENDIX|APPENDICES|INDEX|GLOSSARY|"
    r"BIBLIOGRAPHY|ERRATA|COLOPHON|(?:\w+\s+)?TRANSCRIBER.?S?\s+NOTES?)\b[:.]?\s*$",
    re.IGNORECASE,
)


def classify_heading(title: str) -> str:
    """Classify a chapter heading as "front", "body", or "back" matter."""
    stripped = title.strip()
    if BACK_MATTER_RE.match(stripped):
        return "back"
    if FRONT_MATTER_RE.match(stripped):
        return "front"
    return "body"


# A table of contents lists every heading a line or two apart; real chapters are
# hundreds of lines apart. Any run of at least TOC_MIN_RUN headings packed within
# TOC_MAX_GAP lines of each other is a contents listing, not the body.
TOC_MAX_GAP = 4
TOC_MIN_RUN = 3


def looks_like_chapter_heading(stripped: str) -> bool:
    """True if an already-stripped line has the shape of a chapter heading."""
    return any(p.match(stripped) for p in CHAPTER_PATTERNS)


def drop_toc_clusters(matches: list[dict]) -> list[dict]:
    """Drop headings belonging to densely packed runs (table-of-contents listings).

    Takes and returns dicts carrying a 0-based "line_idx". If filtering would
    discard everything, the input is returned unchanged — better to over-detect
    than to report a book with no chapters at all.
    """
    n = len(matches)
    if n < TOC_MIN_RUN:
        return matches

    close_to_next = [
        matches[i + 1]["line_idx"] - matches[i]["line_idx"] <= TOC_MAX_GAP
        for i in range(n - 1)
    ]

    keep = [True] * n
    i = 0
    while i < n - 1:
        if not close_to_next[i]:
            i += 1
            continue
        # Walk to the end of the cluster; the final member has no close successor
        # of its own but still belongs to the run.
        j = i
        while j < n - 1 and close_to_next[j]:
            j += 1
        if j - i + 1 >= TOC_MIN_RUN:
            for k in range(i, j + 1):
                keep[k] = False
        i = j + 1

    filtered = [m for m, k in zip(matches, keep) if k]
    return filtered or matches


BARE_NUMERAL_RE = re.compile(r"^[IVXLCDM]+\.?$")

# How many blank lines may sit between the numeral and its title before they
# stop reading as one heading. Editions centre them a line or two apart.
_TWO_LINE_HEADING_GAP = 3


def _two_line_heading(lines: list[str], i: int) -> tuple[str, int] | None:
    """Recognize a numeral line whose title sits on the next non-blank line.

    Returns (combined title, index of the title line), or None. The numeral
    must be alone on its line and the title must be ALL CAPS — a bare numeral
    followed by prose is an in-story section marker, not a chapter heading.
    """
    if not BARE_NUMERAL_RE.match(lines[i].strip()):
        return None

    j = i + 1
    while j < len(lines) and not lines[j].strip() and j - i <= _TWO_LINE_HEADING_GAP:
        j += 1
    if j >= len(lines) or j == i + 1:  # title must be separated by a blank line
        return None

    title = lines[j].strip()
    if not re.fullmatch(_ALL_CAPS_TITLE, title):
        return None
    # A heading stands alone; a caps line opening a paragraph does not.
    if j + 1 < len(lines) and lines[j + 1].strip():
        return None
    return f"{lines[i].strip().rstrip('.')}. {title}", j


def detect_chapters_regex(lines: list[str]) -> list[dict]:
    """Detect chapter boundaries using ordered regex patterns.

    Returns list of dicts with keys: number, title, start_line (1-indexed), start_marker.
    Numbering is positional rather than parsed from the heading, so books that
    restart their count per part cannot produce duplicate chapter numbers.
    """
    matches = []
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Two-line form: a numeral alone, then the title on its own line.
        #     I.
        #
        #     PLAYING PILGRIMS.
        # Neither line is a heading by itself, so PG 37106 matched none of its
        # 47 body headings and fell back to its contents listing. Bare numerals
        # are also in-story section markers (PG 1661 has "I." mid-story), so the
        # all-caps title on the next non-blank line is what separates the two.
        pair = _two_line_heading(lines, i)
        if pair is not None:
            title, end_idx = pair
            matches.append({"line_idx": i, "title": title, "block_len": 1})
            continue

        for pattern in CHAPTER_PATTERNS:
            if pattern.match(stripped):
                # How far the contiguous non-blank block starting here runs: a
                # heading stands alone (1) or wraps its title once (2), while a
                # prose line that merely starts like one ("BOOK I. (_Folio_),
                # CHAPTER I. (_Sperm Whale_).—This whale, among the") opens a
                # full paragraph.
                j = i
                while j + 1 < len(lines) and lines[j + 1].strip():
                    j += 1
                matches.append({
                    "line_idx": i,  # 0-based
                    "title": stripped,
                    "block_len": j - i + 1,
                })
                break

    # TOC filtering first: contents entries sit inside one long non-blank block,
    # so the block-length filter would eat most of a TOC run and leave its tail
    # looking like sparse (real) headings.
    matches = drop_toc_clusters(matches)
    matches = [m for m in matches if m["block_len"] <= 2]

    # Number them sequentially
    result = []
    for idx, m in enumerate(matches):
        result.append({
            "number": idx + 1,
            "title": m["title"],
            "start_line": m["line_idx"] + 1,  # 1-indexed
            "start_marker": m["title"],
        })
    return result


def collapse_blank_lines(text: str, max_blank: int = 2) -> str:
    """Collapse runs of more than max_blank blank lines."""
    pattern = re.compile(r"\n{" + str(max_blank + 2) + r",}")
    return pattern.sub("\n" * (max_blank + 1), text)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text at paragraph boundaries into overlapping chunks.

    chunk_size and overlap are measured in words.
    Chunk boundaries snap to the nearest preceding blank line.
    """
    paragraphs = re.split(r"\n\n+", text.strip())
    chunks: list[str] = []

    current_paras: list[str] = []
    current_words = 0

    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        para_words = len(para.split())

        if current_words + para_words > chunk_size and current_paras:
            # Emit current chunk
            chunks.append("\n\n".join(current_paras))

            # Calculate overlap: keep last N words worth of paragraphs
            overlap_paras: list[str] = []
            overlap_words = 0
            for p in reversed(current_paras):
                pw = len(p.split())
                if overlap_words + pw <= overlap:
                    overlap_paras.insert(0, p)
                    overlap_words += pw
                else:
                    break

            current_paras = overlap_paras
            current_words = overlap_words
        else:
            current_paras.append(para)
            current_words += para_words
            i += 1

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks if chunks else [text]


def build_segment_windows(segments: list[dict], word_budget: int) -> list[tuple[int, int]]:
    """Split segment indices into [start, end) windows of ~word_budget words.

    Every LLM pass over a chapter's segments goes window by window: a chapter is
    not bounded in length (Moby Dick's "The Town-Ho's Story" runs 7,900 words) and
    a whole-chapter prompt overruns the server's context window.
    """
    windows: list[tuple[int, int]] = []
    start = 0
    words = 0
    for i, seg in enumerate(segments):
        w = len(seg["text"].split())
        if words + w > word_budget and i > start:
            windows.append((start, i))
            start = i
            words = 0
        words += w
    windows.append((start, len(segments)))
    return windows


# ── Integrity verification ────────────────────────────────────────────────────

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace for comparison.

    Also treats Gutenberg dashes (-- and em-dash —) as whitespace separators,
    since the model often converts them to spaces and TTS treats them as pauses.
    """
    text = re.sub(r"-{2,}|\u2014", " ", text)  # --, ---, em-dash → space
    return re.sub(r"\s+", " ", text).strip()


def verify_segment_coverage(
    original_text: str,
    segments: list[dict],
) -> tuple[bool, list[str]]:
    """Verify that segment texts cover the original exactly.

    Returns (is_valid, list_of_issues).
    """
    reconstructed = " ".join(s["text"] for s in segments)

    orig_norm = normalize_whitespace(original_text)
    recon_norm = normalize_whitespace(reconstructed)

    if orig_norm == recon_norm:
        return True, []

    # Find diffs
    orig_words = orig_norm.split()
    recon_words = recon_norm.split()

    matcher = difflib.SequenceMatcher(None, recon_words, orig_words)
    issues = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            added = " ".join(recon_words[i1:i2])
            issues.append(f"Extra text in segments (not in original): '{added[:100]}'")
        elif tag == "insert":
            missing = " ".join(orig_words[j1:j2])
            issues.append(f"Missing from segments: '{missing[:100]}'")
        elif tag == "replace":
            orig_span = " ".join(orig_words[j1:j2])
            recon_span = " ".join(recon_words[i1:i2])
            issues.append(f"Text altered: original='{orig_span[:80]}' got='{recon_span[:80]}'")

    return False, issues


def _fps_match(orig_fps: list[str], seg_fps: list[str]) -> bool:
    """Compare two fingerprint sequences, tolerating up to 1 edit-distance error per word.

    This handles model spelling errors like 'suspiciions' vs 'suspicions'.
    """
    if len(orig_fps) != len(seg_fps):
        return False
    mismatches = 0
    for o, s in zip(orig_fps, seg_fps):
        if o != s:
            if levenshtein_distance(o, s) <= 2:
                mismatches += 1
                if mismatches > max(1, len(orig_fps) // 10):
                    return False
            else:
                return False
    return True


def repair_segment_texts(original_text: str, segments: list[dict]) -> list[dict] | None:
    """Try to repair segments whose texts differ from the original only in punctuation.

    Strategy: strip all non-alphanumeric characters from each word to get a "fingerprint",
    match segment words against original words by fingerprint, then use the original words
    verbatim. This restores dropped/altered punctuation (e.g. missing leading quotes).

    Returns repaired segment list, or None if alignment fails.
    """
    def fingerprint(word: str) -> str:
        return re.sub(r"[^\w]", "", word).lower()

    orig_norm = normalize_whitespace(original_text)
    orig_words = orig_norm.split()
    orig_fps = [fingerprint(w) for w in orig_words]

    repaired = []
    orig_pos = 0  # how many orig_words consumed so far

    for seg in segments:
        seg_text = normalize_whitespace(seg.get("text", ""))
        if not seg_text:
            continue

        seg_words = seg_text.split()
        # Build fingerprints for this segment, skipping purely-punctuation tokens
        seg_fps = [fingerprint(w) for w in seg_words]
        content_fps = [f for f in seg_fps if f]  # non-empty fingerprints only

        if not content_fps:
            # Segment is all punctuation — just keep it
            repaired.append(dict(seg))
            continue

        n = len(content_fps)

        # Scan forward in orig_fps to find this run of content fingerprints.
        # Allow a small gap for skipped text (model dropped a sentence).
        MAX_GAP_WORDS = 30
        found = False
        for start in range(orig_pos, min(orig_pos + MAX_GAP_WORDS + n, len(orig_fps) - n + 1)):
            # Collect content fingerprints from orig starting at 'start'
            orig_content_fps = [f for f in orig_fps[start:] if f]
            if _fps_match(orig_content_fps[:n], content_fps):
                # If we skipped some orig words, insert them as a narration segment first
                if start > orig_pos:
                    skipped = " ".join(orig_words[orig_pos:start])
                    repaired.append({
                        "type": "narration",
                        "text": skipped,
                        "speaker": None,
                        "pronunciation_hints": [],
                        "notes": "auto-inserted: model skipped this text",
                    })

                # Determine how many orig_words correspond to these n content fingerprints
                end = start
                matched = 0
                while end < len(orig_fps) and matched < n:
                    if orig_fps[end]:
                        matched += 1
                    end += 1

                orig_slice = " ".join(orig_words[start:end])
                repaired.append({**seg, "text": orig_slice})
                orig_pos = end
                found = True
                break

        if not found:
            return None  # alignment failed

    # Insert any trailing skipped text
    if orig_pos < len(orig_words):
        skipped = " ".join(orig_words[orig_pos:])
        repaired.append({
            "type": "narration",
            "text": skipped,
            "speaker": None,
            "pronunciation_hints": [],
            "notes": "auto-inserted: model skipped trailing text",
        })

    ok, _ = verify_segment_coverage(original_text, repaired)
    return repaired if ok else None


def word_count(text: str) -> int:
    return len(text.split())


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def find_closest_character(name: str, known_chars: list[str], max_distance: int = 2) -> str | None:
    """Find the closest matching character name within edit distance."""
    best = None
    best_dist = max_distance + 1
    name_lower = name.lower()
    for char in known_chars:
        dist = levenshtein_distance(name_lower, char.lower())
        if dist < best_dist:
            best_dist = dist
            best = char
    return best if best_dist <= max_distance else None


# ── Speaker attribution anchor propagation ────────────────────────────────────

_SPEECH_VERBS = (
    r"said|replied|answered|cried|asked|returned|exclaimed|whispered|remarked|"
    r"continued|added|observed|repeated|murmured|laughed|called|declared|"
    r"interposed|interrupted|rejoined|responded|urged|insisted|demanded|"
    r"admitted|confessed|agreed|protested|pleaded|began|concluded|sighed|"
    r"shouted|screamed|muttered|stammered|faltered|ventured|suggested|told"
)

SPEECH_VERB_RE = re.compile(r"\b(" + _SPEECH_VERBS + r")\b", re.IGNORECASE)

# "said Lydia," / "cried Miss Bingley" — a proper name directly after a speech
# verb, for anchoring speakers not (yet) in the known character list.
_NAME_PATTERN = (
    r"((?:(?:Mr|Mrs|Dr|St|Mme|Mlle)\.\s+|(?:Miss|Lady|Sir|Lord|Madame|Colonel|Captain)\s+)?"
    r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)"
)
_TAG_NAME_RE = re.compile(
    r"(?i:\b(?:" + _SPEECH_VERBS + r"))\s+" + _NAME_PATTERN
)
# "Mrs. Bennet said only," / "Miss Bartlett continued;" — subject-first order.
_NAME_TAG_RE = re.compile(
    _NAME_PATTERN + r"\s+(?i:\b(?:" + _SPEECH_VERBS + r")\b)"
)

_REPORTED_SPEECH_RE = re.compile(
    r"\b(said|replied|answered|cried|asked|returned|exclaimed|told|informed|"
    r"acknowledged|admitted)\s+that\b",
    re.IGNORECASE,
)


# Sentence split that doesn't break on honorific abbreviations
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!Mr\.)(?<!Mrs\.)(?<!Dr\.)(?<!St\.)(?<!Mme\.)(?<=[.!?])\s+"
)

# A speech verb this deep into a sentence ("...unable to contain herself,
# began scolding...") is not an attribution-tag construction.
_TAG_VERB_LEAD_WORDS = 6


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]


def _has_lead_speech_verb(sentence: str) -> bool:
    """True if a speech verb appears in the sentence's first few words
    ("said his lady...", "Mrs. Bennet said only,")."""
    lead = " ".join(sentence.split()[:_TAG_VERB_LEAD_WORDS])
    return bool(SPEECH_VERB_RE.search(lead)) and not _REPORTED_SPEECH_RE.search(sentence)


def _is_attribution_narration(seg: dict) -> bool:
    """Return True if this narration segment is a speech attribution tag.

    Attribution tags are short narration segments where some sentence LEADS
    with a speech verb ("said she,", "he continued,", "Mrs. Bennet said only,").
    A speech verb buried mid-sentence is action/beat narration, not a tag, and
    reported-speech constructions ("replied that he had not") never qualify.
    """
    if seg.get("type") != "narration":
        return False
    text = seg.get("text", "")
    if len(text.split()) > 20:
        return False
    return any(_has_lead_speech_verb(s) for s in _split_sentences(text))


def _name_tokens(name: str) -> set[str]:
    return {t.strip(".").lower() for t in name.split()}


def _merge_target(i: int, chars: list) -> int | None:
    """Index of the unique roster entry that entry i duplicates, or None."""
    c = chars[i]
    alias_targets = [
        j for j, d in enumerate(chars)
        if j != i and c.name.lower() in (a.lower() for a in d.aliases)
    ]
    if len(alias_targets) == 1:
        return alias_targets[0]
    c_tokens = _name_tokens(c.name)
    subset_targets = [
        j for j, d in enumerate(chars)
        if j != i and any(
            c_tokens < s or c_tokens == s
            for s in [_name_tokens(d.name), *(_name_tokens(a) for a in d.aliases)]
        )
    ]
    if len(subset_targets) == 1:
        return subset_targets[0]
    return None


PLACEHOLDER_NAME_RE = re.compile(
    r"^\s*(n/?a|none|unknown|nobody|no\b.*\bcharacters?\b.*)\s*$",
    re.IGNORECASE,
)


def is_placeholder_name(name: str) -> bool:
    """True for non-names a model emits when a chapter has no characters.

    A chapter of pure exposition tempts discovery into answering "N/A" or
    "No named characters found" rather than an empty list; such an entry then
    becomes a legal speaker for the whole rest of the book. "Unknown" is also
    a placeholder: it is the attribution fallback, never a character.
    """
    return bool(PLACEHOLDER_NAME_RE.match(name))


# The role the story is told from, and the pronouns a first-person narrator
# refers to themselves by. Leading qualifiers are absorbed: told plainly not to
# emit "Narrator", the model answered "Unnamed narrator" instead, so the role
# noun is what matters, not what decorates it. Still whole-name only —
# "Narrator's Wife" is a real person described by relation, and ends on the
# noun that names her.
NARRATOR_ROLE_RE = re.compile(
    r"^\s*(?:(?:the|a|an|our|its|this|unnamed|anonymous|unidentified|main|primary"
    r"|first[-\s]person)\s+)*"
    r"(?:narrator|storyteller|author|writer|speaker)\s*$",
    re.IGNORECASE,
)
# Pronouns stay strict — no qualifiers — so "Mary" and "Io" are untouched.
NARRATOR_PRONOUN_RE = re.compile(r"^\s*(?:i|me|myself|my)\s*$", re.IGNORECASE)


def is_narrator_role(name: str) -> bool:
    """True for "Narrator", "The Narrator", "I" and friends.

    A first-person book invites discovery to file its teller under the role
    rather than the name, so PG 1661 carried "The Narrator" (aliases: I,
    Narrator, my patient) beside "Dr. Watson" — one person, two entries, with
    106 of Watson's lines on the wrong one. The two never merge lexically:
    "The Narrator" and "Dr. Watson" share no tokens.

    "Narrator" is also this pipeline's reserved label for narration, so a
    character by that name is ambiguous by construction.
    """
    return bool(NARRATOR_ROLE_RE.match(name) or NARRATOR_PRONOUN_RE.match(name))


def is_reserved_character_name(name: str) -> bool:
    """True for names that must never become roster entries."""
    return is_placeholder_name(name) or is_narrator_role(name)


def merge_rosters(roster: list, found: list) -> list:
    """Fold newly found characters into a roster, deduplicating by name and aliases.

    A found entry whose name matches a roster entry's name (case-insensitive)
    contributes its aliases to that entry; one whose name matches a roster
    entry's alias is dropped as already known. Only genuinely new names are
    appended. Deeper reconciliation ('Peleg' vs 'Captain Peleg') is
    merge_duplicate_characters' job, once, at assembly.
    """
    by_name = {c.name.lower(): c for c in roster}

    for char in found:
        key = char.name.lower()
        if key in by_name:
            existing = by_name[key]
            for alias in char.aliases:
                if all(alias.lower() != a.lower() for a in existing.aliases):
                    existing.aliases.append(alias)
        else:
            known_alias = any(
                any(a.lower() == key for a in existing.aliases)
                for existing in by_name.values()
            )
            if not known_alias:
                by_name[key] = char

    return list(by_name.values())


def merge_duplicate_characters(characters: list) -> list:
    """Merge roster entries that name the same person.

    Discovery routinely emits the same character several ways ('Lucy',
    'Lucy Honeychurch'), which splits attributions across identities and
    doubles the LLM's speaker enum. Two deterministic signals identify a
    duplicate:

      1. An entry's name appears among another entry's aliases
         ('Lucy' vs 'Lucy Honeychurch' [aliases: Lucy]).
      2. An entry's name tokens are a subset of exactly ONE other entry's
         name or alias tokens ('Sir Harry' ⊂ 'Sir Harry Otway'; 'Charlotte'
         ⊂ alias 'Charlotte Bartlett' of 'Miss Bartlett'). Titles compare
         like any other token, so 'Mrs. Honeychurch' never merges into
         'Lucy Honeychurch', and a subset with two possible supersets
         ('John' vs 'John Smith' and 'John Brown') merges into neither.

    The surviving entry absorbs the duplicate's name and aliases and keeps
    the earliest first_appearance_chapter. Idempotent.
    """
    chars = list(characters)
    merged = True
    while merged:
        merged = False
        for i in range(len(chars)):
            j = _merge_target(i, chars)
            if j is None:
                continue
            c, t = chars[i], chars[j]
            for alias in [c.name, *c.aliases]:
                if alias.lower() != t.name.lower() and all(
                    alias.lower() != a.lower() for a in t.aliases
                ):
                    t.aliases.append(alias)
            for hint in c.pronunciation_hints:
                if hint not in t.pronunciation_hints:
                    t.pronunciation_hints.append(hint)
            t.first_appearance_chapter = min(
                t.first_appearance_chapter, c.first_appearance_chapter
            )
            del chars[i]
            merged = True
            break

    # Canonical must be a proper name, not a description. Discovery sometimes
    # emits the descriptive entry as primary ("Jane's mother", aliases:
    # ["Mrs. Bennet"]); the absorb above keeps the target's name, so 125 P&P
    # segments once shipped labeled "Jane's mother". Promote the best alias
    # when it outranks the name.
    for c in chars:
        best = max(c.aliases, key=_name_quality, default=None)
        if best is not None and _name_quality(best) > _name_quality(c.name):
            c.aliases = [a for a in c.aliases if a.lower() != best.lower()] + [c.name]
            c.name = best
    return chars


def _name_quality(name: str) -> int:
    """Rank how much a string looks like a proper name.

    Capitalized tokens score up; lowercase tokens and possessives score down,
    so "Mrs. Bennet" (2) outranks "Jane's mother" (-2) and "his wife" (-2),
    while "Captain Ahab" (2) keeps its title over plain "Ahab" (1).
    """
    score = 0
    if re.search(r"[’']s\b", name):
        score -= 2
    for token in name.split():
        if token[0].isupper():
            score += 1
        elif token[0].isalpha():
            score -= 1
    return score


def _build_alias_map(characters: list) -> dict[str, str]:
    """Build alias → canonical_name lookup from CharacterInfo-like objects."""
    alias_map: dict[str, str] = {}
    for char in characters:
        name = char.name if hasattr(char, "name") else char["name"]
        aliases = char.aliases if hasattr(char, "aliases") else char.get("aliases", [])
        alias_map[name.lower()] = name
        for alias in aliases:
            alias_map[alias.lower()] = name
    return alias_map


def _find_chars_in_text(text: str, alias_map: dict[str, str]) -> list[str]:
    """Return canonical names of all known characters mentioned in text
    (word-boundary match), longest-alias first."""
    text_lower = text.lower()
    found: dict[str, int] = {}
    for alias, canonical in alias_map.items():
        if len(alias) < 3:
            continue
        if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
            found[canonical] = max(found.get(canonical, 0), len(alias))
    return sorted(found, key=found.get, reverse=True)


def _find_char_in_text(text: str, alias_map: dict[str, str]) -> str | None:
    """Return canonical character name if a known alias appears in text (word-boundary)."""
    matches = _find_chars_in_text(text, alias_map)
    return matches[0] if matches else None


def extract_attribution_anchors(
    segments: list[dict],
    characters: list,
) -> dict[int, str]:
    """Find dialogue segments directly confirmed by adjacent attribution narration.

    When narration says "said Mr. Bennet" (contains a character name), the
    immediately adjacent dialogue segments are mapped to that character.
    The preceding dialogue is always anchored; the following dialogue only when
    the tag ends with continuation punctuation (",", ";", ":") — a tag ending
    with a period ("cried his wife, impatiently.") closes the speech, and the
    next quote may belong to anyone.

    Returns: dict mapping dialogue segment index → canonical character name.
    """
    alias_map = _build_alias_map(characters)
    named_anchors: dict[int, str] = {}

    for i, seg in enumerate(segments):
        if not _is_attribution_narration(seg):
            continue
        text = seg.get("text", "")
        sentences = _split_sentences(text)

        # Backward anchor: the tag's FIRST sentence attributes the preceding
        # dialogue ("said Mr. Bennet; and, as he spoke, he left the room...")
        if (
            i > 0
            and segments[i - 1].get("type") == "dialogue"
            and _has_lead_speech_verb(sentences[0])
        ):
            canonical = _tag_speaker(sentences[0], alias_map)
            if canonical:
                named_anchors[i - 1] = canonical

        # Forward anchor: the tag's LAST sentence introduces the following
        # dialogue ("The girls stared at their father. Mrs. Bennet said only,").
        # Requires continuation punctuation — a tag ending with a period
        # ("cried his wife, impatiently.") closes the speech, and the next
        # quote may belong to anyone.
        if (
            i < len(segments) - 1
            and segments[i + 1].get("type") == "dialogue"
            and text.strip().endswith((",", ";", ":"))
            and _has_lead_speech_verb(sentences[-1])
        ):
            canonical = _tag_speaker(sentences[-1], alias_map)
            if canonical:
                named_anchors[i + 1] = canonical

    return named_anchors


def _tag_speaker(sentence: str, alias_map: dict[str, str]) -> str | None:
    """Resolve an attribution-tag sentence to a speaker name.

    The name adjacent to the speech verb is the speaker — "said Lucy, who had
    been further saddened by the Signora's unexpected accent." attributes to
    Lucy; Signora is merely mentioned. Only when no name sits next to the verb
    does the whole sentence get scanned, and then only an unambiguous single
    mention counts; with several characters named, guessing (e.g. by longest
    alias) is worse than deferring to the LLM tier, so return None.
    """
    m = _TAG_NAME_RE.search(sentence)
    if m:
        name = m.group(1).strip()
        return alias_map.get(name.lower(), name)
    # Subject-first order ("Miss Bartlett continued;"). Sentence-initial
    # capitals also catch pronouns ("She said") and determiners ("The girls
    # cried"), so only a name the alias map recognizes counts here.
    m = _NAME_TAG_RE.search(sentence)
    if m:
        canonical = alias_map.get(m.group(1).strip().lower())
        if canonical:
            return canonical
    matches = _find_chars_in_text(sentence, alias_map)
    return matches[0] if len(matches) == 1 else None


