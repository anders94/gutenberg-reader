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

CHAPTER_PATTERNS = [
    re.compile(rf"^(CHAPTER\s+[IVXLCDM]+{_TITLE_AFTER_PUNCT})\s*$"),
    re.compile(rf"^(CHAPTER\s+\d+{_TITLE_AFTER_SPACE})\s*$"),
    re.compile(rf"^(Chapter\s+\d+{_TITLE_AFTER_SPACE})\s*$"),
    re.compile(rf"^(Chapter\s+[IVXLCDM]+{_TITLE_AFTER_PUNCT})\s*$"),
    re.compile(rf"^(PART\s+[IVXLCDM]+{_TITLE_AFTER_PUNCT})\s*$"),
    re.compile(rf"^(BOOK\s+[IVXLCDM]+{_TITLE_AFTER_PUNCT})\s*$"),
]

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


def detect_chapters_regex(lines: list[str]) -> list[dict]:
    """Detect chapter boundaries using ordered regex patterns.

    Returns list of dicts with keys: number, title, start_line (1-indexed), start_marker.
    Numbering is positional rather than parsed from the heading, so books that
    restart their count per part cannot produce duplicate chapter numbers.
    """
    matches = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        for pattern in CHAPTER_PATTERNS:
            if pattern.match(stripped):
                matches.append({
                    "line_idx": i,  # 0-based
                    "title": stripped,
                })
                break

    matches = drop_toc_clusters(matches)

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
_TAG_NAME_RE = re.compile(
    r"(?i:\b(?:" + _SPEECH_VERBS + r"))\s+"
    r"((?:(?:Mr|Mrs|Dr|St|Mme|Mlle)\.\s+|(?:Miss|Lady|Sir|Lord|Madame|Colonel|Captain)\s+)?"
    r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)"
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


def _group_conversation_chains(segments: list[dict]) -> list[list[int]]:
    """Group segment indices into conversation chains.

    A chain is a maximal sequence of dialogue + attribution-narration segments.
    Any narration that is NOT an attribution tag breaks the chain.
    Chains with fewer than 2 dialogue segments are discarded.
    """
    chains: list[list[int]] = []
    current: list[int] = []

    for i, seg in enumerate(segments):
        if seg.get("type") == "dialogue":
            current.append(i)
        elif _is_attribution_narration(seg):
            if current:  # only attach if we're already in a chain
                current.append(i)
        else:
            dia_count = sum(1 for j in current if segments[j].get("type") == "dialogue")
            if dia_count >= 2:
                chains.append(current)
            current = []

    dia_count = sum(1 for j in current if segments[j].get("type") == "dialogue")
    if dia_count >= 2:
        chains.append(current)

    return chains


def _group_into_speech_units(
    dialogue_idxs: list[int],
    segments: list[dict],
) -> list[list[int]]:
    """Group consecutive dialogue indices into speech units.

    Two adjacent dialogue segments belong to the same speech unit when the only
    segment between them is attribution narration ending with a comma, semicolon,
    or colon — indicating the speech continues — or when the earlier segment is
    an unclosed quotation continuing into the next paragraph (quote-continues).

    Example: '"Part A," said she, "Part B."' → one speech unit [A, B].
    Example: '"Statement." said she. [next dia]' → two separate speech units.
    """
    from gutenberg_reader.segmenter import QUOTE_CONTINUES

    if not dialogue_idxs:
        return []

    units: list[list[int]] = [[dialogue_idxs[0]]]

    for pos in range(1, len(dialogue_idxs)):
        prev_idx = dialogue_idxs[pos - 1]
        curr_idx = dialogue_idxs[pos]
        between = segments[prev_idx + 1:curr_idx]

        is_bridge = (
            len(between) == 1
            and _is_attribution_narration(between[0])
            and between[0].get("text", "").strip().endswith((",", ";", ":"))
        ) or (
            len(between) == 0
            and segments[prev_idx].get("notes") == QUOTE_CONTINUES
        )

        if is_bridge:
            units[-1].append(dialogue_idxs[pos])
        else:
            units.append([dialogue_idxs[pos]])

    return units


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


def _find_char_in_text(text: str, alias_map: dict[str, str]) -> str | None:
    """Return canonical character name if a known alias appears in text (word-boundary)."""
    text_lower = text.lower()
    best: str | None = None
    best_len = 0
    for alias, canonical in alias_map.items():
        if len(alias) < 3:
            continue
        if re.search(r"\b" + re.escape(alias) + r"\b", text_lower) and len(alias) > best_len:
            best_len = len(alias)
            best = canonical
    return best


_REFERENCE_VERB_RE = re.compile(
    r"^\s+(said|says|told|has|had|was|were|did|does|is|are|will|would|can|could|"
    r"may|might|came|went|has|have|replied|asked|answered|cried)",
    re.IGNORECASE,
)


def _extract_scene_cast(
    chain_segs: list[dict],
    confirmed_speakers: set[str],
    char_names: list[str],
) -> set[str]:
    """Detect characters present in a scene from VOCATIVE use of their name in dialogue.

    Only counts a character as present if their full canonical name is directly
    addressed (vocative), NOT merely referenced ("Mrs. Long says that...").

    Detection:
    - Name in first 50 chars AND not immediately followed by a reference verb.
    - OR name follows an address marker ("My dear NAME", "Oh NAME").
    """
    cast = set(confirmed_speakers)
    for seg in chain_segs:
        if seg.get("type") != "dialogue":
            continue
        text = seg.get("text", "")
        text_inner = text.lstrip('"\u201c\u2018_').lstrip()
        for name in char_names:
            name_pat = re.escape(name)
            # Case 1: true vocative \u2014 the name is directly addressed: followed
            # by vocative punctuation and preceded by a clause boundary, so mere
            # references ("I heard it from Mrs. Long has...") don't count. A
            # period after the name only counts mid-utterance ("...will like it,
            # Lizzy.") \u2014 a bare "Bingley." is an answer, not an address.
            found = False
            for m in re.finditer(name_pat + r"(?=[,.!?])", text_inner, re.IGNORECASE):
                before = text_inner[:m.start()].rstrip()
                after_char = text_inner[m.end():m.end() + 1]
                boundary = (
                    before.endswith((",", ";", "!", "?", "\u2014", "--"))
                    or re.search(r"\b(my dear|dear|oh|ah|why|well|pray|come)$", before, re.IGNORECASE)
                )
                ok = boundary if before else after_char != "."
                if ok:
                    cast.add(name)
                    found = True
                    break
            if found:
                continue
            # Case 2: full name follows an address marker anywhere in text
            if re.search(
                r"\b(my dear|dear|oh|ah|pray)\s+" + name_pat + r"\b",
                text,
                re.IGNORECASE,
            ):
                cast.add(name)
    return cast


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

    Prefers the known-character alias map; falls back to a proper name directly
    after the speech verb ("said Lydia,") so characters the discovery stage
    missed still anchor deterministically.
    """
    canonical = _find_char_in_text(sentence, alias_map)
    if canonical:
        return canonical
    m = _TAG_NAME_RE.search(sentence)
    return m.group(1).strip() if m else None


def propagate_anchors(
    segments: list[dict],
    characters: list,
    char_names: list[str],
) -> tuple[list[dict], list[list[int]], int]:
    """Propagate confirmed speakers through 2-person conversation chains.

    Algorithm:
    1. Group segments into conversation chains (dialogue + attribution narration).
    2. For each chain, find confirmed speakers from:
       a. Named anchors (attribution narration containing a character name)
       b. Fallback: LLM-assigned speakers adjacent to attribution narration
    3. Detect scene cast (2nd character from dialogue text name mentions).
    4. If cast == 2: fill unanchored dialogue via strict alternation by speech unit.
    5. If cast != 2 or no anchors: flag chain for Tier-2 LLM review.

    Returns: (corrected_segments, flagged_chains, n_corrections)
    """
    named_anchors = extract_attribution_anchors(segments, characters)
    corrected = [dict(s) for s in segments]
    chains = _group_conversation_chains(segments)
    flagged_chains: list[list[int]] = []
    n_corrections = 0

    for chain in chains:
        dialogue_idxs = [i for i in chain if segments[i].get("type") == "dialogue"]
        if len(dialogue_idxs) < 2:
            continue

        # Step 1: confirmed speakers from named anchors
        confirmed: dict[int, str] = {
            i: named_anchors[i] for i in dialogue_idxs if i in named_anchors
        }

        # Step 2: fallback — LLM-assigned speakers adjacent to attribution narration
        if not confirmed:
            for idx in dialogue_idxs:
                adj_before = idx - 1 >= 0 and _is_attribution_narration(segments[idx - 1])
                adj_after = (
                    idx + 1 < len(segments)
                    and _is_attribution_narration(segments[idx + 1])
                )
                if adj_before or adj_after:
                    spk = segments[idx].get("speaker")
                    if spk and spk not in ("Unknown", "Narrator"):
                        confirmed[idx] = spk

        if not confirmed:
            flagged_chains.append(chain)
            continue

        confirmed_speakers = set(confirmed.values())

        # Step 3: detect scene cast from dialogue text (e.g., "My dear Mr. Bennet,")
        chain_segs = [segments[i] for i in chain]
        scene_cast = _extract_scene_cast(chain_segs, confirmed_speakers, char_names)
        scene_cast = {c for c in scene_cast if c in char_names}

        if len(scene_cast) != 2:
            flagged_chains.append(chain)
            continue

        # Step 4: group dialogue into speech units (handle split-speech patterns)
        speech_units = _group_into_speech_units(dialogue_idxs, segments)

        # Map each speech unit position to its confirmed speaker
        su_confirmed: dict[int, str] = {}
        for su_pos, su in enumerate(speech_units):
            for seg_idx in su:
                if seg_idx in confirmed:
                    su_confirmed[su_pos] = confirmed[seg_idx]
                    break

        if not su_confirmed:
            flagged_chains.append(chain)
            continue

        # Step 5: determine alternation orientation from confirmed speech units
        chars = list(scene_cast)
        anchor_pos = min(su_confirmed.keys())
        anchor_spk = su_confirmed[anchor_pos]

        # Try both orientations; use the one consistent with all confirmed anchors
        best_anchor_char_idx: int | None = None
        for try_idx in range(2):
            ok = True
            for su_pos, spk in su_confirmed.items():
                if spk not in chars:
                    ok = False
                    break
                expected = chars[(try_idx + (su_pos - anchor_pos)) % 2]
                if expected != spk:
                    ok = False
                    break
            if ok:
                best_anchor_char_idx = try_idx
                break

        if best_anchor_char_idx is None:
            flagged_chains.append(chain)
            continue

        # Step 6: apply alternation to unconfirmed speech units
        for su_pos, su in enumerate(speech_units):
            if su_pos in su_confirmed:
                continue
            expected_char_idx = (best_anchor_char_idx + (su_pos - anchor_pos)) % 2
            new_speaker = chars[expected_char_idx]
            for seg_idx in su:
                if corrected[seg_idx].get("speaker") != new_speaker:
                    corrected[seg_idx] = {**corrected[seg_idx], "speaker": new_speaker}
                    n_corrections += 1

    return corrected, flagged_chains, n_corrections
