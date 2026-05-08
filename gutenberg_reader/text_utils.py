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

CHAPTER_PATTERNS = [
    re.compile(r"^(CHAPTER\s+[IVXLCDM]+\.?\s*)$", re.MULTILINE),
    re.compile(r"^(CHAPTER\s+\d+\.?\s*)$", re.MULTILINE),
    re.compile(r"^(Chapter\s+\d+\.?\s*)$", re.MULTILINE),
    re.compile(r"^(Chapter\s+[IVXLCDM]+\.?\s*)$", re.MULTILINE),
    re.compile(r"^(PART\s+[IVXLCDM]+\.?\s*)$", re.MULTILINE),
    re.compile(r"^(BOOK\s+[IVXLCDM]+\.?\s*)$", re.MULTILINE),
]


def detect_chapters_regex(lines: list[str]) -> list[dict]:
    """Detect chapter boundaries using ordered regex patterns.

    Returns list of dicts with keys: number, title, start_line (1-indexed), start_marker.
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

    Also treats Gutenberg double-dashes (--) as whitespace separators, since
    the model often converts them to spaces and TTS treats them as pauses.
    """
    text = re.sub(r"-{2,}", " ", text)  # -- and --- → space
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
