"""Stage 02 — Discover metadata and chapter structure."""

from __future__ import annotations
from pathlib import Path

from rich.console import Console

from gutenberg_reader.cache import atomic_write_json, read_text, stage_complete
from gutenberg_reader.config import Config
from gutenberg_reader.models import BookMetadata, ChapterInfo, DiscoveryResult
from gutenberg_reader.llm import LLMClient
from gutenberg_reader import prompts, schemas
from gutenberg_reader import text_utils

console = Console()


def run(config: Config, client: LLMClient) -> DiscoveryResult:
    """Run discovery and return DiscoveryResult."""
    stage_dir = config.stage_dir(2)
    out_path = stage_dir / "discovery.json"

    if stage_complete(out_path) and (config.force_stage is None or config.force_stage > 2):
        if config.verbose:
            console.print(f"[dim]Stage 02: already complete ({out_path})[/dim]")
        from gutenberg_reader.cache import read_json
        cached = DiscoveryResult.from_dict(read_json(out_path))
        # Re-check the cached structure, not just freshly detected structure: a
        # discovery.json written by an older detector stays on disk and silently
        # feeds every later stage a table of contents (see PG 2701). The checks
        # are pure, so running them on the cached path costs nothing.
        _warn_on_degenerate_chapters(cached.chapters)
        _warn_on_size_outliers(cached.chapters)
        return cached

    raw_path = config.stage_dir(1) / "book.txt"
    raw_text = read_text(raw_path)
    lines = raw_text.splitlines()

    # Find body bounds
    body_start, body_end = text_utils.find_body_bounds(lines)

    preamble = "\n".join(lines[:body_start])
    raw_meta = text_utils.extract_preamble_metadata(preamble)

    metadata = BookMetadata(
        title=raw_meta.get("title", ""),
        author=raw_meta.get("author", ""),
        language=raw_meta.get("language", "en"),
        gutenberg_id=config.book_id,
        release_date=raw_meta.get("release_date", ""),
        credits=raw_meta.get("credits", ""),
    )

    if config.verbose:
        console.print(f"[cyan]Stage 02:[/cyan] Metadata: {metadata.title} by {metadata.author}")

    # Detect chapters within body
    body_lines = lines[body_start:body_end]
    raw_chapters = text_utils.detect_chapters_regex(body_lines)

    # Check for content before the first detected chapter (e.g. P&P chapter 1 has no heading)
    raw_chapters = _maybe_prepend_chapter_one(raw_chapters, body_lines)

    if len(raw_chapters) < 2:
        if config.verbose:
            console.print(
                f"[yellow]Stage 02:[/yellow] Regex found only {len(raw_chapters)} chapters, "
                "falling back to LLM discovery..."
            )
        raw_chapters = _llm_chapter_discovery(body_lines, config, client)

    # Runs after both paths: the title-page split is what a heading-less source
    # produces, and either detector can land on it.
    raw_chapters = _drop_leading_front_matter(
        raw_chapters, body_lines, include_front_matter=config.include_front_matter,
    )

    # Last resort: a body that really has no headings still needs to come apart.
    raw_chapters = _split_headless_body(raw_chapters, body_lines)

    if config.verbose:
        console.print(f"[cyan]Stage 02:[/cyan] Found {len(raw_chapters)} chapters")

    # Build ChapterInfo objects with end_line and word_count
    chapter_infos = _build_chapter_infos(
        raw_chapters, body_lines, body_start,
        include_back_matter=config.include_back_matter,
    )
    _warn_on_degenerate_chapters(chapter_infos)
    _warn_on_size_outliers(chapter_infos)

    result = DiscoveryResult(
        metadata=metadata,
        chapters=chapter_infos,
        body_start_line=body_start,
        body_end_line=body_end,
    )

    atomic_write_json(out_path, result.to_dict())
    return result


def _build_chapter_infos(
    raw_chapters: list[dict],
    body_lines: list[str],
    body_start: int,
    include_back_matter: bool = False,
) -> list[ChapterInfo]:
    """Convert raw chapter dicts into ChapterInfo with word counts and absolute line numbers."""
    infos: list[ChapterInfo] = []
    n = len(raw_chapters)
    seen_numbers: set[int] = set()
    back_matter_start: int | None = None  # rel index of the first back-matter heading

    for i, ch in enumerate(raw_chapters):
        # start_line in raw_chapters is 1-indexed relative to body_lines
        rel_start = ch["start_line"] - 1  # 0-based in body_lines
        abs_start = body_start + rel_start  # 0-based in full file

        # end: next chapter start - 1, or end of body
        if i + 1 < n:
            rel_end = raw_chapters[i + 1]["start_line"] - 2  # line before next chapter header
        else:
            # The final chapter would otherwise run to end-of-body, swallowing
            # any footnote appendix, index, or errata the edition carries.
            rel_end = len(body_lines) - 1
            back_matter_start = _find_back_matter_heading(body_lines, rel_start + 1, rel_end)
            if back_matter_start is not None:
                trimmed = rel_end - back_matter_start + 1
                console.print(
                    f"[cyan]Stage 02:[/cyan] final chapter ends at back-matter heading "
                    f"{body_lines[back_matter_start].strip()!r} — "
                    f"{'keeping' if include_back_matter else 'trimming'} {trimmed:,} trailing lines"
                )
                rel_end = back_matter_start - 1

        if rel_end < rel_start:
            console.print(
                f"[yellow]Stage 02: warning —[/yellow] chapter {ch['number']} "
                f"({ch['title']!r}) spans no lines; its boundaries are out of order. "
                "Check the discovered chapter list."
            )

        abs_end = body_start + rel_end

        # Extract chapter text for word count
        chapter_text = "\n".join(body_lines[rel_start:rel_end + 1])
        chapter_text = text_utils.strip_illustration_blocks(chapter_text)
        wc = text_utils.word_count(chapter_text)

        # Chapter numbers key the per-chapter cache files in stages 03/05 and the
        # accepted-chapter map in stage 07, so a duplicate silently discards a
        # chapter. Fall back to position when a source hands us a repeat.
        number = ch["number"]
        if number in seen_numbers:
            number = max(seen_numbers) + 1
        seen_numbers.add(number)

        infos.append(ChapterInfo(
            number=number,
            title=ch["title"],
            start_line=abs_start + 1,  # 1-indexed in full file
            end_line=abs_end + 1,
            word_count=wc,
            start_marker=ch.get("start_marker", ch["title"]),
            kind=ch.get("kind", "body"),
        ))

    if include_back_matter and back_matter_start is not None and infos:
        rel_end = len(body_lines) - 1
        text = "\n".join(body_lines[back_matter_start:rel_end + 1])
        text = text_utils.strip_illustration_blocks(text)
        infos.append(ChapterInfo(
            number=max(seen_numbers) + 1,
            title=body_lines[back_matter_start].strip(),
            start_line=body_start + back_matter_start + 1,
            end_line=body_start + rel_end + 1,
            word_count=text_utils.word_count(text),
            start_marker=body_lines[back_matter_start].strip(),
            kind="back",
        ))

    return infos


def _find_back_matter_heading(body_lines: list[str], start: int, end: int) -> int | None:
    """Return the 0-based index of the first back-matter heading in [start, end], if any."""
    for j in range(start, end + 1):
        if text_utils.BACK_MATTER_RE.match(body_lines[j].strip()):
            return j
    return None


def _warn_on_degenerate_chapters(infos: list[ChapterInfo], min_words: int = 20) -> None:
    """Warn when many chapters hold nothing but their own heading.

    This is what a table of contents mistaken for the body looks like, and it is
    otherwise invisible until the final JSON comes out nearly empty.
    """
    if not infos:
        return

    degenerate = [ci for ci in infos if ci.word_count < min_words]
    if len(degenerate) * 5 < len(infos):  # under 20% — nothing systematic
        return

    console.print(
        f"[yellow]Stage 02: warning —[/yellow] {len(degenerate)} of {len(infos)} chapters "
        f"contain fewer than {min_words} words. Chapter detection may have matched a "
        "table of contents or an index rather than the book body."
    )


def _warn_on_size_outliers(
    infos: list[ChapterInfo],
    high: float = 2.5,
    low: float = 0.2,
    pair_high: float = 10.0,
) -> None:
    """Warn when a chapter is wildly larger or smaller than the median.

    A chapter several times the median usually means swallowed front or back
    matter, and it costs real TTS time downstream before anyone hears it.

    Two chapters have no useful median — each sits at it — so they are compared
    against each other instead. That is not a corner case but the worst one:
    PG 2131 (Herodotus) carries no chapter headings at all, so discovery split
    it on the only repeated line in the file, its own title, giving a 647-word
    front-matter note and the entire 36,916-word book. The median test could
    never see it.
    """
    import statistics

    if len(infos) < 2:
        return

    if len(infos) == 2:
        larger, smaller = sorted(infos, key=lambda ci: ci.word_count, reverse=True)
        if smaller.word_count > 0 and larger.word_count / smaller.word_count < pair_high:
            return
        console.print(
            f"[yellow]Stage 02: warning —[/yellow] the book split into 2 chapters of "
            f"{larger.word_count:,} and {smaller.word_count:,} words. A split this "
            "lopsided usually means the source carries no chapter headings and "
            "detection matched its title page instead."
        )
        return

    median = statistics.median(ci.word_count for ci in infos)
    if median == 0:
        return

    for ci in infos:
        ratio = ci.word_count / median
        if ratio > high or ratio < low:
            console.print(
                f"[yellow]Stage 02: warning —[/yellow] chapter {ci.number} "
                f"({ci.title!r}) is {ratio:.1f}× the median chapter size "
                f"({ci.word_count:,} vs {median:,.0f} words). "
                "Check for swallowed front or back matter."
            )


# A title page and the editor's note under it run to a few hundred words; a real
# opening chapter runs to thousands. Past this, a leading block is body text
# whatever its headings look like.
MAX_LEADING_FRONT_MATTER_WORDS = 2000
# Below this a block holds no prose at all — just its own heading and a byline.
MIN_LEADING_BODY_WORDS = 20


def _leading_front_matter_reason(
    raw_chapters: list[dict],
    body_lines: list[str],
) -> tuple[str, int] | None:
    """Why the first of raw_chapters is the edition's matter, and its word count.

    Size alone does not identify it; plenty of books open with a short chapter.
    What does is a *small leading* block that repeats the next chapter's title
    verbatim, carries an apparatus heading (NOTE, PREFACE, ...) of its own, or
    holds no prose at all.
    """
    if len(raw_chapters) < 2:
        return None

    first, second = raw_chapters[0], raw_chapters[1]
    lead_start = first["start_line"] - 1
    lead_end = second["start_line"] - 1  # exclusive: the next heading's own line
    lead_lines = body_lines[lead_start:lead_end]

    lead_words = text_utils.word_count(
        text_utils.strip_illustration_blocks("\n".join(lead_lines))
    )
    if lead_words == 0 or lead_words > MAX_LEADING_FRONT_MATTER_WORDS:
        return None

    # The rest of the book has to dwarf it, or this is simply a short chapter one.
    rest_words = text_utils.word_count(
        text_utils.strip_illustration_blocks("\n".join(body_lines[lead_end:]))
    )
    if rest_words < 4 * lead_words:
        return None

    if (first.get("title", "").strip().casefold()
            == second.get("title", "").strip().casefold()):
        return "repeats the next chapter's title", lead_words
    if any(text_utils.FRONT_MATTER_RE.match(line.strip())
           or text_utils.BACK_MATTER_RE.match(line.strip())
           for line in lead_lines):
        return "carries an apparatus heading", lead_words
    if lead_words < MIN_LEADING_BODY_WORDS:
        return "holds no prose of its own", lead_words
    return None


def _drop_leading_front_matter(
    raw_chapters: list[dict],
    body_lines: list[str],
    include_front_matter: bool = False,
) -> list[dict]:
    """Drop leading "chapters" that are really the edition's title page.

    A source with no chapter headings leaves the detector only the book's own
    front page to match on. PG 2131 (Herodotus, "An Account of Egypt") splits
    there, and not once: the title page with the editor's NOTE under it, the
    title repeated over the text, and the subtitle line below that all read as
    headings, so the whole book lands in a chapter sitting behind two stubs.
    Each is stripped in turn — removing one exposes the next.
    """
    kept = list(raw_chapters)
    dropped: list[dict] = []

    while (found := _leading_front_matter_reason(kept, body_lines)) is not None:
        reason, lead_words = found
        head = kept.pop(0)
        dropped.append(dict(head, kind="front"))
        console.print(
            f"[cyan]Stage 02:[/cyan] {head.get('title', '')!r} ({lead_words:,} words) "
            f"is front matter — it {reason}"
        )

    if not dropped:
        return raw_chapters

    result = dropped + kept if include_front_matter else kept
    return [dict(ch, number=i + 1) for i, ch in enumerate(result)]


# A synthesized part aims for the length of an ordinary chapter: long enough to
# be worth its own track, short enough that one attribution pass can hold it.
# P&P runs ~2,000 words a chapter, the Odyssey ~4,500.
TARGET_PART_WORDS = 2500
# Below this a body is a long short story, not an unbroken book; leave it whole.
MIN_HEADLESS_SPLIT_WORDS = 3 * TARGET_PART_WORDS


def _paragraph_spans(
    body_lines: list[str], start: int, end: int
) -> list[tuple[int, int]]:
    """(first line index, word count) for each blank-line-separated paragraph."""
    spans: list[tuple[int, int]] = []
    i = start
    while i < end:
        if not body_lines[i].strip():
            i += 1
            continue
        para_start, words = i, 0
        while i < end and body_lines[i].strip():
            words += len(body_lines[i].split())
            i += 1
        spans.append((para_start, words))
    return spans


def _split_headless_body(
    raw_chapters: list[dict],
    body_lines: list[str],
    target_words: int = TARGET_PART_WORDS,
) -> list[dict]:
    """Split a body with no chapter structure into parts at paragraph boundaries.

    Some editions carry no chapter heading anywhere. PG 2131 prints Herodotus'
    Book II as one unbroken 36,916-word run — its 182 canonical sections are
    unnumbered in this translation — so there is genuinely nothing to detect and
    discovery correctly returns a single chapter. Every later stage then treats
    the whole book as one unit: one attribution pass over 37k words, a cast
    discovered from a single window (88 names for 2131), one enormous track.

    Only a body with *no* structure is split. A long chapter inside a book that
    has chapters is a long chapter — Moby-Dick's "Cetology" runs 3.6x the median
    and is genuinely one — so this never touches a multi-chapter result.
    """
    # Exactly one body chapter, running to the end. Front matter kept by
    # --include-front-matter sits ahead of it and is left alone.
    body_idx = [
        i for i, ch in enumerate(raw_chapters) if ch.get("kind", "body") == "body"
    ]
    if len(body_idx) != 1 or body_idx[0] != len(raw_chapters) - 1:
        return raw_chapters

    head = raw_chapters[:body_idx[0]]
    rel_start = raw_chapters[body_idx[0]]["start_line"] - 1
    spans = _paragraph_spans(body_lines, rel_start, len(body_lines))
    total = sum(words for _, words in spans)
    if total < MIN_HEADLESS_SPLIT_WORDS:
        return raw_chapters

    # Greedy fill: a part closes once it reaches the target, so every cut lands
    # on a paragraph boundary and the parts tile the body exactly.
    groups: list[list[int]] = []  # [first line index, words so far]
    for line_idx, words in spans:
        if not groups or groups[-1][1] >= target_words:
            groups.append([line_idx, words])
        else:
            groups[-1][1] += words

    # A runt tail is worse than one long part: fold it back.
    if len(groups) > 1 and groups[-1][1] < target_words // 2:
        tail = groups.pop()
        groups[-1][1] += tail[1]

    if len(groups) < 2:
        return raw_chapters

    # Keep the original opening line — the title sits above the first paragraph.
    groups[0][0] = rel_start

    console.print(
        f"[cyan]Stage 02:[/cyan] no chapter headings in {total:,} words — "
        f"splitting into {len(groups)} parts at paragraph boundaries"
    )
    parts = [
        {
            "number": 0,  # renumbered below, across any kept front matter
            "title": f"Part {i + 1}",
            "start_line": line_idx + 1,  # 1-indexed within body_lines
            "start_marker": "",
            "kind": "body",
        }
        for i, (line_idx, _words) in enumerate(groups)
    ]
    return [dict(ch, number=i + 1) for i, ch in enumerate(head + parts)]


def _maybe_prepend_chapter_one(
    raw_chapters: list[dict],
    body_lines: list[str],
) -> list[dict]:
    """If there's substantial narrative text before the first detected chapter, prepend a chapter 1.

    Handles books like the illustrated P&P where Chapter I has no standalone heading.
    Strategy: reconstruct the pre-chapter text (ignoring illustration blocks and TOC lines),
    and if it's substantial, find where it starts.
    """
    import re as _re

    first_chapter_start_idx = (raw_chapters[0]["start_line"] - 1) if raw_chapters else len(body_lines)
    MIN_CHAPTER_WORDS = 50

    TOC_RE = _re.compile(
        r"(heading to chapter|tailpiece|list of illustrations|\bcontents?\b"
        r"|^\s*\d+\s*$|\s{3,}\d+\s*$)",
        _re.IGNORECASE,
    )

    def is_toc(line: str) -> bool:
        # A heading-shaped line ahead of the first detected chapter is a contents
        # entry, not narrative — otherwise a full TOC reads as a 500-word chapter one.
        # Takes the raw line: an indented numeral entry ("   I.  A Scandal in
        # Bohemia") is recognizable only by its indentation, since the heading
        # patterns require an all-caps title and contents entries are title case.
        stripped = line.strip()
        return (
            bool(TOC_RE.search(stripped))
            or text_utils.looks_like_chapter_heading(stripped)
            or bool(text_utils.TOC_NUMERAL_ENTRY_RE.match(line))
        )

    # Build set of illustration line indices before the first chapter
    illustration_lines: set[int] = _find_illustration_lines(body_lines, 0, first_chapter_start_idx)

    # Collect prose lines (non-illustration, non-blank, non-TOC) before first chapter
    # Track the earliest one as our candidate chapter-1 start
    first_prose_line: int | None = None
    last_prose_line: int | None = None

    for j in range(first_chapter_start_idx):
        if j in illustration_lines:
            continue
        stripped = body_lines[j].strip()
        if not stripped:
            continue
        if is_toc(body_lines[j]):
            continue
        if last_prose_line is None or j > last_prose_line:
            last_prose_line = j

    if last_prose_line is None:
        return raw_chapters

    # Now scan forward from the last prose line to find the contiguous block it belongs to,
    # then scan backward to find where that block starts (skipping illustrations).
    # Walk back from last_prose_line to find the start of the prose block nearest the chapter.
    block_end = last_prose_line
    block_start = block_end
    j = block_end - 1
    while j >= 0:
        if j in illustration_lines:
            # Skip the illustration block backward
            j -= 1
            continue
        stripped = body_lines[j].strip()
        if not stripped:
            # Blank line — peek further back
            k = j - 1
            while k >= 0 and (k in illustration_lines or not body_lines[k].strip()):
                k -= 1
            if k < 0:
                break
            if is_toc(body_lines[k]):
                break  # Hit TOC boundary
            # Continue over the blank gap
            block_start = k
            j = k - 1
        else:
            if is_toc(body_lines[j]):
                break
            block_start = j
            j -= 1

    # A preface or dedication reads as prose to the walk-back above — nothing in
    # it is TOC-shaped. But such a block announces itself with a heading
    # ("PREFACE TO FIRST EDITION", "DEDICATION", ...); if one is present anywhere
    # in the block, this is the edition's apparatus, not an unlabeled chapter one.
    for j in range(block_start, first_chapter_start_idx):
        if j in illustration_lines:
            continue
        stripped = body_lines[j].strip()
        if (text_utils.FRONT_MATTER_RE.match(stripped)
                or text_utils.BACK_MATTER_RE.match(stripped)):
            return raw_chapters

    # Collect all text in [block_start, first_chapter_start_idx) excluding illustrations
    pre_text_words = []
    paragraphs: list[str] = []
    current: list[str] = []
    for j in range(block_start, first_chapter_start_idx):
        if j in illustration_lines:
            continue
        stripped = body_lines[j].strip()
        if stripped and not is_toc(body_lines[j]):
            pre_text_words.extend(stripped.split())
            current.append(stripped)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))

    if len(pre_text_words) < MIN_CHAPTER_WORDS:
        return raw_chapters

    if not _reads_as_prose(paragraphs):
        return raw_chapters

    # Advance block_start past any leading TOC/blank lines to find actual prose start
    actual_start = block_start
    for j in range(block_start, first_chapter_start_idx):
        if j in illustration_lines:
            continue
        stripped = body_lines[j].strip()
        if not stripped or is_toc(body_lines[j]):
            continue
        actual_start = j
        break

    synthetic = {
        "number": 1,
        "title": "Chapter I",
        "start_line": actual_start + 1,  # 1-indexed
        "start_marker": "",
    }
    # Renumber subsequent chapters starting from 2
    renumbered = []
    for i, ch in enumerate(raw_chapters):
        renumbered.append(dict(ch, number=i + 2))
    return [synthetic] + renumbered


def _reads_as_prose(paragraphs: list[str], min_ratio: float = 0.5) -> bool:
    """True if a block is narrative rather than a list of entries.

    An unlabeled chapter one is made of sentences; a list of illustrations or a
    contents block is made of captions. Measured on PG 1342's genuine chapter
    one, 34 of 34 paragraphs end in sentence punctuation (median 21 words); on
    PG 37106's list of illustrations, 7 of 141 do (median 5 words). That gap is
    what separates them — the front-matter heading that would otherwise name the
    block sits behind a TOC line the backward walk stops on, so it is never seen.
    """
    if not paragraphs:
        return False
    ended = sum(1 for p in paragraphs if p.rstrip().endswith((".", "!", "?", '"', "”", "’", "—")))
    return ended / len(paragraphs) >= min_ratio


def _find_illustration_lines(body_lines: list[str], start: int, end: int) -> set[int]:
    """Return set of 0-based indices that are inside illustration blocks."""
    result: set[int] = set()
    i = start
    while i < end:
        stripped = body_lines[i].strip()
        if stripped.lower().startswith("[illustration"):
            depth = stripped.count("[") - stripped.count("]")
            result.add(i)
            if depth > 0:
                i += 1
                while i < end:
                    result.add(i)
                    depth += body_lines[i].count("[") - body_lines[i].count("]")
                    i += 1
                    if depth <= 0:
                        break
                continue
        i += 1
    return result


def _validate_llm_chapters(
    chapters: list[dict],
    body_line_count: int,
    config: Config,
) -> list[dict]:
    """Sanitize model-provided chapter entries before trusting their positions.

    Chapter ends are derived from the *next* entry's start, so an out-of-order
    or out-of-range start silently produces an empty chapter while its
    neighbour balloons. Entries are range-checked, sorted by position, and
    deduplicated — loudly, never silently. Front and back matter (prefaces,
    footnote appendices) are dropped unless the config keeps them.
    """
    valid: list[dict] = []
    seen_starts: set[int] = set()
    for ch in chapters:
        title = ch.get("title", "")
        try:
            start = int(ch.get("start_line", 0))
        except (TypeError, ValueError):
            start = 0
        if start < 1 or start > body_line_count:
            console.print(
                f"[yellow]Stage 02: warning —[/yellow] dropping LLM chapter {title!r}: "
                f"start_line {ch.get('start_line')!r} is outside the body (1–{body_line_count})"
            )
            continue
        if start in seen_starts:
            console.print(
                f"[yellow]Stage 02: warning —[/yellow] dropping LLM chapter {title!r}: "
                f"duplicate start_line {start}"
            )
            continue
        kind = text_utils.classify_heading(title)
        if kind == "front" and not config.include_front_matter:
            console.print(f"[cyan]Stage 02:[/cyan] skipping front matter: {title!r}")
            continue
        if kind == "back" and not config.include_back_matter:
            console.print(f"[cyan]Stage 02:[/cyan] skipping back matter: {title!r}")
            continue
        seen_starts.add(start)
        valid.append(dict(ch, start_line=start, kind=kind))

    in_order = [ch["start_line"] for ch in valid]
    if in_order != sorted(in_order):
        console.print(
            "[yellow]Stage 02: warning —[/yellow] LLM returned chapters out of "
            "document order; sorting by position"
        )
        valid.sort(key=lambda ch: ch["start_line"])
    return valid


def _llm_chapter_discovery(
    body_lines: list[str],
    config: Config,
    client: LLMClient,
) -> list[dict]:
    """Fall back to LLM for chapter detection."""
    # Send at most first 500 lines to keep context manageable
    sample = "\n".join(
        f"{i+1}: {line}" for i, line in enumerate(body_lines[:500])
    )

    messages = [
        {"role": "system", "content": prompts.llm_chapter_discovery_system()},
        {"role": "user", "content": prompts.llm_chapter_discovery_user(sample)},
    ]

    try:
        data = client.chat_json(config.processing_model, messages, schema=schemas.CHAPTERS_SCHEMA)
        chapters = data.get("chapters", [])
        chapters = _validate_llm_chapters(chapters, len(body_lines), config)
        # Normalize to our format. Numbering is positional, not taken from the
        # model: a heading the model labels "Chapter 1" may be the Nth it found,
        # and a repeated number would collide in the per-chapter caches.
        return [
            {
                "number": i + 1,
                "title": ch.get("title", f"Chapter {i+1}"),
                "start_line": ch.get("start_line", 1),
                "start_marker": ch.get("title", ""),
                "kind": ch.get("kind", "body"),
            }
            for i, ch in enumerate(chapters)
        ]
    except Exception as e:
        console.print(f"[red]Stage 02: LLM discovery failed: {e}[/red]")
        # Return a single "chapter" covering the whole body
        return [{"number": 1, "title": "Chapter 1", "start_line": 1, "start_marker": ""}]
