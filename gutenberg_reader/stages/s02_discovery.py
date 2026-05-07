"""Stage 02 — Discover metadata and chapter structure."""

from __future__ import annotations
from pathlib import Path

from rich.console import Console

from gutenberg_reader.cache import atomic_write_json, read_text, stage_complete
from gutenberg_reader.config import Config
from gutenberg_reader.models import BookMetadata, ChapterInfo, DiscoveryResult
from gutenberg_reader.ollama import OllamaClient
from gutenberg_reader import prompts
from gutenberg_reader import text_utils

console = Console()


def run(config: Config, client: OllamaClient) -> DiscoveryResult:
    """Run discovery and return DiscoveryResult."""
    stage_dir = config.stage_dir(2)
    out_path = stage_dir / "discovery.json"

    if stage_complete(out_path) and (config.force_stage is None or config.force_stage > 2):
        if config.verbose:
            console.print(f"[dim]Stage 02: already complete ({out_path})[/dim]")
        from gutenberg_reader.cache import read_json
        return DiscoveryResult.from_dict(read_json(out_path))

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

    if config.verbose:
        console.print(f"[cyan]Stage 02:[/cyan] Found {len(raw_chapters)} chapters")

    # Build ChapterInfo objects with end_line and word_count
    chapter_infos = _build_chapter_infos(raw_chapters, body_lines, body_start)

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
) -> list[ChapterInfo]:
    """Convert raw chapter dicts into ChapterInfo with word counts and absolute line numbers."""
    infos: list[ChapterInfo] = []
    n = len(raw_chapters)

    for i, ch in enumerate(raw_chapters):
        # start_line in raw_chapters is 1-indexed relative to body_lines
        rel_start = ch["start_line"] - 1  # 0-based in body_lines
        abs_start = body_start + rel_start  # 0-based in full file

        # end: next chapter start - 1, or end of body
        if i + 1 < n:
            rel_end = raw_chapters[i + 1]["start_line"] - 2  # line before next chapter header
        else:
            rel_end = len(body_lines) - 1

        abs_end = body_start + rel_end

        # Extract chapter text for word count
        chapter_text = "\n".join(body_lines[rel_start:rel_end + 1])
        chapter_text = text_utils.strip_illustration_blocks(chapter_text)
        wc = text_utils.word_count(chapter_text)

        infos.append(ChapterInfo(
            number=ch["number"],
            title=ch["title"],
            start_line=abs_start + 1,  # 1-indexed in full file
            end_line=abs_end + 1,
            word_count=wc,
            start_marker=ch.get("start_marker", ch["title"]),
        ))

    return infos


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
        if TOC_RE.search(stripped):
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
            prev_stripped = body_lines[k].strip()
            if TOC_RE.search(prev_stripped):
                break  # Hit TOC boundary
            # Continue over the blank gap
            block_start = k
            j = k - 1
        else:
            if TOC_RE.search(stripped):
                break
            block_start = j
            j -= 1

    # Collect all text in [block_start, first_chapter_start_idx) excluding illustrations
    pre_text_words = []
    for j in range(block_start, first_chapter_start_idx):
        if j in illustration_lines:
            continue
        stripped = body_lines[j].strip()
        if stripped and not TOC_RE.search(stripped):
            pre_text_words.extend(stripped.split())

    if len(pre_text_words) < MIN_CHAPTER_WORDS:
        return raw_chapters

    # Advance block_start past any leading TOC/blank lines to find actual prose start
    actual_start = block_start
    for j in range(block_start, first_chapter_start_idx):
        if j in illustration_lines:
            continue
        stripped = body_lines[j].strip()
        if not stripped or TOC_RE.search(stripped):
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


def _llm_chapter_discovery(
    body_lines: list[str],
    config: Config,
    client: OllamaClient,
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
        data = client.chat_json(config.processing_model, messages)
        chapters = data.get("chapters", [])
        # Normalize to our format
        return [
            {
                "number": ch.get("number", i + 1),
                "title": ch.get("title", f"Chapter {i+1}"),
                "start_line": ch.get("start_line", 1),
                "start_marker": ch.get("title", ""),
            }
            for i, ch in enumerate(chapters)
        ]
    except Exception as e:
        console.print(f"[red]Stage 02: LLM discovery failed: {e}[/red]")
        # Return a single "chapter" covering the whole body
        return [{"number": 1, "title": "Chapter 1", "start_line": 1, "start_marker": ""}]
