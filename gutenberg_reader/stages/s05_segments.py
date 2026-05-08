"""Stage 05 — Segment chapters into narration/dialogue with speaker attribution."""

from __future__ import annotations
from pathlib import Path

from rich.console import Console

from gutenberg_reader.cache import (
    atomic_write_json,
    chapter_file,
    read_json,
    read_text,
    stage_complete,
)
from gutenberg_reader.config import Config
from gutenberg_reader.models import CharacterInfo, ProcessedChapter, Segment
from gutenberg_reader.ollama import OllamaClient, OllamaError
from gutenberg_reader import prompts, text_utils

console = Console()


def run(
    config: Config,
    client: OllamaClient,
    chapter_paths: dict[int, Path],
    characters: list[CharacterInfo],
    chapter_nums: list[int] | None = None,
) -> dict[int, ProcessedChapter]:
    """Segment all (or specified) chapters. Returns {chapter_num: ProcessedChapter}."""
    stage_dir = config.stage_dir(5)
    char_names = [c.name for c in characters]
    nums = chapter_nums if chapter_nums is not None else sorted(chapter_paths.keys())

    results: dict[int, ProcessedChapter] = {}

    for num in nums:
        out_path = chapter_file(stage_dir, num)

        # Resume: skip if complete and not forced to this stage
        if stage_complete(out_path) and (config.force_stage is None or config.force_stage > 5):
            if config.verbose:
                console.print(f"[dim]Stage 05: chapter {num:02d} already complete[/dim]")
            data = read_json(out_path)
            results[num] = ProcessedChapter.from_dict(data)
            continue

        if num not in chapter_paths:
            console.print(f"[yellow]Stage 05: chapter {num} not found in chapter_paths, skipping[/yellow]")
            continue

        chapter_text = read_text(chapter_paths[num])
        if not chapter_text.strip():
            console.print(f"[yellow]Stage 05: chapter {num} is empty, skipping[/yellow]")
            continue

        if config.verbose:
            wc = text_utils.word_count(chapter_text)
            console.print(f"[cyan]Stage 05:[/cyan] Segmenting chapter {num:02d} ({wc:,} words)...")

        processed = _segment_chapter(num, chapter_text, char_names, config, client)
        atomic_write_json(out_path, processed.to_dict())
        results[num] = processed

    return results


def _segment_chapter(
    chapter_num: int,
    chapter_text: str,
    char_names: list[str],
    config: Config,
    client: OllamaClient,
) -> ProcessedChapter:
    """Segment a single chapter into narration/dialogue segments."""
    lines = chapter_text.splitlines()
    chapter_title = next((l.strip() for l in lines if l.strip()), f"Chapter {chapter_num}")

    # Split into non-overlapping chunks; use prev_segments as context
    paragraphs = _split_paragraphs(chapter_text)
    chunks = _build_chunks(paragraphs, config.chunk_size)

    all_segments: list[Segment] = []
    discovered_chars: list[CharacterInfo] = []
    prev_context: list[dict] = []  # Last 3 segments from previous chunk (for LLM context)

    for chunk_idx, chunk_text in enumerate(chunks):
        if config.verbose and len(chunks) > 1:
            console.print(
                f"  [dim]Chunk {chunk_idx+1}/{len(chunks)} "
                f"({text_utils.word_count(chunk_text)} words)[/dim]"
            )

        segments, new_chars = _process_chunk_with_retry(
            chunk=chunk_text,
            char_names=char_names,
            prev_segments=prev_context,
            chunk_idx=chunk_idx,
            config=config,
            client=client,
        )

        for nc in new_chars:
            if not any(c.name == nc.name for c in discovered_chars):
                discovered_chars.append(nc)

        all_segments.extend(segments)
        prev_context = [s.to_dict() for s in segments[-3:]] if segments else []

    wc = text_utils.word_count(chapter_text)
    return ProcessedChapter(
        chapter_number=chapter_num,
        chapter_title=chapter_title,
        segments=all_segments,
        discovered_characters=discovered_chars,
        word_count=wc,
    )


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, normalizing Gutenberg line-wrap artifacts.

    Gutenberg wraps prose at ~70 chars. Those line breaks are NOT sentence
    boundaries — joining them prevents the model from creating mid-sentence
    segment splits.
    """
    import re
    paras = re.split(r"\n\n+", text.strip())
    # Collapse intra-paragraph line breaks (Gutenberg wrapping) to spaces
    return [" ".join(p.split()) for p in paras if p.strip()]


def _build_chunks(paragraphs: list[str], chunk_size: int) -> list[str]:
    """Build non-overlapping chunks of approximately chunk_size words."""
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = text_utils.word_count(para)
        if current_words + para_words > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = []
            current_words = 0
        current.append(para)
        current_words += para_words

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [""]


def _process_chunk_with_retry(
    chunk: str,
    char_names: list[str],
    prev_segments: list[dict],
    chunk_idx: int,
    config: Config,
    client: OllamaClient,
) -> tuple[list[Segment], list[CharacterInfo]]:
    """Process a chunk with up to max_retries attempts."""
    system_msg = prompts.segmentation_system(char_names)
    best_segments: list[Segment] = []
    best_issues: list[str] = []

    for attempt in range(config.max_retries):
        if attempt == 0:
            user_msg = prompts.segmentation_user(chunk, prev_segments if chunk_idx > 0 else None)
        else:
            user_msg = prompts.segmentation_retry(chunk, best_issues, prev_segments if chunk_idx > 0 else None)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            data = client.chat_json(config.processing_model, messages)
        except OllamaError as e:
            console.print(f"  [red]LLM error (attempt {attempt+1}): {e}[/red]")
            if attempt == config.max_retries - 1:
                break
            continue

        raw_segments = data.get("segments", [])
        raw_chars = data.get("discovered_characters", [])

        # Filter valid segments
        valid_raw = [s for s in raw_segments if _valid_segment(s)]
        segments = [Segment.from_dict(s) for s in valid_raw]
        new_chars = _parse_chars(raw_chars)

        ok, issues = text_utils.verify_segment_coverage(chunk, valid_raw)

        if ok:
            return segments, new_chars

        # Try punctuation-repair before giving up on this attempt
        repaired_raw = text_utils.repair_segment_texts(chunk, valid_raw)
        if repaired_raw is not None:
            if config.verbose:
                console.print(f"  [dim]Repaired punctuation on attempt {attempt+1}[/dim]")
            return [Segment.from_dict(s) for s in repaired_raw], new_chars

        # Keep best attempt (fewest issues)
        if not best_segments or len(issues) < len(best_issues):
            best_segments = [Segment.from_dict(s) for s in valid_raw]
            best_issues = issues

        console.print(
            f"  [yellow]Integrity check failed (attempt {attempt+1}/{config.max_retries}): "
            f"{len(issues)} issue(s)[/yellow]"
        )

    # Max retries reached — try splitting the chunk in half as a last resort
    paragraphs = _split_paragraphs(chunk)
    if len(paragraphs) >= 2:
        mid = len(paragraphs) // 2
        half_a = "\n\n".join(paragraphs[:mid])
        half_b = "\n\n".join(paragraphs[mid:])
    else:
        # Single paragraph — split at the midpoint word boundary
        words = chunk.split()
        if len(words) >= 20:
            mid_w = len(words) // 2
            half_a = " ".join(words[:mid_w])
            half_b = " ".join(words[mid_w:])
        else:
            half_a = half_b = ""  # too small to split further

    if half_a and half_b:
        if config.verbose:
            console.print(f"  [dim]Splitting chunk in half as fallback...[/dim]")
        segs_a, chars_a = _process_chunk_with_retry(
            half_a, char_names, prev_segments, chunk_idx, config, client
        )
        segs_b, chars_b = _process_chunk_with_retry(
            half_b, char_names, [s.to_dict() for s in segs_a[-3:]], chunk_idx, config, client
        )
        all_chars = list({c.name: c for c in chars_a + chars_b}.values())
        return segs_a + segs_b, all_chars

    issues_str = "\n    ".join(best_issues[:5])
    raise RuntimeError(
        f"Integrity check failed after {config.max_retries} attempts.\n"
        f"    {issues_str}\n"
        f"Try reducing --chunk-size (currently {config.chunk_size}) or increasing --max-retries."
    )


def _valid_segment(s: object) -> bool:
    """Basic validation of a segment dict."""
    return (
        isinstance(s, dict)
        and s.get("type") in ("narration", "dialogue")
        and isinstance(s.get("text"), str)
        and len(s.get("text", "").strip()) > 0
    )


def _parse_chars(raw: list) -> list[CharacterInfo]:
    result = []
    for c in raw:
        try:
            result.append(CharacterInfo.from_dict(c))
        except Exception:
            pass
    return result
