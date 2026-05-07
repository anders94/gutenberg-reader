"""Stage 03 — Split body text into individual chapter files."""

from __future__ import annotations
from pathlib import Path

from rich.console import Console

from gutenberg_reader.cache import atomic_write_text, chapter_file, read_text, stage_complete
from gutenberg_reader.config import Config
from gutenberg_reader.models import ChapterInfo
from gutenberg_reader import text_utils

console = Console()


def run(config: Config, chapters: list[ChapterInfo]) -> dict[int, Path]:
    """Extract chapter texts and save to 03-chapters/. Returns {chapter_num: path}."""
    stage_dir = config.stage_dir(3)
    raw_path = config.stage_dir(1) / "book.txt"

    result: dict[int, Path] = {}
    all_complete = True

    for ch in chapters:
        out_path = chapter_file(stage_dir, ch.number, ".txt")
        if not (stage_complete(out_path) and (config.force_stage is None or config.force_stage > 3)):
            all_complete = False
            break
        result[ch.number] = out_path

    if all_complete and result:
        if config.verbose:
            console.print(f"[dim]Stage 03: all {len(chapters)} chapters already complete[/dim]")
        return result

    # Read the raw file
    raw_text = read_text(raw_path)
    lines = raw_text.splitlines(keepends=True)

    result = {}
    for ch in chapters:
        # Skip if already complete and not forced
        out_path = chapter_file(stage_dir, ch.number, ".txt")
        if stage_complete(out_path) and (config.force_stage is None or config.force_stage > 3):
            result[ch.number] = out_path
            continue

        # Extract lines (1-indexed, inclusive)
        start_idx = ch.start_line - 1
        end_idx = ch.end_line  # exclusive in slice
        chapter_lines = lines[start_idx:end_idx]
        chapter_text = "".join(chapter_lines)

        # Clean up
        chapter_text = text_utils.strip_illustration_blocks(chapter_text)
        chapter_text = text_utils.collapse_blank_lines(chapter_text)
        chapter_text = chapter_text.strip()

        atomic_write_text(out_path, chapter_text)
        result[ch.number] = out_path

        if config.verbose:
            wc = text_utils.word_count(chapter_text)
            console.print(
                f"[cyan]Stage 03:[/cyan] Chapter {ch.number:02d} "
                f"({wc:,} words) → {out_path.name}"
            )

    return result
