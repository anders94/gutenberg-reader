"""Stage 03 — Split body text into individual chapter files."""

from __future__ import annotations
from pathlib import Path

from rich.console import Console

from gutenberg_reader.cache import (
    atomic_write_json, atomic_write_text, chapter_file, read_json, read_text,
    stage_complete,
)
from gutenberg_reader.config import Config
from gutenberg_reader.models import ChapterInfo
from gutenberg_reader import text_utils

console = Console()

# Bumped when the text written per chapter changes shape. Together with each
# chapter's line range this is what makes a cached chapter file trustworthy.
CHAPTER_FORMAT = 1


def _manifest_path(config: Config) -> Path:
    return config.stage_dir(3) / "manifest.json"


def _fingerprint(ch: ChapterInfo) -> dict:
    return {"start_line": ch.start_line, "end_line": ch.end_line,
            "format": CHAPTER_FORMAT}


def run(config: Config, chapters: list[ChapterInfo]) -> dict[int, Path]:
    """Extract chapter texts and save to 03-chapters/. Returns {chapter_num: path}."""
    stage_dir = config.stage_dir(3)
    raw_path = config.stage_dir(1) / "book.txt"

    # A chapter file is keyed by number, but the number means nothing without
    # the lines behind it: re-running PG 6400 after its structure was fixed would
    # otherwise reuse "chapter 2" from the broken 3-chapter split, a 173,461-word
    # span, as chapter 2 of the corrected twenty.
    manifest = read_json(_manifest_path(config)) if _manifest_path(config).exists() else {}

    def cached_ok(ch: ChapterInfo, path: Path) -> bool:
        return (
            stage_complete(path)
            and (config.force_stage is None or config.force_stage > 3)
            and manifest.get(str(ch.number)) == _fingerprint(ch)
        )

    result: dict[int, Path] = {}
    all_complete = True

    for ch in chapters:
        out_path = chapter_file(stage_dir, ch.number, ".txt")
        if not cached_ok(ch, out_path):
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
        if cached_ok(ch, out_path):
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
        manifest[str(ch.number)] = _fingerprint(ch)
        result[ch.number] = out_path

        if config.verbose:
            wc = text_utils.word_count(chapter_text)
            console.print(
                f"[cyan]Stage 03:[/cyan] Chapter {ch.number:02d} "
                f"({wc:,} words) → {out_path.name}"
            )

    atomic_write_json(_manifest_path(config), manifest)
    return result
