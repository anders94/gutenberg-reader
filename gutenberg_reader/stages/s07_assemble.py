"""Stage 07 — Assemble final output JSON."""

from __future__ import annotations
import time
from pathlib import Path

from rich.console import Console

from gutenberg_reader.cache import atomic_write_json
from gutenberg_reader.config import Config
from gutenberg_reader import text_utils
from gutenberg_reader.models import (
    BookMetadata,
    CharacterInfo,
    ChapterInfo,
    CriticReport,
    ProcessedChapter,
)

console = Console()

PIPELINE_VERSION = "1.0.0"


def _output_path(config: Config) -> Path:
    """Where the final JSON goes.

    A --chapters run holds only part of the book, so it gets its own file:
    writing it to the canonical path would replace a complete book with a
    fragment, and the next full run would happily serve that fragment.
    """
    if config.output_file:
        return config.output_file

    stage_dir = config.stage_dir(7)
    if not config.chapters_only:
        return stage_dir / f"{config.book_id}.json"

    nums = sorted(config.chapters_only)
    span = "-".join(str(n) for n in nums) if len(nums) <= 6 else f"{nums[0]}_{nums[-1]}"
    return stage_dir / f"{config.book_id}-ch{span}.json"


def run(
    config: Config,
    metadata: BookMetadata,
    chapter_infos: list[ChapterInfo],
    accepted: dict[int, tuple[ProcessedChapter, CriticReport | None]],
    characters: list[CharacterInfo],
    start_time: float,
) -> Path:
    """Assemble and save final JSON. Returns path to output file."""
    out_path = _output_path(config)

    # Assembly is always re-run: it takes milliseconds, and its inputs (stages
    # 05/06) change on every resumed run. Skipping it when the file merely
    # exists would keep serving an output assembled from fewer chapters.

    if config.verbose:
        console.print("[cyan]Stage 07:[/cyan] Assembling final output...")

    # Build chapter_info lookup
    info_by_num = {ci.number: ci for ci in chapter_infos}

    chapters_out = []
    # Stage-04 roster first; per-chapter tag discoveries merge in after
    all_chars: dict[str, CharacterInfo] = {c.name.lower(): c for c in characters}
    total_words = 0
    total_segments = 0
    quality_scores: list[float] = []

    for num in sorted(accepted.keys()):
        processed, report = accepted[num]
        ci = info_by_num.get(num)

        chapter_entry = {
            "chapter": {
                "number": processed.chapter_number,
                "title": processed.chapter_title,
                "text": "",  # raw text omitted from final output
                "word_count": processed.word_count,
                "start_marker": ci.start_marker if ci else processed.chapter_title,
            },
            "processed": processed.to_dict(),
            "validation": report.to_dict() if report else None,
        }
        chapters_out.append(chapter_entry)

        # Accumulate stats
        total_words += processed.word_count
        total_segments += len(processed.segments)

        if report:
            quality_scores.append(report.overall_quality)

        # Collect characters
        for char in processed.discovered_characters:
            key = char.name.lower()
            if key not in all_chars:
                all_chars[key] = char
            else:
                # Merge aliases
                for alias in char.aliases:
                    if alias not in all_chars[key].aliases:
                        all_chars[key].aliases.append(alias)

    # Regularize: chapters discover partial names ("said Sir Harry" in a
    # chapter whose roster only knows Sir Harry Otway), so the same person
    # can arrive under several names. Merge the roster, then remap every
    # segment speaker to its canonical name.
    final_chars = text_utils.merge_duplicate_characters(list(all_chars.values()))
    alias_map = text_utils._build_alias_map(final_chars)
    n_remapped = 0
    for entry in chapters_out:
        for seg in entry["processed"]["segments"]:
            speaker = seg.get("speaker")
            if speaker and speaker not in ("Unknown", "Narrator"):
                canonical = alias_map.get(speaker.lower())
                if canonical and canonical != speaker:
                    seg["speaker"] = canonical
                    n_remapped += 1
    if config.verbose and (n_remapped or len(final_chars) != len(all_chars)):
        console.print(
            f"  [dim]Regularized {len(all_chars)} -> {len(final_chars)} characters, "
            f"remapped {n_remapped} speaker labels[/dim]"
        )

    elapsed = time.time() - start_time
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 1.0
    min_quality = min(quality_scores) if quality_scores else 1.0

    output = {
        "metadata": metadata.to_dict(),
        "chapters": chapters_out,
        "characters": [c.to_dict() for c in final_chars],
        "statistics": {
            "total_chapters": len(chapters_out),
            "total_words": total_words,
            "total_segments": total_segments,
            "total_characters": len(final_chars),
            "processing_time_seconds": round(elapsed, 2),
            "validation_performed": not config.no_critic,
            "pipeline_version": PIPELINE_VERSION,
            "discovery_confidence": {
                "avg_confidence": avg_quality,
                "min_confidence": min_quality,
            },
        },
        "processing_config": {
            "processing_model": config.processing_model,
            "validation_model": config.validation_model,
            "dual_llm_validation": config.processing_model != config.validation_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        },
    }

    atomic_write_json(out_path, output)

    if config.verbose:
        console.print(
            f"[green]Stage 07:[/green] Saved {len(chapters_out)} chapters, "
            f"{total_words:,} words, {total_segments:,} segments → {out_path}"
        )

    return out_path
