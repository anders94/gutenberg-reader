"""Stage 07 — Assemble final output JSON."""

from __future__ import annotations
import time
from pathlib import Path

from rich.console import Console

from gutenberg_reader.cache import atomic_write_json, chapter_file, read_json, stage_complete
from gutenberg_reader.config import Config
from gutenberg_reader.models import (
    BookMetadata,
    CharacterInfo,
    ChapterInfo,
    CriticReport,
    ProcessedChapter,
)
from gutenberg_reader import text_utils

console = Console()

PIPELINE_VERSION = "1.0.0"


def run(
    config: Config,
    metadata: BookMetadata,
    chapter_infos: list[ChapterInfo],
    accepted: dict[int, tuple[ProcessedChapter, CriticReport | None]],
    start_time: float,
) -> Path:
    """Assemble and save final JSON. Returns path to output file."""
    stage_dir = config.stage_dir(7)
    out_path = config.output_file or stage_dir / f"{config.book_id}.json"

    if stage_complete(out_path) and (config.force_stage is None or config.force_stage > 7):
        if config.verbose:
            console.print(f"[dim]Stage 07: already complete ({out_path})[/dim]")
        return out_path

    if config.verbose:
        console.print("[cyan]Stage 07:[/cyan] Assembling final output...")

    # Build chapter_info lookup
    info_by_num = {ci.number: ci for ci in chapter_infos}

    chapters_out = []
    all_chars: dict[str, CharacterInfo] = {}
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

    elapsed = time.time() - start_time
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 1.0
    min_quality = min(quality_scores) if quality_scores else 1.0

    output = {
        "metadata": metadata.to_dict(),
        "chapters": chapters_out,
        "characters": [c.to_dict() for c in all_chars.values()],
        "statistics": {
            "total_chapters": len(chapters_out),
            "total_words": total_words,
            "total_segments": total_segments,
            "total_characters": len(all_chars),
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
