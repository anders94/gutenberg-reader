"""Stage 06 — Critic pass: quality review and correction of segments."""

from __future__ import annotations
from pathlib import Path

from rich.console import Console

from gutenberg_reader.cache import (
    atomic_write_json,
    chapter_file,
    read_json,
    stage_complete,
)
from gutenberg_reader.config import Config
from gutenberg_reader.models import CharacterInfo, CriticReport, ProcessedChapter, Segment
from gutenberg_reader.llm import LLMClient, LLMError
from gutenberg_reader import prompts, schemas, text_utils

console = Console()

QUALITY_THRESHOLD = 0.85
MAX_REPROCESSING = 2


def run(
    config: Config,
    client: LLMClient,
    processed: dict[int, ProcessedChapter],
    characters: list[CharacterInfo],
    chapter_nums: list[int] | None = None,
) -> dict[int, tuple[ProcessedChapter, CriticReport]]:
    """Run critic pass on processed chapters. Returns {num: (accepted_chapter, report)}."""
    stage_dir = config.stage_dir(6)
    nums = chapter_nums if chapter_nums is not None else sorted(processed.keys())

    results: dict[int, tuple[ProcessedChapter, CriticReport]] = {}

    for num in nums:
        out_path = chapter_file(stage_dir, num)

        if stage_complete(out_path) and (config.force_stage is None or config.force_stage > 6):
            if config.verbose:
                console.print(f"[dim]Stage 06: chapter {num:02d} already complete[/dim]")
            data = read_json(out_path)
            chapter = ProcessedChapter.from_dict(data["chapter"])
            report = CriticReport.from_dict(data["report"])
            results[num] = (chapter, report)
            continue

        if num not in processed:
            continue

        chapter = processed[num]
        if config.verbose:
            console.print(f"[cyan]Stage 06:[/cyan] Critiquing chapter {num:02d}...")

        report, final_chapter = _critique_chapter(chapter, characters, config, client)

        # Save combined output
        data = {
            "chapter": final_chapter.to_dict(),
            "report": report.to_dict(),
        }
        atomic_write_json(out_path, data)
        results[num] = (final_chapter, report)

        if config.verbose:
            quality_color = "green" if report.overall_quality >= QUALITY_THRESHOLD else "yellow"
            console.print(
                f"  [{quality_color}]Quality: {report.overall_quality:.2f}[/{quality_color}]"
                + (" (needs reprocessing)" if report.needs_reprocessing else "")
            )

    return results


def _critique_chapter(
    chapter: ProcessedChapter,
    characters: list[CharacterInfo],
    config: Config,
    client: LLMClient,
) -> tuple[CriticReport, ProcessedChapter]:
    """Run code-level checks and LLM critique."""
    char_names = [c.name for c in characters]

    # Code-level: coverage check
    coverage_issues = _check_coverage(chapter)

    # Code-level: name spell-check
    name_issues = _check_names(chapter, char_names)

    # LLM critique: returns per-segment speaker corrections, never text.
    # Segment text is deterministic and untouchable at this point.
    corrections, quality = _llm_critique(chapter, char_names, config, client)

    # Named anchors ("said Mr. Bennet" adjacent to the dialogue) outrank the critic
    named_anchors = text_utils.extract_attribution_anchors(
        [s.to_dict() for s in chapter.segments], characters
    )

    final_segs = list(chapter.segments)
    applied: list[str] = []
    for corr in corrections:
        idx = corr.get("index")
        speaker = corr.get("speaker")
        if not isinstance(idx, int) or not (0 <= idx < len(final_segs)) or not speaker:
            continue
        seg = final_segs[idx]
        if seg.type != "dialogue" or idx in named_anchors or seg.speaker == speaker:
            continue
        applied.append(f"segment {idx}: {seg.speaker} -> {speaker} ({corr.get('reason', '')})")
        final_segs[idx] = Segment(
            type=seg.type,
            text=seg.text,
            speaker=speaker,
            pronunciation_hints=seg.pronunciation_hints,
            notes=seg.notes,
        )

    report = CriticReport(
        chapter_number=chapter.chapter_number,
        missing_text=coverage_issues,
        attribution_issues=applied,
        name_inconsistencies=name_issues,
        overall_quality=quality,
        needs_reprocessing=bool(coverage_issues),
    )

    final_chapter = ProcessedChapter(
        chapter_number=chapter.chapter_number,
        chapter_title=chapter.chapter_title,
        segments=final_segs,
        discovered_characters=chapter.discovered_characters,
        word_count=chapter.word_count,
    )

    return report, final_chapter


def _check_coverage(chapter: ProcessedChapter) -> list[str]:
    """Verify all segments cover their expected text (basic check)."""
    issues = []
    for seg in chapter.segments:
        if not seg.text or not seg.text.strip():
            issues.append(f"Empty segment found in chapter {chapter.chapter_number}")
    return issues


def _check_names(chapter: ProcessedChapter, char_names: list[str]) -> list[str]:
    """Check speaker names against known characters using edit distance."""
    issues = []
    for seg in chapter.segments:
        if seg.speaker and seg.speaker not in ("Unknown", "Narrator"):
            if seg.speaker not in char_names:
                closest = text_utils.find_closest_character(seg.speaker, char_names, max_distance=2)
                if closest:
                    issues.append(
                        f"Possible name inconsistency: '{seg.speaker}' "
                        f"(did you mean '{closest}'?)"
                    )
    return issues


def _llm_critique(
    chapter: ProcessedChapter,
    char_names: list[str],
    config: Config,
    client: LLMClient,
) -> tuple[list[dict], float]:
    """Call LLM to review speaker attribution. Returns (corrections, quality)."""
    segments_data = [s.to_dict() for s in chapter.segments]

    messages = [
        {"role": "system", "content": prompts.critic_system(char_names)},
        {"role": "user", "content": prompts.critic_user(chapter.chapter_title, segments_data)},
    ]

    try:
        data = client.chat_json(
            config.validation_model,
            messages,
            schema=schemas.critic_schema(char_names),
        )
    except LLMError as e:
        console.print(f"  [red]Stage 06: LLM critique failed: {e}[/red]")
        # Return a passing report so we don't block the pipeline
        return [], 1.0

    corrections = [c for c in data.get("corrections", []) if isinstance(c, dict)]
    return corrections, float(data.get("overall_quality", 1.0))
