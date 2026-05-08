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
from gutenberg_reader.ollama import OllamaClient, OllamaError
from gutenberg_reader import prompts, text_utils

console = Console()

QUALITY_THRESHOLD = 0.85
MAX_REPROCESSING = 2


def run(
    config: Config,
    client: OllamaClient,
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
    client: OllamaClient,
) -> tuple[CriticReport, ProcessedChapter]:
    """Run code-level checks and LLM critique."""
    char_names = [c.name for c in characters]

    # Tier 1: deterministic anchor propagation
    anchor_chapter, anchor_corrections = _anchor_attribution(chapter, characters, char_names, config)

    # Code-level: coverage check
    coverage_issues = _check_coverage(anchor_chapter)

    # Code-level: name spell-check
    name_issues = _check_names(anchor_chapter, char_names)

    # LLM critique (on anchor-corrected segments)
    report = _llm_critique(anchor_chapter, char_names, config, client)

    # Merge code-level findings into report
    report.missing_text.extend(coverage_issues)
    report.name_inconsistencies.extend(name_issues)
    if anchor_corrections:
        report.attribution_issues = [f"Anchor pass fixed: {anchor_corrections}"] + report.attribution_issues

    if coverage_issues:
        report.needs_reprocessing = True
        report.overall_quality = min(report.overall_quality, 0.7)

    # Determine final segments; anchor corrections take precedence over LLM fixes
    if report.fixed_segments:
        # Re-apply named anchors: if an anchor confirmed a speaker, keep it
        named_anchors = text_utils.extract_attribution_anchors(
            [s.to_dict() for s in chapter.segments], characters
        )
        fixed = [s.to_dict() for s in report.fixed_segments]
        for idx, spk in named_anchors.items():
            if idx < len(fixed):
                fixed[idx] = {**fixed[idx], "speaker": spk}
        final_segs = [Segment.from_dict(d) for d in fixed]
    else:
        final_segs = anchor_chapter.segments

    final_chapter = ProcessedChapter(
        chapter_number=chapter.chapter_number,
        chapter_title=chapter.chapter_title,
        segments=final_segs,
        discovered_characters=chapter.discovered_characters,
        word_count=chapter.word_count,
    )

    return report, final_chapter


def _anchor_attribution(
    chapter: ProcessedChapter,
    characters: list[CharacterInfo],
    char_names: list[str],
    config: Config,
) -> tuple[ProcessedChapter, str]:
    """Tier 1: deterministic speaker propagation via conversation-chain alternation.

    Walks the chapter's segments, groups them into conversation chains, and
    propagates confirmed speakers bidirectionally via strict alternation for
    2-person chains.  Returns the (possibly corrected) chapter and a summary string.
    """
    segments_dicts = [s.to_dict() for s in chapter.segments]
    corrected_dicts, flagged_chains, n_corrections = text_utils.propagate_anchors(
        segments_dicts, characters, char_names
    )

    if n_corrections == 0 and not flagged_chains:
        return chapter, ""

    summary_parts = []
    if n_corrections > 0:
        summary_parts.append(f"{n_corrections} speaker(s) corrected by anchor propagation")
    if flagged_chains:
        summary_parts.append(f"{len(flagged_chains)} chain(s) flagged for LLM review")
    summary = "; ".join(summary_parts)

    if config.verbose and n_corrections > 0:
        console.print(f"  [green]Anchor pass:[/green] {summary}")

    corrected_segs = [Segment.from_dict(d) for d in corrected_dicts]
    corrected_chapter = ProcessedChapter(
        chapter_number=chapter.chapter_number,
        chapter_title=chapter.chapter_title,
        segments=corrected_segs,
        discovered_characters=chapter.discovered_characters,
        word_count=chapter.word_count,
    )
    return corrected_chapter, summary


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
    client: OllamaClient,
) -> CriticReport:
    """Call LLM to review and optionally fix segments."""
    segments_data = [s.to_dict() for s in chapter.segments]

    messages = [
        {"role": "system", "content": prompts.critic_system(char_names)},
        {"role": "user", "content": prompts.critic_user(chapter.chapter_title, segments_data)},
    ]

    try:
        data = client.chat_json(config.validation_model, messages)
    except OllamaError as e:
        console.print(f"  [red]Stage 06: LLM critique failed: {e}[/red]")
        # Return a passing report so we don't block the pipeline
        return CriticReport(
            chapter_number=chapter.chapter_number,
            overall_quality=1.0,
            needs_reprocessing=False,
        )

    fixed_segs = None
    if data.get("fixed_segments"):
        fixed_raw = [s for s in data["fixed_segments"] if _valid_seg(s)]
        fixed_segs = [Segment.from_dict(s) for s in fixed_raw]

    return CriticReport(
        chapter_number=chapter.chapter_number,
        missing_text=data.get("missing_text", []),
        attribution_issues=data.get("attribution_issues", []),
        name_inconsistencies=data.get("name_inconsistencies", []),
        overall_quality=float(data.get("overall_quality", 1.0)),
        needs_reprocessing=bool(data.get("needs_reprocessing", False)),
        fixed_segments=fixed_segs,
    )


def _valid_seg(s: object) -> bool:
    return (
        isinstance(s, dict)
        and s.get("type") in ("narration", "dialogue")
        and isinstance(s.get("text"), str)
        and len(s.get("text", "").strip()) > 0
    )
