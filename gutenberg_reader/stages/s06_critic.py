"""Stage 06 — Critic pass: quality review and correction of segments.

Runs inside stage 05's chapter loop, right after attribution, so its verdicts
can act while they still matter: besides correcting speaker labels, the critic
reviews the characters discovery just added and can strike a ship, a cited
author, or a duplicate from the roster before the next chapter's attribution
ever sees it as an option.
"""

from __future__ import annotations
from dataclasses import replace

from rich.console import Console

from gutenberg_reader.cache import (
    atomic_write_json,
    chapter_file,
    read_json,
    stage_complete,
)
from gutenberg_reader.config import Config
from gutenberg_reader.models import CharacterInfo, CriticReport, ProcessedChapter, Segment
from gutenberg_reader.llm import LLMRouter, call_json_with_retries
from gutenberg_reader import prompts, schemas, text_utils

console = Console()

QUALITY_THRESHOLD = 0.85

# Segments from the previous window included (read-only) for continuity
CONTEXT_SEGMENTS = 8


def run_chapter(
    config: Config,
    client: LLMRouter,
    chapter: ProcessedChapter,
    roster: list[CharacterInfo],
    new_names: list[str],
    force: bool = False,
) -> tuple[ProcessedChapter, CriticReport, list[dict]]:
    """Critique one chapter. Returns (accepted_chapter, report, roster_issues).

    roster_issues are the critic's raw objections to this chapter's roster
    additions; the caller applies them (see apply_roster_issues) so anchor
    protection and the rolling roster stay in one place.
    """
    num = chapter.chapter_number
    out_path = chapter_file(config.stage_dir(6), num)

    if not force and stage_complete(out_path) and (
            config.force_stage is None or config.force_stage > 6):
        if config.verbose:
            console.print(f"[dim]Stage 06: chapter {num:02d} already complete[/dim]")
        data = read_json(out_path)
        return (
            ProcessedChapter.from_dict(data["chapter"]),
            CriticReport.from_dict(data["report"]),
            data.get("roster_issues", []),
        )

    if config.verbose:
        console.print(f"[cyan]Stage 06:[/cyan] Critiquing chapter {num:02d}...")

    report, final_chapter, roster_issues = _critique_chapter(
        chapter, roster, new_names, config, client
    )

    data = {
        "chapter": final_chapter.to_dict(),
        "report": report.to_dict(),
        "roster_issues": roster_issues,
    }
    atomic_write_json(out_path, data)

    if config.verbose:
        quality_color = "green" if report.overall_quality >= QUALITY_THRESHOLD else "yellow"
        console.print(
            f"  [{quality_color}]Quality: {report.overall_quality:.2f}[/{quality_color}]"
            + (" (needs reprocessing)" if report.needs_reprocessing else "")
        )

    return final_chapter, report, roster_issues


def apply_roster_issues(
    roster: list[CharacterInfo],
    issues: list[dict],
    protected: set[str],
) -> tuple[list[CharacterInfo], list[str]]:
    """Apply the critic's roster objections. Pure; returns (roster, log lines).

    Effects are forward-only: they change which names later chapters can
    attribute to, never labels already written (stage 07's remap reconciles
    those). Names in `protected` (lowercase) were anchor-established — the
    author wrote "said X" — and outrank the critic, exactly as named anchors
    outrank its segment corrections.
    """
    by_name = {c.name.lower(): c for c in roster}
    applied: list[str] = []

    for issue in issues:
        if not isinstance(issue, dict):
            continue
        name = issue.get("name", "")
        verdict = issue.get("verdict")
        entry = by_name.get(name.lower())
        if entry is None:
            continue
        if entry.name.lower() in protected or any(a.lower() in protected for a in entry.aliases):
            continue

        if verdict == "not_a_character":
            del by_name[entry.name.lower()]
            applied.append(f"dropped {entry.name!r} ({issue.get('reason', '')})")
        elif verdict == "duplicate":
            canonical = by_name.get(issue.get("canonical", "").lower())
            if canonical is None or canonical is entry:
                continue
            for alias in [entry.name, *entry.aliases]:
                if alias.lower() != canonical.name.lower() and all(
                    alias.lower() != a.lower() for a in canonical.aliases
                ):
                    canonical.aliases.append(alias)
            canonical.first_appearance_chapter = min(
                canonical.first_appearance_chapter, entry.first_appearance_chapter
            )
            del by_name[entry.name.lower()]
            applied.append(
                f"merged {entry.name!r} into {canonical.name!r} ({issue.get('reason', '')})"
            )

    return list(by_name.values()), applied


# Characters per roster-review call, and how much text to show for each. A few
# hundred characters either side of a mention is what settles whether a name is
# a person, a ship or a cited author.
ROSTER_EVIDENCE_SNIPPETS = 3
ROSTER_EVIDENCE_CHARS = 120


def _roster_evidence(chapter: ProcessedChapter, name: str) -> list[str]:
    """Where this name actually appears, sliced by code from the chapter."""
    parts: list[str] = []
    needle = name.lower()
    for seg in chapter.segments:
        low = seg.text.lower()
        pos = low.find(needle)
        if pos < 0:
            continue
        a = max(0, pos - ROSTER_EVIDENCE_CHARS)
        b = min(len(seg.text), pos + len(name) + ROSTER_EVIDENCE_CHARS)
        parts.append(seg.text[a:b].replace("\n", " "))
        if len(parts) >= ROSTER_EVIDENCE_SNIPPETS:
            break
    return parts


def _review_roster(
    chapter: ProcessedChapter,
    char_names: list[str],
    new_names: list[str],
    config: Config,
    client: LLMRouter,
) -> list[dict]:
    """One verdict per new name, asked once with the passages that mention it.

    Attached to the attribution windows this was asked five or ten times a
    chapter, and issues_by_name.setdefault meant the first window's answer won by
    accident of ordering rather than by being the best-informed.
    """
    if not new_names:
        return []
    evidence = {n: _roster_evidence(chapter, n) for n in dict.fromkeys(new_names)}
    data = call_json_with_retries(
        client, config.validation_model,
        [{"role": "system", "content": prompts.roster_review_system()},
         {"role": "user", "content": prompts.roster_review_user(
             chapter.chapter_title, evidence)}],
        schema=schemas.roster_review_schema(char_names, new_names),
        retries=config.max_retries, what="roster review", console=console,
    )
    if data is None:
        return []
    # "keep" is a verdict, not an objection — only the rest travel onward.
    return [
        i for i in data.get("roster_issues", [])
        if isinstance(i, dict) and i.get("verdict") in ("not_a_character", "duplicate")
    ]


def _critique_chapter(
    chapter: ProcessedChapter,
    roster: list[CharacterInfo],
    new_names: list[str],
    config: Config,
    client: LLMRouter,
) -> tuple[CriticReport, ProcessedChapter, list[dict]]:
    """Run code-level checks and LLM critique."""
    char_names = [c.name for c in roster]

    # Code-level: coverage check
    coverage_issues = _check_coverage(chapter)

    # Code-level: name spell-check
    name_issues = _check_names(chapter, char_names)

    # LLM critique: returns per-segment speaker corrections and roster
    # objections, never text. Segment text is deterministic and untouchable.
    corrections, quality, unreviewed = _llm_critique(
        chapter, char_names, new_names, config, client
    )
    # Asked once for the chapter, with evidence, rather than once per window
    # from memory.
    roster_issues = _review_roster(chapter, char_names, new_names, config, client)

    # Named anchors ("said Mr. Bennet" adjacent to the dialogue) outrank the critic
    named_anchors = text_utils.extract_attribution_anchors(
        [s.to_dict() for s in chapter.segments], roster
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
        # replace() rather than rebuilding field by field: the critic changes a
        # speaker label and nothing else, and a hand-listed constructor silently
        # drops whatever was added to the model since it was written.
        final_segs[idx] = replace(seg, speaker=speaker)

    report = CriticReport(
        chapter_number=chapter.chapter_number,
        missing_text=coverage_issues,
        attribution_issues=applied,
        name_inconsistencies=name_issues,
        overall_quality=quality,
        unreviewed_windows=unreviewed,
        # Coverage is an assertion in stage 05 now, so an empty segment cannot
        # reach here. What can is a chapter the critic thinks is poor, or one it
        # only partly saw — both are worth another pass.
        needs_reprocessing=quality < QUALITY_THRESHOLD or bool(unreviewed),
    )

    final_chapter = ProcessedChapter(
        chapter_number=chapter.chapter_number,
        chapter_title=chapter.chapter_title,
        segments=final_segs,
        discovered_characters=chapter.discovered_characters,
        word_count=chapter.word_count,
    )

    return report, final_chapter, roster_issues


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
    new_names: list[str],
    config: Config,
    client: LLMRouter,
) -> tuple[list[dict], float, list[list[int]]]:
    """Review attribution window by window.

    Returns (corrections, quality, unreviewed_windows).

    Runs window by window, like the attribution passes in stage 05: a whole
    chapter in one prompt overruns the server's context window on long chapters
    (Moby Dick's longest is 7,918 words), and the request fails outright.
    """
    segments_data = [s.to_dict() for s in chapter.segments]
    system_msg = prompts.critic_system(char_names, None)
    schema = schemas.critic_schema(char_names)

    corrections: list[dict] = []
    qualities: list[tuple[float, int]] = []  # (quality, dialogue segments judged)
    unreviewed: list[list[int]] = []

    for start, end in text_utils.build_segment_windows(segments_data, config.chunk_size):
        ctx_start = max(0, start - CONTEXT_SEGMENTS)
        window = segments_data[ctx_start:end]
        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": prompts.critic_user(
                    chapter.chapter_title, window, ctx_start, start - ctx_start
                ),
            },
        ]

        # Skip a window rather than the chapter: the rest still reviews. Phase 5
        # records what was skipped so a chapter cannot report a quality score
        # for work no one looked at.
        data = call_json_with_retries(
            client, config.validation_model, messages, schema=schema,
            retries=config.max_retries, what="critic window", console=console,
        )
        if data is None:
            unreviewed.append([start, end])
            continue

        # Corrections aimed at [CONTEXT] lines belong to the window that owned
        # them; that window already judged them with its own full context.
        corrections.extend(
            c for c in data.get("corrections", [])
            if isinstance(c, dict) and isinstance(c.get("index"), int) and start <= c["index"] < end
        )
        n_dialogue = sum(1 for s in segments_data[start:end] if s.get("type") == "dialogue")
        qualities.append((float(data.get("overall_quality", 1.0)), n_dialogue))

    if not qualities:
        # Every window failed. A passing score here would be a lie about work
        # that never happened; 0.0 with the windows recorded is what it is, and
        # needs_reprocessing acts on it.
        return [], 0.0, unreviewed

    judged = sum(n for _, n in qualities)
    if judged == 0:
        quality = sum(q for q, _ in qualities) / len(qualities)
    else:
        quality = sum(q * n for q, n in qualities) / judged
    return corrections, quality, unreviewed
