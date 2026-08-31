"""Stage 05 — Read chapters in order: discover characters, segment, attribute.

Segmentation into narration/dialogue is fully deterministic (quotation marks
delimit dialogue — see segmenter.py), so text coverage is guaranteed by
construction.

Characters are discovered chapter by chapter in this same loop and merged into
a rolling roster, so chapter N's speaker enum contains exactly the characters
the book has introduced by chapter N — a speaker cannot be someone the book
has not met yet. The critic (stage 06) runs here too, right after each
chapter's attribution: besides correcting labels, it reviews the names this
chapter added and can strike a ship or a cited author from the roster before
the next chapter ever sees it as an option. Stage 07's regularize-and-remap
is the backward fix-up that makes forward-only naming safe (early chapters
labeled "Ahab" get remapped once "Captain Ahab" becomes canonical).

Speaker attribution favors accuracy over cost:

  1. Deterministic anchors: "said Mr. Bennet" narration adjacent to dialogue.
     These are the author naming the speaker — kept as hard evidence, and
     they protect the named character from critic roster objections.
  2. Everything else goes through three LLM passes (guided decoding constrains
     speakers to the rolling roster):
       a. opportunistic — best-guess attribution with full context
       b. critical — independently re-derives every non-anchored speaker,
          blind to pass (a)'s answers so its errors don't correlate
       c. tie-breaker — sees both candidates and resolves disagreements

Resume: each chapter's cache file carries a roster_after snapshot (and the
cumulative anchor-name set), so chapter N+1's inputs are recoverable without
re-reading chapters 1..N. Files without a snapshot predate this design and
are treated as incomplete.
"""

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
from gutenberg_reader.models import CharacterInfo, CriticReport, ProcessedChapter, Segment
from gutenberg_reader.llm import LLMRouter, call_json_with_retries
from gutenberg_reader import prompts, schemas, segmenter, text_utils
from gutenberg_reader.stages import s04_characters, s06_critic

console = Console()

# Segments from the previous window included (read-only) for continuity
CONTEXT_SEGMENTS = 8


def run(
    config: Config,
    client: LLMRouter,
    chapter_paths: dict[int, Path],
    chapter_nums: list[int] | None = None,
    chapter_titles: dict[int, str] | None = None,
) -> tuple[dict[int, tuple[ProcessedChapter, CriticReport | None]], list[CharacterInfo]]:
    """Process chapters in reading order.

    Returns ({chapter_num: (accepted_chapter, critic_report_or_None)}, roster).
    """
    stage_dir = config.stage_dir(5)
    nums = chapter_nums if chapter_nums is not None else sorted(chapter_paths.keys())
    critic_on = not config.no_critic

    roster: list[CharacterInfo] = []
    protected: set[str] = set()  # lowercase anchor-established names
    accepted: dict[int, tuple[ProcessedChapter, CriticReport | None]] = {}

    for num in nums:
        out_path = chapter_file(stage_dir, num)

        cached = _load_cached(out_path, config)
        if cached is not None:
            chapter, roster, protected = cached
            if config.verbose:
                console.print(f"[dim]Stage 05: chapter {num:02d} already complete[/dim]")
            report = None
            if critic_on:
                chapter, report, roster, protected = _run_critic(
                    config, client, chapter, roster, protected
                )
            accepted[num] = (chapter, report)
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
            console.print(f"[cyan]Stage 05:[/cyan] Chapter {num:02d} ({wc:,} words)...")

        names_before = {c.name.lower() for c in roster}

        # Discover this chapter's characters into the rolling roster.
        found = s04_characters.discover_in_chapter(chapter_text, num, config, client)
        roster = text_utils.merge_rosters(roster, found)
        # Regularize as we go so the enum never splits one person across two
        # entries ("Peleg" / "Captain Peleg") for the rest of the book.
        roster = text_utils.merge_duplicate_characters(roster)

        # Segment and attribute (may add anchor-named characters to the roster).
        processed, roster, chapter_anchors = _segment_chapter(
            num, chapter_text, roster, config, client,
            title=(chapter_titles or {}).get(num),
        )
        protected = protected | chapter_anchors

        # Provenance: what this chapter added (by name, post-merge).
        processed.discovered_characters = [
            c for c in roster if c.name.lower() not in names_before
        ]

        report = None
        final_chapter = processed
        if critic_on:
            final_chapter, report, roster, protected = _run_critic(
                config, client, processed, roster, protected
            )

        # Snapshot after the critic so chapter N+1 resumes from the exact
        # roster state it would see live.
        data = processed.to_dict()
        data["roster_after"] = [c.to_dict() for c in roster]
        data["anchor_names"] = sorted(protected)
        atomic_write_json(out_path, data)

        accepted[num] = (final_chapter, report)

    # Final roster artifact — same path stage 04 always wrote, for inspection.
    if roster:
        atomic_write_json(
            config.stage_dir(4) / "characters.json",
            {"characters": [c.to_dict() for c in roster]},
        )

    return accepted, roster


def _run_critic(
    config: Config,
    client: LLMRouter,
    chapter: ProcessedChapter,
    roster: list[CharacterInfo],
    protected: set[str],
) -> tuple[ProcessedChapter, CriticReport, list[CharacterInfo], set[str]]:
    """Critique one chapter and apply its roster objections (forward-only)."""
    new_names = [c.name for c in chapter.discovered_characters]
    final_chapter, report, issues = s06_critic.run_chapter(
        config, client, chapter, roster, new_names
    )
    # Idempotent: on a resumed chapter whose snapshot already reflects these
    # issues, the entries are gone and every issue is a no-op.
    roster, applied = s06_critic.apply_roster_issues(roster, issues, protected)
    if applied and config.verbose:
        for line in applied:
            console.print(f"  [dim]roster: {line}[/dim]")
    return final_chapter, report, roster, protected


def _load_cached(
    out_path: Path,
    config: Config,
) -> tuple[ProcessedChapter, list[CharacterInfo], set[str]] | None:
    """Load a chapter's cache entry, or None if absent/forced/pre-snapshot.

    Returns (chapter, roster_after, anchor_names). Files without a
    roster_after snapshot predate the rolling-roster design: without the
    snapshot the next chapter's inputs are unrecoverable, so they count as
    incomplete and the chapter re-runs.
    """
    if not stage_complete(out_path):
        return None
    if config.force_stage is not None and config.force_stage <= 5:
        return None
    data = read_json(out_path)
    if "roster_after" not in data:
        return None
    return (
        ProcessedChapter.from_dict(data),
        [CharacterInfo.from_dict(c) for c in data["roster_after"]],
        set(data.get("anchor_names", [])),
    )


def _segment_chapter(
    chapter_num: int,
    chapter_text: str,
    roster: list[CharacterInfo],
    config: Config,
    client: LLMRouter,
    title: str | None = None,
) -> tuple[ProcessedChapter, list[CharacterInfo], set[str]]:
    """Segment and attribute one chapter.

    Returns (processed, roster, anchor_names): the roster possibly extended
    with anchor-named characters discovery missed, and the lowercase names
    anchored by explicit attribution tags in this chapter.
    """
    lines = chapter_text.splitlines()
    # Stage 02 is the authority on titles. Reading the first non-blank line back
    # out of the text only agrees when the heading is one line: PG 37106 centres
    # the numeral above the title, so that guess yields "I." for a chapter
    # discovery correctly called "I. PLAYING PILGRIMS.".
    chapter_title = title or next(
        (l.strip() for l in lines if l.strip()), f"Chapter {chapter_num}"
    )

    # Tier 0: deterministic segmentation
    segments = segmenter.segment_text(chapter_text)

    ok, issues = text_utils.verify_segment_coverage(chapter_text, segments)
    if not ok:
        # Should be impossible; if it happens, the segmenter has a bug worth surfacing
        console.print(
            f"  [yellow]Coverage warning in chapter {chapter_num}: {issues[:2]}[/yellow]"
        )

    # Tier 1: deterministic attribution anchors ("said Mr. Bennet" adjacent to dialogue)
    anchors = text_utils.extract_attribution_anchors(segments, roster)
    for idx, name in anchors.items():
        segments[idx]["speaker"] = name

    # Tags can name speakers discovery missed ("said Lydia,") — merge them into
    # the roster so the LLM enum and the final JSON know them.
    char_names = [c.name for c in roster]
    extra_names = sorted({n for n in anchors.values() if n not in char_names})
    roster = text_utils.merge_rosters(
        roster,
        [CharacterInfo(name=n, first_appearance_chapter=chapter_num) for n in extra_names],
    )
    char_names = [c.name for c in roster]
    anchor_names = {n.lower() for n in anchors.values()}

    # Tier 1b: LLM-resolve nameless attribution tags ("said his lady", "returned she")
    # to character names — a much easier task than free attribution — then anchor
    # the adjacent dialogue exactly as named tags do.
    n_tag = _resolve_nameless_tags(segments, roster, char_names, config, client)

    # Pass A (opportunistic): best-guess LLM attribution for unanchored dialogue
    unresolved = {
        i for i, s in enumerate(segments)
        if s["type"] == "dialogue" and not s.get("speaker")
    }
    proposed = _llm_window_pass(
        segments, unresolved, config, client,
        system_msg=prompts.attribution_system(char_names),
        user_fn=prompts.attribution_user,
        schema=schemas.attribution_schema(char_names),
    )
    for idx, speaker in proposed.items():
        segments[idx]["speaker"] = speaker

    # Pass B (critical): independently re-derive every non-anchored speaker.
    # Runs on the validator model — pointing --validator at a larger model
    # (it defaults to the processing model) buys accuracy exactly where it
    # matters, and decorrelates the two passes' errors even at equal size.
    verified = _llm_window_pass(
        segments, unresolved, config, client,
        system_msg=prompts.verify_attribution_system(char_names),
        user_fn=prompts.verify_attribution_user,
        schema=schemas.attribution_schema(char_names),
        model=config.validation_model,
    )
    disputes: dict[int, tuple[str, str]] = {}
    for idx, speaker in verified.items():
        current = segments[idx].get("speaker")
        if current is None:
            segments[idx]["speaker"] = speaker
        elif speaker != current:
            disputes[idx] = (current, speaker)

    # Pass C (tie-breaker): resolve disagreements between A and B
    if disputes:
        broken = _llm_window_pass(
            segments, set(disputes), config, client,
            system_msg=prompts.tiebreak_system(char_names),
            user_fn=lambda w, s, f, c: prompts.tiebreak_user(w, s, f, c, disputes),
            schema=schemas.attribution_schema(char_names),
            model=config.validation_model,
        )
        for idx in disputes:
            # Two passes disagreed and the arbiter didn't rule: Unknown is
            # more honest than either guess.
            segments[idx]["speaker"] = broken.get(idx, "Unknown")

    n_unknown = 0
    for s in segments:
        if s["type"] == "dialogue" and not s.get("speaker"):
            s["speaker"] = "Unknown"
            n_unknown += 1

    if config.verbose:
        n_dialogue = sum(1 for s in segments if s["type"] == "dialogue")
        console.print(
            f"  [dim]{len(segments)} segments, {n_dialogue} dialogue: "
            f"{len(anchors)} anchored, {n_tag} tag-resolved, {len(proposed)} proposed, "
            f"{len(disputes)} disputed, {n_unknown} unknown[/dim]"
        )

    processed = ProcessedChapter(
        chapter_number=chapter_num,
        chapter_title=chapter_title,
        segments=[Segment.from_dict(s) for s in segments],
        word_count=text_utils.word_count(chapter_text),
    )
    return processed, roster, anchor_names


def _llm_window_pass(
    segments: list[dict],
    flagged_all: set[int],
    config: Config,
    client: LLMRouter,
    system_msg: str,
    user_fn,
    schema: dict,
    model: str | None = None,
) -> dict[int, str]:
    """Run an LLM pass over flagged segment indices, window by window.

    Returns {segment_index: speaker} for every flagged index the LLM answered
    with something other than "Unknown"-by-omission. Does not mutate segments.
    model defaults to the processing model.
    """
    results: dict[int, str] = {}
    model = model or config.processing_model
    if not flagged_all:
        return results

    for start, end in text_utils.build_segment_windows(segments, config.chunk_size):
        flagged = {i for i in flagged_all if start <= i < end}
        if not flagged:
            continue

        ctx_start = max(0, start - CONTEXT_SEGMENTS)
        window = segments[ctx_start:end]
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_fn(window, ctx_start, flagged, start - ctx_start)},
        ]

        data = call_json_with_retries(
            client, model, messages, schema=schema,
            retries=config.max_retries, what="attribution window", console=console,
        )
        if data is None:
            continue
        for a in data.get("attributions", []):
            idx = a.get("index")
            if idx in flagged and a.get("speaker"):
                results[idx] = a["speaker"]

    return results


def _resolve_nameless_tags(
    segments: list[dict],
    characters: list[CharacterInfo],
    char_names: list[str],
    config: Config,
    client: LLMRouter,
) -> int:
    """Resolve attribution tags that lack a character name and anchor adjacent dialogue.

    "said his lady," identifies the speaker of the surrounding dialogue as
    precisely as "said Mrs. Bennet" — once the referring expression is resolved.
    Mutates segments in place; returns the number of dialogue segments anchored.
    """
    alias_map = text_utils._build_alias_map(characters)
    nameless = {
        i for i, s in enumerate(segments)
        if text_utils._is_attribution_narration(s)
        and text_utils._find_char_in_text(s.get("text", ""), alias_map) is None
    }
    if not nameless:
        return 0

    resolved = _llm_window_pass(
        segments, nameless, config, client,
        system_msg=prompts.tag_resolution_system(char_names),
        user_fn=prompts.tag_resolution_user,
        schema=schemas.attribution_schema(char_names),
    )

    n_anchored = 0
    for idx, name in resolved.items():
        if name == "Unknown":
            continue
        # Preceding dialogue is always the tag's speech; the following one only
        # when the tag ends with continuation punctuation (see
        # text_utils.extract_attribution_anchors for the rationale).
        adjacent = [idx - 1]
        if segments[idx].get("text", "").strip().endswith((",", ";", ":")):
            adjacent.append(idx + 1)
        for adj in adjacent:
            if 0 <= adj < len(segments) and segments[adj]["type"] == "dialogue":
                if not segments[adj].get("speaker"):
                    segments[adj]["speaker"] = name
                    n_anchored += 1

    return n_anchored
