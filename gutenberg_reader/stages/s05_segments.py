"""Stage 05 — Segment chapters and attribute dialogue speakers.

Segmentation into narration/dialogue is fully deterministic (quotation marks
delimit dialogue — see segmenter.py), so text coverage is guaranteed by
construction. Speaker attribution favors accuracy over cost:

  1. Deterministic anchors: "said Mr. Bennet" narration adjacent to dialogue.
     These are the author naming the speaker — kept as hard evidence.
  2. Everything else goes through three LLM passes (guided decoding constrains
     speakers to the known character list):
       a. opportunistic — best-guess attribution with full context
       b. critical — independently re-derives every non-anchored speaker,
          blind to pass (a)'s answers so its errors don't correlate
       c. tie-breaker — sees both candidates and resolves disagreements

No alternation or scene-cast heuristics: real texts break the "speakers
alternate" and "one speaker per paragraph" conventions often enough (e.g.
PG 2641 ch. 1 puts two speakers in one paragraph) that guessing from
typography produces confident errors the LLM passes would have caught.
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
from gutenberg_reader.models import CharacterInfo, ProcessedChapter, Segment
from gutenberg_reader.llm import LLMClient, LLMError
from gutenberg_reader import prompts, schemas, segmenter, text_utils

console = Console()

# Segments from the previous window included (read-only) for continuity
CONTEXT_SEGMENTS = 8


def run(
    config: Config,
    client: LLMClient,
    chapter_paths: dict[int, Path],
    characters: list[CharacterInfo],
    chapter_nums: list[int] | None = None,
) -> dict[int, ProcessedChapter]:
    """Segment all (or specified) chapters. Returns {chapter_num: ProcessedChapter}."""
    stage_dir = config.stage_dir(5)
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

        processed = _segment_chapter(num, chapter_text, characters, config, client)
        atomic_write_json(out_path, processed.to_dict())
        results[num] = processed

    return results


def _segment_chapter(
    chapter_num: int,
    chapter_text: str,
    characters: list[CharacterInfo],
    config: Config,
    client: LLMClient,
) -> ProcessedChapter:
    lines = chapter_text.splitlines()
    chapter_title = next((l.strip() for l in lines if l.strip()), f"Chapter {chapter_num}")
    char_names = [c.name for c in characters]

    # Tier 0: deterministic segmentation
    segments = segmenter.segment_text(chapter_text)

    ok, issues = text_utils.verify_segment_coverage(chapter_text, segments)
    if not ok:
        # Should be impossible; if it happens, the segmenter has a bug worth surfacing
        console.print(
            f"  [yellow]Coverage warning in chapter {chapter_num}: {issues[:2]}[/yellow]"
        )

    # Tier 1: deterministic attribution anchors ("said Mr. Bennet" adjacent to dialogue)
    anchors = text_utils.extract_attribution_anchors(segments, characters)
    for idx, name in anchors.items():
        segments[idx]["speaker"] = name

    # Tags can name speakers the discovery stage missed ("said Lydia,") —
    # record them so alternation, the LLM enum, and the final JSON know them.
    extra_names = sorted({n for n in anchors.values() if n not in char_names})
    discovered = [
        CharacterInfo(name=n, first_appearance_chapter=chapter_num) for n in extra_names
    ]
    characters = characters + discovered
    char_names = char_names + extra_names

    # Tier 1b: LLM-resolve nameless attribution tags ("said his lady", "returned she")
    # to character names — a much easier task than free attribution — then anchor
    # the adjacent dialogue exactly as named tags do.
    n_tag = _resolve_nameless_tags(segments, characters, char_names, config, client)

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

    return ProcessedChapter(
        chapter_number=chapter_num,
        chapter_title=chapter_title,
        segments=[Segment.from_dict(s) for s in segments],
        discovered_characters=discovered,
        word_count=text_utils.word_count(chapter_text),
    )


def _llm_window_pass(
    segments: list[dict],
    flagged_all: set[int],
    config: Config,
    client: LLMClient,
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

        for attempt in range(config.max_retries):
            try:
                data = client.chat_json(model, messages, schema=schema)
            except LLMError as e:
                console.print(f"  [red]LLM pass error (attempt {attempt+1}): {e}[/red]")
                continue
            for a in data.get("attributions", []):
                idx = a.get("index")
                if idx in flagged and a.get("speaker"):
                    results[idx] = a["speaker"]
            break

    return results


def _resolve_nameless_tags(
    segments: list[dict],
    characters: list[CharacterInfo],
    char_names: list[str],
    config: Config,
    client: LLMClient,
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
