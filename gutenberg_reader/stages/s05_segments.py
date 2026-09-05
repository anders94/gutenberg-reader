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
    atomic_write_text,
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

# Bumped when a cached chapter's segments stop being usable as they are.
# 2: segments carry start/end offsets into a reading_text written beside them.
SEGMENT_FORMAT = 2


def run(
    config: Config,
    client: LLMRouter,
    chapter_paths: dict[int, Path],
    chapter_nums: list[int] | None = None,
    chapter_titles: dict[int, str] | None = None,
    chapter_bounds: dict[int, tuple[int, int]] | None = None,
    quote_pair: tuple[str, str] | None = None,
    narrator_name: str = "",
) -> tuple[dict[int, tuple[ProcessedChapter, CriticReport | None]], list[CharacterInfo]]:
    """Process chapters in reading order.

    Returns ({chapter_num: (accepted_chapter, critic_report_or_None)}, roster).
    """
    stage_dir = config.stage_dir(5)
    nums = chapter_nums if chapter_nums is not None else sorted(chapter_paths.keys())
    critic_on = config.critic

    def fingerprint(n: int) -> dict | None:
        bounds = (chapter_bounds or {}).get(n)
        if bounds is None:
            return None
        return {"start_line": bounds[0], "end_line": bounds[1],
                "format": SEGMENT_FORMAT}

    roster: list[CharacterInfo] = []
    protected: set[str] = set()  # lowercase anchor-established names

    # A first-person narrator is a character like any other, and their own speech
    # needs a label the enum can carry. Seeded before chapter one because
    # discovery only finds names the text states, and a memoir rarely names its
    # own author: Augustine speaks 26 times in the Confessions and is mentioned
    # by name never.
    if narrator_name:
        roster = [CharacterInfo(name=narrator_name)]
        protected = {narrator_name.lower()}
        console.print(f"  [dim]narrator seeded into the roster: {narrator_name}[/dim]")
    accepted: dict[int, tuple[ProcessedChapter, CriticReport | None]] = {}

    for num in nums:
        out_path = chapter_file(stage_dir, num)

        cached = _load_cached(out_path, config, fingerprint(num))
        if cached is not None:
            chapter, roster, protected = cached
            if config.verbose:
                console.print(f"[dim]Stage 05: chapter {num:02d} already complete[/dim]")
            report = None
            if critic_on:
                chapter, report, roster, protected = _run_critic(
                    config, client, chapter, roster, protected, narrator_name
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
            quote_pair=quote_pair,
            narrator_name=narrator_name,
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
                config, client, processed, roster, protected, narrator_name
            )

        # Snapshot after the critic so chapter N+1 resumes from the exact
        # roster state it would see live.
        data = processed.to_dict()
        data["roster_after"] = [c.to_dict() for c in roster]
        data["anchor_names"] = sorted(protected)
        data["source"] = fingerprint(num)
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
    narrator_name: str = "",
) -> tuple[ProcessedChapter, CriticReport, list[CharacterInfo], set[str]]:
    """Critique one chapter and apply its roster objections (forward-only)."""
    new_names = [c.name for c in chapter.discovered_characters]
    final_chapter, report, issues = s06_critic.run_chapter(
        config, client, chapter, roster, new_names, narrator_name=narrator_name
    )
    # Idempotent: on a resumed chapter whose snapshot already reflects these
    # issues, the entries are gone and every issue is a no-op.
    roster, applied = s06_critic.apply_roster_issues(roster, issues, protected)
    if applied and config.verbose:
        for line in applied:
            console.print(f"  [dim]roster: {line}[/dim]")

    if report.needs_reprocessing:
        final_chapter, report = _reattribute_and_recheck(
            config, client, final_chapter, report, roster, protected, narrator_name
        )
    return final_chapter, report, roster, protected


def _reattribute_and_recheck(
    config: Config,
    client: LLMRouter,
    chapter: ProcessedChapter,
    report: CriticReport,
    roster: list[CharacterInfo],
    protected: set[str],
    narrator_name: str = "",
) -> tuple[ProcessedChapter, CriticReport]:
    """Give a chapter the critic was unhappy with exactly one more pass.

    needs_reprocessing was set and never read, so a chapter the critic scored
    0.6 shipped identically to one it scored 1.0. What gets redone is the
    low-confidence part — the segments the critic corrected, and the ones still
    Unknown — on the validator model, followed by a single re-critique.

    Deliberately capped at one attempt. A loop here is a way to spend a whole
    night on the chapter the model happens to disagree with itself about.
    """
    segments = [s.to_dict() for s in chapter.segments]
    retry = {
        i for i, seg in enumerate(segments)
        if seg.get("type") == "dialogue"
        and seg.get("notes") != "citation"
        and (seg.get("speaker") in (None, "Unknown"))
    }
    corrected = {
        int(line.split()[1].rstrip(":"))
        for line in report.attribution_issues
        if line.startswith("segment ") and line.split()[1].rstrip(":").isdigit()
    }
    retry |= {i for i in corrected if 0 <= i < len(segments)}
    if not retry:
        return chapter, report

    console.print(
        f"  [yellow]chapter {chapter.chapter_number}: quality "
        f"{report.overall_quality:.2f}"
        + (f", {len(report.unreviewed_windows)} window(s) unreviewed"
           if report.unreviewed_windows else "")
        + f" — re-attributing {len(retry)} segment(s)[/yellow]"
    )

    char_names = [c.name for c in roster]
    # The narrator reaches dialogue only through a first-person tag, never
    # through the free attribution passes. Offered in the enum, they become the
    # sink for everything unattributable: on PG 3296 "Augustine" collected 105
    # lines of which roughly seventy were a personified abstraction, a quoted
    # term, or his mother speaking. A tag is evidence; being plausible is not.
    attributable = text_utils.attributable_names(char_names, narrator_name)
    answers = _llm_window_pass(
        segments, retry, config, client,
        system_msg=prompts.verify_attribution_system(attributable),
        user_fn=prompts.verify_attribution_user,
        schema=schemas.attribution_schema(attributable),
        model=config.validation_model,
    )
    for idx, speaker in answers.items():
        segments[idx]["speaker"] = speaker

    reworked = ProcessedChapter(
        chapter_number=chapter.chapter_number,
        chapter_title=chapter.chapter_title,
        segments=[Segment.from_dict(s) for s in segments],
        discovered_characters=chapter.discovered_characters,
        word_count=chapter.word_count,
    )
    # One re-critique, and the roster is already settled, so no new names.
    rechecked, second, _ = s06_critic.run_chapter(
        config, client, reworked, roster, [], force=True,
        narrator_name=narrator_name,
    )
    if second.overall_quality < report.overall_quality:
        # The second opinion is worse than the first; keep what we had rather
        # than churn the chapter toward whichever pass was luckier.
        return chapter, report
    return rechecked, second


def _load_cached(
    out_path: Path,
    config: Config,
    fingerprint: dict | None = None,
) -> tuple[ProcessedChapter, list[CharacterInfo], set[str]] | None:
    """Load a chapter's cache entry, or None if it cannot be trusted as it is.

    Returns (chapter, roster_after, anchor_names). Rejected when:

    - there is no roster_after snapshot. Those files predate the rolling-roster
      design, and without the snapshot the next chapter's inputs are lost.
    - the fingerprint differs. A cache entry is keyed by chapter number, and the
      number means nothing on its own: re-running PG 6400 after its structure was
      fixed would otherwise load "chapter 2" from the broken 3-chapter split — a
      173,461-word span — as chapter 2 of the corrected twenty. Nothing would say
      so. The fingerprint carries the chapter's line range and the segment
      format, so a changed structure or an older format re-runs instead.
    """
    if not stage_complete(out_path):
        return None
    if config.force_stage is not None and config.force_stage <= 5:
        return None
    data = read_json(out_path)
    if "roster_after" not in data:
        return None
    if fingerprint is not None and data.get("source") != fingerprint:
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
    quote_pair: tuple[str, str] | None = None,
    narrator_name: str = "",
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

    # Tier 0: deterministic segmentation. Segments carry offsets into
    # reading_text and their text is a slice of it, so coverage is an exact
    # property rather than a reconstruction that has to be diffed.
    reading_text, segments = segmenter.segment_text(chapter_text, quote_pair)

    ok, issues = text_utils.verify_reading_text(chapter_text, reading_text)
    assert ok, f"chapter {chapter_num}: normalisation changed the text — {issues}"
    ok, issues = text_utils.verify_span_coverage(reading_text, segments)
    assert ok, f"chapter {chapter_num}: segments do not tile the chapter — {issues}"

    # Tier 0b: quotation marks mark speech and several things that are not.
    # Settle what the sentence says plainly, ask about the rest, and let a
    # verdict remove a boundary rather than rewrite any text.
    settled, ask = segmenter.classify_spans_deterministically(segments, reading_text)
    if config.span_review and ask:
        settled.update(_review_spans(segments, reading_text, ask, config, client))
    demoted = sum(1 for v in settled.values() if v in ("term", "title"))
    if demoted:
        segments = segmenter.apply_span_labels(segments, reading_text, settled)
        ok, issues = text_utils.verify_span_coverage(reading_text, segments)
        assert ok, f"chapter {chapter_num}: span review broke the tiling — {issues}"
        if config.verbose:
            console.print(
                f"  [dim]span review: {demoted} quoted span(s) were terms or "
                f"titles, not speech ({len(ask)} asked)[/dim]"
            )

    # Tier 1: deterministic attribution anchors ("said Mr. Bennet" adjacent to dialogue)
    anchors = text_utils.extract_attribution_anchors(segments, roster)
    # And the narrator's own, which no named tag can reach: a first-person
    # narrator is never named by a tag in their own narration.
    anchors.update(text_utils.extract_first_person_anchors(segments, narrator_name))
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
    # The narrator reaches dialogue only through a first-person tag, never
    # through the free attribution passes below.
    attributable = text_utils.attributable_names(char_names, narrator_name)
    anchor_names = {n.lower() for n in anchors.values()}

    # Tier 1b: LLM-resolve nameless attribution tags ("said his lady", "returned she")
    # to character names — a much easier task than free attribution — then anchor
    # the adjacent dialogue exactly as named tags do.
    n_tag = _resolve_nameless_tags(segments, roster, attributable, config, client)

    # Pass A (opportunistic): best-guess LLM attribution for unanchored dialogue
    # Citations are quoted text with no speaker to find — an oracle, an
    # inscription, a line of Homer the historian is citing. Running them through
    # attribution only ever yields "Unknown", and worse, tempts a pass into
    # pinning them on whoever is nearest.
    unresolved = {
        i for i, s in enumerate(segments)
        if s["type"] == "dialogue" and not s.get("speaker")
        and s.get("notes") != "citation"
    }
    proposed = _llm_window_pass(
        segments, unresolved, config, client,
        system_msg=prompts.attribution_system(attributable),
        user_fn=prompts.attribution_user,
        schema=schemas.attribution_schema(attributable),
    )
    for idx, speaker in proposed.items():
        segments[idx]["speaker"] = speaker

    # Pass B (critical): independently re-derive every non-anchored speaker.
    # Runs on the validator model — pointing --validator at a larger model
    # (it defaults to the processing model) buys accuracy exactly where it
    # matters, and decorrelates the two passes' errors even at equal size.
    verified = _llm_window_pass(
        segments, unresolved, config, client,
        system_msg=prompts.verify_attribution_system(attributable),
        user_fn=prompts.verify_attribution_user,
        schema=schemas.attribution_schema(attributable),
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
            system_msg=prompts.tiebreak_system(attributable),
            user_fn=lambda w, s, f, c: prompts.tiebreak_user(w, s, f, c, disputes),
            schema=schemas.attribution_schema(attributable),
            model=config.validation_model,
        )
        for idx in disputes:
            # Two passes disagreed and the arbiter didn't rule: Unknown is
            # more honest than either guess.
            segments[idx]["speaker"] = broken.get(idx, "Unknown")

    n_unknown = n_cited = 0
    for s in segments:
        if s["type"] != "dialogue" or s.get("speaker"):
            n_cited += s.get("notes") == "citation"
            continue
        if s.get("notes") == "citation":
            s["speaker"] = text_utils.CITATION_SPEAKER
            n_cited += 1
        else:
            s["speaker"] = "Unknown"
            n_unknown += 1

    if config.verbose:
        n_dialogue = sum(1 for s in segments if s["type"] == "dialogue")
        console.print(
            f"  [dim]{len(segments)} segments, {n_dialogue} dialogue: "
            f"{len(anchors)} anchored, {n_tag} tag-resolved, {len(proposed)} proposed, "
            f"{len(disputes)} disputed, {n_cited} cited, {n_unknown} unknown[/dim]"
        )

    # The canonical string every offset indexes, written once so the spans in
    # the output point at something inspectable rather than at a reconstruction.
    atomic_write_text(
        chapter_file(config.stage_dir(5), chapter_num, ".text"), reading_text)

    processed = ProcessedChapter(
        chapter_number=chapter_num,
        chapter_title=chapter_title,
        segments=[Segment.from_dict(s) for s in segments],
        word_count=text_utils.word_count(chapter_text),
    )
    return processed, roster, anchor_names


# Marked spans per request. Small: the model is judging one short span at a
# time and the surrounding sentence is what it needs, not a whole chapter.
SPAN_REVIEW_BATCH = 40


SPAN_CONTEXT_CHARS = 300


def _window_around_span(passage: str, ordinal: int) -> str:
    """The span, plus context either side, rather than the paragraph's opening.

    The passage used to be cut to its first 600 characters, which works for a
    novel and fails for anything with long paragraphs: on PG 3296 that removed
    the span itself from 49% of the questions and on PG 6400 from 59%. The model
    was shown a paragraph with nothing marked in it and asked what the marked
    span was, so those answers were guesses.
    """
    open_tag, close_tag = f"\u27e6{ordinal}\u27e7", f"\u27e6/{ordinal}\u27e7"
    a = passage.find(open_tag)
    b = passage.find(close_tag)
    if a < 0 or b < 0:
        return passage[:2 * SPAN_CONTEXT_CHARS]
    b += len(close_tag)
    lo = max(0, a - SPAN_CONTEXT_CHARS)
    hi = min(len(passage), b + SPAN_CONTEXT_CHARS)
    return (("\u2026" if lo > 0 else "") + passage[lo:hi]
            + ("\u2026" if hi < len(passage) else ""))


def _render_span_passages(
    segments: list[dict], reading_text: str, ask: list[int]
) -> str:
    """One line per span: its paragraph with the span itself bracketed.

    The model sees the sentence around the span, which is the evidence, and
    answers with ordinals — it never repeats the text back.
    """
    by_para: dict[int, list[int]] = {}
    for i, seg in enumerate(segments):
        by_para.setdefault(seg.get("para", 0), []).append(i)

    lines = []
    for ordinal, idx in enumerate(ask):
        para = segments[idx].get("para", 0)
        parts = []
        for j in by_para.get(para, []):
            piece = reading_text[segments[j]["start"]:segments[j]["end"]]
            if j == idx:
                parts.append(f"\u27e6{ordinal}\u27e7{piece}\u27e6/{ordinal}\u27e7")
            else:
                parts.append(piece)
        lines.append(f"{ordinal}| {_window_around_span(' '.join(parts), ordinal)}")
    return "\n".join(lines)


def _review_spans(
    segments: list[dict],
    reading_text: str,
    ask: list[int],
    config: Config,
    client: LLMRouter,
) -> dict[int, str]:
    """Ask what the ambiguous quoted spans actually are.

    Quotation marks mark speech and several things that are not speech, and the
    segmenter cannot tell them apart. In PG 2131 most "dialogue" is scare-quoted
    terminology, which is why 12 of its 29 dialogue segments had no speaker to
    find. A verdict here removes a boundary rather than rewriting anything.
    """
    labels: dict[int, str] = {}
    for start in range(0, len(ask), SPAN_REVIEW_BATCH):
        batch = ask[start:start + SPAN_REVIEW_BATCH]
        passages = _render_span_passages(segments, reading_text, batch)
        data = call_json_with_retries(
            client, config.validation_model,
            [{"role": "system", "content": prompts.span_type_system()},
             {"role": "user", "content": prompts.span_type_user(passages, len(batch))}],
            schema=schemas.span_type_schema(len(batch)),
            retries=config.max_retries, what="span review", console=console,
            # Greedy, like the structure pass and the critic. Left sampling, the
            # same chapter demoted a different set of spans on each run, which
            # made two runs of a change impossible to compare.
            temperature=0.0,
        )
        if data is None:
            continue          # these spans keep the segmenter's verdict
        for item in data.get("spans", []):
            o = item.get("ordinal")
            if isinstance(o, int) and 0 <= o < len(batch):
                labels[batch[o]] = item["label"]
    return labels


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
