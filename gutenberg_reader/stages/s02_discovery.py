"""Stage 02 — Discover metadata and chapter structure."""

from __future__ import annotations
from pathlib import Path

from rich.console import Console

from gutenberg_reader.cache import atomic_write_json, read_text, stage_complete
from gutenberg_reader.config import Config
from gutenberg_reader.models import (
    SCHEMA_VERSION, BookMetadata, ChapterInfo, DiscoveryResult,
)
from gutenberg_reader.llm import LLMError, LLMRouter, call_json_with_retries
from gutenberg_reader import prompts, schemas
from gutenberg_reader import text_utils
from gutenberg_reader import candidates, segmenter, structure_checks

console = Console()


def run(config: Config, client: LLMRouter) -> DiscoveryResult:
    """Run discovery and return DiscoveryResult."""
    stage_dir = config.stage_dir(2)
    out_path = stage_dir / "discovery.json"

    if stage_complete(out_path) and (config.force_stage is None or config.force_stage > 2):
        from gutenberg_reader.cache import read_json
        cached = DiscoveryResult.from_dict(read_json(out_path))
        # A discovery.json written by an older detector stays on disk and feeds
        # every later stage its verdict — PG 1727 and 2641 both have cached files
        # that disagree with the detector supposedly behind them. Recompute
        # instead of trusting them.
        wanted = _detector_id(config)
        if cached.schema_version != SCHEMA_VERSION or cached.detector != wanted:
            console.print(
                f"[cyan]Stage 02:[/cyan] cached structure is {cached.detector} "
                f"v{cached.schema_version}, want {wanted} v{SCHEMA_VERSION} — recomputing"
            )
        else:
            if config.verbose:
                console.print(f"[dim]Stage 02: already complete ({out_path})[/dim]")
        # Re-check the cached structure, not just freshly detected structure: a
        # discovery.json written by an older detector stays on disk and silently
        # feeds every later stage a table of contents (see PG 2701). The checks
        # are pure, so running them on the cached path costs nothing.
            body_lines = _body_lines(config)
            _enforce_structure(cached.chapters, body_lines, cached.body_start_line, config)
            return cached

    raw_path = config.stage_dir(1) / "book.txt"
    raw_text = read_text(raw_path)
    lines = raw_text.splitlines()

    # Find body bounds
    body_start, body_end = text_utils.find_body_bounds(lines)

    preamble = "\n".join(lines[:body_start])
    raw_meta = text_utils.extract_preamble_metadata(preamble)

    metadata = BookMetadata(
        title=raw_meta.get("title", ""),
        author=raw_meta.get("author", ""),
        language=raw_meta.get("language", "en"),
        gutenberg_id=config.book_id,
        release_date=raw_meta.get("release_date", ""),
        credits=raw_meta.get("credits", ""),
    )

    if config.verbose:
        console.print(f"[cyan]Stage 02:[/cyan] Metadata: {metadata.title} by {metadata.author}")

    body_lines = lines[body_start:body_end]
    verdict: dict = {}

    if config.structure_detector == "llm":
        chapter_infos, verdict = _llm_structure(body_lines, config, client)
        chapter_infos = _shift_to_absolute(chapter_infos, body_start)
        _cross_check_against_regex(chapter_infos, body_lines, body_start, config)
    else:
        chapter_infos = _regex_structure(body_lines, body_start, config, client)

    if config.verbose:
        console.print(f"[cyan]Stage 02:[/cyan] Found {len(chapter_infos)} chapters")

    _enforce_structure(chapter_infos, body_lines, body_start, config)

    # Detected over the whole body: a single chapter can be too short, or too
    # apostrophe-heavy, to call the edition's convention correctly.
    quote_pair = segmenter.detect_quote_pair("\n".join(body_lines)) or ("", "")
    person, narrator = _detect_narration(body_lines, metadata, config, client)

    result = DiscoveryResult(
        metadata=metadata,
        chapters=chapter_infos,
        body_start_line=body_start,
        body_end_line=body_end,
        detector=_detector_id(config),
        work_type=verdict.get("work_type", ""),
        has_chapter_structure=verdict.get("has_chapter_structure", True),
        quote_open=quote_pair[0],
        quote_close=quote_pair[1],
        narration_person=person,
        narrator_name=narrator,
    )

    atomic_write_json(out_path, result.to_dict())
    return result


# A heading with nothing under it is a part title, not a chapter: PG 6400 prints
# "LIVES OF THE POETS." directly above "THE LIFE OF TERENCE." as the heading for
# a group of lives. Left alone it becomes a five-word chapter, which downstream
# is a two-second audio track.
MIN_STANDALONE_CHAPTER_WORDS = 20


def _absorb_part_titles(raw: list[dict], body_lines: list[str]) -> list[dict]:
    """Fold a heading-only chapter into the chapter that follows it.

    The part title stays in the text — it is a line or two — but stops being a
    boundary. The following chapter keeps its own title and simply starts higher
    up. The last chapter has nothing to fold into, so it is left alone and the
    degenerate check speaks for it instead.
    """
    if len(raw) < 2:
        return raw

    out: list[dict] = []
    absorbed: list[str] = []
    pending_start: int | None = None
    for i, ch in enumerate(raw):
        end = (raw[i + 1]["start_line"] - 2) if i + 1 < len(raw) else len(body_lines) - 1
        words = text_utils.word_count(
            "\n".join(body_lines[ch["start_line"] - 1:end + 1]))
        if words < MIN_STANDALONE_CHAPTER_WORDS and i + 1 < len(raw):
            # Remember where it began; the next real chapter starts here.
            pending_start = pending_start if pending_start is not None else ch["start_line"]
            absorbed.append(ch["title"])
            continue
        if pending_start is not None:
            ch = dict(ch, start_line=pending_start)
            pending_start = None
        out.append(ch)

    if absorbed:
        shown = ", ".join(repr(t) for t in absorbed[:3])
        more = f" and {len(absorbed) - 3} more" if len(absorbed) > 3 else ""
        console.print(
            f"[cyan]Stage 02:[/cyan] {len(absorbed)} heading(s) hold no text — "
            f"treating as part titles, not chapters: {shown}{more}"
        )
    return [dict(ch, number=i + 1) for i, ch in enumerate(out)]


# Enough opening prose to tell "I" from "he", without paying for a whole chapter.
NARRATION_SAMPLE_WORDS = 400


def _detect_narration(
    body_lines: list[str],
    metadata: BookMetadata,
    config: Config,
    client: LLMRouter,
) -> tuple[str, str]:
    """How the book is told, and the name to file its narrator under.

    Returns (person, narrator_name); narrator_name is "" unless someone tells the
    story in their own voice. This exists because the speaker enum is the roster:
    a first-person narrator missing from it cannot be attributed at all, since
    guided decoding has no token for them. Jane Eyre, Ishmael and Dr. Watson are
    all named by other characters and so get discovered; Augustine never names
    himself in his own Confessions, and 18 of his 26 first-person lines came back
    Unknown.
    """
    opening = " ".join(" ".join(body_lines).split()[:NARRATION_SAMPLE_WORDS])
    data = call_json_with_retries(
        client, config.structure_model,
        [{"role": "system", "content": prompts.narration_system()},
         {"role": "user", "content": prompts.narration_user(
             metadata.title, metadata.author, opening)}],
        schema=schemas.narration_schema(), retries=config.max_retries,
        what="narration analysis", console=console,
        temperature=STRUCTURE_TEMPERATURE,
    )
    if data is None:
        return "", ""

    person = data.get("person", "")
    name = (data.get("narrator_name") or "").strip()
    # A role is not a name. The model is told this, and told plainly it still
    # answered "Unnamed narrator" in an earlier incarnation of this pipeline, so
    # the check is here rather than only in the prompt.
    if person != "first_person" or text_utils.is_reserved_character_name(name):
        name = ""
    if name:
        console.print(
            f"[cyan]Stage 02:[/cyan] narrated in the first person by {name!r} "
            f"(confidence {data.get('confidence', '?')})"
        )
    return person, name


def _detector_id(config: Config) -> str:
    return "llm-v1" if config.structure_detector == "llm" else "regex-v1"


def _shift_to_absolute(infos: list[ChapterInfo], body_start: int) -> list[ChapterInfo]:
    """_llm_structure works in body coordinates; the cache stores file lines."""
    for ci in infos:
        ci.start_line += body_start
        ci.end_line += body_start
    return infos


def _regex_structure(
    body_lines: list[str],
    body_start: int,
    config: Config,
    client: LLMRouter,
) -> list[ChapterInfo]:
    """The previous pattern-matching detector, kept for comparison and for runs
    with no model to hand. Every book that broke it added a rule here."""
    raw = text_utils.detect_chapters_regex(body_lines)
    raw = _maybe_prepend_chapter_one(raw, body_lines)
    if len(raw) < 2:
        if config.verbose:
            console.print(
                f"[yellow]Stage 02:[/yellow] regex found only {len(raw)} chapters, "
                "falling back to LLM discovery..."
            )
        raw = _llm_chapter_discovery(body_lines, config, client)
    raw = _drop_leading_front_matter(
        raw, body_lines, include_front_matter=config.include_front_matter)
    raw = _split_headless_body(raw, body_lines)
    return _build_chapter_infos(
        raw, body_lines, body_start, include_back_matter=config.include_back_matter)


def _cross_check_against_regex(
    infos: list[ChapterInfo],
    body_lines: list[str],
    body_start: int,
    config: Config,
) -> None:
    """Report where the old detector disagrees, without letting it decide.

    Pure string matching, so it costs nothing, and disagreement is the first
    place to look when a model regresses on a book that used to work. It is
    reported only — the regex is what this replaces.
    """
    try:
        regex_starts = {
            ch["start_line"] + body_start
            for ch in text_utils.detect_chapters_regex(body_lines)
        }
    except Exception:  # the old detector must never break the new one
        return

    llm_starts = {ci.start_line for ci in infos}
    only_regex = len(regex_starts - llm_starts)
    only_llm = len(llm_starts - regex_starts)
    if not (only_regex or only_llm):
        return
    console.print(
        f"[dim]Stage 02: regex detector differs — {len(regex_starts)} headings vs "
        f"{len(llm_starts)} chosen ({only_regex} only regex, {only_llm} only model)[/dim]"
    )


# One reprompt is worth it — a repair is a much easier question than the
# original, because it names the specific objection. Two is where a model that
# cannot do the book stops pretending it can.
MAX_STRUCTURE_REPAIRS = 2

# Greedy, for the same reason the critic is: a structure you cannot reproduce is
# one you cannot check against a golden. PG 6400 returned 19 chapters on one run
# and 20 on the next from identical candidates, which makes "did this book come
# out right?" unanswerable.
STRUCTURE_TEMPERATURE = 0.0

# Below this, whatever sits above the first heading is a title page, not a
# chapter the edition forgot to label.
MIN_UNLABELLED_CHAPTER_WORDS = 50


def _structure_to_raw(
    verdict: dict,
    cands: list[candidates.Candidate],
    body_lines: list[str],
    config: Config,
) -> list[dict]:
    """Turn ordinals into chapter dicts. Every string here comes from the book.

    The model never emits a title or a line number: it picks candidates, and the
    text is sliced from the candidate it picked. A hallucinated position is
    impossible because the ordinal is bounded by the schema.
    """
    by_ord = {c.ordinal: c for c in cands}
    keep = {"body"}
    if config.include_front_matter:
        keep.add("front")
    if config.include_back_matter:
        keep.add("back")

    chosen: list[tuple[int, str, str]] = []   # (line, title, kind)
    in_toc = 0
    for h in verdict.get("headings", []):
        c = by_ord.get(h.get("ordinal"))
        if c is None or h.get("kind") not in keep:
            continue
        # A pick inside a densely packed run is a contents entry whatever the
        # model called it. PG 2701 returned all 135 contents lines alongside the
        # 135 real headings; taken at face value the first "chapter" spans the
        # whole front matter and is named after a contents entry.
        if "toc-run" in c.flags:
            in_toc += 1
            continue
        chosen.append((c.line, c.text, h["kind"]))
    chosen.sort()
    if in_toc:
        console.print(
            f"[cyan]Stage 02:[/cyan] ignored {in_toc} heading(s) inside a contents "
            "listing — they are entries, not the body"
        )

    raw = [
        {"number": i + 1, "title": title, "start_line": line + 1,
         "start_marker": title, "kind": kind}
        for i, (line, title, kind) in enumerate(chosen)
    ]

    # The edition prints no heading over its opening chapter (PG 1342). Give the
    # text before the first heading a chapter of its own rather than losing it —
    # but only when it is really the story starting. Nearly every book has *some*
    # text up there: on PG 1260 it is Currer Bell's preface and on PG 2641 the
    # title page, and taking the model's word for it gave both a spurious extra
    # chapter one.
    #
    # _maybe_prepend_chapter_one already decides this correctly and is worth
    # keeping: it walks back only as far as the contents listing, then vetoes the
    # block if it carries a front-matter heading of its own, then requires it to
    # read as sentences rather than captions. That is three books' worth of
    # accumulated knowledge and none of it is guesswork the model should redo.
    if raw and verdict.get("body_starts_before_first_heading"):
        raw = _maybe_prepend_chapter_one(raw, body_lines)
    return raw


def _llm_structure(
    body_lines: list[str],
    config: Config,
    client: LLMRouter,
) -> tuple[list[ChapterInfo], dict]:
    """Classify the whole book's structure at once, then check and repair.

    Structure is a global property: whether a run of heading-shaped lines is a
    contents listing or the body, and whether a lone all-caps line is a chapter
    or an inscription, are only answerable by seeing every candidate together.
    So the book is condensed to its candidate blocks and judged in one pass
    rather than streamed past the model in chunks, which is what produced the
    PG 2701, 37106 and 1661 defects in the first place.
    """
    cands = candidates.extract(body_lines)
    rendered = candidates.render(cands)
    schema = schemas.structure_schema(len(cands))
    system = prompts.structure_system()

    if config.verbose:
        console.print(
            f"[cyan]Stage 02:[/cyan] {len(cands)} candidate blocks "
            f"(~{int(len(rendered.split()) * 1.4):,} tokens) → {config.structure_model}"
        )

    user = prompts.structure_user(rendered, len(cands))
    verdict: dict = {}
    infos: list[ChapterInfo] = []
    findings: list = []

    for attempt in range(MAX_STRUCTURE_REPAIRS + 1):
        data = call_json_with_retries(
            client, config.structure_model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            schema=schema, retries=config.max_retries,
            what="structure analysis", console=console,
            temperature=STRUCTURE_TEMPERATURE,
        )
        if data is None:
            raise LLMError("structure analysis exhausted its retries")
        verdict = data

        raw = _structure_to_raw(verdict, cands, body_lines, config)
        if not verdict.get("has_chapter_structure", True) or not raw:
            # A book with no divisions at all is a real answer, not a failure.
            # Give the body one chapter and let it be cut into parts.
            start = raw[0]["start_line"] if raw else 1
            raw = [{"number": 1, "title": raw[0]["title"] if raw else "",
                    "start_line": start, "start_marker": "", "kind": "body"}]
        raw = _absorb_part_titles(raw, body_lines)
        raw = _split_headless_body(raw, body_lines)

        infos = _build_chapter_infos(
            raw, body_lines, 0, include_back_matter=config.include_back_matter)
        findings = structure_checks.check(infos, cands, 0)
        fails = [f for f in findings if f.severity == "fail"]
        if not fails:
            return infos, verdict

        if attempt == MAX_STRUCTURE_REPAIRS:
            break
        console.print(
            f"[yellow]Stage 02:[/yellow] {len(fails)} structure problem(s) — "
            f"re-asking ({attempt + 1}/{MAX_STRUCTURE_REPAIRS})"
        )
        user = prompts.structure_repair_user(rendered, [f.message for f in fails])

    return infos, verdict


def _build_chapter_infos(
    raw_chapters: list[dict],
    body_lines: list[str],
    body_start: int,
    include_back_matter: bool = False,
) -> list[ChapterInfo]:
    """Convert raw chapter dicts into ChapterInfo with word counts and absolute line numbers."""
    infos: list[ChapterInfo] = []
    n = len(raw_chapters)
    seen_numbers: set[int] = set()
    back_matter_start: int | None = None  # rel index of the first back-matter heading

    for i, ch in enumerate(raw_chapters):
        # start_line in raw_chapters is 1-indexed relative to body_lines
        rel_start = ch["start_line"] - 1  # 0-based in body_lines
        abs_start = body_start + rel_start  # 0-based in full file

        # end: next chapter start - 1, or end of body
        if i + 1 < n:
            rel_end = raw_chapters[i + 1]["start_line"] - 2  # line before next chapter header
        else:
            # The final chapter would otherwise run to end-of-body, swallowing
            # any footnote appendix, index, or errata the edition carries.
            rel_end = len(body_lines) - 1
            back_matter_start = _find_back_matter_heading(body_lines, rel_start + 1, rel_end)
            if back_matter_start is not None:
                trimmed = rel_end - back_matter_start + 1
                console.print(
                    f"[cyan]Stage 02:[/cyan] final chapter ends at back-matter heading "
                    f"{body_lines[back_matter_start].strip()!r} — "
                    f"{'keeping' if include_back_matter else 'trimming'} {trimmed:,} trailing lines"
                )
                rel_end = back_matter_start - 1

        if rel_end < rel_start:
            console.print(
                f"[yellow]Stage 02: warning —[/yellow] chapter {ch['number']} "
                f"({ch['title']!r}) spans no lines; its boundaries are out of order. "
                "Check the discovered chapter list."
            )

        abs_end = body_start + rel_end

        # Extract chapter text for word count
        chapter_text = "\n".join(body_lines[rel_start:rel_end + 1])
        chapter_text = text_utils.strip_illustration_blocks(chapter_text)
        wc = text_utils.word_count(chapter_text)

        # Chapter numbers key the per-chapter cache files in stages 03/05 and the
        # accepted-chapter map in stage 07, so a duplicate silently discards a
        # chapter. Fall back to position when a source hands us a repeat.
        number = ch["number"]
        if number in seen_numbers:
            number = max(seen_numbers) + 1
        seen_numbers.add(number)

        infos.append(ChapterInfo(
            number=number,
            title=ch["title"],
            start_line=abs_start + 1,  # 1-indexed in full file
            end_line=abs_end + 1,
            word_count=wc,
            start_marker=ch.get("start_marker", ch["title"]),
            kind=ch.get("kind", "body"),
        ))

    if include_back_matter and back_matter_start is not None and infos:
        rel_end = len(body_lines) - 1
        text = "\n".join(body_lines[back_matter_start:rel_end + 1])
        text = text_utils.strip_illustration_blocks(text)
        infos.append(ChapterInfo(
            number=max(seen_numbers) + 1,
            title=body_lines[back_matter_start].strip(),
            start_line=body_start + back_matter_start + 1,
            end_line=body_start + rel_end + 1,
            word_count=text_utils.word_count(text),
            start_marker=body_lines[back_matter_start].strip(),
            kind="back",
        ))

    return infos


def _find_back_matter_heading(body_lines: list[str], start: int, end: int) -> int | None:
    """Return the 0-based index of the first back-matter heading in [start, end], if any."""
    for j in range(start, end + 1):
        if text_utils.BACK_MATTER_RE.match(body_lines[j].strip()):
            return j
    return None


def _body_lines(config: Config) -> list[str]:
    """The book body, for checks that need to see the text behind a decision."""
    lines = read_text(config.stage_dir(1) / "book.txt").splitlines()
    start, end = text_utils.find_body_bounds(lines)
    return lines[start:end]


def _enforce_structure(
    chapters: list[ChapterInfo],
    body_lines: list[str],
    body_start: int,
    config: Config,
) -> None:
    """Run the structure checks and refuse to continue on a failure.

    Warnings used to print and be ignored, which is how PG 6400 shipped with
    173,461 words in one chapter and PG 1727 shipped 87 minutes of footnotes to
    the synthesiser. A re-run is cheap; those were not.
    """
    findings = structure_checks.check(
        chapters, candidates.extract(body_lines), body_start
    )
    for f in findings:
        colour = "red" if f.severity == "fail" else "yellow"
        console.print(f"[{colour}]Stage 02 {f.severity}:[/{colour}] {f.message}")

    fails = [f for f in findings if f.severity == "fail"]
    if not fails:
        return
    if config.accept_structure_warnings:
        console.print(
            f"[yellow]Stage 02:[/yellow] continuing past {len(fails)} structure "
            "failure(s) — --accept-structure-warnings was given"
        )
        return
    console.print(
        f"[red]Stage 02: refusing to continue —[/red] {len(fails)} structure "
        "check(s) failed. Fix detection, or pass --accept-structure-warnings to "
        "ship this structure anyway."
    )
    raise SystemExit(2)


# A title page and the editor's note under it run to a few hundred words; a real
# opening chapter runs to thousands. Past this, a leading block is body text
# whatever its headings look like.
MAX_LEADING_FRONT_MATTER_WORDS = 2000
# Below this a block holds no prose at all — just its own heading and a byline.
MIN_LEADING_BODY_WORDS = 20


def _leading_front_matter_reason(
    raw_chapters: list[dict],
    body_lines: list[str],
) -> tuple[str, int] | None:
    """Why the first of raw_chapters is the edition's matter, and its word count.

    Size alone does not identify it; plenty of books open with a short chapter.
    What does is a *small leading* block that repeats the next chapter's title
    verbatim, carries an apparatus heading (NOTE, PREFACE, ...) of its own, or
    holds no prose at all.
    """
    if len(raw_chapters) < 2:
        return None

    first, second = raw_chapters[0], raw_chapters[1]
    lead_start = first["start_line"] - 1
    lead_end = second["start_line"] - 1  # exclusive: the next heading's own line
    lead_lines = body_lines[lead_start:lead_end]

    lead_words = text_utils.word_count(
        text_utils.strip_illustration_blocks("\n".join(lead_lines))
    )
    if lead_words == 0 or lead_words > MAX_LEADING_FRONT_MATTER_WORDS:
        return None

    # The rest of the book has to dwarf it, or this is simply a short chapter one.
    rest_words = text_utils.word_count(
        text_utils.strip_illustration_blocks("\n".join(body_lines[lead_end:]))
    )
    if rest_words < 4 * lead_words:
        return None

    if (first.get("title", "").strip().casefold()
            == second.get("title", "").strip().casefold()):
        return "repeats the next chapter's title", lead_words
    if any(text_utils.FRONT_MATTER_RE.match(line.strip())
           or text_utils.BACK_MATTER_RE.match(line.strip())
           for line in lead_lines):
        return "carries an apparatus heading", lead_words
    if lead_words < MIN_LEADING_BODY_WORDS:
        return "holds no prose of its own", lead_words
    return None


def _drop_leading_front_matter(
    raw_chapters: list[dict],
    body_lines: list[str],
    include_front_matter: bool = False,
) -> list[dict]:
    """Drop leading "chapters" that are really the edition's title page.

    A source with no chapter headings leaves the detector only the book's own
    front page to match on. PG 2131 (Herodotus, "An Account of Egypt") splits
    there, and not once: the title page with the editor's NOTE under it, the
    title repeated over the text, and the subtitle line below that all read as
    headings, so the whole book lands in a chapter sitting behind two stubs.
    Each is stripped in turn — removing one exposes the next.
    """
    kept = list(raw_chapters)
    dropped: list[dict] = []

    while (found := _leading_front_matter_reason(kept, body_lines)) is not None:
        reason, lead_words = found
        head = kept.pop(0)
        dropped.append(dict(head, kind="front"))
        console.print(
            f"[cyan]Stage 02:[/cyan] {head.get('title', '')!r} ({lead_words:,} words) "
            f"is front matter — it {reason}"
        )

    if not dropped:
        return raw_chapters

    result = dropped + kept if include_front_matter else kept
    return [dict(ch, number=i + 1) for i, ch in enumerate(result)]


# A synthesized part aims for the length of an ordinary chapter: long enough to
# be worth its own track, short enough that one attribution pass can hold it.
# P&P runs ~2,000 words a chapter, the Odyssey ~4,500.
TARGET_PART_WORDS = 2500
# Below this a body is a long short story, not an unbroken book; leave it whole.
MIN_HEADLESS_SPLIT_WORDS = 3 * TARGET_PART_WORDS


def _paragraph_spans(
    body_lines: list[str], start: int, end: int
) -> list[tuple[int, int]]:
    """(first line index, word count) for each blank-line-separated paragraph."""
    spans: list[tuple[int, int]] = []
    i = start
    while i < end:
        if not body_lines[i].strip():
            i += 1
            continue
        para_start, words = i, 0
        while i < end and body_lines[i].strip():
            words += len(body_lines[i].split())
            i += 1
        spans.append((para_start, words))
    return spans


def _split_headless_body(
    raw_chapters: list[dict],
    body_lines: list[str],
    target_words: int = TARGET_PART_WORDS,
) -> list[dict]:
    """Split a body with no chapter structure into parts at paragraph boundaries.

    Some editions carry no chapter heading anywhere. PG 2131 prints Herodotus'
    Book II as one unbroken 36,916-word run — its 182 canonical sections are
    unnumbered in this translation — so there is genuinely nothing to detect and
    discovery correctly returns a single chapter. Every later stage then treats
    the whole book as one unit: one attribution pass over 37k words, a cast
    discovered from a single window (88 names for 2131), one enormous track.

    Only a body with *no* structure is split. A long chapter inside a book that
    has chapters is a long chapter — Moby-Dick's "Cetology" runs 3.6x the median
    and is genuinely one — so this never touches a multi-chapter result.
    """
    # Exactly one body chapter, running to the end. Front matter kept by
    # --include-front-matter sits ahead of it and is left alone.
    body_idx = [
        i for i, ch in enumerate(raw_chapters) if ch.get("kind", "body") == "body"
    ]
    if len(body_idx) != 1 or body_idx[0] != len(raw_chapters) - 1:
        return raw_chapters

    head = raw_chapters[:body_idx[0]]
    rel_start = raw_chapters[body_idx[0]]["start_line"] - 1
    spans = _paragraph_spans(body_lines, rel_start, len(body_lines))
    total = sum(words for _, words in spans)
    if total < MIN_HEADLESS_SPLIT_WORDS:
        return raw_chapters

    # Greedy fill: a part closes once it reaches the target, so every cut lands
    # on a paragraph boundary and the parts tile the body exactly.
    groups: list[list[int]] = []  # [first line index, words so far]
    for line_idx, words in spans:
        if not groups or groups[-1][1] >= target_words:
            groups.append([line_idx, words])
        else:
            groups[-1][1] += words

    # A runt tail is worse than one long part: fold it back.
    if len(groups) > 1 and groups[-1][1] < target_words // 2:
        tail = groups.pop()
        groups[-1][1] += tail[1]

    if len(groups) < 2:
        return raw_chapters

    # Keep the original opening line — the title sits above the first paragraph.
    groups[0][0] = rel_start

    console.print(
        f"[cyan]Stage 02:[/cyan] no chapter headings in {total:,} words — "
        f"splitting into {len(groups)} parts at paragraph boundaries"
    )
    parts = [
        {
            "number": 0,  # renumbered below, across any kept front matter
            "title": f"Part {i + 1}",
            "start_line": line_idx + 1,  # 1-indexed within body_lines
            "start_marker": "",
            "kind": "body",
        }
        for i, (line_idx, _words) in enumerate(groups)
    ]
    return [dict(ch, number=i + 1) for i, ch in enumerate(head + parts)]


def _maybe_prepend_chapter_one(
    raw_chapters: list[dict],
    body_lines: list[str],
) -> list[dict]:
    """If there's substantial narrative text before the first detected chapter, prepend a chapter 1.

    Handles books like the illustrated P&P where Chapter I has no standalone heading.
    Strategy: reconstruct the pre-chapter text (ignoring illustration blocks and TOC lines),
    and if it's substantial, find where it starts.
    """
    import re as _re

    first_chapter_start_idx = (raw_chapters[0]["start_line"] - 1) if raw_chapters else len(body_lines)
    MIN_CHAPTER_WORDS = 50

    TOC_RE = _re.compile(
        r"(heading to chapter|tailpiece|list of illustrations|\bcontents?\b"
        r"|^\s*\d+\s*$|\s{3,}\d+\s*$)",
        _re.IGNORECASE,
    )

    def is_toc(line: str) -> bool:
        # A heading-shaped line ahead of the first detected chapter is a contents
        # entry, not narrative — otherwise a full TOC reads as a 500-word chapter one.
        # Takes the raw line: an indented numeral entry ("   I.  A Scandal in
        # Bohemia") is recognizable only by its indentation, since the heading
        # patterns require an all-caps title and contents entries are title case.
        stripped = line.strip()
        return (
            bool(TOC_RE.search(stripped))
            or text_utils.looks_like_chapter_heading(stripped)
            or bool(text_utils.TOC_NUMERAL_ENTRY_RE.match(line))
        )

    # Build set of illustration line indices before the first chapter
    illustration_lines: set[int] = _find_illustration_lines(body_lines, 0, first_chapter_start_idx)

    # Collect prose lines (non-illustration, non-blank, non-TOC) before first chapter
    # Track the earliest one as our candidate chapter-1 start
    first_prose_line: int | None = None
    last_prose_line: int | None = None

    for j in range(first_chapter_start_idx):
        if j in illustration_lines:
            continue
        stripped = body_lines[j].strip()
        if not stripped:
            continue
        if is_toc(body_lines[j]):
            continue
        if last_prose_line is None or j > last_prose_line:
            last_prose_line = j

    if last_prose_line is None:
        return raw_chapters

    # Now scan forward from the last prose line to find the contiguous block it belongs to,
    # then scan backward to find where that block starts (skipping illustrations).
    # Walk back from last_prose_line to find the start of the prose block nearest the chapter.
    block_end = last_prose_line
    block_start = block_end
    j = block_end - 1
    while j >= 0:
        if j in illustration_lines:
            # Skip the illustration block backward
            j -= 1
            continue
        stripped = body_lines[j].strip()
        if not stripped:
            # Blank line — peek further back
            k = j - 1
            while k >= 0 and (k in illustration_lines or not body_lines[k].strip()):
                k -= 1
            if k < 0:
                break
            if is_toc(body_lines[k]):
                break  # Hit TOC boundary
            # Continue over the blank gap
            block_start = k
            j = k - 1
        else:
            if is_toc(body_lines[j]):
                break
            block_start = j
            j -= 1

    # A preface or dedication reads as prose to the walk-back above — nothing in
    # it is TOC-shaped. But such a block announces itself with a heading
    # ("PREFACE TO FIRST EDITION", "DEDICATION", ...); if one is present anywhere
    # in the block, this is the edition's apparatus, not an unlabeled chapter one.
    for j in range(block_start, first_chapter_start_idx):
        if j in illustration_lines:
            continue
        stripped = body_lines[j].strip()
        if (text_utils.FRONT_MATTER_RE.match(stripped)
                or text_utils.BACK_MATTER_RE.match(stripped)):
            return raw_chapters

    # Collect all text in [block_start, first_chapter_start_idx) excluding illustrations
    pre_text_words = []
    paragraphs: list[str] = []
    current: list[str] = []
    for j in range(block_start, first_chapter_start_idx):
        if j in illustration_lines:
            continue
        stripped = body_lines[j].strip()
        if stripped and not is_toc(body_lines[j]):
            pre_text_words.extend(stripped.split())
            current.append(stripped)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))

    if len(pre_text_words) < MIN_CHAPTER_WORDS:
        return raw_chapters

    if not _reads_as_prose(paragraphs):
        return raw_chapters

    # Advance block_start past any leading TOC/blank lines to find actual prose start
    actual_start = block_start
    for j in range(block_start, first_chapter_start_idx):
        if j in illustration_lines:
            continue
        stripped = body_lines[j].strip()
        if not stripped or is_toc(body_lines[j]):
            continue
        actual_start = j
        break

    synthetic = {
        "number": 1,
        "title": "Chapter I",
        "start_line": actual_start + 1,  # 1-indexed
        "start_marker": "",
    }
    # Renumber subsequent chapters starting from 2
    renumbered = []
    for i, ch in enumerate(raw_chapters):
        renumbered.append(dict(ch, number=i + 2))
    return [synthetic] + renumbered


def _reads_as_prose(paragraphs: list[str], min_ratio: float = 0.5) -> bool:
    """True if a block is narrative rather than a list of entries.

    An unlabeled chapter one is made of sentences; a list of illustrations or a
    contents block is made of captions. Measured on PG 1342's genuine chapter
    one, 34 of 34 paragraphs end in sentence punctuation (median 21 words); on
    PG 37106's list of illustrations, 7 of 141 do (median 5 words). That gap is
    what separates them — the front-matter heading that would otherwise name the
    block sits behind a TOC line the backward walk stops on, so it is never seen.
    """
    if not paragraphs:
        return False
    ended = sum(1 for p in paragraphs if p.rstrip().endswith((".", "!", "?", '"', "”", "’", "—")))
    return ended / len(paragraphs) >= min_ratio


def _find_illustration_lines(body_lines: list[str], start: int, end: int) -> set[int]:
    """Return set of 0-based indices that are inside illustration blocks."""
    result: set[int] = set()
    i = start
    while i < end:
        stripped = body_lines[i].strip()
        if stripped.lower().startswith("[illustration"):
            depth = stripped.count("[") - stripped.count("]")
            result.add(i)
            if depth > 0:
                i += 1
                while i < end:
                    result.add(i)
                    depth += body_lines[i].count("[") - body_lines[i].count("]")
                    i += 1
                    if depth <= 0:
                        break
                continue
        i += 1
    return result


def _validate_llm_chapters(
    chapters: list[dict],
    body_line_count: int,
    config: Config,
) -> list[dict]:
    """Sanitize model-provided chapter entries before trusting their positions.

    Chapter ends are derived from the *next* entry's start, so an out-of-order
    or out-of-range start silently produces an empty chapter while its
    neighbour balloons. Entries are range-checked, sorted by position, and
    deduplicated — loudly, never silently. Front and back matter (prefaces,
    footnote appendices) are dropped unless the config keeps them.
    """
    valid: list[dict] = []
    seen_starts: set[int] = set()
    for ch in chapters:
        title = ch.get("title", "")
        try:
            start = int(ch.get("start_line", 0))
        except (TypeError, ValueError):
            start = 0
        if start < 1 or start > body_line_count:
            console.print(
                f"[yellow]Stage 02: warning —[/yellow] dropping LLM chapter {title!r}: "
                f"start_line {ch.get('start_line')!r} is outside the body (1–{body_line_count})"
            )
            continue
        if start in seen_starts:
            console.print(
                f"[yellow]Stage 02: warning —[/yellow] dropping LLM chapter {title!r}: "
                f"duplicate start_line {start}"
            )
            continue
        kind = text_utils.classify_heading(title)
        if kind == "front" and not config.include_front_matter:
            console.print(f"[cyan]Stage 02:[/cyan] skipping front matter: {title!r}")
            continue
        if kind == "back" and not config.include_back_matter:
            console.print(f"[cyan]Stage 02:[/cyan] skipping back matter: {title!r}")
            continue
        seen_starts.add(start)
        valid.append(dict(ch, start_line=start, kind=kind))

    in_order = [ch["start_line"] for ch in valid]
    if in_order != sorted(in_order):
        console.print(
            "[yellow]Stage 02: warning —[/yellow] LLM returned chapters out of "
            "document order; sorting by position"
        )
        valid.sort(key=lambda ch: ch["start_line"])
    return valid


def _llm_chapter_discovery(
    body_lines: list[str],
    config: Config,
    client: LLMRouter,
) -> list[dict]:
    """Fall back to LLM for chapter detection."""
    # Send at most first 500 lines to keep context manageable
    sample = "\n".join(
        f"{i+1}: {line}" for i, line in enumerate(body_lines[:500])
    )

    messages = [
        {"role": "system", "content": prompts.llm_chapter_discovery_system()},
        {"role": "user", "content": prompts.llm_chapter_discovery_user(sample)},
    ]

    try:
        data = call_json_with_retries(
            client, config.structure_model, messages, schema=schemas.CHAPTERS_SCHEMA,
            retries=config.max_retries, what="chapter discovery", console=console,
        )
        if data is None:
            raise LLMError("chapter discovery exhausted its retries")
        chapters = data.get("chapters", [])
        chapters = _validate_llm_chapters(chapters, len(body_lines), config)
        # Normalize to our format. Numbering is positional, not taken from the
        # model: a heading the model labels "Chapter 1" may be the Nth it found,
        # and a repeated number would collide in the per-chapter caches.
        return [
            {
                "number": i + 1,
                "title": ch.get("title", f"Chapter {i+1}"),
                "start_line": ch.get("start_line", 1),
                "start_marker": ch.get("title", ""),
                "kind": ch.get("kind", "body"),
            }
            for i, ch in enumerate(chapters)
        ]
    except Exception as e:
        console.print(f"[red]Stage 02: LLM discovery failed: {e}[/red]")
        # Return a single "chapter" covering the whole body
        return [{"number": 1, "title": "Chapter 1", "start_line": 1, "start_marker": ""}]
