"""Whole-book cast regularization — one person, one name, at assembly.

Discovery names people chapter by chapter, and a rolling roster names them
forward-only, so the same person ends a long novel under several entries:
on PG 37106 Laurie spoke 83 lines as "Mr. Laurence" (his grandfather, who had
been handed the alias "Laurie" in chapter 2) before "Theodore Laurence" was
discovered in chapter 10; Meg was "Meg March" for 322 lines and "Margaret
March" for 112, both entries claiming the alias "Meg"; Amy became "Aunt Amy"
from chapter 45 and the book-wide remap sent her chapter-one lines there too.
For an audiobook each split is one character read in two voices.

The evidence that settles these is already deterministic: two entries claiming
the same alias is a conflict, and the line counts say who is who. This module
lists the conflicts and the roster with counts, asks the validator model once
which entries are one person and who owns each disputed alias, applies the
answer, and then re-anchors every tag-backed line ("said Laurie") against the
settled alias map so the text's own attributions follow their owner.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console

from gutenberg_reader import prompts, schemas, text_utils
from gutenberg_reader.cache import atomic_write_json, read_json
from gutenberg_reader.config import Config
from gutenberg_reader.llm import LLMRouter, call_json_with_retries
from gutenberg_reader.models import CharacterInfo

console = Console()

# A judgment; sampling made the roster review one that changed its mind.
CAST_REVIEW_TEMPERATURE = 0.0

# Aliases that are a role rather than a name are claimed by half a cast
# legitimately ("Mother" of every child, "the old lady") and settle nothing.
_ROLE_ALIAS_RE = re.compile(
    r"^(the |his |her |their |my |our |old |little )*"
    r"(mother|father|mamma|papa|mama|grandfather|grandmother|grandpa|grandma|"
    r"uncle|aunt|sister|brother|wife|husband|lady|gentleman|doctor|"
    r"professor|captain|king|queen|count|countess|narrator|boy|girl|man|"
    r"woman|child|baby|stranger|tutor|servant|maid|cook|nurse)$",
    re.IGNORECASE,
)


def alias_conflicts(roster: list[CharacterInfo]) -> dict[str, list[str]]:
    """Every alias claimed by more than one entry: {alias: [entry names]}.

    Names count as claims too — "Amy" the entry against "Amy" the alias of
    "Aunt Amy". Role words are skipped; a real name claimed twice is the
    signal.
    """
    claims: dict[str, list[str]] = defaultdict(list)
    surface: dict[str, str] = {}
    for c in roster:
        for form in [c.name, *c.aliases]:
            key = form.strip().lower()
            if not key or _ROLE_ALIAS_RE.match(key):
                continue
            if c.name not in claims[key]:
                claims[key].append(c.name)
                surface.setdefault(key, form.strip())
    return {surface[k]: names for k, names in claims.items() if len(names) > 1}


def line_counts(chapters_out: list[dict]) -> Counter:
    counts: Counter = Counter()
    for entry in chapters_out:
        for seg in entry["processed"]["segments"]:
            if seg.get("type") == "dialogue" and seg.get("speaker"):
                counts[seg["speaker"]] += 1
    return counts


def _fingerprint(roster: list[CharacterInfo], counts: Counter) -> dict:
    return {
        "roster": sorted((c.name, sorted(a.lower() for a in c.aliases)) for c in roster),
        "lines": sorted(counts.items()),
    }


def review(
    config: Config,
    client: LLMRouter,
    roster: list[CharacterInfo],
    counts: Counter,
) -> tuple[list[dict], dict[str, str]]:
    """Ask once which entries are one person and who owns each disputed alias.

    Returns (merges, alias_owners). Cached beside the assembled book, keyed by
    the roster and the line counts, so re-assembling does not re-ask.
    """
    conflicts = alias_conflicts(roster)
    names = [c.name for c in roster]
    if len(names) < 2:
        return [], {}

    cache_path = config.stage_dir(7) / "cast-review.json"
    fp = _fingerprint(roster, counts)
    if cache_path.exists():
        cached = read_json(cache_path)
        if cached.get("fingerprint") == fp:
            return cached.get("merges", []), cached.get("alias_owners", {})

    entries = [
        {"name": c.name, "aliases": c.aliases, "first_chapter": c.first_appearance_chapter,
         "lines": counts.get(c.name, 0)}
        for c in sorted(roster, key=lambda c: (-counts.get(c.name, 0), c.first_appearance_chapter))
    ]
    data = call_json_with_retries(
        client, config.validation_model,
        [{"role": "system", "content": prompts.cast_review_system()},
         {"role": "user", "content": prompts.cast_review_user(entries, conflicts)}],
        schema=schemas.cast_review_schema(names, conflicts),
        retries=config.max_retries, what="cast review", console=console,
        temperature=CAST_REVIEW_TEMPERATURE,
    )
    if data is None:
        return [], {}
    merges = [m for m in data.get("merges", []) if isinstance(m, dict)]
    owners = {a: o for a, o in (data.get("alias_owners") or {}).items()
              if a in conflicts and o in conflicts[a]}
    atomic_write_json(cache_path, {"fingerprint": fp, "merges": merges, "alias_owners": owners})
    return merges, owners


# Tagged turns this close together, by two different labels, are two people
# talking. One person does not answer themself.
CONVERSATION_SPAN = 6
CONVERSATION_MIN = 2


def conversing_pairs(chapters_out: list[dict]) -> Counter:
    """How often each pair of labels trade tagged lines within a few segments.

    The model folded "Mr. Laurence" into "Theodore Laurence" — grandfather
    into grandson — against an explicit instruction. The text refutes it:
    "said Mr. Laurence" and "said Laurie" sit two segments apart in chapter
    21 and again in 35. Only tag-backed labels count; an inferred label is a
    guess and can be the split it is meant to detect.
    """
    pairs: Counter = Counter()
    for entry in chapters_out:
        tagged = [
            (i, seg["speaker"]) for i, seg in enumerate(entry["processed"]["segments"])
            if seg.get("type") == "dialogue" and seg.get("evidence") == "tag"
            and seg.get("speaker")
        ]
        for (i, a), (j, b) in zip(tagged, tagged[1:]):
            if a != b and j - i <= CONVERSATION_SPAN:
                pairs[frozenset((a, b))] += 1
    return pairs


def _given_names(c: CharacterInfo) -> set[str]:
    """Name tokens that identify the person rather than the family.

    Strong tokens, minus the last one when there are two or more: "Hannah"
    from "Hannah March", nothing from "Aunt March", "Demi" from "Master Demi".
    A shared surname is what a family has in common.
    """
    out: set[str] = set()
    for form in [c.name, *c.aliases]:
        strong = [t for t in form.split() if t.strip(".").lower() not in text_utils._WEAK_NAME_TOKENS]
        given = strong[:-1] if len(strong) >= 2 else strong
        out |= {t.strip(".,'\u2019").lower() for t in given}
    return out


def linked(a: CharacterInfo, b: CharacterInfo) -> bool:
    """Whether the text ties two entries to one person at all.

    A shared full form ("Meg" as alias of both), or a shared given name
    ("Sallie Moffat" / "Sallie Gardiner"). "Aunt March" and "Hannah March"
    share only the family name and the model merged them anyway; without a
    link there is nothing to merge on.
    """
    forms_a = {f.lower() for f in [a.name, *a.aliases]}
    forms_b = {f.lower() for f in [b.name, *b.aliases]}
    if forms_a & forms_b:
        return True
    return bool(_given_names(a) & _given_names(b))


def apply(
    roster: list[CharacterInfo],
    merges: list[dict],
    alias_owners: dict[str, str],
    conversing: Counter | None = None,
) -> tuple[list[CharacterInfo], dict[str, str], list[str], list[tuple[str, str, str]]]:
    """Apply a review. Returns (roster, {old label: new label}, log lines,
    [(alias, owner, loser)] for every alias taken away from an entry).

    A merge folds an entry into its canonical: the name and every alias travel
    with it. A chain (A into B, B into C) resolves to its end; a cycle is
    ignored. Ownership removes a disputed alias from every entry but its owner.

    Two refusals the text decides: entries with no name in common are not
    merged, and entries that converse are not merged.
    """
    by_name = {c.name: c for c in roster}
    conversing = conversing or Counter()
    log: list[str] = []
    into: dict[str, str] = {}
    for m in merges:
        name, canonical = m.get("name"), m.get("canonical")
        if not (name in by_name and canonical in by_name and name != canonical):
            continue
        if not linked(by_name[name], by_name[canonical]):
            log.append(f"refused {name!r} into {canonical!r}: no name in common")
            continue
        if conversing[frozenset((name, canonical))] >= CONVERSATION_MIN:
            log.append(f"refused {name!r} into {canonical!r}: they talk to each other")
            continue
        into[name] = canonical

    def resolve(name: str) -> str:
        seen = {name}
        while name in into:
            name = into[name]
            if name in seen:
                return name
            seen.add(name)
        return name

    relabel: dict[str, str] = {}
    for name in list(into):
        target = resolve(name)
        if target == name or target not in by_name or name not in by_name:
            continue
        src, dst = by_name[name], by_name[target]
        for alias in [src.name, *src.aliases]:
            if alias.lower() != dst.name.lower() and all(
                    alias.lower() != a.lower() for a in dst.aliases):
                dst.aliases.append(alias)
        dst.first_appearance_chapter = min(
            dst.first_appearance_chapter, src.first_appearance_chapter)
        relabel[name] = target
        del by_name[name]
        log.append(f"{name!r} is {target!r}")

    moved: list[tuple[str, str, str]] = []   # (alias, owner, loser)
    for alias, owner in alias_owners.items():
        owner = resolve(owner)
        if owner not in by_name:
            continue
        for c in by_name.values():
            if c.name != owner:
                before = len(c.aliases)
                c.aliases = [a for a in c.aliases if a.lower() != alias.lower()]
                if len(c.aliases) != before:
                    log.append(f"{alias!r} belongs to {owner!r}, not {c.name!r}")
                    moved.append((alias, owner, c.name))

    return list(by_name.values()), relabel, log, moved


def reanchor(chapters_out: list[dict], roster: list[CharacterInfo]) -> int:
    """Re-derive every tag-backed label against the settled alias map.

    "said Laurie" was anchored to whichever entry held the alias "Laurie"
    when the chapter was read. With the aliases settled, the same tag names
    the same line's owner, and the label follows. Returns how many changed.
    """
    changed = 0
    for entry in chapters_out:
        segments = entry["processed"]["segments"]
        anchors = text_utils.extract_attribution_anchors(segments, roster)
        for idx, name in anchors.items():
            seg = segments[idx]
            if seg.get("evidence") == "tag" and seg.get("speaker") != name:
                seg["speaker"] = name
                changed += 1
    return changed


def suspect_lines(
    chapters_out: list[dict],
    roster: list[CharacterInfo],
    moved: list[tuple[str, str, str]],
) -> dict[int, set[int]]:
    """Inferred lines an alias transfer casts doubt on: {chapter index: {segment index}}.

    "Laurie" moved from "Mr. Laurence" to "Theodore Laurence", who entered the
    roster in chapter 10. In chapters 3 to 9 the grandfather was the only
    Laurence the attribution passes could choose, and Laurie's 55 untagged
    lines there are his. Every inferred line under the loser, in a chapter
    before the owner existed, is a guess made without the right answer on
    offer.
    """
    first = {c.name: c.first_appearance_chapter for c in roster}
    out: dict[int, set[int]] = defaultdict(set)
    for _alias, owner, loser in moved:
        for ci, entry in enumerate(chapters_out):
            if entry["processed"]["chapter_number"] >= first.get(owner, 0):
                continue
            for si, seg in enumerate(entry["processed"]["segments"]):
                if (seg.get("type") == "dialogue" and seg.get("speaker") == loser
                        and seg.get("evidence") == "inferred"):
                    out[ci].add(si)
    return dict(out)


def reattribute(
    config: Config,
    client: LLMRouter,
    chapters_out: list[dict],
    roster: list[CharacterInfo],
    suspects: dict[int, set[int]],
) -> int:
    """Ask again about the suspect lines, with the settled roster on offer.

    The critical pass's prompt on the validator model, window by window, as
    stage 05 runs it; the re-anchored tag lines around them are now labelled
    with the owner and serve as anchors. Returns how many labels changed.
    """
    from gutenberg_reader.stages.s05_segments import _llm_window_pass

    names = [c.name for c in roster]
    # Cached, like the review itself: assembly runs on every resume, and an
    # answer sampled twice can differ, which would make two assemblies of
    # the same book disagree for no reason.
    cache_path = config.stage_dir(7) / "cast-reattribution.json"
    cached: dict = read_json(cache_path) if cache_path.exists() else {}
    changed = 0
    for ci, flagged in suspects.items():
        segments = chapters_out[ci]["processed"]["segments"]
        key = json.dumps({"chapter": chapters_out[ci]["processed"]["chapter_number"],
                          "lines": sorted(flagged), "names": names})
        if key in cached:
            answers = {int(k): v for k, v in cached[key].items()}
        else:
            answers = _llm_window_pass(
                segments, flagged, config, client,
                system_msg=prompts.verify_attribution_system(names),
                user_fn=prompts.verify_attribution_user,
                schema=schemas.attribution_schema(names),
                model=config.validation_model,
            )
            cached[key] = {str(k): v for k, v in answers.items()}
            atomic_write_json(cache_path, cached)
        for idx, speaker in answers.items():
            if speaker != segments[idx].get("speaker"):
                segments[idx]["speaker"] = speaker
                changed += 1
    return changed
