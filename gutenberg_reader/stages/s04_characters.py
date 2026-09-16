"""Stage 04 — Character discovery, one chapter at a time.

There is no separate discovery sweep. Stage 05's chapter loop calls
discover_in_chapter() as it reads, merging each chapter's finds into a rolling
roster, so chapter N's speaker enum contains exactly the characters the book
has introduced by chapter N. The final roster is written to this stage's
directory (characters.json) by that loop, purely as an inspection artifact.

A novel's cast is not front-loaded — Moby Dick's first ten chapters name
Ishmael and Queequeg but not Ahab, Starbuck, Stubb or Flask, who between them
speak most of the book — and a name absent from the roster cannot be chosen by
the constrained attribution passes. Discovery therefore has to read everything;
reading it in the same loop that attributes means it is read exactly once.
"""

from __future__ import annotations

from rich.console import Console

from gutenberg_reader.config import Config
from gutenberg_reader.models import CharacterInfo
from gutenberg_reader.llm import (
    STALL_RETRY_REPETITION_PENALTY, LLMRouter, call_json_with_retries,
)
from gutenberg_reader import prompts, schemas, text_utils

console = Console()

# Words of chapter text per discovery call. A chapter is not bounded in length
# — and when structure detection goes wrong it is not bounded at all: PG 1661's
# story headings ("I. A SCANDAL IN BOHEMIA") once went undetected and left a
# single 100,984-word "chapter", which is ~130k tokens against a 65k context.
# Every other LLM call in the pipeline windows; this one does too.
DISCOVERY_WORD_BUDGET = 6000


def discover_in_chapter(
    text: str,
    chapter_num: int,
    config: Config,
    client: LLMRouter,
) -> list[CharacterInfo]:
    """Discover characters in one chapter's text.

    Long chapters are read in windows and their finds unioned. A window that
    fails costs its own text, not the chapter's.
    """
    found: list[CharacterInfo] = []
    for window in _split_for_discovery(text, DISCOVERY_WORD_BUDGET):
        found = text_utils.merge_rosters(found, _discover(window, chapter_num, config, client))
    return found


def _split_for_discovery(text: str, word_budget: int) -> list[str]:
    """Split text into ~word_budget windows at paragraph boundaries."""
    if text_utils.word_count(text) <= word_budget:
        return [text]

    windows: list[str] = []
    current: list[str] = []
    words = 0
    for para in text.split("\n\n"):
        w = text_utils.word_count(para)
        if current and words + w > word_budget:
            windows.append("\n\n".join(current))
            current, words = [], 0
        current.append(para)
        words += w
    if current:
        windows.append("\n\n".join(current))
    return windows


# A name listed this many times is a loop, not a cast list.
DEGENERATE_REPEATS = 3


def _is_degenerate(entries: list) -> bool:
    """True when the list repeats a name enough times to be a loop."""
    counts: dict[str, int] = {}
    for c in entries:
        name = (c.get("name") or "").strip().lower() if isinstance(c, dict) else ""
        if name:
            counts[name] = counts.get(name, 0) + 1
    return any(n >= DEGENERATE_REPEATS for n in counts.values())


def _discover(
    text: str,
    chapter_num: int,
    config: Config,
    client: LLMRouter,
) -> list[CharacterInfo]:
    """One discovery call. Returns [] on LLM failure."""
    messages = [
        {"role": "system", "content": prompts.character_discovery_system()},
        {"role": "user", "content": prompts.character_discovery_user(text)},
    ]

    # One bad window costs its text, not the roster built so far — but it is
    # worth retrying before giving up on it.
    data = call_json_with_retries(
        client, config.processing_model, messages, schema=schemas.CHARACTERS_SCHEMA,
        retries=config.max_retries,
        what=f"character discovery (chapter {chapter_num})", console=console,
    )
    if data is not None and _is_degenerate(data.get("characters", [])):
        # The list is the model repeating two names until the schema's array
        # bound closed it (PG 1184 chapter 14). Valid JSON, but the names that
        # came after the loop began were never written. Asked once more with a
        # repetition penalty the same window answers with its seven names.
        console.print(
            f"  [yellow]character discovery (chapter {chapter_num}): the model "
            f"repeated itself — asking again with a repetition penalty[/yellow]"
        )
        data = call_json_with_retries(
            client, config.processing_model, messages, schema=schemas.CHARACTERS_SCHEMA,
            retries=config.max_retries,
            what=f"character discovery (chapter {chapter_num})", console=console,
            sampling={"repetition_penalty": STALL_RETRY_REPETITION_PENALTY},
        )
    if data is None:
        return []

    chars = [
        CharacterInfo.from_dict(c)
        for c in data.get("characters", [])
        if c.get("name") and not text_utils.is_reserved_character_name(c["name"])
    ]
    for c in chars:
        # The model sees a single chapter; its guess at a first appearance is
        # meaningless. The loop calls us in reading order, so this is exact.
        c.first_appearance_chapter = chapter_num
    return chars
