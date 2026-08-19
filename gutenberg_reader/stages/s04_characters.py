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
from gutenberg_reader.llm import LLMClient
from gutenberg_reader import prompts, schemas, text_utils

console = Console()


def discover_in_chapter(
    text: str,
    chapter_num: int,
    config: Config,
    client: LLMClient,
) -> list[CharacterInfo]:
    """Discover characters in one chapter's text. Returns [] on LLM failure."""
    messages = [
        {"role": "system", "content": prompts.character_discovery_system()},
        {"role": "user", "content": prompts.character_discovery_user(text)},
    ]

    try:
        data = client.chat_json(config.processing_model, messages, schema=schemas.CHARACTERS_SCHEMA)
    except Exception as e:
        # One bad chapter costs its discoveries, not the roster built so far.
        console.print(f"  [red]Stage 04: character discovery failed for chapter {chapter_num}: {e}[/red]")
        return []

    chars = [
        CharacterInfo.from_dict(c)
        for c in data.get("characters", [])
        if c.get("name") and not text_utils.is_placeholder_name(c["name"])
    ]
    for c in chars:
        # The model sees a single chapter; its guess at a first appearance is
        # meaningless. The loop calls us in reading order, so this is exact.
        c.first_appearance_chapter = chapter_num
    return chars
