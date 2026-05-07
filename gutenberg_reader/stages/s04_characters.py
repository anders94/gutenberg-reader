"""Stage 04 — Discover characters using LLM on first N chapters."""

from __future__ import annotations
from pathlib import Path

from rich.console import Console

from gutenberg_reader.cache import atomic_write_json, read_json, read_text, stage_complete
from gutenberg_reader.config import Config
from gutenberg_reader.models import CharacterInfo
from gutenberg_reader.ollama import OllamaClient
from gutenberg_reader import prompts

console = Console()

FIRST_PASS_CHAPTERS = 5
SECOND_PASS_CHAPTERS = 5  # Sample from middle/later chapters


def run(
    config: Config,
    client: OllamaClient,
    chapter_paths: dict[int, Path],
) -> list[CharacterInfo]:
    """Discover characters from chapters. Returns list of CharacterInfo."""
    stage_dir = config.stage_dir(4)
    out_path = stage_dir / "characters.json"

    if stage_complete(out_path) and (config.force_stage is None or config.force_stage > 4):
        if config.verbose:
            console.print(f"[dim]Stage 04: already complete ({out_path})[/dim]")
        data = read_json(out_path)
        return [CharacterInfo.from_dict(c) for c in data["characters"]]

    sorted_nums = sorted(chapter_paths.keys())

    # First pass: first N chapters
    first_nums = sorted_nums[:FIRST_PASS_CHAPTERS]
    first_text = _load_chapters_text(first_nums, chapter_paths)

    if config.verbose:
        console.print(f"[cyan]Stage 04:[/cyan] First pass on {len(first_nums)} chapters...")

    first_chars = _discover_characters(first_text, config, client)

    # Second pass: sample from later chapters if book is long enough
    all_chars = first_chars
    if len(sorted_nums) > FIRST_PASS_CHAPTERS + SECOND_PASS_CHAPTERS:
        later_nums = sorted_nums[FIRST_PASS_CHAPTERS:FIRST_PASS_CHAPTERS + SECOND_PASS_CHAPTERS]
        later_text = _load_chapters_text(later_nums, chapter_paths)

        if config.verbose:
            console.print(f"[cyan]Stage 04:[/cyan] Second pass on chapters {later_nums}...")

        later_chars = _discover_characters(later_text, config, client, existing=first_chars)
        all_chars = _merge_characters(first_chars, later_chars)

    if config.verbose:
        console.print(f"[cyan]Stage 04:[/cyan] Found {len(all_chars)} unique characters")

    data = {"characters": [c.to_dict() for c in all_chars]}
    atomic_write_json(out_path, data)
    return all_chars


def _load_chapters_text(nums: list[int], chapter_paths: dict[int, Path]) -> str:
    parts = []
    for n in nums:
        if n in chapter_paths:
            try:
                text = read_text(chapter_paths[n])
                parts.append(f"--- Chapter {n} ---\n{text[:3000]}")
            except FileNotFoundError:
                pass
    return "\n\n".join(parts)


def _discover_characters(
    text: str,
    config: Config,
    client: OllamaClient,
    existing: list[CharacterInfo] | None = None,
) -> list[CharacterInfo]:
    """Call LLM to discover characters in the given text."""
    messages = [
        {"role": "system", "content": prompts.character_discovery_system()},
        {"role": "user", "content": prompts.character_discovery_user(text)},
    ]

    try:
        data = client.chat_json(config.processing_model, messages)
        chars = data.get("characters", [])
        return [CharacterInfo.from_dict(c) for c in chars]
    except Exception as e:
        console.print(f"[red]Stage 04: Character discovery failed: {e}[/red]")
        return existing or []


def _merge_characters(
    first: list[CharacterInfo],
    second: list[CharacterInfo],
) -> list[CharacterInfo]:
    """Merge two character lists, deduplicating by name and aliases."""
    by_name: dict[str, CharacterInfo] = {}

    for char in first:
        by_name[char.name.lower()] = char

    for char in second:
        key = char.name.lower()
        if key in by_name:
            # Merge aliases
            existing = by_name[key]
            for alias in char.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
        else:
            # Check if it's an alias of an existing character
            found = False
            for existing in by_name.values():
                if any(a.lower() == key for a in existing.aliases):
                    found = True
                    break
            if not found:
                by_name[key] = char

    return list(by_name.values())
