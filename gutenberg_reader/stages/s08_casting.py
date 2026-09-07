"""Stage 08 — Casting: production notes and per-character voice specs.

Runs after assembly, on the final JSON, because only stage 07's roster is
real: it has merged duplicates and remapped every segment speaker to its
canonical name. Casting the rolling roster would produce voices for "Ahab"
and "Captain Ahab" separately, and for names no segment carries any more.

The enrichment is cached under 08-casting/ keyed to a fingerprint of the
speaking roster, so a resumed run (stage 07 always reassembles) re-merges
the cached casting instead of paying for the LLM again, and the installed
library file stays byte-stable. Decoding is greedy for the same reason.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from gutenberg_reader import prompts, schemas, text_utils
from gutenberg_reader.cache import atomic_write_json, read_json, stage_complete
from gutenberg_reader.config import Config
from gutenberg_reader.llm import LLMRouter, call_json_with_retries

console = Console()

CASTING_TEMPERATURE = 0.0
CASTING_VERSION = "casting-v1"

# Characters cast per LLM call. Small enough that one bad batch loses little
# and the model attends to each character; large enough not to pay the
# system-prompt tax per character.
VOICE_BATCH = 10

# A character must actually speak to need a voice. Non-speaking roster
# entries (mentioned relatives, addressees) still ship, just without specs —
# downstream tolerates an absent voice block.
MIN_DIALOGUE_SEGMENTS = 1

# Dialogue evidence shown per character: first, middle and last line, each
# clipped — enough to hear a manner of speaking without paying for a chapter.
SAMPLE_LINES = 3
SAMPLE_LINE_CHARS = 160


def run(
    config: Config,
    client: LLMRouter,
    final_path: Path,
    *,
    work_type: str = "",
    narration_person: str = "",
    narrator_name: str = "",
) -> Path:
    """Enrich the assembled book JSON in place. Never raises past a lost
    LLM answer: a book without production notes still installs."""
    data = read_json(final_path)

    counts, samples = _dialogue_evidence(data)
    roster = [c["name"] for c in data.get("characters", [])]
    cast_names = [n for n in roster if counts.get(n, 0) >= MIN_DIALOGUE_SEGMENTS]

    cache_path = config.stage_dir(8) / "casting.json"
    fingerprint = {
        "version": CASTING_VERSION,
        "model": config.validation_model,
        "narration": [narration_person, narrator_name],
        "speakers": {n: counts.get(n, 0) for n in cast_names},
    }

    cached = None
    if stage_complete(cache_path) and (
            config.force_stage is None or config.force_stage > 8):
        payload = read_json(cache_path)
        if payload.get("source") == fingerprint:
            cached = payload

    if cached is not None:
        production = cached.get("production")
        castings = cached.get("castings", [])
        if config.verbose:
            console.print("[dim]Stage 08: cached casting reused[/dim]")
    else:
        production = _cast_production(
            config, client, data, cast_names, counts,
            work_type=work_type, narration_person=narration_person,
            narrator_name=narrator_name)
        castings = _cast_voices(config, client, data, cast_names, counts,
                                samples, production)
        atomic_write_json(cache_path, {
            "source": fingerprint,
            "production": production,
            "castings": castings,
        })

    _merge(data, production, castings, counts)
    atomic_write_json(final_path, data)

    console.print(
        f"  [dim]cast {len(castings)}/{len(cast_names)} speaking characters"
        + ("" if production else "; production notes lost") + "[/dim]"
    )
    return final_path


def _dialogue_evidence(data: dict) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Per-canonical-name dialogue counts and sample lines, from segments.

    Counted here, not upstream: stage 07 has already remapped speakers, so
    these totals are what an audiobook renderer will actually see.
    """
    counts: dict[str, int] = {}
    lines: dict[str, list[str]] = {}
    reserved = {"Unknown", "Narrator", text_utils.CITATION_SPEAKER}
    for entry in data.get("chapters", []):
        for seg in entry.get("processed", {}).get("segments", []):
            speaker = seg.get("speaker")
            if seg.get("type") != "dialogue" or not speaker or speaker in reserved:
                continue
            counts[speaker] = counts.get(speaker, 0) + 1
            lines.setdefault(speaker, []).append(seg.get("text", ""))

    samples: dict[str, list[str]] = {}
    for name, all_lines in lines.items():
        picks = [all_lines[0]]
        if len(all_lines) > 2:
            picks.append(all_lines[len(all_lines) // 2])
        if len(all_lines) > 1:
            picks.append(all_lines[-1])
        samples[name] = [ln[:SAMPLE_LINE_CHARS] for ln in picks[:SAMPLE_LINES]]
    return counts, samples


def _cast_production(config, client, data, cast_names, counts, *,
                     work_type, narration_person, narrator_name):
    meta = data.get("metadata", {})
    speaker_lines = [
        f"- {n} ({counts.get(n, 0)})"
        for n in sorted(cast_names, key=lambda n: -counts.get(n, 0))[:30]
    ]
    result = call_json_with_retries(
        client, config.validation_model,
        [{"role": "system", "content": prompts.casting_production_system()},
         {"role": "user", "content": prompts.casting_production_user(
             meta.get("title", ""), meta.get("author", ""), work_type,
             narration_person, narrator_name, speaker_lines)}],
        schema=schemas.production_schema(cast_names),
        retries=config.max_retries, what="production casting",
        console=console, temperature=CASTING_TEMPERATURE,
    )
    if result is None:
        return None
    # A narrator link must point at a real roster entry; guided decoding
    # already guarantees that, but the pairing must also make sense — a
    # third-person book has no narrator character to link.
    narration = result.get("narration", {})
    if narration.get("person") != "first_person":
        narration["narrator_character"] = ""
    return result


def _cast_voices(config, client, data, cast_names, counts, samples, production):
    meta = data.get("metadata", {})
    casting_notes = (production or {}).get("casting_notes", "")
    house_locale = (((production or {}).get("narration") or {})
                    .get("voice", {}).get("accent", {}).get("locale", ""))

    ordered = sorted(cast_names, key=lambda n: (-counts.get(n, 0), n))
    castings: list[dict] = []
    for i in range(0, len(ordered), VOICE_BATCH):
        batch = ordered[i:i + VOICE_BATCH]
        blocks = []
        for name in batch:
            quoted = "\n".join(f"  {ln}" for ln in samples.get(name, []))
            blocks.append(f"{name} — {counts.get(name, 0)} dialogue segments\n"
                          f"{quoted}")
        result = call_json_with_retries(
            client, config.validation_model,
            [{"role": "system", "content": prompts.casting_voices_system()},
             {"role": "user", "content": prompts.casting_voices_user(
                 meta.get("title", ""), meta.get("author", ""),
                 casting_notes, house_locale, blocks)}],
            schema=schemas.voices_schema(batch),
            retries=config.max_retries,
            what=f"voice casting ({batch[0]}…)",
            console=console, temperature=CASTING_TEMPERATURE,
        )
        if result is None:
            # This batch's characters stay uncast; the book still ships.
            continue
        seen: set[str] = set()
        for casting in result.get("castings", []):
            name = casting.get("name")
            if name in batch and name not in seen:
                seen.add(name)
                castings.append(casting)
    return castings


def _merge(data: dict, production, castings: list[dict],
           counts: dict[str, int]) -> None:
    if production is not None:
        data["production"] = production
    by_name = {c["name"]: c for c in castings}
    for char in data.get("characters", []):
        char["dialogue_segments"] = counts.get(char["name"], 0)
        cast = by_name.get(char["name"])
        if cast:
            char["description"] = cast.get("description", "")
            char["voice"] = cast.get("voice")
            char["confidence"] = cast.get("confidence", "")
            char["basis"] = cast.get("basis", "")
    stats = data.setdefault("statistics", {})
    stats["cast_characters"] = len(by_name)
