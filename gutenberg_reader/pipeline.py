"""Pipeline orchestrator — runs stages in order with resumability."""

from __future__ import annotations
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from gutenberg_reader.config import Config
from gutenberg_reader.models import CriticReport, ProcessedChapter
from gutenberg_reader.ollama import OllamaClient
from gutenberg_reader.stages import (
    s01_download,
    s02_discovery,
    s03_chapters,
    s04_characters,
    s05_segments,
    s06_critic,
    s07_assemble,
)

console = Console()


def run_pipeline(config: Config) -> Path:
    """Run the full pipeline and return path to final output."""
    start_time = time.time()

    # Create Ollama client and verify connectivity
    client = OllamaClient(base_url=config.ollama_url)

    console.print(f"[bold]gutenberg-reader[/bold] book_id={config.book_id} model={config.processing_model}")

    try:
        client.health_check(config.processing_model)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1) from None

    # Set up stage dirs
    for stage_num in range(1, 8):
        config.stage_dir(stage_num).mkdir(parents=True, exist_ok=True)

    # ── Stage 01: Download ────────────────────────────────────────────────────
    _log_stage(1, "Download", config)
    if _should_run(1, config):
        s01_download.run(config)
    else:
        console.print("[dim]Stage 01: skipped (cached)[/dim]")

    # ── Stage 02: Discovery ───────────────────────────────────────────────────
    _log_stage(2, "Discovery", config)
    discovery = s02_discovery.run(config, client)
    chapter_infos = discovery.chapters
    metadata = discovery.metadata

    console.print(
        f"  [dim]{metadata.title} by {metadata.author} — "
        f"{len(chapter_infos)} chapters[/dim]"
    )

    # Apply chapter filter
    if config.chapters_only:
        chapter_infos = [ci for ci in chapter_infos if ci.number in config.chapters_only]

    # ── Stage 03: Chapter Splitting ───────────────────────────────────────────
    _log_stage(3, "Chapter Splitting", config)
    chapter_paths = s03_chapters.run(config, chapter_infos)

    # ── Stage 04: Character Discovery ────────────────────────────────────────
    _log_stage(4, "Character Discovery", config)
    characters = s04_characters.run(config, client, chapter_paths)
    console.print(f"  [dim]{len(characters)} characters identified[/dim]")

    # ── Stage 05: Segmentation ────────────────────────────────────────────────
    _log_stage(5, "Segmentation", config)
    chapter_nums = [ci.number for ci in chapter_infos]
    processed = s05_segments.run(config, client, chapter_paths, characters, chapter_nums)

    # ── Stage 06: Critic Pass ─────────────────────────────────────────────────
    accepted: dict[int, tuple[ProcessedChapter, CriticReport | None]]

    if config.no_critic:
        console.print("[dim]Stage 06: skipped (--no-critic)[/dim]")
        accepted = {num: (ch, None) for num, ch in processed.items()}
    else:
        _log_stage(6, "Critic Pass", config)
        critic_results = s06_critic.run(config, client, processed, characters, chapter_nums)
        accepted = {}
        for num, (ch, report) in critic_results.items():
            accepted[num] = (ch, report)
        # Include any chapters that weren't critiqued (shouldn't happen, but safety)
        for num, ch in processed.items():
            if num not in accepted:
                accepted[num] = (ch, None)

    # ── Stage 07: Assembly ────────────────────────────────────────────────────
    _log_stage(7, "Assembly", config)
    out_path = s07_assemble.run(
        config,
        metadata,
        chapter_infos,
        accepted,
        start_time,
    )

    elapsed = time.time() - start_time
    console.print(
        f"\n[bold green]Done![/bold green] "
        f"{len(accepted)} chapters in {elapsed:.1f}s → [cyan]{out_path}[/cyan]"
    )
    return out_path


def _log_stage(num: int, name: str, config: Config) -> None:
    console.print(f"\n[bold blue]Stage {num:02d}[/bold blue] — {name}")


def _should_run(stage_num: int, config: Config) -> bool:
    """Return True if this stage should be re-run (forced or not yet complete)."""
    if config.force_stage is not None and config.force_stage <= stage_num:
        return True
    return True  # Let each stage decide based on its own cache check
