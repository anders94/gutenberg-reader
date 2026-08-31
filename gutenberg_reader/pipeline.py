"""Pipeline orchestrator — runs stages in order with resumability."""

from __future__ import annotations
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from gutenberg_reader.config import Config
from gutenberg_reader.llm import LLMRouter
from gutenberg_reader.stages import (
    s01_download,
    s02_discovery,
    s03_chapters,
    s05_segments,
    s07_assemble,
)

console = Console()


def run_pipeline(config: Config) -> Path:
    """Run the full pipeline and return path to final output."""
    start_time = time.time()

    # Register every endpoint the run will use and resolve its model name.
    # Each role is health-checked against the server that actually serves it:
    # validation_model used to be checked against the *processing* endpoint, so
    # pointing --validator at a model only the second box serves killed the run
    # at startup.
    client = LLMRouter()

    try:
        config.processing_model = client.register(
            config.base_url, config.api_key,
            config.processing_model, config.processing_timeout,
        )
        config.validation_model = client.register(
            config.validator_base_url, config.api_key,
            config.validation_model, config.judgment_timeout,
        )
        config.structure_model = client.register(
            config.structure_base_url, config.api_key,
            config.structure_model, config.judgment_timeout,
            template_kwargs=None if config.structure_thinking
            else {"enable_thinking": False, "thinking": False},
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1) from None

    console.print(f"[bold]gutenberg-reader[/bold] book_id={config.book_id} model={config.processing_model}")
    if config.validation_model != config.processing_model or (
        client.endpoint_for(config.validation_model) != client.endpoint_for(config.processing_model)
    ):
        console.print(
            f"  [dim]judgment passes: {config.validation_model} "
            f"@ {client.endpoint_for(config.validation_model)}[/dim]"
        )

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

    # ── Stages 04–06: Read, Discover & Attribute ──────────────────────────────
    # One loop in reading order: per chapter, discover characters into the
    # rolling roster, segment, attribute, and (unless --no-critic) critique —
    # the critic can prune roster additions before later chapters see them.
    _log_stage(5, "Read, Discover & Attribute", config)
    if config.no_critic:
        console.print("[dim]critic: off (--no-critic)[/dim]")
    chapter_nums = [ci.number for ci in chapter_infos]
    accepted, characters = s05_segments.run(
        config, client, chapter_paths, chapter_nums,
        chapter_titles={ci.number: ci.title for ci in chapter_infos},
        quote_pair=(discovery.quote_open, discovery.quote_close)
        if discovery.quote_open else None,
    )
    console.print(f"  [dim]{len(characters)} characters identified[/dim]")

    # ── Stage 07: Assembly ────────────────────────────────────────────────────
    _log_stage(7, "Assembly", config)
    out_path = s07_assemble.run(
        config,
        metadata,
        chapter_infos,
        accepted,
        characters,
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
