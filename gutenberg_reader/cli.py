"""Click CLI entry point for gutenberg-reader."""

from __future__ import annotations
import sys
from pathlib import Path

import click
from rich.console import Console

from gutenberg_reader.config import Config
from gutenberg_reader.pipeline import run_pipeline

console = Console()


@click.command()
@click.argument("book_id")
@click.option("--model", default="qwen2.5:14b", show_default=True, help="Ollama processing model")
@click.option("--validator", default="", help="Critic model (default: same as --model)")
@click.option("--ollama-url", default="http://localhost:11434", show_default=True, help="Ollama base URL")
@click.option("--cache-dir", default="./cache", show_default=True, type=click.Path(), help="Cache directory")
@click.option("--output", default=None, type=click.Path(), help="Output file path")
@click.option("--chunk-size", default=400, show_default=True, type=int, help="Words per chunk")
@click.option("--overlap", default=150, show_default=True, type=int, help="Overlap words between chunks")
@click.option("--no-critic", is_flag=True, default=False, help="Skip Stage 06 critic pass")
@click.option("--force-stage", default=None, type=int, metavar="STAGE", help="Re-run from this stage (1-7)")
@click.option("--chapters", default=None, help="Process only these chapters (e.g. 1,2,5)")
@click.option("--max-retries", default=3, show_default=True, type=int, help="Max retries per chunk")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output")
def main(
    book_id: str,
    model: str,
    validator: str,
    ollama_url: str,
    cache_dir: str,
    output: str | None,
    chunk_size: int,
    overlap: int,
    no_critic: bool,
    force_stage: int | None,
    chapters: str | None,
    max_retries: int,
    verbose: bool,
) -> None:
    """Download a Project Gutenberg book and produce structured JSON for TTS audiobook generation.

    BOOK_ID is the numeric Project Gutenberg book ID (e.g., 1342 for Pride and Prejudice).
    """
    # Parse chapter list
    chapters_only = None
    if chapters:
        try:
            chapters_only = [int(x.strip()) for x in chapters.split(",")]
        except ValueError:
            console.print(f"[red]Invalid --chapters value: {chapters!r}[/red]")
            sys.exit(1)

    output_path = Path(output) if output else None

    config = Config(
        book_id=str(book_id),
        ollama_url=ollama_url,
        processing_model=model,
        validation_model=validator or model,
        cache_dir=Path(cache_dir),
        output_file=output_path,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        max_retries=max_retries,
        verbose=verbose,
        no_critic=no_critic,
        force_stage=force_stage,
        chapters_only=chapters_only,
    )

    try:
        run_pipeline(config)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Run again to resume from where it left off.[/yellow]")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
