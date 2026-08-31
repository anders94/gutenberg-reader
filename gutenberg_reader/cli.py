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
@click.option("--model", default="", help="Processing model (default: first model served by the API)")
@click.option("--validator", default="",
              help="Model for the critical/tie-breaker attribution passes and the Stage 06 "
                   "critic (default: same as --model)")
@click.option("--base-url", default="http://localhost:8000/v1", show_default=True,
              help="OpenAI-compatible API base URL (vLLM, llama.cpp server, LM Studio, ...)")
@click.option("--validator-base-url", default="",
              help="API base URL serving --validator (default: --base-url). Point this "
                   "at a second server to run the judgment passes on a larger model")
@click.option("--structure-base-url", default="",
              help="API base URL for chapter-structure analysis (default: --validator-base-url)")
@click.option("--structure-model", default="",
              help="Model for chapter-structure analysis (default: --validator)")
@click.option("--api-key", default="EMPTY", help="API key, if the server requires one")
@click.option("--cache-dir", default="./cache", show_default=True, type=click.Path(), help="Cache directory")
@click.option("--output", default=None, type=click.Path(), help="Output file path")
@click.option("--chunk-size", default=1000, show_default=True, type=int,
              help="Words per LLM attribution window")
@click.option("--critic/--no-critic", "critic", default=True, show_default=True,
              help="Run the Stage 06 LLM critic pass: reviews speaker attribution and "
                   "the characters discovery just added. Most useful with a larger "
                   "--validator model")
@click.option("--include-front-matter", is_flag=True, default=False,
              help="Keep prefaces, introductions, and dedications as chapters (default: skip them)")
@click.option("--include-back-matter", is_flag=True, default=False,
              help="Keep footnotes, appendices, and indexes as a final chapter (default: trim them)")
@click.option("--span-review/--no-span-review", default=True, show_default=True,
              help="Ask whether an ambiguous quoted span is speech, a term being "
                   "discussed, or a title. Off, every quoted phrase is dialogue")
@click.option("--structure", "structure_detector",
              type=click.Choice(["llm", "regex"]), default="llm", show_default=True,
              help="How to find chapter boundaries. 'regex' is the previous "
                   "pattern-matching detector, kept for comparison and offline runs")
@click.option("--accept-structure-warnings", is_flag=True, default=False,
              help="Ship a book whose detected structure failed its checks "
                   "(default: refuse; a bad structure costs hours of TTS downstream)")
@click.option("--force-stage", default=None, type=int, metavar="STAGE",
              help="Re-run from this stage (1-7; discovery/segmentation/critic share one "
                   "loop, so 4 and 5 are equivalent)")
@click.option("--chapters", default=None, help="Process only these chapters (e.g. 1,2,5)")
@click.option("--max-retries", default=3, show_default=True, type=int, help="Max retries per chunk")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output")
def main(
    book_id: str,
    model: str,
    validator: str,
    base_url: str,
    validator_base_url: str,
    structure_base_url: str,
    structure_model: str,
    api_key: str,
    cache_dir: str,
    output: str | None,
    chunk_size: int,
    critic: bool,
    include_front_matter: bool,
    include_back_matter: bool,
    span_review: bool,
    structure_detector: str,
    accept_structure_warnings: bool,
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
        base_url=base_url,
        api_key=api_key,
        processing_model=model,
        validation_model=validator or model,
        validator_base_url=validator_base_url,
        structure_base_url=structure_base_url,
        structure_model=structure_model,
        cache_dir=Path(cache_dir),
        output_file=output_path,
        chunk_size=chunk_size,
        max_retries=max_retries,
        verbose=verbose,
        critic=critic,
        include_front_matter=include_front_matter,
        include_back_matter=include_back_matter,
        span_review=span_review,
        structure_detector=structure_detector,
        accept_structure_warnings=accept_structure_warnings,
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
