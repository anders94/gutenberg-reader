"""Stage 01 — Download book from Project Gutenberg."""

from __future__ import annotations
import time
from pathlib import Path

import chardet
import httpx
from rich.console import Console

from gutenberg_reader.cache import atomic_write_text, stage_complete
from gutenberg_reader.config import Config
from gutenberg_reader.text_utils import START_MARKER_RE

console = Console()

PRIMARY_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
FALLBACK_URL = "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt"
MIRROR_URL = "https://www.gutenberg.org/files/{id}/{id}-0.txt"


def run(config: Config) -> Path:
    """Download the book and save to 01-raw/book.txt. Returns path to saved file."""
    stage_dir = config.stage_dir(1)
    out_path = stage_dir / "book.txt"

    if stage_complete(out_path) and config.force_stage is None or (
        config.force_stage is not None and config.force_stage > 1 and stage_complete(out_path)
    ):
        if config.verbose:
            console.print(f"[dim]Stage 01: already complete ({out_path})[/dim]")
        return out_path

    book_id = config.book_id
    urls = [
        PRIMARY_URL.format(id=book_id),
        FALLBACK_URL.format(id=book_id),
        MIRROR_URL.format(id=book_id),
    ]

    text = None
    last_error = None

    for url in urls:
        if config.verbose:
            console.print(f"[cyan]Stage 01:[/cyan] Downloading from {url}")
        for attempt in range(3):
            try:
                with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    raw_bytes = resp.content
                    # Try UTF-8 first, fall back to chardet
                    try:
                        text = raw_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        detected = chardet.detect(raw_bytes)
                        encoding = detected.get("encoding") or "latin-1"
                        text = raw_bytes.decode(encoding, errors="replace")
                break
            except httpx.HTTPError as e:
                last_error = e
                if attempt < 2:
                    wait = 2 ** attempt
                    if config.verbose:
                        console.print(f"[yellow]  Attempt {attempt+1} failed, retrying in {wait}s...[/yellow]")
                    time.sleep(wait)
        if text is not None:
            break

    if text is None:
        raise RuntimeError(f"Stage 01: Failed to download book {book_id}: {last_error}")

    # Validate that it's a real Gutenberg file
    if not START_MARKER_RE.search(text):
        raise RuntimeError(
            f"Stage 01: Downloaded file does not contain Gutenberg START marker. "
            f"The file may not be a valid Project Gutenberg text."
        )

    atomic_write_text(out_path, text)
    if config.verbose:
        console.print(f"[green]Stage 01:[/green] Saved {len(text):,} chars to {out_path}")
    return out_path
